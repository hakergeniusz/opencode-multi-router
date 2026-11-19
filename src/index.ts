/**
 * opencode-multi-router — native OpenCode V2 port of pi-multi-router.
 *
 * Registers a virtual OpenAI-compatible provider (`multirouter`) with four
 * router models that walk a model ladder and rotate multi-account API keys
 * until something answers:
 *
 *   multirouter/gpt-oss    ladder: openai/gpt-oss-120b -> openai/gpt-oss-20b
 *   multirouter/qwen       ladder: qwen/qwen3.8-27b
 *   multirouter/command-a  ladder: command-a-plus-05-2026 -> command-a-03-2025
 *   multirouter/command-r  ladder: command-r-plus-08-2024 -> command-r-08-2024
 *
 * The model and key actually used are applied at the last possible moment in
 * an `http.request` hook (URL, Authorization header and body `model` are
 * rewritten). The `retry` hook advances the (model, key) pair on errors that
 * OpenCode would otherwise treat as terminal (401, 429, 5xx, timeouts, ...).
 * A pair that succeeds is held; nothing advances on success.
 *
 * Keys are read only from environment variables or plugin options at setup.
 * They are never written to storage, logged, or printed by the status command.
 */

import { Model, Plugin, Provider } from "@opencode/plugin"
import {
  advanceState,
  findRoutable,
  initialState,
  isRotatableError,
  norm,
  normalizeState,
  type RouterSpec,
  type RouterState,
} from "./router-core.js"

const PROVIDER_ID = Provider.ID.make("multirouter")

const GROQ_CONTEXT_WINDOW = 131072
const GROQ_MAX_TOKENS = 16384
const COHERE_CONTEXT_WINDOW = 288000
const COHERE_MAX_TOKENS = 8192

const DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
const DEFAULT_COHERE_BASE_URL = "https://api.cohere.com/v1"

const SPECS: RouterSpec[] = [
  {
    group: "groq",
    id: "gpt-oss",
    name: "GPT OSS (router)",
    ladder: ["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
    contextWindow: GROQ_CONTEXT_WINDOW,
    maxTokens: GROQ_MAX_TOKENS,
  },
  {
    group: "groq",
    id: "qwen",
    name: "Qwen 3.8 27B (router)",
    ladder: ["qwen/qwen3.8-27b"],
    contextWindow: GROQ_CONTEXT_WINDOW,
    maxTokens: GROQ_MAX_TOKENS,
  },
  {
    group: "cohere",
    id: "command-a",
    name: "Command A (router)",
    ladder: ["command-a-plus-05-2026", "command-a-03-2025"],
    contextWindow: COHERE_CONTEXT_WINDOW,
    maxTokens: COHERE_MAX_TOKENS,
  },
  {
    group: "cohere",
    id: "command-r",
    name: "Command R (router)",
    ladder: ["command-r-plus-08-2024", "command-r-08-2024"],
    contextWindow: 128000,
    maxTokens: 4096,
  },
]

interface Options {
  /** API keys as arrays (prefer env vars — config files can leak secrets). */
  groqKeys?: string[]
  cohereKeys?: string[]
  /** Override the OpenAI-compatible endpoint per group. */
  groqBaseURL?: string
  cohereBaseURL?: string
}

interface KeyInfo {
  /** Stable, secret-free label, e.g. `groq-key1`. */
  label: string
  /** The actual secret, held only in this closure. */
  value: string
}

interface GroupConfig {
  baseURL: string
  keys: KeyInfo[]
}

/** Discovered keys per group. Values are never persisted or printed. */
function discoverKeys(group: "groq" | "cohere", options: Options): KeyInfo[] {
  const prefix = group.toUpperCase()
  const seen = new Set<string>()
  const values: string[] = []

  // Numbered env vars first, numerically ordered: GROQ_API_KEY_2, _10, ...
  const numbered = Object.keys(process.env)
    .map((name) => {
      const match = new RegExp(`^${prefix}_API_KEY_(\\d+)$`).exec(name)
      return match ? { n: Number(match[1]), name } : undefined
    })
    .filter((x): x is { n: number; name: string } => x !== undefined)
    .sort((a, b) => a.n - b.n)
  for (const entry of numbered) {
    const value = process.env[entry.name]
    if (value && !seen.has(value)) {
      seen.add(value)
      values.push(value)
    }
  }

  // A single unnumbered key is key 1 when no numbered keys exist.
  const single = process.env[`${prefix}_API_KEY`]
  if (single && !seen.has(single)) {
    seen.add(single)
    values.push(single)
  }

  // Plugin options (from the `plugins` object form in opencode.jsonc).
  const fromOptions = group === "groq" ? options.groqKeys : options.cohereKeys
  if (Array.isArray(fromOptions)) {
    for (const value of fromOptions) {
      if (typeof value === "string" && value && !seen.has(value)) {
        seen.add(value)
        values.push(value)
      }
    }
  }

  return values.map((value, index) => ({ label: `${group}-key${index + 1}`, value }))
}

function stateKey(sessionID: string, specID: string): string {
  return `router:${sessionID}:${specID}`
}

function epochKey(specID: string): string {
  return `router-epoch:${specID}`
}

function ladderLabel(spec: RouterSpec): string {
  return spec.ladder.join(" -> ")
}

export default Plugin.define({
  id: "opencode-multi-router",
  async setup(ctx) {
    const options = (ctx.options ?? {}) as Options

    const groups: Record<"groq" | "cohere", GroupConfig> = {
      groq: {
        baseURL: options.groqBaseURL ?? process.env.GROQ_BASE_URL ?? DEFAULT_GROQ_BASE_URL,
        keys: discoverKeys("groq", options),
      },
      cohere: {
        baseURL: options.cohereBaseURL ?? process.env.COHERE_BASE_URL ?? DEFAULT_COHERE_BASE_URL,
        keys: discoverKeys("cohere", options),
      },
    }

    // Last routed pair per virtual model — a hint for non-primary requests
    // (compaction, title, generate), which carry no router state by design.
    const lastGood = new Map<string, { m: number; k: number }>()
    // Reset epochs live in memory; a restart naturally resumes at the head.
    const epochs = new Map<string, number>()

    const epochOf = (specID: string): number => epochs.get(specID) ?? 0

    // --- provider + model registration -------------------------------------

    await ctx.provider.transform((editor) => {
      editor.add({
        info: {
          ...Provider.Info.empty(PROVIDER_ID),
          name: "Multi Router (Groq / Cohere)",
          activation: "enabled",
          package: "@opencode/ai/providers/openai-compatible",
          settings: {
            // Placeholder values only; the real endpoint/key are applied per
            // request in the http.request hook. No secret is hardcoded here.
            baseURL: groups.groq.baseURL,
            apiKey: groups.groq.keys[0]?.value ?? "",
          },
        },
        models: SPECS.map((spec) => {
          const info = Model.Info.default(PROVIDER_ID, Model.ID.make(spec.id))
          info.name = spec.name
          info.capabilities = { ...info.capabilities, input: ["text"] }
          info.limit = { context: spec.contextWindow, output: spec.maxTokens }
          return info
        }),
      })
    })

    // Model policy: keep router models enabled and, when the host exposes them,
    // honour an output-token override from plugin options.
    await ctx.model.transform((editor) => {
      for (const spec of SPECS) {
        editor.update(PROVIDER_ID, spec.id, (model) => {
          model.enabled = true
          model.limit = { ...model.limit, output: spec.maxTokens }
        })
      }
    })

    // --- state helpers ------------------------------------------------------

    const readState = async (sessionID: string, spec: RouterSpec): Promise<RouterState | undefined> => {
      const raw = await ctx.storage.get(stateKey(sessionID, spec.id))
      if (!raw || typeof raw !== "object" || Array.isArray(raw)) return undefined
      return raw as unknown as RouterState
    }

    const writeState = async (sessionID: string, spec: RouterSpec, state: RouterState): Promise<void> => {
      await ctx.storage.set(stateKey(sessionID, spec.id), state as unknown as Record<string, never>)
    }

    const resetTried = async (sessionID: string): Promise<void> => {
      const page = await ctx.storage.scan({ prefix: `router:${sessionID}:`, limit: 1000 })
      for (const entry of page.entries) {
        if (!entry.value || typeof entry.value !== "object" || Array.isArray(entry.value)) continue
        const state = entry.value as unknown as RouterState
        if (!Number.isInteger(state.m)) continue
        await ctx.storage.set(entry.key, { ...state, tried: 0 } as unknown as Record<string, never>)
      }
    }

    // --- hooks --------------------------------------------------------------

    // A new user prompt starts a fresh walk (pi: before_agent_start).
    await ctx.session.hook("prompt", async (event) => {
      await resetTried(event.sessionID)
    })

    // A completed turn resets the failure counter (pi: turn_end/completed).
    // experimental.idle deliberately does not fire on failure/interruption.
    await ctx.session.hook("experimental.idle", async (event) => {
      await resetTried(event.sessionID)
    })

    // Apply the chosen (model, key) pair at the last possible moment. The host
    // keeps the virtual model selected; only the outgoing HTTP call is rewritten.
    await ctx.session.hook(
      "http.request",
      async (event) => {
        if (event.model.providerID !== PROVIDER_ID) return
        const spec = SPECS.find((s) => s.id === event.model.id)
        if (!spec) return

        const config = groups[spec.group]
        if (config.keys.length === 0) {
          throw new Error(
            `multi-router: no API keys for ${spec.group} — set ${spec.group.toUpperCase()}_API_KEY_1, ...`,
          )
        }

        const epoch = epochOf(spec.id)
        const stored = await readState(event.sessionID, spec)
        let state = normalizeState(stored, spec.ladder.length, config.keys.length, epoch)

        let pair: { m: number; k: number }
        if (event.kind !== "primary") {
          // Auxiliary requests carry no state: use the last routed pair as a hint.
          const hint = lastGood.get(spec.id)
          const found = findRoutable(
            { m: norm(hint?.m, spec.ladder.length), k: norm(hint?.k, config.keys.length) },
            spec.ladder.length,
            config.keys.length,
            () => true,
          )
          pair = found ?? { m: 0, k: 0 }
        } else {
          const found = findRoutable(
            { m: state.m, k: state.k },
            spec.ladder.length,
            config.keys.length,
            () => true,
          )
          pair = found ?? { m: 0, k: 0 }
          state = { ...state, ...pair }
          lastGood.set(spec.id, pair)
          await writeState(event.sessionID, spec, state)
        }

        const concreteModel = spec.ladder[pair.m]
        const key = config.keys[pair.k].value

        // Rewrite body `model` to the concrete ladder entry.
        let body: Record<string, unknown> = {}
        try {
          const parsed = await event.request.clone().json()
          if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
            body = parsed as Record<string, unknown>
          }
        } catch {
          // No JSON body to preserve (should not happen for chat completions).
        }
        body.model = concreteModel

        // Rewrite the endpoint to this key's provider (same OpenAI-compatible path).
        const original = new URL(event.request.url)
        const placeholder = new URL(config.baseURL)
        let suffix = original.pathname
          .replace(placeholder.pathname.replace(/\/+$/, ""), "")
          .replace(/^\/+/, "")
        if (!suffix) suffix = "chat/completions"
        const target = new URL(config.baseURL.replace(/\/+$/, "") + "/" + suffix + original.search)

        const headers = new Headers(event.request.headers)
        headers.set("Authorization", `Bearer ${key}`)

        event.request = new Request(target, {
          method: event.request.method,
          headers,
          body: JSON.stringify(body),
        })
      },
      { providerID: PROVIDER_ID },
    )

    // Rotate on failures OpenCode would otherwise leave terminal. The next
    // attempt re-runs http.request, which picks up the advanced pair.
    await ctx.session.hook(
      "retry",
      async (event) => {
        if (event.model.providerID !== PROVIDER_ID) return
        const spec = SPECS.find((s) => s.id === event.model.id)
        if (!spec) return

        const message = event.error?.message
        if (!isRotatableError(message)) return

        const config = groups[spec.group]
        if (config.keys.length === 0) return

        const epoch = epochOf(spec.id)
        const stored = await readState(event.sessionID, spec)
        const state = normalizeState(stored, spec.ladder.length, config.keys.length, epoch)
        const totalPairs = spec.ladder.length * config.keys.length

        // One full walk covers every pair exactly once; then surface the error.
        if (state.tried >= totalPairs || event.attempt > totalPairs) {
          event.decision = { retry: false }
          return
        }

        const next = advanceState(state, spec.ladder.length, config.keys.length, epoch)
        await writeState(event.sessionID, spec, next)
        event.decision = { retry: true, delay: 0 }
      },
      { providerID: PROVIDER_ID },
    )

    // --- command ------------------------------------------------------------

    await ctx.command.transform((editor) => {
      editor.add({
        name: "multi-router",
        description: "Show the Groq/Cohere router ladders and discovered keys, or reset a ladder",
        execute: async ({ sessionID, prompt }) => {
          const raw = (prompt.text ?? "").replace(/^\/?multi-router\b/i, "").trim()
          const parts = raw.toLowerCase().split(/\s+/).filter(Boolean)

          if (parts[0] === "reset") {
            const target = parts[1]
            const ids = SPECS.filter((s) => !target || s.group === target).map((s) => s.id)
            if (ids.length === 0) {
              await ctx.session.synthetic({
                sessionID,
                text: "multi-router: unknown group (use groq or cohere)",
              })
              return
            }
            for (const id of ids) {
              epochs.set(id, epochOf(id) + 1)
              lastGood.delete(id)
            }
            // Drop persisted states so the next prompt starts at the head.
            const page = await ctx.storage.scan({ prefix: "router:", limit: 1000 })
            for (const entry of page.entries) {
              if (ids.some((id) => entry.key.endsWith(":" + id))) {
                await ctx.storage.remove(entry.key)
              }
            }
            await ctx.session.synthetic({
              sessionID,
              text: `multi-router: reset (${ids.join(", ")}) — next request starts at the ladder head`,
            })
            return
          }

          const lines: string[] = ["multi-router ladders:"]
          for (const spec of SPECS) {
            const count = groups[spec.group].keys.length
            lines.push(
              `  ${(PROVIDER_ID + "/" + spec.id).padEnd(26)}${ladderLabel(spec)}  [${count} keys]`,
            )
          }
          for (const group of ["groq", "cohere"] as const) {
            const labels = groups[group].keys.map((k) => k.label)
            lines.push(
              `${group} keys: ` +
                (labels.length ? labels.join(", ") : `none — set ${group.toUpperCase()}_API_KEY_1, ...`),
            )
          }
          lines.push(
            "advance: key first (all keys on one model), then next model; holds (model, key) until it errors",
          )
          lines.push("/multi-router reset [groq|cohere] — restart a ladder at the head")
          await ctx.session.synthetic({ sessionID, text: lines.join("\n") })
        },
      })
    })

    return () => {
      // In-memory epochs and last-good hints go away with the plugin; persisted
      // router states are left intact so an in-flight session can resume.
      epochs.clear()
      lastGood.clear()
    }
  },
})
