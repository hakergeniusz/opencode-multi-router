# opencode-multi-router

Native OpenCode V2 port of `pi-multi-router`. Registers four router models that
walk a model ladder and rotate multi-account API keys until something answers:

| Router model | Ladder |
|---|---|
| `multirouter/gpt-oss` | `openai/gpt-oss-120b` → `openai/gpt-oss-20b` |
| `multirouter/qwen` | `qwen/qwen3.8-27b` |
| `multirouter/command-a` | `command-a-plus-05-2026` → `command-a-03-2025` |
| `multirouter/command-r` | `command-r-plus-08-2024` → `command-r-08-2024` |

Each ladder entry is tried on **every** key before the next model: key index
advances fastest, model index slowest. A `(model, key)` pair that succeeds is
held — nothing advances on success. A new prompt or a completed turn starts a
fresh walk. Groq and Cohere rate-limit per key and per model, so rotating keys
multiplies capacity without changing per-request limits.

## How it works

- `ctx.provider.transform` registers one virtual OpenAI-compatible provider,
  `multirouter`, with the four router models (limits set via
  `ctx.model.transform`).
- An `http.request` hook applies the chosen pair at the last possible moment:
  it rewrites the request URL to the key's provider endpoint, sets
  `Authorization: Bearer <key>`, and replaces the body's `model` field with the
  concrete ladder entry. Non-primary requests (compaction, title, generate)
  reuse the last routed pair as a hint instead of advancing.
- A `retry` hook advances the pair on failures OpenCode would otherwise treat
  as terminal (401, 403, 429, 5xx, timeouts, connection errors). A walk covers
  every `(model, key)` pair exactly once; when it is exhausted the error is
  surfaced. This preserves the original extension's key feature: an invalid
  key (401) rotates to the next key instead of stalling.
- Router state is persisted with `ctx.storage` and keyed by session + model, so
  sessions rotate independently. A new prompt resets the failure counter
  (`prompt` / `experimental.idle` hooks).

## Mapping from the pi original

| pi concept | OpenCode V2 equivalent |
|---|---|
| `registerVirtualModel` + `route()` | `ctx.provider.transform` + `ctx.model.transform` + `session.hook("http.request")` |
| multi-account `groq-keyN` / `cohere-keyN` providers | keys discovered from env vars / `ctx.options`; endpoint + key applied in `http.request` |
| `agent_before_settle` takeover | `session.hook("retry")` forcing `event.decision = { retry: true, delay: 0 }` and advancing state |
| `turn_end` / `before_agent_start` | `session.hook("experimental.idle")` / `session.hook("prompt")` |
| session-entry router state | `ctx.storage` (key `router:<sessionID>:<modelID>`) |
| `/multi-router` command | `ctx.command.transform` (`multi-router`) |
| `ctx.ui.notify(...)` output | `ctx.session.synthetic({ sessionID, text })` |

## Load

Add the plugin to `opencode.json(c)`:

```jsonc
{
  "plugins": ["opencode-multi-router"]
}
```

With options (keys should stay in environment variables, not config files):

```jsonc
{
  "plugins": [
    {
      "package": "opencode-multi-router",
      "options": {
        "groqBaseURL": "https://api.groq.com/openai/v1",
        "cohereBaseURL": "https://api.cohere.com/v1"
      }
    }
  ]
}
```

Select one of the router models in OpenCode (e.g. `/model multirouter/gpt-oss`).

## Keys (never hardcoded)

Keys are discovered only at setup, in this order, and are de-duplicated:

- `GROQ_API_KEY_1`, `GROQ_API_KEY_2`, … (numerically sorted) and `COHERE_API_KEY_1`, …
- `GROQ_API_KEY` / `COHERE_API_KEY` (single account)
- `ctx.options.groqKeys` / `ctx.options.cohereKeys` (arrays, from the `plugins` object form)

No key is written to disk, logged, or printed by the status command — only
labels such as `groq-key1` are shown. Endpoint overrides: `GROQ_BASE_URL` /
`COHERE_BASE_URL` env vars or `groqBaseURL` / `cohereBaseURL` options.

## Commands

- `/multi-router` — show ladders and discovered key **labels**
- `/multi-router reset [groq|cohere]` — restart one or both ladders at the head

## Tests

`test/router-core.test.ts` ports the original rotation tests (rotation order,
hold-on-success, stale-epoch restart, corrupt-state normalization, missing-pair
skipping, ladder wrap, and the rotatable-error classifier) against the pure
core module, with no pi dependency. Run with `bun test`. There is no TUI part —
the original extension was not TUI-focused.

## Known gaps / differences

- **Model identity:** OpenCode keeps `multirouter/<id>` selected and rewrites
  only the outgoing HTTP call. Token/cost accounting and message metadata
  therefore reflect the virtual router model, not the concrete ladder entry.
  (`experimental.model.resolve` cannot switch providers, which is why the
  rewrite happens at the HTTP layer.)
- **Attempt ceiling:** OpenCode enforces its own hard maximum retry count. A
  ladder with more `(model, key)` pairs than that ceiling will stop early. This
  is a host limit the plugin cannot override.
- **OpenAI-compatible assumption:** both providers are assumed to expose an
  OpenAI-compatible `chat/completions` endpoint and Bearer auth. Non-standard
  gateways may need `groqBaseURL` / `cohereBaseURL` tuning.
- **Key discovery:** the original read `~/.pi/agent/multi-account.json`; this
  port expects env vars or plugin options instead (no pi runtime dependency).
- **Epoch persistence:** reset epochs are in-memory, so a process restart
  resumes at the ladder head rather than preserving an in-flight reset.
