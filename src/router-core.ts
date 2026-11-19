/**
 * Pure routing core for opencode-multi-router.
 *
 * No runtime dependencies, no `@opencode/plugin` import, no secrets: this module
 * only does integer bookkeeping over a (model, key) grid. Kept separate so the
 * rotation behaviour can be unit-tested without a running OpenCode host.
 */

export interface RouterSpec {
  /** Provider group the ladder belongs to. */
  group: "groq" | "cohere"
  /** Virtual model id shown in the model picker. */
  id: string
  /** Display name. */
  name: string
  /** Concrete backend model ids, newest first. */
  ladder: string[]
  contextWindow: number
  maxTokens: number
}

export interface RouterState {
  /** Index into the ladder. */
  m: number
  /** Index into the discovered key list. */
  k: number
  /** Reset epoch; a mismatch restarts the walk at the head. */
  epoch: number
  /** Number of (model, key) pairs that failed in the current walk. */
  tried: number
}

/** Coerce any stored value to a valid index. NaN/negative/out-of-range never poison the walk. */
export function norm(value: number | undefined, length: number): number {
  if (!Number.isInteger(value)) return 0
  const n = length > 0 ? length : 1
  return (((value as number) % n) + n) % n
}

export function initialState(epoch: number): RouterState {
  return { m: 0, k: 0, epoch, tried: 0 }
}

/** Normalize a persisted state; stale epochs and corrupt values restart at the head. */
export function normalizeState(
  input: RouterState | undefined,
  ladderLength: number,
  keyCount: number,
  epoch: number,
): RouterState {
  if (!input || input.epoch !== epoch) return initialState(epoch)
  return {
    m: norm(input.m, ladderLength),
    k: norm(input.k, keyCount),
    epoch,
    tried: Number.isInteger(input.tried) && (input.tried as number) > 0 ? (input.tried as number) : 0,
  }
}

/** Advance the pair by one step: key index fastest, model index slowest. */
export function advanceState(
  state: RouterState,
  ladderLength: number,
  keyCount: number,
  epoch: number,
): RouterState {
  const l = Math.max(1, ladderLength)
  const k = Math.max(1, keyCount)
  const total = l * k
  const from = norm(state.m, l) * k + norm(state.k, k)
  const next = (from + 1) % total
  return {
    m: Math.floor(next / k) % l,
    k: next % k,
    epoch,
    tried: state.tried + 1,
  }
}

/**
 * First routable pair at or after `from`, wrapping once. A pair is skipped when
 * `isAvailable` says the (model, key) combination does not exist.
 */
export function findRoutable(
  from: { m: number; k: number },
  ladderLength: number,
  keyCount: number,
  isAvailable: (m: number, k: number) => boolean,
): { m: number; k: number } | undefined {
  const l = Math.max(1, ladderLength)
  const k = Math.max(1, keyCount)
  const total = l * k
  const start = norm(from.m, l) * k + norm(from.k, k)
  for (let step = 0; step < total; step++) {
    const idx = (start + step) % total
    const m = Math.floor(idx / k) % l
    const key = idx % k
    if (isAvailable(m, key)) return { m, k: key }
  }
  return undefined
}

/**
 * Errors a different key (or the next model) can still answer. Pi's own retry
 * classifier does not retry 401 or a quota error that reads as billing, so the
 * router forces those to rotate instead of sitting on a broken key forever.
 */
export const ROTATABLE_ERROR_PATTERN =
  /401|403|invalid.?api.?key|Incorrect API key|Unauthorized|PERMISSION_DENIED|429|rate.?limit|too many requests|quota|RESOURCE_EXHAUSTED|overloaded|high demand|UNAVAILABLE|500|502|503|504|timed? out|timeout|connection.?error|fetch failed/i

export function isRotatableError(message: string | undefined): boolean {
  return typeof message === "string" && ROTATABLE_ERROR_PATTERN.test(message)
}
