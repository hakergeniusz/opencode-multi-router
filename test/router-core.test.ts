/**
 * Ported (pi-dependency-free) tests for the rotation core.
 * Original pi-multi-router/test.ts drove a fake ExtensionAPI; these tests keep
 * the same behavioural coverage against the pure core module instead.
 * Run with `bun test`.
 */

import { describe, expect, test } from "bun:test"
import {
  advanceState,
  findRoutable,
  initialState,
  isRotatableError,
  norm,
  normalizeState,
  type RouterState,
} from "../src/router-core.js"

const LADDER = 2
const KEYS = 3

/** Walk `n` retries from a state and return the resulting state. */
function retries(state: RouterState, n: number): RouterState {
  let current = state
  for (let i = 0; i < n; i++) current = advanceState(current, LADDER, KEYS, 0)
  return current
}

describe("index normalization", () => {
  test("NaN and negative values normalize to integers", () => {
    expect(Number.isInteger(norm(NaN, LADDER))).toBe(true)
    expect(Number.isInteger(norm(-7, KEYS))).toBe(true)
  })

  test("out-of-range values wrap deterministically", () => {
    expect(norm(99, LADDER)).toBe(1)
    expect(norm(57, KEYS)).toBe(0)
  })

  test("non-integers fall back to zero", () => {
    expect(norm(1.5, LADDER)).toBe(0)
    expect(norm(undefined, KEYS)).toBe(0)
  })
})

describe("(model, key) rotation", () => {
  test("head state starts at model 0, key 0", () => {
    expect(initialState(0)).toEqual({ m: 0, k: 0, epoch: 0, tried: 0 })
  })

  test("key index advances fastest, model holds", () => {
    expect(advanceState(initialState(0), LADDER, KEYS, 0)).toMatchObject({ m: 0, k: 1, tried: 1 })
  })

  test("after all keys fail the model steps down", () => {
    // 2 retries: head (0,0) -> (0,1) -> (0,2) -> (1,0)
    expect(retries(initialState(0), 3)).toMatchObject({ m: 1, k: 0 })
  })

  test("an exhausted walk wraps to the head", () => {
    // 2 models * 3 keys = 6 pairs, so 6 advances wrap to (0,0).
    expect(retries(initialState(0), 6)).toMatchObject({ m: 0, k: 0, tried: 6 })
  })

  test("single-model ladders rotate only the key", () => {
    let state = initialState(0)
    expect(advanceState(state, 1, 3, 0)).toMatchObject({ m: 0, k: 1 })
    for (let i = 0; i < 3; i++) state = advanceState(state, 1, 3, 0)
    expect(state).toMatchObject({ m: 0, k: 0 })
  })
})

describe("state normalization / epochs", () => {
  test("stale epoch restarts at the ladder head", () => {
    const stale = normalizeState({ m: 1, k: 2, epoch: 999, tried: 4 }, LADDER, KEYS, 0)
    expect(stale).toEqual({ m: 0, k: 0, epoch: 0, tried: 0 })
  })

  test("matching epoch keeps normalized indices and tried counter", () => {
    const kept = normalizeState({ m: 99, k: 57, epoch: 0, tried: 2 }, LADDER, KEYS, 0)
    expect(kept).toEqual({ m: 1, k: 0, epoch: 0, tried: 2 })
  })

  test("missing state falls back to the head", () => {
    expect(normalizeState(undefined, LADDER, KEYS, 0)).toEqual({ m: 0, k: 0, epoch: 0, tried: 0 })
  })
})

describe("pair availability", () => {
  test("skips missing (model, key) pairs", () => {
    // key2 and key3 lack model 0; after advancing from (0,0) to (0,1), the
    // next routable pair is model 1 / key 1.
    const available = (m: number, k: number) => !(m === 0 && k > 0)
    expect(findRoutable({ m: 0, k: 1 }, LADDER, KEYS, available)).toEqual({ m: 1, k: 0 })
  })

  test("wraps when nothing after the start is routable", () => {
    const available = (m: number, k: number) => m === 0 && k === 0
    expect(findRoutable({ m: 1, k: 2 }, LADDER, KEYS, available)).toEqual({ m: 0, k: 0 })
  })

  test("returns undefined when no pair is routable", () => {
    expect(findRoutable({ m: 0, k: 0 }, LADDER, KEYS, () => false)).toBeUndefined()
  })
})

describe("rotatable errors", () => {
  test("401 / 429 / 5xx / timeout rotate", () => {
    for (const message of [
      "401 Invalid API key",
      "429 too many requests",
      "503 Service Unavailable",
      "fetch failed",
      "Request timed out",
    ]) {
      expect(isRotatableError(message)).toBe(true)
    }
  })

  test("user aborts and ordinary errors do not rotate", () => {
    expect(isRotatableError("Request was aborted by user")).toBe(false)
    expect(isRotatableError("bad request")).toBe(false)
    expect(isRotatableError(undefined)).toBe(false)
  })
})
