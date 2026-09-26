/**
 * Tests for the KV-stable context controller (kv-control.ts).
 *
 * Load-bearing properties:
 *  - under budget → pure append (no fold, zero perturbation, full cache);
 *  - the flat zone (attended window) and pins are never folded;
 *  - W is the only hard wall: the controller escalates rather than exceeding it;
 *  - amortization: a NARROWER hysteresis band lowers the worst-case per-turn
 *    perturbation (gentler continuity) at the cost of more fold events — i.e.
 *    the band width really is the cost↔continuity operating knob;
 *  - deterministic.
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { SummaryTree } from '../../src/adaptive/summary-tree.js';
import {
  replayControlled,
  planControlledFrontier,
  foldDepthCap,
  type ControlOptions,
} from '../../src/adaptive/kv-control.js';
import { buildChronicleWithChain, type MockChronicle } from './harness.js';
import type { PickerInputs } from '../../src/adaptive/picker.js';
import type { PickerChunk } from '../../src/adaptive/picker.js';

function inputsOf(ch: MockChronicle): PickerInputs {
  return {
    chunks: ch.chunks, summaries: ch.summaries, recallPairTokens: ch.recallPairTokens,
    headTokens: 0, tailTokens: 0, headChunkIds: new Set(), tailChunkIds: new Set(),
  };
}

function session(): PickerInputs {
  return inputsOf(buildChronicleWithChain({
    chunkCount: 90, tokensPerChunk: 1000, mergeThreshold: 6, recallPairTokens: 200,
  }));
}

const BASE: ControlOptions = {
  windowTokens: 60_000,
  budgetTokens: 30_000,
  attendedWindowTokens: 6_000,
  mergeThreshold: 6,
  markerCount: 4,
  targetSteps: 90,
};

test('under budget → pure append: no folds, zero perturbation, full cache', () => {
  // 12 small chunks, generous budget → never folds.
  const ch = buildChronicleWithChain({ chunkCount: 12, tokensPerChunk: 500, mergeThreshold: 6, recallPairTokens: 200 });
  const r = replayControlled(inputsOf(ch), { ...BASE, budgetTokens: 1_000_000, windowTokens: 2_000_000, highWatermark: 1_000_000 });
  assert.equal(r.foldEvents, 0, 'never folds under budget');
  assert.equal(r.maxPerturbation, 0, 'no prefix churn');
  for (let i = 1; i < r.steps.length; i++) {
    assert.equal(r.steps[i].perturbation, 0, `step ${i} pure append`);
  }
});

test('flat zone and pins are never folded', () => {
  const ch = buildChronicleWithChain({ chunkCount: 90, tokensPerChunk: 1000, mergeThreshold: 6, recallPairTokens: 200 });
  // Pin a deep-old chunk that the controller would otherwise fold.
  ch.chunks[3].pinned = true;
  const inputs = inputsOf(ch);
  const r = replayControlled(inputs, { ...BASE });
  assert.ok(r.foldEvents > 0, 'fixture should force folding');

  // foldDepthCap must report 0 for the pin and for the flat zone, at any age.
  const tree = new SummaryTree(inputs);
  const flat = new Set<string>([ch.chunks[89].id]);
  assert.equal(foldDepthCap(ch.chunks[3], 89, new Set(), 6, 6), 0, 'pinned chunk cap 0');
  assert.equal(foldDepthCap(ch.chunks[89], 89, flat, 6, 6), 0, 'flat-zone chunk cap 0');
  // An old un-pinned chunk should be allowed to fold.
  assert.ok(foldDepthCap(ch.chunks[10], 89, new Set(), 6, 6) >= 1, 'old chunk foldable');
  void tree;
});

test('W is the only hard wall: escalates instead of exceeding it', () => {
  // Tiny window, lots of pinned (unfoldable) content → cannot fit under W.
  const ch = buildChronicleWithChain({ chunkCount: 60, tokensPerChunk: 1000, mergeThreshold: 6, recallPairTokens: 200 });
  for (const c of ch.chunks) c.pinned = true; // nothing may fold
  const r = replayControlled(inputsOf(ch), {
    ...BASE, windowTokens: 20_000, budgetTokens: 10_000, highWatermark: 10_000, attendedWindowTokens: 2_000,
  });
  assert.ok(r.escalations > 0, 'unfoldable content over W must escalate, not silently overflow');
});

test('trust region P never distorts the outcome — only the payment schedule (§13)', () => {
  // Rev 5.0 retires the old premise (tighter reach → gentler): the old
  // controller achieved gentleness by folding the WRONG (recent) content —
  // the mechanism behind the 2026-07-12 inversion incident. Under the
  // single-path solve, P bounds per-turn perturbation when the relevance-
  // correct repair can be amortized (suffix adoption), and is overridden when
  // it can't — so WHAT the profile converges to is P-invariant.
  const inputs = session();
  const shallow = replayControlled(inputs, { ...BASE, reachTokens: 8_000 });
  const deep = replayControlled(inputs, { ...BASE, reachTokens: 60_000 });

  assert.equal(shallow.escalations, 0, 'tight P stays feasible');
  assert.equal(deep.escalations, 0, 'wide P stays feasible');
  const lastS = shallow.steps[shallow.steps.length - 1];
  const lastD = deep.steps[deep.steps.length - 1];
  assert.equal(lastS.renderedTokens, lastD.renderedTokens, 'same final render size');
  assert.equal(lastS.deepestLevel, lastD.deepestLevel, 'same final fold depth');
  // The relevance-correct shed folds at the prefix start, which cannot be
  // amortized below the invalidation floor — so a tight P pays the same bill.
  assert.equal(
    shallow.totalRecomputed, deep.totalRecomputed,
    'P does not change what gets folded at these settings',
  );
});

test('bidirectional: under budget with a folded F_prev, un-folds to use headroom', () => {
  const inputs = session(); // 72 chunks × 1000 = 72k raw
  const tree = new SummaryTree(inputs);
  const now = 71;
  // 1) Fold tight (expandAt 0 → fold-only this call).
  const folded = planControlledFrontier(inputs, tree, {
    previous: new Map(), foldAtTokens: 18_000, expandAtTokens: 0, targetTokens: 18_000,
    windowTokens: 72_000, rawZone: new Set(), now, mergeThreshold: 6,
  });
  assert.ok(folded.folded && !folded.expanded, 'first call folds down');

  // 2) From that folded state, a generous budget should UN-FOLD to use headroom.
  const expanded = planControlledFrontier(inputs, tree, {
    previous: folded.resolutions, foldAtTokens: 50_000, expandAtTokens: 50_000, targetTokens: 50_000,
    windowTokens: 72_000, rawZone: new Set(), now, mergeThreshold: 6,
  });
  assert.ok(expanded.expanded && !expanded.folded, 'un-folds, does not fold');
  assert.ok(expanded.tokens > folded.tokens, `used headroom: ${expanded.tokens} > ${folded.tokens}`);
  assert.ok(expanded.tokens <= 50_000, `stays within target: ${expanded.tokens} ≤ 50000`);

  // 3) Dead band: within [expandAt, foldAt] nothing changes (zero perturbation).
  const quiet = planControlledFrontier(inputs, tree, {
    previous: expanded.resolutions, foldAtTokens: 60_000, expandAtTokens: 40_000, targetTokens: 50_000,
    windowTokens: 72_000, rawZone: new Set(), now, mergeThreshold: 6,
  });
  assert.ok(!quiet.folded && !quiet.expanded, 'in-band → no perturbation');
});

test('controller is deterministic', () => {
  const inputs = session();
  const a = replayControlled(inputs, { ...BASE });
  const b = replayControlled(inputs, { ...BASE });
  assert.deepEqual(a, b);
});

test('foldDepthCap is monotone non-decreasing in age and capped', () => {
  const mk = (seq: number): PickerChunk => ({
    id: `c${seq}`, sequence: seq, rawTokens: 100, currentResolution: 0, lockedByAgent: false, pinned: false,
  });
  let prev = -1;
  for (let age = 0; age <= 5000; age += 50) {
    const cap = foldDepthCap(mk(0), age, new Set(), 10, 6);
    assert.ok(cap >= 0 && cap <= 8, 'cap in [0, MAX_FOLD_LEVEL]');
    assert.ok(cap >= prev, 'non-decreasing in age');
    prev = cap;
  }
});

// Prompt-cache-aware timing (opt-in): refold when the cache is cold anyway,
// defer voluntary refolds while it is warm (Ian + Opus 5.5, 2026-09-26).
test('cache-aware: warm defers a voluntary expansion; cold adopts it; a required shed still happens warm', () => {
  const inputs = session();
  const tree = new SummaryTree(inputs);
  const now = 71;
  const folded = planControlledFrontier(inputs, tree, {
    previous: new Map(), foldAtTokens: 18_000, expandAtTokens: 0, targetTokens: 18_000,
    windowTokens: 72_000, rawZone: new Set(), now, mergeThreshold: 6,
  });
  const roomy = {
    previous: folded.resolutions, foldAtTokens: 50_000, expandAtTokens: 50_000, targetTokens: 50_000,
    windowTokens: 72_000, rawZone: new Set<string>(), now, mergeThreshold: 6,
  };
  // Classic: expands into the headroom (a prefix change).
  const classic = planControlledFrontier(inputs, tree, roomy);
  assert.ok(classic.expanded && classic.perturbation > 0);

  // Warm: the same voluntary expansion is deferred, zero perturbation, flagged.
  const warm = planControlledFrontier(inputs, tree, { ...roomy, deferWhenWarm: true });
  assert.equal(warm.deferred, true);
  assert.equal(warm.perturbation, 0);
  assert.ok(!warm.expanded && !warm.folded);
  assert.equal(warm.tokens, folded.tokens);

  // Cold: the change lands (same frontier as classic), marked coldAdopt.
  const cold = planControlledFrontier(inputs, tree, { ...roomy, cacheCold: true });
  assert.equal(cold.coldAdopt, true);
  assert.equal(cold.deferred, undefined);
  assert.ok(cold.expanded);

  // A required shed (carried over foldAt) is NOT deferred, warm or not.
  const raw = planControlledFrontier(inputs, tree, {
    previous: new Map(), foldAtTokens: 72_000, expandAtTokens: 0, targetTokens: 72_000,
    windowTokens: 72_000, rawZone: new Set(), now, mergeThreshold: 6,
  });
  const mustShed = planControlledFrontier(inputs, tree, {
    previous: raw.resolutions, foldAtTokens: 30_000, expandAtTokens: 20_000, targetTokens: 25_000,
    windowTokens: 72_000, rawZone: new Set(), now, mergeThreshold: 6, deferWhenWarm: true,
  });
  assert.ok(mustShed.folded, 'over foldAt: sheds even when warm');
  assert.notEqual(mustShed.deferred, true);
  assert.ok(mustShed.tokens <= 30_000, `back under foldAt: ${mustShed.tokens}`);
});

test('cache-aware: an in-band hold is a plain hold, not a deferral; unset state = classic ladder', () => {
  const inputs = session();
  const tree = new SummaryTree(inputs);
  const now = 71;
  const base = planControlledFrontier(inputs, tree, {
    previous: new Map(), foldAtTokens: 50_000, expandAtTokens: 0, targetTokens: 50_000,
    windowTokens: 72_000, rawZone: new Set(), now, mergeThreshold: 6,
  });
  const band = {
    previous: base.resolutions, foldAtTokens: 60_000, expandAtTokens: 40_000, targetTokens: 50_000,
    windowTokens: 72_000, rawZone: new Set<string>(), now, mergeThreshold: 6,
  };
  const held = planControlledFrontier(inputs, tree, { ...band, deferWhenWarm: true });
  assert.equal(held.perturbation, 0);
  assert.equal(held.deferred, undefined, 'nothing was waiting: not a deferral');
  const classic = planControlledFrontier(inputs, tree, band);
  assert.deepEqual([...classic.resolutions], [...held.resolutions]);
});
