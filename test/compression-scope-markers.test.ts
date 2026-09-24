import { after, describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, rmSync } from 'node:fs';
import type { ContentBlock, NormalizedRequest, ToolDefinition } from '@animalabs/membrane';

import { ContextManager, AutobiographicalStrategy } from '../src/index.js';
import type { Chunk } from '../src/strategies/autobiographical.js';
import type { StrategyContext, SummaryEntry } from '../src/types/index.js';

// compressionScopeMarkers (opt-in): BEGIN/END markers + scope rule around the
// new L1 slice and each merge's source span, no recall pairs in merges, and a
// silent-identity suffix. Off by default: prompts are unchanged.

const BASE = './test-scope-markers';
const MARKER_SUBSTR = 'You will soon form a new memory';
let sequence = 0;
const paths: string[] = [];
function freshPath(): string { const p = `${BASE}-${sequence++}`; paths.push(p); return p; }
function cleanup(): void { for (const p of paths) if (existsSync(p)) rmSync(p, { recursive: true, force: true }); }

function text(t: string): ContentBlock { return { type: 'text', text: t }; }

function capturingMembrane(stopReason: 'end_turn' | 'refusal' = 'end_turn') {
  const calls: NormalizedRequest[] = [];
  return {
    calls,
    membrane: {
      complete: async (request: NormalizedRequest) => {
        calls.push(structuredClone(request));
        if (stopReason === 'refusal') {
          return { content: [], stopReason: 'refusal', usage: { inputTokens: 100, outputTokens: 0 },
            raw: { response: { stop_details: { category: 'cyber' } } } };
        }
        return { content: [text('memory body')], stopReason: 'end_turn', usage: { inputTokens: 100, outputTokens: 20 } };
      },
    } as never,
  };
}

class ProbeStrategy extends AutobiographicalStrategy {
  seed(entry: SummaryEntry): void { this.pushSummary(entry); }
  run(chunk: Chunk, ctx: StrategyContext): Promise<void> { return this.compressChunkHierarchical(chunk, ctx); }
  runMerge(level: number, sourceIds: string[], ctx: StrategyContext): Promise<void> {
    return (this as unknown as { executeMerge(l: number, ids: string[], c: StrategyContext): Promise<void> })
      .executeMerge(level, sourceIds, ctx);
  }
}

function managerContext(manager: ContextManager): StrategyContext {
  return (manager as unknown as { createStrategyContext(): StrategyContext }).createStrategyContext();
}

function summary(id: string, first: string, last: string, sourceIds: string[]): SummaryEntry {
  return { id, level: 1, content: `authored ${id}`, tokens: 20, sourceLevel: 0, sourceIds, sourceRange: { first, last }, created: Number(id.replace(/\D/g, '')) || 1 };
}

function texts(req: NormalizedRequest): string[] {
  return req.messages.flatMap((m) => m.content.filter((b): b is Extract<ContentBlock, { type: 'text' }> => b.type === 'text').map((b) => b.text));
}
function recallIds(req: NormalizedRequest): string[] {
  return texts(req).flatMap((t) => { const m = /^\[CM\] Recall memory (.+)\.$/.exec(t); return m ? [m[1]!] : []; });
}
function blockTypes(req: NormalizedRequest): Set<string> {
  const s = new Set<string>(); for (const m of req.messages) for (const b of m.content) s.add(b.type); return s;
}


async function build(membrane: unknown, scope: boolean | undefined, identityReminder?: string, extra: Record<string, unknown> = {}) {
  const strategy = new ProbeStrategy({
    compressionModel: 'same-model',
    targetChunkTokens: 100,
    recentWindowTokens: 0,
    headWindowTokens: 50,
    autoTickOnNewMessage: false,
    minChunkCharsForLLM: 0,
    mergeThreshold: 99,
    compressionScopeMarkers: scope,
    identityReminder,
    ...extra,
  } as never);
  const manager = await ContextManager.open({ path: freshPath(), strategy, membrane: membrane as never });
  const ids: string[] = [];
  for (let i = 0; i < 12; i++) ids.push(manager.addMessage(i % 2 ? 'Claude' : 'User', [text(`raw-${i} ` + 'substantive '.repeat(12))]));
  strategy.seed(summary('L1-100', ids[2]!, ids[3]!, [ids[2]!, ids[3]!]));
  strategy.seed(summary('L1-200', ids[6]!, ids[7]!, [ids[6]!, ids[7]!]));
  strategy.seed(summary('L1-300', ids[8]!, ids[9]!, [ids[8]!, ids[9]!]));
  const all = managerContext(manager).messageStore.getAll();
  const targetMessages = all.filter((m) => [ids[10]!, ids[11]!].includes(m.id));
  const target: Chunk = { index: 999, startIndex: 10, endIndex: 12, messages: targetMessages, tokens: 100, compressed: false };
  return { manager, strategy, target };
}

describe('compressionScopeMarkers', () => {
  after(cleanup);

  it('L1: markers, scope rule and silent-identity suffix when on; recall still present', async () => {
    const { calls, membrane } = capturingMembrane();
    const fx = await build(membrane, true, 'You are Probe.');
    await fx.strategy.run(fx.target, managerContext(fx.manager));
    const all = texts(calls[0]!).join('\n');
    assert.match(all, /\[BEGIN NEW SOURCE SLICE/);
    assert.match(all, /\[END NEW SOURCE SLICE\]/);
    assert.match(all, /Scope boundary: write only the events/);
    assert.match(all, /earlier raw messages shown verbatim/);
    assert.match(all, /You are Probe\.\n\nApply this identity and attribution guidance silently/);
    assert.ok(recallIds(calls[0]!).length > 0, 'L1 keeps continuity recall');
  });

  it('L1: unchanged when off (no markers, no suffix)', async () => {
    const { calls, membrane } = capturingMembrane();
    const fx = await build(membrane, undefined, 'You are Probe.');
    await fx.strategy.run(fx.target, managerContext(fx.manager));
    const all = texts(calls[0]!).join('\n');
    assert.doesNotMatch(all, /BEGIN NEW SOURCE SLICE|Scope boundary|apply silently|guidance silently/);
  });

  it('merge: source-span markers and no recall pairs when on; recall pairs when off', async () => {
    const on = capturingMembrane();
    const a = await build(on.membrane, true);
    await a.strategy.runMerge(2, ['L1-200', 'L1-300'], managerContext(a.manager));
    const req = on.calls[0]!;
    assert.deepEqual(recallIds(req), [], 'merges replay no earlier summaries');
    const all = texts(req).join('\n');
    assert.match(all, /\[BEGIN MERGE SOURCE SPAN/);
    assert.match(all, /\[END MERGE SOURCE SPAN\]/);

    const off = capturingMembrane();
    const b = await build(off.membrane, undefined);
    await b.strategy.runMerge(2, ['L1-200', 'L1-300'], managerContext(b.manager));
    assert.ok(recallIds(off.calls[0]!).includes('L1-100'), 'default merges still recall earlier summaries');
    assert.doesNotMatch(texts(off.calls[0]!).join('\n'), /MERGE SOURCE SPAN/);
  });

  it('merge: compressionMergeRecall keeps the span markers and restores earlier recall as context', async () => {
    const cap = capturingMembrane();
    const fx = await build(cap.membrane, true, undefined, { compressionMergeRecall: true });
    await fx.strategy.runMerge(2, ['L1-200', 'L1-300'], managerContext(fx.manager));
    const req = cap.calls[0]!;
    assert.ok(recallIds(req).includes('L1-100'), 'earlier summary is visible to the merge');
    const all = texts(req).join('\n');
    assert.match(all, /\[BEGIN MERGE SOURCE SPAN/);
    assert.match(all, /\[END MERGE SOURCE SPAN\]/);
  });
});
