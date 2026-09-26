/**
 * kvStableCacheAware end to end: the host-reported prompt-cache state reaches
 * the kv-stable solver through a real ContextManager compile.
 */
import { describe, it, before, after } from 'node:test';
import assert from 'node:assert';
import { rmSync, existsSync } from 'node:fs';
import { ContextManager, AutobiographicalStrategy } from '../src/index.js';
import type { ContentBlock, NormalizedRequest } from '@animalabs/membrane';

const STORE = './test-zz-cache-aware';
const t = (text: string): ContentBlock => ({ type: 'text', text });
const cleanup = () => { if (existsSync(STORE)) rmSync(STORE, { recursive: true, force: true }); };
const summarizer = () => ({
  complete: async (_r: NormalizedRequest) => ({
    stopReason: 'end_turn',
    content: [{ type: 'text', text: 'A brief memory of this stretch. ' + 'detail '.repeat(10) }],
    usage: { input_tokens: 1000, output_tokens: 40 },
  }),
});
const strategy = (cacheAware: boolean) => new AutobiographicalStrategy({
  compressionModel: 'zz-model', targetChunkTokens: 120, headWindowTokens: 0, recentWindowTokens: 400,
  hierarchical: true, mergeThreshold: 3, adaptiveResolution: true, foldingStrategy: 'kv-stable',
  kvStableCacheAware: cacheAware,
} as ConstructorParameters<typeof AutobiographicalStrategy>[0]);

describe('kvStableCacheAware', () => {
  before(async () => {
    cleanup();
    const m = await ContextManager.open({ path: STORE, strategy: strategy(false), membrane: summarizer() as any });
    for (let i = 0; i < 120; i++) {
      m.addMessage(i % 2 ? 'agent' : 'user', [t(`turn ${i}: substantive traffic with specifics `.repeat(4))]);
      for (let k = 0; k < 200 && !m.isReady(); k++) await m.tick();
    }
    await m.compile({ maxTokens: 4_000, reserveForResponse: 0 }); // fold tight
    for (let k = 0; k < 500 && !m.isReady(); k++) await m.tick();
    await m.close();
  });
  after(cleanup);

  it('warm: a bigger budget is deferred (layout held, flagged); cold: adopted', async () => {
    const s = strategy(true);
    const m = await ContextManager.open({ path: STORE, strategy: s, membrane: summarizer() as any });
    await m.compile({ maxTokens: 4_000, reserveForResponse: 0 });
    const tight = m.getRenderStats()!.total.tokens;

    s.setPromptCacheState('warm');
    await m.compile({ maxTokens: 16_000, reserveForResponse: 0 });
    assert.equal(s.isRefoldDeferred(), true, 'the expansion waits for a cold cache');
    assert.equal(m.getRenderStats()!.total.tokens, tight, 'layout held');

    s.setPromptCacheState('cold');
    await m.compile({ maxTokens: 16_000, reserveForResponse: 0 });
    assert.equal(s.isRefoldDeferred(), false);
    assert.ok(m.getRenderStats()!.total.tokens > tight, 'expanded once cold');
    await m.close();
  });

  it('off (default): the reported state is ignored', async () => {
    const s = strategy(false);
    const m = await ContextManager.open({ path: STORE, strategy: s, membrane: summarizer() as any });
    await m.compile({ maxTokens: 4_000, reserveForResponse: 0 });
    const tight = m.getRenderStats()!.total.tokens;
    s.setPromptCacheState('warm');
    await m.compile({ maxTokens: 16_000, reserveForResponse: 0 });
    assert.equal(s.isRefoldDeferred(), false);
    assert.ok(m.getRenderStats()!.total.tokens > tight, 'classic: expands immediately');
    await m.close();
  });
});
