import { describe, it } from 'node:test';
import assert from 'node:assert';
import type { ContentBlock } from '@animalabs/membrane';
import { moveSkipReasonsToResults, SKIP_REASON_PLACEHOLDER } from '../src/index.js';

const skip = (id: string, reason: string) => ({ type: 'tool_use', id, name: 'skip_reply', input: { reason } }) as unknown as ContentBlock;
const res = (id: string, content: unknown = '{"skipped":true,"note":"Turn ended; nothing sent."}') => ({ type: 'tool_result', toolUseId: id, content }) as unknown as ContentBlock;
const use = (id: string) => ({ type: 'tool_use', id, name: 'fn', input: { reason: 'not a skip' } }) as unknown as ContentBlock;
const e = (participant: string, ...content: ContentBlock[]) => ({ participant, content });

describe('liveSkipReasonInResult view', () => {
  it('keeps the call, moves the full reason onto the result side', () => {
    const out = moveSkipReasonsToResults([e('agent', skip('s1', 'a long reflection')), e('user', res('s1'))]);
    const call = out[0].content[0] as unknown as { name: string; input: { reason: string } };
    assert.equal(call.name, 'skip_reply');
    assert.equal(call.input.reason, SKIP_REASON_PLACEHOLDER);
    const text = JSON.stringify(out[1].content);
    assert.ok(text.includes('a long reflection'), 'reason must not be lost');
    assert.ok(text.includes('Turn ended; nothing sent.'));
  });

  it('leaves a skip whose result is not in view unchanged', () => {
    const input = [e('agent', skip('s2', 'orphan reason'))];
    assert.deepEqual(moveSkipReasonsToResults(input), input);
  });

  it('does not touch other tools, and is idempotent', () => {
    const input = [e('agent', use('u1'), skip('s3', 'r3')), e('user', res('u1', 'ok'), res('s3'))];
    const once = moveSkipReasonsToResults(input);
    assert.deepEqual(once[0].content[0], input[0].content[0]);
    assert.deepEqual(once[1].content[0], input[1].content[0]);
    assert.deepEqual(moveSkipReasonsToResults(once), once);
  });
});
