import type { ContentBlock } from '@animalabs/membrane';

/**
 * Live-view rendering for `liveSkipReasonInResult` (AutobiographicalConfig).
 *
 * Residents often write a long first-person reflection as a skip_reply
 * reason. Replayed as a tool_use payload (model output with its reasoning
 * no longer attached), a run of these can trip reasoning_extraction on
 * every later turn. This keeps each past skip_reply call, with a short
 * placeholder reason, and moves the full reason onto the result side as a
 * quoted note, so nothing is lost and the resident still sees itself
 * calling the tool. Calls whose result isn't in view are left unchanged.
 * (Librarian, 2026-09-24: live turn refused as-is, passes rendered this way.)
 */
export const SKIP_REASON_PLACEHOLDER = '(private note; kept with the result)';

type Block = ContentBlock & {
  id?: string;
  name?: string;
  input?: { reason?: unknown } & Record<string, unknown>;
  toolUseId?: string;
};

export function moveSkipReasonsToResults<T extends { content: ContentBlock[] }>(entries: readonly T[]): T[] {
  const resultIds = new Set<string>();
  for (const e of entries) {
    for (const b of e.content as Block[]) {
      if (b.type === 'tool_result' && typeof b.toolUseId === 'string') resultIds.add(b.toolUseId);
    }
  }
  const reasons = new Map<string, string>();
  return entries.map((e) => {
    let changed = false;
    const content = (e.content as Block[]).map((b) => {
      if (b.type === 'tool_use' && b.name === 'skip_reply' && typeof b.id === 'string' && resultIds.has(b.id)) {
        const reason = b.input?.reason;
        if (typeof reason === 'string' && reason && reason !== SKIP_REASON_PLACEHOLDER) {
          reasons.set(b.id, reason);
          changed = true;
          return { ...b, input: { ...b.input, reason: SKIP_REASON_PLACEHOLDER } } as ContentBlock;
        }
      }
      if (b.type === 'tool_result' && typeof b.toolUseId === 'string' && reasons.has(b.toolUseId)) {
        changed = true;
        return {
          ...b,
          content: [{ type: 'text', text: `Turn ended; nothing sent. Your private note, as you wrote it:\n\n${reasons.get(b.toolUseId)}` }],
        } as unknown as ContentBlock;
      }
      return b as ContentBlock;
    });
    return changed ? { ...e, content } : e;
  });
}
