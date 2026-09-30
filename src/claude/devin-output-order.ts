import type { AdapterEvent } from "../types";
import {
  isTranslatorBudgetExceededError,
  releaseTranslatedEvent,
  retainTranslatedEvent,
  type TranslatorBudget,
} from "../lib/translator-budget";

/**
 * Cognition signs reasoning after text/tools. Claude Code treats a trailing empty
 * thinking block as its final result, so Messages needs reasoning before content.
 * Hold only this physical turn's semantic events; keep progress and reasoning live.
 */
export function createDevinMessagesOutputOrder(
  emit: (event: AdapterEvent) => void,
  budget: TranslatorBudget,
  signal: AbortSignal,
  abort: () => void,
) {
  const held: AdapterEvent[] = [];
  let ended = false;
  const release = () => {
    for (const event of held) releaseTranslatedEvent(event, budget);
    held.length = 0;
  };
  const cancel = () => { ended = true; release(); };
  signal.addEventListener("abort", cancel, { once: true });
  if (signal.aborted) cancel();
  const flush = () => {
    if (ended) return;
    try {
      for (const event of held) {
        emit(event);
        releaseTranslatedEvent(event, budget);
      }
    } finally { release(); }
  };
  return {
    emit(event: AdapterEvent) {
      if (ended) return;
      if (event.type === "done" || event.type === "error" || event.type === "incomplete") {
        flush();
        ended = true;
        emit(event);
        return;
      }
      if (event.type === "heartbeat" || event.type === "thinking_delta"
        || event.type === "thinking_signature" || event.type === "redacted_thinking"
        || event.type === "reasoning_raw_delta" || event.type === "kiro_redacted_reasoning") {
        emit(event);
        return;
      }
      const copy = { ...event } as AdapterEvent;
      try {
        retainTranslatedEvent(copy, budget, held.at(-1));
        held.push(copy);
      } catch (error) {
        if (!isTranslatorBudgetExceededError(error)) throw error;
        ended = true;
        release();
        emit({ type: "error", status: 413, errorType: "request_too_large",
          code: "translation_buffer_limit", message: error.message });
        abort();
        return;
      }
      // Progress still feeds the stall watchdog while text/tool output waits for
      // the signature. Preserve original replay-unsafe heartbeats above verbatim.
      emit({ type: "heartbeat" });
    },
    flush,
    dispose() {
      signal.removeEventListener("abort", cancel);
      cancel();
    },
  };
}
