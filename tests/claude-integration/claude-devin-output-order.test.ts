import { afterAll, afterEach, beforeEach, describe, expect, mock, test } from "bun:test";
import { createDevinMessagesOutputOrder } from "../../src/claude/devin-output-order";
import { collectAnthropicMessage, responsesSseToAnthropicSse } from "../../src/claude/outbound";
import { bridgeToResponsesSSE } from "../../src/bridge";
import { createTestTranslatorBudget } from "../helpers/translator-budget";
import { encodeDevinSignature } from "../../src/adapters/devin/reasoning-signature";
import { mapOcxMessagesToDevin } from "../../src/adapters/devin";
import { messagesToResponsesTranslation } from "../../src/protocols/codecs/messages";
import { parseRequest } from "../../src/responses/parser";
import type { AdapterEvent, OcxConfig, OcxProviderConfig } from "../../src/types";
import type { ProviderAdapter } from "../../src/adapters/base";
import { createTempHome } from "../helpers/temp-home";
import { acquireOwnedSpendHome } from "../helpers/owned-spend-home";
import { installIsolatedCodexHome } from "../helpers/isolated-codex-home";

const signature = encodeDevinSignature("sealed.v1.synthetic-attestation", "sealed");
const usage = { inputTokens: 12, outputTokens: 3, totalTokens: 15 };
const terminal: AdapterEvent = { type: "done", usage };
let upstreamEvents: AdapterEvent[] = [];
const resolver = await import("../../src/server/adapter-resolve");
const originalResolver = { ...resolver };
mock.module("../../src/server/adapter-resolve", () => ({ ...originalResolver,
  resolveAdapter(provider: OcxProviderConfig, cache?: "none" | "short" | "long") {
    if (provider.adapter !== "devin") return originalResolver.resolveAdapter(provider, cache);
    return {
      name: "devin",
      buildRequest: () => ({ url: provider.baseUrl, method: "POST", headers: {}, body: "" }),
      async *parseStream() { yield terminal; },
      async runTurn(_parsed, _incoming, emit) { upstreamEvents.forEach(emit); },
    } satisfies ProviderAdapter;
  },
}));
const { handleClaudeMessages } = await import("../../src/server/claude-messages");
const { handleResponses } = await import("../../src/server/responses");
afterAll(() => { mock.module("../../src/server/adapter-resolve", () => originalResolver); });

let home: ReturnType<typeof createTempHome>;
let codexHome: ReturnType<typeof installIsolatedCodexHome>;
let releaseSpend: () => void;
beforeEach(() => {
  home = createTempHome("ocx-claude-devin-output-");
  codexHome = installIsolatedCodexHome("ocx-claude-devin-codex-");
  releaseSpend = acquireOwnedSpendHome();
  upstreamEvents = [];
});
afterEach(() => { releaseSpend(); codexHome.restore(); home.remove(); });

function ordered(events: AdapterEvent[]) {
  const budget = createTestTranslatorBudget();
  const abort = new AbortController();
  const output: AdapterEvent[] = [];
  const order = createDevinMessagesOutputOrder(event => output.push(event), budget, abort.signal, () => abort.abort());
  try { events.forEach(order.emit); order.flush(); } finally { order.dispose(); }
  return { output, budget };
}

async function message(events: AdapterEvent[]) {
  const { output, budget } = ordered(events);
  const source = (async function* () { yield* output; })();
  const bridged = bridgeToResponsesSSE(source, "swe-2", undefined, undefined, undefined, undefined, 0, { translatorBudget: budget });
  return collectAnthropicMessage(responsesSseToAnthropicSse(bridged, "devin/swe-2", {
    translatorBudget: budget, pingIntervalMs: 0,
  }), "devin/swe-2", budget);
}

describe("Devin Messages late signatures", () => {
  test("late signature keeps the answer last, signed reasoning intact, and exact usage", async () => {
    const result = await message([
      { type: "thinking_delta", thinking: "A thought" },
      { type: "text_delta", text: "O" }, { type: "text_delta", text: "K" },
      { type: "thinking_signature", signature }, terminal,
    ]);
    expect(result.content).toEqual([
      { type: "thinking", thinking: "A thought", signature }, { type: "text", text: "OK" },
    ]);
    expect(result.stop_reason).toBe("end_turn");
    expect(result.usage).toMatchObject({ input_tokens: 12, output_tokens: 3 });
  });

  test("signature-only turns preserve both the signature and final text", async () => {
    const result = await message([
      { type: "text_delta", text: "OK" }, { type: "thinking_signature", signature }, terminal,
    ]);
    expect(result.content).toEqual([
      { type: "thinking", thinking: "", signature }, { type: "text", text: "OK" },
    ]);
  });

  test("a signed tool turn survives Messages inbound replay to the Devin prompt", async () => {
    const result = await message([
      { type: "thinking_delta", thinking: "Read the file" },
      { type: "tool_call_start", id: "call_read", name: "Read" },
      { type: "tool_call_delta", arguments: '{"file_path":"/synthetic/file.txt"}' },
      { type: "tool_call_end" }, { type: "thinking_signature", signature }, terminal,
    ]);
    expect(result.stop_reason).toBe("tool_use");
    expect(result.content.at(-1)).toMatchObject({ type: "tool_use", id: "call_read", name: "Read",
      input: { file_path: "/synthetic/file.txt" } });
    const translated = messagesToResponsesTranslation({ model: "devin/swe-2", max_tokens: 64,
      messages: [{ role: "user", content: "Read" }, { role: "assistant", content: result.content },
        { role: "user", content: [{ type: "tool_result", tool_use_id: "call_read", content: "marker" }] }],
    }, undefined, createTestTranslatorBudget());
    const history = mapOcxMessagesToDevin(parseRequest(translated.body));
    expect(history.find(row => row.role === "assistant")).toMatchObject({
      thinking: "Read the file", signature: "sealed.v1.synthetic-attestation", signature_type: "sealed",
    });
    expect(history.find(row => row.role === "tool")?.content).toBe("marker");
  });

  test("independently signed reasoning blocks retain their associations", async () => {
    const result = await message([
      { type: "thinking_delta", thinking: "First" }, { type: "thinking_signature", signature: "sig-first" },
      { type: "text_delta", text: "answer" },
      { type: "thinking_delta", thinking: "Second" }, { type: "thinking_signature", signature: "sig-second" }, terminal,
    ]);
    expect(result.content).toEqual([
      { type: "thinking", thinking: "First", signature: "sig-first" },
      { type: "thinking", thinking: "Second", signature: "sig-second" }, { type: "text", text: "answer" },
    ]);
  });

  for (const ending of [terminal, { type: "incomplete", reason: "max_tokens", usage },
    { type: "error", status: 502, message: "synthetic reset", usage }] as AdapterEvent[]) {
    test(`${ending.type} preserves partial content and the original terminal`, () => {
      const { output, budget } = ordered([{ type: "text_delta", text: "partial" }, ending]);
      expect(output.filter(event => event.type !== "heartbeat")).toEqual([{ type: "text_delta", text: "partial" }, ending]);
      expect(budget.snapshot().currentBytes).toBe(0);
    });
  }

  test("holding output keeps progress live and cancellation releases it immediately", () => {
    const budget = createTestTranslatorBudget();
    const abort = new AbortController();
    const output: AdapterEvent[] = [];
    const order = createDevinMessagesOutputOrder(event => output.push(event), budget, abort.signal, () => abort.abort());
    order.emit({ type: "text_delta", text: "still generating" });
    expect(output).toEqual([{ type: "heartbeat" }]);
    expect(budget.snapshot().currentBytes).toBeGreaterThan(0);
    order.emit({ type: "heartbeat", replayUnsafe: true });
    expect(output.at(-1)).toEqual({ type: "heartbeat", replayUnsafe: true });
    abort.abort();
    expect(budget.snapshot().currentBytes).toBe(0);
    order.emit(terminal);
    order.dispose();
    expect(output.some(event => event.type === "text_delta" || event.type === "done")).toBe(false);
  });

  test("overflow emits one typed error, aborts the producer, and releases held output", () => {
    const budget = createTestTranslatorBudget({ maxTurnBytes: 160 });
    const abort = new AbortController();
    const output: AdapterEvent[] = [];
    const order = createDevinMessagesOutputOrder(event => output.push(event), budget, abort.signal, () => abort.abort());
    order.emit({ type: "text_delta", text: "small" });
    order.emit({ type: "text_delta", text: "x".repeat(200) });
    order.emit(terminal);
    expect(output.filter(event => event.type === "error")).toHaveLength(1);
    expect(output.at(-1)).toMatchObject({ type: "error", status: 413, code: "translation_buffer_limit" });
    expect(abort.signal.aborted).toBe(true);
    expect(budget.snapshot().currentBytes).toBe(0);
    order.dispose();
  });
});

const config = (): OcxConfig => ({ port: 0, defaultProvider: "cognition-custom", claudeCode: { enabled: true },
  providers: { "cognition-custom": { adapter: "devin", baseUrl: "https://synthetic.invalid", apiKey: "synthetic-key", models: ["swe-2"] } },
});

for (const stream of [true, false]) {
  test(`actual Messages ingress orders a renamed Devin provider, stream=${stream}`, async () => {
    upstreamEvents = [{ type: "text_delta", text: "OK" }, { type: "thinking_signature", signature }, terminal];
    const response = await handleClaudeMessages(new Request("http://localhost/v1/messages", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: "cognition-custom/swe-2", max_tokens: 64, stream,
        messages: [{ role: "user", content: "Reply OK" }] }),
    }), config(), {});
    expect(response.status).toBe(200);
    const result = stream ? await collectAnthropicMessage(response.body!, "cognition-custom/swe-2", createTestTranslatorBudget())
      : await response.json();
    expect(result.content.at(-1)).toEqual({ type: "text", text: "OK" });
    expect(result.content[0]).toEqual({ type: "thinking", thinking: "", signature });
  });
}

test("the Responses ingress retains incremental text before the late signature", async () => {
  upstreamEvents = [{ type: "text_delta", text: "OK" }, { type: "thinking_signature", signature }, terminal];
  const response = await handleResponses(new Request("http://localhost/v1/responses", {
    method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ model: "cognition-custom/swe-2", input: "Reply OK", stream: true }),
  }), config(), {});
  const text = await response.text();
  expect(text.indexOf("response.output_text.delta")).toBeLessThan(text.indexOf("\"type\":\"reasoning\""));
});
