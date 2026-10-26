#!/usr/bin/env python3
"""The Muse serve streams exactly the text Glimmer generated (plan Task 10b). CPU only; real Muse tokenizer.
Run: pixi run --no-install python3 pyhost/test_stream_exact.py -v"""
import json
import os
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from serve_protocol import AtemServeProtocol, AtemStreamParser, atem_exact_fields, normalize_atem_messages  # noqa: E402

TOK_DIR = Path(os.environ.get("MUSE_TOKENIZER", os.path.expanduser("~/nomos_data/muse-glimmer-tokenizer")))
# Glimmer's own split of an emoji, from a real generation: 85275 is ' ' plus the emoji's first bytes, 148676 the rest.
SPLIT = [85275, 148676]
# A real thought: 4,500 tokens Glimmer generated for Task 14's long-thought measurement (header, then all thought).
LONG = Path(__file__).resolve().parent.parent / "tools/muse_long_thought_tokens.json"


class WholeBodyParser(AtemStreamParser):
    """The parser as it was before ledger R58, the reference for the incremental one: it decoded the WHOLE body again at
    every token, held back a text ending in U+FFFD while the body could still grow, and emitted what lay past the text
    it had emitted. The incremental parser must emit the same deltas at the same tokens."""

    def _begin_body(self):
        super()._begin_body()
        self.body_emitted = ""

    def _reset_header(self):
        super()._reset_header()
        self.body_emitted = ""

    def _body_delta(self, final):
        if self.recipient not in ("self", "user") or not self.body_ids:
            return []
        full = self.tokenizer.decode(self.body_ids, skip_special_tokens=True)
        if not final and full.endswith("�"):
            return []
        if len(full) <= len(self.body_emitted):
            return []
        delta, self.body_emitted = full[len(self.body_emitted):], full
        if self.recipient == "self":
            self.reasoning_emitted += delta
            return [("reasoning_content", delta)]
        self.content_emitted += delta
        return [("content", delta)]


@unittest.skipUnless(TOK_DIR.is_dir(), f"Muse tokenizer absent: {TOK_DIR}")
class ExactStreamTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tok = AutoTokenizer.from_pretrained(TOK_DIR)
        cls.proto = AtemServeProtocol(cls.tok, "muse")

    def enc(self, s):
        return self.tok.encode(s, add_special_tokens=False)

    def stream(self, ids):
        """What a streaming client joins: every feed() delta, then finish()'s residual, as the SSE path sends them."""
        p = self.proto.stream_parser(self.tok)
        got = {"reasoning_content": "", "content": ""}
        for t in ids:
            for f, d in p.feed(t):
                got[f] += d
        _, residual = p.finish()
        for f, d in residual:
            got[f] += d
        return got

    def render(self, msgs):
        s = self.tok.apply_chat_template(normalize_atem_messages(msgs), tokenize=False, add_generation_prompt=True,
                                         enable_thinking=True)
        return self.enc(self.proto.finish_prompt(s, True))

    def test_the_split_is_a_real_split(self):
        self.assertTrue(self.tok.decode(SPLIT[:1]).endswith("�"))
        self.assertNotIn("�", self.tok.decode(SPLIT))

    def test_a_character_split_across_tokens_streams_whole(self):
        body = self.enc("Probably") + SPLIT + self.enc(" or sun")
        got = self.stream(self.enc(" to=self<|message|>") + body + [self.proto.eom_id])
        self.assertEqual(got["reasoning_content"], self.tok.decode(body, skip_special_tokens=True))
        self.assertNotIn("�", got["reasoning_content"])

    def test_a_split_character_that_ends_a_body_is_emitted_at_eom_and_at_stop(self):
        body = self.enc("Sky") + SPLIT
        for end in (self.proto.eom_id, self.proto.eot_id):
            got = self.stream(self.enc(" to=user<|message|>") + body + [end])
            self.assertEqual(got["content"], self.tok.decode(body, skip_special_tokens=True))

    def test_a_body_cut_off_between_the_halves_is_flushed_by_finish(self):
        body = self.enc("Sky") + SPLIT[:1]                          # max_tokens stopped it mid-character
        got = self.stream(self.enc(" to=user<|message|>") + body)
        self.assertEqual(got["content"], self.tok.decode(body, skip_special_tokens=True))

    def test_nonstream_fields_are_the_streamed_text_unstripped(self):
        gen = (self.enc(" to=self<|message|>\nThinking it over.\n") + [self.proto.eom_id]
               + self.enc("<|start|>assistant to=user<|message|> Blue") + SPLIT + self.enc(" \n") + [self.proto.eot_id])
        reasoning, content, parsed = atem_exact_fields(self.tok, self.proto, gen)
        got = self.stream(gen)
        self.assertEqual((reasoning, content), (got["reasoning_content"], got["content"]))
        self.assertEqual(reasoning, "\nThinking it over.\n")
        self.assertEqual(content, self.tok.decode(self.enc(" Blue") + SPLIT + self.enc(" \n"), skip_special_tokens=True))
        self.assertIsNone(parsed.tool_call)

    def test_the_history_a_client_sends_back_re_renders_to_the_held_ids(self):
        msgs = [{"role": "user", "content": "Three emoji for the sky?"}]
        prompt = self.render(msgs)
        gen = (self.enc(" to=self<|message|>Maybe") + SPLIT + self.enc(" works.") + [self.proto.eom_id]
               + self.enc("<|start|>assistant to=user<|message|>Here:") + SPLIT + [self.proto.eot_id])
        held = prompt + gen[:-1]                                     # the stop token is sampled, never stepped
        reasoning, content, _ = atem_exact_fields(self.tok, self.proto, gen)
        nxt = self.render(msgs + [{"role": "assistant", "reasoning_content": reasoning, "content": content},
                                  {"role": "user", "content": "And one more?"}])
        self.assertEqual(nxt[:len(held)], held)
        self.assertEqual(nxt[len(held)], self.proto.eot_id)

    def test_a_half_character_ending_a_body_is_flushed_by_the_terminator(self):
        # A leading newline/space keeps finish()'s stripped fallback from masking a lost tail.
        for head, field, end in ((" to=self<|message|>\nThink", "reasoning_content", self.proto.eom_id),
                                 (" to=user<|message|> Sky", "content", self.proto.eom_id),
                                 (" to=user<|message|> Sky", "content", self.proto.eot_id)):
            with self.subTest(field=field, end=end):
                p = self.proto.stream_parser(self.tok)
                for t in self.enc(head) + SPLIT[:1]:
                    p.feed(t)
                self.assertEqual(p.feed(end), [(field, " �")])      # the terminator itself returns the held tail

    def test_a_body_with_leading_whitespace_cut_off_between_the_halves_is_flushed_by_finish(self):
        body = self.enc(" Sky") + SPLIT[:1]
        got = self.stream(self.enc(" to=user<|message|>") + body)
        self.assertEqual(got["content"], self.tok.decode(body, skip_special_tokens=True))

    def test_a_real_replacement_character_is_delayed_never_lost(self):
        fffd = self.enc("�")            # one token (id 4468): a complete U+FFFD, indistinguishable from a half
        for tail, end in ((self.enc("b"), [self.proto.eot_id]), ([], [self.proto.eot_id]), ([], [])):
            with self.subTest(tail=tail, end=end):
                got = self.stream(self.enc(" to=user<|message|> a") + fffd + tail + end)
                self.assertEqual(got["content"], " a�" + self.tok.decode(tail))

    # -- ledger R58: the body is decoded incrementally --------------------------------------------------------------------
    def long_thought(self):
        """The real 4,500-token thought, with Glimmer's split emoji put in every 97 tokens and a real U+FFFD token every 499,
        so the long body holds characters back again and again. Returns (header, body)."""
        ids = json.loads(LONG.read_text())["tokens"]
        head, body = ids[:3], ids[3:]
        self.assertEqual(self.tok.decode(head, skip_special_tokens=False), " to=self<|message|>")
        fffd = self.enc("�")
        out = []
        for i, t in enumerate(body):
            out.append(t)
            if i % 97 == 96:
                out += SPLIT
            if i % 499 == 498:
                out += fffd
        return head, out

    def trace(self, parser, ids):
        """Every delta a parser emits for ids, fed as the serve feeds them (it stops at the stop token), with the index of
        the token that produced it (len(ids) for finish())."""
        out = []
        for i, t in enumerate(ids):
            out += [(i, f, d) for f, d in parser.feed(t)]
            if parser.saw_stop:
                break
        _, residual = parser.finish()
        return out + [(len(ids), f, d) for f, d in residual]

    def test_a_long_thought_streams_exactly(self):
        head, body = self.long_thought()
        self.assertGreaterEqual(len(body), 4000)
        whole = self.tok.decode(body, skip_special_tokens=True)
        self.assertNotIn("�" * 2, whole)
        for end in ([], [self.proto.eom_id]):           # cut off by max_tokens (finish() flushes it), or closed by <|eom|>
            with self.subTest(end=end):
                got = self.stream(head + body + end)
                self.assertEqual(got["reasoning_content"], whole)
                self.assertEqual(got["content"], "")

    def test_the_incremental_decode_emits_what_the_whole_body_decode_emitted(self):
        """Delta for delta, at the same tokens: the long thought followed by an answer, and 400 random replies built from
        tokens that split characters, complete them, stand alone as continuation bytes, or are a real U+FFFD."""
        head, body = self.long_thought()
        answer = self.enc("<|start|>assistant to=user<|message|> Ana is a knave") + SPLIT + [self.proto.eot_id]
        cases = [head + body + [self.proto.eom_id] + answer]
        pool = self.enc(" sky, ¬p ∧ q ↔ r. Done") + SPLIT + [SPLIT[0]] * 3 + [SPLIT[1]] * 2 + self.enc("�") * 2
        rng = random.Random(58)
        for _ in range(400):
            reply = []
            for _ in range(rng.randint(1, 3)):
                reply += self.enc(rng.choice(["<|start|>assistant to=self<|message|>",
                                              "<|start|>assistant to=user<|message|>"]))
                reply += [rng.choice(pool) for _ in range(rng.randint(0, 12))]
                reply.append(rng.choice([self.proto.eom_id, self.proto.eom_id, self.proto.eot_id]))
            if rng.random() < 0.3:
                reply = reply[:rng.randint(1, len(reply))]       # cut off by max_tokens anywhere, even mid-header
            cases.append(reply[len(self.enc("<|start|>assistant")):])  # the template has already emitted that much
        for k, ids in enumerate(cases):
            with self.subTest(case=k):
                new = self.trace(self.proto.stream_parser(self.tok), ids)
                self.assertEqual(new, self.trace(WholeBodyParser(self.tok, self.proto), ids))


if __name__ == "__main__":
    unittest.main()
