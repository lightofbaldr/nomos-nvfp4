#!/usr/bin/env python3
"""serve._fit_request, the prompt/capacity guard (CPU; importing serve loads no model)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import serve  # noqa: E402
from fastapi import HTTPException  # noqa: E402


class FitRequestTest(unittest.TestCase):
    def setUp(self):
        self.saved = (serve.MAX_SEQ, serve.PREFILL_CHUNK, serve.MAX_PROMPT_TOKENS)
        serve.MAX_SEQ, serve.MAX_PROMPT_TOKENS = 32768, 6200

    def tearDown(self):
        serve.MAX_SEQ, serve.PREFILL_CHUNK, serve.MAX_PROMPT_TOKENS = self.saved

    def test_unchunked_keeps_the_prompt_cap(self):
        serve.PREFILL_CHUNK = 0
        with self.assertRaises(HTTPException) as cm:
            serve._fit_request(6201, 64)
        self.assertEqual(cm.exception.status_code, 413)
        self.assertEqual(serve._fit_request(6200, 64), 64)

    def test_chunked_accepts_a_long_conversation(self):
        serve.PREFILL_CHUNK = 2048
        self.assertEqual(serve._fit_request(20000, 2048), 2048)

    def test_generation_is_clamped_to_the_capacity(self):
        serve.PREFILL_CHUNK = 2048
        self.assertEqual(serve._fit_request(32000, 2048), 32768 - 32000 - 1)

    def test_a_full_context_is_refused(self):
        serve.PREFILL_CHUNK = 2048
        with self.assertRaises(HTTPException) as cm:
            serve._fit_request(32767, 64)
        self.assertEqual(cm.exception.status_code, 413)


if __name__ == "__main__":
    unittest.main()
