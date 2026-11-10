"""Synthetic-only tests for remote privacy boundaries and planner failures."""
import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from jarvis_planner import JarvisError, build_messages, extract_plan, plan_reply


class PlannerTests(unittest.TestCase):
    def test_redaction_and_speaker_identity(self):
        messages = build_messages([
            {"role": "user", "content": '[intimate photo: FICTIONAL_FIXTURE] hello'},
            {"role": "assistant", "content": "My promise"},
        ], "Memory [INTIMATE VIDEO: SECRET]")
        payload = json.loads(messages[1]["content"])
        self.assertNotIn("PRIVATE", messages[1]["content"])
        self.assertNotIn("SECRET", messages[1]["content"])
        self.assertEqual(payload["conversation"][0]["speaker"], "Sam")
        self.assertEqual(payload["conversation"][1]["speaker"], "Alex")
        self.assertEqual(payload["next_speaker"], "Alex")
        self.assertIn("[intimate photo]", payload["conversation"][0]["text"])

    def test_never_uses_private_reasoning_as_the_response_plan(self):
        with self.assertRaises(JarvisError):
            extract_plan({"choices": [{"finish_reason": "stop", "message": {
                "content": "", "reasoning_content": "hidden internal reasoning"}}]})

    def test_rejects_incomplete_plan(self):
        with self.assertRaises(JarvisError):
            extract_plan({"choices": [{"finish_reason": "length", "message": {
                "content": "An unfinished plan"}}]})

    @patch("jarvis_planner.KEY_PATH")
    @patch("jarvis_planner.urllib.request.urlopen")
    def test_live_request_contract_and_plan_extraction(self, urlopen, key_path):
        key_path.read_text.return_value = "synthetic-key"
        urlopen.return_value.__enter__.return_value.read.return_value = json.dumps({
            "choices": [{"finish_reason": "stop", "message": {
                "content": "I will answer her question directly.",
                "reasoning_content": "not passed to the texter"}}]}).encode()
        result = plan_reply([{"role": "user", "content": "Hello"}], "")
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload["chat_template_kwargs"], {"thinking": True, "reasoning_effort": "high"})
        self.assertEqual(result, "I will answer her question directly.")

    @patch("jarvis_planner.KEY_PATH")
    @patch("jarvis_planner.urllib.request.urlopen")
    def test_http_error_does_not_disclose_provider_body(self, urlopen, key_path):
        key_path.read_text.return_value = "synthetic-key"
        urlopen.side_effect = urllib.error.HTTPError("https://example.test", 429,
                                                     "private error text", {}, io.BytesIO(b"PRIVATE"))
        with self.assertRaises(JarvisError) as caught:
            plan_reply([{"role": "user", "content": "Hello"}], "")
        self.assertEqual(str(caught.exception), "Jarvis HTTP 429; response body withheld.")


if __name__ == "__main__":
    unittest.main()
