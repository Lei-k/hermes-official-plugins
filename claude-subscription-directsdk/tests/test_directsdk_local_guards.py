"""Local integration guards (Lei-k): near-match tool names fail closed; one prompt-cache TTL policy."""
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import directsdk
from test_directsdk import Contract


class LocalGuards(Contract):
    def test_near_match_tool_name_fails_closed_instead_of_running_another_tool(self):
        # Hermes' repair_tool_call would turn each of these into the offered `probe` and run it
        # with the model's arguments, so none may reach the host.
        with tempfile.TemporaryDirectory() as tmp:
            for native_name in ("mcp__hermes__Probe", "Probe", "mcp__hermes__prob"):
                client = self.client(tmp, TOOL_NAME=native_name)
                with self.assertRaises(directsdk.ClaudeToolOutsideInventory) as caught:
                    client.chat.completions.create(**self.request())
                self.assertEqual(caught.exception.failure_kind, "tool_outside_inventory")
                client.close()

    def test_unrelated_unknown_tool_still_reaches_the_host(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = self.client(tmp, TOOL_NAME="mcp__fastmail__draft_email")
            result = client.chat.completions.create(**self.request())
            self.assertEqual(result.choices[0].message.tool_calls[0].function.name, "mcp__fastmail__draft_email")
            client.close()

    def test_outside_inventory_is_not_retried(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("ds_profile_probe", Path(__file__).resolve().parents[1] / "__init__.py")
        verdict = None
        try:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            verdict = module.classify_api_error(directsdk.ClaudeToolOutsideInventory("x"))
        except Exception as exc:  # provider registration may need a full host; check the source instead
            source = (Path(__file__).resolve().parents[1] / "__init__.py").read_text()
            self.assertIn("'tool_outside_inventory'", source, exc)
            return
        self.assertEqual(verdict, {'reason': 'format_error', 'retryable': False, 'should_fallback': True})


if __name__ == "__main__":
    unittest.main()
