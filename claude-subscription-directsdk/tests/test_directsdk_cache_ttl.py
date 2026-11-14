"""Every native spawn gets one prompt-cache TTL, whatever the host environment forces."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_directsdk
from test_directsdk import Contract

# The fake native writes the TTL-related variables it was started with, then answers as usual.
RECORD = r"""
import json as _j, os as _o
if _o.environ.get('TTL_RECORD'):
    open(_o.environ['TTL_RECORD'], 'w').write(_j.dumps({k: _o.environ.get(k) for k in (
        'FORCE_PROMPT_CACHING_5M', 'CLAUDE_CODE_PROMPT_CACHE_TTL', 'CLAUDE_CODE_SUBAGENT_PROMPT_CACHE_TTL')}))
"""


class CacheTTL(Contract):
    def spawn_env(self, tmp, **kw):
        original = test_directsdk.FAKE
        test_directsdk.FAKE = RECORD + original
        try:
            record = Path(tmp) / "ttl.json"
            client = self.client(tmp, TTL_RECORD=str(record), **kw)
            client.chat.completions.create(**self.request())
            client.close()
            return json.loads(record.read_text())
        finally:
            test_directsdk.FAKE = original

    def test_inherited_5m_forcing_is_replaced_by_the_1h_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = self.spawn_env(tmp, FORCE_PROMPT_CACHING_5M="1", CLAUDE_CODE_PROMPT_CACHE_TTL="5m")
        self.assertEqual(env, {'FORCE_PROMPT_CACHING_5M': None, 'CLAUDE_CODE_PROMPT_CACHE_TTL': '1h',
                               'CLAUDE_CODE_SUBAGENT_PROMPT_CACHE_TTL': '1h'})

    def test_plugin_setting_selects_5m_and_invalid_values_fall_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = self.spawn_env(tmp, CLAUDE_SUBSCRIPTION_DIRECTSDK_CACHE_TTL="5m")
        self.assertEqual(env['CLAUDE_CODE_PROMPT_CACHE_TTL'], '5m')
        self.assertEqual(env['CLAUDE_CODE_SUBAGENT_PROMPT_CACHE_TTL'], '5m')
        with tempfile.TemporaryDirectory() as tmp:
            env = self.spawn_env(tmp, CLAUDE_SUBSCRIPTION_DIRECTSDK_CACHE_TTL="1d")
        self.assertEqual(env['CLAUDE_CODE_PROMPT_CACHE_TTL'], '1h')


if __name__ == "__main__":
    unittest.main()
