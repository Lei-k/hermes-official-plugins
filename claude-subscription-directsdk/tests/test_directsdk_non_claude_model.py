"""A non-Claude model id fails before native starts, as model_not_found; every Claude spelling still spawns."""
import asyncio
import json
import os
import sys
from types import SimpleNamespace

import pytest

import directsdk

NATIVE = '''import json, os, pathlib, sys
log = pathlib.Path(os.environ["HOME"]) / "spawns.jsonl"
with log.open("a") as out:
    out.write(json.dumps(sys.argv[sys.argv.index("--model") + 1]) + "\\n")
for line in sys.stdin: pass
print(json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "ok"}], "id": "m", "stop_reason": "end_turn"}}), flush=True)
print(json.dumps({"type": "stream_event", "event": {"type": "message_stop"}}), flush=True)
print(json.dumps({"type": "result", "subtype": "success", "usage": {"input_tokens": 1, "output_tokens": 1}}), flush=True)
'''

FOREIGN = ['gpt-6-astra', 'gpt-6-astra[1m]', 'gemini-3-pro', 'o3', 'o4-mini', 'o1-preview',
           'llama-4-maverick', 'meta-llama/llama-4', 'mistral-large', 'deepseek-v4', 'grok-5', ' GPT-6-ASTRA[1M] ']
# Pinned ids, their [1m] forms, plugin and CLI aliases, short names, and ids no table knows yet.
CLAUDE = ['claude-opus-5-5', 'claude-opus-5-5[1m]', 'sonnet', 'opus', 'haiku', 'fable', 'default', 'best',
          'opusplan', 'opusplan[1m]', 'opus[1m]', 'claude-haiku-4-5', 'sonnet-5.5', 'Opus', 'claude-foo-9',
          'claude-foo-9[1m]', 'sonnet-9', 'anthropic/claude-opus-5-5']


def _client(tmp_path, create_client=directsdk.Client):
    script = tmp_path / 'native.py'
    script.write_text(NATIVE)
    return create_client(command=[sys.executable, str(script)], env={'PATH': os.environ['PATH'], 'HOME': str(tmp_path)})


def _spawns(tmp_path):
    log = tmp_path / 'spawns.jsonl'
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def _request(model, streaming=False):
    return dict(model=model, messages=[{'role': 'system', 'content': 's'}, {'role': 'user', 'content': 'go'}],
                stream=streaming, timeout=SimpleNamespace(read=10))


@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('model', FOREIGN)
def test_non_claude_model_never_spawns_native(tmp_path, model, streaming):
    client = _client(tmp_path)
    try:
        with pytest.raises(Exception) as raised:
            result = client.chat.completions.create(**_request(model, streaming))
            if streaming:
                list(result)
        assert not client._requests
    finally:
        client.close()
    assert _spawns(tmp_path) == [], f'native spawned for {model}; error={raised.value!r}'
    assert getattr(raised.value, 'status_code', None) == 404
    assert model in str(raised.value) and 'claude-subscription-directsdk-experimental' in str(raised.value)


def test_non_claude_model_is_refused_on_the_async_path(tmp_path):
    client = _client(tmp_path)

    async def call():
        return await client.create(**_request('gpt-6-astra'))

    try:
        with pytest.raises(directsdk.ClaudeModelNotFound):
            asyncio.run(call())
        assert not client._requests
    finally:
        client.close()
    assert _spawns(tmp_path) == []


def test_non_claude_model_falls_back_instead_of_retrying(profile, tmp_path):
    from agent.error_classifier import FailoverReason, classify_api_error

    client = _client(tmp_path, profile.create_client)
    try:
        with pytest.raises(RuntimeError) as raised:
            client.create(**_request('gpt-6-astra'))
    finally:
        client.close()
    assert profile.classify_api_error(raised.value) == {'reason': 'model_not_found', 'retryable': False, 'should_fallback': True}
    verdict = classify_api_error(raised.value, provider=profile.name, model='gpt-6-astra')
    assert (verdict.reason, verdict.retryable, verdict.should_fallback) == (FailoverReason.model_not_found, False, True)
    # Hermes core reads the same verdict off the 404 and its text alone, without the profile hook.
    verdict = classify_api_error(raised.value, provider='', model='gpt-6-astra')
    assert (verdict.reason, verdict.retryable, verdict.should_fallback) == (FailoverReason.model_not_found, False, True)


@pytest.mark.parametrize('model', CLAUDE)
def test_claude_models_and_aliases_still_spawn_native(tmp_path, model):
    client = _client(tmp_path)
    try:
        result = client.chat.completions.create(**_request(model))
    finally:
        client.close()
    assert result.choices[0].message.content == 'ok'
    assert _spawns(tmp_path) == [directsdk.native_model(model)]


def test_guard_allows_every_route_the_profile_knows(profile):
    from model_catalog import names_claude

    # Discovery and the picker are untouched; the guard must never refuse a route they offer.
    for model in (*profile.fallback_models, *profile.model_aliases, *profile.model_aliases.values()):
        assert names_claude(model), model
        assert names_claude(model + '[1m]'), model
    for model in FOREIGN:
        assert not names_claude(model), model
