"""Per-request prompt-cache write tiers (5m/1h): reported in usage and one INFO line, never at the
expense of the provider result, the original error or process/socket cleanup.

The fake native is test_directsdk's FAKE, extended here only: ``CACHE_CREATION`` (JSON) adds the
API's ``usage.cache_creation`` split to the result event.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import directsdk
from test_directsdk import FAKE

USAGE_LINE = "u={'input_tokens':3,'output_tokens':5,'cache_read_input_tokens':7,'cache_creation_input_tokens':11,'output_tokens_details':{'thinking_tokens':4}}\n"
assert USAGE_LINE in FAKE
TIERED = FAKE.replace(USAGE_LINE, USAGE_LINE + "if 'CACHE_CREATION' in os.environ:\n u['cache_creation']=json.loads(os.environ['CACHE_CREATION'])\n")
TTL = 'CLAUDE_CODE_PROMPT_CACHE_TTL'
POLICY_TTL = 'CLAUDE_SUBSCRIPTION_DIRECTSDK_CACHE_TTL'
SECRET = 'unsafe-value@example.invalid'
INFO = 'DirectSDK cache tiers requested_ttl=%s write_5m=%s write_1h=%s'
MISMATCH = 'DirectSDK cache tier mismatch requested_ttl=%s write_5m=%s write_1h=%s'


def client(tmp_path, fake=TIERED, **env):
    script = tmp_path / 'native.py'
    script.write_text(fake)
    return directsdk.Client(command=[sys.executable, str(script)],
                            env={'PATH': os.environ['PATH'], 'HOME': str(tmp_path), **env})


def request():
    return dict(model='sonnet', messages=[{'role': 'user', 'content': 'go'}],
                tools=[{'type': 'function', 'function': {'name': 'probe', 'description': 'long ' * 600 + 'TAIL',
                        'parameters': {'type': 'object', 'properties': {'value': {'type': 'string'}}}}}])


def records(caplog, level):
    return [r.getMessage() for r in caplog.records if r.name == directsdk.logger.name and r.levelname == level]


def unknown(value):
    return 'unknown' if value is None else value


@pytest.mark.parametrize('usage,expected', [
    ({}, (None, None)),
    ({'cache_creation': None}, (None, None)),
    ({'cache_creation': 'x'}, (None, None)),
    ({'cache_creation': {'ephemeral_5m_input_tokens': 0, 'ephemeral_1h_input_tokens': 11}}, (0, 11)),
    ({'cache_creation': {'ephemeral_5m_input_tokens': 4}}, (4, None)),
    ({'cache_creation': {'ephemeral_1h_input_tokens': 0}}, (None, 0)),
    ({'cache_creation': {'ephemeral_5m_input_tokens': SECRET, 'ephemeral_1h_input_tokens': -1}}, (None, None)),
    ({'cache_creation': {'ephemeral_5m_input_tokens': True, 'ephemeral_1h_input_tokens': 2.5}}, (None, None)),
])
def test_native_cache_tiers_reads_only_valid_counts(usage, expected):
    assert directsdk.native_cache_tiers(usage, '1h') == {'requested_ttl': '1h', 'write_5m': expected[0], 'write_1h': expected[1]}


def test_native_cache_tiers_never_raises():
    assert directsdk.native_cache_tiers(None, 'unknown') == {'requested_ttl': 'unknown', 'write_5m': None, 'write_1h': None}


@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('creation,expected', [
    (None, (None, None)),
    ({'ephemeral_5m_input_tokens': 0, 'ephemeral_1h_input_tokens': 11}, (0, 11)),
    ({'ephemeral_5m_input_tokens': SECRET, 'ephemeral_1h_input_tokens': -1}, (None, None)),
])
def test_tiers_reach_usage_and_one_info_line(tmp_path, caplog, streaming, creation, expected):
    env = {TTL: '1h'} if creation is None else {TTL: '1h', 'CACHE_CREATION': json.dumps(creation)}
    native = client(tmp_path, **env)
    try:
        with caplog.at_level(logging.INFO, logger=directsdk.logger.name):
            result = native.create(**request(), stream=streaming)
            final = list(result)[-1] if streaming else result
        usage = final.usage.model_dump()
        assert usage['native_cache_tiers'] == {'requested_ttl': '1h', 'write_5m': expected[0], 'write_1h': expected[1]}
        assert usage['native_usage'].get('cache_creation') == creation
        assert usage['cache_creation_input_tokens'] == 11  # The aggregate stays as upstream normalizes it.
        assert records(caplog, 'INFO') == [INFO % ('1h', unknown(expected[0]), unknown(expected[1]))]
        assert records(caplog, 'WARNING') == []
        assert SECRET not in caplog.text and 'TAIL' not in caplog.text and 'go' not in caplog.text.split()
    finally:
        native.close()


@pytest.mark.parametrize('value,requested', [(None, 'unknown'), ('1h', '1h'), ('5m', '5m'), (SECRET, 'unknown'), ('', 'unknown')])
def test_requested_ttl_helper_sanitizes_raw_native_env(value, requested):
    env = {} if value is None else {TTL: value}
    assert directsdk._requested_cache_ttl(env) == requested


@pytest.mark.parametrize('policy,inherited,requested', [
    (None, None, '1h'),
    (None, '5m', '1h'),
    ('1h', '5m', '1h'),
    ('5m', '1h', '5m'),
    (SECRET, '5m', '1h'),
    ('', '5m', '1h'),
    (None, SECRET, '1h'),
    (None, '', '1h'),
])
def test_requested_ttl_matches_policy_and_actual_native_env(tmp_path, caplog, policy, inherited, requested):
    record = tmp_path / 'ttl.json'
    fake = TIERED.replace('rows=[]', "pathlib.Path(os.environ['TTL_RECORD']).write_text(json.dumps({k: os.environ.get(k) for k in ('CLAUDE_CODE_PROMPT_CACHE_TTL', 'CLAUDE_CODE_SUBAGENT_PROMPT_CACHE_TTL', 'FORCE_PROMPT_CACHING_5M')}))\nrows=[]")
    env = {'CACHE_CREATION': json.dumps({'ephemeral_5m_input_tokens': 0, 'ephemeral_1h_input_tokens': 0}),
           'TTL_RECORD': str(record), 'FORCE_PROMPT_CACHING_5M': '1'}
    if policy is not None:
        env[POLICY_TTL] = policy
    if inherited is not None:
        env[TTL] = inherited
    native = client(tmp_path, fake=fake, **env)
    try:
        with caplog.at_level(logging.INFO, logger=directsdk.logger.name):
            result = native.create(**request())
        assert json.loads(record.read_text()) == {
            TTL: requested, 'CLAUDE_CODE_SUBAGENT_PROMPT_CACHE_TTL': requested,
            'FORCE_PROMPT_CACHING_5M': None}
        assert result.usage.model_dump()['native_cache_tiers']['requested_ttl'] == requested
        assert records(caplog, 'INFO') == [INFO % (requested, 0, 0)]
        assert SECRET not in caplog.text
    finally:
        native.close()


@pytest.mark.parametrize('requested,creation', [
    ('1h', {'ephemeral_5m_input_tokens': 3, 'ephemeral_1h_input_tokens': 8}),
    ('5m', {'ephemeral_5m_input_tokens': 3, 'ephemeral_1h_input_tokens': 8}),
])
def test_opposite_tier_warns_once_per_client(tmp_path, caplog, requested, creation):
    env = {POLICY_TTL: requested, 'CACHE_CREATION': json.dumps(creation)}
    native = client(tmp_path, **env)
    try:
        with caplog.at_level(logging.INFO, logger=directsdk.logger.name):
            for _ in range(2):
                native.create(**request())
        assert records(caplog, 'WARNING') == [MISMATCH % (requested, 3, 8)]
        assert len(records(caplog, 'INFO')) == 2
    finally:
        native.close()
    other = client(tmp_path, **env)
    try:
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=directsdk.logger.name):
            other.create(**request())
        assert records(caplog, 'WARNING') == [MISMATCH % (requested, 3, 8)]
    finally:
        other.close()


@pytest.mark.parametrize('env', [
    {TTL: '1h', 'CACHE_CREATION': json.dumps({'ephemeral_5m_input_tokens': 0, 'ephemeral_1h_input_tokens': 8})},
    {POLICY_TTL: '5m', 'CACHE_CREATION': json.dumps({'ephemeral_5m_input_tokens': 8, 'ephemeral_1h_input_tokens': 0})},
    {'CACHE_CREATION': json.dumps({'ephemeral_5m_input_tokens': 0, 'ephemeral_1h_input_tokens': 8})},
    {TTL: '1h'},
])
def test_no_warning_without_an_opposite_tier_write(tmp_path, caplog, env):
    native = client(tmp_path, **env)
    try:
        with caplog.at_level(logging.INFO, logger=directsdk.logger.name):
            native.create(**request())
        assert records(caplog, 'WARNING') == []
        assert len(records(caplog, 'INFO')) == 1
    finally:
        native.close()


def test_concurrent_mismatch_warning_is_rate_limited(tmp_path, caplog):
    native = client(tmp_path, **{TTL: '1h', 'CACHE_CREATION': json.dumps({'ephemeral_5m_input_tokens': 11})})
    try:
        with caplog.at_level(logging.INFO, logger=directsdk.logger.name), ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(native.create, **request()) for _ in range(4)]
            for future in futures:
                assert future.result(timeout=30).usage.model_dump()['native_cache_tiers']['write_5m'] == 11
        assert len(records(caplog, 'INFO')) == 4
        assert records(caplog, 'WARNING') == [MISMATCH % ('1h', 11, 'unknown')]
    finally:
        native.close()


def test_failed_native_call_records_unknown_and_keeps_error(tmp_path, caplog):
    native = client(tmp_path, **{TTL: '1h', 'NATIVE_ERROR': 'unknown:Public fixture failure'})
    try:
        with caplog.at_level(logging.INFO, logger=directsdk.logger.name), \
                pytest.raises(directsdk.ClaudeAPIError, match='Public fixture failure'):
            native.create(**request())
        assert records(caplog, 'INFO') == [INFO % ('1h', 'unknown', 'unknown')]
    finally:
        native.close()


@pytest.mark.parametrize('failure', ['info', 'mismatch', 'tiers'])
@pytest.mark.parametrize('native_failure', [False, True])
@pytest.mark.parametrize('streaming', [False, True])
def test_telemetry_failure_keeps_result_error_and_cleanup(tmp_path, monkeypatch, failure, native_failure, streaming):
    attempts = []
    prefix = {'info': 'DirectSDK cache tiers ', 'mismatch': 'DirectSDK cache tier mismatch', 'tiers': None}[failure]

    class BrokenHandler(logging.Handler):
        def emit(self, record):
            if prefix and record.getMessage().startswith(prefix):
                attempts.append(record.levelname)
                raise OSError('Public fixture logging sink failure')

    env, fake = {TTL: '1h'}, TIERED
    expected_error = (directsdk.ClaudeAPIError, 'Native API error: Public fixture native failure')
    if failure == 'mismatch':
        env['CACHE_CREATION'] = json.dumps({'ephemeral_5m_input_tokens': 3, 'ephemeral_1h_input_tokens': 8})
        if native_failure:
            # Fail after complete native usage exists, so the mismatch warning is attempted on the error path too.
            fake = TIERED.replace("'text':b['text']", "'text':'Public fixture different text'")
            expected_error = (RuntimeError, 'Native final text differs from incremental stream')
    elif native_failure:
        env['NATIVE_ERROR'] = 'unknown:Public fixture native failure'
    if failure == 'tiers':
        def broken_tiers(*_):
            attempts.append('tiers')
            raise ValueError('Public fixture telemetry failure')
        monkeypatch.setattr(directsdk, 'native_cache_tiers', broken_tiers)
    native = client(tmp_path, fake, **env)

    gates, processes = [], []
    original_admission, original_spawn = directsdk.Admission, directsdk.Request.spawn

    def gate(*args, **kwargs):
        gates.append(original_admission(*args, **kwargs))
        return gates[-1]

    def spawn(request, *args, **kwargs):
        processes.append(original_spawn(request, *args, **kwargs))
        return processes[-1]

    monkeypatch.setattr(directsdk, 'Admission', gate)
    monkeypatch.setattr(directsdk.Request, 'spawn', spawn)
    handler, level = BrokenHandler(), directsdk.logger.level
    directsdk.logger.addHandler(handler)
    directsdk.logger.setLevel(logging.INFO)
    try:
        result = error = None
        try:
            result = native.create(**request(), stream=streaming)
            if streaming:
                result = list(result)[-1]._response
        except Exception as exc:
            error = exc
        # Lifecycle first, before the explicit cleanup below can hide a skipped teardown.
        assert len(gates) == 1 and len(processes) == 1
        assert all(not g.thread.is_alive() and g.server.fileno() == -1 for g in gates)
        assert all(p.poll() is not None and p.stdin.closed and p.stdout.closed for p in processes)
        assert not native._requests
        assert attempts and set(attempts) == {'tiers' if failure == 'tiers' else 'INFO' if failure == 'info' else 'WARNING'}
        if native_failure:
            assert type(error) is expected_error[0] and str(error) == expected_error[1]
        else:
            assert error is None, error
            assert result.choices[0].message.content == 'hello\n'
            assert result.choices[0].message.tool_calls[0].function.name == 'probe'
            tiers = result.usage.model_dump().get('native_cache_tiers')
            if failure == 'tiers':
                assert tiers == {'requested_ttl': '1h', 'write_5m': None, 'write_1h': None}
            elif failure == 'mismatch':
                assert tiers == {'requested_ttl': '1h', 'write_5m': 3, 'write_1h': 8}
    finally:
        directsdk.logger.removeHandler(handler)
        directsdk.logger.setLevel(level)
        native.close()
        for g in gates:
            if g.thread.is_alive() or g.server.fileno() != -1:
                g.close()
        for p in processes:
            directsdk.kill_process_tree(p)
            p.wait(timeout=5)
            for pipe in (p.stdin, p.stdout):
                if pipe is not None and not pipe.closed:
                    pipe.close()
