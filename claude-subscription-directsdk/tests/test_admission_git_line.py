"""Native's git line for the plugin's own workdir must not move the cached prefix.

Native puts an environment message right after the first user turn of every request. Its
``Is a git repository`` line is a fresh walk from the cwd up to ``/`` for any ``.git``, so a
``.git`` appearing above the shared scratch workdir (another tool running ``git init`` in the
same temp tree) flipped it for just those requests, and each flip re-wrote the whole
conversation to the prompt cache. The relay pins that one line, for that one directory, to a
value computed once per workdir; everything else is forwarded byte-identical.
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import admission
from admission import pin_message_breakpoint, stable_request_context
from test_admission_first_turn import MARKER, cached_prefix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import directsdk  # noqa: E402

WORKDIR = '/synthetic/tmp/claude-directsdk-cwd-1000'


@pytest.fixture(autouse=True)
def fresh_answers(monkeypatch):
    """Each test starts as a new process: no workdir's git answer computed yet."""
    monkeypatch.setattr(admission, '_workdir_git', {}, raising=False)


def env_block(git, workdir=WORKDIR, worktree=False):
    lines = ['# Environment', 'You have been invoked in the following environment: ',
             f' - Primary working directory: {workdir}']
    if worktree:
        lines.append(' - This is a git worktree \u2014 an isolated copy of the repository.')
    lines += [f' - Is a git repository: {git}', ' - Platform: linux', ' - Shell: bash',
              ' - OS Version: Linux 7.0.0-38-generic', '',
              'You are powered by the model named Opus 5.5 (1M context).']
    return '\n'.join(lines)


FIRST = {'type': 'text', 'text': 'Run the probe.'}
RESULT = {'type': 'tool_result', 'tool_use_id': 't1', 'content': 'probe output ' * 50}


def request(git, rounds, **env):
    """Native's tool-loop layout: first turn, environment message, rounds, date line last."""
    messages = [{'role': 'user', 'content': [FIRST]}, {'role': 'system', 'content': env_block(git, **env)}]
    for n in range(rounds):
        messages.append({'role': 'assistant', 'content': [
            {'type': 'tool_use', 'id': f't{n + 1}', 'name': 'mcp__hermes__probe', 'input': {}}]})
        messages.append({'role': 'user', 'content': [{**RESULT, 'tool_use_id': f't{n + 1}'}]})
    messages.append({'role': 'system', 'content': [{'type': 'text', 'text': "Today's date is 2026-11-14.",
                                                    'cache_control': MARKER}]})
    queried = messages[-2]['content'] if rounds else [FIRST]
    return json.dumps({'model': 'm', 'messages': messages}).encode(), queried


def relay(payload, queried, workdir=WORKDIR):
    """The relay's order: stable context, then the git line, then the breakpoint."""
    payload = stable_request_context(payload, queried)
    payload = admission.stable_workdir_git(payload, workdir)
    return pin_message_breakpoint(payload, queried)


def forwarded(git, rounds, workdir=WORKDIR, **env):
    return relay(*request(git, rounds, **env), workdir=workdir)


def rows(payload):
    out = []
    for message in json.loads(payload)['messages']:
        content = message['content']
        for block in ([{'type': 'text', 'text': content}] if isinstance(content, str) else content):
            out.append((message['role'], json.dumps({k: v for k, v in block.items() if k != 'cache_control'},
                                                    sort_keys=True)))
    return out


def test_ancestor_git_flip_does_not_change_forwarded_prefix():
    """false, true (an ancestor .git appeared), false: every request replays the prior cached span."""
    calls = [forwarded('false', 1), forwarded('true', 2), forwarded('false', 3)]
    for before, after in zip(calls, calls[1:]):
        prefix = cached_prefix(before)
        assert prefix is not None
        assert rows(after)[:len(prefix)] == prefix
    env = [json.loads(c)['messages'][1]['content'] for c in calls]
    assert env[0] == env[1] == env[2]
    assert 'Is a git repository: false\n' in env[0]


def test_only_the_git_line_changes():
    payload, queried = request('true', 1)
    expected = relay(*request('false', 1), workdir=None)
    assert relay(payload, queried) == expected


def test_list_form_is_pinned_and_keeps_its_directive():
    payload, queried = request('true', 1)
    body = json.loads(payload)
    body['messages'][1]['content'] = [{'type': 'text', 'text': env_block('true'), 'cache_control': MARKER}]
    out = json.loads(admission.stable_workdir_git(json.dumps(body).encode(), WORKDIR))
    assert out['messages'][1]['content'] == [{'type': 'text', 'text': env_block('false'), 'cache_control': MARKER}]


def test_answer_is_computed_once_per_workdir(tmp_path):
    """The workdir's own answer, fixed for the process: later filesystem changes cannot flip it."""
    plain, repo = tmp_path / 'plain', tmp_path / 'repo'
    plain.mkdir()
    (repo / '.git').mkdir(parents=True)

    def git_line(native_answer, workdir):
        out = admission.stable_workdir_git(request(native_answer, 1, workdir=str(workdir))[0], str(workdir))
        return json.loads(out)['messages'][1]['content'].split('\n')[3]

    assert git_line('true', plain) == ' - Is a git repository: false'
    (plain / '.git').mkdir()
    assert git_line('true', plain) == ' - Is a git repository: false'
    # A workdir that is itself a repository keeps true, and a false for it is pinned too.
    assert git_line('false', repo) == ' - Is a git repository: true'
    (repo / '.git').rmdir()
    assert git_line('false', repo) == ' - Is a git repository: true'


def _quoted_in_user_turn():
    """Host content never arrives as a system-role message: a user turn quoting the line keeps it."""
    payload, _ = request('false', 1)
    body = json.loads(payload)
    body['messages'][0]['content'].append({'type': 'text', 'text': env_block('true')})
    return json.dumps(body).encode()


def _no_git_line():
    payload, _ = request('true', 1)
    body = json.loads(payload)
    body['messages'][1]['content'] = env_block('true').replace(' - Is a git repository: true\n', '')
    return json.dumps(body).encode()


@pytest.mark.parametrize('case', ['no_git_line', 'already_pinned', 'other_workdir', 'path_prefix', 'worktree',
                                  'unknown_value', 'no_workdir', 'user_role_quote', 'unparseable'])
def test_everything_else_is_forwarded_byte_identical(case):
    workdir = WORKDIR
    if case == 'no_git_line':
        payload = _no_git_line()
    elif case == 'already_pinned':
        payload = request('false', 1)[0]
    elif case == 'other_workdir':
        payload = request('true', 1, workdir='/home/someone/project')[0]
    elif case == 'path_prefix':
        payload = request('true', 1, workdir=WORKDIR + '-other')[0]
    elif case == 'worktree':
        payload = request('true', 1, worktree=True)[0]
    elif case == 'unknown_value':
        payload = request('yes', 1)[0]
    elif case == 'no_workdir':
        payload, workdir = request('true', 1)[0], None
    elif case == 'user_role_quote':
        payload = _quoted_in_user_turn()
    else:
        payload = b'{"messages": [' + env_block('true').encode()
    assert admission.stable_workdir_git(payload, workdir) is payload


# The production path: the client hands its canonical cwd to the relay that forwards native's body.
NATIVE = r'''
import json, os, sys, urllib.request
for line in sys.stdin:
    frame = json.loads(line)
    if frame.get('shouldQuery') is False:
        print(json.dumps({'type': 'result', 'num_turns': 0}), flush=True)
        continue
    break
# Native's own answer: a walk from getcwd() up to / for any .git.
cwd, git = os.getcwd(), 'false'
probe = cwd
while True:
    if os.path.exists(os.path.join(probe, '.git')):
        git = 'true'
        break
    parent = os.path.dirname(probe)
    if parent == probe:
        break
    probe = parent
env = ('# Environment\nYou have been invoked in the following environment: \n - Primary working directory: '
       + cwd + '\n - Is a git repository: ' + git + '\n - Platform: linux')
body = {'model': 'm', 'messages': [
    {'role': 'user', 'content': [{'type': 'text', 'text': 'fixture'}]},
    {'role': 'system', 'content': env},
    {'role': 'system', 'content': [{'type': 'text', 'text': 'date', 'cache_control': {'type': 'ephemeral'}}]}]}
urllib.request.urlopen(urllib.request.Request(os.environ['ANTHROPIC_BASE_URL'] + '/v1/messages',
                       data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'}), timeout=5).read()
print(json.dumps({'type': 'assistant', 'message': {'id': 'm', 'role': 'assistant', 'content': [{'type': 'text', 'text': 'ok'}]}}))
print(json.dumps({'type': 'stream_event', 'event': {'type': 'message_stop'}}))
print(json.dumps({'type': 'result', 'subtype': 'success', 'usage': {'input_tokens': 0, 'output_tokens': 0}}))
'''


def _serve(bodies):
    usage = {'input_tokens': 0, 'output_tokens': 0}

    class Peer(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            bodies.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            events = [
                {'type': 'message_start', 'message': {'id': 'm', 'role': 'assistant', 'model': 'm', 'content': [], 'usage': usage}},
                {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}},
                {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'ok'}},
                {'type': 'content_block_stop', 'index': 0},
                {'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': usage},
                {'type': 'message_stop'},
            ]
            self.wfile.write(''.join('data: ' + json.dumps(e) + '\n\n' for e in events).encode())

    peer = ThreadingHTTPServer(('127.0.0.1', 0), Peer)
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    return peer


@pytest.mark.skipif(not hasattr(os, 'getuid'), reason='the shared workspace is keyed by the POSIX uid')
@pytest.mark.parametrize('symlinked', [False, True], ids=['real_tmpdir', 'symlinked_tmpdir'])
def test_client_relays_its_canonical_workdir_and_the_flip_never_reaches_upstream(tmp_path, monkeypatch, symlinked):
    root = tmp_path / 'tmp'
    root.mkdir(mode=0o700)
    tempdir = root
    if symlinked:
        # A TMPDIR reached through a symlink: native reports the canonical getcwd().
        tempdir = tmp_path / 'tmp-link'
        tempdir.symlink_to(root, target_is_directory=True)
    monkeypatch.setattr(directsdk.tempfile, 'tempdir', str(tempdir))
    bodies = []
    peer = _serve(bodies)
    script = tmp_path / 'native.py'
    script.write_text(NATIVE)
    client = directsdk.Client(command=[sys.executable, str(script)],
                              env={'PATH': os.defpath, 'HOME': str(tmp_path),
                                   'ANTHROPIC_BASE_URL': f'http://127.0.0.1:{peer.server_port}'})
    try:
        client.create(model='sonnet', messages=[{'role': 'user', 'content': 'fixture'}])
        (root / '.git').mkdir()  # The unrelated ancestor .git that flips native's own answer.
        client.create(model='sonnet', messages=[{'role': 'user', 'content': 'fixture'}])
        (root / '.git').rmdir()
        client.create(model='sonnet', messages=[{'role': 'user', 'content': 'fixture'}])
    finally:
        client.close()
        peer.shutdown()
        peer.server_close()
    workdir = os.path.realpath(root / f'claude-directsdk-cwd-{os.getuid()}')
    envs = [body['messages'][1]['content'] for body in bodies]
    assert len(envs) == 3
    assert envs[0] == envs[1] == envs[2]
    assert f'Primary working directory: {workdir}\n - Is a git repository: false\n' in envs[0]
