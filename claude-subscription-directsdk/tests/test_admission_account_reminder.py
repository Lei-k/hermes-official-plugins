"""Native's account-email reminder appended after Hermes' frame never reaches Claude.

Claude Code adds an account-email ``<system-reminder>`` to the turn it answers. Prepended to the
frame it is dropped by ``_frame_first`` / ``stable_request_context`` (#195); appended after it
(a suffix of the frame's text, inside a tool_result, or a block of its own) it used to stay in
the request, where Claude read it as part of the user's message and the next request, which
replays the turn without it, missed the cache. Only the exact native template goes; anything
else is forwarded as native built it (only the message breakpoint may move, as for any turn).
"""
import copy
import json

import pytest

from admission import QueriedTurnMismatch, pin_message_breakpoint, restore_queried_turn, stable_request_context
from test_admission_first_turn import FRAME, MARKER, call1, wire

EMAIL = 'a@example.invalid'
_HEAD = ("<system-reminder>\nAs you answer the user's questions, you can use the following context:\n"
         "# userEmail\nThe user's email address is " + EMAIL + ". Use it only to identify the user, "
         "such as for authorship, attribution, or filtering their own work. Never send it to an "
         "unrelated service, such as in a request header, URL, or payload, unless the user explicitly "
         "asks.")
# Native 2.1.283: usage sentence plus the IMPORTANT paragraph.
REMINDER_283 = (_HEAD + "\n\nIMPORTANT: this context may or may not be relevant to your tasks. You should "
                "not respond to this context unless it is highly relevant to your task.\n</system-reminder>\n")
# Native 2.1.296: an attribution paragraph instead.
REMINDER_296 = (_HEAD + "\n\nClaude Code attached this context automatically; it isn't part of the user's "
                "message. It describes the user's own account and workspace, so they don't need it reported "
                "back.\n</system-reminder>\n")
FIRST = {'type': 'tool_result', 'tool_use_id': 't1', 'content': 'unaltered'}
RESULT = {'type': 'tool_result', 'tool_use_id': 't2', 'content': 'result'}
ASSISTANT = {'role': 'assistant', 'content': [{'type': 'text', 'text': 'tools'}]}


def plain(value):
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items() if k != 'cache_control'}
    if isinstance(value, list):
        return [plain(v) for v in value]
    return value


def content(payload):
    """Message content without cache markers: the breakpoint may still move, the text may not."""
    return plain(json.loads(payload)['messages'])


def relay(payload, queried):
    """The relay's transforms in ``Handler.do_POST`` order (restoration mismatches forward as is)."""
    try:
        payload = restore_queried_turn(payload, queried)
    except QueriedTurnMismatch:
        pass
    return pin_message_breakpoint(stable_request_context(payload, queried), queried)


def native_turn(placement, reminder, lead=(FIRST,)):
    host = RESULT if placement == 'tool_suffix' else FRAME
    queried = [*lead, host]
    if placement == 'standalone_after':
        sent = [*lead, {**host, 'cache_control': MARKER}, {'type': 'text', 'text': reminder}]
    else:
        field = 'content' if placement == 'tool_suffix' else 'text'
        sent = [*lead, {**host, field: host[field] + '\n\n' + reminder, 'cache_control': MARKER}]
    return queried, [ASSISTANT, {'role': 'user', 'content': sent}]


@pytest.mark.parametrize('reminder', [REMINDER_283, REMINDER_296], ids=['2.1.283', '2.1.296'])
@pytest.mark.parametrize('lead', [(FIRST,), ()], ids=['after_results', 'alone'])
@pytest.mark.parametrize('placement', ['text_suffix', 'tool_suffix', 'standalone_after'])
def test_account_reminder_after_host_frame_is_removed(placement, lead, reminder):
    queried, messages = native_turn(placement, reminder, lead)
    original = copy.deepcopy(queried)
    out = relay(wire(messages), queried).decode()
    assert EMAIL not in out
    body = json.loads(out)
    assert plain(body['messages'][1]['content']) == queried
    assert body['messages'][0] == ASSISTANT
    assert queried == original
    # The marker stays on the frame's last block, exactly once.
    assert body['messages'][1]['content'][-1]['cache_control'] == MARKER
    assert out.count('cache_control') == 1


def test_marker_on_a_dropped_standalone_reminder_moves_onto_the_frame():
    sent = [FIRST, FRAME, {'type': 'text', 'text': REMINDER_296, 'cache_control': MARKER}]
    out = json.loads(relay(wire([ASSISTANT, {'role': 'user', 'content': sent}]), [FIRST, FRAME]))
    assert out['messages'][1]['content'] == [FIRST, {**FRAME, 'cache_control': MARKER}]


def test_reminder_inside_structured_tool_result_content_is_removed():
    host = {'type': 'tool_result', 'tool_use_id': 't2', 'content': [{'type': 'text', 'text': 'listing'}]}
    for content in ([{'type': 'text', 'text': 'listing\n\n' + REMINDER_283}],
                    [{'type': 'text', 'text': 'listing'}, {'type': 'text', 'text': REMINDER_283}]):
        sent = [FIRST, {**host, 'content': content, 'cache_control': MARKER}]
        out = relay(wire([ASSISTANT, {'role': 'user', 'content': sent}]), [FIRST, host]).decode()
        assert EMAIL not in out
        assert plain(json.loads(out)['messages'][1]['content']) == [FIRST, host]


VARIANTS = [
    REMINDER_296.replace('reported back.', 'reported back. Extra.'),
    REMINDER_283.replace('# userEmail', '# userAccount'),
    REMINDER_296 + 'trailing native text',
    REMINDER_283.replace('\n</system-reminder>', "\n# currentDate\nToday's date is 2026-10-03.\n</system-reminder>"),
]
VARIANT_IDS = ['extra_sentence', 'other_heading', 'trailing_text', 'extra_section']


@pytest.mark.parametrize('variant', VARIANTS, ids=VARIANT_IDS)
@pytest.mark.parametrize('placement', ['text_suffix', 'tool_suffix', 'standalone_after'])
def test_unknown_reminder_variant_is_not_dropped_as_the_account_reminder(placement, variant):
    """Only the exact template is recognized. (Trailing plain text in a tool-result frame is
    ``restore_queried_turn``'s own, wording-independent rule, so this checks the steps after it.)"""
    queried, messages = native_turn(placement, variant)
    raw = wire(messages)
    assert content(pin_message_breakpoint(stable_request_context(raw, queried), queried)) == content(raw)


@pytest.mark.parametrize('variant', VARIANTS, ids=VARIANT_IDS)
@pytest.mark.parametrize('placement', ['text_suffix', 'standalone_after'])
def test_unknown_reminder_after_a_plain_frame_reaches_upstream_unchanged(placement, variant):
    queried, messages = native_turn(placement, variant, lead=())
    raw = wire(messages)
    assert content(relay(raw, queried)) == content(raw)


def test_user_text_containing_an_email_is_untouched():
    """Host text that quotes the template (or any email) is Hermes' own content, not native's."""
    host = {'type': 'text', 'text': 'Please reply to ' + EMAIL + ':\n\n' + REMINDER_296}
    queried = [FIRST, host]
    messages = [ASSISTANT, {'role': 'user', 'content': [FIRST, {**host, 'cache_control': MARKER}]}]
    out = json.loads(relay(wire(messages), queried))
    assert plain(out['messages'][1]['content']) == queried
    # A standalone reminder block that Hermes itself queried is not native's addition.
    queried = [FRAME, {'type': 'text', 'text': REMINDER_296}]
    messages = [ASSISTANT, {'role': 'user', 'content': [FRAME, {'type': 'text', 'text': REMINDER_296,
                                                                 'cache_control': MARKER}]}]
    raw = wire(messages)
    assert content(relay(raw, queried)) == content(raw)


def test_ambiguous_frame_is_forwarded_unchanged():
    sent = [FRAME, FRAME, {'type': 'text', 'text': REMINDER_296, 'cache_control': MARKER}]
    raw = wire([ASSISTANT, {'role': 'user', 'content': sent}])
    assert content(relay(raw, [FRAME])) == content(raw)


def test_changed_tool_identity_is_forwarded_unchanged():
    sent = [{**RESULT, 'tool_use_id': 'other', 'content': RESULT['content'] + '\n\n' + REMINDER_296,
             'cache_control': MARKER}]
    raw = wire([ASSISTANT, {'role': 'user', 'content': sent}])
    assert content(relay(raw, [RESULT])) == content(raw)


@pytest.mark.parametrize('reminder', [REMINDER_283, REMINDER_296], ids=['2.1.283', '2.1.296'])
def test_prepended_reminder_on_the_opening_turn_is_still_dropped(reminder):
    messages = call1()
    messages[0]['content'][0] = {'type': 'text', 'text': reminder}
    out = relay(wire(messages), [FRAME]).decode()
    assert EMAIL not in out
    assert plain(json.loads(out)['messages'][0]['content']) == [FRAME]


def test_transform_is_idempotent():
    queried, messages = native_turn('text_suffix', REMINDER_296)
    once = relay(wire(messages), queried)
    assert relay(once, queried) == once
