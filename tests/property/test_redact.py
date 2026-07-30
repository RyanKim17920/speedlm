from __future__ import annotations

import json
import string
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from speedlm.traces.redact import Redactor

SECRET_CHARS = string.ascii_letters + string.digits


def _openai_secret(suffix: str) -> tuple[str, str]:
    return "openai_key", f"sk-{suffix}"


def _anthropic_secret(suffix: str) -> tuple[str, str]:
    return "anthropic_key", f"sk-ant-{suffix}"


def _github_secret(suffix: str) -> tuple[str, str]:
    return "github_token", f"ghp_{suffix}"


SECRET_SUFFIX = st.text(alphabet=SECRET_CHARS, min_size=24, max_size=60)
SECRETS = st.one_of(
    st.builds(_openai_secret, SECRET_SUFFIX),
    st.builds(_anthropic_secret, SECRET_SUFFIX),
    st.builds(_github_secret, SECRET_SUFFIX),
)

JSON_SCALAR = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(max_size=50),
)
JSON_VALUE = st.recursive(
    JSON_SCALAR,
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.dictionaries(st.text(max_size=16), children, max_size=4),
    ),
    max_leaves=15,
)


@given(SECRETS, st.text(max_size=80))
@settings(max_examples=80, deadline=None)
def test_output_never_contains_original_provider_secret(
    classified_secret: tuple[str, str],
    surrounding_text: str,
) -> None:
    category, secret = classified_secret
    redacted, report = Redactor().redact_text(f"{surrounding_text} {secret}")

    assert secret not in redacted
    assert f"<REDACTED:{category}>" in redacted
    assert report[category] >= 1


@given(JSON_VALUE)
@settings(max_examples=80, deadline=None)
def test_redaction_is_idempotent(value: Any) -> None:
    redactor = Redactor()
    once, _ = redactor.redact(value)
    twice, second_report = redactor.redact(once)

    assert twice == once
    assert second_report.total == 0


@given(JSON_VALUE, SECRETS)
@settings(max_examples=70, deadline=None)
def test_redacted_json_text_stays_valid(
    value: Any,
    classified_secret: tuple[str, str],
) -> None:
    _, secret = classified_secret
    encoded = json.dumps({"value": value, "token": secret}, ensure_ascii=False)

    redacted, _ = Redactor().redact_text(encoded)

    decoded = json.loads(redacted)
    assert isinstance(decoded, dict)
    assert secret not in redacted


@given(JSON_VALUE, SECRETS)
@settings(max_examples=70, deadline=None)
def test_tool_argument_json_stays_valid_after_redaction(
    value: Any,
    classified_secret: tuple[str, str],
) -> None:
    _, secret = classified_secret
    arguments = json.dumps(
        {"value": value, "credentials": {"token": secret}},
        ensure_ascii=False,
    )
    trace = {"tool_calls": [{"function": {"arguments": arguments}}]}

    redacted, _ = Redactor().redact(trace)
    encoded = redacted["tool_calls"][0]["function"]["arguments"]
    decoded = json.loads(encoded)

    assert isinstance(decoded, dict)
    assert secret not in encoded
    assert decoded["value"] == value


@given(SECRETS)
@settings(max_examples=50, deadline=None)
def test_same_secret_maps_to_same_placeholder(
    classified_secret: tuple[str, str],
) -> None:
    category, secret = classified_secret
    redacted, report = Redactor().redact([secret, f"token={secret}"])

    assert isinstance(redacted, list)
    assert redacted == [
        f"<REDACTED:{category}>",
        f"token=<REDACTED:{category}>",
    ]
    assert report[category] == 2


@given(st.text(alphabet=" abcdefghijklmnopqrstuvwxyz.,!?0123456789", max_size=24))
@settings(max_examples=80, deadline=None)
def test_text_without_secrets_is_byte_identical(text: str) -> None:
    redacted, report = Redactor().redact_text(text)

    assert redacted.encode("utf-8") == text.encode("utf-8")
    assert report.total == 0


@given(st.text(max_size=500))
@settings(max_examples=100, deadline=None)
def test_arbitrary_unicode_never_raises(text: str) -> None:
    redacted, _ = Redactor().redact_text(text)
    assert isinstance(redacted, str)


def test_provider_secret_assignment_redaction_is_idempotent() -> None:
    secret = "ghp_abcdefghijklmnopqrstuvwxyz1234567890"
    redactor = Redactor()
    once, _ = redactor.redact_text(f"api_key={secret}")
    twice, _ = redactor.redact_text(once)
    assert twice == once
