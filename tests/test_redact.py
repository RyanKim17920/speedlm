from __future__ import annotations

import ast
import base64
import json
import time
from typing import Any
from urllib.parse import quote

import pytest

from speedlm.traces.redact import RedactionPolicy, Redactor


@pytest.mark.parametrize(
    ("category", "secret"),
    [
        ("aws_key", "AKIAIOSFODNN7EXAMPLE"),
        ("github_token", "ghp_abcdefghijklmnopqrstuvwxyz1234567890"),
        ("github_token", "ghu_abcdefghijklmnopqrstuvwxyz1234567890"),
        ("github_token", "gho_abcdefghijklmnopqrstuvwxyz1234567890"),
        ("github_token", "ghs_abcdefghijklmnopqrstuvwxyz1234567890"),
        ("github_token", "gpr_abcdefghijklmnopqrstuvwxyz1234567890"),
        ("openai_key", "sk-abcdefghijklmnopqrstuvwxyz1234567890"),
        ("anthropic_key", "sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890"),
        ("google_key", "AIzaSyA1b2C3d4E5f6G7h8I9j0K1l2M3n4O5PQ7"),
        ("google_key", "ya29.a0AfH6SMBabcdefghijklmnopqrstuvwxyz123456"),
        ("slack_token", "xoxb-123456789012-abcdefghijklmnopqrstuv"),
        ("jwt", "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signature123456789"),
    ],
)
def test_provider_secret_classes(category: str, secret: str) -> None:
    redacted, report = Redactor().redact_text(f"credential: {secret}")

    assert secret not in redacted
    assert f"<REDACTED:{category}>" in redacted
    assert report[category] == 1
    assert report.total == 1


@pytest.mark.parametrize(
    ("source", "category", "preserved"),
    [
        ("Authorization: opaqueBearerValue12345", "authorization", "Authorization: "),
        ("Authorization: Bearer opaqueBearerValue12345", "authorization", "Bearer "),
        ("Bearer opaqueBearerValue12345", "bearer_token", "Bearer "),
        ("api_key=not-a-provider-secret", "api_key", "api_key="),
        ("password: 'correct horse battery staple'", "password", "password: '"),
        ('"token": "opaque-token-value"', "token", '"token": "'),
        ("client_secret=hunter2", "secret", "client_secret="),
    ],
)
def test_assignments_and_headers(source: str, category: str, preserved: str) -> None:
    redacted, report = Redactor().redact_text(source)

    assert preserved in redacted
    assert f"<REDACTED:{category}>" in redacted
    assert report[category] == 1


def test_private_key_pem_block() -> None:
    private_key = (
        "-----BEGIN PRIVATE KEY-----\n"
        "MIIEvQIBADANBgkqhkiG9w0BAQEFAASC\n"
        "very-secret-material\n"
        "-----END PRIVATE KEY-----"
    )

    redacted, report = Redactor().redact_text(f"before\n{private_key}\nafter")

    assert redacted == "before\n<REDACTED:private_key>\nafter"
    assert report.counts == {"private_key": 1}


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/home/alice/project/src/main.py", "<REDACTED:home_path>/project/src/main.py"),
        ("/Users/bob/Desktop/note.txt", "<REDACTED:home_path>/Desktop/note.txt"),
        (
            "/admin/home/service/.config/speedlm/config.json",
            "<REDACTED:home_path>/.config/speedlm/config.json",
        ),
    ],
)
def test_home_path_preserves_remainder(path: str, expected: str) -> None:
    redacted, report = Redactor().redact_text(path)

    assert redacted == expected
    assert report["home_path"] == 1


def test_email_redaction_can_be_disabled() -> None:
    text = "Contact alice.smith+training@example.co.uk"
    redacted, report = Redactor().redact_text(text)
    unchanged, disabled_report = Redactor(
        RedactionPolicy(redact_emails=False)
    ).redact_text(text)

    assert redacted == "Contact <REDACTED:email>"
    assert report["email"] == 1
    assert unchanged == text
    assert disabled_report.total == 0


def test_network_redaction_is_policy_controlled() -> None:
    text = "connect 10.2.3.4 or fd00::1234 via model-cache.service.internal"
    default, default_report = Redactor().redact_text(text)
    private, private_report = Redactor(
        RedactionPolicy(
            redact_ipv4=True,
            redact_ipv6=True,
            redact_internal_hostnames=True,
        )
    ).redact_text(text)

    assert default == text
    assert default_report.total == 0
    assert private == (
        "connect <REDACTED:ipv4> or <REDACTED:ipv6> "
        "via <REDACTED:internal_hostname>"
    )
    assert private_report.counts == {
        "ipv4": 1,
        "ipv6": 1,
        "internal_hostname": 1,
    }


def test_generic_high_entropy_hex_and_base64() -> None:
    hex_secret = "0123456789abcdefABCDEF9876543210fedcba98765432100123456789ABCDEF"
    base64_secret = "Q7vN2mK9pL4xR8sT1uW5yZ0aB3cD6eF9gH2jK5mN8pQ1rS4tV7wX0yZ3"

    redacted, report = Redactor().redact_text(f"{hex_secret} {base64_secret}")

    assert redacted == "<REDACTED:high_entropy_hex> <REDACTED:high_entropy_base64>"
    assert report.counts == {
        "high_entropy_hex": 1,
        "high_entropy_base64": 1,
    }


def test_entropy_threshold_is_tunable() -> None:
    candidate = "Q7vN2mK9pL4xR8sT1uW5yZ0aB3cD6eF9gH2jK5mN8pQ1rS4tV7wX0yZ3"

    redacted, report = Redactor(
        RedactionPolicy(entropy_threshold=8.0)
    ).redact_text(candidate)

    assert redacted == candidate
    assert report.total == 0


def test_tool_call_json_arguments_remain_valid() -> None:
    secret = "ghp_abcdefghijklmnopqrstuvwxyz1234567890"
    arguments = json.dumps(
        {
            "repo": "speedlm",
            "credentials": {"token": secret},
            "recipients": ["dev@example.com"],
        },
        indent=2,
    )
    trace = {
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "publish", "arguments": arguments},
                    }
                ],
            }
        ]
    }

    redacted, report = Redactor().redact(trace)
    encoded = redacted["messages"][0]["tool_calls"][0]["function"]["arguments"]
    decoded = json.loads(encoded)

    assert decoded == {
        "repo": "speedlm",
        "credentials": {"token": "<REDACTED:github_token>"},
        "recipients": ["<REDACTED:email>"],
    }
    assert report.counts == {"github_token": 1, "email": 1}


def test_unchanged_tool_arguments_keep_original_serialization() -> None:
    arguments = '{ "query": "ordinary code", "limit": 3 }'
    trace = {"tool_calls": [{"function": {"arguments": arguments}}]}

    redacted, report = Redactor().redact(trace)

    assert redacted is trace
    assert redacted["tool_calls"][0]["function"]["arguments"] == arguments
    assert report.total == 0


def test_malformed_tool_arguments_still_get_text_redaction() -> None:
    secret = "sk-abcdefghijklmnopqrstuvwxyz1234567890"
    trace = {"tool_calls": [{"function": {"arguments": f'{{"key": "{secret}"'}}]}

    redacted, report = Redactor().redact(trace)

    arguments = redacted["tool_calls"][0]["function"]["arguments"]
    assert secret not in arguments
    assert "<REDACTED:openai_key>" in arguments
    assert report["openai_key"] == 1


def test_nested_structures_and_all_string_fields_are_walked() -> None:
    trace: dict[str, Any] = {
        "messages": (
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "mail me at person@example.com"},
                    {"type": "metadata", "value": "/home/alice/work/tree"},
                ],
            },
        ),
        "metadata": {"deep": [{"password": "low entropy passphrase"}]},
    }

    redacted, report = Redactor().redact(trace)

    message = redacted["messages"][0]
    assert message["content"][0]["text"] == "mail me at <REDACTED:email>"
    assert message["content"][1]["value"] == "<REDACTED:home_path>/work/tree"
    assert redacted["metadata"]["deep"][0]["password"] == "<REDACTED:password>"
    assert report.counts == {"email": 1, "home_path": 1, "password": 1}


def test_same_secret_gets_same_placeholder_and_is_counted_twice() -> None:
    secret = "ghp_abcdefghijklmnopqrstuvwxyz1234567890"
    trace = {
        "messages": [
            {"role": "user", "content": secret},
            {"role": "assistant", "content": f"token={secret}"},
        ]
    }

    redacted, report = Redactor().redact(trace)

    first = redacted["messages"][0]["content"]
    second = redacted["messages"][1]["content"].removeprefix("token=")
    assert first == second == "<REDACTED:github_token>"
    assert report.counts == {"github_token": 2}


def test_existing_placeholder_is_immune_to_all_detection_passes() -> None:
    placeholder = "<REDACTED:github_token>"

    redacted, report = Redactor().redact_text(placeholder)

    assert redacted == placeholder
    assert report.total == 0


@pytest.mark.parametrize(
    "encoded",
    [
        base64.b64encode(base64.b64encode(b"api_key=hunter2")).decode("ascii"),
        quote(quote("api_key=hunter2", safe=""), safe=""),
    ],
)
def test_decoded_secret_scan_is_depth_limited_to_two_levels(encoded: str) -> None:
    redacted, report = Redactor().redact_text(encoded)

    assert redacted == "<REDACTED:api_key>"
    assert report.counts == {"api_key": 1}


def test_ordinary_code_and_prose_are_not_over_redacted() -> None:
    git_sha = "9f2c7b40e18d9474b847eb01ff6c3e80a3d218fc"
    uuid = "123e4567-e89b-12d3-a456-426614174000"
    image_blob = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    )
    long_english = (
        "This is a deliberately long English sentence about model serving, "
        "ordinary source code, tests, trace quality, and careful engineering."
    )
    text = "\n".join(
        [
            f"commit {git_sha}",
            f"id = UUID('{uuid}')",
            image_blob,
            long_english,
            "def calculate_total(items: list[int]) -> int: return sum(items)",
        ]
    )

    redacted, report = Redactor().redact_text(text)

    assert redacted == text
    assert report.total == 0


def test_bare_base64_image_blob_is_not_redacted() -> None:
    image_blob = base64.b64encode(
        b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 2
    ).decode("ascii")

    redacted, report = Redactor().redact_text(image_blob)

    assert redacted == image_blob
    assert report.total == 0


def test_record_without_sensitive_data_passes_through_byte_identical() -> None:
    record = {
        "id": "trace-1",
        "messages": [
            {"role": "user", "content": "Explain merge sort."},
            {"role": "assistant", "content": "Split, sort each half, then merge."},
        ],
        "temperature": 0.2,
    }
    before = json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode()

    redacted, report = Redactor().redact(record)
    after = json.dumps(redacted, ensure_ascii=False, separators=(",", ":")).encode()

    assert redacted is record
    assert after == before
    assert report.counts == {}
    assert report.total == 0


def test_large_message_performance_is_sane() -> None:
    secret = "sk-abcdefghijklmnopqrstuvwxyz1234567890"
    message = ("ordinary model output with code and prose; " * 15_000) + secret
    started = time.perf_counter()

    redacted, report = Redactor().redact({"content": message})
    elapsed = time.perf_counter() - started

    assert secret not in redacted["content"]
    assert report["openai_key"] == 1
    assert elapsed < 2.0


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"entropy_threshold": -0.1}, ValueError),
        ({"entropy_threshold": True}, TypeError),
        ({"entropy_min_length": 8}, ValueError),
        ({"entropy_min_length": 32.5}, TypeError),
    ],
)
def test_invalid_policy_values(kwargs: dict[str, Any], error: type[Exception]) -> None:
    with pytest.raises(error):
        RedactionPolicy(**kwargs)


# ── Corpus-preserving redaction ─────────────────────────────────────────────
#
# Every test below pairs the two directions the security argument needs: the
# secret must still be gone, AND the surrounding syntax must survive. A test
# that only asserted one direction would pass for a redactor that either leaks
# everything or destroys everything.


@pytest.mark.parametrize(
    ("text", "closer"),
    [
        ("call(secret=hunter2secretvalue)", ")"),
        ("login(user, password=hunter2xyzvalue)", ")"),
        ("fn(api_key=ABCDEF123456);", ")"),
        ("lookup[token=ABCDEF123456]", "]"),
        ("{token=ABCDEF123456}", "}"),
    ],
)
def test_assignment_value_stops_at_a_closing_delimiter(
    text: str, closer: str
) -> None:
    """Both directions: the credential goes, the caller's delimiter stays."""
    redacted, report = Redactor().redact_text(text)

    assert report.total == 1
    # Direction 1: the secret is still redacted.
    assert text.split("=", 1)[1].rstrip(")];") not in redacted
    # Direction 2: the closing delimiter survived.
    assert redacted.count(closer) == text.count(closer)
    assert redacted.endswith(text[text.index(closer) :])


def test_nested_call_keeps_every_closing_delimiter_balanced() -> None:
    """Nested closers compound: the old pattern ate the whole ``))`` tail."""
    text = "authorize(build(api_key=SECRETVALUE123))"

    redacted, report = Redactor().redact_text(text)

    assert "SECRETVALUE123" not in redacted
    assert report.total == 1
    assert redacted == "authorize(build(api_key=<REDACTED:api_key>))"
    for opener, closer in (("(", ")"), ("[", "]"), ("{", "}")):
        assert redacted.count(opener) == redacted.count(closer)


def test_annotated_parameter_default_is_redacted_and_still_parses() -> None:
    """The annotation is a type, not a value; the default literal is the secret."""
    source = 'def handler(api_key: str = "SECRETVALUE123") -> None:\n    return None\n'

    redacted, report = Redactor().redact_text(source)

    assert "SECRETVALUE123" not in redacted
    assert report.total == 1
    # The type annotation must survive: replacing it is what broke parsing.
    assert "api_key: str = " in redacted
    ast.parse(redacted)


def test_typescript_annotated_parameter_keeps_its_signature() -> None:
    source = 'function auth(token: string = "SECRETVALUE123") { return token; }'

    redacted, report = Redactor().redact_text(source)

    assert "SECRETVALUE123" not in redacted
    assert report.total == 1
    assert "token: string = " in redacted
    assert redacted.endswith(') { return token; }')


@pytest.mark.parametrize(
    "source",
    [
        "if token == '':\n    pass\n",
        "while password == other:\n    pass\n",
        "assert secret == expected\n",
    ],
)
def test_equality_comparison_is_not_mistaken_for_an_assignment(source: str) -> None:
    """``token == x`` offers the regex a bare ``=`` as its value; reject it."""
    redacted, report = Redactor().redact_text(source)

    assert redacted == source
    assert report.total == 0
    ast.parse(redacted)


def test_html_closing_tag_after_a_password_label_survives() -> None:
    markup = (
        '<div class="password-wrapper">\n'
        '  <label for="pwd">Password:</label>\n'
        '  <input type="password" id="pwd">\n'
        "</div>"
    )

    redacted, report = Redactor().redact_text(markup)

    assert redacted == markup
    assert report.total == 0


@pytest.mark.parametrize(
    "source",
    [
        "import java.util.Scanner\n",
        "import java.io.FileReader\n",
        "from speedlm.traces.redact import Redactor\n",
        "org.apache.commons.lang3.StringUtils\n",
    ],
)
def test_dotted_import_paths_are_not_treated_as_jwts(source: str) -> None:
    redacted, report = Redactor().redact_text(source)

    assert redacted == source
    assert report.total == 0


@pytest.mark.parametrize(
    "jwt",
    [
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0"
        ".dBjftJeZ4CVPmB92K27uhbUJU1p1r_wW1gFWFOEjXk",
        "eyJhbGciOiJub25lIn0.eyJhIjoxfQ.aaaaaaaaaaaaaaa",
    ],
)
def test_real_jwts_are_still_redacted(jwt: str) -> None:
    """The ``eyJ`` anchor is the JSON header, so no real JWT is excluded."""
    redacted, report = Redactor().redact_text(f"Authorization: Bearer {jwt}")

    assert jwt not in redacted
    assert report["jwt"] == 1


@pytest.mark.parametrize(
    ("text", "address", "expected"),
    [
        (
            "So the contact email is kelsei.etchison@mga.edu. The address is",
            "kelsei.etchison@mga.edu",
            "So the contact email is <REDACTED:email>. The address is",
        ),
        (
            "Reach him at kenthomas@envisorco.com.",
            "kenthomas@envisorco.com",
            "Reach him at <REDACTED:email>.",
        ),
        (
            "Write to bob.smith@example.co.uk. Then call.",
            "bob.smith@example.co.uk",
            "Write to <REDACTED:email>. Then call.",
        ),
        (
            "Write to bob.smith@example.co.uk, then call.",
            "bob.smith@example.co.uk",
            "Write to <REDACTED:email>, then call.",
        ),
        (
            "Write to bob.smith@example.co.uk and wait",
            "bob.smith@example.co.uk",
            "Write to <REDACTED:email> and wait",
        ),
    ],
)
def test_sentence_final_email_addresses_are_redacted(
    text: str, address: str, expected: str
) -> None:
    """A trailing period must not make an address unmatchable."""
    redacted, report = Redactor().redact_text(text)

    # Direction 1: the address is gone, whole -- no ``.uk`` tail left behind.
    assert address not in redacted
    assert ".uk" not in redacted
    assert report["email"] == 1
    # Direction 2: every surrounding character, punctuation included, survives.
    assert redacted == expected


def test_email_is_not_truncated_before_its_final_label() -> None:
    redacted, _ = Redactor().redact_text("mail ops@a.co.uk now")

    assert redacted == "mail <REDACTED:email> now"


@pytest.mark.parametrize(
    "secret",
    ["P@ssw0rdValue", "!@#$%^&*()_+", "hunter2", "ABCDEF1234567890"],
)
def test_unquoted_credentials_are_still_redacted(secret: str) -> None:
    """Guardrail against the closing-delimiter fix narrowing coverage."""
    redacted, report = Redactor().redact_text(f"password={secret}")

    assert secret not in redacted
    assert report["password"] == 1
