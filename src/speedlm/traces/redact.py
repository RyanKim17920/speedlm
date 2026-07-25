"""Inline, structure-preserving redaction for captured traces."""

from __future__ import annotations

import ipaddress
import json
import math
import re
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

_PEM_RE = re.compile(
    r"-----BEGIN (?P<label>[A-Z0-9 ]*PRIVATE KEY)-----"
    r".*?"
    r"-----END (?P=label)-----",
    re.DOTALL,
)
_AUTHORIZATION_RE = re.compile(
    r"(?i)\b(?P<prefix>authorization\s*:\s*)"
    r"(?P<scheme>bearer\s+|basic\s+)?"
    r"(?P<value>[A-Za-z0-9+/_.=-]{8,})"
)
_BEARER_RE = re.compile(
    r"(?i)\b(?P<prefix>bearer\s+)(?P<value>[A-Za-z0-9+/_.=-]{8,})"
)
_ASSIGNMENT_RE = re.compile(
    r"""(?ix)
    (?P<prefix>
        \b(?P<name>
            api[_-]?key|x[_-]?api[_-]?key|password|passwd|pwd|
            access[_-]?token|refresh[_-]?token|id[_-]?token|token|
            client[_-]?secret|secret
        )\b
        ["']?\s*[:=]\s*
    )
    (?:
        (?P<quote>["'])(?P<quoted>[^"'\r\n]+)(?P=quote)
        |
        (?P<plain>[^\s,;}\]]+)
    )
    """
)
_HOME_PATH_RE = re.compile(
    r"(?<![\w.-])(?:/admin/home|/home|/Users)/[A-Za-z0-9._-]+"
)
_EMAIL_RE = re.compile(
    r"(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,63}(?![\w.-])",
    re.IGNORECASE,
)
_IPV4_CANDIDATE_RE = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
_IPV6_CANDIDATE_RE = re.compile(r"(?<![\w:])[0-9A-Fa-f:]{3,39}(?![\w:])")
_INTERNAL_HOST_RE = re.compile(
    r"(?<![@\w.-])"
    r"(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+"
    r"(?:internal|intranet|corp|lan|local|home)"
    r"(?![\w.-])",
    re.IGNORECASE,
)
_JWT_RE = re.compile(
    r"(?<![A-Za-z0-9_-])"
    r"[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{10,}"
    r"(?![A-Za-z0-9_-])"
)
_HEX_RE = re.compile(r"(?<![A-Fa-f0-9])[A-Fa-f0-9]{32,}(?![A-Fa-f0-9])")
_BASE64_RE = re.compile(
    r"(?<![A-Za-z0-9+/_-])[A-Za-z0-9+/_-]{32,}={0,2}(?![A-Za-z0-9+/_=-])"
)
_IMAGE_DATA_URI_RE = re.compile(
    r"data:image/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=\r\n]+",
    re.IGNORECASE,
)

_PROVIDER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "aws_key",
        re.compile(r"\b(?:AKIA|ASIA|AIDA|AROA|AIPA|ANPA|ANVA|ASCA)[A-Z0-9]{16}\b"),
    ),
    (
        "github_token",
        re.compile(r"\b(?:ghp|ghu|gho|ghs|gpr)_[A-Za-z0-9]{20,255}\b"),
    ),
    (
        "anthropic_key",
        re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"),
    ),
    (
        "openai_key",
        re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}\b"),
    ),
    (
        "google_key",
        re.compile(r"\b(?:AIza[0-9A-Za-z_-]{35}|ya29\.[0-9A-Za-z_-]{20,})\b"),
    ),
    (
        "slack_token",
        re.compile(r"\b(?:xox[baprs]-[A-Za-z0-9-]{10,}|xapp-[A-Za-z0-9-]{10,})\b"),
    ),
)

_SENSITIVE_FIELD_CATEGORIES = {
    "api_key": "api_key",
    "apikey": "api_key",
    "x_api_key": "api_key",
    "password": "password",
    "passwd": "password",
    "pwd": "password",
    "authorization": "authorization",
    "proxy_authorization": "authorization",
    "access_token": "token",
    "refresh_token": "token",
    "id_token": "token",
    "token": "token",
    "client_secret": "secret",
    "secret": "secret",
    "aws_secret_access_key": "aws_secret",
    "private_key": "private_key",
}


@dataclass(frozen=True, slots=True)
class RedactionPolicy:
    """Explicit controls for redaction behavior.

    ``entropy_threshold`` is Shannon entropy in bits per character. The default
    favors false-negative risk over damaging ordinary code and prose.
    """

    redact_emails: bool = True
    redact_home_paths: bool = True
    redact_ipv4: bool = False
    redact_ipv6: bool = False
    redact_internal_hostnames: bool = False
    entropy_enabled: bool = True
    entropy_threshold: float = 3.7
    entropy_min_length: int = 32

    def __post_init__(self) -> None:
        if isinstance(self.entropy_threshold, bool) or not isinstance(
            self.entropy_threshold, (int, float)
        ):
            raise TypeError("entropy_threshold must be a number")
        if not 0.0 <= self.entropy_threshold <= 8.0:
            raise ValueError("entropy_threshold must be between 0 and 8")
        if isinstance(self.entropy_min_length, bool) or not isinstance(
            self.entropy_min_length, int
        ):
            raise TypeError("entropy_min_length must be an integer")
        if self.entropy_min_length < 16:
            raise ValueError("entropy_min_length must be at least 16")


@dataclass(frozen=True, slots=True)
class RedactionReport(Mapping[str, int]):
    """Immutable category counts for one redaction operation."""

    counts: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "counts", MappingProxyType(dict(self.counts)))

    def __getitem__(self, category: str) -> int:
        return self.counts[category]

    def __iter__(self) -> Iterator[str]:
        return iter(self.counts)

    def __len__(self) -> int:
        return len(self.counts)

    @property
    def total(self) -> int:
        """Total number of replacements, including repeated values."""
        return sum(self.counts.values())


@dataclass(slots=True)
class _RedactionContext:
    counts: Counter[str]
    placeholders: dict[str, str]


class Redactor:
    """Redact secrets from arbitrary JSON-like trace structures."""

    def __init__(self, policy: RedactionPolicy | None = None) -> None:
        self.policy = policy or RedactionPolicy()

    def redact(self, value: Any) -> tuple[Any, RedactionReport]:
        """Return a redacted value and per-category replacement counts.

        Traversal is copy-on-write: when no sensitive content is found, the
        original object is returned. Placeholder state is scoped to this call,
        which corresponds to one trace.
        """
        context = _RedactionContext(counts=Counter(), placeholders={})
        redacted, _ = self._walk(value, context, field_name=None)
        return redacted, RedactionReport(context.counts)

    def redact_trace(self, trace: Any) -> tuple[Any, RedactionReport]:
        """Alias for :meth:`redact` that makes call sites self-documenting."""
        return self.redact(trace)

    def redact_text(self, text: str) -> tuple[str, RedactionReport]:
        """Redact one text value using the same policy and reporting API."""
        redacted, report = self.redact(text)
        return redacted, report

    def _walk(
        self,
        value: Any,
        context: _RedactionContext,
        *,
        field_name: str | None,
    ) -> tuple[Any, bool]:
        if isinstance(value, str):
            category = self._sensitive_field_category(field_name)
            if category is not None and value:
                return self._redact_sensitive_field(value, category, context), True
            if field_name is not None and field_name.casefold() == "arguments":
                json_result = self._redact_json_arguments(value, context)
                if json_result is not None:
                    return json_result
            redacted = self._redact_string(value, context)
            return redacted, redacted != value

        if isinstance(value, Mapping):
            changed = False
            result: dict[Any, Any] = {}
            for key, item in value.items():
                child, child_changed = self._walk(
                    item,
                    context,
                    field_name=key if isinstance(key, str) else None,
                )
                result[key] = child
                changed = changed or child_changed
            return (result, True) if changed else (value, False)

        if isinstance(value, list):
            changed = False
            result_list: list[Any] = []
            for item in value:
                child, child_changed = self._walk(item, context, field_name=None)
                result_list.append(child)
                changed = changed or child_changed
            return (result_list, True) if changed else (value, False)

        if isinstance(value, tuple):
            changed = False
            result_tuple: list[Any] = []
            for item in value:
                child, child_changed = self._walk(item, context, field_name=None)
                result_tuple.append(child)
                changed = changed or child_changed
            return (tuple(result_tuple), True) if changed else (value, False)

        return value, False

    def _redact_json_arguments(
        self,
        value: str,
        context: _RedactionContext,
    ) -> tuple[str, bool] | None:
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None
        redacted, changed = self._walk(parsed, context, field_name=None)
        if not changed:
            return value, False
        return (
            json.dumps(redacted, ensure_ascii=False, separators=(",", ":")),
            True,
        )

    def _redact_sensitive_field(
        self,
        value: str,
        fallback_category: str,
        context: _RedactionContext,
    ) -> str:
        if fallback_category == "authorization":
            match = re.fullmatch(r"(?i)(bearer\s+|basic\s+)?(.+)", value)
            if match is not None:
                scheme = match.group(1) or ""
                secret = match.group(2)
                category = self._classify_secret(secret, "authorization")
                return f"{scheme}{self._placeholder(secret, category, context)}"
        category = self._classify_secret(value, fallback_category)
        return self._placeholder(value, category, context)

    @staticmethod
    def _sensitive_field_category(field_name: str | None) -> str | None:
        if field_name is None:
            return None
        normalized = field_name.casefold().replace("-", "_")
        return _SENSITIVE_FIELD_CATEGORIES.get(normalized)

    def _redact_string(self, text: str, context: _RedactionContext) -> str:
        result = _PEM_RE.sub(
            lambda match: self._placeholder(match.group(0), "private_key", context),
            text,
        )
        result = _AUTHORIZATION_RE.sub(
            lambda match: self._replace_authorization(match, context),
            result,
        )
        result = _ASSIGNMENT_RE.sub(
            lambda match: self._replace_assignment(match, context),
            result,
        )
        result = _BEARER_RE.sub(
            lambda match: self._replace_bearer(match, context),
            result,
        )

        for category, pattern in _PROVIDER_PATTERNS:
            def _make_replacer(cat: str) -> Callable[[re.Match[str]], str]:
                def replacer(m: re.Match[str]) -> str:
                    return self._placeholder(m.group(0), cat, context)
                return replacer
            result = pattern.sub(
                _make_replacer(category),
                result,
            )

        result = _JWT_RE.sub(
            lambda match: self._placeholder(match.group(0), "jwt", context),
            result,
        )

        if self.policy.redact_home_paths:
            result = _HOME_PATH_RE.sub(
                lambda match: self._placeholder(match.group(0), "home_path", context),
                result,
            )
        if self.policy.redact_emails:
            result = _EMAIL_RE.sub(
                lambda match: self._placeholder(match.group(0), "email", context),
                result,
            )
        if self.policy.redact_ipv4:
            result = _IPV4_CANDIDATE_RE.sub(
                lambda match: self._replace_ip(match, context, version=4),
                result,
            )
        if self.policy.redact_ipv6:
            result = _IPV6_CANDIDATE_RE.sub(
                lambda match: self._replace_ip(match, context, version=6),
                result,
            )
        if self.policy.redact_internal_hostnames:
            result = _INTERNAL_HOST_RE.sub(
                lambda match: self._placeholder(
                    match.group(0), "internal_hostname", context
                ),
                result,
            )
        if self.policy.entropy_enabled:
            result = self._redact_entropy(result, context)
        return result

    def _replace_authorization(
        self,
        match: re.Match[str],
        context: _RedactionContext,
    ) -> str:
        secret = match.group("value")
        category = self._classify_secret(secret, "authorization")
        return (
            f"{match.group('prefix')}{match.group('scheme') or ''}"
            f"{self._placeholder(secret, category, context)}"
        )

    def _replace_bearer(
        self,
        match: re.Match[str],
        context: _RedactionContext,
    ) -> str:
        secret = match.group("value")
        category = self._classify_secret(secret, "bearer_token")
        return f"{match.group('prefix')}{self._placeholder(secret, category, context)}"

    def _replace_assignment(
        self,
        match: re.Match[str],
        context: _RedactionContext,
    ) -> str:
        quote = match.group("quote") or ""
        secret = match.group("quoted") or match.group("plain")
        fallback = self._sensitive_field_category(match.group("name")) or "secret"
        category = self._classify_secret(secret, fallback)
        placeholder = self._placeholder(secret, category, context)
        return f"{match.group('prefix')}{quote}{placeholder}{quote}"

    def _replace_ip(
        self,
        match: re.Match[str],
        context: _RedactionContext,
        *,
        version: int,
    ) -> str:
        candidate = match.group(0)
        try:
            address = ipaddress.ip_address(candidate)
        except ValueError:
            return candidate
        if address.version != version:
            return candidate
        return self._placeholder(candidate, f"ipv{version}", context)

    def _redact_entropy(self, text: str, context: _RedactionContext) -> str:
        protected_spans = tuple(match.span() for match in _IMAGE_DATA_URI_RE.finditer(text))

        def replace_hex(match: re.Match[str]) -> str:
            candidate = match.group(0)
            minimum = max(48, self.policy.entropy_min_length)
            if len(candidate) < minimum or len(candidate) == 40:
                return candidate
            if self._inside_spans(match.start(), match.end(), protected_spans):
                return candidate
            if self._looks_like_hash_label(text, match.start()):
                return candidate
            if self._shannon_entropy(candidate) < self.policy.entropy_threshold:
                return candidate
            return self._placeholder(candidate, "high_entropy_hex", context)

        result = _HEX_RE.sub(replace_hex, text)
        protected_spans = tuple(match.span() for match in _IMAGE_DATA_URI_RE.finditer(result))

        def replace_base64(match: re.Match[str]) -> str:
            candidate = match.group(0)
            if len(candidate.rstrip("=")) < self.policy.entropy_min_length:
                return candidate
            if self._inside_spans(match.start(), match.end(), protected_spans):
                return candidate
            if not (
                any(char.islower() for char in candidate)
                and any(char.isupper() for char in candidate)
                and any(char.isdigit() for char in candidate)
            ):
                return candidate
            if self._shannon_entropy(candidate.rstrip("=")) < self.policy.entropy_threshold:
                return candidate
            return self._placeholder(candidate, "high_entropy_base64", context)

        return _BASE64_RE.sub(replace_base64, result)

    @staticmethod
    def _inside_spans(start: int, end: int, spans: tuple[tuple[int, int], ...]) -> bool:
        return any(start >= span_start and end <= span_end for span_start, span_end in spans)

    @staticmethod
    def _looks_like_hash_label(text: str, start: int) -> bool:
        prefix = text[max(0, start - 20) : start].casefold()
        return bool(re.search(r"(?:commit|git|sha(?:1|256)?)[\s:=_-]*$", prefix))

    @staticmethod
    def _shannon_entropy(value: str) -> float:
        counts = Counter(value)
        length = len(value)
        return -sum(
            (count / length) * math.log2(count / length)
            for count in counts.values()
        )

    @staticmethod
    def _classify_secret(secret: str, fallback: str) -> str:
        for category, pattern in _PROVIDER_PATTERNS:
            if pattern.fullmatch(secret):
                return category
        if _JWT_RE.fullmatch(secret):
            return "jwt"
        if _PEM_RE.fullmatch(secret):
            return "private_key"
        return fallback

    @staticmethod
    def _placeholder(
        secret: str,
        category: str,
        context: _RedactionContext,
    ) -> str:
        existing = context.placeholders.get(secret)
        context.counts[category] += 1
        if existing is not None:
            return existing
        placeholder = f"<REDACTED:{category}>"
        context.placeholders[secret] = placeholder
        return placeholder
