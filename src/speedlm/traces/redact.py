"""Inline, structure-preserving redaction for captured traces."""

from __future__ import annotations

import base64
import binascii
import ipaddress
import json
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any
from urllib.parse import unquote_to_bytes

_PEM_RE = re.compile(
    r"-{5}\s*BEGIN\s+(?:[A-Z0-9]+\s+)*PRIVATE\s+KEY\s*-{5}"
    r".*?"
    r"-{5}\s*END\s+(?:[A-Z0-9]+\s+)*PRIVATE\s+KEY\s*-{5}",
    re.DOTALL | re.IGNORECASE,
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
_PLACEHOLDER_RE = re.compile(r"<REDACTED:[a-z0-9_]+>")
_PERCENT_CANDIDATE_RE = re.compile(
    r"(?<![A-Za-z0-9._~%-])(?:[A-Za-z0-9._~-]|%[0-9A-Fa-f]{2})*"
    r"%[0-9A-Fa-f]{2}(?:[A-Za-z0-9._~-]|%[0-9A-Fa-f]{2})*"
    r"(?![A-Za-z0-9._~%-])"
)
_HEX_DECODE_CANDIDATE_RE = re.compile(
    r"(?<![A-Fa-f0-9])[A-Fa-f0-9]{16,}(?![A-Fa-f0-9])"
)
_BASE64_DECODE_CANDIDATE_RE = re.compile(
    r"(?<![A-Za-z0-9+/_-])[A-Za-z0-9+/_-]{16,}={0,2}"
    r"(?![A-Za-z0-9+/_=-])"
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


@dataclass(frozen=True, slots=True)
class _TextView:
    text: str
    source_spans: tuple[tuple[int, int], ...]

    def source_span(self, start: int, end: int) -> tuple[int, int]:
        if start >= end:
            raise ValueError("text-view spans must be non-empty")
        return self.source_spans[start][0], self.source_spans[end - 1][1]


@dataclass(frozen=True, slots=True)
class _Match:
    start: int
    end: int
    secret: str
    category: str
    priority: int


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
                redacted_key: Any = key
                key_changed = False
                if isinstance(key, str):
                    redacted_key = self._redact_string(key, context)
                    key_changed = redacted_key != key
                child, child_changed = self._walk(
                    item,
                    context,
                    field_name=key if isinstance(key, str) else None,
                )
                result[redacted_key] = child
                changed = changed or key_changed or child_changed
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
        if _PLACEHOLDER_RE.fullmatch(value):
            return value
        if fallback_category == "authorization":
            match = re.fullmatch(r"(?i)(bearer\s+|basic\s+)?(.+)", value)
            if match is not None:
                scheme = match.group(1) or ""
                secret = match.group(2)
                if _PLACEHOLDER_RE.fullmatch(secret):
                    return value
                category = self._classify_secret(secret, "authorization")
                return f"{scheme}{self._placeholder(secret, category, context)}"
        category = self._classify_secret(value, fallback_category)
        return self._placeholder(value, category, context)

    @staticmethod
    def _sensitive_field_category(field_name: str | None) -> str | None:
        if field_name is None:
            return None
        normalized = "".join(
            char
            for char in unicodedata.normalize("NFKC", field_name)
            if unicodedata.category(char) != "Cf"
        )
        normalized = normalized.casefold().replace("-", "_")
        return _SENSITIVE_FIELD_CATEGORIES.get(normalized)

    def _redact_string(self, text: str, context: _RedactionContext) -> str:
        protected_spans = tuple(match.span() for match in _PLACEHOLDER_RE.finditer(text))
        view = self._normalized_view(text)
        matches = self._collect_direct_matches(text, view, protected_spans)
        matches.extend(self._collect_decoded_matches(text, protected_spans))
        if self.policy.entropy_enabled:
            matches.extend(self._collect_entropy_matches(text, protected_spans))

        selected = self._non_overlapping(matches)
        if not selected:
            return text

        parts: list[str] = []
        cursor = 0
        for match in selected:
            parts.append(text[cursor : match.start])
            parts.append(self._placeholder(match.secret, match.category, context))
            cursor = match.end
        parts.append(text[cursor:])
        return "".join(parts)

    @staticmethod
    def _normalized_view(text: str) -> _TextView:
        characters: list[str] = []
        spans: list[tuple[int, int]] = []
        for index, source_char in enumerate(text):
            normalized = unicodedata.normalize("NFKC", source_char)
            for char in normalized:
                if unicodedata.category(char) == "Cf":
                    continue
                characters.append(char)
                spans.append((index, index + 1))
        return _TextView("".join(characters), tuple(spans))

    def _collect_direct_matches(
        self,
        source: str,
        view: _TextView,
        protected_spans: tuple[tuple[int, int], ...],
    ) -> list[_Match]:
        matches: list[_Match] = []

        def add(
            match: re.Match[str],
            category: str,
            *,
            group: str | int = 0,
            priority: int,
        ) -> None:
            view_start, view_end = match.span(group)
            source_start, source_end = view.source_span(view_start, view_end)
            if self._overlaps_spans(source_start, source_end, protected_spans):
                return
            secret = source[source_start:source_end]
            matches.append(
                _Match(source_start, source_end, secret, category, priority)
            )

        for match in _PEM_RE.finditer(view.text):
            add(match, "private_key", priority=0)

        for match in _AUTHORIZATION_RE.finditer(view.text):
            secret = match.group("value")
            add(
                match,
                self._classify_secret(secret, "authorization"),
                group="value",
                priority=10,
            )
        for match in _ASSIGNMENT_RE.finditer(view.text):
            group = "quoted" if match.group("quoted") is not None else "plain"
            secret = match.group(group)
            fallback = self._sensitive_field_category(match.group("name")) or "secret"
            add(
                match,
                self._classify_secret(secret, fallback),
                group=group,
                priority=10,
            )
        for match in _BEARER_RE.finditer(view.text):
            secret = match.group("value")
            add(
                match,
                self._classify_secret(secret, "bearer_token"),
                group="value",
                priority=10,
            )
        for category, pattern in _PROVIDER_PATTERNS:
            for match in pattern.finditer(view.text):
                add(match, category, priority=20)
        for match in _JWT_RE.finditer(view.text):
            add(match, "jwt", priority=20)

        if self.policy.redact_home_paths:
            for match in _HOME_PATH_RE.finditer(view.text):
                add(match, "home_path", priority=30)
        if self.policy.redact_emails:
            for match in _EMAIL_RE.finditer(view.text):
                add(match, "email", priority=30)
        if self.policy.redact_ipv4:
            self._collect_ip_matches(
                view,
                source,
                _IPV4_CANDIDATE_RE,
                version=4,
                protected_spans=protected_spans,
                matches=matches,
            )
        if self.policy.redact_ipv6:
            self._collect_ip_matches(
                view,
                source,
                _IPV6_CANDIDATE_RE,
                version=6,
                protected_spans=protected_spans,
                matches=matches,
            )
        if self.policy.redact_internal_hostnames:
            for match in _INTERNAL_HOST_RE.finditer(view.text):
                add(match, "internal_hostname", priority=30)
        return matches

    @staticmethod
    def _collect_ip_matches(
        view: _TextView,
        source: str,
        pattern: re.Pattern[str],
        *,
        version: int,
        protected_spans: tuple[tuple[int, int], ...],
        matches: list[_Match],
    ) -> None:
        for match in pattern.finditer(view.text):
            candidate = match.group(0)
            try:
                address = ipaddress.ip_address(candidate)
            except ValueError:
                continue
            if address.version != version:
                continue
            start, end = view.source_span(*match.span())
            if Redactor._overlaps_spans(start, end, protected_spans):
                continue
            matches.append(
                _Match(start, end, source[start:end], f"ipv{version}", 30)
            )

    def _collect_decoded_matches(
        self,
        text: str,
        protected_spans: tuple[tuple[int, int], ...],
    ) -> list[_Match]:
        image_spans = tuple(match.span() for match in _IMAGE_DATA_URI_RE.finditer(text))
        matches: list[_Match] = []
        patterns = (
            ("percent", _PERCENT_CANDIDATE_RE),
            ("hex", _HEX_DECODE_CANDIDATE_RE),
            ("base64", _BASE64_DECODE_CANDIDATE_RE),
        )
        for encoding, pattern in patterns:
            for candidate_match in pattern.finditer(text):
                start, end = candidate_match.span()
                if self._overlaps_spans(start, end, protected_spans):
                    continue
                if self._inside_spans(start, end, image_spans):
                    continue
                candidate = candidate_match.group(0)
                decoded = self._safe_decode(candidate, encoding)
                if decoded is None:
                    continue
                category = self._decoded_secret_category(decoded, depth=1)
                if category is not None:
                    matches.append(
                        _Match(start, end, candidate, category, 5)
                    )
        return matches

    def _decoded_secret_category(self, text: str, *, depth: int) -> str | None:
        view = self._normalized_view(text)
        direct = self._collect_direct_matches(text, view, ())
        if direct:
            return min(direct, key=lambda match: match.priority).category
        if depth <= 0:
            return None
        for encoding, pattern in (
            ("percent", _PERCENT_CANDIDATE_RE),
            ("hex", _HEX_DECODE_CANDIDATE_RE),
            ("base64", _BASE64_DECODE_CANDIDATE_RE),
        ):
            for candidate_match in pattern.finditer(text):
                decoded = self._safe_decode(candidate_match.group(0), encoding)
                if decoded is None:
                    continue
                category = self._decoded_secret_category(decoded, depth=depth - 1)
                if category is not None:
                    return category
        return None

    @staticmethod
    def _safe_decode(candidate: str, encoding: str) -> str | None:
        try:
            if encoding == "percent":
                decoded_bytes = unquote_to_bytes(candidate)
                if decoded_bytes == candidate.encode("utf-8"):
                    return None
            elif encoding == "hex":
                if len(candidate) % 2:
                    return None
                decoded_bytes = bytes.fromhex(candidate)
            else:
                padding = "=" * (-len(candidate) % 4)
                decoded_bytes = base64.b64decode(
                    candidate + padding,
                    altchars=b"-_",
                    validate=True,
                )
            decoded = decoded_bytes.decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            return None
        if not decoded or any(
            unicodedata.category(char) == "Cc" and char not in "\t\r\n"
            for char in decoded
        ):
            return None
        return decoded

    def _collect_entropy_matches(
        self,
        text: str,
        protected_spans: tuple[tuple[int, int], ...],
    ) -> list[_Match]:
        image_spans = tuple(match.span() for match in _IMAGE_DATA_URI_RE.finditer(text))
        matches: list[_Match] = []

        for match in _HEX_RE.finditer(text):
            candidate = match.group(0)
            minimum = max(48, self.policy.entropy_min_length)
            if len(candidate) < minimum or len(candidate) == 40:
                continue
            if self._inside_spans(match.start(), match.end(), image_spans):
                continue
            if self._overlaps_spans(match.start(), match.end(), protected_spans):
                continue
            if self._looks_like_hash_label(text, match.start()):
                continue
            if self._shannon_entropy(candidate) < self.policy.entropy_threshold:
                continue
            matches.append(
                _Match(
                    match.start(),
                    match.end(),
                    candidate,
                    "high_entropy_hex",
                    40,
                )
            )

        for match in _BASE64_RE.finditer(text):
            candidate = match.group(0)
            unpadded = candidate.rstrip("=")
            if len(unpadded) < self.policy.entropy_min_length:
                continue
            if self._inside_spans(match.start(), match.end(), image_spans):
                continue
            if self._overlaps_spans(match.start(), match.end(), protected_spans):
                continue
            if self._looks_like_base64_image(candidate):
                continue
            if not (
                any(char.islower() for char in candidate)
                and any(char.isupper() for char in candidate)
                and any(char.isdigit() for char in candidate)
            ):
                continue
            if self._shannon_entropy(unpadded) < self.policy.entropy_threshold:
                continue
            matches.append(
                _Match(
                    match.start(),
                    match.end(),
                    candidate,
                    "high_entropy_base64",
                    40,
                )
            )
        return matches

    @staticmethod
    def _looks_like_base64_image(candidate: str) -> bool:
        try:
            padding = "=" * (-len(candidate) % 4)
            decoded = base64.b64decode(
                candidate + padding,
                altchars=b"-_",
                validate=True,
            )
        except (binascii.Error, ValueError):
            return False
        signatures = (
            b"\x89PNG\r\n\x1a\n",
            b"\xff\xd8\xff",
            b"GIF87a",
            b"GIF89a",
            b"BM",
            b"II*\x00",
            b"MM\x00*",
            b"\x00\x00\x01\x00",
        )
        return decoded.startswith(signatures) or (
            decoded.startswith(b"RIFF")
            and len(decoded) >= 12
            and decoded[8:12] == b"WEBP"
        )

    @staticmethod
    def _non_overlapping(matches: list[_Match]) -> tuple[_Match, ...]:
        selected: list[_Match] = []
        cursor = 0
        for match in sorted(
            matches,
            key=lambda item: (item.start, item.priority, -(item.end - item.start)),
        ):
            if match.start < cursor:
                continue
            selected.append(match)
            cursor = match.end
        return tuple(selected)

    @staticmethod
    def _inside_spans(start: int, end: int, spans: tuple[tuple[int, int], ...]) -> bool:
        return any(start >= span_start and end <= span_end for span_start, span_end in spans)

    @staticmethod
    def _overlaps_spans(
        start: int,
        end: int,
        spans: tuple[tuple[int, int], ...],
    ) -> bool:
        return any(start < span_end and end > span_start for span_start, span_end in spans)

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
