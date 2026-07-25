# SpeedLM Security Audit -- Adversary-Focused Findings

**Repository**: speedlm-fr
**Date**: 2026-07-25
**Scope**: 6 threat surfaces, 26 tests, 20 CONFIRMED vulnerabilities, 6 regressions passing

All findings are CONFIRMED -- each is backed by a strict xfail test in `tests/security/` that demonstrates the issue. The 6 passing tests are regression checks for already-hardened surfaces.

---

## Severity Ranking

### CRITICAL -- Secret Exposure (9 findings)

#### 1. Base64/Hex/Percent-encoded secrets bypass redaction
**Tests**: `test_base64_encoded_secret_must_not_reach_disk`, `test_hex_encoded_secret_must_not_reach_disk`, `test_percent_encoded_secret_must_not_reach_disk`
**PoC**: `base64.b64encode(b"api_key=hunter2")` stored as-is in traces.jsonl
**Impact**: Credentials survive in raw encoded form on disk and become training data.
**Fix direction**: Decode common encodings (base64, hex, percent) before applying redaction rules.

#### 2. Unicode homoglyph / zero-width character obfuscation
**Tests**: `test_unicode_homoglyph_secret_must_not_reach_disk`, `test_zero_width_obfuscated_secret_must_not_reach_disk`
**PoC**: Zero-width space splits the key name; full-width characters in assignment
**Impact**: Sensitive assignment names are not normalized before matching.
**Fix direction**: Unicode NFKC normalization and zero-width character stripping before redaction.

#### 3. Secret split across message boundaries
**Test**: `test_secret_split_across_messages_must_not_reach_disk`
**PoC**: Secret split across two user messages evades per-message redaction
**Impact**: Redaction is message-level, not conversation-level.
**Fix direction**: Maintain a rolling window of recent message content for redaction matching.

#### 4. Secrets in JSON field names (not values)
**Test**: `test_secret_embedded_in_field_name_must_not_reach_disk`
**PoC**: Secret used as a dict key rather than value
**Impact**: Redaction scans string values but copies mapping keys verbatim.
**Fix direction**: Apply redaction to both keys and values of all nested mappings.

#### 5. Obfuscated tool_call argument secrets
**Test**: `test_obfuscated_tool_argument_secret_must_not_reach_disk`
**PoC**: Zero-width obfuscation inside tool argument JSON
**Impact**: Tool argument strings are not normalized/redacted.
**Fix direction**: Normalize and redact tool argument strings with the same rules as message content.

#### 6. Whitespace-obfuscated PEM blocks
**Test**: `test_whitespace_obfuscated_pem_must_not_reach_disk`
**PoC**: Tab instead of space in PEM header
**Impact**: PEM pattern matching requires exact whitespace and dash placement.
**Fix direction**: Use regex-based PEM detection that tolerates whitespace variations.

---

### HIGH -- Path Traversal and Local Confidentiality (5 findings)

#### 7. Profile loader follows symlinks outside profile directory
**Test**: `test_profile_loader_must_reject_symlink_escape`
**PoC**: Symlink in profiles/ pointing to external file
**Impact**: Arbitrary files can be loaded as profiles.
**Fix direction**: Resolve symlinks and reject files not physically inside the profiles directory.

#### 8. Run prefix can escape runs directory via ```
**Test**: `test_run_prefix_must_not_escape_runs_directory`
**PoC**: `new_run_dir(layout, prefix="../escaped", ...)`
**Impact**: Attacker-controlled prefix creates directories outside intended runs folder.
**Fix direction**: Validate that prefix is a basename or resolve and check containment.

#### 9. ~/.speedlm directories created with process umask (not 0700)
**Test**: `test_speedlm_directories_must_be_owner_only`
**Impact**: Other local users can list and read trace files containing secrets.
**Fix direction**: Force 0o700 in ensure_layout() via explicit chmod.

#### 10. Trace files created with mode 0644 (not 0600)
**Test**: `test_trace_files_must_be_owner_only`
**Impact**: Any local user can read raw trace data including API keys, prompts, and responses.
**Fix direction**: Open trace files with mode=0o600 or chmod after creation.

#### 11. Atomic metadata files created with mode 0644
**Test**: `test_atomic_metadata_files_must_be_owner_only`
**Impact**: Metadata files may contain sensitive configuration.
**Fix direction**: Use os.open(path, flags, 0o600) in the atomic write path.

---

### HIGH -- Proxy Destination Bypass (1 finding)

#### 12. Dot-segment path traversal bypasses allowlist
**Test**: `test_dot_segment_cannot_reach_blocked_admin_endpoint`
**PoC**: `/v1/harmless/../sleep` -- allowlist check runs on raw path before httpx normalizes ..
**Impact**: Client can reach vLLM admin endpoints by embedding ../ in the path.
**Fix direction**: Normalize the path (resolve .. segments) before checking against blocked paths.

---

### HIGH -- Resource Exhaustion (3 findings)

#### 13. TraceStore.append does not enforce configured token budget
**Test**: `test_append_must_enforce_configured_trace_token_budget`
**PoC**: Append traces until total tokens far exceed max_tokens -- no pruning occurs.
**Impact**: Disk exhaustion via unbounded trace storage.
**Fix direction**: Call store.prune() at end of append() when token budget is exceeded.

#### 14. Zero reported usage defeats prune accounting
**Test**: `test_zero_reported_usage_must_not_bypass_prune_budget`
**PoC**: 40KB message with usage: {prompt_tokens: 0, completion_tokens: 0} bypasses token-based pruning.
**Impact**: Massive content accumulates without triggering the token budget.
**Fix direction**: When usage is zero but content is large, estimate token count from actual content length.

#### 15. SSE capture has no byte limit (unlike non-streaming)
**Test**: `test_sse_capture_must_obey_capture_body_limit`
**PoC**: Feed 1KB+ of SSE data with _MAX_CAPTURE_BODY_BYTES=256 -- captured without truncation.
**Impact**: Unbounded memory consumption during streaming responses, potential OOM.
**Fix direction**: Apply the same byte ceiling to SSE _ResponseObserver as non-streaming capture.

---

### MEDIUM -- Training Data Integrity (1 finding)

#### 16. Client-supplied assistant turns are supervised in training
**Test**: `test_client_supplied_assistant_turn_must_not_be_supervised`
**PoC**: Client sends assistant role messages that are marked as supervised training targets.
**Impact**: Any client can inject arbitrary content that the draft model will learn to generate.
**Fix direction**: Only supervise tokens from the provider response (last assistant turn).

---

### MEDIUM -- Log and Error Disclosure (1 finding)

#### 17. TrainingError exposes subprocess stderr verbatim
**Test**: `test_training_error_must_not_expose_secret_stderr`
**PoC**: stderr field in TrainingError contains raw credential values.
**Impact**: Secret values from subprocess output appear in logs.
**Fix direction**: Truncate or redact the stderr field in TrainingError.__str__.

---

## What Looks Solid (6 Passing Tests)

1. Trailing slash on blocked paths -- /v1/sleep/, /v1/sleep//// correctly rejected
2. Non-v1 path rejection -- /metrics, //attacker.example/v1/chat/completions blocked
3. Loss mask correctly weights large messages -- 8000-token message dominates loss as intended
4. Non-streaming body cap -- non-streaming responses capped at 32 MiB
5. Upstream host/port from config -- not from request headers
6. Redaction failures drop traces -- dropped rather than persisted raw

---

## Summary Table

| # | Severity | Area | Test File | Status |
|---|----------|------|-----------|--------|
| 1-3 | CRITICAL | Encoded secret bypass | test_redaction_evasions.py | CONFIRMED |
| 4-5 | CRITICAL | Obfuscation bypass | test_redaction_evasions.py | CONFIRMED |
| 6-7 | CRITICAL | Split/field-name secrets | test_redaction_evasions.py | CONFIRMED |
| 8 | CRITICAL | PEM obfuscation | test_redaction_evasions.py | CONFIRMED |
| 9-10 | HIGH | Path traversal | test_path_and_permissions.py | CONFIRMED |
| 11-13 | HIGH | File permissions | test_path_and_permissions.py | CONFIRMED |
| 14 | HIGH | Proxy dot-segment bypass | test_proxy_destination.py | CONFIRMED |
| 15-17 | HIGH | Resource exhaustion | test_resource_bounds.py | CONFIRMED |
| 18 | MEDIUM | Training data provenance | test_training_integrity.py | CONFIRMED |
| 19 | MEDIUM | Error message disclosure | test_log_safety.py | CONFIRMED |
