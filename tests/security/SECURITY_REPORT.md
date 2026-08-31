# SpeedLM Security Status

**Repository:** `speedlm-fr`

**Updated:** 2026-08-31

**Automated status:** 43 passed, 1 deliberate strict xfail

Run the focused suite with:

```bash
python -m pytest tests/security -q -ra
```

The security suite is regression coverage for specific, previously discovered
failure modes. It is not an independent penetration test, a complete data-loss
prevention system, or a guarantee that captured traffic contains no sensitive
data.

## Known limitation: secrets split across structured leaves

`test_secret_split_across_messages_must_not_reach_disk` remains a deliberate
strict xfail. `Redactor._walk` scans each string leaf independently, so a secret
split across messages, roles, or tool-call arguments may not match any pattern;
the individually harmless fragments can therefore survive trace persistence.

A correct general fix needs an explicit ordering and concatenation contract for
all string leaves, a way to map matches back to per-leaf spans, and a measured
false-positive budget. Joining only adjacent message text would make the
existing example pass without closing the broader evasion. Until that contract
exists, SpeedLM redaction should be treated as defense in depth, not as a DLP
boundary. Operators should regard raw captured exchanges as sensitive, restrict
access to the SpeedLM home and run directories, and apply appropriate retention
controls.

The expected failure is documented at
`tests/security/test_redaction_evasions.py:80` and is marked `strict=True`, so an
unexpected pass or a change in behavior is visible in CI.

## Remediated findings under regression coverage

The 2026-07-25 adversarial audit counted 20 test-backed findings. Current tests
pass for 19 of them; the remaining split-secret case is the documented expected
failure above.

| Surface | Passing regression coverage |
| --- | --- |
| Secret redaction | Base64-, hex-, and percent-encoded assignments; Unicode homoglyphs; zero-width obfuscation; sensitive mapping keys; tool-call arguments; whitespace-obfuscated PEM blocks |
| Path containment | Profile symlink escape and run-prefix directory escape |
| Local confidentiality | Owner-only SpeedLM directories, trace/lock files, and atomic metadata files |
| Proxy routing | Dot-segment normalization, trailing-slash variants, unrelated non-API paths, and installed vLLM development-route denylisting |
| Resource bounds | Trace token-budget enforcement, zero-usage accounting, finite streaming capture limits, and under-limit stream retention |
| Training provenance | Client-supplied and provider-generated assistant messages receive distinct provenance tags |
| Diagnostic safety | Training errors redact credentials from captured subprocess stderr |

The suite also retains passing checks for upstream destination control and other
already-hardened behavior alongside these audit regressions.

## Scope and interpretation

- The result above describes the current `tests/security/` suite only. Run it
  for the exact revision being released; do not copy the count forward after
  changing tests or security-sensitive code.
- Exact prompts, completions, tool arguments, traces, and training examples may
  be confidential even when pattern-based redaction succeeds. Do not publish
  raw run directories by default.
- Deployment controls outside this repository—including host access, network
  policy, model-server hardening, secret management, and GPU isolation—remain
  the operator's responsibility.
- Third-party model and vLLM behavior is outside the claim made by these tests.

## Historical note

This document supersedes the original 2026-07-25 point-in-time report, which
accurately described the failing suite at that revision but became stale after
the fixes landed. The known limitation is kept explicit instead of describing
the current tree as fully remediated.
