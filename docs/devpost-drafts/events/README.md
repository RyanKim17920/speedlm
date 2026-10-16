# Event packets

Create one file per researched Devpost event, named with the event's Devpost
slug, for example `example-hackathon.md`. Start from [`_template.md`](_template.md).

Research every plausible event, including events that become ineligible. A
complete rejection packet is useful: it prevents duplicate research and makes
the reason auditable.

## Allowed statuses

- `researching` — official requirements are incomplete or stale.
- `direct-active` — SpeedLM itself fits and the remaining work is submission
  packaging plus owner attestations.
- `direct-conditional` — SpeedLM itself fits, but an entrant-specific or
  conflicting-rules condition must be resolved first.
- `future-direct` — SpeedLM itself fits, but submissions are not open yet.
- `excluded-needs-different-project` — entering would require a separate app,
  sponsor integration, or themed deployment, so it is outside this campaign.
- `excluded-from-direct-campaign` — the project may be topically relevant, but
  an entrant or timing condition is unverified, so it is not an actionable
  SpeedLM submission.
- `owner-review` — copy and assets are ready; only owner attestations or the
  final form review remain.
- `submitted` — use only after recording a real Devpost confirmation URL and
  timestamp.
- `ineligible` — a verified rule conflicts with the entrant/project and cannot
  be fixed truthfully before the deadline.

Do not use `submitted` or a similar status based only on a prepared Markdown
file.

## Minimal event index row

```text
Event | slug | deadline UTC | status | packet path | last official check
```
