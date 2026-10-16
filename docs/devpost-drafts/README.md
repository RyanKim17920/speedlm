# SpeedLM Devpost draft workspace

This directory is the offline source of truth for events that can accept
**SpeedLM itself**. Campaigns that require a separate application, mandatory
sponsor integration, or a different themed deployment are excluded from the
active list. It contains reusable project copy, evidence-backed facts, an asset
inventory, and a per-event worksheet.

Nothing here means that an event has accepted SpeedLM, that an online draft was
saved, or that the entrant is eligible. Devpost's final submit action includes
legal and rules attestations; the entrant must review and make those
attestations personally.

## Directory map

- [`shared-facts.md`](shared-facts.md) — claims that may be reused without
  event-specific rewriting, plus their evidence and limits.
- [`reusable-answers.md`](reusable-answers.md) — paste-ready copy at common
  character and word budgets.
- [`assets.md`](assets.md) — local videos, screenshots, hashes, and hosting
  requirements.
- [`owner-inputs.md`](owner-inputs.md) — identity, timeline, team, hosting, and
  legal facts that only the entrant can supply or attest.
- [`form-readiness.md`](form-readiness.md) — exact standard Devpost field map,
  per-event coverage, upload gaps, and the organizer-only questions still
  hidden behind authenticated drafts.
- [`events/_template.md`](events/_template.md) — copy this once per event and
  complete every gate before an online draft is created.
- [`events/README.md`](events/README.md) — naming, status, and indexing rules for
  the event packets.

The older [`../devpost-campaign.md`](../devpost-campaign.md) contains a dated
August 31, 2026 research pass and several event-specific drafts. Treat its
deadlines and eligibility conclusions as historical until each official event
page is checked again. Use this directory for the direct-submission campaign.

## Safe campaign workflow

1. Duplicate `events/_template.md` as `events/<devpost-slug>.md`.
2. Record the official overview, rules, eligibility, and submission-form URLs.
   Record when those pages were checked and preserve the exact event timezone.
3. Classify the entry as `researching`, `direct-active`,
   `direct-active-with-date-warning`, `direct-conditional`, `future-direct`,
   `owner-review`, `submitted`, `excluded-from-direct-campaign`, or
   `excluded-needs-different-project`.
4. Fill event-specific relevance and sponsor-technology answers. Do not paste a
   generic answer into a mandatory track that SpeedLM does not implement.
5. Choose copy from `reusable-answers.md`, then adapt it only with facts already
   supported by `shared-facts.md` or by newly retained evidence.
6. Match the event's duration and audio rules to `assets.md`. A local MP4 is not
   a judge-facing URL; upload it to an accepted host and test it logged out.
7. Resolve all `TODO(owner)` fields. These include identity, school, country,
   team, actual project timing, prior submissions, license choice, and any
   declaration of eligibility.
8. Save the online entry as a draft when possible. Record the draft URL and
   timestamp in the event packet, but leave final submission to the entrant.

## Evidence and wording rules

- Say the archived gate measured **15.0% more accepted tokens per verifier
  step** and **rejected promotion** because its wall-clock channel was
  non-stationary. A separate retained 125×8 gate measured **+13.4% accepted
  length and promoted**. Never merge those runs or say “universally 15%
  faster.”
- Keep the verifier authoritative: a speculative draft proposes tokens; the
  target model accepts or rejects them.
- A real recorded request produced the same reply in the normal-vLLM and
  SpeedLM demonstrations. Do not generalize that single check into a guarantee
  of byte-identical output across independently batched GPU runs.
- The five-second-ish idle setting shown in the demo is a recording setting.
  The software default described in the README is five minutes.
- The code is a research preview live-tested on one pinned H100 stack. Do not
  claim universal GPU, model, or runtime support.
- Never imply that a repository URL is public, a license exists, or a video is
  hosted until a logged-out check proves it.
- Existing-project rules are event-specific. If the rules are silent, disclose
  the earlier repository history; if they require new work, list and date the
  genuinely new work; if unchanged work is forbidden, mark the event
  `ineligible`.

## Active direct submissions

These events accept SpeedLM as the submitted project. They do not require a
separate app or sponsor integration.

| Event | Deadline (UTC) | Status | Video | Remaining owner checks |
| --- | --- | --- | --- | --- |
| [VoltHacks](events/volthacks.md) | **2026-09-05 21:00** | `direct-active` | verified 119s | confirm repository timing; jurisdiction; team; hosted asset |
| [Hyperbloom September AI/ML](events/hyperbloom-september-ai-ml.md) | **2026-09-14 21:00** | `direct-active` | none required | jurisdiction; enrollment; team; final attestation |
| [GatewayGS Hackathon 2](events/gatewaygs-hackathon-2.md) | **2026-09-16 21:00** | `direct-active-with-date-warning` | verified 59s | use live overview deadline; disclose July 25 start; final attestation |
| [ML Empowerment 3.0](events/ml-empowerment-build-challenge-3-0.md) | **2026-10-01 06:45** (use earlier conflicting date) | `direct-active` | none required | jurisdiction; enrollment; team; final attestation |

## Conditional direct submissions

SpeedLM itself fits these events, but a rules or entrant condition must be
resolved before submission. They do not call for inventing a different project.

| Event | Deadline (UTC) | Status | Blocking condition |
| --- | --- | --- | --- |
| [GIBC V2 Track 03](events/global-innovation-build-challenge-v2.md) | **2026-09-21 15:45** | `direct-conditional` | no substantially identical prior hackathon entry; complete July 11–September 21 timing attestation |

## Future direct submission

| Event | Opens / deadline (UTC) | Status | Requirement |
| --- | --- | --- | --- |
| [Arbiter Hacks V1](events/arbiter-hacks-v1.md) | opens **2026-10-20**; closes **2026-12-21 05:00** | `future-direct` | public repository and hosted 2–3-minute demo |

## Expired/archive

| Event | Closed (UTC) | Status | Note |
| --- | --- | --- | --- |
| [HackSocial 2026](events/hacksocial-2026.md) | **2026-09-01 03:45** | `expired-unconfirmed` | The form is closed. If an entry was finalized externally before the deadline, record it here and resolve the GIBC prior-entry rule. |

## Excluded from this campaign

| Event | Status | Reason |
| --- | --- | --- |
| [BunnieX](events/buuniex-hackathon.md) | `excluded-from-direct-campaign` | Its “created for this hackathon” attestation has not been established; do not submit SpeedLM while that is unknown. |
| [Zero Origin](events/zero-origin.md) | `excluded-from-direct-campaign` | It requires an eligible-country check and an exact 2–4-person team; this campaign has neither verified. |
| [Code for Humanity](events/code-for-humanity.md) | `excluded-needs-different-project` | SpeedLM alone does not demonstrate the required humanitarian deployment or beneficiary impact. |

### Cross-submission stop

GIBC V2 rejects work substantially the same as a previous hackathon entry.
Preparing several offline drafts does not trigger that rule. If SpeedLM was
finalized for HackSocial or another event, unchanged SpeedLM may be ineligible
for GIBC; the entrant must disclose the prior entry and obtain organizer
clarification or create and document a materially distinct version that the
rules permit.
