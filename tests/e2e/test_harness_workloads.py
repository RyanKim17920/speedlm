"""Tests for the declarative workload system.

CI-SAFE BY CONSTRUCTION
-----------------------
Nothing here carries the ``e2e`` marker, needs a GPU, or needs the multi-gigabyte
corpora under ``/data``.  The logic tests build a small synthetic workload in
``tmp_path`` *through the real builder*, so they cover the same code path that
produces the shipped manifests.  The handful of tests that genuinely need the
real corpora are guarded by ``skipif`` on file existence, and every one of them
has a pure-fixture twin above it so CI still covers the logic.

EVERY TEST HERE HAS BEEN DEMONSTRATED TO FAIL
---------------------------------------------
The mutation used for each is recorded in the docstring of the test.  A test
whose failure mode was not observed is not evidence; this repository's recurring
defect is a green assertion that cannot go red.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_workload as builder  # noqa: E402

from tests.e2e.harness import workloads as W  # noqa: E402

SHIPPED_SPECS = W.spec_directory()


# ---------------------------------------------------------------------------
# Synthetic fixture: a real workload, small enough to live in tmp_path
# ---------------------------------------------------------------------------


def _tokenizer() -> builder.Tokenizer:
    """A tokenizer that needs no transformers install, so CI can build corpora."""
    return builder.Tokenizer("fixture", chars_per_token=4.0)


def _fixture_records(count: int) -> list[dict]:
    """``count`` records whose prompt length increases strictly with the index.

    Strictly increasing lengths make percentile windows checkable by index, and
    distinct bodies keep the distinctness filter from collapsing the corpus.
    """
    tokenizer = _tokenizer()
    records = []
    for index in range(count):
        body = f"request {index:04d} " + f"word{index} " * (index + 1)
        records.append(
            builder.make_record(
                f"fixture-{index:04d}",
                [
                    {"role": "system", "content": f"system prompt {index % 3}"},
                    {"role": "user", "content": body},
                ],
                tokenizer,
                tools=[{"function": {"name": "read", "description": "read a file"}}],
                completion={"role": "assistant", "content": f"answer {index}"},
                timestamps=[f"2026-08-0{index % 9 + 1}T00:00:0{index % 10}Z"] * 2,
            )
        )
    return records


def _build_fixture_workload(tmp_path: Path, *, count: int = 200, name: str = "fixture-chat"):
    """Materialize a workload and its manifest exactly the way the builder does."""
    records = _fixture_records(count)
    records_path = tmp_path / "corpora" / name / "records.jsonl"
    builder.write_records(records_path, records)
    spec_dir = tmp_path / "specs"
    builder.emit_manifest(
        name=name,
        version="1.0.0",
        description="synthetic fixture workload",
        domain="fixture",
        records_path=records_path,
        records=records,
        provenance={"upstream": [], "method": "synthesized in a test"},
        tokenizer=_tokenizer(),
        output_reserve=128,
        needs_tool_support=True,
        ground_truth=None,
        spec_dir=spec_dir,
    )
    return W.load_spec(name, directory=spec_dir), records_path, spec_dir


@pytest.fixture
def fixture_workload(tmp_path: Path):
    return _build_fixture_workload(tmp_path)


def _rewrite_manifest(spec_dir: Path, name: str, mutate) -> W.WorkloadSpec:
    path = spec_dir / f"{name}.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return W.load_spec(name, directory=spec_dir)


# ---------------------------------------------------------------------------
# Band semantics
# ---------------------------------------------------------------------------


def test_band_is_a_percentile_window_and_excludes_the_extremes(fixture_workload):
    """A band must be a window over the distribution, never its tail.

    MUTATION: in ``workloads.band_window`` change
    ``start = min(total - 1, max(0, math.floor(band.lower * total)))`` to
    ``start = 0``.  RED: "short band reached the shortest record".
    """
    spec, _, _ = fixture_workload
    records = W.load_records(spec)
    shortest = min(record.prompt_chars for record in records)
    longest = max(record.prompt_chars for record in records)

    window = W.band_window(spec, "short", records)
    lengths = [record.prompt_chars for record in window]
    assert min(lengths) > shortest, (
        "short band reached the shortest record; a band taken from the extreme is "
        "the tail, not a sample of the distribution"
    )

    long_window = W.band_window(spec, "long", records)
    long_lengths = [record.prompt_chars for record in long_window]
    assert max(long_lengths) < longest, "long band reached the longest record"
    assert min(long_lengths) > max(lengths), "short and long bands overlap"

    medium = [r.prompt_chars for r in W.band_window(spec, "medium", records)]
    assert max(lengths) < min(medium) < max(medium) < min(long_lengths)


def test_band_prompts_are_verbatim_corpus_records(fixture_workload):
    """No truncation, no repetition padding: sampled text must exist in the corpus.

    MUTATION: in ``workloads.band_prompts`` return
    ``tuple(record.final_user_text[:64] for record in chosen)``.
    RED: "band produced text that is not any corpus record verbatim".
    """
    spec, _, _ = fixture_workload
    records = W.load_records(spec)
    corpus_texts = {record.final_user_text for record in records}
    for band in spec.bands:
        for prompt in W.band_prompts(spec, band, 5):
            assert prompt in corpus_texts, (
                "band produced text that is not any corpus record verbatim -- it was "
                "truncated, padded, or synthesized"
            )
            # Repetition padding leaves a text whose second half repeats its first.
            half = len(prompt) // 2
            assert half == 0 or prompt[:half] != prompt[half : 2 * half]


def test_band_refuses_to_return_fewer_than_requested(fixture_workload):
    """Exhaustion must be loud and must report both counts.

    MUTATION: in ``workloads.band_records`` replace the ``raise
    BandExhaustedError(...)`` with ``count = len(window)``.
    RED: "DID NOT RAISE tests.e2e.harness.workloads.BandExhaustedError".
    """
    spec, _, _ = fixture_workload
    records = W.load_records(spec)
    available = len(W.band_window(spec, "short", records))
    with pytest.raises(W.BandExhaustedError) as caught:
        W.band_records(spec, "short", available + 1, records=records)
    assert caught.value.requested == available + 1
    assert caught.value.available == available
    assert str(available) in str(caught.value)
    # ... and the boundary case still works, so this is not a blanket refusal.
    assert len(W.band_records(spec, "short", available, records=records)) == available


def test_band_sampling_is_deterministic_and_seed_sensitive(fixture_workload):
    """Same inputs, same sample; different seed, different sample.

    MUTATION: in ``workloads._seeded_random`` return ``random.Random()``.
    RED: "band sampling is not deterministic".
    """
    spec, _, _ = fixture_workload
    first = W.band_prompts(spec, "medium", 6)
    second = W.band_prompts(spec, "medium", 6)
    assert first == second, "band sampling is not deterministic"
    other = W.band_prompts(spec, "medium", 6, seed=spec.seed + 1)
    assert other != first, "changing the seed did not change the sample"
    assert set(other) <= {r.final_user_text for r in W.load_records(spec)}


def test_band_sample_holds_distinct_records(tmp_path: Path):
    """Duplicated prompts collapse; a band never pads its count with copies.

    MUTATION: in ``workloads.band_window`` delete the ``if digest in seen:
    continue`` guard.  RED: "band returned duplicate prompts".
    """
    tokenizer = _tokenizer()
    records = []
    for index in range(60):
        # Every prompt appears three times: a corpus with 20 distinct prompts.
        body = "duplicated body " * (index // 3 + 1)
        records.append(
            builder.make_record(
                f"dup-{index:03d}", [{"role": "user", "content": body}], tokenizer
            )
        )
    records_path = tmp_path / "dup" / "records.jsonl"
    builder.write_records(records_path, records)
    spec_dir = tmp_path / "specs"
    builder.emit_manifest(
        name="dup",
        version="1.0.0",
        description="duplicates",
        domain="fixture",
        records_path=records_path,
        records=records,
        provenance={"upstream": []},
        tokenizer=tokenizer,
        output_reserve=16,
        needs_tool_support=False,
        ground_truth=None,
        spec_dir=spec_dir,
    )
    spec = W.load_spec("dup", directory=spec_dir)
    window = W.band_window(spec, "long", W.load_records(spec))
    texts = [record.prompt_text for record in window]
    assert len(texts) == len(set(texts)), "band returned duplicate prompts"


def test_unknown_band_names_are_rejected(fixture_workload):
    """MUTATION: in ``WorkloadSpec.band`` return ``Band(name, 0.0, 1.0)`` on KeyError.

    RED: "DID NOT RAISE tests.e2e.harness.workloads.WorkloadError".
    """
    spec, _, _ = fixture_workload
    with pytest.raises(W.WorkloadError, match="has no band 'enormous'"):
        W.band_prompts(spec, "enormous", 1)


# ---------------------------------------------------------------------------
# Anti-vacuity verification -- the most important function in the module
# ---------------------------------------------------------------------------


def test_verification_accepts_the_corpus_it_describes(fixture_workload):
    """The green control: a validator that rejects everything proves nothing.

    MUTATION: in ``workloads.verify_workload`` append an unconditional
    ``failures.append("synthetic")``.  RED: "workload 'fixture-chat' failed
    verification: - synthetic".
    """
    spec, _, _ = fixture_workload
    records = W.verify_workload(spec)
    assert len(records) == spec.characteristics["record_count"] == 200


def test_verification_fails_when_the_corpus_is_truncated(fixture_workload):
    """A corpus that lost records must go red on the digest AND the counts.

    MUTATION: in ``workloads.verify_workload`` change the digest comparison to
    ``if False:`` and delete the record-count comparison.  RED: verification
    passes on a corpus missing half its records (test asserts it raised).
    """
    spec, records_path, _ = fixture_workload
    lines = records_path.read_text(encoding="utf-8").splitlines(keepends=True)
    records_path.write_text("".join(lines[:100]), encoding="utf-8")
    with pytest.raises(W.WorkloadVerificationError) as caught:
        W.verify_workload(spec)
    text = str(caught.value)
    assert "source.sha256" in text
    assert "characteristics.record_count" in text
    assert "declared 200, recomputed 100" in text


def test_verification_fails_when_a_record_body_is_edited_in_place(fixture_workload):
    """An edit that preserves byte count still moves the digest and the rendering.

    This is the "swapped data file" case: the file is the same size and the same
    line count, and only the content changed.

    MUTATION: in ``workloads.verify_workload`` delete the per-record
    ``prompt_chars`` block AND the digest check.  RED: an edited corpus verifies
    clean (test asserts it raised).
    """
    spec, records_path, _ = fixture_workload
    lines = records_path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    original = record["messages"][1]["content"]
    # Same length, different bytes: prompt_chars stays "correct", content does not.
    record["messages"][1]["content"] = "Z" * len(original)
    lines[0] = W.stable_json(record)
    records_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(W.WorkloadVerificationError, match="source.sha256"):
        W.verify_workload(spec)


def test_verification_fails_when_prompt_chars_disagrees_with_the_prompt(fixture_workload):
    """A stale cached length cannot hide behind the manifest.

    Digest checking is disabled here so the per-record integrity check is the
    only thing that can catch it -- otherwise this test would pass for the wrong
    reason.

    MUTATION: in ``workloads.verify_workload`` delete the ``mismatched`` block.
    RED: "DID NOT RAISE tests.e2e.harness.workloads.WorkloadVerificationError".
    """
    spec, records_path, _ = fixture_workload
    lines = records_path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["messages"][1]["content"] += " an appended sentence that costs characters"
    lines[0] = W.stable_json(record)
    records_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(W.WorkloadVerificationError, match="prompt_chars declared"):
        W.verify_workload(spec, check_digest=False)


def test_verification_fails_on_a_hand_typed_percentile(fixture_workload):
    """The whole point: declared statistics must be recomputed, not believed.

    MUTATION: in ``workloads.verify_workload`` skip the percentile block loop.
    RED: "DID NOT RAISE".
    """
    spec, _, spec_dir = fixture_workload
    spec = _rewrite_manifest(
        spec_dir,
        "fixture-chat",
        lambda m: m["characteristics"]["prompt_tokens"].update({"p50": 999_999}),
    )
    with pytest.raises(W.WorkloadVerificationError, match=r"prompt_tokens\.p50"):
        W.verify_workload(spec, check_digest=False)


def test_verification_fails_on_a_hand_typed_fraction(fixture_workload):
    """MUTATION: drop ``_FRACTION_KEYS`` from the comparison loop in verify_workload.

    RED: "DID NOT RAISE".
    """
    spec, _, spec_dir = fixture_workload
    spec = _rewrite_manifest(
        spec_dir,
        "fixture-chat",
        lambda m: m["characteristics"].update({"tool_call_fraction": 0.99}),
    )
    with pytest.raises(W.WorkloadVerificationError, match="tool_call_fraction"):
        W.verify_workload(spec, check_digest=False)


def test_verification_tolerance_is_narrow_enough_to_be_meaningful(fixture_workload):
    """A tolerance wide enough to swallow a real change is not a tolerance.

    MUTATION: set ``DEFAULT_TOLERANCE["percentile_relative"] = 0.5`` in
    build_workload.  RED: "DID NOT RAISE" (a 10% percentile shift passes).
    """
    spec, _, spec_dir = fixture_workload
    declared = spec.characteristics["prompt_chars"]["p50"]
    spec = _rewrite_manifest(
        spec_dir,
        "fixture-chat",
        lambda m: m["characteristics"]["prompt_chars"].update({"p50": declared * 1.10}),
    )
    with pytest.raises(W.WorkloadVerificationError, match=r"prompt_chars\.p50"):
        W.verify_workload(spec, check_digest=False)


def test_verification_fails_on_an_understated_serving_window(fixture_workload):
    """A requirement that would truncate the workload must not verify.

    MUTATION: in ``workloads.verify_workload`` delete the
    ``requirements.min_max_model_len`` block.  RED: "DID NOT RAISE".
    """
    spec, _, spec_dir = fixture_workload
    spec = _rewrite_manifest(
        spec_dir,
        "fixture-chat",
        lambda m: m["requirements"].update({"min_max_model_len": 16}),
    )
    with pytest.raises(W.WorkloadVerificationError, match="would truncate the workload"):
        W.verify_workload(spec, check_digest=False)


def test_verification_reports_every_disagreement_not_just_the_first(fixture_workload):
    """MUTATION: in ``WorkloadVerificationError.__init__`` keep only ``failures[:1]``.

    RED: "expected at least 3 failures, got 1".
    """
    spec, _, spec_dir = fixture_workload

    def mutate(manifest):
        manifest["characteristics"]["record_count"] = 7
        manifest["characteristics"]["tool_call_fraction"] = 0.5
        manifest["characteristics"]["prompt_chars"]["p90"] = 3

    spec = _rewrite_manifest(spec_dir, "fixture-chat", mutate)
    with pytest.raises(W.WorkloadVerificationError) as caught:
        W.verify_workload(spec, check_digest=False)
    assert len(caught.value.failures) >= 3, (
        f"expected at least 3 failures, got {len(caught.value.failures)}"
    )


def test_missing_corpus_is_an_error_not_an_empty_workload(fixture_workload):
    """MUTATION: in ``workloads.load_records`` return ``()`` instead of raising on a
    missing file.  RED: "DID NOT RAISE".
    """
    spec, records_path, _ = fixture_workload
    records_path.unlink()
    with pytest.raises(W.WorkloadVerificationError, match="corpus file is missing"):
        W.verify_workload(spec)


# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------


def test_prompts_by_context_has_the_legacy_shape(fixture_workload):
    """Drop-in for ``_prompts_by_context``: band name -> tuple of prompt strings.

    MUTATION: in ``workloads.prompts_by_context`` return
    ``{name: list(...)}``.  RED: "band 'short' is not a tuple".
    """
    spec, _, _ = fixture_workload
    result = W.prompts_by_context(spec, {"short": 4, "long": 4})
    assert set(result) == {"short", "long"}
    for band, prompts in result.items():
        assert isinstance(prompts, tuple), f"band {band!r} is not a tuple"
        assert len(prompts) == 4
        assert all(isinstance(prompt, str) and prompt for prompt in prompts)
    # An integer applies to every declared band.
    assert set(W.prompts_by_context(spec, 3)) == set(spec.bands)


def test_messages_by_context_preserves_system_and_history(fixture_workload):
    """The richer accessor must not throw away what makes agentic traffic costly.

    MUTATION: in ``workloads.band_messages`` return
    ``tuple(record.messages[-1:] for record in chosen)``.
    RED: "message list lost its system prompt".
    """
    spec, _, _ = fixture_workload
    result = W.messages_by_context(spec, {"medium": 3})
    for messages in result["medium"]:
        roles = [message["role"] for message in messages]
        assert roles[0] == "system", "message list lost its system prompt"
        assert "user" in roles
        assert all(isinstance(message.get("content"), str) for message in messages)
    # The lossy accessor really is lossy, and the rich one really is richer.
    flat = W.prompts_by_context(spec, {"medium": 3})["medium"]
    pairs = zip(result["medium"], flat, strict=True)
    assert all(len(W.flatten_prompt(m)) > len(p) for m, p in pairs)


def test_preflight_refuses_a_window_that_would_truncate(fixture_workload):
    """Both directions: refuse below the requirement, allow at or above it.

    MUTATION: in ``workloads.preflight_refusals`` change ``<`` to ``>``.
    RED: "preflight allowed a launch that would truncate".
    """
    spec, _, _ = fixture_workload
    required = int(spec.requirements["min_max_model_len"])
    refusals = W.preflight_refusals(spec, max_model_len=required - 1, tool_support=True)
    assert refusals, "preflight allowed a launch that would truncate the workload"
    assert str(required) in refusals[0]
    assert W.preflight_refusals(spec, max_model_len=required, tool_support=True) == ()
    tool_refusals = W.preflight_refusals(spec, max_model_len=required, tool_support=False)
    assert any("tool" in reason for reason in tool_refusals)


def test_adding_a_workload_needs_no_test_code_change(tmp_path: Path):
    """The declarative promise: a new JSON file is a new workload.

    MUTATION: in ``workloads.available_workloads`` return a hardcoded tuple of the
    four shipped names.  RED: "a manifest dropped into the spec directory was not
    discovered".
    """
    spec, _, spec_dir = _build_fixture_workload(tmp_path, count=80, name="alpha")
    assert W.available_workloads(directory=spec_dir) == ("alpha",)
    _build_fixture_workload(tmp_path, count=80, name="beta")
    assert W.available_workloads(directory=spec_dir) == ("alpha", "beta"), (
        "a manifest dropped into the spec directory was not discovered"
    )
    assert {s.name for s in W.iter_specs(directory=spec_dir)} == {"alpha", "beta"}


def test_a_manifest_with_the_wrong_schema_version_is_refused(tmp_path: Path):
    """MUTATION: in ``workloads.load_spec_file`` delete the schema_version check.

    RED: "DID NOT RAISE".
    """
    spec, _, spec_dir = _build_fixture_workload(tmp_path, count=40, name="gamma")
    path = spec_dir / "gamma.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 99
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(W.WorkloadError, match="schema_version"):
        W.load_spec("gamma", directory=spec_dir)


def test_a_manifest_missing_required_keys_is_refused(tmp_path: Path):
    """MUTATION: in ``workloads.load_spec_file`` drop the ``missing`` check.

    RED: "DID NOT RAISE" (the KeyError surfaces later, at sampling time).
    """
    spec, _, spec_dir = _build_fixture_workload(tmp_path, count=40, name="delta")
    path = spec_dir / "delta.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    del manifest["requirements"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(W.WorkloadError, match="missing keys: requirements"):
        W.load_spec("delta", directory=spec_dir)


# ---------------------------------------------------------------------------
# The ported trace exporter (long-context-sessions)
# ---------------------------------------------------------------------------

COLLEAGUE_PATH = "/admin/home/other.person/secrets/notes.txt"


def test_redaction_covers_this_clusters_home_layout():
    """The ported exporter fixes a real leak; this test proves the leak existed.

    Upstream ``ABS_PATH_PATTERN`` is ``(?<![\\w.-])/(?:home|Users)/...``.  Against
    ``/admin/home/other.person/...`` the lookbehind sees the ``n`` of ``admin``
    and refuses, so a colleague's paths pass through unredacted.

    MUTATION: set ``build_workload.ABS_PATH_PATTERN =
    build_workload.UPSTREAM_ABS_PATH_PATTERN``.  RED: "the ported pattern did not
    redact /admin/home/other.person/secrets/notes.txt".
    """
    # The defect, demonstrated rather than asserted from memory.
    assert builder.UPSTREAM_ABS_PATH_PATTERN.search(COLLEAGUE_PATH) is None, (
        "the upstream pattern unexpectedly matched; this test's premise is stale"
    )
    assert builder.UPSTREAM_ABS_PATH_PATTERN.search("/home/other.person/x") is not None

    redacted = builder.redact(f"see {COLLEAGUE_PATH} for details")
    assert COLLEAGUE_PATH not in redacted, (
        f"the ported pattern did not redact {COLLEAGUE_PATH}"
    )
    assert "other.person" not in redacted
    # Plain /home and /Users layouts still work.
    assert "/home/someone/a.py" not in builder.redact("/home/someone/a.py")
    assert "/Users/someone/a.py" not in builder.redact("/Users/someone/a.py")
    # A relative path inside a diff must NOT be mangled -- over-redaction would
    # destroy exactly the code content that makes these prompts realistic.
    assert builder.redact("src/home/widget.py") == "src/home/widget.py"


def _session_file(path: Path, turns: list[tuple[str, object, str]]) -> Path:
    """Write a Claude-Code-shaped session .jsonl (uuid/parentUuid/timestamp chain)."""
    lines = []
    parent = None
    for index, (role, content, timestamp) in enumerate(turns):
        uuid = f"uuid-{index:03d}"
        message: dict = {"role": role, "content": content}
        if role == "assistant":
            message["model"] = "claude-opus-4-8"
        lines.append(
            json.dumps(
                {
                    "uuid": uuid,
                    "parentUuid": parent,
                    "type": role,
                    "sessionId": "session-1",
                    "timestamp": timestamp,
                    "message": message,
                }
            )
        )
        parent = uuid
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _long(text: str, n: int = 400) -> str:
    return (text + " ") * n


def _demo_session(tmp_path: Path) -> Path:
    return _session_file(
        tmp_path / "session-1.jsonl",
        [
            ("user", _long("please fix the failing test"), "2026-08-01T00:00:00Z"),
            ("assistant", _long("I will look at it"), "2026-08-01T00:00:05Z"),
            ("user", _long("here is the traceback"), "2026-08-01T00:00:09Z"),
            # The modal agent turn: a short tool call, 41 characters.
            ("assistant", "running the tests now to reproduce it", "2026-08-01T00:00:20Z"),
        ],
    )


def test_low_assistant_floor_keeps_the_modal_short_tool_turn(tmp_path: Path):
    """The upstream 800-char floor deletes short turns and inverts decode lengths.

    MUTATION: in ``build_workload.traces_from_session_file`` hardcode
    ``minimum_assistant_chars = 800``.  RED: "the low floor dropped the short
    assistant turn".
    """
    session = _demo_session(tmp_path)
    kept = builder.traces_from_session_file(
        session,
        minimum_assistant_chars=1,
        minimum_prompt_chars=0,
        per_file_limit=5,
        model_counter=__import__("collections").Counter(),
    )
    completions = [trace.completion["content"] for trace in kept]
    assert any(len(text) < 800 for text in completions), (
        "the low floor dropped the short assistant turn"
    )

    # And the upstream floor really would have dropped it -- the defect is real.
    with_upstream_floor = builder.traces_from_session_file(
        session,
        minimum_assistant_chars=800,
        minimum_prompt_chars=0,
        per_file_limit=5,
        model_counter=__import__("collections").Counter(),
    )
    assert all(
        len(trace.completion["content"]) >= 800 for trace in with_upstream_floor
    )
    assert len(with_upstream_floor) < len(kept)


def test_per_turn_timestamps_are_carried(tmp_path: Path):
    """Arrival shape must stay recoverable; upstream discarded these.

    MUTATION: in ``build_workload.traces_from_session_file`` replace
    ``timestamps=tuple(timestamps)`` with ``timestamps=()``.
    RED: "trace carried no per-turn timestamps".
    """
    session = _demo_session(tmp_path)
    traces = builder.traces_from_session_file(
        session,
        minimum_assistant_chars=1,
        minimum_prompt_chars=0,
        per_file_limit=1,
        model_counter=__import__("collections").Counter(),
    )
    assert traces, "fixture session produced no traces"
    trace = traces[0]
    assert trace.timestamps, "trace carried no per-turn timestamps"
    assert len(trace.timestamps) == len(trace.messages)
    assert trace.timestamps[0] == "2026-08-01T00:00:00Z"
    assert re.match(r"\d{4}-\d{2}-\d{2}T", trace.timestamps[-1])


def test_exporter_accepts_non_qwen_assistants(tmp_path: Path):
    """Upstream hard-filtered on one Qwen model id and would export ZERO here.

    MUTATION: in ``build_workload._real_model`` return
    ``model if model == "Qwen/Qwen3.6-27B" else None``.
    RED: "the exporter yielded nothing from a claude-* session".
    """
    session = _demo_session(tmp_path)
    counter = __import__("collections").Counter()
    traces = builder.traces_from_session_file(
        session,
        minimum_assistant_chars=1,
        minimum_prompt_chars=0,
        per_file_limit=5,
        model_counter=counter,
    )
    assert traces, "the exporter yielded nothing from a claude-* session"
    assert "claude-opus-4-8" in counter
    # A synthetic placeholder model is still refused.
    placeholder = _session_file(
        tmp_path / "synthetic.jsonl",
        [("user", _long("hello"), "2026-08-01T00:00:00Z")],
    )
    assert (
        builder.traces_from_session_file(
            placeholder,
            minimum_assistant_chars=1,
            minimum_prompt_chars=0,
            per_file_limit=5,
            model_counter=__import__("collections").Counter(),
        )
        == []
    )


def test_exported_trace_prompt_ends_on_a_user_turn(tmp_path: Path):
    """A prompt prefix that ends on an assistant turn is not a servable request.

    MUTATION: in ``build_workload.traces_from_session_file`` delete the
    ``messages[-1]["role"] != "user"`` guard.  RED: this test still passes on the
    demo session, so the mutation was shown against a session whose deepest
    assistant turn follows another assistant turn (see the extra fixture below).
    """
    session = _session_file(
        tmp_path / "trailing.jsonl",
        [
            ("user", _long("start"), "2026-08-01T00:00:00Z"),
            ("assistant", _long("thinking out loud"), "2026-08-01T00:00:01Z"),
            ("assistant", _long("still thinking"), "2026-08-01T00:00:02Z"),
        ],
    )
    traces = builder.traces_from_session_file(
        session,
        minimum_assistant_chars=1,
        minimum_prompt_chars=0,
        per_file_limit=5,
        model_counter=__import__("collections").Counter(),
    )
    for trace in traces:
        assert trace.messages[-1]["role"] == "user", (
            "exported a prompt whose last message is not a user turn"
        )


# ---------------------------------------------------------------------------
# The shipped manifests -- structure in CI, data when /data is present
# ---------------------------------------------------------------------------

SHIPPED = ("generic-chat", "agentic-tool-loop", "agentic-mixed-outcome", "long-context-sessions")


@pytest.mark.parametrize("name", SHIPPED)
def test_shipped_manifest_is_structurally_complete(name: str):
    """Runs in CI with no /data: parse the manifest and check it declares enough.

    MUTATION: delete ``"p99"`` from ``workloads.PERCENTILE_KEYS`` and rebuild is
    not needed -- instead remove ``"p99"`` from a shipped manifest's
    ``characteristics.prompt_tokens``.  RED: "generic-chat is missing declared
    percentile p99".
    """
    if not (SHIPPED_SPECS / f"{name}.json").is_file():
        pytest.skip(f"{name} manifest has not been built yet")
    spec = W.load_spec(name)
    assert spec.description and len(spec.description) > 200, (
        "a workload description that does not state its limitations is how an "
        "unrepresentative corpus gets treated as representative"
    )
    for block in ("prompt_chars", "prompt_tokens"):
        for key in W.PERCENTILE_KEYS:
            assert key in spec.characteristics[block], (
                f"{name} is missing declared percentile {key} in {block}"
            )
    assert spec.characteristics["record_count"] > 0
    assert set(spec.bands) == {"short", "medium", "long"}
    assert spec.requirements["min_max_model_len"] >= 512
    assert spec.provenance["token_basis"].startswith("tokenizer:"), (
        f"{name} token counts are estimates, not measurements"
    )
    assert spec.provenance["upstream"], f"{name} declares no upstream provenance"


def test_the_agentic_workloads_exceed_the_current_serving_window():
    """The realism constraint, encoded: three of four do not fit --max-model-len 4096.

    MUTATION: set ``requirements.min_max_model_len`` to 4096 in
    agentic-tool-loop.json.  RED: "agentic-tool-loop fits in 4096".
    """
    present = [name for name in SHIPPED if (SHIPPED_SPECS / f"{name}.json").is_file()]
    if len(present) < len(SHIPPED):
        pytest.skip("not every manifest has been built yet")
    for name in ("agentic-tool-loop", "agentic-mixed-outcome", "long-context-sessions"):
        spec = W.load_spec(name)
        assert W.preflight_refusals(spec, max_model_len=4096, tool_support=True), (
            f"{name} fits in 4096; the config matrix could run it untruncated, "
            "which contradicts the measured prompt lengths"
        )
    control = W.load_spec("generic-chat")
    assert W.preflight_refusals(control, max_model_len=4096, tool_support=True) == (), (
        "generic-chat should fit in 4096 -- it is the control"
    )


def test_agentic_mixed_outcome_carries_measured_ground_truth():
    """The measured acceptance rate is the only external check this project has.

    MUTATION: in ``build_workload.mixed_ground_truth`` return ``{}``, rebuild.
    Cheaper equivalent: delete the ``ground_truth`` key from the manifest.
    RED: "agentic-mixed-outcome declares no ground_truth block".
    """
    path = SHIPPED_SPECS / "agentic-mixed-outcome.json"
    if not path.is_file():
        pytest.skip("agentic-mixed-outcome manifest has not been built yet")
    spec = W.load_spec("agentic-mixed-outcome")
    truth = spec.ground_truth
    assert truth, "agentic-mixed-outcome declares no ground_truth block"
    assert truth["metric"] == "acceptance_fraction"
    assert 0.0 < truth["p5"] < truth["p50"] < truth["p95"] <= 1.0
    assert truth["n"] == spec.characteristics["record_count"]


def _data_available(name: str) -> bool:
    path = SHIPPED_SPECS / f"{name}.json"
    if not path.is_file():
        return False
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return Path(manifest["source"]["path"]).is_file()


@pytest.mark.parametrize("name", SHIPPED)
@pytest.mark.skipif(
    not any(_data_available(n) for n in SHIPPED),
    reason="materialized workload corpora are not present (they live under /data)",
)
def test_shipped_workload_verifies_against_its_real_corpus(name: str):
    """The real thing, when /data is mounted.  Its logic twin above runs in CI.

    MUTATION: append one line to
    /data/ryan.kim/speedlm-workloads/generic-chat/records.jsonl.
    RED: "source.sha256: declared ... on disk ...".
    """
    if not _data_available(name):
        pytest.skip(f"{name} corpus is not present")
    spec = W.load_spec(name)
    records = W.verify_workload(spec)
    assert len(records) == spec.characteristics["record_count"]
    for band in spec.bands:
        sample = W.band_records(spec, band, 4, records=records)
        assert len({record.id for record in sample}) == 4
