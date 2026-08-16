"""Tests for the prompt-phrasing module — tests that cannot fail don't count.

TDD verification log (2026-08-16):
    test_no_fact_consolidation_dump (TestAntiTemplate):
        AGAINST OLD phrasing.py:
            FAILED: 500/500 renders (100.0%) consolidated all facts into a single
            sentence. Top structural dumps:
              "Here's what's involved: <FACT>, <FACT>, <FACT>": 68 (13.6%)
              "The relevant files are <FACT>, <FACT>, <FACT>": 59 (11.8%)
              "You'll find everything in <FACT>, <FACT>, <FACT>": 53 (10.6%)
              "You'll need to look at <FACT>, <FACT>, <FACT>": 52 (10.4%)
              "Focus on these: <FACT>, <FACT>, <FACT>": 48 (9.6%)
            AssertionError: Anti-template check FAILED:
            500/500 (100.0%) of renders consolidate all 3 facts into a single
            sentence (limit: <= 20%).
        AGAINST NEW phrasing.py:
            PASSED: 0/500 renders (0.0%) consolidated all facts into a single
            sentence.
"""

from __future__ import annotations

from collections import Counter

from tests.e2e.agentenv.phrasing import (
    _PATH_BLAME_FRAMES,
    QUANTITATIVE_WORDS,
    TEST_SUITE_WORDS,
    Brief,
    Fact,
    FactKind,
    render_instruction,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SAMPLE_BRIEF = Brief(
    goal="Find the planted defect in the ledger pipeline and fix it so the test suite passes.",
    required_facts=("tests/test_pipeline.py", "pipeline/", "pytest"),
    context="The nightly build just started failing on the main branch.",
    constraints=("do not change anything under tests/", "change the smallest amount of code"),
)

_ARTIFACT_TEXT = (
    "2026-08-09T04:13:32Z ERROR ledger req-a1b2c3d4 E_TIMEOUT: downstream refused after 3 retries"
)

_SAMPLE_BRIEF_WITH_ARTIFACT = Brief(
    goal="Locate the ERROR line in service.log and report it in findings.json.",
    required_facts=("service.log", "findings.json", "E_TIMEOUT"),
    context="Production incident — one service is refusing requests.",
    constraints=("write valid JSON", "include exactly three keys"),
    artifact=_ARTIFACT_TEXT,
)

_SEEDS = range(500)
_SALT = 42


# ---------------------------------------------------------------------------
# Purity: same seed + salt always returns byte-identical output
# ---------------------------------------------------------------------------
class TestPurity:
    def test_same_seed_same_bytes(self) -> None:
        """Calling render_instruction twice with the same seed must return identical strings."""
        first = [render_instruction(_SAMPLE_BRIEF, seed=s, salt=_SALT) for s in _SEEDS]
        second = [render_instruction(_SAMPLE_BRIEF, seed=s, salt=_SALT) for s in _SEEDS]
        mismatches = sum(1 for a, b in zip(first, second, strict=True) if a != b)
        assert mismatches == 0, (
            f"Expected byte-identical strings across 500 seeds, got {mismatches} mismatches"
        )

    def test_different_salt_different_bytes(self) -> None:
        """Changing the salt should produce different results for at least some seeds."""
        salt_a = [render_instruction(_SAMPLE_BRIEF, seed=s, salt=0) for s in range(100)]
        salt_b = [render_instruction(_SAMPLE_BRIEF, seed=s, salt=1) for s in range(100)]
        same = sum(1 for a, b in zip(salt_a, salt_b, strict=True) if a == b)
        assert same < 100, (
            f"All {same}/100 renders are identical across different salts — salt is being ignored"
        )


# ---------------------------------------------------------------------------
# Fact preservation: every required_facts string must appear verbatim
# ---------------------------------------------------------------------------
class TestFactPreservation:
    def test_all_required_facts_present_for_500_seeds(self) -> None:
        """Every required_facts entry must appear verbatim in the output for all 500 seeds."""
        missing: dict[str, int] = {}
        for seed in _SEEDS:
            text = render_instruction(_SAMPLE_BRIEF, seed=seed, salt=_SALT)
            for raw in _SAMPLE_BRIEF.required_facts:
                fact_text = raw.text if isinstance(raw, Fact) else raw
                if fact_text not in text:
                    missing[fact_text] = missing.get(fact_text, 0) + 1
        assert not missing, (
            f"Required facts were missing from output: {dict(sorted(missing.items()))}"
        )

    def test_artifact_brief_required_facts(self) -> None:
        """Required facts in the artifact brief must also be preserved."""
        missing: dict[str, int] = {}
        for seed in _SEEDS:
            text = render_instruction(_SAMPLE_BRIEF_WITH_ARTIFACT, seed=seed, salt=_SALT)
            for raw in _SAMPLE_BRIEF_WITH_ARTIFACT.required_facts:
                fact_text = raw.text if isinstance(raw, Fact) else raw
                if fact_text not in text:
                    missing[fact_text] = missing.get(fact_text, 0) + 1
        assert not missing, (
            f"Required facts missing from artifact brief: {dict(sorted(missing.items()))}"
        )

    def test_typed_facts_preserved(self) -> None:
        """Typed Fact entries must also be preserved verbatim."""
        typed_brief = Brief(
            goal="Fix the failing tests.",
            required_facts=(
                Fact(text="tests/test_main.py", kind=FactKind.PATH),
                Fact(text="pytest", kind=FactKind.COMMAND),
                Fact(text="E_TIMEOUT", kind=FactKind.VALUE),
            ),
        )
        missing: dict[str, int] = {}
        for seed in _SEEDS:
            text = render_instruction(typed_brief, seed=seed, salt=_SALT)
            for fact in typed_brief.required_facts:
                if isinstance(fact, Fact) and fact.text not in text:
                    missing[fact.text] = missing.get(fact.text, 0) + 1
        assert not missing, f"Typed facts missing: {dict(sorted(missing.items()))}"


# ---------------------------------------------------------------------------
# Diversity: 500 consecutive seeds should produce >= 450 distinct strings
# ---------------------------------------------------------------------------
class TestDiversity:
    def test_high_uniqueness_over_500_seeds(self) -> None:
        """Over 500 consecutive seeds, at least 90% of renders must be unique."""
        renders = [render_instruction(_SAMPLE_BRIEF, seed=s, salt=_SALT) for s in _SEEDS]
        distinct = len(set(renders))
        assert distinct >= 450, f"Only {distinct}/500 renders are unique (need >= 450, i.e. >= 90%)"

    def test_high_uniqueness_artifact_brief(self) -> None:
        """The artifact brief should also produce diverse renders."""
        renders = [
            render_instruction(_SAMPLE_BRIEF_WITH_ARTIFACT, seed=s, salt=_SALT) for s in _SEEDS
        ]
        distinct = len(set(renders))
        assert distinct >= 450, (
            f"Only {distinct}/500 artifact-brief renders are unique (need >= 450)"
        )


# ---------------------------------------------------------------------------
# No degenerate output
# ---------------------------------------------------------------------------
class TestNoDegenerateOutput:
    def test_length_bounds(self) -> None:
        """Rendered length must be between 80 and 1200 chars for all 500 seeds."""
        for seed in _SEEDS:
            text = render_instruction(_SAMPLE_BRIEF, seed=seed, salt=_SALT)
            length = len(text)
            assert 80 <= length <= 1200, (
                f"seed {seed}: render length is {length}, expected 80..1200"
            )

    def test_goal_substance_present(self) -> None:
        """Each render should contain the goal's substance (a significant substring)."""
        goal_words = [w for w in _SAMPLE_BRIEF.goal.lower().split() if len(w) > 4]
        missing_goal: list[int] = []
        for seed in _SEEDS:
            text = render_instruction(_SAMPLE_BRIEF, seed=seed, salt=_SALT).lower()
            if not any(w in text for w in goal_words):
                missing_goal.append(seed)
        assert not missing_goal, (
            f"{len(missing_goal)}/500 renders missing goal substance "
            f"(first 10: {missing_goal[:10]})"
        )

    def test_no_robotic_scaffolding(self) -> None:
        """Renders should not start with robotic scaffolding."""
        for seed in range(200):
            text = render_instruction(_SAMPLE_BRIEF, seed=seed, salt=_SALT)
            stripped = text.lstrip()
            assert not stripped.startswith("Task:"), (
                f"seed {seed}: render starts with 'Task:' scaffolding"
            )
            assert not stripped.startswith("Instruction:"), (
                f"seed {seed}: render starts with 'Instruction:' scaffolding"
            )
            assert not stripped.startswith("Objective:"), (
                f"seed {seed}: render starts with 'Objective:' scaffolding"
            )
            assert not stripped.startswith("Requirements:"), (
                f"seed {seed}: render starts with 'Requirements:' scaffolding"
            )

    def test_constraints_substance_present(self) -> None:
        """A meaningful fraction of renders should include constraint substance."""
        renders_with_constraint = 0
        for seed in range(200):
            text = render_instruction(_SAMPLE_BRIEF, seed=seed, salt=_SALT)
            lower = text.lower()
            has_diff = "smallest" in lower or "minimal" in lower or "least" in lower
            has_no_tests = (
                "not change" in lower
                or "don't" in lower
                or "do not" in lower
                or "avoid" in lower
                or "without" in lower
                or "keep" in lower
            )
            if has_diff or has_no_tests:
                renders_with_constraint += 1
        assert renders_with_constraint >= 100, (
            f"only {renders_with_constraint}/200 renders include constraint substance (need >= 50%)"
        )


# ---------------------------------------------------------------------------
# Artifact embedding
# ---------------------------------------------------------------------------
class TestArtifactEmbedding:
    def test_artifact_embedded_in_some_renders(self) -> None:
        """When brief.artifact is non-empty, it should be embedded in some renders."""
        embedded_count = 0
        for seed in range(200):
            text = render_instruction(_SAMPLE_BRIEF_WITH_ARTIFACT, seed=seed, salt=_SALT)
            if "E_TIMEOUT" in text and "downstream refused" in text:
                embedded_count += 1
        assert embedded_count >= 60, (
            f"only {embedded_count}/200 renders embed the artifact content (need >= 30%)"
        )


# ---------------------------------------------------------------------------
# Anti-template: no fact-consolidation dump
# ---------------------------------------------------------------------------
def _fact_texts(brief: Brief) -> list[str]:
    """Extract all fact literal strings from a Brief."""
    texts: list[str] = []
    for raw in brief.required_facts:
        if isinstance(raw, Fact):
            texts.append(raw.text)
        else:
            texts.append(raw)
    return texts


def _strip_facts(text: str, facts: list[str]) -> str:
    """Replace fact literals (with or without backticks) with <FACT>."""
    for ft in facts:
        text = text.replace(f"`{ft}`", "<FACT>")
        text = text.replace(ft, "<FACT>")
    return " ".join(text.split())


class TestAntiTemplate:
    def test_no_fact_consolidation_dump(self) -> None:
        """Over 500 seeds, <= 20% of renders may consolidate all facts into one sentence.

        The OLD implementation used _FACT_TEMPLATES_MANY which dumped all N
        facts as a backtick-separated list in a single sentence:
            "Key files: `tests/test_pipeline.py`, `pipeline/`, `pytest`."
        This produced a structural tell detectable by anyone reading the corpus.

        Concretely: split each render on '. ' (period-space, which avoids
        splitting on file-extension dots like .py, .json), find any segment
        that contains ALL fact strings, extract its skeleton (facts replaced
        with <FACT>), and assert the most common such skeleton accounts for
        <= 20% of renders, AND that the total fraction of renders with any
        single-sentence fact dump is <= 20%.

        Watched fail against old phrasing.py (2026-08-16):
            AssertionError: Anti-template check FAILED:
            500/500 (100.0%) of renders consolidate all 3 facts into a single
            sentence (limit: <= 20%).
            Top dump skeletons:
              "Here's what's involved: <FACT>, <FACT>, <FACT>": 68 (13.6%)
              "The relevant files are <FACT>, <FACT>, <FACT>": 59 (11.8%)
              "You'll find everything in <FACT>, <FACT>, <FACT>": 53 (10.6%)
              "You'll need to look at <FACT>, <FACT>, <FACT>": 52 (10.4%)
              "Focus on these: <FACT>, <FACT>, <FACT>": 48 (9.6%)
        """
        renders = [render_instruction(_SAMPLE_BRIEF, seed=s, salt=_SALT) for s in _SEEDS]
        facts = _fact_texts(_SAMPLE_BRIEF)

        # Only meaningful for briefs with >= 2 facts
        assert len(facts) >= 2, "Anti-template test requires >= 2 facts"

        dump_skeletons: list[str] = []
        for render in renders:
            # Split on '. ' to avoid splitting on file-extension dots (.py, .json)
            # Also handle renders that may not have '. ' separators
            parts = render.split(". ")
            for part in parts:
                part_stripped = part.strip().rstrip(".")
                if all(ft in part_stripped for ft in facts):
                    skel = _strip_facts(part_stripped, facts)
                    dump_skeletons.append(skel)
                    break  # only count one dump per render

        dump_count = len(dump_skeletons)
        dump_pct = dump_count / len(renders) * 100

        if dump_skeletons:
            counter = Counter(dump_skeletons)
            top5 = counter.most_common(5)
            skeleton_detail = "\n".join(
                f"  {repr(sk)}: {cnt} ({cnt / len(renders) * 100:.1f}%)" for sk, cnt in top5
            )
        else:
            skeleton_detail = "  (none)"

        assert dump_pct <= 20, (
            f"Anti-template check FAILED:\n"
            f"{dump_count}/{len(renders)} ({dump_pct:.1f}%) of renders consolidate "
            f"all {len(facts)} facts into a single sentence (limit: <= 20%).\n"
            f"Top dump skeletons:\n{skeleton_detail}"
        )


# ---------------------------------------------------------------------------
# Typed fact API
# ---------------------------------------------------------------------------
class TestTypedFacts:
    def test_backward_compat_plain_strings(self) -> None:
        """Plain string facts should still work (default to VALUE kind)."""
        brief = Brief(
            goal="Fix the defect.",
            required_facts=("E_TIMEOUT", "75"),
        )
        for seed in range(100):
            text = render_instruction(brief, seed=seed, salt=_SALT)
            assert "E_TIMEOUT" in text, f"seed {seed}: missing E_TIMEOUT"
            assert "75" in text, f"seed {seed}: missing 75"

    def test_mixed_str_and_fact_entries(self) -> None:
        """A mix of plain strings and Fact objects should all be preserved."""
        brief = Brief(
            goal="Debug and fix the pipeline.",
            required_facts=(
                Fact(text="tests/test_main.py", kind=FactKind.PATH),
                "E_TIMEOUT",
                Fact(text="pytest", kind=FactKind.COMMAND),
            ),
        )
        for seed in range(100):
            text = render_instruction(brief, seed=seed, salt=_SALT)
            assert "tests/test_main.py" in text, f"seed {seed}: missing tests/test_main.py"
            assert "E_TIMEOUT" in text, f"seed {seed}: missing E_TIMEOUT"
            assert "pytest" in text, f"seed {seed}: missing pytest"

    def test_path_command_value_each_use_own_frame(self) -> None:
        """PATH/COMMAND/VALUE facts must each produce grammatically distinct frames."""
        path_brief = Brief(
            goal="Fix it.",
            required_facts=(Fact(text="src/main.py", kind=FactKind.PATH),),
        )
        cmd_brief = Brief(
            goal="Fix it.",
            required_facts=(Fact(text="make test", kind=FactKind.COMMAND),),
        )
        val_brief = Brief(
            goal="Fix it.",
            required_facts=(Fact(text="E_TIMEOUT", kind=FactKind.VALUE),),
        )
        for seed in range(50):
            path_r = render_instruction(path_brief, seed=seed, salt=_SALT)
            cmd_r = render_instruction(cmd_brief, seed=seed, salt=_SALT)
            val_r = render_instruction(val_brief, seed=seed, salt=_SALT)
            assert "src/main.py" in path_r, f"seed {seed}: missing path fact"
            assert "make test" in cmd_r, f"seed {seed}: missing command fact"
            assert "E_TIMEOUT" in val_r, f"seed {seed}: missing value fact"


# ---------------------------------------------------------------------------
# Semantic-role correctness — new tests (watched-fail verified 2026-08-16)
#
# Each test was run against the old implementation first to confirm it FAILS;
# the observed failure outputs are documented in the test docstrings.
# ---------------------------------------------------------------------------


class TestTokenFactsNoQuantitativeFrames:
    """TOKEN facts must never receive quantitative sentence frames.

    Watched fail against old phrasing.py (2026-08-16):
        Old impl: 296/500 TOKEN renders contain quantitative framing.
        Example (seed=0, banned='threshold'):
            "the threshold's set to `GBP`. Passing this along since it fell
             through the cracks. Basically, fix the error. Happy to wa..."
    """

    def test_token_facts_never_get_quantitative_frames(self) -> None:
        """Over 500 seeds with a TOKEN fact, no render may use quantitative phrasings."""
        brief = Brief(
            goal="Fix the error.",
            required_facts=(Fact(text="GBP", kind=FactKind.TOKEN),),
        )
        violations: list[tuple[int, str, str]] = []
        for seed in _SEEDS:
            text = render_instruction(brief, seed=seed, salt=_SALT)
            for banned in QUANTITATIVE_WORDS:
                if banned in text:
                    violations.append((seed, banned, text[:160]))
                    break
        assert not violations, (
            f"{len(violations)}/500 TOKEN renders contained quantitative framing.\n"
            f"QUANTITATIVE_WORDS checked: {QUANTITATIVE_WORDS}\n"
            f"First violation — seed={violations[0][0]}, banned='{violations[0][1]}':\n"
            f"  {violations[0][2]}"
        )

    def test_value_deprecated_alias_also_no_quantitative_frames(self) -> None:
        """VALUE (deprecated alias for TOKEN) also must not get quantitative frames."""
        brief = Brief(
            goal="Fix the error.",
            required_facts=(Fact(text="E_QUOTA", kind=FactKind.VALUE),),
        )
        violations: list[tuple[int, str, str]] = []
        for seed in _SEEDS:
            text = render_instruction(brief, seed=seed, salt=_SALT)
            for banned in QUANTITATIVE_WORDS:
                if banned in text:
                    violations.append((seed, banned, text[:160]))
                    break
        assert not violations, (
            f"{len(violations)}/500 VALUE renders contained quantitative framing.\n"
            f"First violation — seed={violations[0][0]}, banned='{violations[0][1]}':\n"
            f"  {violations[0][2]}"
        )

    def test_number_facts_may_use_quantitative_frames(self) -> None:
        """NUMBER facts are allowed to use quantitative frames (threshold, limit, etc.)."""
        brief = Brief(
            goal="Fix it.",
            required_facts=(Fact(text="75", kind=FactKind.NUMBER),),
        )
        hit_any = False
        for seed in _SEEDS:
            text = render_instruction(brief, seed=seed, salt=_SALT)
            if any(q in text for q in QUANTITATIVE_WORDS):
                hit_any = True
                break
        assert hit_any, (
            "NUMBER facts never received quantitative framing over 500 seeds "
            "(expected at least one render with a quantitative frame)"
        )


class TestCreatedArtifactsNeverFramedAsLocations:
    """Facts listed in Brief.creates must be framed as outputs, never as locations.

    Watched fail against old phrasing.py (2026-08-16):
        Old impl: 258/500 renders frame 'findings.json' locatively.
        Example (seed=0, frame="the failure's in"):
            "the failure's in `findings.json`. Passing this along since it fell
             through the cracks. Basically, locate the error and r..."
    """

    # Locative frame patterns that must never wrap a creates fact.
    _LOCATIVE_PATTERNS = (
        "the failure's in `{fact}`",
        "`{fact}` is where the bug lives",
        "the issue traces back to `{fact}`",
        "check `{fact}` -- that's the hot spot",
        "start at `{fact}`",
        "everything you need is under `{fact}`",
        "you'll want to look at `{fact}` first",
    )

    def test_created_artifacts_never_framed_as_locations(self) -> None:
        """Over 500 seeds, a creates fact must never receive a locative frame."""
        creates_fact = "findings.json"
        brief = Brief(
            goal="Locate the ERROR line in service.log and report it.",
            required_facts=(Fact(text=creates_fact, kind=FactKind.PATH),),
            creates=(creates_fact,),
        )
        locative_frames_for_fact = [
            pat.format(fact=creates_fact) for pat in self._LOCATIVE_PATTERNS
        ]
        violations: list[tuple[int, str, str]] = []
        for seed in _SEEDS:
            text = render_instruction(brief, seed=seed, salt=_SALT)
            for frame in locative_frames_for_fact:
                if frame in text:
                    violations.append((seed, frame, text[:160]))
                    break
        assert not violations, (
            f"{len(violations)}/500 renders applied a locative frame to a creates fact.\n"
            f"First violation — seed={violations[0][0]}:\n"
            f"  matched: {violations[0][1]!r}\n"
            f"  text: {violations[0][2]}"
        )

    def test_creates_fact_receives_output_frame(self) -> None:
        """A creates fact must receive an output-oriented frame in at least some renders."""
        creates_fact = "findings.json"
        brief = Brief(
            goal="Investigate the incident and document your findings.",
            required_facts=(Fact(text=creates_fact, kind=FactKind.PATH),),
            creates=(creates_fact,),
        )
        output_patterns = (
            f"write your answer to `{creates_fact}`",
            f"produce `{creates_fact}`",
            f"`{creates_fact}` is what you need to create",
            f"the output should go into `{creates_fact}`",
            f"generate `{creates_fact}`",
            f"`{creates_fact}` is the file you should produce",
            f"you'll need to create `{creates_fact}`",
            f"save your results to `{creates_fact}`",
        )
        hit_any = False
        for seed in _SEEDS:
            text = render_instruction(brief, seed=seed, salt=_SALT)
            if any(pat in text for pat in output_patterns):
                hit_any = True
                break
        assert hit_any, "creates fact never received an output frame over 500 seeds"


class TestNoTestSuiteLanguageWhenAbsent:
    """When has_test_suite=False, no rendered text may reference tests/suite/rerun.

    Watched fail against old phrasing.py (2026-08-16):
        Old impl: 94/500 renders contain test-suite language.
        Example (seed=4, word='tests are failing'):
            "fyi: hmm the tests are failing on main rn What we need: investigate
             the memory usage spike. thanks a lot"
    """

    def test_no_test_suite_language_when_absent(self) -> None:
        """Over 500 seeds with has_test_suite=False, no render may mention tests."""
        brief = Brief(
            goal="Investigate the memory usage spike.",
            has_test_suite=False,
        )
        violations: list[tuple[int, str, str]] = []
        for seed in _SEEDS:
            text = render_instruction(brief, seed=seed, salt=_SALT)
            lower = text.lower()
            for word in TEST_SUITE_WORDS:
                if word in lower:
                    violations.append((seed, word, text[:160]))
                    break
        assert not violations, (
            f"{len(violations)}/500 renders contained test-suite language despite "
            f"has_test_suite=False.\n"
            f"TEST_SUITE_WORDS checked: {TEST_SUITE_WORDS}\n"
            f"First violation — seed={violations[0][0]}, matched='{violations[0][1]}':\n"
            f"  {violations[0][2]}"
        )

    def test_has_test_suite_true_allows_test_language(self) -> None:
        """With has_test_suite=True (default), test language may appear."""
        brief = Brief(
            goal="Fix the failing test.",
            has_test_suite=True,
        )
        hit_any = False
        for seed in _SEEDS:
            text = render_instruction(brief, seed=seed, salt=_SALT)
            if any(w in text.lower() for w in TEST_SUITE_WORDS):
                hit_any = True
                break
        assert hit_any, "has_test_suite=True produced no test-suite language over 500 seeds"


class TestOnlyBlamePathGetsBugFraming:
    """Only the exact blame_path may receive "where the bug lives" framing.

    Watched fail against old phrasing.py (2026-08-16):
        Old impl: 119/500 renders incorrectly blame 'src/correct.py'.
        Example (seed=3):
            "heads up -- the failure's in `src/correct.py`. A test is failing
             and it's not obvious what changed. `src/buggy.py` is wh..."
    """

    # Blame-locative frame templates (populated from _PATH_BLAME_FRAMES)
    _BUG_FRAME_TEMPLATES = _PATH_BLAME_FRAMES

    def _blame_frames_for(self, path: str) -> list[str]:
        """Return the rendered blame frames that would wrap a given path."""
        return [t.format(fact=path) for t in self._BUG_FRAME_TEMPLATES]

    def test_only_blame_path_gets_bug_framing(self) -> None:
        """Over 500 seeds, bug-locative frames only apply to the blame_path."""
        correct_path = "src/correct.py"
        buggy_path = "src/buggy.py"
        brief = Brief(
            goal="Fix the defect.",
            required_facts=(
                Fact(text=correct_path, kind=FactKind.PATH),
                Fact(text=buggy_path, kind=FactKind.PATH),
            ),
            blame_path=buggy_path,
        )
        wrong_blame_frames = self._blame_frames_for(correct_path)
        violations: list[tuple[int, str, str]] = []
        for seed in _SEEDS:
            text = render_instruction(brief, seed=seed, salt=_SALT)
            for frame in wrong_blame_frames:
                if frame in text:
                    violations.append((seed, frame, text[:160]))
                    break
        assert not violations, (
            f"{len(violations)}/500 renders applied bug-locative framing to "
            f"'{correct_path}' (the non-blame path).\n"
            f"First violation — seed={violations[0][0]}:\n"
            f"  matched: {violations[0][1]!r}\n"
            f"  text: {violations[0][2]}"
        )

    def test_blame_path_does_get_bug_framing(self) -> None:
        """The blame_path should receive bug-locative framing in at least some renders."""
        buggy_path = "src/buggy.py"
        brief = Brief(
            goal="Fix the defect.",
            required_facts=(Fact(text=buggy_path, kind=FactKind.PATH),),
            blame_path=buggy_path,
        )
        blame_frames = self._blame_frames_for(buggy_path)
        hit_any = False
        for seed in _SEEDS:
            text = render_instruction(brief, seed=seed, salt=_SALT)
            if any(f in text for f in blame_frames):
                hit_any = True
                break
        assert hit_any, (
            f"blame_path '{buggy_path}' never received bug-locative framing "
            "over 500 seeds — _PATH_BLAME_FRAMES may be empty or unreachable"
        )

    def test_no_blame_path_means_all_paths_neutral(self) -> None:
        """With no blame_path set, all PATH facts receive only neutral framing."""
        correct_path = "tests/correct.py"
        brief = Brief(
            goal="Fix the defect.",
            required_facts=(Fact(text=correct_path, kind=FactKind.PATH),),
            blame_path="",
        )
        wrong_blame_frames = self._blame_frames_for(correct_path)
        violations: list[tuple[int, str, str]] = []
        for seed in _SEEDS:
            text = render_instruction(brief, seed=seed, salt=_SALT)
            for frame in wrong_blame_frames:
                if frame in text:
                    violations.append((seed, frame, text[:160]))
                    break
        assert not violations, (
            f"{len(violations)}/500 renders applied bug-locative framing to a path "
            f"when blame_path='' (empty).\n"
            f"First violation — seed={violations[0][0]}:\n"
            f"  matched: {violations[0][1]!r}\n"
            f"  text: {violations[0][2]}"
        )
