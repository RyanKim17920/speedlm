"""The real gate, measuring a real engine over real HTTP, for every verdict.

Nothing here fakes a :class:`~speedlm.gate.decide.Decision`.  The suite is
frozen from real prompts by :func:`~speedlm.gate.suite.build_suite`, replayed
by the shipped :class:`~speedlm.gate.runner.HttpReplayExecutor` against a
simulated engine, scraped through :func:`~speedlm.gate.metrics.parse_metrics`,
and judged by :func:`~speedlm.gate.decide.decide_promotion`.  The only thing
dialled in is the engine's behaviour -- which is the thing under measurement.

The distinction this file exists to defend: a gate that *measured and declined*
and a gate that *never measured* must not report the same way.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from simulation.corpus import simulation_traces
from simulation.engine import (
    DraftProfile,
    EngineFaults,
    SimulatedEngine,
    running_engine,
)
from simulation.harness import (
    EngineEndpoint,
    EngineMetrics,
    MutableClock,
    RecordSource,
    real_gate,
    simulation_config,
)
from speedlm.config import PromotionConfig
from speedlm.gate.decide import DispersionBasis, Reason, Verdict
from speedlm.gate.runner import BenchmarkGateRunner
from speedlm.gate.suite import SuiteError
from speedlm.tuner.orchestrator import GateFailure

STOCK_REFERENCE = "sim/stock-draft"

#: Held-out contexts per benchmark.  Five is enough for the split, the hashing
#: and the per-repeat statistics to be real, and small enough that ten suite
#: passes across two arms stay well under a second.
SUITE_SIZE = 5

#: Thresholds for tests whose subject is *not* throughput.
#:
#: The gating throughput statistic is client-measured wall clock, so it carries
#: real scheduling jitter -- on a loaded machine a few milliseconds of delay
#: swamps the few milliseconds of injected latency that separate the two arms,
#: and a test about output divergence starts failing on THROUGHPUT_BELOW_
#: THRESHOLD.  Widening the regression guard removes that variable without
#: weakening anything under test: acceptance, the criterion these tests are
#: actually about, is counter arithmetic and is exact.  The one test whose
#: subject *is* the throughput guard uses the shipped defaults.
ACCEPTANCE_ONLY = PromotionConfig(
    min_acceptance_delta_pp=1.0,
    min_throughput_delta_pct=-95.0,
)


@dataclass(frozen=True, slots=True)
class GateFixture:
    """Everything needed to run one real benchmark against one engine."""

    root: Path
    engine: SimulatedEngine
    candidate: Path

    def gate(
        self,
        *,
        clock: MutableClock | None = None,
        promotion: PromotionConfig | None = None,
    ) -> BenchmarkGateRunner:
        return real_gate(
            self.engine,
            simulation_traces(SUITE_SIZE),
            self.root / "suite",
            stock_draft=STOCK_REFERENCE,
            config=simulation_config(promotion=promotion or ACCEPTANCE_ONLY),
            clock=clock,
        )


@contextmanager
def _fixture(
    tmp_path: Path,
    stock: DraftProfile,
    candidate: DraftProfile,
    *,
    faults: EngineFaults | None = None,
) -> Iterator[GateFixture]:
    candidate_dir = tmp_path / "candidate-draft"
    candidate_dir.mkdir()
    with running_engine(default_profile=stock, faults=faults) as engine:
        engine.register(STOCK_REFERENCE, stock)
        engine.register(str(candidate_dir), candidate)
        yield GateFixture(root=tmp_path, engine=engine, candidate=candidate_dir)


def _profiles(
    *,
    stock_acceptance: float = 0.40,
    candidate_acceptance: float = 0.55,
    stock_seconds: float = 0.006,
    candidate_seconds: float = 0.002,
    candidate_divergence_at: int | None = None,
    candidate_invalid_every: int | None = None,
) -> tuple[DraftProfile, DraftProfile]:
    return (
        DraftProfile(
            name="stock",
            acceptance_rate=stock_acceptance,
            seconds_per_request=stock_seconds,
        ),
        DraftProfile(
            name="candidate",
            acceptance_rate=candidate_acceptance,
            seconds_per_request=candidate_seconds,
            divergence_at_token=candidate_divergence_at,
            invalid_every=candidate_invalid_every,
        ),
    )


class TestPromotion:
    def test_a_better_head_is_promoted_on_measured_evidence(self, tmp_path: Path) -> None:
        stock, candidate = _profiles(stock_acceptance=0.40, candidate_acceptance=0.55)
        with _fixture(tmp_path, stock, candidate) as fixture:
            result = fixture.gate().benchmark(
                fixture.candidate, timeout_seconds=600.0, should_abort=lambda: False
            )

            assert result.passed is True
            assert result.failure is None
            decision = result.decision
            assert decision is not None
            assert decision.verdict is Verdict.PROMOTE
            assert decision.reason is Reason.BOTH_THRESHOLDS_MET
            # The acceptance the gate reports is the acceptance that was dialled
            # in, recovered end to end through the Prometheus counters.
            assert decision.stock_avg_acceptance == pytest.approx(0.40, abs=1e-6)
            assert decision.candidate_avg_acceptance == pytest.approx(0.55, abs=1e-6)
            assert decision.acceptance_delta_pp == pytest.approx(15.0, abs=1e-4)
            assert decision.throughput_delta_pct is not None
            # Asserted on the Prometheus decode-time figure rather than the
            # client-timed replay one: the former is derived from the
            # decode-seconds counter the engine advances by exactly the injected
            # latency, so it is deterministic, whereas the replay statistic
            # carries the host's scheduling jitter.
            assert decision.prometheus_throughput_delta_pct is not None
            assert decision.prometheus_throughput_delta_pct > 0
            # Three scored repeats, and the published array really has three
            # independent acceptance samples rather than one stamped thrice.
            assert decision.num_repeats == 3
            assert len(decision.per_repeat) == 3
            assert decision.acceptance_statistic == "per_repeat_mean"
            assert decision.throughput_statistic == "replay_per_repeat_mean"

    def test_a_deterministic_acceptance_replay_is_labelled_degenerate(
        self, tmp_path: Path
    ) -> None:
        """Repeats do not sample acceptance, and the record has to admit it.

        The simulated engine advances its ``spec_decode`` counters by the same
        amount on every suite pass, which is exactly what the real engine does
        under greedy replay of a frozen suite -- jobs 369161/369162 produced
        bit-identical counter deltas across five repeats.  The published
        standard deviation is therefore 0.0, and the danger is that a consumer
        reads ``min_acceptance_delta_pp / standard_error`` as infinite headroom.
        The decision must label the reading ``degenerate`` and publish a null
        standard error rather than a zero one.
        """
        stock, candidate = _profiles()
        with _fixture(tmp_path, stock, candidate) as fixture:
            result = fixture.gate().benchmark(
                fixture.candidate, timeout_seconds=600.0, should_abort=lambda: False
            )

            decision = result.decision
            assert decision is not None
            assert decision.num_repeats >= 2
            assert len({r.stock_acceptance_rate for r in decision.per_repeat}) == 1
            assert decision.stock_acceptance_stdev == 0.0
            assert decision.candidate_acceptance_stdev == 0.0
            assert decision.acceptance_dispersion is DispersionBasis.DEGENERATE
            assert decision.acceptance_delta_standard_error_pp is None

            record = decision.to_dict()
            assert record["acceptance_dispersion"] == "degenerate"
            assert record["acceptance_delta_standard_error_pp"] is None
            # Throughput is the quantity the repeats are actually there for, so
            # it must not be labelled the same way: real wall-clock timing over
            # more than one repeat always disperses.
            assert decision.throughput_dispersion is DispersionBasis.MEASURED
            assert decision.throughput_delta_standard_error_pct is not None
            assert decision.candidate_throughput_trend_pct_per_repeat is not None
            # The warming diagnostic has to reach ``decision.json`` on every
            # run, not only when it happens to find a plateau -- accumulating
            # the answer across runs is the entire mechanism.  Its *value* is
            # not asserted: three repeats of a simulated engine is not a
            # warming curve, and pinning one here would be pinning noise.
            assert "candidate_throughput_flat_from_repeat" in record
            assert "stock_throughput_flat_from_repeat" in record

    def test_every_scrape_is_kept_verbatim_as_evidence(self, tmp_path: Path) -> None:
        stock, candidate = _profiles()
        with _fixture(tmp_path, stock, candidate) as fixture:
            result = fixture.gate().benchmark(
                fixture.candidate, timeout_seconds=600.0, should_abort=lambda: False
            )

            # repeats + 1 scrapes per arm, both arms: the pooled window and
            # every per-repeat window are reconcilable against raw counters.
            assert sorted(result.metrics_bodies) == [
                "candidate-after",
                "candidate-after-repeat-0",
                "candidate-after-repeat-1",
                "candidate-before",
                "stock-after",
                "stock-after-repeat-0",
                "stock-after-repeat-1",
                "stock-before",
            ]
            for body in result.metrics_bodies.values():
                assert "vllm:spec_decode_num_accepted_tokens_total" in body

    def test_a_divergence_deep_into_the_answer_does_not_block_promotion(
        self, tmp_path: Path
    ) -> None:
        # Position, not equality.  Parting at token 40 -- past
        # ``min_divergence_token_index`` (16) -- is measurement noise on
        # non-reproducible hardware, not a broken head.
        stock, candidate = _profiles(candidate_divergence_at=40)
        with _fixture(tmp_path, stock, candidate) as fixture:
            result = fixture.gate().benchmark(
                fixture.candidate, timeout_seconds=600.0, should_abort=lambda: False
            )

            decision = result.decision
            assert decision is not None
            assert decision.output_divergences, "the divergence must still be recorded"
            assert all(not d.early for d in decision.output_divergences)
            assert all(d.first_divergence_index == 40 for d in decision.output_divergences)
            assert all(d.basis == "token" for d in decision.output_divergences)
            assert decision.verdict is Verdict.PROMOTE


class TestMeasuredRejections:
    """Each of these is a *scientific* result: two arms were compared."""

    def test_acceptance_below_threshold(self, tmp_path: Path) -> None:
        stock, candidate = _profiles(stock_acceptance=0.40, candidate_acceptance=0.35)
        with _fixture(tmp_path, stock, candidate) as fixture:
            result = fixture.gate().benchmark(
                fixture.candidate, timeout_seconds=600.0, should_abort=lambda: False
            )

            assert result.passed is False
            assert result.failure is None, "a measured rejection is not a gate failure"
            decision = result.decision
            assert decision is not None
            assert decision.reason is Reason.ACCEPTANCE_BELOW_THRESHOLD
            assert decision.acceptance_delta_pp == pytest.approx(-5.0, abs=1e-4)
            # The throughput delta is still reported: a rejection on one
            # criterion must not blank the other.
            assert decision.throughput_delta_pct is not None

    def test_throughput_below_threshold_despite_better_acceptance(
        self, tmp_path: Path
    ) -> None:
        # Acceptance clears its bar, so the only thing that can reject is the
        # throughput regression guard.
        stock, candidate = _profiles(
            stock_acceptance=0.40,
            candidate_acceptance=0.60,
            stock_seconds=0.001,
            candidate_seconds=0.020,
        )
        with _fixture(tmp_path, stock, candidate) as fixture:
            result = fixture.gate(promotion=PromotionConfig()).benchmark(
                fixture.candidate, timeout_seconds=600.0, should_abort=lambda: False
            )

            decision = result.decision
            assert decision is not None
            assert decision.reason is Reason.THROUGHPUT_BELOW_THRESHOLD
            assert decision.acceptance_delta_pp == pytest.approx(20.0, abs=1e-4)
            assert decision.throughput_delta_pct is not None
            assert decision.throughput_delta_pct < decision.min_throughput_delta_pct

    def test_early_output_divergence(self, tmp_path: Path) -> None:
        stock, candidate = _profiles(candidate_divergence_at=2)
        with _fixture(tmp_path, stock, candidate) as fixture:
            result = fixture.gate().benchmark(
                fixture.candidate, timeout_seconds=600.0, should_abort=lambda: False
            )

            decision = result.decision
            assert decision is not None
            assert decision.reason is Reason.OUTPUT_MISMATCH
            assert decision.output_early_divergences == SUITE_SIZE
            assert all(d.early for d in decision.output_divergences)
            assert all(d.first_divergence_index == 2 for d in decision.output_divergences)

    def test_raising_the_threshold_reclassifies_a_real_divergence_as_early(
        self, tmp_path: Path
    ) -> None:
        """The rehearsal for the GPU experiment that has never run.

        Every divergence recorded on live hardware has classified LATE: job
        369161 measured first-divergence offsets of 36/67/78/91/127 and job
        369162 measured 58 of them spanning 20..125, against a threshold of 16.
        So the EARLY branch -- the one that actually rejects -- has only ever
        executed against synthetic unit-test data.

        The cheapest decisive experiment is not a deliberately broken drafter;
        it is the *same* benign divergences judged against a threshold above
        them.  That is what this pins in simulation, over the real gate and a
        real replay, and it is exactly the shape of the sbatch that runs it:
        one config key, no code path that exists only for the experiment.

        What it establishes: the classification boundary and the reject path
        work on divergences the engine genuinely produced.  What it does not:
        that a genuinely broken drafter parts early.  Those are different
        claims and only the first one is testable this cheaply.
        """
        # The default (16) promotes this exact candidate -- see
        # ``test_a_divergence_deep_into_the_answer_does_not_block_promotion``.
        stock, candidate = _profiles(candidate_divergence_at=40)
        promotion = replace(ACCEPTANCE_ONLY, min_divergence_token_index=128)
        with _fixture(tmp_path, stock, candidate) as fixture:
            result = fixture.gate(promotion=promotion).benchmark(
                fixture.candidate, timeout_seconds=600.0, should_abort=lambda: False
            )

            decision = result.decision
            assert decision is not None
            assert result.passed is False
            assert result.failure is None, "a measured rejection is not a gate failure"
            assert decision.verdict is Verdict.REJECT
            assert decision.reason is Reason.OUTPUT_MISMATCH
            # The divergences are the same ones; only the criterion moved.
            assert all(d.first_divergence_index == 40 for d in decision.output_divergences)
            assert all(d.early for d in decision.output_divergences)
            assert decision.output_early_divergences == SUITE_SIZE
            assert decision.output_total_divergences == SUITE_SIZE
            # The archived record has to say which criterion produced this, or
            # a reader cannot tell the experiment from a production rejection.
            assert decision.min_divergence_token_index == 128
            assert decision.to_dict()["min_divergence_token_index"] == 128

    def test_an_early_divergence_rejection_carries_no_deltas_but_is_measured(
        self, tmp_path: Path
    ) -> None:
        """The exact record shape job 369373 produced, pinned without a GPU.

        That job rejected 48 early divergences and reported
        ``acceptance_delta_pp: null`` -- correctly, because
        :func:`~speedlm.gate.decide.decide_promotion` returns at the divergence
        check, which sits *above* the line that computes the deltas.  The e2e
        harness nevertheless asserted the delta was non-null for every reason
        it called "measured", ``output_mismatch`` among them, so the assertion
        could never hold and the job reported FAILED on a pipeline that worked.

        Two claims are pinned here.  First, the gate's own contract: a
        short-circuited rejection publishes no deltas and instead publishes the
        divergence evidence that caused it.  Second, the harness contract: the
        shipped e2e assertion accepts this record while still refusing one from
        a gate that measured nothing.
        """
        from e2e.test_live_idle_tuning import (  # noqa: PLC0415
            DELTA_REASONS,
            MEASURED_REASONS,
            SHORT_CIRCUIT_MEASURED_REASONS,
            _assert_gate_measured_something,
        )

        stock, candidate = _profiles(candidate_divergence_at=2)
        with _fixture(tmp_path, stock, candidate) as fixture:
            result = fixture.gate().benchmark(
                fixture.candidate, timeout_seconds=600.0, should_abort=lambda: False
            )

            decision = result.decision
            assert decision is not None
            assert decision.verdict is Verdict.REJECT
            assert decision.reason is Reason.OUTPUT_MISMATCH
            # Null by design: the short circuit is above the delta computation.
            assert decision.acceptance_delta_pp is None
            assert decision.throughput_delta_pct is None
            # But the arms did run, and the evidence is on the record.
            assert decision.num_repeats > 0
            assert decision.output_early_divergences > 0

            record = decision.to_dict()
            # The correctness pass is its own replay: it made ONE pass, so only
            # ``per_repeat`` row 0 can carry a non-zero ``output_mismatches``
            # and the later rows are unmeasured, not clean.  Without this field
            # a reader cannot tell those two apart.
            assert record["correctness_repeats"] == 1
            assert record["per_repeat"][0]["output_mismatches"] > 0
            assert all(
                row["output_mismatches"] == 0 for row in record["per_repeat"][1:]
            )

            # The harness contract, checked against the shipped assertion.
            assert decision.reason.value in SHORT_CIRCUIT_MEASURED_REASONS
            assert decision.reason.value not in DELTA_REASONS
            assert decision.reason.value in MEASURED_REASONS
            _assert_gate_measured_something(record)

    def test_the_harness_still_rejects_a_gate_that_measured_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The coverage the ``output_mismatch`` split must not have cost.

        Dropping ``output_mismatch`` from the measured set would have made the
        e2e assertion pass, at the price of no longer distinguishing a gate
        that measured and declined from one that collected zero samples.  This
        pins that the zero-sample record still fails, and that a short-circuit
        reason with no divergence evidence behind it fails too -- the loophole
        the split would otherwise open.
        """
        from e2e import test_live_idle_tuning as harness  # noqa: PLC0415

        _assert_gate_measured_something = harness._assert_gate_measured_something
        # The escape hatch is an operator override, not part of the contract
        # under test; pin it off so this passes regardless of the environment.
        monkeypatch.setattr(harness, "ALLOW_UNMEASURED_GATE", False)

        # The record that actually shipped: rejected having measured nothing.
        with pytest.raises(AssertionError, match="without comparing the two arms"):
            _assert_gate_measured_something(
                {
                    "reason": "acceptance_unavailable",
                    "num_repeats": 0,
                    "per_repeat": [],
                }
            )

        # An ``output_mismatch`` claimed with no divergence behind it.
        with pytest.raises(AssertionError):
            _assert_gate_measured_something(
                {
                    "reason": "output_mismatch",
                    "num_repeats": 1,
                    "per_repeat": [{"output_mismatches": 0}],
                    "output_early_divergences": 0,
                    "output_divergences": [],
                    "correctness_repeats": 1,
                }
            )

    def test_counter_reset_under_the_gate(self, tmp_path: Path) -> None:
        # An engine that restarts mid-arm.  Per arm the schedule is: 5 warmup
        # generations, then three scored repeats of 5.  A reset on generation 12
        # lands inside the third repeat's window, so that window runs backwards
        # while the pooled endpoints alone might not.
        stock, candidate = _profiles()
        faults = EngineFaults(reset_counters_after_requests=12)
        with _fixture(tmp_path, stock, candidate, faults=faults) as fixture:
            result = fixture.gate().benchmark(
                fixture.candidate, timeout_seconds=600.0, should_abort=lambda: False
            )

            assert result.passed is False
            # Still a completed benchmark, so it carries a real decision: the
            # gate measured, discovered the measurement was invalid, and said so.
            assert result.failure is None
            decision = result.decision
            assert decision is not None
            assert decision.reason is Reason.COUNTER_RESET
            assert "counter-reset" in fixture.engine.journal.events
            # A reset window publishes no throughput or acceptance: an
            # unmeasurable run must not look measured.
            assert decision.acceptance_delta_pp is None
            assert decision.throughput_delta_pct is None

    def test_acceptance_unavailable_on_a_non_speculative_engine(
        self, tmp_path: Path
    ) -> None:
        stock, candidate = _profiles()
        faults = EngineFaults(omit_spec_counters=True)
        with _fixture(tmp_path, stock, candidate, faults=faults) as fixture:
            result = fixture.gate().benchmark(
                fixture.candidate, timeout_seconds=600.0, should_abort=lambda: False
            )

            decision = result.decision
            assert decision is not None
            # Not a measured 0% acceptance -- an absent measurement.
            assert decision.reason is Reason.ACCEPTANCE_UNAVAILABLE
            assert decision.stock_avg_acceptance == 0.0

    def test_high_invalid_rate_from_a_flaky_engine(self, tmp_path: Path) -> None:
        stock, candidate = _profiles(candidate_invalid_every=2)
        with _fixture(tmp_path, stock, candidate) as fixture:
            result = fixture.gate().benchmark(
                fixture.candidate, timeout_seconds=600.0, should_abort=lambda: False
            )

            decision = result.decision
            assert decision is not None
            # Checked before output mismatch: half the responses being HTTP 500
            # is an infrastructure story, and reporting it as a divergence would
            # blame the head for the engine.
            assert decision.reason is Reason.HIGH_INVALID_RATE
            assert max(r.invalid_rate for r in decision.per_repeat) > 0.1

    def test_a_reasoning_model_truncated_at_the_cap_is_not_a_flaky_engine(
        self, tmp_path: Path
    ) -> None:
        """SLURM 369147: ``high_invalid_rate 0.7379`` on a healthy engine.

        Both arms are thinking models whose generations run past
        ``benchmark_max_tokens``, so every held-out response comes back with
        ``content: null`` and ``finish_reason: "length"``.  Under the old
        predicate that was a 100% invalid rate and an automatic rejection.  The
        engine generated the full 512 tokens for each one and its acceptance
        counters moved exactly as dialled, so the gate must measure and judge on
        the merits -- here, promote.
        """
        stock, candidate = _profiles()
        stock = replace(stock, completion_tokens=4096, reasoning_model=True)
        candidate = replace(candidate, completion_tokens=4096, reasoning_model=True)
        with _fixture(tmp_path, stock, candidate) as fixture:
            result = fixture.gate().benchmark(
                fixture.candidate, timeout_seconds=600.0, should_abort=lambda: False
            )

            decision = result.decision
            assert decision is not None
            assert all(r.invalid_rate == 0.0 for r in decision.per_repeat)
            assert decision.reason is not Reason.HIGH_INVALID_RATE
            assert decision.verdict is Verdict.PROMOTE
            assert decision.acceptance_delta_pp == pytest.approx(15.0, abs=1e-4)


class TestGateFailuresThatNeverMeasured:
    """Each of these produced *no* decision, and must not read as a rejection."""

    def test_a_timeout_is_not_a_measured_rejection(self, tmp_path: Path) -> None:
        stock, candidate = _profiles()
        with _fixture(tmp_path, stock, candidate) as fixture:
            # The runner only consults its clock at stage boundaries, so a clock
            # that ticks on every read is what makes a deadline expire without
            # anything actually being slow.
            clock = MutableClock(advance_per_call=0.05)
            result = fixture.gate(clock=clock).benchmark(
                fixture.candidate, timeout_seconds=1.0, should_abort=lambda: False
            )

            assert result.passed is False
            assert result.failure is GateFailure.TIMED_OUT
            # The whole point: no Decision, so nothing downstream can mistake
            # this for "the candidate was measured and lost".
            assert result.decision is None
            assert result.metrics == {"timed_out": True}
            assert "timed out" in result.reason

    def test_an_abort_is_not_a_measured_rejection(self, tmp_path: Path) -> None:
        stock, candidate = _profiles()
        with _fixture(tmp_path, stock, candidate) as fixture:
            # Serving activity arrives once the benchmark is under way.
            calls = {"n": 0}

            def should_abort() -> bool:
                calls["n"] += 1
                return calls["n"] > 6

            result = fixture.gate().benchmark(
                fixture.candidate, timeout_seconds=600.0, should_abort=should_abort
            )

            assert result.passed is False
            assert result.failure is GateFailure.ABORTED
            assert result.decision is None
            assert result.metrics == {"aborted": True}
            assert "aborted" in result.reason

    def test_partial_evidence_survives_a_gate_that_never_finished(
        self, tmp_path: Path
    ) -> None:
        stock, candidate = _profiles()
        with _fixture(tmp_path, stock, candidate) as fixture:
            # Abort once the gate has actually taken measurements, rather than
            # after a fixed number of polls: the abort check is consulted from
            # inside the replay loop too, so a poll count says nothing about
            # how far the benchmark got.
            def should_abort() -> bool:
                return fixture.engine.journal.scrapes >= 2

            result = fixture.gate().benchmark(
                fixture.candidate, timeout_seconds=600.0, should_abort=should_abort
            )

            assert result.failure is GateFailure.ABORTED
            # Scrapes taken before the abort are kept: a benchmark that died
            # partway still saw counters, and that partial evidence is worth
            # more than nothing when diagnosing why it died.
            assert result.metrics_bodies
            assert "stock-before" in result.metrics_bodies

    def test_an_engine_that_never_becomes_ready_fails_the_gate_loudly(
        self, tmp_path: Path
    ) -> None:
        stock, candidate = _profiles()
        faults = EngineFaults(never_ready=True)
        # Not an abort and not a timeout: an activation that cannot succeed is
        # an error, and swallowing it into a rejection would promote a broken
        # deployment into a scientific claim.
        with (
            _fixture(tmp_path, stock, candidate, faults=faults) as fixture,
            pytest.raises(RuntimeError, match="never became ready"),
        ):
            fixture.gate().benchmark(
                fixture.candidate, timeout_seconds=600.0, should_abort=lambda: False
            )


class TestSuiteIntegrity:
    def test_leakage_between_training_and_benchmark_contexts_is_refused(
        self, tmp_path: Path
    ) -> None:
        records = simulation_traces(SUITE_SIZE)
        stock, candidate = _profiles()
        candidate_dir = tmp_path / "candidate-draft"
        candidate_dir.mkdir()
        with running_engine(default_profile=stock) as engine:
            engine.register(STOCK_REFERENCE, stock)
            engine.register(str(candidate_dir), candidate)
            # Claim every benchmark context was also trained on.
            leaked = {
                __import__("hashlib")
                .sha256(
                    __import__("json")
                    .dumps(
                        [{"role": "user", "content": r.messages[0]["content"]}],
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    )
                    .encode("utf-8")
                )
                .hexdigest()
                for r in records
            }
            runner = BenchmarkGateRunner(
                config=simulation_config(),
                trace_source=RecordSource(records),
                suite_dir=tmp_path / "suite",
                stock_draft=STOCK_REFERENCE,
                endpoint=EngineEndpoint(engine),
                metrics_source=EngineMetrics(engine),
                held_out_fraction=1.0,
                training_context_hashes=leaked,
                clock=lambda: 0.0,
            )

            with pytest.raises(SuiteError, match="leakage"):
                runner.benchmark(
                    candidate_dir, timeout_seconds=600.0, should_abort=lambda: False
                )

    def test_a_gate_that_cannot_prove_the_suite_is_held_out_refuses_to_run(
        self, tmp_path: Path
    ) -> None:
        stock, candidate = _profiles()
        candidate_dir = tmp_path / "candidate-draft"
        candidate_dir.mkdir()
        with running_engine(default_profile=stock) as engine:
            runner = BenchmarkGateRunner(
                config=simulation_config(),
                trace_source=RecordSource(simulation_traces(SUITE_SIZE)),
                suite_dir=tmp_path / "suite",
                stock_draft=STOCK_REFERENCE,
                endpoint=EngineEndpoint(engine),
                metrics_source=EngineMetrics(engine),
                held_out_fraction=1.0,
                training_context_hashes=None,
                clock=lambda: 0.0,
            )

            with pytest.raises(SuiteError, match="Cannot prove"):
                runner.benchmark(
                    candidate_dir, timeout_seconds=600.0, should_abort=lambda: False
                )

    def test_the_frozen_suite_is_reused_across_benchmarks(self, tmp_path: Path) -> None:
        stock, candidate = _profiles()
        with _fixture(tmp_path, stock, candidate) as fixture:
            gate = fixture.gate()
            # ``estimated_benchmark_seconds`` freezes the suite so the pass that
            # follows measures the same contexts the deadline was sized against.
            estimate = gate.estimated_benchmark_seconds()
            assert estimate is not None and estimate > 0
            assert (fixture.root / "suite" / "suite_manifest.json").exists()

            first = gate.benchmark(
                fixture.candidate, timeout_seconds=600.0, should_abort=lambda: False
            )
            second = gate.benchmark(
                fixture.candidate, timeout_seconds=600.0, should_abort=lambda: False
            )
            assert first.metrics["suite_hash"] == second.metrics["suite_hash"]
            assert first.metrics["num_contexts"] == SUITE_SIZE


def test_the_suite_is_built_from_real_prompt_lengths(tmp_path: Path) -> None:
    # The point of using the real corpus: a suite of identical-length synthetic
    # strings would make the held-out split and the hashing degenerate.
    records = simulation_traces(12)
    lengths = {len(str(record.messages[0]["content"])) for record in records}
    assert len(lengths) > 1, "the corpus must supply varied prompt lengths"
    # Every record round-trips through the production trace contract, including
    # the provenance tag that marks the provider-authored turn.
    assert all(r.messages[-1].get("provenance_tag") == "generated" for r in records)
