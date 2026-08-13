#!/bin/bash
# Persistent monitor for agentenv idle-tuning runs.
#
# Survives individual jobs: give it a job id and run dir and it polls until that
# job reaches a terminal state, then appends ONE line of verdict to a durable
# ledger and prints the decision. Designed to be re-invoked for each run so the
# ledger accumulates across replications -- an N=1 promotion is not a result,
# and the ledger is what turns a sequence of runs into one.
#
# Usage: agentenv_monitor.sh <jobid> <run_dir> [ledger]
set -uo pipefail

JOB="${1:?usage: agentenv_monitor.sh <jobid> <run_dir> [ledger]}"
RUN="${2:?usage: agentenv_monitor.sh <jobid> <run_dir> [ledger]}"
LEDGER="${3:-/data/ryan.kim/speedlm-runs/agentenv-ledger.jsonl}"
PY=/admin/home/ryan.kim/speedlm-fr/.venv/bin/python

note() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

record() {
  # One JSON line per terminal outcome. Written with python so a partial write
  # cannot corrupt the ledger into unparseable JSON.
  "$PY" - "$LEDGER" "$JOB" "$RUN" "$1" "${2:-}" <<'PYEOF'
import json, sys, os, time, glob
ledger, job, run, outcome, detail = sys.argv[1:6]
entry = {"job": job, "run_dir": run, "outcome": outcome, "detail": detail[:2000],
         "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
# DECISION_GLOB is the single definition of where a decision artifact lives;
# the shell helper below globs the identical shape so the poller and the ledger
# can never disagree about whether one exists.
DECISION_GLOB = os.path.join(run, "speedlm_home", "runs", "*", "decision.json")
unreadable = []
for path in sorted(glob.glob(DECISION_GLOB)):
    try:
        d = json.load(open(path))
    except Exception as exc:
        # Never swallow this: an unparseable decision.json is a different fact
        # from no decision at all, and reporting it as absence hides a bug.
        unreadable.append("%s: %s: %s" % (path, type(exc).__name__, exc))
        continue
    # The OUTCOME first. "verdict" is only what the thresholds said: a promotion
    # the gate vetoed for a non-stationary throughput delta is written as
    # verdict=promote while the cycle rolls back, so a ledger keyed off it
    # records promotions that never happened. Derived here for records written
    # before the gate persisted "final_verdict".
    stationarity = d.get("throughput_stationarity")
    vetoed = d.get("vetoed")
    if vetoed is None:
        vetoed = bool(
            d.get("verdict") == "promote"
            and isinstance(stationarity, dict)
            and stationarity.get("required_for_promotion")
            and stationarity.get("status") == "non_stationary"
        )
    entry["decision"] = {
        "final_verdict": d.get("final_verdict") or ("reject" if vetoed else d.get("verdict")),
        "final_reason": (
            d.get("final_reason")
            or ("throughput_not_stationary" if vetoed else d.get("reason"))
        ),
        "vetoed": bool(vetoed),
    }
    entry["decision"].update({k: d.get(k) for k in (
        "verdict", "reason", "accepted_length_delta", "acceptance_delta_pp",
        "throughput_delta_pct", "stock_avg_accepted_length",
        "stock_avg_tok_per_sec", "candidate_avg_tok_per_sec",
        "stock_truncation_regime", "candidate_truncation_regime",
        "acceptance_dispersion", "num_contexts", "num_repeats",
        "divergence_rate", "control_divergence_rate",
        "divergence_control_comparable", "divergence_sampling")})
    break

if "decision" not in entry:
    # Say WHY there is no decision. "no decision artifact" is compatible with a
    # job that never started, a cycle that never reached a verdict, and a lookup
    # pointed at the wrong path -- three failures that need different responses.
    home = os.path.join(run, "speedlm_home")
    runs_dir = os.path.join(home, "runs")
    if unreadable:
        why = "decision.json exists but could not be parsed -- " + "; ".join(unreadable)
    elif not os.path.isdir(home):
        why = "no %s: the job never got as far as starting the server" % home
    elif not os.path.isdir(runs_dir):
        why = "no %s: the idle tuner never opened a cycle run directory" % runs_dir
    else:
        siblings = sorted(os.listdir(runs_dir))
        if not siblings:
            why = "%s is empty: the idle tuner never opened a cycle run directory" % runs_dir
        else:
            why = ("%d cycle run dir(s) under %s (%s) but none holds decision.json: "
                   "a cycle started and did not reach a verdict"
                   % (len(siblings), runs_dir, ", ".join(siblings[:5])))
    entry["no_decision_reason"] = why

summary = os.path.join(run, "traffic", "summary.json")
if os.path.isfile(summary):
    try:
        entry["traffic"] = json.load(open(summary))["totals"]
    except Exception:
        pass
os.makedirs(os.path.dirname(ledger), exist_ok=True)
with open(ledger, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(entry, sort_keys=True) + "\n")
print(json.dumps(
    entry.get("decision", {"no_decision_reason": entry.get("no_decision_reason", "unknown")}),
    indent=2, sort_keys=True))
PYEOF
}

# Exactly the path shape record()'s DECISION_GLOB uses. Keep the two in step:
# a poller that looks somewhere the ledger does not is how a run gets reported
# as "no decision" while its decision.json sits on disk.
decision_present() {
  compgen -G "$RUN/speedlm_home/runs/*/decision.json" >/dev/null 2>&1
}

note "monitoring job $JOB -> $RUN (ledger: $LEDGER)"

# 1200 polls at 40s = 13.3h of monitoring against the job's --time=12:00:00,
# i.e. ~1.3h of slack to cover queue wait. There is no separate grace period --
# the queue wait comes out of the same budget as the run itself, so a job that
# sits pending for more than ~1.3h can outlive this monitor.
for _ in $(seq 1 1200); do
  state=$(squeue -j "$JOB" -h -o "%T" 2>/dev/null)

  if [ -n "$state" ]; then
    # Still queued or running. Report the decisive in-flight signals as they
    # appear, but do NOT exit on them -- the cycle retries after a cooldown and
    # an early failure is not necessarily terminal.
    if [ -f "$RUN/gateway-and-vllm.log" ] && \
       rg -q "idle tuning cycle failed" "$RUN/gateway-and-vllm.log" 2>/dev/null; then
      note "cycle failure seen (may retry): $(rg -m1 -o 'idle tuning cycle failed.{0,180}' "$RUN/gateway-and-vllm.log" 2>/dev/null)"
    fi
    if decision_present; then
      note "DECISION ARTIFACT APPEARED"
      record "decision" "job still running when decision landed"
      exit 0
    fi
    sleep 40
    continue
  fi

  # Absent from squeue. That is NOT yet proof of termination: squeue can fail
  # transiently, and sacct lags the queue -- it reports RUNNING, COMPLETING or
  # nothing at all for a while after squeue stops listing the job. Treating the
  # first sample as final records a wrong (or empty) outcome permanently.
  # So: require the SAME terminal state from sacct on two consecutive samples,
  # and abandon the whole conclusion if the job turns up in the queue again.
  final=""
  stable=0
  requeued=0
  previous=""
  for _ in $(seq 1 18); do
    if [ -n "$(squeue -j "$JOB" -h -o '%T' 2>/dev/null)" ]; then
      note "job $JOB reappeared in the queue; not terminal after all"
      requeued=1
      break
    fi
    sample=$(sacct -j "$JOB" --format=State -n -X 2>/dev/null | head -1 | tr -d ' ')
    case "$sample" in
      COMPLETED*|FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*|PREEMPTED*|BOOT_FAIL*|DEADLINE*|REVOKED*)
        if [ "$sample" = "$previous" ]; then
          final="$sample"
          stable=1
          break
        fi
        previous="$sample"
        ;;
      *)
        # RUNNING, COMPLETING, REQUEUED, or empty: accounting has not settled.
        previous=""
        ;;
    esac
    sleep 10
  done

  if [ "$requeued" = 1 ]; then
    sleep 40
    continue
  fi

  if [ "$stable" = 1 ]; then
    note "job $JOB terminal: $final"
  else
    final="${previous:-unknown}"
    note "job $JOB left the queue but sacct never settled; last sample: $final"
  fi

  if decision_present; then
    record "decision" "$final"
    exit 0
  fi
  detail=$(rg -m1 -o 'idle tuning cycle failed.{0,300}' "$RUN/gateway-and-vllm.log" 2>/dev/null)
  if [ "$stable" != 1 ]; then
    final="$final (unconfirmed: sacct gave no stable terminal state in 180s)"
  fi
  record "no_decision" "${final}: ${detail:-no cycle failure logged}"
  exit 1
done

note "monitor timed out"
record "monitor_timeout" "no terminal state within the monitor window"
exit 2
