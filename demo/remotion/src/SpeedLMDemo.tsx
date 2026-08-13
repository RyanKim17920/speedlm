import React from "react";
import {
  AbsoluteFill,
  OffthreadVideo,
  interpolate,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import data from "./data.json";
import { AccuracyChart } from "./charts/AccuracyChart";
import { LossChart } from "./charts/LossChart";
import { ThroughputChart } from "./charts/ThroughputChart";
import { BG, DIM, MONO, MUTED, PANEL, RULE, TEXT } from "./theme";

export const TERMINAL_VIDEO = "terminal.mp4";

/**
 * FALLBACK timing only, used when src/data.json carries no frame anchors —
 * i.e. extract_series.py could not find the recording and the renderer's timing
 * sidecar, so nothing knows when the terminal actually printed each number.
 * These are fractions of the composition, not measurements. When anchors are
 * present (the normal case) they are ignored entirely and every datum appears
 * on the frame where its log line appeared on the left.
 */
const REVEAL = {
  training: [0.06, 0.62] as const, // loss + per-position accuracy fill in together
  gate: [0.66, 0.9] as const, // gate replay bars grow after training completes
};

/**
 * How long the gate bars take to grow once the verdict is on screen. The
 * *onset* is the measured moment; only this easing is presentational, and it is
 * short enough that the bars are still arriving while the verdict block is being
 * printed in the terminal.
 */
const GATE_GROW_FRAMES = 24;

/**
 * How long a burst of co-anchored points is allowed to take to fill in.
 *
 * The fast cut tails an already-complete training log, so the terminal prints
 * all 218 loss / accuracy lines inside a couple of hundredths of a second and
 * extract_series.py honestly stamps every one of them with the same frame. Drawn
 * literally that is a whole chart popping into existence in one frame: truthful,
 * but unreadable.
 *
 * So the ONSET stays measured and the FILL-IN is eased. Nothing is ever drawn
 * before the frame on which the terminal printed it -- `easedSchedule` only ever
 * moves a point *later* than its anchor, never earlier -- and the easing is
 * capped so it can never run past the next point's own anchor. The stagger is
 * presentation; the "not before" is the measurement, and only the former is
 * invented here.
 */
const BURST_SPREAD_FRAMES = 60;

const frac = (frame: number, durationInFrames: number, [a, b]: readonly [number, number]) =>
  interpolate(frame / durationInFrames, [a, b], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

/**
 * The frame extract_series.py stamped on a datum: the moment its line hit the
 * recorded terminal, mapped through the renderer's compressed timeline.
 */
const frameOf = (point: unknown): number | undefined =>
  (point as { frame?: number }).frame;

/**
 * Spread runs of identically-anchored frames into distinct (fractional) reveal
 * times, so a burst draws in instead of popping.
 *
 * Given non-decreasing anchor frames, each maximal run sharing a frame F is laid
 * out evenly across [F, F + span), where span = min(BURST_SPREAD_FRAMES, G - F)
 * and G is the next distinct anchor. Two invariants make this honest rather than
 * decorative:
 *
 *   eased[i] >= frames[i]     -- a point is never revealed before the terminal
 *                                printed it; the measured onset is a floor.
 *   eased is non-decreasing   -- and strictly below the next run's anchor, so
 *                                easing one burst never spills into the next.
 */
const easedSchedule = (frames: readonly number[], maxSpread: number): number[] => {
  const eased = new Array<number>(frames.length);
  let i = 0;
  while (i < frames.length) {
    let end = i;
    while (end + 1 < frames.length && frames[end + 1] === frames[i]) end += 1;
    const at = frames[i];
    const n = end - i + 1;
    const next = end + 1 < frames.length ? frames[end + 1] : Infinity;
    const span = Math.min(maxSpread, next - at);
    for (let k = 0; k < n; k += 1) eased[i + k] = at + (span * k) / n;
    i = end + 1;
  }
  return eased;
};

/**
 * Fractional index of the leading edge for a series with (eased) reveal frames.
 * Whole part = last point that has arrived; fraction = progress toward the next
 * one, so the line still moves between points instead of snapping.
 * Returns -1 when none has arrived yet, which is a real state: before that frame
 * the chart has nothing to honestly draw.
 */
const revealedAt = (schedule: readonly number[], frame: number): number => {
  let last = -1;
  for (let i = 0; i < schedule.length; i += 1) {
    if (schedule[i] > frame) break;
    last = i;
  }
  if (last < 0) return -1;
  const here = schedule[last];
  const next = last + 1 < schedule.length ? schedule[last + 1] : undefined;
  if (next === undefined || next <= here) return last;
  return last + Math.min(1, (frame - here) / (next - here));
};

const Divider: React.FC = () => (
  <div style={{ height: 1, background: RULE, flexShrink: 0 }} />
);

export const SpeedLMDemo: React.FC = () => {
  const frame = useCurrentFrame();
  const { durationInFrames, width, height } = useVideoConfig();

  const train = data.training.train;
  const val = data.training.val;

  // Anchored mode is the real one: each point knows the frame on which the
  // terminal to the left printed it. The fallback only runs for data extracted
  // without a recording + timing sidecar to measure against.
  const anchored =
    train.every((p) => frameOf(p) !== undefined) && val.every((p) => frameOf(p) !== undefined);
  const gateFrame = frameOf(data.gate);

  // Train and val are eased as ONE sequence, in the same reading order the
  // terminal used (by step; a val line follows the train step that closed its
  // epoch). Easing them separately would let a diamond arrive before the train
  // point it sits above, which the terminal never did.
  const schedules = React.useMemo(() => {
    if (!anchored) return null;
    const order = [
      ...train.map((p, i) => ({ kind: "train" as const, i, step: p.step, tie: 0 })),
      ...val.map((p, i) => ({ kind: "val" as const, i, step: p.step, tie: 1 })),
    ].sort((a, b) => a.step - b.step || a.tie - b.tie);
    const eased = easedSchedule(
      order.map((o) => frameOf((o.kind === "train" ? train : val)[o.i]) as number),
      BURST_SPREAD_FRAMES,
    );
    const trainAt = new Array<number>(train.length);
    const valAt = new Array<number>(val.length);
    order.forEach((o, k) => {
      (o.kind === "train" ? trainAt : valAt)[o.i] = eased[k];
    });
    return { train: trainAt, val: valAt };
  }, [anchored, train, val]);

  // Fractional step count so the leading edge moves every frame, not every point.
  const revealed = schedules
    ? revealedAt(schedules.train, frame)
    : frac(frame, durationInFrames, REVEAL.training) * (train.length - 1);

  const gateProgress =
    gateFrame === undefined
      ? frac(frame, durationInFrames, REVEAL.gate)
      : interpolate(frame, [gateFrame, gateFrame + GATE_GROW_FRAMES], [0, 1], {
          extrapolateLeft: "clamp",
          extrapolateRight: "clamp",
        });

  // A validation diamond appears when its own line was printed; unanchored, it
  // appears once the training line has advanced past the epoch's last step.
  const shownVal = val.filter((v, i) =>
    schedules
      ? schedules.val[i] <= frame
      : revealed >= 0 && v.step <= (train[Math.floor(revealed)]?.step ?? -1),
  );

  const leftW = Math.round(width * 0.62);
  const rightW = width - leftW;

  return (
    <AbsoluteFill style={{ background: BG, flexDirection: "row" }}>
      {/* LEFT — the real recorded terminal, composited as-is. */}
      <div
        style={{
          width: leftW,
          height,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: BG,
          overflow: "hidden",
        }}
      >
        <OffthreadVideo
          src={staticFile(TERMINAL_VIDEO)}
          style={{ width: "100%", height: "100%", objectFit: "contain" }}
          muted
        />
      </div>

      {/* RIGHT — live charts driven by the real run artifacts. */}
      <div
        style={{
          width: rightW,
          height,
          background: PANEL,
          borderLeft: `1px solid ${RULE}`,
          boxSizing: "border-box",
          padding: "22px 32px 10px",
          display: "flex",
          flexDirection: "column",
          // Tightened from 20 -> 12 -> 9 as the gate card grew: it now carries
          // the "+15.0% per verifier step" reading under accepted length AND a
          // two-line caused-range caveat under throughput. At 12 the caveat's
          // second line fell off the bottom of the panel.
          gap: 9,
        }}
      >
        <header style={{ display: "flex", flexDirection: "column", gap: 6, flexShrink: 0 }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
            <span style={{ font: `700 25px ${MONO}`, color: TEXT, letterSpacing: 0.5 }}>
              SpeedLM
            </span>
            <span style={{ font: `400 16px ${MONO}`, color: MUTED }}>idle-tuning cycle</span>
          </div>
          <span style={{ font: `400 14px ${MONO}`, color: DIM }}>
            stock draft: {data.gate.stock_draft ?? "n/a"}
          </span>
        </header>

        <Divider />

        <LossChart train={train} val={shownVal} revealed={revealed} />

        <Divider />

        <AccuracyChart points={train} metric="full_acc" revealed={revealed} />

        <Divider />

        <ThroughputChart
          throughput={{
            stock: data.gate.throughput.stock,
            tuned: data.gate.throughput.tuned,
            delta_pct: data.gate.throughput.delta_pct,
            delta_standard_error_pct: data.gate.throughput.delta_standard_error_pct,
            stationary: data.gate.throughput.stationary,
          }}
          acceptedLength={{
            stock: data.gate.accepted_length.stock,
            tuned: data.gate.accepted_length.tuned,
            delta: data.gate.accepted_length.delta,
            delta_standard_error: data.gate.accepted_length.delta_standard_error,
          }}
          perRepeat={data.gate.per_repeat}
          verdict={data.gate.verdict}
          contexts={data.gate.num_contexts}
          repeats={data.gate.num_repeats}
          progress={gateProgress}
        />
      </div>
    </AbsoluteFill>
  );
};
