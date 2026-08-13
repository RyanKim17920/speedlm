import React from "react";
import { ChartCard, LegendItem } from "./frame";
import { DIM, MONO, MUTED, PANEL, SERIES_BLUE, SERIES_GREEN, TEXT } from "../theme";

const W = 666;
const ROW_H = 30;
const GAP = 6; // 2px surface gap rule, widened for 1080p legibility
const LABEL_W = 150;
const VALUE_W = 118;

/** Half-height of an error-bar cap, and its stroke weight. */
const CAP_H = 9;
const ERR_STROKE = 2;

type Measure = {
  label: string;
  unit: string;
  stock: number;
  tuned: number;
  format: (v: number) => string;
  delta: string;
  /**
   * The delta's interval, as the two ABSOLUTE endpoints it puts on the tuned
   * bar, in the same units as `stock` / `tuned`. Asymmetric on purpose: for
   * accepted length these are mean ±1 SE (sub-pixel), for throughput they are
   * the lowest and highest per-repeat delta actually observed, which is not
   * symmetric about the central estimate. Undefined = the gate measured no
   * dispersion for this measure, and nothing is drawn (an absent interval must
   * not read as a tight one).
   */
  intervalAbs?: readonly [number, number];
  /** Shown under the group when the measure is not safe to read as settled. */
  caveat?: string;
  /** Second line under the group: what the number MEANS, in one clause. */
  note?: string;
};

/**
 * One paired-bar group on its own scale — never a second y axis.
 *
 * The tuned bar carries a ±1 SE whisker when the gate measured one. This is the
 * whole point of the chart on this cut: the accepted-length delta's interval is
 * so tight it is sub-pixel, while the throughput delta's interval is a fifth of
 * the bar. Drawing both on the same rules lets the viewer see that difference
 * instead of being told about it.
 */
const Group: React.FC<{ m: Measure; progress: number }> = ({ m, progress }) => {
  const max = Math.max(m.stock, m.tuned);
  const track = W - LABEL_W - VALUE_W;
  const px = (v: number) => (v / max) * track;
  const w = (v: number) => Math.max(2, px(v) * progress);

  const rows: Array<[string, number, string]> = [
    ["stock", m.stock, SERIES_BLUE],
    ["tuned", m.tuned, SERIES_GREEN],
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
        <span style={{ font: `500 16px ${MONO}`, color: MUTED }}>{m.label}</span>
        <span style={{ font: `400 14px ${MONO}`, color: DIM }}>{m.unit}</span>
        <span
          style={{
            marginLeft: "auto",
            font: `600 16px ${MONO}`,
            color: TEXT,
            opacity: progress > 0.98 ? 1 : 0,
          }}
        >
          {m.delta}
        </span>
      </div>
      <svg width={W} height={ROW_H * 2 + GAP} style={{ display: "block" }}>
        {rows.map(([name, value, color], i) => {
          const y = i * (ROW_H + GAP);
          const mid = y + ROW_H / 2;
          const end = LABEL_W + w(value);
          const iv = name === "tuned" ? m.intervalAbs : undefined;
          // Interpolated from the bar's own end so the whisker grows with it and
          // never floats away from the mark it qualifies.
          const err = iv && progress > 0.02 ? 1 : 0;
          const lo = iv ? Math.max(LABEL_W, end - (px(value) - px(iv[0])) * progress) : end;
          const hi = iv ? end + (px(iv[1]) - px(value)) * progress : end;
          return (
            <g key={name}>
              <text
                x={0}
                y={mid}
                dominantBaseline="middle"
                style={{ font: `400 16px ${MONO}`, fill: MUTED }}
              >
                {name}
              </text>
              <rect
                x={LABEL_W}
                y={y + 4}
                width={w(value)}
                height={ROW_H - 8}
                rx={4}
                fill={color}
              />
              {err > 0 ? (
                <g>
                  {/* 2px surface ring first, so the whisker stays legible where
                      it crosses the fill it is drawn on top of. */}
                  <path
                    d={`M${lo} ${mid - CAP_H}V${mid + CAP_H}M${lo} ${mid}H${hi}M${hi} ${
                      mid - CAP_H
                    }V${mid + CAP_H}`}
                    stroke={PANEL}
                    strokeWidth={ERR_STROKE + 4}
                    strokeLinecap="butt"
                    fill="none"
                  />
                  <path
                    d={`M${lo} ${mid - CAP_H}V${mid + CAP_H}M${lo} ${mid}H${hi}M${hi} ${
                      mid - CAP_H
                    }V${mid + CAP_H}`}
                    stroke={TEXT}
                    strokeWidth={ERR_STROKE}
                    strokeLinecap="butt"
                    fill="none"
                  />
                </g>
              ) : null}
              <text
                x={W}
                y={mid}
                dominantBaseline="middle"
                textAnchor="end"
                style={{ font: `600 18px ${MONO}`, fill: TEXT }}
              >
                {m.format(value * progress)}
              </text>
            </g>
          );
        })}
      </svg>
      {m.note ? (
        <div
          style={{
            font: `600 14px ${MONO}`,
            color: TEXT,
            opacity: progress > 0.98 ? 1 : 0,
          }}
        >
          {m.note}
        </div>
      ) : null}
      {m.caveat ? (
        <div
          style={{
            display: "flex",
            gap: 8,
            alignItems: "baseline",
            font: `400 14px ${MONO}`,
            color: MUTED,
            opacity: progress > 0.98 ? 1 : 0,
          }}
        >
          <span style={{ font: `700 14px ${MONO}`, color: TEXT }}>[!]</span>
          <span>{m.caveat}</span>
        </div>
      ) : null}
    </div>
  );
};

export const ThroughputChart: React.FC<{
  throughput: {
    stock: number;
    tuned: number;
    delta_pct: number;
    delta_standard_error_pct?: number | null;
    stationary?: boolean | null;
  };
  acceptedLength: {
    stock: number;
    tuned: number;
    delta: number;
    delta_standard_error?: number | null;
  };
  perRepeat?: ReadonlyArray<{ stock_tok_per_sec: number; tuned_tok_per_sec: number }>;
  verdict: string;
  contexts: number;
  repeats: number;
  progress: number;
}> = ({ throughput, acceptedLength, perRepeat, verdict, contexts, repeats, progress }) => {
  const notStationary = throughput.stationary === false;

  // The throughput interval is NOT ±1 SE about the central estimate. The gate
  // vetoed this measure precisely because the per-repeat delta moved, so the
  // honest interval is the spread that actually occurred: the lowest and highest
  // per-repeat delta, mapped back onto the tuned bar through the delta's own
  // definition (tuned = stock * (1 + delta/100)). Falling back to ±1 SE when no
  // per-repeat rows are present keeps an interval on screen either way.
  const deltas = (perRepeat ?? []).map(
    (r) => (r.tuned_tok_per_sec / r.stock_tok_per_sec - 1) * 100,
  );
  const tokSE =
    throughput.delta_standard_error_pct == null
      ? undefined
      : (throughput.stock * throughput.delta_standard_error_pct) / 100;
  const tokInterval: readonly [number, number] | undefined = deltas.length
    ? [
        throughput.stock * (1 + Math.min(...deltas) / 100),
        throughput.stock * (1 + Math.max(...deltas) / 100),
      ]
    : tokSE === undefined
      ? undefined
      : [throughput.tuned - tokSE, throughput.tuned + tokSE];

  // The headline, stated the way a reader can use it: a 15% lift in how much of
  // each draft survives verification. Derived, not typed in.
  const acceptedPct =
    acceptedLength.delta == null || !acceptedLength.stock
      ? undefined
      : (acceptedLength.delta / acceptedLength.stock) * 100;

  const measures: Measure[] = [
    {
      label: "accepted length",
      unit: "tokens/step",
      stock: acceptedLength.stock,
      tuned: acceptedLength.tuned,
      format: (v) => v.toFixed(4),
      delta:
        `${acceptedLength.delta >= 0 ? "+" : ""}${acceptedLength.delta.toFixed(4)}` +
        (acceptedLength.delta_standard_error == null
          ? ""
          : ` ± ${acceptedLength.delta_standard_error.toFixed(4)} SE`),
      intervalAbs:
        acceptedLength.delta_standard_error == null
          ? undefined
          : [
              acceptedLength.tuned - acceptedLength.delta_standard_error,
              acceptedLength.tuned + acceptedLength.delta_standard_error,
            ],
      note:
        acceptedPct === undefined
          ? undefined
          : `+${acceptedPct.toFixed(1)}% more tokens accepted per verifier step — reproduced three times`,
    },
    {
      label: "decode throughput",
      unit: "tok/s",
      stock: throughput.stock,
      tuned: throughput.tuned,
      // 2dp, matching the `decode tok/s 144.71 124.59` line the terminal prints
      // on the left -- rounding to 1dp here made the two halves disagree.
      format: (v) => v.toFixed(2),
      // A range, never a settled point: the delta is quoted by its observed
      // per-repeat extremes with the central estimate in parentheses.
      delta: deltas.length
        ? `+${Math.min(...deltas).toFixed(1)}% to +${Math.max(...deltas).toFixed(1)}%  (central +${throughput.delta_pct.toFixed(1)}%)`
        : `${throughput.delta_pct >= 0 ? "+" : ""}${throughput.delta_pct.toFixed(2)}%`,
      intervalAbs: tokInterval,
      caveat: notStationary
        // One line, because the panel has exactly one line of room here: any
        // second line falls off the bottom of the 1080p frame. It has to name
        // the veto, the cause, and which arm moved -- so it does, in numbers.
        ? "vetoed, not stationary: STOCK drifted 127->121 tok/s; tuned flat 143-147"
        : undefined,
    },
  ];

  const promoted = verdict === "promote";

  return (
    <ChartCard
      title="Gate replay: stock vs tuned"
      subtitle={`${contexts} contexts × ${repeats} repeats`}
      legend={
        <>
          <LegendItem color={SERIES_BLUE} label="stock draft" shape="square" />
          <LegendItem color={SERIES_GREEN} label="tuned draft" shape="square" />
          <span style={{ display: "flex", alignItems: "center", gap: 7 }}>
            <svg width={22} height={13} style={{ display: "block" }}>
              <path
                d="M2 1V12M2 6.5H20M20 1V12"
                stroke={TEXT}
                strokeWidth={2}
                fill="none"
              />
            </svg>
            <span style={{ font: `400 15px/1 ${MONO}`, color: MUTED }}>delta interval</span>
          </span>
          <span
            style={{
              marginLeft: "auto",
              display: "flex",
              alignItems: "center",
              gap: 7,
              padding: "3px 12px",
              borderRadius: 999,
              border: `1px solid ${promoted ? SERIES_GREEN : DIM}`,
              font: `600 15px ${MONO}`,
              color: TEXT,
              // Hidden outright until the replay starts filling in: the verdict
              // is the one number the video must not show before the terminal
              // has printed it, and a dimmed-but-readable pill still shows it.
              opacity: progress <= 0 ? 0 : progress > 0.98 ? 1 : 0.25,
            }}
          >
            <span
              style={{
                width: 9,
                height: 9,
                borderRadius: 999,
                background: promoted ? SERIES_GREEN : DIM,
              }}
            />
            verdict: {verdict}
          </span>
        </>
      }
    >
      <div style={{ display: "flex", flexDirection: "column", gap: 14, paddingTop: 4 }}>
        {measures.map((m) => (
          <Group key={m.label} m={m} progress={progress} />
        ))}
      </div>
    </ChartCard>
  );
};
