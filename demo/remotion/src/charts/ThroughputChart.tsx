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
   * ±1 standard error of the delta, expressed in the SAME units as `stock` /
   * `tuned` so it can be drawn on the tuned bar. Undefined = the gate did not
   * measure dispersion for this measure, and nothing is drawn (an absent
   * interval must not read as a tight one).
   */
  errorAbs?: number;
  /** Shown under the group when the measure is not safe to read as settled. */
  caveat?: string;
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
          const err = name === "tuned" && m.errorAbs ? px(m.errorAbs) * progress : 0;
          const lo = Math.max(LABEL_W, end - err);
          const hi = end + err;
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
  verdict: string;
  contexts: number;
  repeats: number;
  progress: number;
}> = ({ throughput, acceptedLength, verdict, contexts, repeats, progress }) => {
  // The throughput SE is quoted in percentage POINTS of the delta, so it has to
  // be pushed back through the delta's own definition to land in tok/s:
  //   delta_pct = (tuned - stock) / stock * 100  =>  d(tuned) = stock * dpct/100
  const tokSE =
    throughput.delta_standard_error_pct == null
      ? undefined
      : (throughput.stock * throughput.delta_standard_error_pct) / 100;
  const notStationary = throughput.stationary === false;

  const measures: Measure[] = [
    {
      label: "decode throughput",
      unit: "tok/s",
      stock: throughput.stock,
      tuned: throughput.tuned,
      // 2dp, matching the `decode tok/s 141.31 117.85` line the terminal prints
      // on the left -- rounding to 1dp here made the two halves disagree.
      format: (v) => v.toFixed(2),
      delta:
        `${throughput.delta_pct >= 0 ? "+" : ""}${throughput.delta_pct.toFixed(2)}%` +
        (throughput.delta_standard_error_pct == null
          ? ""
          : ` ± ${throughput.delta_standard_error_pct.toFixed(2)}pp`),
      errorAbs: tokSE,
      caveat: notStationary
        ? "not stationary: the node was contended and both arms slowed in later repeats. Read the range, not the point."
        : undefined,
    },
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
          : ` ± ${acceptedLength.delta_standard_error.toFixed(4)}`),
      errorAbs: acceptedLength.delta_standard_error ?? undefined,
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
            <span style={{ font: `400 15px/1 ${MONO}`, color: MUTED }}>±1 SE of delta</span>
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
