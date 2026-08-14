import React from "react";
import { ChartCard, LegendItem } from "./frame";
import { DIM, MONO, MUTED, SERIES_BLUE, SERIES_GREEN, TEXT } from "../theme";

const W = 666;
const ROW_H = 26;
const GAP = 6; // 2px surface gap rule, widened for 1080p legibility
const LABEL_W = 150;
const VALUE_W = 118;

/**
 * The one measure this card is about.
 *
 * Throughput used to be charted here as a second paired-bar group with an
 * asymmetric whisker. It is no longer: the gate vetoed that channel as
 * non-stationary, so putting it on the same visual footing as the measure the
 * gate actually promotes on gave equal weight to a settled result and an
 * unsettled one. It is now a single supporting line of prose beneath the chart.
 */
type Measure = {
  label: string;
  unit: string;
  stock: number;
  tuned: number;
  format: (v: number) => string;
  delta: string;
};

/** One paired-bar group on its own scale — never a second y axis. */
const Group: React.FC<{ m: Measure; progress: number }> = ({ m, progress }) => {
  const max = Math.max(m.stock, m.tuned);
  const track = W - LABEL_W - VALUE_W;
  const w = (v: number) => Math.max(2, (v / max) * track * progress);

  const rows: Array<[string, number, string]> = [
    ["stock", m.stock, SERIES_BLUE],
    ["tuned", m.tuned, SERIES_GREEN],
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
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
                y={y + 3}
                width={w(value)}
                height={ROW_H - 6}
                rx={4}
                fill={color}
              />
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
  /**
   * The same head's accepted-length delta as measured by independent gates.
   * Read off real decision.json records by extract_series.py, never typed in, so
   * the "reproduced Nx" reading can never outrun the evidence behind it.
   */
  reproductions?: ReadonlyArray<{ run: string; delta: number }>;
  verdict: string;
  contexts: number;
  repeats: number;
  progress: number;
}> = ({
  throughput,
  acceptedLength,
  perRepeat,
  reproductions,
  verdict,
  contexts,
  repeats,
  progress,
}) => {
  const notStationary = throughput.stationary === false;

  // The headline, stated the way a reader can use it: a 15% lift in how much of
  // each draft survives verification. Derived, not typed in.
  const acceptedPct = (acceptedLength.delta / acceptedLength.stock) * 100;

  const se = acceptedLength.delta_standard_error;

  // Every throughput figure on this card is derived from the per-repeat rows the
  // gate wrote, so the supporting line cannot drift from the record: the range is
  // the observed per-repeat extremes, the tuned band is that arm's own min..max,
  // and the baseline drift is stock's first repeat against its last.
  const rows = perRepeat ?? [];
  const deltas = rows.map((r) => (r.tuned_tok_per_sec / r.stock_tok_per_sec - 1) * 100);
  const tunedTok = rows.map((r) => r.tuned_tok_per_sec);
  const throughputRange = deltas.length
    ? `+${Math.min(...deltas).toFixed(1)}% to +${Math.max(...deltas).toFixed(1)}%`
    : `${throughput.delta_pct >= 0 ? "+" : ""}${throughput.delta_pct.toFixed(1)}%`;
  const tunedBand = tunedTok.length
    ? `${Math.round(Math.min(...tunedTok))}-${Math.round(Math.max(...tunedTok))}`
    : throughput.tuned.toFixed(0);
  const stockDrift = rows.length
    ? `${Math.round(rows[0].stock_tok_per_sec)}->${Math.round(
        rows[rows.length - 1].stock_tok_per_sec,
      )}`
    : throughput.stock.toFixed(0);

  const measure: Measure = {
    label: "accepted length",
    unit: "tokens/step",
    stock: acceptedLength.stock,
    tuned: acceptedLength.tuned,
    format: (v) => v.toFixed(4),
    delta:
      `${acceptedLength.delta >= 0 ? "+" : ""}${acceptedLength.delta.toFixed(4)}` +
      (se == null ? "" : ` ± ${se.toFixed(4)} SE`),
  };

  // Nothing about the result is drawn before the terminal on the left has
  // printed it: at progress 0 the bars sit at zero and the hero block is absent
  // outright, exactly as the whole card behaved before.
  const arrived = progress > 0;
  const settled = progress > 0.98;

  const reps = reproductions ?? [];

  return (
    <ChartCard
      title="Gate replay: stock vs tuned"
      subtitle={`${contexts} contexts × ${repeats} repeats · greedy`}
      legend={
        <>
          <LegendItem color={SERIES_BLUE} label="stock draft" shape="square" />
          <LegendItem color={SERIES_GREEN} label="tuned draft" shape="square" />
        </>
      }
    >
      <div style={{ display: "flex", flexDirection: "column", gap: 11, paddingTop: 2 }}>
        {/* HERO — the result, as one number a viewer can carry away. Text tokens
            only: the green belongs to the tuned bar directly beneath it. */}
        <div
          style={{
            display: "flex",
            alignItems: "flex-end",
            gap: 20,
            opacity: arrived ? 1 : 0,
          }}
        >
          <span
            style={{
              font: `700 52px/0.95 ${MONO}`,
              color: TEXT,
              letterSpacing: -1,
              fontVariantNumeric: "tabular-nums",
            }}
          >
            +{(acceptedPct * progress).toFixed(1)}%
          </span>
          <div style={{ display: "flex", flexDirection: "column", gap: 3, paddingBottom: 3 }}>
            <span style={{ font: `600 16px/1.2 ${MONO}`, color: TEXT }}>
              more tokens accepted per verifier step
            </span>
            <span style={{ font: `400 15px/1.2 ${MONO}`, color: MUTED }}>
              {acceptedLength.stock.toFixed(2)} → {acceptedLength.tuned.toFixed(2)} tokens/step
              {se == null ? "" : `  ·  +${acceptedLength.delta.toFixed(4)} SE ${se.toFixed(4)}`}
            </span>
            {reps.length > 1 ? (
              <span
                style={{
                  font: `400 15px/1.2 ${MONO}`,
                  color: MUTED,
                  opacity: settled ? 1 : 0,
                }}
              >
                reproduced {reps.length}x:{" "}
                {reps.map((r) => `+${r.delta.toFixed(4)}`).join(" / ")}
              </span>
            ) : null}
          </div>
        </div>

        <Group m={measure} progress={progress} />

        {/* SUPPORTING — throughput, once, as prose. Not a second chart. */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 2,
            font: `400 14px/1.35 ${MONO}`,
            color: MUTED,
            opacity: settled ? 1 : 0,
          }}
        >
          <span>
            wall-clock throughput {throughputRange} across repeats — the tuned arm held
          </span>
          <span>
            {tunedBand} tok/s while the shared-node baseline drifted {stockDrift} tok/s.
          </span>
          {notStationary ? (
            <span style={{ color: DIM, paddingTop: 2 }}>
              gate vetoed the throughput channel as non-stationary (final verdict: {verdict}); the
              accepted-length channel passed at +{acceptedLength.delta.toFixed(4)}.
            </span>
          ) : null}
        </div>
      </div>
    </ChartCard>
  );
};
