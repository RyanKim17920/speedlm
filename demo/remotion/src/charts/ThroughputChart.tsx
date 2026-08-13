import React from "react";
import { ChartCard, LegendItem } from "./frame";
import { DIM, MONO, MUTED, SERIES_BLUE, SERIES_GREEN, TEXT } from "../theme";

const W = 666;
const ROW_H = 30;
const GAP = 6; // 2px surface gap rule, widened for 1080p legibility
const LABEL_W = 150;
const VALUE_W = 118;

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
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
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
          return (
            <g key={name}>
              <text
                x={0}
                y={y + ROW_H / 2}
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
              <text
                x={W}
                y={y + ROW_H / 2}
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
  throughput: { stock: number; tuned: number; delta_pct: number };
  acceptedLength: { stock: number; tuned: number; delta: number };
  verdict: string;
  contexts: number;
  repeats: number;
  progress: number;
}> = ({ throughput, acceptedLength, verdict, contexts, repeats, progress }) => {
  const measures: Measure[] = [
    {
      label: "decode throughput",
      unit: "tok/s",
      stock: throughput.stock,
      tuned: throughput.tuned,
      // 2dp, matching the `decode tok/s 143.65 130.66` line the terminal prints
      // on the left -- rounding to 1dp here made the two halves disagree.
      format: (v) => v.toFixed(2),
      delta: `${throughput.delta_pct >= 0 ? "+" : ""}${throughput.delta_pct.toFixed(2)}%`,
    },
    {
      label: "accepted length",
      unit: "tokens/step",
      stock: acceptedLength.stock,
      tuned: acceptedLength.tuned,
      format: (v) => v.toFixed(3),
      delta: `${acceptedLength.delta >= 0 ? "+" : ""}${acceptedLength.delta.toFixed(3)}`,
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
      <div style={{ display: "flex", flexDirection: "column", gap: 20, paddingTop: 4 }}>
        {measures.map((m) => (
          <Group key={m.label} m={m} progress={progress} />
        ))}
      </div>
    </ChartCard>
  );
};
