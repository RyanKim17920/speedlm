import React from "react";
import { ChartCard, LegendItem, YGrid, scale } from "./frame";
import { DIM, MONO, PANEL, POSITION_RAMP, TEXT } from "../theme";

export type AccPoint = { step: number; full_acc: number[]; cond_acc: number[] };

const W = 666;
const H = 258;
const PAD = { t: 12, r: 74, b: 34, l: 52 };

/**
 * Per-draft-position accuracy over steps. Positions are ordered, so they take a
 * single-hue ordinal ramp (light = position 0, dark = position 2) rather than
 * unrelated categorical hues.
 */
export const AccuracyChart: React.FC<{
  points: AccPoint[];
  metric: "full_acc" | "cond_acc";
  revealed: number;
}> = ({ points, metric, revealed }) => {
  const lastStep = points[points.length - 1].step;
  const x = scale(points[0].step, lastStep, PAD.l, W - PAD.r);
  const y = scale(0, 1, H - PAD.b, PAD.t);

  // `revealed < 0`: the terminal has not printed a step yet, so there is nothing
  // to draw but the axes.
  const started = revealed >= 0;
  const whole = Math.floor(revealed);
  const frac = revealed - whole;
  const shown = started ? points.slice(0, Math.min(points.length, whole + 1)) : [];

  const seriesPts = (i: number) => {
    const pts = shown.map((p) => [x(p.step), y(p[metric][i])] as const);
    if (whole > 0 && whole < points.length && frac > 0) {
      const a = points[whole - 1];
      const b = points[whole];
      pts[pts.length - 1] = [
        x(a.step + (b.step - a.step) * frac),
        y(a[metric][i] + (b[metric][i] - a[metric][i]) * frac),
      ] as const;
    }
    return pts;
  };

  const label = metric === "full_acc" ? "full_acc" : "cond_acc";

  // The three positions converge as training proceeds (94/88/83%), so their
  // direct labels can end up within a few pixels of each other and overlap.
  // Nudge them apart to a minimum gap -- the marker stays on the true value, only
  // the text moves, so nothing is misplaced on the y axis that a reader measures.
  const LABEL_GAP = 19;
  const labelYs: number[] = [];
  if (started) {
    const heads = [0, 1, 2].map((i) => ({ i, y: y(shown[shown.length - 1][metric][i]) }));
    heads.sort((a, b) => a.y - b.y);
    let prev = -Infinity;
    for (const h of heads) {
      const at = Math.max(h.y, prev + LABEL_GAP);
      labelYs[h.i] = at;
      prev = at;
    }
  }

  return (
    <ChartCard
      title="Per-position draft accuracy"
      subtitle={label}
      legend={
        <>
          {[0, 1, 2].map((i) => (
            <LegendItem key={i} color={POSITION_RAMP[i]} label={`${label}_${i}`} />
          ))}
        </>
      }
    >
      <svg width={W} height={H} style={{ display: "block", overflow: "visible" }}>
        <YGrid
          ticks={[0, 0.5, 1]}
          y={y}
          x0={PAD.l}
          x1={W - PAD.r}
          format={(v) => `${(v * 100).toFixed(0)}%`}
        />

        {(started ? [0, 1, 2] : []).map((i) => {
          const pts = seriesPts(i);
          const head = pts[pts.length - 1];
          const value = shown[shown.length - 1][metric][i];
          return (
            <g key={i}>
              <polyline
                points={pts.map(([px, py]) => `${px},${py}`).join(" ")}
                fill="none"
                stroke={POSITION_RAMP[i]}
                strokeWidth={2.5}
                strokeLinejoin="round"
                strokeLinecap="round"
              />
              <circle
                cx={head[0]}
                cy={head[1]}
                r={4.5}
                fill={POSITION_RAMP[i]}
                stroke={PANEL}
                strokeWidth={2}
              />
              {/* Direct label at the leading edge only — 3 series, so all get one. */}
              <text
                x={Math.min(head[0] + 11, W - PAD.r + 6)}
                y={labelYs[i]}
                dominantBaseline="middle"
                style={{ font: `600 16px ${MONO}`, fill: TEXT }}
              >
                {(value * 100).toFixed(0)}%
              </text>
            </g>
          );
        })}

        <line
          x1={PAD.l}
          x2={W - PAD.r}
          y1={H - PAD.b}
          y2={H - PAD.b}
          stroke={DIM}
          strokeWidth={1}
        />
        <text x={PAD.l} y={H - PAD.b + 20} style={{ font: `400 14px ${MONO}`, fill: DIM }}>
          step 0
        </text>
        <text
          x={W - PAD.r}
          y={H - PAD.b + 20}
          textAnchor="end"
          style={{ font: `400 14px ${MONO}`, fill: DIM }}
        >
          step {lastStep}
        </text>
      </svg>
    </ChartCard>
  );
};
