import React from "react";
import { ChartCard, LegendItem, YGrid, scale } from "./frame";
import { DIM, MONO, PANEL, SERIES_AMBER, SERIES_BLUE, TEXT } from "../theme";

export type TrainPoint = { step: number; epoch: number; loss: number };
export type ValPoint = { step: number; epoch: number; loss: number };

const W = 666;
const H = 258;
const PAD = { t: 12, r: 66, b: 34, l: 52 };

/**
 * Training loss over optimizer steps, revealed progressively.
 *
 * `revealed` is a fractional step count (not an index) so the leading edge of the
 * line advances smoothly between points instead of snapping once per frame.
 * `revealed < 0` means the terminal has not printed a single step yet, and the
 * chart draws its axes and nothing else -- there is no first point to imply.
 *
 * `val` is already filtered by the caller to the diamonds that have been printed;
 * this component does not decide what is visible, only how it looks.
 */
export const LossChart: React.FC<{
  train: TrainPoint[];
  val: ValPoint[];
  revealed: number;
}> = ({ train, val, revealed }) => {
  const lastStep = train[train.length - 1].step;
  const maxLoss = Math.max(...train.map((p) => p.loss));
  const top = Math.ceil(maxLoss / 2) * 2;

  const x = scale(train[0].step, lastStep, PAD.l, W - PAD.r);
  const y = scale(0, top, H - PAD.b, PAD.t);

  const started = revealed >= 0;
  const whole = Math.floor(revealed);
  const frac = revealed - whole;
  const shown = started ? train.slice(0, Math.min(train.length, whole + 1)) : [];

  const pts = shown.map((p) => [x(p.step), y(p.loss)] as const);
  // Interpolate the leading edge toward the next real point.
  if (whole > 0 && whole < train.length && frac > 0) {
    const a = train[whole - 1];
    const b = train[whole];
    pts[pts.length - 1] = [
      x(a.step + (b.step - a.step) * frac),
      y(a.loss + (b.loss - a.loss) * frac),
    ] as const;
  }

  const head = pts.length ? pts[pts.length - 1] : null;
  const headLoss = shown.length ? shown[shown.length - 1].loss : null;

  const ticks = [0, top / 2, top];

  return (
    <ChartCard
      title="Draft training loss"
      subtitle={`${train.length} steps · 3 epochs`}
      legend={
        <>
          <LegendItem color={SERIES_BLUE} label="train/loss" />
          <LegendItem color={SERIES_AMBER} label="val/loss_epoch" shape="diamond" />
        </>
      }
    >
      <svg width={W} height={H} style={{ display: "block", overflow: "visible" }}>
        <YGrid ticks={ticks} y={y} x0={PAD.l} x1={W - PAD.r} format={(v) => v.toFixed(0)} />

        <polyline
          points={pts.map(([px, py]) => `${px},${py}`).join(" ")}
          fill="none"
          stroke={SERIES_BLUE}
          strokeWidth={2.5}
          strokeLinejoin="round"
          strokeLinecap="round"
        />

        {/* Validation loss: one diamond per completed epoch, direct-labeled. */}
        {val.map((v) => (
          <g key={v.epoch}>
            <rect
              x={x(v.step) - 6}
              y={y(v.loss) - 6}
              width={12}
              height={12}
              rx={2}
              transform={`rotate(45 ${x(v.step)} ${y(v.loss)})`}
              fill={SERIES_AMBER}
              stroke={PANEL}
              strokeWidth={2}
            />
            <text
              x={x(v.step)}
              y={y(v.loss) - 15}
              textAnchor="middle"
              style={{ font: `500 14px ${MONO}`, fill: TEXT }}
            >
              {v.loss.toFixed(2)}
            </text>
          </g>
        ))}

        {/* Leading edge: a marker plus the live value, so no number is on every point. */}
        {head && headLoss !== null ? (
          <>
            <circle
              cx={head[0]}
              cy={head[1]}
              r={5}
              fill={SERIES_BLUE}
              stroke={PANEL}
              strokeWidth={2}
            />
            <text
              x={Math.min(head[0] + 12, W - PAD.r + 8)}
              y={head[1]}
              dominantBaseline="middle"
              style={{ font: `600 17px ${MONO}`, fill: TEXT }}
            >
              {headLoss.toFixed(2)}
            </text>
          </>
        ) : null}

        <line
          x1={PAD.l}
          x2={W - PAD.r}
          y1={H - PAD.b}
          y2={H - PAD.b}
          stroke={DIM}
          strokeWidth={1}
        />
        <text
          x={PAD.l}
          y={H - PAD.b + 20}
          style={{ font: `400 14px ${MONO}`, fill: DIM }}
        >
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
