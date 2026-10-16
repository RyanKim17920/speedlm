import React from "react";
import { AbsoluteFill, Easing, interpolate, useCurrentFrame } from "remotion";
import { BG, DIM, MONO, PANEL, POSITION_RAMP, RULE, SERIES_BLUE, SERIES_GREEN, TEXT } from "../theme";
import data from "../data.json";

const TRAINING = data.training.train;
const LAST_STEP = TRAINING[TRAINING.length - 1].step;

const WIDTH = 760;
const HEIGHT = 390;
const PAD = { top: 40, right: 34, bottom: 52, left: 72 };

const xFor = (step: number) => PAD.left + (step / LAST_STEP) * (WIDTH - PAD.left - PAD.right);
const yForLoss = (loss: number) => PAD.top + (1 - Math.min(loss, 11) / 11) * (HEIGHT - PAD.top - PAD.bottom);
const yForAccuracy = (value: number) => PAD.top + (1 - value) * (HEIGHT - PAD.top - PAD.bottom);

const pathFor = (points: Array<{ x: number; y: number }>) =>
  points.map((point, index) => `${index === 0 ? "M" : "L"}${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(" ");

const PlotFrame: React.FC<{ children: React.ReactNode; endLabel: string; title: string }> = ({ children, endLabel, title }) => (
  <section style={{ flex: 1, background: PANEL, border: `1px solid ${RULE}`, borderRadius: 22, padding: "30px 30px 22px" }}>
    <div style={{ color: TEXT, fontSize: 31, fontWeight: 680, marginBottom: 8 }}>{title}</div>
    <div style={{ color: DIM, fontSize: 22, marginBottom: 2 }}>{endLabel}</div>
    <svg width={WIDTH} height={HEIGHT} viewBox={`0 0 ${WIDTH} ${HEIGHT}`}>
      {[0, 0.5, 1].map((position) => {
        const y = PAD.top + position * (HEIGHT - PAD.top - PAD.bottom);
        return <line key={position} x1={PAD.left} x2={WIDTH - PAD.right} y1={y} y2={y} stroke={RULE} strokeWidth={1} />;
      })}
      {children}
      <text x={PAD.left} y={HEIGHT - 18} fill={DIM} fontFamily={MONO} fontSize={18}>step 0</text>
      <text x={WIDTH - PAD.right} y={HEIGHT - 18} fill={DIM} fontFamily={MONO} fontSize={18} textAnchor="end">step {LAST_STEP}</text>
    </svg>
  </section>
);

export const TrainingScene: React.FC = () => {
  const frame = useCurrentFrame();
  const progress = interpolate(frame, [20, 330], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.inOut(Easing.cubic),
  });
  const visibleStep = progress * LAST_STEP;
  const visible = TRAINING.filter((point) => point.step <= visibleStep);
  const lossPath = pathFor(visible.map((point) => ({ x: xFor(point.step), y: yForLoss(point.loss) })));
  const accuracyPaths = [0, 1, 2].map((position) =>
    pathFor(visible.map((point) => ({ x: xFor(point.step), y: yForAccuracy(point.full_acc[position]) }))),
  );
  const latest = visible.length > 0 ? visible[visible.length - 1] : TRAINING[0];
  return (
    <AbsoluteFill style={{ background: BG, padding: "66px 78px", fontFamily: MONO }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 36 }}>
        <div>
          <div style={{ color: TEXT, fontSize: 52, fontWeight: 720 }}>Idle GPU time trains the candidate draft</div>
          <div style={{ color: DIM, fontSize: 24, marginTop: 12 }}>1,024-record window · 922 optimizer steps · 8 epochs</div>
        </div>
        <div style={{ color: SERIES_GREEN, border: `1px solid ${SERIES_GREEN}`, borderRadius: 999, padding: "12px 22px", fontSize: 21, fontWeight: 700 }}>
          RECORDED TRAINER STDOUT
        </div>
      </div>

      <div style={{ display: "flex", gap: 34 }}>
        <PlotFrame title="Draft training loss · lower is better" endLabel={`${latest.loss.toFixed(2)} loss`}>
          <path d={lossPath} fill="none" stroke={SERIES_BLUE} strokeWidth={5} strokeLinecap="round" strokeLinejoin="round" />
          {visible.length > 0 ? <circle cx={xFor(latest.step)} cy={yForLoss(latest.loss)} r={8} fill={SERIES_BLUE} /> : null}
          <text x={PAD.left - 16} y={PAD.top + 6} fill={DIM} fontFamily={MONO} fontSize={18} textAnchor="end">11</text>
          <text x={PAD.left - 16} y={HEIGHT - PAD.bottom + 6} fill={DIM} fontFamily={MONO} fontSize={18} textAnchor="end">0</text>
        </PlotFrame>
        <PlotFrame title="Draft accuracy · higher is better" endLabel={`${(latest.full_acc[2] * 100).toFixed(1)}% at position 3`}>
          {accuracyPaths.map((path, index) => (
            <path key={index} d={path} fill="none" stroke={POSITION_RAMP[index]} strokeWidth={4} strokeLinecap="round" strokeLinejoin="round" />
          ))}
          {visible.length > 0 ? [0, 1, 2].map((position) => (
            <circle key={position} cx={xFor(latest.step)} cy={yForAccuracy(latest.full_acc[position])} r={7} fill={POSITION_RAMP[position]} />
          )) : null}
          <text x={PAD.left - 16} y={PAD.top + 6} fill={DIM} fontFamily={MONO} fontSize={18} textAnchor="end">100%</text>
          <text x={PAD.left - 16} y={HEIGHT - PAD.bottom + 6} fill={DIM} fontFamily={MONO} fontSize={18} textAnchor="end">0%</text>
        </PlotFrame>
      </div>
    </AbsoluteFill>
  );
};
