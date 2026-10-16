import React from "react";
import {Easing, interpolate, useCurrentFrame} from "remotion";
import {DIM, MONO, PANEL, RULE, SERIES_BLUE, SERIES_GREEN, TEXT} from "../theme";
import {SubmissionFrame} from "./SubmissionFrame";
import {RETAINED_RUN, RETAINED_TRAINING} from "./retainedRunData";

const WIDTH = 790;
const HEIGHT = 385;
const PAD = {top: 32, right: 28, bottom: 48, left: 70};
const LAST_STEP = 261;

const xFor = (step: number) => PAD.left + (step / LAST_STEP) * (WIDTH - PAD.left - PAD.right);
const yForLoss = (value: number) => PAD.top + (1 - value / 10) * (HEIGHT - PAD.top - PAD.bottom);
const yForAccuracy = (value: number) => PAD.top + (1 - value) * (HEIGHT - PAD.top - PAD.bottom);
const pathFor = (points: ReadonlyArray<{x: number; y: number}>) =>
  points.map((point, index) => `${index === 0 ? "M" : "L"}${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(" ");

const Plot: React.FC<{children: React.ReactNode; endLabel: string; title: string}> = ({children, endLabel, title}) => (
  <section style={{flex: 1, minWidth: 0, background: PANEL, border: `1px solid ${RULE}`, borderRadius: 17, padding: "20px 22px 12px"}}>
    <div style={{display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 16}}>
      <div style={{color: TEXT, fontSize: 27, fontWeight: 780}}>{title}</div>
      <div style={{color: DIM, fontSize: 18}}>{endLabel}</div>
    </div>
    <svg width="100%" height={HEIGHT} viewBox={`0 0 ${WIDTH} ${HEIGHT}`}>
      {[0, 0.5, 1].map((position) => {
        const y = PAD.top + position * (HEIGHT - PAD.top - PAD.bottom);
        return <line key={position} x1={PAD.left} x2={WIDTH - PAD.right} y1={y} y2={y} stroke={RULE} strokeWidth={1} />;
      })}
      {children}
      <text x={PAD.left} y={HEIGHT - 15} fill={DIM} fontFamily={MONO} fontSize={16}>step 0</text>
      <text x={WIDTH - PAD.right} y={HEIGHT - 15} fill={DIM} fontFamily={MONO} fontSize={16} textAnchor="end">step 261</text>
    </svg>
  </section>
);

export const RetainedTrainingScene: React.FC<{revealEndFrame?: number}> = ({revealEndFrame = 500}) => {
  const frame = useCurrentFrame();
  const progress = interpolate(frame, [Math.min(24, revealEndFrame - 1), revealEndFrame], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.inOut(Easing.cubic),
  });
  const visibleStep = progress * LAST_STEP;
  const visible = RETAINED_TRAINING.filter((point) => point.step <= visibleStep);
  const latest = visible.length > 0 ? visible[visible.length - 1] : RETAINED_TRAINING[0];
  const lossPath = pathFor(visible.map((point) => ({x: xFor(point.step), y: yForLoss(point.loss)})));
  const accuracyPath = pathFor(visible.map((point) => ({x: xFor(point.step), y: yForAccuracy(point.p3)})));

  return (
    <SubmissionFrame
      eyebrow="REAL IDLE-TIME TRAINING · 262 UNSMOOTHED POINTS"
      title="Idle GPU time becomes workload-specific learning"
      provenance={`run ${RETAINED_RUN.runId.slice(0, 8)} · stdout ${RETAINED_RUN.trainingLogSha} · events ${RETAINED_RUN.eventsSha}`}
      caption="In one retained three-epoch run, draft training loss falls 9.46 → 1.42 and held-out position-3 accuracy reaches 75.5%. Training lasts 06:23.6."
    >
      <div style={{width: "100%", display: "flex", flexDirection: "column", minHeight: 0}}>
        <div style={{display: "flex", alignItems: "stretch", gap: 16, marginBottom: 16}}>
          <div style={{flex: 1, background: PANEL, border: `1px solid ${RULE}`, borderRadius: 13, padding: "12px 17px"}}>
            <div style={{color: DIM, fontSize: 13, fontWeight: 760}}>REAL TRAINING LOG</div>
            <div style={{color: TEXT, fontSize: 18, marginTop: 7}}>global_step=0 · train/loss=9.461 → global_step=261 · train/loss=1.416</div>
          </div>
          <div style={{width: 470, background: PANEL, border: `1px solid ${SERIES_GREEN}`, borderRadius: 13, padding: "12px 17px"}}>
            <div style={{color: DIM, fontSize: 13, fontWeight: 760}}>FINAL HELD-OUT CHECK</div>
            <div style={{color: SERIES_GREEN, fontSize: 18, marginTop: 7}}>val/full_acc_2_epoch=0.755 · TRAINING 06:23.6</div>
          </div>
        </div>
        <div style={{display: "flex", flex: 1, minHeight: 0, gap: 20}}>
          <Plot title="Draft training loss · lower is better" endLabel={`${latest.loss.toFixed(3)} at step ${latest.step}`}>
            <path d={lossPath} fill="none" stroke={SERIES_BLUE} strokeWidth={4} strokeLinecap="round" strokeLinejoin="round" />
            <circle cx={xFor(latest.step)} cy={yForLoss(latest.loss)} r={7} fill={SERIES_BLUE} />
            <text x={PAD.left - 13} y={PAD.top + 5} fill={DIM} fontFamily={MONO} fontSize={16} textAnchor="end">10</text>
            <text x={PAD.left - 13} y={HEIGHT - PAD.bottom + 5} fill={DIM} fontFamily={MONO} fontSize={16} textAnchor="end">0</text>
          </Plot>
          <Plot title="Draft accuracy · higher is better" endLabel={`${(latest.p3 * 100).toFixed(1)}% train · 75.5% held-out`}>
            <path d={accuracyPath} fill="none" stroke={SERIES_GREEN} strokeWidth={4} strokeLinecap="round" strokeLinejoin="round" />
            <line x1={PAD.left} x2={WIDTH - PAD.right} y1={yForAccuracy(RETAINED_RUN.finalValidationP3)} y2={yForAccuracy(RETAINED_RUN.finalValidationP3)} stroke={TEXT} strokeWidth={2} strokeDasharray="8 7" opacity={0.65} />
            <circle cx={xFor(latest.step)} cy={yForAccuracy(latest.p3)} r={7} fill={SERIES_GREEN} />
            <text x={WIDTH - PAD.right - 4} y={yForAccuracy(RETAINED_RUN.finalValidationP3) - 9} fill={TEXT} fontFamily={MONO} fontSize={15} textAnchor="end">held-out 75.5%</text>
            <text x={PAD.left - 13} y={PAD.top + 5} fill={DIM} fontFamily={MONO} fontSize={16} textAnchor="end">100%</text>
            <text x={PAD.left - 13} y={HEIGHT - PAD.bottom + 5} fill={DIM} fontFamily={MONO} fontSize={16} textAnchor="end">0%</text>
          </Plot>
        </div>
      </div>
    </SubmissionFrame>
  );
};
