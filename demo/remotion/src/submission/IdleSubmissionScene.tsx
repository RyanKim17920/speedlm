import React from "react";
import {AbsoluteFill, Sequence} from "remotion";
import {BG, DIM, MONO, PANEL, RULE, SERIES_BLUE, SERIES_GREEN, TEXT} from "../theme";
import {PtyIdleScene} from "../short/PtyIdleScene";
import {SubmissionFrame} from "./SubmissionFrame";

const Milestone: React.FC<{label: string; time: string}> = ({label, time}) => (
  <div style={{flex: 1, minWidth: 0, background: PANEL, border: `1px solid ${RULE}`, borderRadius: 16, padding: "24px 20px"}}>
    <div style={{color: DIM, fontSize: 17, fontWeight: 760}}>{label}</div>
    <div style={{color: label === "TUNING STARTS" ? SERIES_GREEN : TEXT, fontSize: 38, fontWeight: 840, marginTop: 13}}>{time}</div>
  </div>
);

const IdleSummary: React.FC = () => (
  <SubmissionFrame
    eyebrow="REAL LOGS · EXACT IDLE-TO-TRAINING CLOCK"
    title="When traffic stops, SpeedLM uses the quiet compute"
    provenance="job 394208 · demo threshold 5s · software default 5 min · waits time-lapsed"
    caption="Normal vLLM stays ready without starting a tuner. SpeedLM confirms quiet, releases serving GPU memory, and trains locally. Demo threshold 5s · software default 5 min."
  >
    <div style={{width: "100%", display: "flex", flexDirection: "column", justifyContent: "center"}}>
      <div style={{display: "flex", gap: 18}}>
        <Milestone label="REQUESTS STOP" time="00:00.0" />
        <Milestone label="QUIET CONFIRMED" time="+00:06.9" />
        <Milestone label="vLLM ASLEEP" time="+00:16.1" />
        <Milestone label="TUNING STARTS" time="+03:39.4" />
      </div>
      <div style={{display: "flex", gap: 22, marginTop: 30}}>
        <div style={{flex: 1, background: PANEL, border: `1px solid ${RULE}`, borderRadius: 16, padding: "25px 28px"}}>
          <div style={{color: SERIES_BLUE, fontSize: 24, fontWeight: 800}}>Normal vLLM</div>
          <div style={{color: TEXT, fontSize: 34, fontWeight: 780, marginTop: 13}}>READY · NO TUNING</div>
          <div style={{color: DIM, fontSize: 19, marginTop: 10}}>No workload tuner starts.</div>
        </div>
        <div style={{flex: 1, background: PANEL, border: `1px solid ${SERIES_GREEN}`, borderRadius: 16, padding: "25px 28px"}}>
          <div style={{color: SERIES_GREEN, fontSize: 24, fontWeight: 800}}>SpeedLM vLLM</div>
          <div style={{color: SERIES_GREEN, fontSize: 34, fontWeight: 780, marginTop: 13}}>TUNING FOR SPEED</div>
          <div style={{color: DIM, fontSize: 19, marginTop: 10}}>The draft learns from your completed traffic.</div>
        </div>
      </div>
    </div>
  </SubmissionFrame>
);

export const IdleSubmissionScene: React.FC = () => (
  <AbsoluteFill style={{background: BG, fontFamily: MONO}}>
    <Sequence name="Literal side-by-side idle lifecycle PTYs" durationInFrames={360} premountFor={45}>
      <PtyIdleScene />
      <div
        style={{
          position: "absolute",
          left: 58,
          right: 58,
          bottom: 20,
          minHeight: 78,
          display: "flex",
          alignItems: "center",
          padding: "12px 20px",
          background: PANEL,
          border: `1px solid ${RULE}`,
          borderRadius: 14,
          color: TEXT,
          fontSize: 21,
          fontWeight: 620,
        }}
      >
        <span style={{color: SERIES_GREEN, marginRight: 12}}>●</span>
        <span style={{flex: 1}}>The real logs show requests stop on both servers. Only SpeedLM starts tuning; its wait is time-lapsed.</span>
        <span style={{color: SERIES_GREEN, marginLeft: 24, whiteSpace: "nowrap"}}>demo threshold 5s · software default 5 min</span>
      </div>
    </Sequence>
    <Sequence name="Artifact-backed idle timing summary" from={360} durationInFrames={210} premountFor={30}>
      <IdleSummary />
    </Sequence>
  </AbsoluteFill>
);
