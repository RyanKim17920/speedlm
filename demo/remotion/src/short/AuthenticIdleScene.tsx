import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { BG, DIM, MONO, PANEL, RULE, SERIES_BLUE, SERIES_GREEN, TEXT } from "../theme";
import { AUTHENTIC_CAPTURE } from "./authenticCapture";

const formatTime = (seconds: number) => {
  const rounded = Math.round(seconds);
  return `${String(Math.floor(rounded / 60)).padStart(2, "0")}:${String(rounded % 60).padStart(2, "0")}`;
};

const phaseFor = (frame: number) => {
  if (frame < 50) return { state: "READY", detail: "idle timer running", seconds: interpolate(frame, [0, 50], [0, AUTHENTIC_CAPTURE.idle.quiescingSeconds]) };
  if (frame < 95) return { state: "SLEEPING", detail: "serving engine released", seconds: interpolate(frame, [50, 95], [AUTHENTIC_CAPTURE.idle.quiescingSeconds, AUTHENTIC_CAPTURE.idle.extractingSeconds]) };
  if (frame < 185) return { state: "EXTRACTING", detail: "building training signals", seconds: interpolate(frame, [95, 185], [AUTHENTIC_CAPTURE.idle.extractingSeconds, AUTHENTIC_CAPTURE.idle.trainingSeconds]) };
  if (frame < 240) return { state: "TRAINING", detail: "trainer running", seconds: interpolate(frame, [185, 240], [AUTHENTIC_CAPTURE.idle.trainingSeconds, AUTHENTIC_CAPTURE.idle.firstOptimizerSeconds]) };
  return { state: "TRAINING", detail: `first optimizer step · loss ${AUTHENTIC_CAPTURE.idle.firstLoss}`, seconds: AUTHENTIC_CAPTURE.idle.firstOptimizerSeconds };
};

const StatusPanel: React.FC<{ accent: string; children: React.ReactNode; label: string; status: string }> = ({ accent, children, label, status }) => (
  <section style={{ flex: 1, height: 690, background: PANEL, border: `1px solid ${RULE}`, borderRadius: 22, overflow: "hidden" }}>
    <div style={{ height: 82, padding: "0 28px", display: "flex", alignItems: "center", justifyContent: "space-between", borderBottom: `1px solid ${RULE}` }}>
      <div style={{ color: accent, fontSize: 32, fontWeight: 760 }}>{label}</div>
      <div style={{ color: status === "TRAINING" ? SERIES_GREEN : TEXT, fontSize: 21, fontWeight: 740 }}>{status}</div>
    </div>
    {children}
  </section>
);

export const AuthenticIdleScene: React.FC = () => {
  const frame = useCurrentFrame();
  const phase = phaseFor(frame);
  const trainVisible = frame >= 185;

  return (
    <AbsoluteFill style={{ background: BG, padding: "58px 80px", fontFamily: MONO }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 34 }}>
        <div>
          <div style={{ color: TEXT, fontSize: 54, fontWeight: 740 }}>Incoming traffic stops.</div>
          <div style={{ color: DIM, fontSize: 23, marginTop: 10 }}>Both servers stay running. Only SpeedLM starts idle tuning.</div>
        </div>
        <div style={{ color: SERIES_GREEN, fontSize: 46, fontWeight: 780 }}>{formatTime(phase.seconds)}</div>
      </div>

      <div style={{ display: "flex", gap: 34 }}>
        <StatusPanel accent={SERIES_BLUE} label="Normal vLLM" status="IDLE">
          <div style={{ padding: "48px 34px" }}>
            <div style={{ color: DIM, fontSize: 19 }}>LIVE ENGINE METRICS</div>
            <div style={{ marginTop: 38, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
              {[['0.0', 'prompt tok/s'], ['0.0', 'generation tok/s'], ['0', 'running requests'], ['0', 'waiting requests']].map(([value, label]) => (
                <div key={label} style={{ padding: "24px", border: `1px solid ${RULE}`, borderRadius: 16 }}>
                  <div style={{ color: TEXT, fontSize: 46, fontWeight: 760 }}>{value}</div>
                  <div style={{ color: DIM, fontSize: 18, marginTop: 8 }}>{label}</div>
                </div>
              ))}
            </div>
            <div style={{ color: DIM, fontSize: 22, marginTop: 42 }}>Server remains ready. No training process starts.</div>
          </div>
        </StatusPanel>

        <StatusPanel accent={SERIES_GREEN} label="SpeedLM vLLM" status={phase.state}>
          <div style={{ padding: "48px 34px" }}>
            <div style={{ color: DIM, fontSize: 19 }}>RECORDED IDLE LIFECYCLE</div>
            <div style={{ marginTop: 42, display: "flex", flexDirection: "column", gap: 24 }}>
              {[['READY', 0], ['SLEEPING', 50], ['EXTRACTING', 95], ['TRAINING', 185]].map(([label, at]) => {
                const reached = frame >= Number(at);
                return (
                  <div key={label} style={{ display: "flex", alignItems: "center", gap: 18, opacity: reached ? 1 : 0.24 }}>
                    <div style={{ width: 18, height: 18, borderRadius: 999, background: reached ? SERIES_GREEN : RULE }} />
                    <div style={{ color: reached ? TEXT : DIM, fontSize: 28, fontWeight: 700 }}>{label}</div>
                  </div>
                );
              })}
            </div>
            <div style={{ color: trainVisible ? SERIES_GREEN : TEXT, fontSize: 24, fontWeight: 700, marginTop: 42 }}>{phase.detail}</div>
            {frame >= 108 && frame < 240 ? <div style={{ color: DIM, fontSize: 18, marginTop: 14 }}>time lapse · recorded wall clock preserved above</div> : null}
          </div>
        </StatusPanel>
      </div>
    </AbsoluteFill>
  );
};
