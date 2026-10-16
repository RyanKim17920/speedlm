import React from "react";
import { AbsoluteFill, Easing, interpolate, useCurrentFrame } from "remotion";
import { BG, DIM, MONO, RULE, SERIES_GREEN, TEXT } from "../theme";
import evidence from "./trafficEvidence.json";

const stats = [
  ["707", "real sessions"],
  [evidence.summary.records.toLocaleString("en-US"), "completed requests"],
  [evidence.summary.completion_tokens.toLocaleString("en-US"), "predicted tokens"],
  [(evidence.summary.prompt_tokens + evidence.summary.completion_tokens).toLocaleString("en-US"), "total tokens"],
] as const;

const TRAFFIC_SAMPLES = evidence.examples;

export const TrafficStreamScene: React.FC = () => {
  const frame = useCurrentFrame();
  const sampleFrames = 96;
  const index = Math.min(TRAFFIC_SAMPLES.length - 1, Math.floor(frame / sampleFrames));
  const local = frame % sampleFrames;
  const sample = TRAFFIC_SAMPLES[index];
  const visibleChars = Math.floor(interpolate(local, [12, 90], [0, sample.output.length], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.linear,
  }));
  const revealed = sample.output.slice(0, visibleChars);
  const visibleOutput = revealed.slice(Math.max(0, revealed.length - 1160));

  return (
    <AbsoluteFill style={{ background: BG, padding: "42px 74px", fontFamily: MONO }}>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between" }}>
        <div style={{ color: TEXT, fontSize: 44, fontWeight: 740 }}>Your real traffic becomes local training data</div>
        <div style={{ color: DIM, fontSize: 15 }}>trace {sample.trace_id} · sha256 {sample.source_sha256.slice(0, 10)}…</div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 30, marginTop: 22, paddingBottom: 24, borderBottom: `1px solid ${RULE}` }}>
        {stats.map(([value, label]) => (
          <div key={label}>
            <div style={{ color: label === "predicted tokens" ? SERIES_GREEN : TEXT, fontSize: value.length > 7 ? 32 : 38, fontWeight: 780 }}>{value}</div>
            <div style={{ color: DIM, fontSize: 16, marginTop: 5 }}>{label}</div>
          </div>
        ))}
      </div>

      <main style={{ marginTop: 28, overflow: "hidden" }}>
        <div style={{ color: SERIES_GREEN, fontSize: 17, fontWeight: 760, marginBottom: 10 }}>INPUT</div>
        <div style={{ color: TEXT, fontSize: 21, lineHeight: 1.42, height: 160, overflow: "hidden" }}>{sample.input}</div>

        <div style={{ height: 1, background: RULE, margin: "22px 0" }} />
        <div style={{ color: SERIES_GREEN, fontSize: 17, fontWeight: 760, marginBottom: 10 }}>PREDICTED OUTPUT TOKENS</div>
        <pre style={{ margin: 0, height: 510, overflow: "hidden", whiteSpace: "pre-wrap", color: TEXT, font: `500 20px/1.4 ${MONO}` }}>
          {visibleOutput}{visibleChars < sample.output.length ? "▌" : ""}
        </pre>
      </main>
    </AbsoluteFill>
  );
};
