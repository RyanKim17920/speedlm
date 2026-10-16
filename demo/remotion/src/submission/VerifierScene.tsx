import React from "react";
import {Video} from "@remotion/media";
import {AbsoluteFill, Easing, interpolate, staticFile, useCurrentFrame} from "remotion";
import {BG, DIM, MONO, PANEL, RULE, SERIES_GREEN, TEXT} from "../theme";
import {SubmissionFrame} from "./SubmissionFrame";

const Step: React.FC<{accent?: boolean; label: string; note: string}> = ({accent = false, label, note}) => (
  <div style={{flex: 1, background: PANEL, border: `1px solid ${accent ? SERIES_GREEN : RULE}`, borderRadius: 18, padding: "28px 30px"}}>
    <div style={{color: accent ? SERIES_GREEN : TEXT, fontSize: 30, fontWeight: 800}}>{label}</div>
    <div style={{color: DIM, fontSize: 20, lineHeight: 1.4, marginTop: 16}}>{note}</div>
  </div>
);

export const VerifierScene: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <SubmissionFrame
      eyebrow="YOUR TARGET MODEL STAYS IN CONTROL"
      title="Better-matched draft. Same target-model verification."
      provenance="real Qwen3-8B server stdout beneath · source: real-live-server.mp4"
      caption="SpeedLM changes only the disposable draft. Qwen3-8B still verifies every emitted token, and clients keep the OpenAI-compatible serving contract."
    >
      <AbsoluteFill style={{top: 128, bottom: 160, opacity: 0.13, overflow: "hidden"}}>
        <Video src={staticFile("real-live-server.mp4")} muted loop objectFit="cover" style={{width: "100%", height: "100%"}} />
      </AbsoluteFill>
      <div
        style={{
          position: "relative",
          zIndex: 2,
          width: "100%",
          display: "flex",
          alignItems: "center",
          gap: 22,
          opacity: interpolate(frame, [8, 28], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.bezier(0.16, 1, 0.3, 1),
          }),
        }}
      >
        <Step label="1 · SAME CLIENT REQUEST" note="Your existing OpenAI-compatible request reaches the target model." />
        <div style={{color: SERIES_GREEN, fontSize: 42}}>→</div>
        <Step accent label="2 · WORKLOAD-TUNED DRAFT" note="A small EAGLE-3 head suggests several likely next tokens at once." />
        <div style={{color: SERIES_GREEN, fontSize: 42}}>→</div>
        <Step label="3 · TARGET VERIFIES" note="Qwen3-8B accepts valid suggestions and produces the final response." />
      </div>
    </SubmissionFrame>
  );
};
