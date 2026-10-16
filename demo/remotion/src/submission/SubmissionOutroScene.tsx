import React from "react";
import {AbsoluteFill, Easing, interpolate, useCurrentFrame} from "remotion";
import {BG, DIM, MONO, SERIES_GREEN, TEXT} from "../theme";

export const SubmissionOutroScene: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill style={{background: BG, padding: "110px", alignItems: "center", justifyContent: "center", textAlign: "center", fontFamily: MONO}}>
      <div
        style={{
          color: TEXT,
          fontSize: 92,
          fontWeight: 800,
          letterSpacing: -4,
          opacity: interpolate(frame, [0, 20], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.bezier(0.16, 1, 0.3, 1),
          }),
        }}
      >
        SpeedLM
      </div>
      <div style={{color: SERIES_GREEN, fontSize: 44, fontWeight: 740, marginTop: 30}}>
        Workload-tuned drafting. Same OpenAI-compatible serving API.
      </div>
      <div style={{color: TEXT, fontSize: 31, marginTop: 66}}>github.com/RyanKim17920/speedlm</div>
      <div style={{color: DIM, fontSize: 22, lineHeight: 1.5, marginTop: 34, maxWidth: 1320}}>
        Research preview validated on an NVIDIA H100 stack. The repository documents supported profiles,
        prerequisites, benchmark evidence, and current limitations.
      </div>
    </AbsoluteFill>
  );
};
