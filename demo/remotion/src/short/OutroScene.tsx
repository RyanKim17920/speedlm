import React from "react";
import { AbsoluteFill, Easing, interpolate, useCurrentFrame } from "remotion";
import { BG, MONO, SERIES_GREEN, TEXT } from "../theme";

export const OutroScene: React.FC = () => {
  const frame = useCurrentFrame();

  return (
    <AbsoluteFill
      style={{
        background: BG,
        padding: "110px",
        justifyContent: "center",
        alignItems: "center",
        textAlign: "center",
      }}
    >
      <div
        style={{
          color: TEXT,
          font: `700 86px ${MONO}`,
          lineHeight: 1.16,
          maxWidth: 1540,
          opacity: interpolate(frame, [0, 18], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.bezier(0.16, 1, 0.3, 1),
          }),
        }}
      >
        Automatically speed up local LLMs.
      </div>
      <div style={{ color: SERIES_GREEN, font: `600 38px ${MONO}`, marginTop: 52 }}>
        idle GPU time · same OpenAI-compatible serving API
      </div>
    </AbsoluteFill>
  );
};
