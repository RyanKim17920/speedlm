import React from "react";
import { AbsoluteFill, Easing, interpolate, useCurrentFrame } from "remotion";
import { BG, DIM, MONO, TEXT } from "../theme";

export const TitleScene: React.FC = () => {
  const frame = useCurrentFrame();

  return (
    <AbsoluteFill
      style={{
        background: BG,
        padding: "120px 110px",
        justifyContent: "center",
      }}
    >
      <div
        style={{
          color: TEXT,
          font: `700 92px ${MONO}`,
          letterSpacing: -4,
          opacity: interpolate(frame, [0, 18], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.bezier(0.16, 1, 0.3, 1),
          }),
          translate: interpolate(frame, [0, 18], ["0px 28px", "0px 0px"], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.bezier(0.16, 1, 0.3, 1),
          }),
        }}
      >
        SpeedLM vs stock
      </div>
      <div
        style={{
          color: "#3fb950",
          font: `700 76px ${MONO}`,
          marginTop: 36,
          opacity: interpolate(frame, [12, 32], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.bezier(0.16, 1, 0.3, 1),
          }),
        }}
      >
        +15% accepted per verifier step
      </div>
      <div
        style={{
          color: TEXT,
          font: `500 46px ${MONO}`,
          marginTop: 28,
          opacity: interpolate(frame, [26, 46], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.bezier(0.16, 1, 0.3, 1),
          }),
        }}
      >
        49 matching recorded outputs. 11% sooner in this replay.
      </div>
      <div
        style={{
          color: DIM,
          font: `400 28px ${MONO}`,
          marginTop: 54,
          maxWidth: 1500,
          lineHeight: 1.45,
          opacity: interpolate(frame, [42, 62], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        Archived replay. The gate measured the drafting gain but rejected promotion because
        its wall-clock throughput channel was non-stationary.
      </div>
    </AbsoluteFill>
  );
};
