import React from "react";
import {AbsoluteFill, Easing, interpolate, useCurrentFrame} from "remotion";
import {BG, DIM, MONO, PANEL, RULE, SERIES_BLUE, SERIES_GREEN, TEXT} from "../theme";

export const ProblemScene: React.FC = () => {
  const frame = useCurrentFrame();

  return (
    <AbsoluteFill style={{background: BG, padding: "116px 110px", fontFamily: MONO, justifyContent: "center"}}>
      <div style={{color: TEXT, fontSize: 72, fontWeight: 780, letterSpacing: -3, lineHeight: 1.12}}>
        Static drafts learn somebody else&apos;s traffic.
      </div>
      <div style={{display: "flex", gap: 34, marginTop: 64}}>
        <div
          style={{
            flex: 1,
            minHeight: 300,
            background: PANEL,
            border: `1px solid ${RULE}`,
            borderRadius: 22,
            padding: "42px",
            opacity: interpolate(frame, [18, 42], [0, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
              easing: Easing.bezier(0.16, 1, 0.3, 1),
            }),
          }}
        >
          <div style={{color: SERIES_BLUE, fontSize: 25, fontWeight: 780}}>NORMAL SPECULATIVE DECODING</div>
          <div style={{color: TEXT, fontSize: 37, fontWeight: 700, lineHeight: 1.32, marginTop: 28}}>
            One generic draft stays frozen as the workload changes.
          </div>
        </div>
        <div
          style={{
            flex: 1,
            minHeight: 300,
            background: PANEL,
            border: `1px solid ${SERIES_GREEN}`,
            borderRadius: 22,
            padding: "42px",
            opacity: interpolate(frame, [34, 58], [0, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
              easing: Easing.bezier(0.16, 1, 0.3, 1),
            }),
          }}
        >
          <div style={{color: SERIES_GREEN, fontSize: 25, fontWeight: 780}}>SPEEDLM</div>
          <div style={{color: TEXT, fontSize: 37, fontWeight: 700, lineHeight: 1.32, marginTop: 28}}>
            Completed traffic becomes a candidate draft during verified idle time.
          </div>
        </div>
      </div>
      <div style={{color: DIM, fontSize: 27, marginTop: 42}}>
        The target model still verifies every emitted token.
      </div>
    </AbsoluteFill>
  );
};
