import React from "react";
import {AbsoluteFill, Easing, interpolate, useCurrentFrame} from "remotion";
import {BG, DIM, MONO, PANEL, RULE, SERIES_BLUE, SERIES_GREEN, TEXT} from "../theme";

export const MetricScene: React.FC = () => {
  const frame = useCurrentFrame();

  return (
    <AbsoluteFill
      style={{
        background: BG,
        padding: "108px 110px",
        fontFamily: MONO,
        justifyContent: "center",
      }}
    >
      <div
        style={{
          color: SERIES_GREEN,
          fontSize: 96,
          fontWeight: 820,
          letterSpacing: -4,
          opacity: interpolate(frame, [0, 20], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.bezier(0.16, 1, 0.3, 1),
          }),
          translate: interpolate(frame, [0, 20], ["0px 24px", "0px 0px"], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.bezier(0.16, 1, 0.3, 1),
          }),
        }}
      >
        +15.0%
      </div>
      <div style={{color: TEXT, fontSize: 52, fontWeight: 720, marginTop: 16}}>
        accepted tokens per verifier step
      </div>

      <div style={{display: "flex", gap: 28, marginTop: 58}}>
        <div style={{flex: 1, background: PANEL, border: `1px solid ${RULE}`, borderRadius: 18, padding: "30px 34px"}}>
          <div style={{color: SERIES_BLUE, fontSize: 24, fontWeight: 760}}>STOCK DRAFT</div>
          <div style={{color: TEXT, fontSize: 60, fontWeight: 800, marginTop: 12}}>2.3051</div>
        </div>
        <div style={{flex: 1, background: PANEL, border: `1px solid ${RULE}`, borderRadius: 18, padding: "30px 34px"}}>
          <div style={{color: SERIES_GREEN, fontSize: 24, fontWeight: 760}}>IDLE-TUNED DRAFT</div>
          <div style={{color: TEXT, fontSize: 60, fontWeight: 800, marginTop: 12}}>2.6507</div>
        </div>
        <div style={{width: 390, background: PANEL, border: `1px solid ${RULE}`, borderRadius: 18, padding: "30px 34px"}}>
          <div style={{color: DIM, fontSize: 24, fontWeight: 760}}>HELD-OUT GATE</div>
          <div style={{color: TEXT, fontSize: 48, fontWeight: 800, marginTop: 18}}>287 × 8</div>
          <div style={{color: DIM, fontSize: 20, marginTop: 8}}>contexts × repeats</div>
        </div>
      </div>

      <div style={{color: DIM, fontSize: 25, lineHeight: 1.5, marginTop: 44, maxWidth: 1560}}>
        Reproducible acceptance result. This gate did not promote because its wall-clock channel was non-stationary;
        it is not a universal “15% faster” claim.
      </div>
    </AbsoluteFill>
  );
};
