import React from "react";
import {AbsoluteFill, Sequence} from "remotion";
import {BG, MONO, PANEL, SERIES_GREEN, TEXT} from "../theme";
import {RaceScene} from "../short/RaceScene";
import {ResultScene} from "../short/ResultScene";

export const HookScene: React.FC = () => (
  <AbsoluteFill style={{background: BG, fontFamily: MONO}}>
    <Sequence name="Authentic stock versus tuned replay" durationInFrames={210} premountFor={30}>
      <RaceScene />
    </Sequence>
    <Sequence name="Archived replay result hold" from={210} durationInFrames={60} premountFor={30}>
      <ResultScene />
    </Sequence>
    <div
      style={{
        position: "absolute",
        left: 70,
        right: 70,
        bottom: 38,
        minHeight: 106,
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: 28,
        padding: "16px 24px",
        background: PANEL,
        borderRadius: 16,
      }}
    >
      <div>
        <div style={{color: SERIES_GREEN, fontSize: 44, fontWeight: 840}}>+15.0% accepted length</div>
        <div style={{color: TEXT, fontSize: 21, marginTop: 5}}>accepted draft tokens per verifier step · 287 contexts × 8 repeats</div>
      </div>
      <div style={{color: TEXT, fontSize: 20, lineHeight: 1.35, maxWidth: 660, textAlign: "right"}}>
        Acceptance improved, but unstable throughput blocked promotion. A separate promoted gate appears later.
      </div>
    </div>
  </AbsoluteFill>
);
