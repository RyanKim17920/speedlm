import React from "react";
import { Video } from "@remotion/media";
import { AbsoluteFill, Easing, interpolate, Sequence, staticFile, useCurrentFrame } from "remotion";
import { BG, DIM, MONO, PANEL, RULE, SERIES_BLUE, SERIES_GREEN, TEXT } from "../theme";
import { RaceScene } from "./RaceScene";

const CommandPanel: React.FC<{ accent: string; label: string; src: string }> = ({ accent, label, src }) => (
  <section style={{ flex: 1, height: 650, border: `1px solid ${RULE}`, borderRadius: 20, overflow: "hidden", background: PANEL }}>
    <div style={{ height: 78, display: "flex", alignItems: "center", padding: "0 26px", color: accent, font: `700 30px ${MONO}`, borderBottom: `1px solid ${RULE}` }}>
      {label}
    </div>
    <div style={{ position: "relative", height: 572 }}>
      <Video
        src={staticFile(src)}
        durationInFrames={90}
        playbackRate={0.65}
        muted
        loop
        objectFit="contain"
        style={{ width: "100%", height: "100%" }}
      />
    </div>
  </section>
);

export const RunLiveScene: React.FC = () => {
  const frame = useCurrentFrame();

  return (
    <AbsoluteFill style={{ background: BG, fontFamily: MONO }}>
      <AbsoluteFill
        style={{
          padding: "58px 80px",
          opacity: interpolate(frame, [66, 82], [1, 0], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.inOut(Easing.cubic),
          }),
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 34 }}>
          <div style={{ color: TEXT, fontSize: 52, fontWeight: 720 }}>Same vLLM command. One extra flag.</div>
          <div style={{ color: DIM, fontSize: 23 }}>original terminal captures</div>
        </div>
        <div style={{ display: "flex", gap: 34 }}>
          <CommandPanel accent={SERIES_BLUE} label="Normal vLLM" src="real-stock-command-clean.mp4" />
          <CommandPanel accent={SERIES_GREEN} label="SpeedLM vLLM" src="real-speedlm-command-clean.mp4" />
        </div>
      </AbsoluteFill>

      <Sequence from={74} durationInFrames={226}>
        <AbsoluteFill
          style={{
            opacity: interpolate(frame, [74, 90], [0, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
              easing: Easing.inOut(Easing.cubic),
            }),
          }}
        >
          <RaceScene />
        </AbsoluteFill>
      </Sequence>
    </AbsoluteFill>
  );
};
