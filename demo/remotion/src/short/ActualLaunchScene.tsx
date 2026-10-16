import React from "react";
import { Video } from "@remotion/media";
import { AbsoluteFill, staticFile } from "remotion";
import { BG, DIM, MONO, PANEL, RULE, SERIES_BLUE, SERIES_GREEN, TEXT } from "../theme";

const CaptureBand: React.FC<{
  accent: string;
  label: string;
  playbackRate: number;
  src: string;
}> = ({ accent, label, playbackRate, src }) => (
  <section style={{ height: 315, border: `1px solid ${RULE}`, borderRadius: 18, overflow: "hidden", background: PANEL }}>
    <div style={{ height: 52, display: "flex", alignItems: "center", padding: "0 20px", color: accent, fontSize: 23, fontWeight: 700, borderBottom: `1px solid ${RULE}` }}>
      {label}
    </div>
    <div style={{ position: "relative", height: 263 }}>
      <Video
        src={staticFile(src)}
        durationInFrames={210}
        playbackRate={playbackRate}
        muted
        loop
        objectFit="contain"
        style={{ width: "100%", height: "100%" }}
      />
      <div style={{ position: "absolute", left: 0, right: 0, top: 0, height: 70, background: PANEL }} />
    </div>
  </section>
);

export const ActualLaunchScene: React.FC = () => (
  <AbsoluteFill style={{ background: BG, padding: "58px 80px", fontFamily: MONO }}>
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 28 }}>
      <div style={{ color: TEXT, fontSize: 50, fontWeight: 720 }}>The real launch commands</div>
      <div style={{ color: DIM, fontSize: 23 }}>original terminal capture</div>
    </div>
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      <CaptureBand accent={SERIES_BLUE} label="stock vLLM" src="real-stock-command-crop.mp4" playbackRate={0.28} />
      <CaptureBand accent={SERIES_GREEN} label="SpeedLM" src="real-speedlm-command-crop.mp4" playbackRate={0.28} />
    </div>
  </AbsoluteFill>
);
