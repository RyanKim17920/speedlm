import React from "react";
import { Video } from "@remotion/media";
import { AbsoluteFill, staticFile } from "remotion";
import { BG, DIM, MONO, RULE, TEXT } from "../theme";

export const IdleTransitionScene: React.FC = () => (
  <AbsoluteFill style={{ background: BG, padding: "58px 80px", fontFamily: MONO }}>
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 30 }}>
      <div style={{ color: TEXT, fontSize: 50, fontWeight: 720 }}>When traffic stops, tuning starts</div>
      <div style={{ color: DIM, fontSize: 23 }}>recorded idle-cycle events</div>
    </div>
    <div style={{ position: "relative", height: 560, marginTop: 100, border: `1px solid ${RULE}`, borderRadius: 18, overflow: "hidden", background: "#0d1117" }}>
      <Video
        src={staticFile("real-idle-transition-crop.mp4")}
        durationInFrames={180}
        playbackRate={0.25}
        muted
        objectFit="contain"
        style={{ width: "100%", height: "100%" }}
      />
      <div style={{ position: "absolute", left: 0, right: 0, top: 0, height: 185, background: "#0d1117" }} />
      <div style={{ position: "absolute", left: 0, right: 0, bottom: 0, height: 190, background: "#0d1117" }} />
    </div>
  </AbsoluteFill>
);
