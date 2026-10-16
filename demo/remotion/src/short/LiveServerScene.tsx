import React from "react";
import { Video } from "@remotion/media";
import { AbsoluteFill, Sequence, staticFile } from "remotion";
import { BG, MONO, RULE, SERIES_GREEN, TEXT } from "../theme";

export const LiveServerScene: React.FC = () => (
  <AbsoluteFill style={{ background: BG, padding: "58px 80px", fontFamily: MONO }}>
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 30 }}>
      <div style={{ color: TEXT, fontSize: 50, fontWeight: 720 }}>Then it serves real requests</div>
      <div style={{ color: SERIES_GREEN, fontSize: 23 }}>original GPU capture</div>
    </div>
    <div style={{ height: 820, border: `1px solid ${RULE}`, borderRadius: 18, overflow: "hidden", position: "relative" }}>
      <Sequence durationInFrames={75}>
        <Video
          src={staticFile("real-live-server.mp4")}
          durationInFrames={75}
          playbackRate={0.8}
          muted
          objectFit="contain"
          style={{ width: "100%", height: "100%" }}
        />
      </Sequence>
      <Sequence from={75} durationInFrames={135}>
        <Video
          src={staticFile("real-live-server.mp4")}
          trimBefore={540}
          durationInFrames={135}
          playbackRate={0.67}
          muted
          objectFit="contain"
          style={{ width: "100%", height: "100%" }}
        />
      </Sequence>
    </div>
  </AbsoluteFill>
);
