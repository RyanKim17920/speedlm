import React from "react";
import { Video } from "@remotion/media";
import { AbsoluteFill, staticFile } from "remotion";
import { BG, MONO, PANEL, SERIES_GREEN, TEXT } from "../theme";

export const ResultScene: React.FC = () => (
  <AbsoluteFill style={{ background: BG, fontFamily: MONO }}>
    <Video
      src={staticFile("short-race.mp4")}
      trimBefore={660}
      durationInFrames={120}
      muted
      objectFit="contain"
      style={{ width: "100%", height: "100%" }}
    />
    <div
      style={{
        position: "absolute",
        left: 100,
        right: 118,
        top: 562,
        height: 70,
        border: `3px solid ${SERIES_GREEN}`,
        borderRadius: 12,
        boxShadow: "0 0 0 999px rgba(13,17,23,.10)",
      }}
    />
    <div
      style={{
        position: "absolute",
        top: 30,
        right: 52,
        padding: "10px 16px",
        borderRadius: 10,
        background: PANEL,
        color: TEXT,
        font: `700 18px ${MONO}`,
      }}
    >
      ARCHIVED GATE · NOT PROMOTED
    </div>
    <div
      style={{
        position: "absolute",
        left: 0,
        right: 0,
        top: 760,
        bottom: 0,
        background: BG,
      }}
    />
  </AbsoluteFill>
);
