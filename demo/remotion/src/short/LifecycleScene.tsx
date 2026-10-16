import React from "react";
import { Video } from "@remotion/media";
import { AbsoluteFill, staticFile } from "remotion";

export const LifecycleScene: React.FC = () => (
  <AbsoluteFill style={{ background: "#0d1117" }}>
    <Video
      src={staticFile("walkthrough.mp4")}
      trimBefore={217}
      durationInFrames={769}
      muted
      objectFit="contain"
      style={{ width: "100%", height: "100%" }}
    />
  </AbsoluteFill>
);
