import React from "react";
import { Video } from "@remotion/media";
import { AbsoluteFill, Sequence, staticFile } from "remotion";

export const CommandScene: React.FC = () => (
  <AbsoluteFill style={{ background: "#0d1117" }}>
    <Sequence name="Stock vLLM command" durationInFrames={85}>
      <Video
        src={staticFile("stock-command.mp4")}
        durationInFrames={85}
        muted
        objectFit="contain"
        style={{ width: "100%", height: "100%" }}
      />
    </Sequence>
    <Sequence name="SpeedLM command" from={85} durationInFrames={89}>
      <Video
        src={staticFile("speedlm-command.mp4")}
        durationInFrames={89}
        muted
        objectFit="contain"
        style={{ width: "100%", height: "100%" }}
      />
    </Sequence>
  </AbsoluteFill>
);
