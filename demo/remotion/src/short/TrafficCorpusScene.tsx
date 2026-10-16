import React from "react";
import { Video } from "@remotion/media";
import { AbsoluteFill, Sequence, staticFile } from "remotion";
import { BG } from "../theme";

export const TrafficCorpusScene: React.FC = () => (
  <AbsoluteFill style={{ background: BG }}>
    <Sequence durationInFrames={300}>
      <Video
        src={staticFile("traffic-montage-v3.mp4")}
        trimBefore={78}
        durationInFrames={300}
        playbackRate={3.05}
        muted
        objectFit="contain"
        style={{ width: "100%", height: "100%" }}
      />
    </Sequence>
    <Sequence from={300} durationInFrames={60}>
      <Video
        src={staticFile("traffic-montage-v3.mp4")}
        trimBefore={1001}
        durationInFrames={60}
        playbackRate={0.1}
        muted
        objectFit="contain"
        style={{ width: "100%", height: "100%" }}
      />
    </Sequence>
  </AbsoluteFill>
);
