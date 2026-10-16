import React from "react";
import { Video } from "@remotion/media";
import { AbsoluteFill, Sequence, staticFile, useCurrentFrame } from "remotion";
import { BG, MONO, PANEL, SERIES_BLUE, SERIES_GREEN, TEXT } from "../theme";

const SimplifyingMasks: React.FC<{ identical: boolean }> = ({ identical }) => (
  <>
    <div style={{ position: "absolute", inset: "0 0 auto 0", height: 148, background: BG }} />
    <div style={{ position: "absolute", left: 40, top: 34, width: 900, color: SERIES_BLUE, font: `700 42px ${MONO}` }}>
      Normal vLLM
    </div>
    <div style={{ position: "absolute", left: 980, top: 34, width: 900, color: SERIES_GREEN, font: `700 42px ${MONO}` }}>
      SpeedLM vLLM
    </div>
    <div style={{ position: "absolute", left: 40, top: 94, color: TEXT, font: `500 21px ${MONO}` }}>
      same prompts · same final output
    </div>
    <div style={{ position: "absolute", right: 40, top: 90, color: SERIES_GREEN, font: `700 25px ${MONO}` }}>
      {identical ? "MATCHING OUTPUT" : "SAME PROMPTS"}
    </div>
    <div style={{ position: "absolute", right: 40, top: 34, color: TEXT, font: `700 17px ${MONO}` }}>
      ARCHIVED GATE · NOT PROMOTED
    </div>

    <div style={{ position: "absolute", left: 40, top: 155, width: 901, height: 70, background: PANEL }} />
    <div style={{ position: "absolute", left: 980, top: 155, width: 900, height: 70, background: PANEL }} />
    <div style={{ position: "absolute", left: 62, top: 172, color: SERIES_BLUE, font: `700 31px ${MONO}` }}>Normal vLLM</div>
    <div style={{ position: "absolute", left: 1002, top: 172, color: SERIES_GREEN, font: `700 31px ${MONO}` }}>SpeedLM vLLM</div>

    <div style={{ position: "absolute", left: 50, top: 224, width: 154, height: 54, background: PANEL }} />
    <div style={{ position: "absolute", left: 485, top: 224, width: 190, height: 54, background: PANEL }} />
    <div style={{ position: "absolute", left: 990, top: 224, width: 154, height: 54, background: PANEL }} />
    <div style={{ position: "absolute", left: 1425, top: 224, width: 190, height: 54, background: PANEL }} />
    <div style={{ position: "absolute", left: 40, top: 274, width: 901, height: 35, background: PANEL }} />
    <div style={{ position: "absolute", left: 980, top: 274, width: 900, height: 35, background: PANEL }} />
    <div style={{ position: "absolute", inset: "1005px 0 0 0", background: BG }} />
  </>
);

export const RaceScene: React.FC = () => {
  const frame = useCurrentFrame();

  return <AbsoluteFill style={{ background: BG }}>
    <Sequence durationInFrames={195}>
      <Video
        src={staticFile("short-race.mp4")}
        trimBefore={390}
        durationInFrames={195}
        playbackRate={15 / 13}
        muted
        objectFit="contain"
        style={{ width: "100%", height: "100%" }}
      />
    </Sequence>
    <Sequence from={195} durationInFrames={45}>
      <Video
        src={staticFile("short-race.mp4")}
        trimBefore={615}
        durationInFrames={45}
        playbackRate={0.1}
        muted
        objectFit="contain"
        style={{ width: "100%", height: "100%" }}
      />
    </Sequence>
    <SimplifyingMasks identical={frame >= 195} />
  </AbsoluteFill>
};
