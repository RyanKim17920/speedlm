import React from "react";
import { Video } from "@remotion/media";
import { AbsoluteFill, Sequence, staticFile, useCurrentFrame } from "remotion";
import { BG, DIM, MONO, PANEL, RULE, SERIES_GREEN, TEXT } from "../theme";

const milestoneFor = (frame: number) => {
  if (frame < 90) return { clock: "LIVE", state: "SERVING" };
  if (frame < 112) return { clock: "00:00", state: "TRAFFIC STOPPED" };
  if (frame < 135) return { clock: "00:05", state: "QUIESCING" };
  if (frame < 165) return { clock: "00:39", state: "EXTRACTING" };
  return { clock: "06:30", state: "TRAINING" };
};

const revealBottomFor = (frame: number) => {
  if (frame < 112) return 370;
  if (frame < 135) return 430;
  if (frame < 165) return 455;
  return 810;
};

export const IdleWatchScene: React.FC = () => {
  const frame = useCurrentFrame();
  const milestone = milestoneFor(frame);
  const isServing = frame < 90;
  const revealBottom = revealBottomFor(frame);

  return (
    <AbsoluteFill style={{ background: BG, padding: "58px 80px", fontFamily: MONO }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 30 }}>
        <div>
          <div style={{ color: TEXT, fontSize: 52, fontWeight: 720 }}>
            {isServing ? "SpeedLM serves normal vLLM traffic." : "Traffic stops. SpeedLM waits."}
          </div>
          <div style={{ color: DIM, fontSize: 24, marginTop: 10 }}>
            {isServing ? "Real requests from the original GPU capture" : "Separately recorded idle-cycle log"}
          </div>
        </div>
        <div style={{ minWidth: 260, textAlign: "right" }}>
          <div style={{ color: SERIES_GREEN, fontSize: 44, fontWeight: 760 }}>{milestone.clock}</div>
          <div style={{ color: DIM, fontSize: 18, marginTop: 4 }}>recorded elapsed time</div>
          <div style={{ color: milestone.state === "TRAINING" ? SERIES_GREEN : TEXT, fontSize: 22, fontWeight: 700, marginTop: 8 }}>{milestone.state}</div>
        </div>
      </div>

      <div style={{ position: "relative", height: 810, border: `1px solid ${RULE}`, borderRadius: 20, overflow: "hidden", background: PANEL }}>
        <Sequence durationInFrames={90}>
          <Video
            src={staticFile("real-live-server.mp4")}
            trimBefore={540}
            durationInFrames={90}
            muted
            objectFit="contain"
            style={{ width: "100%", height: "100%" }}
          />
          {revealBottom < 810 ? (
            <div style={{ position: "absolute", left: 0, right: 0, top: revealBottom, bottom: 0, background: PANEL }} />
          ) : null}
        </Sequence>
        <Sequence from={90} durationInFrames={120}>
          <Video
            src={staticFile("real-idle-transition-clean.mp4")}
            durationInFrames={120}
            playbackRate={0.375}
            muted
            objectFit="contain"
            style={{ width: "100%", height: "100%" }}
          />
        </Sequence>
      </div>
    </AbsoluteFill>
  );
};
