import React from "react";
import {Video} from "@remotion/media";
import {Sequence, staticFile, useCurrentFrame} from "remotion";
import {PANEL, RULE, SERIES_GREEN, TEXT} from "../theme";
import {SubmissionFrame} from "./SubmissionFrame";

const clipLabel = (frame: number, segmentFrames: number) => {
  if (frame < segmentFrames) return "source 02:00 / 31:55";
  if (frame < segmentFrames * 2) return "source 16:00 / 31:55";
  if (frame < segmentFrames * 3) return "source 30:00 / 31:55";
  return "real agent harness output";
};

const videoStyle: React.CSSProperties = {
  width: "100%",
  height: "100%",
  objectFit: "contain",
  background: "#1c2229",
  filter: "grayscale(1) contrast(1.06)",
};

export const TrafficProofScene: React.FC<{segmentFrames?: number}> = ({segmentFrames = 90}) => {
  const frame = useCurrentFrame();
  const harnessStart = segmentFrames * 3;
  return (
    <SubmissionFrame
      eyebrow="YOUR REAL WORKLOAD · AUTHENTIC 31:55 CAPTURE"
      title="It learns from traffic you already serve"
      provenance="server clip fc1cb445 · harness clip 9ca9a9e9 · all source audio muted"
      caption="This real coding-agent run produced 6,176 completed exchanges: 26.6M prompt tokens and 2.55M model-predicted tokens for local draft training."
    >
      <div style={{position: "relative", width: "100%", overflow: "hidden", border: `1px solid ${RULE}`, borderRadius: 18, background: PANEL}}>
        <Sequence name="Live server at 02:00" durationInFrames={segmentFrames} premountFor={20}>
          <Video src={staticFile("speedlm-vllm-serve-live.mp4")} trimBefore={3600} muted style={videoStyle} />
        </Sequence>
        <Sequence name="Live server at 16:00" from={segmentFrames} durationInFrames={segmentFrames} premountFor={20}>
          <Video src={staticFile("speedlm-vllm-serve-live.mp4")} trimBefore={28800} muted style={videoStyle} />
        </Sequence>
        <Sequence name="Live server at 30:00" from={segmentFrames * 2} durationInFrames={segmentFrames} premountFor={20}>
          <Video src={staticFile("speedlm-vllm-serve-live.mp4")} trimBefore={54000} muted style={videoStyle} />
        </Sequence>
        <Sequence name="Real agent harness results" from={harnessStart} durationInFrames={segmentFrames} premountFor={20}>
          <Video src={staticFile("traffic-results-live.mp4")} trimBefore={60} muted style={videoStyle} />
        </Sequence>
        <div style={{position: "absolute", top: 18, left: 18, display: "flex", gap: 12}}>
          <div style={{background: PANEL, border: `1px solid ${SERIES_GREEN}`, borderRadius: 999, padding: "9px 14px", color: SERIES_GREEN, fontSize: 17, fontWeight: 800}}>TIME-LAPSED</div>
          <div style={{background: PANEL, border: `1px solid ${RULE}`, borderRadius: 999, padding: "9px 14px", color: TEXT, fontSize: 17, fontWeight: 760}}>{clipLabel(frame, segmentFrames)}</div>
        </div>
        {frame >= harnessStart ? (
          <div style={{position: "absolute", left: 18, right: 18, bottom: 18, display: "flex", gap: 12, justifyContent: "center"}}>
            {[
              "6,176 exchanges",
              "26.6M prompt tokens",
              "2.55M predicted tokens",
            ].map((value) => (
              <div key={value} style={{background: PANEL, border: `1px solid ${RULE}`, borderRadius: 12, padding: "12px 18px", color: TEXT, fontSize: 20, fontWeight: 760}}>{value}</div>
            ))}
          </div>
        ) : null}
      </div>
    </SubmissionFrame>
  );
};
