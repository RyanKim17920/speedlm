import React from "react";
import {Video} from "@remotion/media";
import {Sequence, staticFile, useCurrentFrame} from "remotion";
import {PANEL, RULE, SERIES_BLUE, SERIES_GREEN, TEXT} from "../theme";
import {EvidencePill, SubmissionFrame} from "./SubmissionFrame";

const ClientPane: React.FC<{accent: string; label: string; src: string}> = ({accent, label, src}) => {
  const frame = useCurrentFrame();
  const held = frame >= 240;
  return (
    <section style={{flex: 1, minWidth: 0, display: "flex", flexDirection: "column"}}>
      <div style={{color: accent, fontSize: 23, fontWeight: 800, margin: "0 0 10px 4px"}}>{label}</div>
      <div style={{position: "relative", flex: 1, minHeight: 0, overflow: "hidden", background: PANEL, border: `1px solid ${RULE}`, borderRadius: 16}}>
        {held ? (
          <Video
            src={staticFile(src)}
            trimBefore={120}
            playbackRate={0.02}
            muted
            style={{width: "100%", height: "100%", objectFit: "cover", scale: 1.18, filter: "grayscale(1) contrast(1.08)"}}
          />
        ) : (
          <Video
            src={staticFile(src)}
            muted
            style={{width: "100%", height: "100%", objectFit: "cover", scale: 1.18, filter: "grayscale(1) contrast(1.08)"}}
          />
        )}
      </div>
    </section>
  );
};

export const CurlProofScene: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <SubmissionFrame
      eyebrow="REAL CURL REQUESTS · MATCHING RECORDED REPLIES"
      title="Same client API. One matching recorded reply."
      provenance="request + both response JSON files retained · equality asserted by VERIFIED.json"
      caption="Same greedy request, same recorded reply, same 30 prompt and 10 completion tokens—through the OpenAI-compatible API."
    >
      <div style={{display: "flex", width: "100%", gap: 24}}>
        <ClientPane accent={SERIES_BLUE} label="Normal vLLM · recorded reply" src="pty-normal-client.mp4" />
        <ClientPane accent={SERIES_GREEN} label="SpeedLM · recorded reply" src="pty-speedlm-client.mp4" />
      </div>
      {frame >= 240 ? (
        <div
          style={{
            position: "absolute",
            left: 250,
            right: 250,
            top: 700,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 18,
            padding: "18px 24px",
            background: PANEL,
            border: `2px solid ${SERIES_GREEN}`,
            borderRadius: 16,
            color: TEXT,
            fontSize: 27,
            fontWeight: 760,
          }}
        >
          <span>MATCHING RECORDED REPLY · same OpenAI-compatible serving API</span>
          <EvidencePill green>30 prompt · 10 completion</EvidencePill>
        </div>
      ) : null}
    </SubmissionFrame>
  );
};
