import React from "react";
import {Video} from "@remotion/media";
import {Img, Sequence, staticFile} from "remotion";
import {DIM, PANEL, RULE, SERIES_BLUE, SERIES_GREEN, TEXT} from "../theme";
import {SubmissionFrame} from "./SubmissionFrame";

const Pane: React.FC<{accent: string; children: React.ReactNode; label: string}> = ({accent, children, label}) => (
  <section style={{flex: 1, minWidth: 0, display: "flex", flexDirection: "column"}}>
    <div style={{color: accent, fontSize: 24, fontWeight: 800, margin: "0 0 10px 4px"}}>{label}</div>
    <div style={{position: "relative", flex: 1, minHeight: 0, overflow: "hidden", background: PANEL, border: `1px solid ${RULE}`, borderRadius: 16}}>
      {children}
    </div>
  </section>
);

const serverStyle: React.CSSProperties = {
  width: "100%",
  height: "100%",
  objectFit: "contain",
  objectPosition: "left center",
  scale: 1.12,
  transformOrigin: "left center",
  filter: "grayscale(1) contrast(1.08)",
};

export const LaunchScene: React.FC = () => (
  <SubmissionFrame
    eyebrow="REAL COMMANDS · REAL SERVER LOGS"
    title="Keep your vLLM workflow. Add one flag."
    provenance="two NVIDIA H100s · verified at commit d1acee5 · manifest b2ee2108"
    caption="Set up the trainer once, then add --enable-idle-tuning. Unconsumed vLLM arguments pass through."
  >
    <div style={{position: "relative", width: "100%"}}>
      <Sequence name="Readable complete commands" durationInFrames={180} premountFor={30}>
        <div style={{display: "flex", flexDirection: "column", width: "100%", height: "100%", gap: 28, justifyContent: "center"}}>
        <div>
          <div style={{color: SERIES_BLUE, fontSize: 23, fontWeight: 800, marginBottom: 10}}>YOUR CURRENT vLLM COMMAND</div>
          <div style={{height: 180, display: "flex", alignItems: "center", overflow: "hidden", background: PANEL, border: `1px solid ${RULE}`, borderRadius: 18, padding: "20px 30px"}}>
            <Img src={staticFile("pty-normal-command.png")} style={{width: "100%", height: "100%", objectFit: "contain", filter: "grayscale(1) contrast(1.08)"}} />
          </div>
        </div>
        <div>
          <div style={{color: SERIES_GREEN, fontSize: 23, fontWeight: 800, marginBottom: 10}}>SPEEDLM · FAMILIAR SERVING COMMAND</div>
          <div style={{height: 180, display: "flex", alignItems: "center", overflow: "hidden", background: PANEL, border: `1px solid ${SERIES_GREEN}`, borderRadius: 18, padding: "20px 30px"}}>
            <Img src={staticFile("pty-speedlm-command.png")} style={{width: "100%", height: "100%", objectFit: "contain", filter: "grayscale(1) contrast(1.08)"}} />
          </div>
        </div>
        <div style={{color: DIM, fontSize: 18}}>One-time trainer paths were configured in SPEEDLM_HOME before this real capture.</div>
        </div>
      </Sequence>
      <Sequence name="Both literal server PTYs start" from={180} durationInFrames={360} premountFor={45}>
        <div style={{display: "flex", width: "100%", height: "100%", gap: 24}}>
          <Pane accent={SERIES_BLUE} label="Normal vLLM · real server log">
            <Video src={staticFile("pty-normal-server.mp4")} trimBefore={6} muted style={serverStyle} />
          </Pane>
          <Pane accent={SERIES_GREEN} label="SpeedLM · real server log">
            <Video src={staticFile("pty-speedlm-server.mp4")} trimBefore={6} muted style={serverStyle} />
          </Pane>
        </div>
      </Sequence>
    </div>
  </SubmissionFrame>
);
