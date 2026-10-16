import React from "react";
import {AbsoluteFill} from "remotion";
import {BG, DIM, MONO, PANEL, RULE, SERIES_GREEN, TEXT} from "../theme";

const Node: React.FC<{accent?: boolean; label: string; note: string}> = ({accent = false, label, note}) => (
  <div style={{width: 280, minHeight: 190, background: PANEL, border: `1px solid ${accent ? SERIES_GREEN : RULE}`, borderRadius: 18, padding: "30px 26px"}}>
    <div style={{color: accent ? SERIES_GREEN : TEXT, fontSize: 29, fontWeight: 780}}>{label}</div>
    <div style={{color: DIM, fontSize: 19, lineHeight: 1.4, marginTop: 18}}>{note}</div>
  </div>
);

export const SafetyScene: React.FC = () => (
  <AbsoluteFill style={{background: BG, padding: "104px 84px", fontFamily: MONO, justifyContent: "center"}}>
    <div style={{color: TEXT, fontSize: 66, fontWeight: 780, letterSpacing: -3}}>What if traffic returns mid-training?</div>
    <div style={{display: "flex", alignItems: "center", justifyContent: "space-between", marginTop: 72}}>
      <Node label="REQUEST" note="The gateway sees new activity before it reaches the sleeping engine." />
      <div style={{color: RULE, fontSize: 46}}>→</div>
      <Node label="QUEUE" note="The request waits outside vLLM while the active stage is cancelled." />
      <div style={{color: RULE, fontSize: 46}}>→</div>
      <Node label="RESTORE" note="Workers stop and the durable incumbent is woken and verified." />
      <div style={{color: RULE, fontSize: 46}}>→</div>
      <Node accent label="SERVE" note="Admission reopens only against a ready, verified draft." />
    </div>
    <div style={{color: SERIES_GREEN, fontSize: 30, fontWeight: 760, marginTop: 54}}>
      never sleeping vLLM · never partial weights · never an unverified candidate
    </div>
    <div style={{color: DIM, fontSize: 22, marginTop: 24}}>
      Current limitation: the server-side wait is unbounded, so clients still need an appropriate timeout.
    </div>
  </AbsoluteFill>
);
