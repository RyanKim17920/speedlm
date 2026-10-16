import React from "react";
import {AbsoluteFill} from "remotion";
import {BG, DIM, MONO, PANEL, RULE, SERIES_GREEN, TEXT} from "../theme";

const Step: React.FC<{detail: string; index: string; title: string}> = ({detail, index, title}) => (
  <div style={{flex: 1, minWidth: 0, background: PANEL, border: `1px solid ${RULE}`, borderRadius: 20, padding: "32px 28px"}}>
    <div style={{color: SERIES_GREEN, fontSize: 23, fontWeight: 800}}>{index}</div>
    <div style={{color: TEXT, fontSize: 31, fontWeight: 760, marginTop: 22}}>{title}</div>
    <div style={{color: DIM, fontSize: 21, lineHeight: 1.45, marginTop: 18}}>{detail}</div>
  </div>
);

export const WorkflowScene: React.FC = () => (
  <AbsoluteFill style={{background: BG, padding: "104px 84px", fontFamily: MONO, justifyContent: "center"}}>
    <div style={{color: TEXT, fontSize: 66, fontWeight: 780, letterSpacing: -3}}>One guarded lifecycle.</div>
    <div style={{color: DIM, fontSize: 27, marginTop: 16}}>No separate offline serving workflow.</div>
    <div style={{display: "flex", gap: 20, marginTop: 60}}>
      <Step index="01" title="CAPTURE" detail="Pair completed inputs and outputs while the real vLLM endpoint serves." />
      <Step index="02" title="WAIT" detail="Confirm the default five-minute quiet window before closing admission." />
      <Step index="03" title="TRAIN" detail="Warm-start a disposable EAGLE-3 draft outside the live verifier." />
      <Step index="04" title="GATE" detail="Promote only after a session-disjoint incumbent-versus-candidate replay." />
    </div>
    <div style={{color: SERIES_GREEN, fontSize: 30, fontWeight: 760, marginTop: 46}}>
      failed candidate → automatic rollback
    </div>
  </AbsoluteFill>
);
