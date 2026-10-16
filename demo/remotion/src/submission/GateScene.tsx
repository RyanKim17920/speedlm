import React from "react";
import {AbsoluteFill} from "remotion";
import {BG, DIM, MONO, PANEL, RULE, SERIES_BLUE, SERIES_GREEN, TEXT} from "../theme";

export const GateScene: React.FC = () => (
  <AbsoluteFill style={{background: BG, padding: "96px 96px", fontFamily: MONO, justifyContent: "center"}}>
    <div style={{color: TEXT, fontSize: 66, fontWeight: 780, letterSpacing: -3}}>Training does not equal promotion.</div>
    <div style={{color: DIM, fontSize: 27, marginTop: 16}}>Whole sessions stay on one side of the train / evaluation boundary.</div>
    <div style={{display: "flex", gap: 28, marginTop: 58}}>
      <div style={{flex: 1, background: PANEL, border: `1px solid ${RULE}`, borderRadius: 20, padding: "36px"}}>
        <div style={{color: SERIES_BLUE, fontSize: 25, fontWeight: 780}}>INCUMBENT</div>
        <div style={{color: TEXT, fontSize: 38, fontWeight: 760, marginTop: 24}}>Known serving state</div>
        <div style={{color: DIM, fontSize: 22, lineHeight: 1.5, marginTop: 18}}>Frozen held-out traffic is replayed repeatedly against the current draft.</div>
      </div>
      <div style={{flex: 1, background: PANEL, border: `1px solid ${RULE}`, borderRadius: 20, padding: "36px"}}>
        <div style={{color: SERIES_GREEN, fontSize: 25, fontWeight: 780}}>CANDIDATE</div>
        <div style={{color: TEXT, fontSize: 38, fontWeight: 760, marginTop: 24}}>Immutable new artifact</div>
        <div style={{color: DIM, fontSize: 22, lineHeight: 1.5, marginTop: 18}}>The same contexts measure accepted length and guard against a throughput regression.</div>
      </div>
    </div>
    <div style={{display: "flex", gap: 24, marginTop: 32}}>
      <div style={{flex: 1, color: SERIES_GREEN, background: PANEL, border: `1px solid ${SERIES_GREEN}`, borderRadius: 16, padding: "24px 28px", fontSize: 28, fontWeight: 760}}>
        PASS → atomic promotion
      </div>
      <div style={{flex: 1, color: TEXT, background: PANEL, border: `1px solid ${RULE}`, borderRadius: 16, padding: "24px 28px", fontSize: 28, fontWeight: 760}}>
        FAIL OR NOISY → restore incumbent
      </div>
    </div>
  </AbsoluteFill>
);
