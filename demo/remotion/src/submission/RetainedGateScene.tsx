import React from "react";
import {Easing, interpolate, useCurrentFrame} from "remotion";
import {DIM, PANEL, RULE, SERIES_BLUE, SERIES_GREEN, TEXT} from "../theme";
import {EvidencePill, SubmissionFrame} from "./SubmissionFrame";
import {RETAINED_RUN} from "./retainedRunData";

const Arm: React.FC<{accent: string; label: string; note: string}> = ({accent, label, note}) => (
  <div style={{flex: 1, background: PANEL, border: `1px solid ${accent}`, borderRadius: 15, padding: "22px 24px"}}>
    <div style={{color: accent, fontSize: 22, fontWeight: 820}}>{label}</div>
    <div style={{color: TEXT, fontSize: 24, fontWeight: 720, marginTop: 11}}>{note}</div>
  </div>
);

const ResultBar: React.FC<{accent: string; label: string; scale: number; value: string}> = ({accent, label, scale, value}) => (
  <div style={{marginTop: 22}}>
    <div style={{display: "flex", justifyContent: "space-between", color: TEXT, fontSize: 20}}>
      <span>{label}</span><span style={{fontWeight: 800}}>{value}</span>
    </div>
    <div style={{height: 18, background: RULE, borderRadius: 999, overflow: "hidden", marginTop: 9}}>
      <div style={{height: "100%", width: "100%", background: accent, borderRadius: 999, scale: `${scale} 1`, transformOrigin: "left center"}} />
    </div>
  </div>
);

export const RetainedGateScene: React.FC<{
  promotedFrame?: number;
  revealEndFrame?: number;
  revealStartFrame?: number;
}> = ({promotedFrame = 315, revealEndFrame = 340, revealStartFrame = 185}) => {
  const frame = useCurrentFrame();
  const reveal = interpolate(frame, [revealStartFrame, revealEndFrame], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  return (
    <SubmissionFrame
      eyebrow="AUTOMATIC SAFETY GATE · HELD-OUT SESSIONS"
      title="Only a gated candidate can replace your draft"
      provenance={`decision ${RETAINED_RUN.decisionSha} · run ${RETAINED_RUN.runId.slice(0, 8)} · final_verdict=promote`}
      caption="SpeedLM compares the current and tuned drafts on held-out sessions. This separate retained 125×8 gate promoted +13.4% accepted length."
    >
      <div style={{width: "100%", display: "grid", gridTemplateColumns: "1.05fr 0.95fr", gap: 26}}>
        <section style={{display: "flex", flexDirection: "column", background: PANEL, border: `1px solid ${RULE}`, borderRadius: 18, padding: "26px"}}>
          <div style={{display: "flex", alignItems: "center", justifyContent: "space-between"}}>
            <div style={{color: TEXT, fontSize: 27, fontWeight: 780}}>Real requests kept out of training</div>
            <EvidencePill>{RETAINED_RUN.gate.contexts} contexts × {RETAINED_RUN.gate.repeats} repeats</EvidencePill>
          </div>
          <div style={{color: DIM, fontSize: 19, lineHeight: 1.4, marginTop: 16}}>Whole sessions stay on one side of the train/evaluation split.</div>
          <div style={{display: "flex", gap: 18, marginTop: 30}}>
            <Arm accent={SERIES_BLUE} label="CURRENT DRAFT" note="Already serving" />
            <Arm accent={SERIES_GREEN} label="TUNED DRAFT" note="Not live yet" />
          </div>
          <div style={{color: SERIES_GREEN, fontSize: 25, fontWeight: 780, marginTop: 28}}>PASS → promote all at once</div>
          <div style={{color: DIM, fontSize: 19, marginTop: 12}}>FAIL → keep the current draft; partial weights never serve.</div>
        </section>
        <section style={{background: PANEL, border: `1px solid ${SERIES_GREEN}`, borderRadius: 18, padding: "26px 30px", opacity: 0.35 + reveal * 0.65}}>
          <div style={{display: "flex", justifyContent: "space-between", alignItems: "center"}}>
            <div>
              <div style={{color: DIM, fontSize: 16, fontWeight: 800}}>SEPARATE RETAINED RUN</div>
              <div style={{color: SERIES_GREEN, fontSize: 46, fontWeight: 860, marginTop: 8}}>+13.4%</div>
              <div style={{color: TEXT, fontSize: 21}}>accepted length</div>
            </div>
            {frame >= promotedFrame ? <EvidencePill green>PROMOTED</EvidencePill> : null}
          </div>
          <ResultBar accent={SERIES_BLUE} label="Stock draft" value="2.429" scale={reveal * (2.429364721351517 / 2.9)} />
          <ResultBar accent={SERIES_GREEN} label="Idle-tuned draft" value="2.755" scale={reveal * (2.7554151459116025 / 2.9)} />
          <div style={{height: 1, background: RULE, margin: "27px 0 20px"}} />
          <div style={{color: TEXT, fontSize: 19, lineHeight: 1.45}}>Same acceptance direction as the archived +15.0% gate, which did not promote because its timing channel was unstable.</div>
          <div style={{color: DIM, fontSize: 16, lineHeight: 1.4, marginTop: 16}}>Wall clock here: +8.9% ± 4.6% (noisy, so it is not the headline).</div>
        </section>
      </div>
    </SubmissionFrame>
  );
};
