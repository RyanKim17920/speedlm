import React from "react";
import {Easing, interpolate, useCurrentFrame} from "remotion";
import {DIM, MONO, PANEL, RULE, SERIES_GREEN, TEXT} from "../theme";
import evidence from "../short/trafficEvidence.json";
import {SubmissionFrame} from "./SubmissionFrame";

const samples = [evidence.examples[0], evidence.examples[2]] as const;

export const TraceProofScene: React.FC<{sampleFrames?: number}> = ({sampleFrames = 330}) => {
  const frame = useCurrentFrame();
  const sampleIndex = frame < sampleFrames ? 0 : 1;
  const localFrame = frame < sampleFrames ? frame : frame - sampleFrames;
  const sample = samples[sampleIndex];
  const revealLimit = sampleIndex === 0 ? Math.min(sample.output.length, 2600) : sample.output.length;
  const visibleChars = Math.floor(interpolate(localFrame, [Math.min(18, sampleFrames - 2), Math.max(1, sampleFrames - 25)], [0, revealLimit], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.linear,
  }));
  const revealed = sample.output.slice(0, visibleChars);
  const visibleOutput = revealed.slice(Math.max(0, revealed.length - 1500));

  return (
    <SubmissionFrame
      eyebrow="THE REAL DATA THAT TRAINS YOUR DRAFT"
      title="Your completed inputs become workload-specific training"
      provenance={`trace ${sample.trace_id} · row sha ${sample.source_sha256.slice(0, 10)} · store sha 1d052dbe`}
      caption="Shown verbatim: completed input and model-predicted output, including reasoning and tool calls. Whole sessions never cross the train/evaluation split."
    >
      <div style={{width: "100%", display: "flex", flexDirection: "column", minHeight: 0}}>
        <div style={{display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16}}>
          {[
            ["707", "real sessions"],
            [evidence.summary.records.toLocaleString("en-US"), "completed requests"],
            [evidence.summary.completion_tokens.toLocaleString("en-US"), "predicted tokens"],
            [(evidence.summary.prompt_tokens + evidence.summary.completion_tokens).toLocaleString("en-US"), "total tokens"],
          ].map(([value, label]) => (
            <div key={label} style={{background: PANEL, border: `1px solid ${RULE}`, borderRadius: 12, padding: "11px 16px"}}>
              <div style={{color: label === "predicted tokens" ? SERIES_GREEN : TEXT, fontSize: 28, fontWeight: 820}}>{value}</div>
              <div style={{color: DIM, fontSize: 14, marginTop: 3}}>{label}</div>
            </div>
          ))}
        </div>
        <div style={{display: "grid", gridTemplateColumns: "0.8fr 1.4fr", gap: 18, flex: 1, minHeight: 0, marginTop: 16}}>
          <section style={{background: PANEL, border: `1px solid ${RULE}`, borderRadius: 14, padding: "18px 20px", overflow: "hidden"}}>
            <div style={{color: SERIES_GREEN, fontSize: 16, fontWeight: 820}}>INPUT</div>
            <div style={{color: TEXT, fontSize: 19, lineHeight: 1.42, marginTop: 10}}>{sample.input}</div>
          </section>
          <section style={{background: PANEL, border: `1px solid ${RULE}`, borderRadius: 14, padding: "18px 20px", overflow: "hidden"}}>
            <div style={{display: "flex", justifyContent: "space-between", alignItems: "baseline"}}>
              <div style={{color: SERIES_GREEN, fontSize: 16, fontWeight: 820}}>PREDICTED OUTPUT TOKENS</div>
              <div style={{color: DIM, fontSize: 14}}>{sampleIndex === 0 ? "reasoning included" : "tool call included"}</div>
            </div>
            <pre style={{height: 500, margin: "10px 0 0", overflow: "hidden", whiteSpace: "pre-wrap", color: TEXT, font: `500 18px/1.36 ${MONO}`}}>
              {visibleOutput}{visibleChars < revealLimit ? "▌" : ""}
            </pre>
          </section>
        </div>
      </div>
    </SubmissionFrame>
  );
};
