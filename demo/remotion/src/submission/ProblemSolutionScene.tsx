import React from "react";
import {Easing, interpolate, useCurrentFrame} from "remotion";
import {BG, DIM, PANEL, RULE, SERIES_BLUE, SERIES_GREEN, TEXT} from "../theme";
import {SubmissionFrame} from "./SubmissionFrame";

const columns = [
  {
    accent: DIM,
    title: "Hosted API",
    data: "Vendor-controlled",
    optimization: "Policy-dependent",
  },
  {
    accent: SERIES_BLUE,
    title: "Local LLM",
    data: "On your infrastructure",
    optimization: "Manual / static by default",
  },
  {
    accent: SERIES_GREEN,
    title: "Local LLM + SpeedLM",
    data: "On your infrastructure",
    optimization: "Automatic from completed traffic",
  },
] as const;

const Cell: React.FC<{accent: string; children: React.ReactNode; highlighted?: boolean}> = ({accent, children, highlighted = false}) => (
  <div
    style={{
      minWidth: 0,
      display: "flex",
      alignItems: "center",
      background: PANEL,
      border: `1px solid ${highlighted ? SERIES_GREEN : RULE}`,
      borderRadius: 14,
      padding: "17px 20px",
      color: accent,
      fontSize: 20,
      fontWeight: highlighted ? 760 : 650,
      lineHeight: 1.32,
    }}
  >
    {children}
  </div>
);

const IdleWindow: React.FC<{filled: boolean; frame: number}> = ({filled, frame}) => (
  <div style={{width: "100%"}}>
    <div style={{color: filled ? SERIES_GREEN : DIM, fontSize: 14, fontWeight: 800, letterSpacing: 0.8}}>
      WHEN TRAFFIC STOPS
    </div>
    <div
      style={{
        position: "relative",
        height: 54,
        marginTop: 10,
        overflow: "hidden",
        background: BG,
        border: `1px dashed ${filled ? SERIES_GREEN : RULE}`,
        borderRadius: 10,
      }}
    >
      {filled ? (
        <div
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: SERIES_GREEN,
            color: BG,
            fontSize: 17,
            fontWeight: 850,
            scale: interpolate(frame, [55, 105], ["0 1", "1 1"], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
              easing: Easing.bezier(0.16, 1, 0.3, 1),
            }),
            transformOrigin: "left center",
          }}
        >
          TRAINS FOR SPEED
        </div>
      ) : (
        <div style={{height: "100%", display: "flex", alignItems: "center", justifyContent: "center", color: DIM, fontSize: 17}}>
          NO TUNER STARTS
        </div>
      )}
    </div>
    <div style={{color: filled ? TEXT : DIM, fontSize: 16, marginTop: 9}}>
      {filled ? "Automatic after the idle check" : "No workload adaptation runs"}
    </div>
  </div>
);

export const ProblemSolutionScene: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <SubmissionFrame
      eyebrow="SUPPORTED PROFILES · GUARDED SPEEDUP"
      title="Faster local LLMs, automatically."
      provenance="product comparison · provider behavior varies by service and policy"
      caption="On supported profiles, keep captured traffic on your infrastructure, keep the OpenAI-compatible serving API, and use quiet compute for gated draft tuning."
    >
      <div
        style={{
          width: "100%",
          display: "grid",
          gridTemplateColumns: "245px repeat(3, minmax(0, 1fr))",
          gridTemplateRows: "72px 120px 120px 178px",
          gap: 12,
          opacity: interpolate(frame, [4, 24], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.bezier(0.16, 1, 0.3, 1),
          }),
        }}
      >
        <div />
        {columns.map((column, index) => (
          <div
            key={column.title}
            style={{
              display: "flex",
              alignItems: "center",
              padding: "0 20px",
              color: column.accent,
              background: index === 2 ? "rgba(63,185,80,.08)" : "transparent",
              borderBottom: `2px solid ${column.accent}`,
              borderRadius: "10px 10px 0 0",
              fontSize: 23,
              fontWeight: 820,
              lineHeight: 1.18,
            }}
          >
            {column.title}
          </div>
        ))}

        <div style={{display: "flex", alignItems: "center", color: DIM, fontSize: 17, fontWeight: 800, letterSpacing: 0.8}}>YOUR DATA</div>
        {columns.map((column, index) => <Cell key={column.title} accent={index === 2 ? TEXT : column.accent} highlighted={index === 2}>{column.data}</Cell>)}

        <div style={{display: "flex", alignItems: "center", color: DIM, fontSize: 17, fontWeight: 800, letterSpacing: 0.8}}>WORKLOAD ADAPTATION</div>
        {columns.map((column, index) => <Cell key={column.title} accent={index === 2 ? TEXT : column.accent} highlighted={index === 2}>{column.optimization}</Cell>)}

        <div style={{display: "flex", alignItems: "center", color: DIM, fontSize: 17, fontWeight: 800, letterSpacing: 0.8}}>QUIET GPU WINDOW</div>
        <Cell accent={DIM}>Not yours</Cell>
        <Cell accent={DIM}><IdleWindow filled={false} frame={frame} /></Cell>
        <Cell accent={TEXT} highlighted><IdleWindow filled frame={frame} /></Cell>
      </div>
    </SubmissionFrame>
  );
};
