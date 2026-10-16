import React from "react";
import {AbsoluteFill} from "remotion";
import {BG, DIM, MONO, PANEL, RULE, SERIES_GREEN, TEXT} from "../theme";

export const SubmissionFrame: React.FC<{
  caption: string;
  children: React.ReactNode;
  eyebrow: string;
  provenance: string;
  title: string;
}> = ({caption, children, eyebrow, provenance, title}) => (
  <AbsoluteFill
    style={{
      background: BG,
      color: TEXT,
      fontFamily: MONO,
      padding: "38px 74px 170px",
    }}
  >
    <header style={{display: "flex", alignItems: "flex-end", justifyContent: "space-between", gap: 30}}>
      <div>
        <div style={{color: SERIES_GREEN, fontSize: 18, fontWeight: 800, letterSpacing: 1.6}}>{eyebrow}</div>
        <div style={{color: TEXT, fontSize: 50, fontWeight: 780, letterSpacing: -2.2, marginTop: 8}}>{title}</div>
      </div>
      <div style={{color: DIM, fontSize: 16, textAlign: "right", maxWidth: 620, lineHeight: 1.35}}>{provenance}</div>
    </header>
    <main style={{display: "flex", flex: 1, minHeight: 0, marginTop: 24}}>{children}</main>
    <div
      style={{
        position: "absolute",
        left: 74,
        right: 74,
        bottom: 42,
        minHeight: 88,
        display: "flex",
        alignItems: "center",
        padding: "14px 22px",
        background: PANEL,
        border: `1px solid ${RULE}`,
        borderRadius: 15,
        color: TEXT,
        fontSize: 24,
        fontWeight: 600,
        lineHeight: 1.32,
      }}
    >
      <span style={{color: SERIES_GREEN, marginRight: 14}}>●</span>
      {caption}
    </div>
  </AbsoluteFill>
);

export const EvidencePill: React.FC<{children: React.ReactNode; green?: boolean}> = ({children, green = false}) => (
  <div
    style={{
      color: green ? SERIES_GREEN : TEXT,
      background: PANEL,
      border: `1px solid ${green ? SERIES_GREEN : RULE}`,
      borderRadius: 999,
      padding: "10px 16px",
      fontSize: 18,
      fontWeight: 760,
      whiteSpace: "nowrap",
    }}
  >
    {children}
  </div>
);
