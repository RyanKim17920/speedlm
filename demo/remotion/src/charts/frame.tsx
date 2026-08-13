import React from "react";
import { DIM, MUTED, RULE, TEXT, MONO } from "../theme";

export type Pad = { t: number; r: number; b: number; l: number };

export const scale =
  (d0: number, d1: number, r0: number, r1: number) =>
  (v: number) =>
    d1 === d0 ? r0 : r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);

/** A chart card: title row, optional legend, and the plot slot. */
export const ChartCard: React.FC<{
  title: string;
  subtitle?: string;
  legend?: React.ReactNode;
  children: React.ReactNode;
}> = ({ title, subtitle, legend, children }) => (
  <section
    style={{
      display: "flex",
      flexDirection: "column",
      gap: 10,
      flex: 1,
      minHeight: 0,
    }}
  >
    <header style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
      <h2
        style={{
          margin: 0,
          font: `600 21px/1.2 ${MONO}`,
          color: TEXT,
          letterSpacing: 0.2,
        }}
      >
        {title}
      </h2>
      {subtitle ? (
        <span style={{ font: `400 15px/1.2 ${MONO}`, color: DIM }}>{subtitle}</span>
      ) : null}
    </header>
    {legend ? <div style={{ display: "flex", gap: 18, flexWrap: "wrap" }}>{legend}</div> : null}
    <div style={{ flex: 1, minHeight: 0 }}>{children}</div>
  </section>
);

/** Legend entry — a colored mark plus a text-token label (never colored text). */
export const LegendItem: React.FC<{
  color: string;
  label: string;
  shape?: "line" | "square" | "diamond";
}> = ({ color, label, shape = "line" }) => (
  <span style={{ display: "flex", alignItems: "center", gap: 7 }}>
    {shape === "line" ? (
      <span style={{ width: 20, height: 3, borderRadius: 2, background: color }} />
    ) : shape === "diamond" ? (
      <span
        style={{
          width: 11,
          height: 11,
          background: color,
          transform: "rotate(45deg)",
          borderRadius: 2,
        }}
      />
    ) : (
      <span style={{ width: 13, height: 13, borderRadius: 3, background: color }} />
    )}
    <span style={{ font: `400 15px/1 ${MONO}`, color: MUTED }}>{label}</span>
  </span>
);

/** Recessive horizontal gridlines + y tick labels. */
export const YGrid: React.FC<{
  ticks: number[];
  y: (v: number) => number;
  x0: number;
  x1: number;
  format: (v: number) => string;
}> = ({ ticks, y, x0, x1, format }) => (
  <g>
    {ticks.map((t) => (
      <g key={t}>
        <line x1={x0} x2={x1} y1={y(t)} y2={y(t)} stroke={RULE} strokeWidth={1} />
        <text
          x={x0 - 10}
          y={y(t)}
          textAnchor="end"
          dominantBaseline="middle"
          style={{ font: `400 14px ${MONO}`, fill: DIM }}
        >
          {format(t)}
        </text>
      </g>
    ))}
  </g>
);
