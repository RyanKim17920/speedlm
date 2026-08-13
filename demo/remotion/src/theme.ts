// Chrome colors are the same constants demo/render.py draws the stills with, so
// the Remotion composition and the PIL renderer read as one system.
export const BG = "rgb(13,17,23)";
export const PANEL = "rgb(22,27,34)";
export const RULE = "rgb(48,54,61)";
export const TEXT = "rgb(201,209,217)";
export const DIM = "rgb(110,118,129)";
export const MUTED = "rgb(139,148,158)";

// render.py's brand accents. Kept for chrome (rules, verdict pill) but NOT used as
// an adjacent categorical pair in a chart: amber vs green collide under
// protanopia/deuteranopia (validated ΔE 5.1, a hard fail), so the series colors
// below are re-stepped into the dark-mode lightness band and validated instead.
export const STOCK_ACCENT_BRAND = "rgb(210,153,34)";
export const TUNED_ACCENT_BRAND = "rgb(63,185,80)";

// Categorical series steps — all validated against surface PANEL in dark mode
// (scripts/validate_palette.js: lightness band, chroma floor, CVD separation,
// normal-vision floor, contrast).
export const SERIES_BLUE = "#3D8BE2"; // stock / baseline arm, and train loss
export const SERIES_GREEN = "#31AE45"; // tuned / candidate arm
export const SERIES_AMBER = "#BF8600"; // validation loss

// Ordinal ramp for draft positions 0..2 — one hue, monotone lightness, validated.
export const POSITION_RAMP = ["#8FE0A0", "#4FC463", "#1E8F38"] as const;

export const MONO =
  '"JetBrains Mono", "DejaVu Sans Mono", "Liberation Mono", "Menlo", monospace';
