import React from "react";

export const TRIAD_COLORS = {
  navy: "#0A2E65",
  cobalt: "#1D63B8",
  teal: "#0D9488",
  skyHighlight: "#38BDF8",
  tealHighlight: "#2DD4BF",
  negative: "#E11D48",
  warning: "#D97706",
} as const;

/**
 * Computes CSS custom property `--range-pct` for live 3-color relational
 * gradient slider tracks across all interactive range inputs.
 */
export function getSliderFillStyle(value: number, min: number, max: number): React.CSSProperties {
  const span = max - min;
  const pct = span > 0 ? Math.max(0, Math.min(100, ((value - min) / span) * 100)) : 0;
  return { "--range-pct": `${pct.toFixed(2)}%` } as React.CSSProperties;
}

/**
 * Shared SVG <defs> implementing the 3-color relational triad gradient
 * (Sovereign Navy #0A2E65 -> Oceanic Cobalt #1D63B8 -> Emerald Teal #0D9488)
 * across Recharts Line, Area, Bar, Pie, and Composed charts.
 */
export const TriadSvgDefs: React.FC<{ idPrefix?: string }> = ({ idPrefix = "triad" }) => (
  <defs>
    {/* Horizontal Stroke Gradients */}
    <linearGradient id={`${idPrefix}-stroke-full`} x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stopColor="#0A2E65" />
      <stop offset="50%" stopColor="#1D63B8" />
      <stop offset="100%" stopColor="#0D9488" />
    </linearGradient>
    <linearGradient id={`${idPrefix}-stroke-navy-cobalt`} x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stopColor="#0A2E65" />
      <stop offset="100%" stopColor="#1D63B8" />
    </linearGradient>
    <linearGradient id={`${idPrefix}-stroke-cobalt-teal`} x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stopColor="#1D63B8" />
      <stop offset="100%" stopColor="#0D9488" />
    </linearGradient>
    <linearGradient id={`${idPrefix}-stroke-dark-hist`} x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stopColor="#1D63B8" />
      <stop offset="55%" stopColor="#38BDF8" />
      <stop offset="100%" stopColor="#2DD4BF" />
    </linearGradient>
    <linearGradient id={`${idPrefix}-stroke-dark-sim`} x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stopColor="#38BDF8" />
      <stop offset="50%" stopColor="#14B8A6" />
      <stop offset="100%" stopColor="#34D399" />
    </linearGradient>
    <linearGradient id={`${idPrefix}-stroke-dark-stress`} x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stopColor="#14B8A6" />
      <stop offset="45%" stopColor="#F59E0B" />
      <stop offset="100%" stopColor="#FB7185" />
    </linearGradient>

    {/* Vertical Area Fill Washes */}
    <linearGradient id={`${idPrefix}-area-navy`} x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stopColor="#0A2E65" stopOpacity={0.26} />
      <stop offset="60%" stopColor="#1D63B8" stopOpacity={0.1} />
      <stop offset="100%" stopColor="#1D63B8" stopOpacity={0.01} />
    </linearGradient>
    <linearGradient id={`${idPrefix}-area-cobalt`} x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stopColor="#1D63B8" stopOpacity={0.24} />
      <stop offset="65%" stopColor="#0D9488" stopOpacity={0.09} />
      <stop offset="100%" stopColor="#0D9488" stopOpacity={0.01} />
    </linearGradient>
    <linearGradient id={`${idPrefix}-area-teal`} x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stopColor="#0D9488" stopOpacity={0.28} />
      <stop offset="70%" stopColor="#14B8A6" stopOpacity={0.08} />
      <stop offset="100%" stopColor="#14B8A6" stopOpacity={0.01} />
    </linearGradient>

    {/* Confidence / FX-Propagated Cone Horizontal Gradients */}
    <linearGradient id={`${idPrefix}-band-dark`} x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stopColor="#1D63B8" stopOpacity={0.22} />
      <stop offset="50%" stopColor="#0D9488" stopOpacity={0.26} />
      <stop offset="100%" stopColor="#2DD4BF" stopOpacity={0.22} />
    </linearGradient>
    <linearGradient id={`${idPrefix}-band-light`} x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stopColor="#0A2E65" stopOpacity={0.14} />
      <stop offset="50%" stopColor="#1D63B8" stopOpacity={0.18} />
      <stop offset="100%" stopColor="#0D9488" stopOpacity={0.2} />
    </linearGradient>

    {/* Vertical & Horizontal Bar Gradients */}
    <linearGradient id={`${idPrefix}-bar-navy-cobalt`} x1="0" y1="1" x2="0" y2="0">
      <stop offset="0%" stopColor="#0A2E65" />
      <stop offset="100%" stopColor="#1D63B8" />
    </linearGradient>
    <linearGradient id={`${idPrefix}-bar-cobalt-teal`} x1="0" y1="1" x2="0" y2="0">
      <stop offset="0%" stopColor="#1D63B8" />
      <stop offset="100%" stopColor="#0D9488" />
    </linearGradient>
    <linearGradient id={`${idPrefix}-bar-triad-v`} x1="0" y1="1" x2="0" y2="0">
      <stop offset="0%" stopColor="#0A2E65" />
      <stop offset="52%" stopColor="#1D63B8" />
      <stop offset="100%" stopColor="#0D9488" />
    </linearGradient>
    <linearGradient id={`${idPrefix}-bar-triad-h`} x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stopColor="#0A2E65" />
      <stop offset="52%" stopColor="#1D63B8" />
      <stop offset="100%" stopColor="#0D9488" />
    </linearGradient>
    <linearGradient id={`${idPrefix}-bar-cobalt-teal-h`} x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stopColor="#1D63B8" />
      <stop offset="100%" stopColor="#0D9488" />
    </linearGradient>
    <linearGradient id={`${idPrefix}-bar-amber-h`} x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stopColor="#B45309" />
      <stop offset="100%" stopColor="#F59E0B" />
    </linearGradient>
    <linearGradient id={`${idPrefix}-bar-negative`} x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stopColor="#F43F5E" />
      <stop offset="100%" stopColor="#BE123C" />
    </linearGradient>
  </defs>
);

export const TRIAD_TOOLTIP_STYLE: React.CSSProperties = {
  backgroundColor: "rgba(10, 23, 45, 0.96)",
  border: "1px solid rgba(56, 189, 248, 0.25)",
  borderRadius: "10px",
  color: "#F8FAFC",
  fontSize: "12px",
  boxShadow: "0 10px 25px -5px rgba(10, 30, 61, 0.35)",
  padding: "8px 12px",
};

export const TRIAD_TOOLTIP_LABEL_STYLE: React.CSSProperties = {
  color: "#94A3B8",
  fontWeight: 600,
  marginBottom: "4px",
};

export const TRIAD_TOOLTIP_ITEM_STYLE: React.CSSProperties = {
  color: "#F8FAFC",
  padding: "1px 0",
};
