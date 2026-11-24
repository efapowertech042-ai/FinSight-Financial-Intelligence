import React, { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { AlertCircle, CheckCircle2, Database, GitBranch, SlidersHorizontal } from "lucide-react";
import { FeatureIntelligenceItem, FeaturesIntelligenceResponse } from "../types/financial";
import { useScenario } from "../context/ScenarioContext";
import { formatNGN, formatPct } from "../utils/format";

interface DataEngineeringViewProps {
  data: FeaturesIntelligenceResponse | null;
}

const CATEGORIES = [
  "All",
  "Macroeconomic",
  "Sales & Commercial",
  "Import & Cost",
  "Financial Performance",
  "Temporal & Contextual",
];

// Positioned nodes for interactive dependency graph SVG
const GRAPH_POSITIONS: Record<string, { x: number; y: number }> = {
  fx: { x: 40, y: 36 },
  imp_usd: { x: 40, y: 120 },
  imp_ngn: { x: 250, y: 78 },
  units: { x: 250, y: 196 },
  price: { x: 250, y: 280 },
  cogs: { x: 470, y: 112 },
  rev: { x: 470, y: 238 },
  gp: { x: 690, y: 175 },
  mkt: { x: 690, y: 46 },
  opex: { x: 690, y: 295 },
  op: { x: 910, y: 175 },
};

export const DataEngineeringView: React.FC<DataEngineeringViewProps> = ({ data }) => {
  const { activePresetLabel, singlePeriodResult, setDrawerOpen } = useScenario();
  const [selectedCategory, setSelectedCategory] = useState<string>("All");
  const [selectedFeatureName, setSelectedFeatureName] = useState<string>("USD_NGN_Planning_Rate");

  const features = data?.features || [];
  const qr = data?.quality_report;
  const graph = data?.dependency_graph;

  const filteredFeatures = useMemo(() => {
    if (selectedCategory === "All") return features;
    return features.filter((f) => f.category === selectedCategory);
  }, [features, selectedCategory]);

  const activeFeature: FeatureIntelligenceItem | undefined =
    features.find((f) => f.name === selectedFeatureName) || features[0];

  if (!data || !qr) {
    return (
      <div className="rounded-xl border border-slate-200 bg-white p-8 text-center text-sm text-slate-500">
        Loading data engineering &amp; feature integrity report...
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* 1. Data Integrity & Quality Report Strip */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <div className="text-xs font-medium text-slate-500">Records Processed</div>
          <div className="mt-1 font-display text-2xl font-bold text-[#0A1E3D] tabular-nums">
            {qr.records_processed}
          </div>
          <div className="mt-0.5 text-[11px] text-slate-500">Chronologically sorted</div>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <div className="text-xs font-medium text-slate-500">Missing Values</div>
          <div className="mt-1 font-display text-2xl font-bold text-amber-700 tabular-nums">
            {qr.missing_values_total}
          </div>
          <div className="mt-0.5 text-[11px] text-slate-500">
            First-row FX MoM &amp; nulls checked
          </div>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <div className="text-xs font-medium text-slate-500">Duplicate Records</div>
          <div className="mt-1 font-display text-2xl font-bold text-[#0A1E3D] tabular-nums">
            {qr.duplicate_records}
          </div>
          <div className="mt-0.5 text-[11px] text-slate-500">
            Missing periods: {qr.missing_periods.length}
          </div>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <div className="text-xs font-medium text-slate-500">Material Discrepancies</div>
          <div
            className={`mt-1 font-display text-2xl font-bold tabular-nums ${
              qr.material_discrepancies_count > 0 ? "text-red-600" : "text-emerald-700"
            }`}
          >
            {qr.material_discrepancies_count}
          </div>
          <div className="mt-0.5 text-[11px] text-slate-500">
            Rounding variances: {qr.rounding_variances_count}
          </div>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <div className="text-xs font-medium text-slate-500">Negative Value Alerts</div>
          <div className="mt-1 font-display text-2xl font-bold text-[#0A1E3D] tabular-nums">
            {qr.negative_value_flags.length}
          </div>
          <div className="mt-0.5 text-[11px] text-slate-500">
            Inconsistent rels: {qr.inconsistent_relationships.length}
          </div>
        </div>
      </div>

      {/* 2. Interactive Feature Dependency Graph */}
      <div className="rounded-xl border border-slate-200 bg-white p-5">
        <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-2">
            <GitBranch className="h-4 w-4 text-[#0A2E65]" />
            <h2 className="font-display text-sm font-bold text-[#0A1E3D]">
              Interactive Financial Feature Dependency Graph
            </h2>
          </div>
          <span className="text-xs text-slate-500">
            Select any node to inspect its formula, distribution, and profitability correlation below
          </span>
        </div>

        <div className="mt-4 overflow-x-auto rounded-lg border border-slate-100 bg-slate-50/70 p-4">
          <svg viewBox="0 0 1120 355" className="min-w-[820px] w-full h-[300px]">
            <defs>
              <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3.5" orient="auto">
                <polygon points="0 0, 8 3.5, 0 7" fill="#64748B" />
              </marker>
            </defs>

            {/* Edges */}
            {graph?.edges.map((edge, idx) => {
              const s = GRAPH_POSITIONS[edge.source];
              const t = GRAPH_POSITIONS[edge.target];
              if (!s || !t) return null;
              const x1 = s.x + 165;
              const y1 = s.y + 24;
              const x2 = t.x;
              const y2 = t.y + 24;
              const mx = (x1 + x2) / 2;
              const my = (y1 + y2) / 2;
              return (
                <g key={`${edge.source}-${edge.target}-${idx}`}>
                  <line
                    x1={x1}
                    y1={y1}
                    x2={x2 - 4}
                    y2={y2}
                    stroke="#94A3B8"
                    strokeWidth="1.6"
                    markerEnd="url(#arrow)"
                  />
                  <rect x={mx - 22} y={my - 9} width="44" height="17" rx="4" fill="#F8FAFC" />
                  <text
                    x={mx}
                    y={my + 3}
                    textAnchor="middle"
                    fontSize="10"
                    fontWeight="600"
                    fill="#475569"
                  >
                    {edge.label}
                  </text>
                </g>
              );
            })}

            {/* Clickable Nodes */}
            {graph?.nodes.map((node) => {
              const pos = GRAPH_POSITIONS[node.id];
              if (!pos) return null;
              const isSelected = selectedFeatureName === node.label;
              const isTarget = node.label === "Operating_Profit_NGN";
              return (
                <g
                  key={node.id}
                  transform={`translate(${pos.x}, ${pos.y})`}
                  onClick={() => setSelectedFeatureName(node.label)}
                  className="cursor-pointer"
                >
                  <rect
                    width="165"
                    height="48"
                    rx="8"
                    fill={isSelected ? "#0A2E65" : isTarget ? "#0D9488" : "#FFFFFF"}
                    stroke={isSelected ? "#0A2E65" : isTarget ? "#0F766E" : "#CBD5E1"}
                    strokeWidth={isSelected ? "2" : "1.2"}
                  />
                  <text
                    x="10"
                    y="19"
                    fontSize="10.5"
                    fontWeight="700"
                    fill={isSelected || isTarget ? "#FFFFFF" : "#0F172A"}
                  >
                    {node.label.length > 22 ? `${node.label.slice(0, 21)}…` : node.label}
                  </text>
                  <text
                    x="10"
                    y="35"
                    fontSize="9.5"
                    fill={isSelected || isTarget ? "#E2E8F0" : "#64748B"}
                  >
                    {node.formula.length > 26 ? `${node.formula.slice(0, 25)}…` : node.formula}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>

        {singlePeriodResult && (
          <div className="mt-4 rounded-lg border border-slate-200 bg-slate-50 p-4 space-y-2.5">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="text-xs font-bold text-[#0A1E3D]">
                Live Formula Stress-Test Propagation ({activePresetLabel})
              </div>
              <button
                type="button"
                onClick={() => setDrawerOpen(true)}
                className="inline-flex items-center gap-1 rounded-md border border-slate-300 bg-white px-2.5 py-1 text-[11px] font-semibold text-[#0A2E65] hover:bg-slate-100 whitespace-nowrap"
              >
                <SlidersHorizontal className="h-3 w-3" />
                <span>Adjust Input Nodes</span>
              </button>
            </div>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-5 text-xs tabular-nums">
              <div className="rounded-lg bg-white border border-slate-200 p-2.5">
                <div className="text-[11px] text-slate-500">Import Cost / Unit</div>
                <div className="font-mono font-bold text-[#0A1E3D]">
                  {formatNGN(singlePeriodResult.simulated.accounting.Import_Cost_NGN_Per_Unit)}
                </div>
                <div className="text-[10px] text-slate-400">
                  Base: {formatNGN(singlePeriodResult.baseline.accounting.Import_Cost_NGN_Per_Unit)}
                </div>
              </div>
              <div className="rounded-lg bg-white border border-slate-200 p-2.5">
                <div className="text-[11px] text-slate-500">COGS_NGN</div>
                <div className="font-mono font-bold text-[#0A1E3D]">
                  {formatNGN(singlePeriodResult.simulated.accounting.COGS_NGN, true)}
                </div>
                <div className="text-[10px] text-slate-500">
                  Δ {formatNGN(singlePeriodResult.deltas.cogs_delta, true)}
                </div>
              </div>
              <div className="rounded-lg bg-white border border-slate-200 p-2.5">
                <div className="text-[11px] text-slate-500">Sales_Revenue_NGN</div>
                <div className="font-mono font-bold text-[#0A1E3D]">
                  {formatNGN(singlePeriodResult.simulated.accounting.Sales_Revenue_NGN, true)}
                </div>
                <div className="text-[10px] text-slate-500">
                  Δ {formatNGN(singlePeriodResult.deltas.revenue_delta, true)}
                </div>
              </div>
              <div className="rounded-lg bg-white border border-slate-200 p-2.5">
                <div className="text-[11px] text-slate-500">Gross_Profit_NGN</div>
                <div className="font-mono font-bold text-[#0A1E3D]">
                  {formatNGN(singlePeriodResult.simulated.accounting.Gross_Profit_NGN, true)}
                </div>
                <div className="text-[10px] text-slate-500">
                  {formatPct(singlePeriodResult.simulated.accounting.Gross_Margin_Pct, 1)} GM
                </div>
              </div>
              <div className="rounded-lg bg-white border border-slate-200 p-2.5">
                <div className="text-[11px] text-slate-500">Operating_Profit_NGN</div>
                <div
                  className={`font-mono font-bold ${
                    singlePeriodResult.simulated.accounting.Accounting_Operating_Profit_NGN >= 0
                      ? "text-teal-700"
                      : "text-red-600"
                  }`}
                >
                  {formatNGN(singlePeriodResult.simulated.accounting.Accounting_Operating_Profit_NGN, true)}
                </div>
                <div className="text-[10px] text-slate-500">
                  Δ {formatNGN(singlePeriodResult.deltas.accounting_profit_delta, true)}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* 3. Feature Intelligence Explorer: Catalog + Selected Feature Deep Dive */}
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-12">
        {/* Left List */}
        <div className="rounded-xl border border-slate-200 bg-white p-5 lg:col-span-5">
          <div className="flex items-center gap-2">
            <Database className="h-4 w-4 text-[#0A2E65]" />
            <h2 className="font-display text-sm font-bold text-[#0A1E3D]">
              Dataset Schema &amp; Feature Catalog ({features.length})
            </h2>
          </div>

          <div className="mt-3 flex flex-wrap gap-1 rounded-lg bg-slate-100 p-1">
            {CATEGORIES.map((cat) => (
              <button
                key={cat}
                type="button"
                onClick={() => setSelectedCategory(cat)}
                className={`rounded-md px-2 py-1 text-[11px] font-medium transition-colors ${
                  selectedCategory === cat
                    ? "bg-white text-[#0A1E3D] shadow-sm"
                    : "text-slate-600 hover:text-slate-900"
                }`}
              >
                {cat}
              </button>
            ))}
          </div>

          <div className="mt-3 max-h-[430px] space-y-1.5 overflow-y-auto pr-1">
            {filteredFeatures.map((f) => {
              const active = activeFeature?.name === f.name;
              return (
                <button
                  key={f.name}
                  type="button"
                  onClick={() => setSelectedFeatureName(f.name)}
                  className={`flex w-full items-center justify-between rounded-lg border px-3 py-2.5 text-left transition-colors ${
                    active
                      ? "border-[#0A2E65] bg-[#0A2E65]/5"
                      : "border-slate-200 bg-white hover:bg-slate-50"
                  }`}
                >
                  <div className="min-w-0">
                    <div className="truncate font-mono text-xs font-semibold text-[#0A1E3D]">
                      {f.name}
                    </div>
                    <div className="mt-0.5 text-[11px] text-slate-500">
                      {f.category} · {f.data_type}
                    </div>
                  </div>
                  {f.stats && (
                    <div className="ml-2 text-right font-mono text-[11px] text-slate-700 tabular-nums shrink-0">
                      r = {f.stats.corr_with_operating_profit.toFixed(2)}
                    </div>
                  )}
                </button>
              );
            })}
          </div>
        </div>

        {/* Right Detail Panel */}
        {activeFeature && (
          <div className="rounded-xl border border-slate-200 bg-white p-5 lg:col-span-7 space-y-4">
            <div className="flex flex-wrap items-start justify-between gap-2 border-b border-slate-100 pb-3">
              <div>
                <div className="text-xs font-medium text-slate-500">
                  {activeFeature.category} · {activeFeature.data_type}
                </div>
                <h3 className="font-mono text-lg font-bold text-[#0A1E3D]">
                  {activeFeature.name}
                </h3>
              </div>
              {activeFeature.stats && (
                <div className="text-right">
                  <div className="text-[11px] text-slate-500">Correlation w/ Operating Profit</div>
                  <div className="font-mono text-base font-bold text-teal-700 tabular-nums">
                    r = {activeFeature.stats.corr_with_operating_profit.toFixed(4)}
                  </div>
                </div>
              )}
            </div>

            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 text-xs">
              <div className="rounded-lg bg-slate-50 p-3">
                <div className="font-semibold text-slate-700">Business Definition</div>
                <p className="mt-1 text-slate-600 leading-relaxed">
                  {activeFeature.business_definition}
                </p>
              </div>
              <div className="rounded-lg bg-slate-50 p-3">
                <div className="font-semibold text-slate-700">Formula &amp; Mathematical Role</div>
                <p className="mt-1 font-mono text-[11px] text-[#0A2E65]">{activeFeature.formula}</p>
                <p className="mt-1 text-slate-600">{activeFeature.mathematical_role}</p>
              </div>
              <div className="rounded-lg bg-slate-50 p-3">
                <div className="font-semibold text-slate-700">Financial Interpretation</div>
                <p className="mt-1 text-slate-600 leading-relaxed">
                  {activeFeature.financial_interpretation}
                </p>
              </div>
              <div className="rounded-lg bg-slate-50 p-3">
                <div className="font-semibold text-slate-700">
                  Influence on Operating Profit &amp; Availability
                </div>
                <p className="mt-1 text-slate-600">{activeFeature.profitability_influence}</p>
                <p className="mt-1 text-[11px] font-medium text-slate-500">
                  Forecast availability: {activeFeature.forecast_availability}
                </p>
              </div>
            </div>

            {activeFeature.stats && (
              <>
                <div className="grid grid-cols-3 gap-2 sm:grid-cols-6 rounded-lg border border-slate-200 p-3 text-xs">
                  <div>
                    <div className="text-[11px] text-slate-500">Min</div>
                    <div className="font-mono font-semibold tabular-nums">
                      {activeFeature.stats.min.toLocaleString()}
                    </div>
                  </div>
                  <div>
                    <div className="text-[11px] text-slate-500">Median</div>
                    <div className="font-mono font-semibold tabular-nums">
                      {activeFeature.stats.median.toLocaleString()}
                    </div>
                  </div>
                  <div>
                    <div className="text-[11px] text-slate-500">Mean</div>
                    <div className="font-mono font-semibold tabular-nums">
                      {activeFeature.stats.mean.toLocaleString()}
                    </div>
                  </div>
                  <div>
                    <div className="text-[11px] text-slate-500">Max</div>
                    <div className="font-mono font-semibold tabular-nums">
                      {activeFeature.stats.max.toLocaleString()}
                    </div>
                  </div>
                  <div>
                    <div className="text-[11px] text-slate-500">Std Dev</div>
                    <div className="font-mono font-semibold tabular-nums">
                      {activeFeature.stats.std.toLocaleString()}
                    </div>
                  </div>
                  <div>
                    <div className="text-[11px] text-slate-500">IQR Outliers</div>
                    <div className="font-mono font-semibold text-amber-700 tabular-nums">
                      {activeFeature.stats.iqr_outliers}
                    </div>
                  </div>
                </div>

                <div>
                  <div className="mb-1.5 text-xs font-semibold text-slate-700">
                    Statistical Distribution Histogram
                  </div>
                  <div className="h-[150px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={activeFeature.stats.histogram}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                        <XAxis dataKey="bin" tick={{ fontSize: 9, fill: "#64748B" }} />
                        <YAxis tick={{ fontSize: 10, fill: "#64748B" }} width={30} />
                        <Tooltip />
                        <Bar dataKey="count" name="Observations" fill="#0D9488" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </>
            )}

            {activeFeature.warnings.length > 0 && (
              <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs text-amber-900">
                <div className="flex items-center gap-1.5 font-semibold">
                  <AlertCircle className="h-3.5 w-3.5" />
                  <span>Data Quality &amp; Governance Notes</span>
                </div>
                <ul className="mt-1 list-disc pl-5 space-y-0.5">
                  {activeFeature.warnings.map((w, i) => (
                    <li key={i}>{w}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>

      {/* 4. Calculation Discrepancy & Integrity Audit */}
      <div className="rounded-xl border border-slate-200 bg-white p-5">
        <div className="flex items-center gap-2">
          <CheckCircle2 className="h-4 w-4 text-teal-700" />
          <h2 className="font-display text-sm font-bold text-[#0A1E3D]">
            Independent Recalculation vs. Supplied Columns Audit
          </h2>
        </div>
        <p className="mt-1 text-xs text-slate-500">
          Compares every supplied calculated column against formulas independently recomputed from raw inputs.
          Detected scale conventions:{" "}
          {Object.entries(qr.scale_conventions_detected)
            .map(([k, v]) => `${k} (${v})`)
            .join(" · ")}
        </p>

        {qr.discrepancy_samples.length === 0 ? (
          <div className="mt-3 rounded-lg border border-emerald-200 bg-emerald-50 p-3.5 text-xs text-emerald-900">
            All {qr.records_processed} records reconciled with 0 material calculation discrepancies across
            Import_Cost_NGN_Per_Unit, COGS_NGN, Sales_Revenue_NGN, Gross_Profit_NGN, Gross_Margin_Pct,
            Operating_Profit_NGN, Operating_Margin_Pct, and FX_Change_MoM_Pct.
          </div>
        ) : (
          <div className="mt-3 max-h-[260px] overflow-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-50 border-b border-slate-200 text-slate-600">
                <tr>
                  <th className="py-2 px-3">Month</th>
                  <th className="py-2 px-3">Column</th>
                  <th className="py-2 px-3 text-right">Supplied (Norm.)</th>
                  <th className="py-2 px-3 text-right">Recalculated</th>
                  <th className="py-2 px-3 text-right">Abs Diff</th>
                  <th className="py-2 px-3">Classification</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {qr.discrepancy_samples.map((d, idx) => (
                  <tr key={`${d.month}-${d.column}-${idx}`}>
                    <td className="py-2 px-3 font-mono">{d.month}</td>
                    <td className="py-2 px-3 font-mono">{d.column}</td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums">
                      {d.supplied_normalized.toLocaleString()}
                    </td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums">
                      {d.recalculated.toLocaleString()}
                    </td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums">
                      {d.abs_difference.toLocaleString()}
                    </td>
                    <td className="py-2 px-3 font-medium text-amber-800">{d.classification}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {qr.negative_value_flags.length > 0 && (
          <div className="mt-4">
            <h3 className="text-xs font-bold text-slate-800">
              Negative Financial Value Flags ({qr.negative_value_flags.length})
            </h3>
            <div className="mt-2 space-y-1.5">
              {qr.negative_value_flags.map((nf, i) => (
                <div
                  key={i}
                  className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-amber-200 bg-amber-50/70 px-3 py-2 text-xs"
                >
                  <span>
                    <strong className="font-mono">{nf.month}</strong> · {nf.column} ={" "}
                    <strong className="font-mono text-red-700">{nf.value.toLocaleString()}</strong>
                  </span>
                  <span className="text-slate-700">{nf.reason}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
