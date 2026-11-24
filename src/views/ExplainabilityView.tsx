import React, { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { BrainCircuit, Info, SlidersHorizontal } from "lucide-react";
import { ExplainabilityResponse } from "../types/financial";
import { formatNGN } from "../utils/format";
import { GraphicalRagPanel } from "../components/GraphicalRagPanel";
import { useScenario } from "../context/ScenarioContext";
import {
  TRIAD_TOOLTIP_ITEM_STYLE,
  TRIAD_TOOLTIP_LABEL_STYLE,
  TRIAD_TOOLTIP_STYLE,
  TriadSvgDefs,
} from "../utils/chartTheme";

interface ExplainabilityViewProps {
  data: ExplainabilityResponse | null;
}

export const ExplainabilityView: React.FC<ExplainabilityViewProps> = ({ data }) => {
  const {
    activePresetLabel,
    isScenarioActive,
    driverDeltas,
    baselineInputs,
    simulatedInputs,
    singlePeriodResult,
    setDrawerOpen,
  } = useScenario();

  const dependenceKeys = Object.keys(data?.dependence_plots || {});
  const [selectedDepFeature, setSelectedDepFeature] = useState<string>(
    dependenceKeys[0] || "USD_NGN_Planning_Rate"
  );
  const [beeswarmFeature, setBeeswarmFeature] = useState<string>("ALL");

  const beeswarmData = useMemo(() => {
    if (!data) return [];
    const topFeatures = data.global_importance.slice(0, 6).map((g) => g.feature);
    const pts = data.beeswarm_points.filter((p) =>
      beeswarmFeature === "ALL" ? topFeatures.includes(p.feature) : p.feature === beeswarmFeature
    );
    return pts.map((p) => ({
      ...p,
      featureIndex: topFeatures.indexOf(p.feature),
      color: p.normalized_value >= 0.66 ? "#0A2E65" : p.normalized_value <= 0.33 ? "#0D9488" : "#1D63B8",
    }));
  }, [data, beeswarmFeature]);

  if (!data || !data.available) {
    return (
      <div className="rounded-xl border border-slate-200 bg-white p-8 text-center text-sm text-slate-500">
        {data?.reason || "Loading SHAP and permutation importance explanations..."}
      </div>
    );
  }

  const depPoints = data.dependence_plots[selectedDepFeature] || [];

  return (
    <div className="space-y-6">
      {/* Methodology Clarification Banner */}
      <div className="rounded-xl border border-slate-200 bg-white p-4 text-xs text-slate-700 flex items-start gap-3">
        <Info className="h-4 w-4 text-[#0A2E65] shrink-0 mt-0.5" />
        <div>
          <span className="font-bold text-[#0A1E3D]">
            Explainable AI Scope ({data.model_name} · Baseline Expected Profit:{" "}
            {formatNGN(data.base_value_ngn, true)}) —{" "}
          </span>
          <span>{data.methodology_note}</span>
        </div>
      </div>

      {/* A. Global SHAP Importance + Permutation Importance */}
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-12">
        <div className="rounded-xl border border-slate-200 bg-white p-5 lg:col-span-6">
          <div className="flex items-center gap-2">
            <BrainCircuit className="h-4 w-4 text-[#0A2E65]" />
            <h2 className="font-display text-sm font-bold text-[#0A1E3D]">
              Global Feature Importance (Mean |SHAP| Share %)
            </h2>
          </div>
          <p className="mt-0.5 text-xs text-slate-500">
            Teal bars indicate positive value-to-SHAP correlation; Amber indicates negative; Slate indicates mixed/non-linear.
          </p>
          <div className="mt-3 h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                layout="vertical"
                data={data.global_importance.slice(0, 10)}
                margin={{ top: 4, right: 20, left: 10, bottom: 4 }}
              >
                <TriadSvgDefs idPrefix="ex-shap" />
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis
                  type="number"
                  tick={{ fontSize: 10, fill: "#64748B" }}
                  tickFormatter={(v) => `${v}%`}
                />
                <YAxis
                  type="category"
                  dataKey="feature"
                  tick={{ fontSize: 10, fill: "#334155" }}
                  width={155}
                />
                <Tooltip
                  contentStyle={TRIAD_TOOLTIP_STYLE}
                  labelStyle={TRIAD_TOOLTIP_LABEL_STYLE}
                  itemStyle={TRIAD_TOOLTIP_ITEM_STYLE}
                  formatter={(val: any, _name: any, item: any) => [
                    `${val}% (Mean |SHAP| = ${formatNGN(item?.payload?.mean_abs_shap, true)})`,
                    "Contribution Share",
                  ]}
                />
                <Bar dataKey="share_pct" name="SHAP Share %" radius={[0, 4, 4, 0]}>
                  {data.global_importance.slice(0, 10).map((entry) => (
                    <Cell
                      key={entry.feature}
                      fill={
                        entry.value_shap_correlation > 0.15
                          ? "url(#ex-shap-bar-cobalt-teal-h)"
                          : entry.value_shap_correlation < -0.15
                            ? "url(#ex-shap-bar-amber-h)"
                            : "url(#ex-shap-bar-navy-cobalt)"
                      }
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-5 lg:col-span-6">
          <h2 className="font-display text-sm font-bold text-[#0A1E3D]">
            Out-of-Time Permutation Importance (R² Drop on Test Holdout)
          </h2>
          <p className="mt-0.5 text-xs text-slate-500">
            Validates feature predictive utility by shuffling each predictor on the chronological holdout set (15 repeats).
          </p>
          <div className="mt-3 h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                layout="vertical"
                data={[...data.global_importance]
                  .sort((a, b) => b.permutation_importance_mean - a.permutation_importance_mean)
                  .slice(0, 10)}
                margin={{ top: 4, right: 20, left: 10, bottom: 4 }}
              >
                <TriadSvgDefs idPrefix="ex-perm" />
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis type="number" tick={{ fontSize: 10, fill: "#64748B" }} />
                <YAxis
                  type="category"
                  dataKey="feature"
                  tick={{ fontSize: 10, fill: "#334155" }}
                  width={155}
                />
                <Tooltip
                  contentStyle={TRIAD_TOOLTIP_STYLE}
                  labelStyle={TRIAD_TOOLTIP_LABEL_STYLE}
                  itemStyle={TRIAD_TOOLTIP_ITEM_STYLE}
                  formatter={(val: any) => [Number(val).toFixed(4), "Mean R² Drop"]}
                />
                <ReferenceLine x={0} stroke="#64748B" />
                <Bar dataKey="permutation_importance_mean" name="Permutation R² Drop" fill="url(#ex-perm-bar-triad-h)" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* B. SHAP Summary Plot & C. SHAP Dependence Plot */}
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-12">
        {/* SHAP Summary Distribution */}
        <div className="rounded-xl border border-slate-200 bg-white p-5 lg:col-span-6">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h2 className="font-display text-sm font-bold text-[#0A1E3D]">
                SHAP Summary Distribution Across Observations
              </h2>
              <p className="text-xs text-slate-500">
                Navy = High feature value · Cobalt = Mid · Teal = Low feature value
              </p>
            </div>
            <select
              value={beeswarmFeature}
              onChange={(e) => setBeeswarmFeature(e.target.value)}
              className="rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs font-medium"
            >
              <option value="ALL">Top 6 Features</option>
              {data.global_importance.map((g) => (
                <option key={g.feature} value={g.feature}>
                  {g.feature}
                </option>
              ))}
            </select>
          </div>
          <div className="mt-3 h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 10, right: 16, left: 0, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis
                  type="number"
                  dataKey="shap_value"
                  name="SHAP Impact"
                  tick={{ fontSize: 10, fill: "#64748B" }}
                  tickFormatter={(v) => `₦${(v / 1e6).toFixed(0)}M`}
                />
                <YAxis
                  type="number"
                  dataKey="normalized_value"
                  name="Normalized Value"
                  domain={[0, 1]}
                  tick={{ fontSize: 10, fill: "#64748B" }}
                  width={40}
                />
                <ZAxis range={[28, 28]} />
                <ReferenceLine x={0} stroke="#64748B" />
                <Tooltip
                  contentStyle={TRIAD_TOOLTIP_STYLE}
                  labelStyle={TRIAD_TOOLTIP_LABEL_STYLE}
                  itemStyle={TRIAD_TOOLTIP_ITEM_STYLE}
                  formatter={(val: any, name: any) => [
                    name === "SHAP Impact" ? formatNGN(Number(val)) : Number(val).toFixed(2),
                    name,
                  ]}
                  labelFormatter={(_, payload) => {
                    const p = payload?.[0]?.payload;
                    return p ? `${p.feature} · ${p.month_label} (val: ${p.feature_value})` : "";
                  }}
                />
                <Scatter name="Observation" data={beeswarmData}>
                  {beeswarmData.map((entry, idx) => (
                    <Cell key={`${entry.month}-${idx}`} fill={entry.color} fillOpacity={0.78} />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* SHAP Dependence Plot */}
        <div className="rounded-xl border border-slate-200 bg-white p-5 lg:col-span-6">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h2 className="font-display text-sm font-bold text-[#0A1E3D]">
                SHAP Dependence Plot
              </h2>
              <p className="text-xs text-slate-500">
                Inspect how feature magnitude shifts predicted Operating Profit
              </p>
            </div>
            <select
              value={selectedDepFeature}
              onChange={(e) => setSelectedDepFeature(e.target.value)}
              className="rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs font-medium"
            >
              {dependenceKeys.map((k) => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))}
            </select>
          </div>
          <div className="mt-3 h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 10, right: 16, left: 0, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis
                  type="number"
                  dataKey="x"
                  name={selectedDepFeature}
                  tick={{ fontSize: 10, fill: "#64748B" }}
                />
                <YAxis
                  type="number"
                  dataKey="shap_value"
                  name="SHAP Contribution"
                  tick={{ fontSize: 10, fill: "#64748B" }}
                  tickFormatter={(v) => `₦${(v / 1e6).toFixed(0)}M`}
                  width={54}
                />
                <ZAxis range={[36, 36]} />
                <ReferenceLine y={0} stroke="#64748B" />
                <Tooltip
                  contentStyle={TRIAD_TOOLTIP_STYLE}
                  labelStyle={TRIAD_TOOLTIP_LABEL_STYLE}
                  itemStyle={TRIAD_TOOLTIP_ITEM_STYLE}
                  formatter={(val: any, name: any) => [
                    name === "SHAP Contribution" ? formatNGN(Number(val)) : Number(val).toLocaleString(),
                    name,
                  ]}
                  labelFormatter={(_, payload) => {
                    const p = payload?.[0]?.payload;
                    return p ? `${p.month_label} · Actual OP: ${formatNGN(p.actual_profit, true)}` : "";
                  }}
                />
                <Legend wrapperStyle={{ fontSize: "11px" }} />
                <Scatter
                  name={`${selectedDepFeature} vs SHAP Impact`}
                  data={depPoints}
                  fill="#1D63B8"
                  fillOpacity={0.78}
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Feature Direction Table */}
      <div className="rounded-xl border border-slate-200 bg-white p-5">
        <h2 className="font-display text-sm font-bold text-[#0A1E3D]">
          Feature Importance &amp; Directional Relationship Matrix
        </h2>
        <div className="mt-3 overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-200 text-slate-500">
                <th className="py-2 pr-4 font-semibold">Feature</th>
                <th className="py-2 pr-4 text-right font-semibold">Mean |SHAP| (₦)</th>
                <th className="py-2 pr-4 text-right font-semibold">Share (%)</th>
                <th className="py-2 pr-4 text-right font-semibold">Perm. Importance (ΔR²)</th>
                <th className="py-2 pr-4 text-right font-semibold">Value–SHAP Corr</th>
                <th className="py-2 font-semibold">Relationship Direction</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {data.global_importance.map((g) => (
                <tr key={g.feature}>
                  <td className="py-2.5 pr-4 font-mono font-semibold text-[#0A1E3D]">{g.feature}</td>
                  <td className="py-2.5 pr-4 text-right font-mono tabular-nums">
                    {formatNGN(g.mean_abs_shap, true)}
                  </td>
                  <td className="py-2.5 pr-4 text-right font-mono tabular-nums">{g.share_pct}%</td>
                  <td className="py-2.5 pr-4 text-right font-mono tabular-nums">
                    {g.permutation_importance_mean.toFixed(4)} ± {g.permutation_importance_std.toFixed(3)}
                  </td>
                  <td className="py-2.5 pr-4 text-right font-mono tabular-nums">
                    {g.value_shap_correlation.toFixed(3)}
                  </td>
                  <td className="py-2.5 text-slate-700">{g.direction}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Scenario Driver Sensitivity Mapping */}
      <div className="rounded-xl border border-slate-200 bg-white p-5">
        <div className="flex flex-col gap-3 border-b border-slate-100 pb-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="font-display text-sm font-bold text-[#0A1E3D]">
              Active Scenario Driver Sensitivity Mapping ({activePresetLabel})
            </h2>
            <p className="mt-0.5 text-xs text-slate-500">
              Maps top SHAP profit drivers to your active global scenario lever shifts and simulated single-period ML output.
            </p>
          </div>
          <div className="flex items-center gap-3">
            {singlePeriodResult && (
              <div className="text-right text-xs tabular-nums">
                <div className="text-[11px] text-slate-500">ML Profit Delta vs. Baseline</div>
                <div
                  className={`font-mono font-bold ${
                    singlePeriodResult.deltas.ml_predicted_profit_delta >= 0 ? "text-teal-700" : "text-red-600"
                  }`}
                >
                  {singlePeriodResult.deltas.ml_predicted_profit_delta >= 0 ? "+" : ""}
                  {formatNGN(singlePeriodResult.deltas.ml_predicted_profit_delta, true)}
                </div>
              </div>
            )}
            <button
              type="button"
              onClick={() => setDrawerOpen(true)}
              className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs font-semibold text-[#0A1E3D] hover:bg-slate-100 whitespace-nowrap"
            >
              <SlidersHorizontal className="h-3.5 w-3.5 text-[#0A2E65]" />
              <span>Tune Levers</span>
            </button>
          </div>
        </div>

        <div className="mt-3 grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-4">
          {[
            {
              feature: "USD_NGN_Planning_Rate",
              label: "USD/NGN Rate",
              shift: `${driverDeltas.fxPct >= 0 ? "+" : ""}${driverDeltas.fxPct.toFixed(1)}%`,
              baseVal: `₦${baselineInputs.USD_NGN_Planning_Rate.toLocaleString()}`,
              simVal: `₦${simulatedInputs.USD_NGN_Planning_Rate.toLocaleString()}`,
              changed: Math.abs(driverDeltas.fxPct) > 0.05,
            },
            {
              feature: "Avg_Selling_Price_NGN",
              label: "Avg Selling Price",
              shift: `${driverDeltas.pricePct >= 0 ? "+" : ""}${driverDeltas.pricePct.toFixed(1)}%`,
              baseVal: `₦${baselineInputs.Avg_Selling_Price_NGN.toLocaleString()}`,
              simVal: `₦${simulatedInputs.Avg_Selling_Price_NGN.toLocaleString()}`,
              changed: Math.abs(driverDeltas.pricePct) > 0.05,
            },
            {
              feature: "Units_Sold",
              label: "Units Sold",
              shift: `${driverDeltas.unitsPct >= 0 ? "+" : ""}${driverDeltas.unitsPct.toFixed(1)}%`,
              baseVal: baselineInputs.Units_Sold.toLocaleString(),
              simVal: simulatedInputs.Units_Sold.toLocaleString(),
              changed: Math.abs(driverDeltas.unitsPct) > 0.05,
            },
            {
              feature: "Import_Cost_NGN_Per_Unit",
              label: "Landed Import Cost/Unit",
              shift: `${driverDeltas.fxPct + driverDeltas.impUsdPct >= 0 ? "+" : ""}${(
                ((simulatedInputs.USD_NGN_Planning_Rate * simulatedInputs.Import_Cost_USD_Per_Unit) /
                  Math.max(1, baselineInputs.USD_NGN_Planning_Rate * baselineInputs.Import_Cost_USD_Per_Unit) -
                  1) *
                100
              ).toFixed(1)}%`,
              baseVal: `₦${Math.round(
                baselineInputs.USD_NGN_Planning_Rate * baselineInputs.Import_Cost_USD_Per_Unit
              ).toLocaleString()}`,
              simVal: `₦${Math.round(
                simulatedInputs.USD_NGN_Planning_Rate * simulatedInputs.Import_Cost_USD_Per_Unit
              ).toLocaleString()}`,
              changed: Math.abs(driverDeltas.fxPct) > 0.05 || Math.abs(driverDeltas.impUsdPct) > 0.05,
            },
            {
              feature: "Unit_Spread_NGN",
              label: "Unit Spread (Price − Import)",
              shift: isScenarioActive ? "Derived" : "0.0%",
              baseVal: `₦${Math.round(
                baselineInputs.Avg_Selling_Price_NGN -
                  baselineInputs.USD_NGN_Planning_Rate * baselineInputs.Import_Cost_USD_Per_Unit
              ).toLocaleString()}`,
              simVal: `₦${Math.round(
                simulatedInputs.Avg_Selling_Price_NGN -
                  simulatedInputs.USD_NGN_Planning_Rate * simulatedInputs.Import_Cost_USD_Per_Unit
              ).toLocaleString()}`,
              changed: isScenarioActive,
            },
            {
              feature: "Marketing_Spend_NGN",
              label: "Marketing Spend",
              shift: `${driverDeltas.mktPct >= 0 ? "+" : ""}${driverDeltas.mktPct.toFixed(1)}%`,
              baseVal: formatNGN(baselineInputs.Marketing_Spend_NGN, true),
              simVal: formatNGN(simulatedInputs.Marketing_Spend_NGN, true),
              changed: Math.abs(driverDeltas.mktPct) > 0.05,
            },
            {
              feature: "Other_Operating_Cost_NGN",
              label: "Other Operating Cost",
              shift: `${driverDeltas.opexPct >= 0 ? "+" : ""}${driverDeltas.opexPct.toFixed(1)}%`,
              baseVal: formatNGN(baselineInputs.Other_Operating_Cost_NGN, true),
              simVal: formatNGN(simulatedInputs.Other_Operating_Cost_NGN, true),
              changed: Math.abs(driverDeltas.opexPct) > 0.05,
            },
            {
              feature: "Inflation_YoY_Pct",
              label: "Inflation YoY",
              shift: `${driverDeltas.inflationDeltaPp >= 0 ? "+" : ""}${driverDeltas.inflationDeltaPp.toFixed(1)} pp`,
              baseVal: `${baselineInputs.Inflation_YoY_Pct.toFixed(1)}%`,
              simVal: `${simulatedInputs.Inflation_YoY_Pct.toFixed(1)}%`,
              changed: Math.abs(driverDeltas.inflationDeltaPp) > 0.05,
            },
          ].map((item) => {
            const shapMatch = data.global_importance.find((g) => g.feature === item.feature);
            return (
              <div
                key={item.feature}
                className={`rounded-lg border p-3 ${
                  item.changed ? "border-[#0A2E65]/30 bg-slate-50/90" : "border-slate-200 bg-white"
                }`}
              >
                <div className="flex items-center justify-between gap-2 text-[11px]">
                  <span className="font-semibold text-[#0A1E3D]">{item.label}</span>
                  <span className="font-mono font-bold text-[#0A2E65] tabular-nums">{item.shift}</span>
                </div>
                <div className="mt-1.5 flex items-baseline justify-between text-xs tabular-nums">
                  <span className="text-slate-400 line-through font-mono text-[11px]">{item.baseVal}</span>
                  <span className="font-mono font-bold text-[#0A1E3D]">{item.simVal}</span>
                </div>
                <div className="mt-1.5 text-[11px] text-slate-500 tabular-nums">
                  {shapMatch
                    ? `SHAP Share: ${shapMatch.share_pct}% · Mean |SHAP| ${formatNGN(shapMatch.mean_abs_shap, true)}`
                    : "Core ML Sensitivity Feature"}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <GraphicalRagPanel
        defaultLens="profit_intelligence"
        title="Explainability & Operating Profit Graphical RAG"
      />
    </div>
  );
};
