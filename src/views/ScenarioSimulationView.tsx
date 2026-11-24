import React, { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { RotateCcw, SlidersHorizontal, TrendingUp } from "lucide-react";
import { SCENARIO_PRESETS, useScenario } from "../context/ScenarioContext";
import { Accounting5YrSimulator } from "../components/Accounting5YrSimulator";
import { FinancialRecord } from "../types/financial";
import { formatNGN, formatPct } from "../utils/format";
import {
  getSliderFillStyle,
  TRIAD_TOOLTIP_ITEM_STYLE,
  TRIAD_TOOLTIP_LABEL_STYLE,
  TRIAD_TOOLTIP_STYLE,
  TriadSvgDefs,
} from "../utils/chartTheme";

interface ScenarioSimulationViewProps {
  latestRecord: FinancialRecord | null;
  datasetName?: string;
  onNavigateToForecasting?: () => void;
  onNavigateToEngineering?: () => void;
}

export const ScenarioSimulationView: React.FC<ScenarioSimulationViewProps> = ({
  latestRecord,
  datasetName = "default_financial_data.csv",
  onNavigateToForecasting,
  onNavigateToEngineering,
}) => {
  const [simMode, setSimMode] = useState<"5yr" | "bridge">("5yr");
  const {
    presetId,
    activePresetLabel,
    anchorRecord,
    baselineInputs: baseline,
    simulatedInputs: simulated,
    setExactSimulatedInputs,
    applyPreset,
    resetScenario,
    singlePeriodResult: result,
    singleLoading: loading,
    forecastScenarios,
    customVerdict,
  } = useScenario();

  // Transform waterfall steps into floating bar [start, end]
  const waterfallData = React.useMemo(() => {
    if (!result) return [];
    let running = 0;
    return result.waterfall.map((w) => {
      if (w.type === "baseline") {
        running = w.value;
        return { step: w.step, range: [0, w.value], value: w.value, color: "url(#scen-wf-bar-navy-cobalt)" };
      }
      if (w.type === "total") {
        return {
          step: w.step,
          range: [0, w.value],
          value: w.value,
          color: w.value >= 0 ? "url(#scen-wf-bar-cobalt-teal)" : "url(#scen-wf-bar-negative)",
        };
      }
      const prev = running;
      running += w.value;
      const low = Math.min(prev, running);
      const high = Math.max(prev, running);
      return {
        step: w.step,
        range: [low, high],
        value: w.value,
        color: w.value >= 0 ? "url(#scen-wf-bar-cobalt-teal)" : "url(#scen-wf-bar-negative)",
      };
    });
  }, [result]);

  const base2030 = forecastScenarios?.cases.base.summary;
  const custom2030 = forecastScenarios?.cases.custom.summary;
  const cum2030Delta =
    base2030 && custom2030 ? custom2030.cumulative_profit_ngn - base2030.cumulative_profit_ngn : 0;

  return (
    <div className="space-y-5">
      {/* Mode Switcher */}
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-slate-200 bg-white px-4 py-3">
        <div className="flex items-center gap-1 rounded-lg bg-slate-100 p-1">
          <button
            type="button"
            onClick={() => setSimMode("5yr")}
            className={`rounded-md px-3.5 py-1.5 text-xs font-semibold transition-all whitespace-nowrap ${
              simMode === "5yr"
                ? "bg-gradient-to-r from-[#0A2E65] via-[#1D63B8] to-[#0D9488] text-white shadow-sm"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            5-Year Accounting Simulator (A–G Levers)
          </button>
          <button
            type="button"
            onClick={() => setSimMode("bridge")}
            className={`rounded-md px-3.5 py-1.5 text-xs font-semibold transition-all whitespace-nowrap ${
              simMode === "bridge"
                ? "bg-gradient-to-r from-[#0A2E65] via-[#1D63B8] to-[#0D9488] text-white shadow-sm"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            1-Period ML &amp; Accounting Bridge Waterfall
          </button>
        </div>

        {onNavigateToForecasting && (
          <button
            type="button"
            onClick={onNavigateToForecasting}
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs font-semibold text-[#0A1E3D] hover:bg-slate-100 whitespace-nowrap"
          >
            <TrendingUp className="h-3.5 w-3.5 text-[#0A2E65]" />
            <span>Open 2030 Forecasting Centre</span>
          </button>
        )}
      </div>

      {simMode === "5yr" ? (
        <Accounting5YrSimulator
          datasetName={datasetName}
          onNavigateToEngineering={onNavigateToEngineering}
        />
      ) : (
        <div className="space-y-6">
          <div className="rounded-xl border border-slate-200 bg-white p-5">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex items-center gap-2">
                <SlidersHorizontal className="h-4 w-4 text-[#0A2E65]" />
                <div>
                  <h2 className="font-display text-sm font-bold text-[#0A1E3D]">
                    What-If Business &amp; Macro Scenario Simulator ({activePresetLabel})
                  </h2>
                  <p className="text-xs text-slate-500">
                    Baseline anchored to {anchorRecord?.Month_Label || latestRecord?.Month_Label || "latest period"}. Synchronized live with 2030 Forecasting &amp; all diagnostic views.
                  </p>
                </div>
              </div>

              <div className="flex flex-wrap gap-1.5">
                {SCENARIO_PRESETS.filter((p) => p.id !== "base").map((p) => (
                  <button
                    key={p.id}
                    type="button"
                    onClick={() => applyPreset(p.id)}
                    className={`rounded-lg border px-2.5 py-1.5 text-xs font-semibold transition-all whitespace-nowrap ${
                      presetId === p.id
                        ? "border-transparent bg-gradient-to-r from-[#0A2E65] via-[#1D63B8] to-[#0D9488] text-white shadow-sm"
                        : "border-slate-200 bg-slate-50 text-slate-700 hover:bg-slate-100"
                    }`}
                  >
                    {p.shortLabel}
                  </button>
                ))}
                <button
                  type="button"
                  onClick={resetScenario}
                  className="inline-flex items-center gap-1 rounded-lg border border-slate-200 px-2.5 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50 whitespace-nowrap"
                >
                  <RotateCcw className="h-3 w-3" />
                  <span>Reset</span>
                </button>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-5 xl:grid-cols-12">
        {/* Left Sliders */}
        <div className="rounded-xl border border-slate-200 bg-white p-5 xl:col-span-5 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-bold text-[#0A1E3D]">Single-Period Driver Levers</h3>
            <span className="text-[11px] text-slate-500">
              Anchor: {anchorRecord?.Month_Label || "Latest"}
            </span>
          </div>
          {[
            {
              key: "USD_NGN_Planning_Rate" as const,
              label: "USD/NGN Exchange Rate (₦/$)",
              min: Math.round(baseline.USD_NGN_Planning_Rate * 0.6),
              max: Math.round(baseline.USD_NGN_Planning_Rate * 1.6),
              step: 5,
            },
            {
              key: "Inflation_YoY_Pct" as const,
              label: "Inflation YoY (%)",
              min: 5,
              max: 45,
              step: 0.5,
            },
            {
              key: "Units_Sold" as const,
              label: "Units Sold",
              min: Math.round(baseline.Units_Sold * 0.5),
              max: Math.round(baseline.Units_Sold * 1.6),
              step: 50,
            },
            {
              key: "Import_Cost_USD_Per_Unit" as const,
              label: "Import Cost (USD / Unit)",
              min: 5,
              max: 20,
              step: 0.1,
            },
            {
              key: "Avg_Selling_Price_NGN" as const,
              label: "Avg Selling Price (NGN / Unit)",
              min: Math.round(baseline.Avg_Selling_Price_NGN * 0.6),
              max: Math.round(baseline.Avg_Selling_Price_NGN * 1.6),
              step: 100,
            },
            {
              key: "Marketing_Spend_NGN" as const,
              label: "Marketing Spend (NGN)",
              min: Math.round(baseline.Marketing_Spend_NGN * 0.4),
              max: Math.round(baseline.Marketing_Spend_NGN * 2.0),
              step: 100000,
            },
            {
              key: "Other_Operating_Cost_NGN" as const,
              label: "Other Operating Costs (NGN)",
              min: Math.round(baseline.Other_Operating_Cost_NGN * 0.5),
              max: Math.round(baseline.Other_Operating_Cost_NGN * 1.8),
              step: 100000,
            },
          ].map((ctrl) => {
            const val = Number(simulated[ctrl.key]);
            return (
              <div key={ctrl.key} className="space-y-1">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-medium text-slate-700">{ctrl.label}</span>
                  <input
                    type="number"
                    step={ctrl.step}
                    value={val}
                    onChange={(e) =>
                      setExactSimulatedInputs({ ...simulated, [ctrl.key]: Number(e.target.value) })
                    }
                    className="w-28 rounded border border-slate-200 bg-slate-50 px-2 py-0.5 text-right font-mono text-xs tabular-nums"
                  />
                </div>
                <input
                  type="range"
                  min={ctrl.min}
                  max={ctrl.max}
                  step={ctrl.step}
                  value={val}
                  onChange={(e) =>
                    setExactSimulatedInputs({ ...simulated, [ctrl.key]: Number(e.target.value) })
                  }
                  style={getSliderFillStyle(val, ctrl.min, ctrl.max)}
                  className="triad-slider w-full"
                />
              </div>
            );
          })}
        </div>

        {/* Right Comparison & Waterfall */}
        <div className="xl:col-span-7 space-y-5">
          {result && (
            <>
              <div className="rounded-xl border border-slate-200 bg-white p-5">
                <div className="flex items-center justify-between">
                  <h3 className="font-display text-sm font-bold text-[#0A1E3D]">
                    Baseline vs. Simulated Financial Outcomes (1-Period Driver Bridge)
                  </h3>
                  {loading && <span className="text-[11px] text-slate-400">Updating...</span>}
                </div>

                <div className="mt-3 overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead>
                      <tr className="border-b border-slate-200 text-slate-500">
                        <th className="py-2 pr-4 font-semibold">Metric</th>
                        <th className="py-2 pr-4 text-right font-semibold">Baseline</th>
                        <th className="py-2 pr-4 text-right font-semibold">Simulated</th>
                        <th className="py-2 text-right font-semibold">Delta</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      <tr>
                        <td className="py-2 pr-4 font-medium">Import Cost / Unit (NGN)</td>
                        <td className="py-2 pr-4 text-right font-mono tabular-nums">
                          {formatNGN(result.baseline.accounting.Import_Cost_NGN_Per_Unit)}
                        </td>
                        <td className="py-2 pr-4 text-right font-mono tabular-nums">
                          {formatNGN(result.simulated.accounting.Import_Cost_NGN_Per_Unit)}
                        </td>
                        <td className="py-2 text-right font-mono tabular-nums">
                          {formatNGN(
                            result.simulated.accounting.Import_Cost_NGN_Per_Unit -
                              result.baseline.accounting.Import_Cost_NGN_Per_Unit
                          )}
                        </td>
                      </tr>
                      <tr>
                        <td className="py-2 pr-4 font-medium">COGS (NGN)</td>
                        <td className="py-2 pr-4 text-right font-mono tabular-nums">
                          {formatNGN(result.baseline.accounting.COGS_NGN, true)}
                        </td>
                        <td className="py-2 pr-4 text-right font-mono tabular-nums">
                          {formatNGN(result.simulated.accounting.COGS_NGN, true)}
                        </td>
                        <td className="py-2 text-right font-mono tabular-nums">
                          {formatNGN(result.deltas.cogs_delta, true)}
                        </td>
                      </tr>
                      <tr>
                        <td className="py-2 pr-4 font-medium">Sales Revenue (NGN)</td>
                        <td className="py-2 pr-4 text-right font-mono tabular-nums">
                          {formatNGN(result.baseline.accounting.Sales_Revenue_NGN, true)}
                        </td>
                        <td className="py-2 pr-4 text-right font-mono tabular-nums">
                          {formatNGN(result.simulated.accounting.Sales_Revenue_NGN, true)}
                        </td>
                        <td className="py-2 text-right font-mono tabular-nums">
                          {formatNGN(result.deltas.revenue_delta, true)}
                        </td>
                      </tr>
                      <tr>
                        <td className="py-2 pr-4 font-medium">Gross Profit &amp; Margin</td>
                        <td className="py-2 pr-4 text-right font-mono tabular-nums">
                          {formatNGN(result.baseline.accounting.Gross_Profit_NGN, true)} (
                          {formatPct(result.baseline.accounting.Gross_Margin_Pct, 1)})
                        </td>
                        <td className="py-2 pr-4 text-right font-mono tabular-nums">
                          {formatNGN(result.simulated.accounting.Gross_Profit_NGN, true)} (
                          {formatPct(result.simulated.accounting.Gross_Margin_Pct, 1)})
                        </td>
                        <td className="py-2 text-right font-mono tabular-nums">
                          {formatPct(result.deltas.gross_margin_pp_delta, 2, true)}
                        </td>
                      </tr>
                      <tr className="bg-slate-50 font-semibold">
                        <td className="py-2.5 pr-4 text-[#0A1E3D]">
                          Accounting Operating Profit
                        </td>
                        <td className="py-2.5 pr-4 text-right font-mono tabular-nums">
                          {formatNGN(result.baseline.accounting.Accounting_Operating_Profit_NGN, true)}
                        </td>
                        <td className="py-2.5 pr-4 text-right font-mono tabular-nums">
                          {formatNGN(result.simulated.accounting.Accounting_Operating_Profit_NGN, true)}
                        </td>
                        <td
                          className={`py-2.5 text-right font-mono tabular-nums ${
                            result.deltas.accounting_profit_delta >= 0
                              ? "text-teal-700"
                              : "text-red-600"
                          }`}
                        >
                          {formatNGN(result.deltas.accounting_profit_delta, true)}
                        </td>
                      </tr>
                      <tr className="bg-blue-50/50 font-semibold">
                        <td className="py-2.5 pr-4 text-[#0A2E65]">
                          ML Predicted Operating Profit
                        </td>
                        <td className="py-2.5 pr-4 text-right font-mono tabular-nums">
                          {formatNGN(result.baseline.ml_predicted_operating_profit_ngn, true)}
                        </td>
                        <td className="py-2.5 pr-4 text-right font-mono tabular-nums">
                          {formatNGN(result.simulated.ml_predicted_operating_profit_ngn, true)}
                        </td>
                        <td
                          className={`py-2.5 text-right font-mono tabular-nums ${
                            result.deltas.ml_predicted_profit_delta >= 0
                              ? "text-teal-700"
                              : "text-red-600"
                          }`}
                        >
                          {formatNGN(result.deltas.ml_predicted_profit_delta, true)}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Linked 2030 Multi-Year Cumulative Horizon Card */}
              {custom2030 && base2030 && (
                <div className="rounded-xl border border-slate-200 bg-white p-5 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                  <div className="space-y-1">
                    <div className="text-xs font-bold text-[#0A1E3D]">
                      Synchronized 2030 Multi-Year Horizon Outcome ({activePresetLabel})
                    </div>
                    <div className="flex flex-wrap items-baseline gap-4 text-xs tabular-nums">
                      <span>
                        Dec 2030:{" "}
                        <strong className="font-mono text-[#0A1E3D]">
                          {formatNGN(custom2030.dec_2030_operating_profit_ngn, true)}
                        </strong>
                      </span>
                      <span>
                        2026–2030 Cumulative:{" "}
                        <strong className="font-mono text-[#0A1E3D]">
                          {formatNGN(custom2030.cumulative_profit_ngn, true)}
                        </strong>{" "}
                        <span
                          className={`font-mono font-semibold ${
                            cum2030Delta >= 0 ? "text-teal-700" : "text-red-600"
                          }`}
                        >
                          ({cum2030Delta >= 0 ? "+" : ""}
                          {formatNGN(cum2030Delta, true)} vs Base)
                        </span>
                      </span>
                      {customVerdict && (
                        <span>
                          Verdict: <strong className="text-[#0A2E65]">{customVerdict.category}</strong>
                        </span>
                      )}
                    </div>
                  </div>
                  {onNavigateToForecasting && (
                    <button
                      type="button"
                      onClick={onNavigateToForecasting}
                      className="triad-btn-primary inline-flex items-center gap-1.5 rounded-lg px-3.5 py-2 text-xs font-semibold text-white whitespace-nowrap shrink-0"
                    >
                      <TrendingUp className="h-3.5 w-3.5" />
                      <span>Open 2030 Lab</span>
                    </button>
                  )}
                </div>
              )}

              {/* Profit Bridge Waterfall Chart */}
              <div className="rounded-xl border border-slate-200 bg-white p-5">
                <h3 className="font-display text-sm font-bold text-[#0A1E3D]">
                  Operating Profit Bridge Waterfall (Baseline → Simulated)
                </h3>
                <p className="mt-0.5 text-xs text-slate-500">
                  Decomposes accounting operating profit movement into Volume, Pricing, COGS (FX/Import), Marketing, and OpEx effects.
                </p>
                <div className="mt-3 h-[250px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={waterfallData} margin={{ top: 10, right: 16, left: 4, bottom: 16 }}>
                      <TriadSvgDefs idPrefix="scen-wf" />
                      <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                      <XAxis dataKey="step" tick={{ fontSize: 9.5, fill: "#334155" }} interval={0} />
                      <YAxis
                        tick={{ fontSize: 10, fill: "#64748B" }}
                        tickFormatter={(v) => `₦${(v / 1e6).toFixed(0)}M`}
                        width={54}
                      />
                      <Tooltip
                        contentStyle={TRIAD_TOOLTIP_STYLE}
                        labelStyle={TRIAD_TOOLTIP_LABEL_STYLE}
                        itemStyle={TRIAD_TOOLTIP_ITEM_STYLE}
                        formatter={(_val: any, _name: any, item: any) => [
                          formatNGN(item?.payload?.value),
                          item?.payload?.step,
                        ]}
                      />
                      <ReferenceLine y={0} stroke="#64748B" />
                      <Bar dataKey="range" radius={[4, 4, 4, 4]}>
                        {waterfallData.map((entry) => (
                          <Cell key={entry.step} fill={entry.color} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
        </div>
      )}
    </div>
  );
};
