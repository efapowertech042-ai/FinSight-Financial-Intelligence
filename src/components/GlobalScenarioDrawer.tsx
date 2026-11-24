import React, { useState } from "react";
import { ArrowRight, RotateCcw, SlidersHorizontal, TrendingUp, X } from "lucide-react";
import { SCENARIO_PRESETS, useScenario } from "../context/ScenarioContext";
import { formatNGN, formatPct } from "../utils/format";
import { SectionId } from "./Sidebar";
import { getSliderFillStyle } from "../utils/chartTheme";

interface GlobalScenarioDrawerProps {
  onNavigate: (section: SectionId) => void;
}

export const GlobalScenarioDrawer: React.FC<GlobalScenarioDrawerProps> = ({ onNavigate }) => {
  const {
    presetId,
    isScenarioActive,
    anchorRecord,
    forecastAssumptions,
    setForecastAssumptions,
    driverDeltas,
    setDriverDeltas,
    applyPreset,
    resetScenario,
    forecastScenarios,
    customVerdict,
    singlePeriodResult,
    forecastLoading,
    singleLoading,
    drawerOpen,
    setDrawerOpen,
  } = useScenario();

  const [tab, setTab] = useState<"horizon2030" | "singlePeriod">("horizon2030");

  if (!drawerOpen) return null;

  const baseCase = forecastScenarios?.cases.base.summary;
  const customCase = forecastScenarios?.cases.custom.summary;
  const dec2030Delta =
    baseCase && customCase
      ? customCase.dec_2030_operating_profit_ngn - baseCase.dec_2030_operating_profit_ngn
      : 0;
  const cum2030Delta =
    baseCase && customCase ? customCase.cumulative_profit_ngn - baseCase.cumulative_profit_ngn : 0;

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <div
        className="fixed inset-0 bg-slate-950/40 transition-opacity"
        onClick={() => setDrawerOpen(false)}
        aria-hidden="true"
      />
      <aside
        aria-label="Global Interactive Scenario Simulator"
        className="relative z-10 flex h-full w-full max-w-[460px] flex-col bg-white shadow-xl border-l border-slate-200"
      >
        {/* Drawer Header */}
        <div className="flex items-start justify-between gap-3 border-b border-slate-200 bg-gradient-to-r from-[#071D3F] via-[#11468A] to-[#0B6F73] px-5 py-4 text-white">
          <div>
            <div className="flex items-center gap-2 text-xs font-semibold text-teal-300">
              <SlidersHorizontal className="h-3.5 w-3.5" />
              <span>Synchronized Cross-Page Engine</span>
            </div>
            <h2 className="mt-0.5 font-display text-base font-bold">
              Interactive Scenario Simulator
            </h2>
            <p className="mt-0.5 text-[11px] text-slate-300">
              Adjusting presets or levers updates 2030 Forecasting and all diagnostic pages live.
            </p>
          </div>
          <button
            type="button"
            onClick={() => setDrawerOpen(false)}
            aria-label="Close Scenario Simulator"
            className="rounded-lg p-1.5 text-slate-300 hover:bg-white/10 hover:text-white"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Scrollable Content */}
        <div className="flex-1 overflow-y-auto p-5 space-y-5">
          {/* Scenario Presets */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-bold text-[#0A1E3D]">Strategic Presets</span>
              {isScenarioActive && (
                <button
                  type="button"
                  onClick={resetScenario}
                  className="inline-flex items-center gap-1 text-[11px] font-semibold text-slate-600 hover:text-[#0A2E65]"
                >
                  <RotateCcw className="h-3 w-3" />
                  <span>Reset to Base</span>
                </button>
              )}
            </div>
            <div className="grid grid-cols-2 gap-2">
              {SCENARIO_PRESETS.map((p) => {
                const selected = presetId === p.id;
                return (
                  <button
                    key={p.id}
                    type="button"
                    onClick={() => applyPreset(p.id)}
                    className={`rounded-lg border px-3 py-2 text-left transition-colors ${
                      selected
                        ? "border-[#0A2E65] bg-[#0A2E65]/5 text-[#0A1E3D]"
                        : "border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
                    }`}
                  >
                    <div className="text-xs font-bold">{p.shortLabel}</div>
                    <div className="mt-0.5 text-[10px] text-slate-500 line-clamp-2">
                      {p.description}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Live Cross-Horizon Outcome Strip */}
          <div className="grid grid-cols-2 gap-3 rounded-xl border border-slate-200 bg-slate-50 p-3.5">
            <div>
              <div className="text-[11px] font-medium text-slate-500">
                Dec 2030 Simulated Profit
              </div>
              <div className="mt-0.5 font-display text-base font-bold text-[#0A1E3D] tabular-nums">
                {customCase ? formatNGN(customCase.dec_2030_operating_profit_ngn, true) : "—"}
              </div>
              <div
                className={`text-[11px] font-mono font-semibold tabular-nums ${
                  dec2030Delta >= 0 ? "text-teal-700" : "text-red-600"
                }`}
              >
                {dec2030Delta >= 0 ? "+" : ""}
                {formatNGN(dec2030Delta, true)} vs Base
              </div>
            </div>

            <div>
              <div className="text-[11px] font-medium text-slate-500">
                2026–2030 Cumulative Delta
              </div>
              <div
                className={`mt-0.5 font-display text-base font-bold tabular-nums ${
                  cum2030Delta >= 0 ? "text-teal-700" : "text-red-600"
                }`}
              >
                {cum2030Delta >= 0 ? "+" : ""}
                {formatNGN(cum2030Delta, true)}
              </div>
              <div className="text-[11px] text-slate-600 tabular-nums">
                Verdict: <strong className="text-[#0A1E3D]">{customVerdict?.category || "Base"}</strong>
              </div>
            </div>
          </div>

          {/* Mode Switcher */}
          <div className="flex rounded-lg bg-slate-100 p-1 text-xs font-semibold">
            <button
              type="button"
              onClick={() => setTab("horizon2030")}
              className={`flex-1 rounded-md py-1.5 transition-colors whitespace-nowrap ${
                tab === "horizon2030" ? "bg-white text-[#0A1E3D] shadow-sm" : "text-slate-600"
              }`}
            >
              2030 Multi-Year Levers
            </button>
            <button
              type="button"
              onClick={() => setTab("singlePeriod")}
              className={`flex-1 rounded-md py-1.5 transition-colors whitespace-nowrap ${
                tab === "singlePeriod" ? "bg-white text-[#0A1E3D] shadow-sm" : "text-slate-600"
              }`}
            >
              Driver Shock ({anchorRecord?.Month_Label || "Latest"})
            </button>
          </div>

          {tab === "horizon2030" ? (
            <div className="space-y-3.5">
              {[
                {
                  key: "revenue_growth_adj_pp" as const,
                  label: "Annual Revenue Growth Adj (pp/yr)",
                  min: -10,
                  max: 12,
                  step: 0.5,
                  suffix: " pp/yr",
                },
                {
                  key: "opex_growth_adj_pp" as const,
                  label: "Annual OpEx Growth Adj (pp/yr)",
                  min: -8,
                  max: 14,
                  step: 0.5,
                  suffix: " pp/yr",
                },
                {
                  key: "margin_efficiency_adj_pp" as const,
                  label: "Operating Margin Efficiency Shift (pp)",
                  min: -6,
                  max: 6,
                  step: 0.5,
                  suffix: " pp",
                },
                {
                  key: "fx_cost_pressure_adj_pct" as const,
                  label: "FX / Import Cost Shock (%)",
                  min: -10,
                  max: 25,
                  step: 1,
                  suffix: "%",
                },
                {
                  key: "volume_productivity_adj_pct" as const,
                  label: "Volume & Throughput Shift (%)",
                  min: -15,
                  max: 20,
                  step: 1,
                  suffix: "%",
                },
              ].map((s) => {
                const v = forecastAssumptions[s.key];
                return (
                  <div key={s.key} className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="font-medium text-slate-700">{s.label}</span>
                      <span className="font-mono font-semibold text-[#0A1E3D] tabular-nums">
                        {v >= 0 ? "+" : ""}
                        {v}
                        {s.suffix}
                      </span>
                    </div>
                    <input
                      type="range"
                      min={s.min}
                      max={s.max}
                      step={s.step}
                      value={v}
                      style={getSliderFillStyle(v, s.min, s.max)}
                      onChange={(e) =>
                        setForecastAssumptions({
                          ...forecastAssumptions,
                          [s.key]: Number(e.target.value),
                        })
                      }
                      className="triad-slider w-full"
                    />
                  </div>
                );
              })}
              {forecastLoading && (
                <div className="text-[11px] text-slate-500">Recalculating 2030 trajectory...</div>
              )}
            </div>
          ) : (
            <div className="space-y-3.5">
              {[
                { key: "fxPct" as const, label: "USD/NGN FX Rate Shock (%)", min: -30, max: 40, step: 1, suffix: "%" },
                { key: "pricePct" as const, label: "Selling Price Adjustment (%)", min: -25, max: 35, step: 1, suffix: "%" },
                { key: "unitsPct" as const, label: "Sales Volume Shift (%)", min: -35, max: 40, step: 1, suffix: "%" },
                { key: "impUsdPct" as const, label: "Import USD Cost Shift (%)", min: -25, max: 30, step: 1, suffix: "%" },
                { key: "mktPct" as const, label: "Marketing Spend Shift (%)", min: -40, max: 50, step: 1, suffix: "%" },
                { key: "opexPct" as const, label: "Other OpEx Shift (%)", min: -40, max: 50, step: 1, suffix: "%" },
              ].map((s) => {
                const v = driverDeltas[s.key];
                return (
                  <div key={s.key} className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="font-medium text-slate-700">{s.label}</span>
                      <span className="font-mono font-semibold text-[#0A1E3D] tabular-nums">
                        {v >= 0 ? "+" : ""}
                        {v}
                        {s.suffix}
                      </span>
                    </div>
                    <input
                      type="range"
                      min={s.min}
                      max={s.max}
                      step={s.step}
                      value={v}
                      style={getSliderFillStyle(v, s.min, s.max)}
                      onChange={(e) =>
                        setDriverDeltas({
                          ...driverDeltas,
                          [s.key]: Number(e.target.value),
                        })
                      }
                      className="triad-slider w-full"
                    />
                  </div>
                );
              })}

              {singlePeriodResult && (
                <div className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-xs space-y-1">
                  <div className="flex justify-between">
                    <span className="text-slate-500">Single-Period Simulated Profit:</span>
                    <span className="font-mono font-bold text-[#0A1E3D] tabular-nums">
                      {formatNGN(
                        singlePeriodResult.simulated.accounting.Accounting_Operating_Profit_NGN,
                        true
                      )}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Delta vs Baseline:</span>
                    <span
                      className={`font-mono font-bold tabular-nums ${
                        singlePeriodResult.deltas.accounting_profit_delta >= 0
                          ? "text-teal-700"
                          : "text-red-600"
                      }`}
                    >
                      {singlePeriodResult.deltas.accounting_profit_delta >= 0 ? "+" : ""}
                      {formatNGN(singlePeriodResult.deltas.accounting_profit_delta, true)} (
                      {formatPct(singlePeriodResult.deltas.operating_margin_pp_delta, 2, true)} margin)
                    </span>
                  </div>
                  {singleLoading && (
                    <div className="text-[11px] text-slate-400">Updating driver bridge...</div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer Quick Navigation */}
        <div className="border-t border-slate-200 bg-slate-50 px-5 py-3.5 flex items-center justify-between gap-2">
          <button
            type="button"
            onClick={() => {
              onNavigate("forecasting");
              setDrawerOpen(false);
            }}
            className="triad-btn-primary inline-flex flex-1 items-center justify-center gap-1.5 rounded-lg px-3 py-2 text-xs font-semibold text-white whitespace-nowrap"
          >
            <TrendingUp className="h-3.5 w-3.5" />
            <span>Open 2030 Simulator</span>
            <ArrowRight className="h-3.5 w-3.5" />
          </button>
          <button
            type="button"
            onClick={() => {
              onNavigate("scenario");
              setDrawerOpen(false);
            }}
            className="inline-flex items-center justify-center rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-100 whitespace-nowrap"
          >
            Full What-If View
          </button>
        </div>
      </aside>
    </div>
  );
};
