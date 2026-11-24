import React from "react";
import { RotateCcw, SlidersHorizontal, TrendingUp } from "lucide-react";
import { SCENARIO_PRESETS, useScenario } from "../context/ScenarioContext";
import { formatNGN, formatPct } from "../utils/format";
import { SectionId } from "./Sidebar";

interface GlobalScenarioBarProps {
  activeSection: SectionId;
  onNavigate: (section: SectionId) => void;
}

export const GlobalScenarioBar: React.FC<GlobalScenarioBarProps> = ({
  activeSection,
  onNavigate,
}) => {
  const {
    presetId,
    isScenarioActive,
    applyPreset,
    resetScenario,
    forecastScenarios,
    customVerdict,
    singlePeriodResult,
    setDrawerOpen,
  } = useScenario();

  const baseSummary = forecastScenarios?.cases.base.summary;
  const customSummary = forecastScenarios?.cases.custom.summary;
  const dec2030Delta =
    baseSummary && customSummary
      ? customSummary.dec_2030_operating_profit_ngn - baseSummary.dec_2030_operating_profit_ngn
      : 0;
  const cum2030Delta =
    baseSummary && customSummary
      ? customSummary.cumulative_profit_ngn - baseSummary.cumulative_profit_ngn
      : 0;

  return (
    <div className="mb-5 rounded-xl border border-slate-200 bg-white px-4 py-3 flex flex-col gap-3 xl:flex-row xl:items-center xl:justify-between">
      <div className="flex flex-wrap items-center gap-2">
        <div className="flex items-center gap-1.5 text-xs font-bold text-[#0A1E3D] mr-1">
          <SlidersHorizontal className="h-3.5 w-3.5 text-[#0D9488]" />
          <span>Interactive Scenario:</span>
        </div>

        <div className="flex flex-wrap items-center gap-1 rounded-lg bg-slate-100 p-1">
          {SCENARIO_PRESETS.map((p) => {
            const selected = presetId === p.id;
            return (
              <button
                key={p.id}
                type="button"
                onClick={() => applyPreset(p.id)}
                className={`rounded-md px-2.5 py-1 text-[11px] font-semibold transition-colors whitespace-nowrap ${
                  selected
                    ? "bg-gradient-to-r from-[#0A2E65] via-[#1D63B8] to-[#0D9488] text-white shadow-sm"
                    : "text-slate-600 hover:text-slate-900"
                }`}
              >
                {p.shortLabel}
              </button>
            );
          })}
          {presetId === "custom" && (
            <span className="rounded-md bg-gradient-to-r from-[#0A2E65] via-[#1D63B8] to-[#0D9488] px-2.5 py-1 text-[11px] font-semibold text-white whitespace-nowrap">
              Custom
            </span>
          )}
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-4 text-xs">
        {customSummary && (
          <div className="flex flex-wrap items-center gap-4 tabular-nums">
            <div>
              <span className="text-slate-500">Dec 2030: </span>
              <strong className="font-mono text-[#0A1E3D]">
                {formatNGN(customSummary.dec_2030_operating_profit_ngn, true)}
              </strong>
              {isScenarioActive && (
                <span
                  className={`ml-1 font-mono font-semibold ${
                    dec2030Delta >= 0 ? "text-teal-700" : "text-red-600"
                  }`}
                >
                  ({dec2030Delta >= 0 ? "+" : ""}
                  {formatNGN(dec2030Delta, true)})
                </span>
              )}
            </div>

            <div>
              <span className="text-slate-500">2026–2030 Δ: </span>
              <strong
                className={`font-mono ${
                  !isScenarioActive
                    ? "text-slate-700"
                    : cum2030Delta >= 0
                      ? "text-teal-700"
                      : "text-red-600"
                }`}
              >
                {cum2030Delta >= 0 ? "+" : ""}
                {formatNGN(cum2030Delta, true)}
              </strong>
            </div>

            {singlePeriodResult && isScenarioActive && (
              <div>
                <span className="text-slate-500">1M Δ: </span>
                <strong
                  className={`font-mono ${
                    singlePeriodResult.deltas.accounting_profit_delta >= 0
                      ? "text-teal-700"
                      : "text-red-600"
                  }`}
                >
                  {singlePeriodResult.deltas.accounting_profit_delta >= 0 ? "+" : ""}
                  {formatNGN(singlePeriodResult.deltas.accounting_profit_delta, true)} (
                  {formatPct(singlePeriodResult.deltas.operating_margin_pp_delta, 1, true)})
                </strong>
              </div>
            )}

            {customVerdict && (
              <div className="hidden md:block">
                <span className="text-slate-500">Outlook: </span>
                <strong className="text-[#0A2E65]">{customVerdict.category}</strong>
              </div>
            )}
          </div>
        )}

        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={() => setDrawerOpen(true)}
            className="triad-btn-primary inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-[11px] font-semibold text-white whitespace-nowrap"
          >
            <SlidersHorizontal className="h-3 w-3" />
            <span>Tune Levers</span>
          </button>

          {activeSection !== "forecasting" && (
            <button
              type="button"
              onClick={() => onNavigate("forecasting")}
              className="inline-flex items-center gap-1 rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-[11px] font-semibold text-[#0A1E3D] hover:bg-slate-50 whitespace-nowrap"
            >
              <TrendingUp className="h-3 w-3 text-teal-700" />
              <span>2030 Lab</span>
            </button>
          )}

          {isScenarioActive && (
            <button
              type="button"
              onClick={resetScenario}
              title="Reset scenario to Base Model"
              className="inline-flex items-center gap-1 rounded-lg border border-slate-200 px-2.5 py-1.5 text-[11px] font-semibold text-slate-600 hover:bg-slate-50 whitespace-nowrap"
            >
              <RotateCcw className="h-3 w-3" />
              <span>Reset</span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
