import React, { useState } from "react";
import { Network, SlidersHorizontal } from "lucide-react";
import { GraphicalRagPanel } from "../components/GraphicalRagPanel";
import { useScenario } from "../context/ScenarioContext";
import { formatNGN } from "../utils/format";

interface GraphicalRagViewProps {
  selectedMonth: string | null;
}

export const GraphicalRagView: React.FC<GraphicalRagViewProps> = ({ selectedMonth }) => {
  const {
    activePresetLabel,
    driverDeltas,
    forecastAssumptions,
    forecastScenarios,
    customVerdict,
    singlePeriodResult,
    setDrawerOpen,
  } = useScenario();

  const [trigger, setTrigger] = useState<{
    question: string;
    lens?: "fx_intelligence" | "forecast_intelligence" | "profit_intelligence";
    seq: number;
  } | null>(null);

  const custom2030 = forecastScenarios?.cases?.custom?.summary;

  const handleGroundScenario = (focus: "fx" | "forecast") => {
    const q =
      focus === "fx"
        ? `How did historical FX depreciation episodes like Jun 2023 impact import costs and operating profit compared to a ${driverDeltas.fxPct >= 0 ? "+" : ""}${driverDeltas.fxPct.toFixed(1)}% FX shock and ${driverDeltas.pricePct >= 0 ? "+" : ""}${driverDeltas.pricePct.toFixed(1)}% price change?`
        : `What is the 2030 forecast verdict and hypothesis p-value under ${activePresetLabel} (revenue ${forecastAssumptions.revenue_growth_adj_pp >= 0 ? "+" : ""}${forecastAssumptions.revenue_growth_adj_pp} pp/yr, FX cost ${forecastAssumptions.fx_cost_pressure_adj_pct >= 0 ? "+" : ""}${forecastAssumptions.fx_cost_pressure_adj_pct}%)?`;

    setTrigger((prev) => ({
      question: q,
      lens: focus === "fx" ? "fx_intelligence" : "forecast_intelligence",
      seq: (prev?.seq || 0) + 1,
    }));
  };

  return (
    <div className="space-y-5">
      <div className="rounded-xl border border-slate-200 bg-white p-5">
        <h2 className="font-display text-sm font-bold text-[#0A1E3D]">
          Evidence-Grounded Graphical Retrieval-Augmented Generation
        </h2>
        <p className="mt-1 text-xs leading-relaxed text-slate-600">
          Every explanation below performs live multi-store lookups across the original dataset ledger, Market Notes,
          recalculated financial formulas, chronological ML benchmarks, SHAP attributions, Isolation Forest anomalies,
          and the 2030 Forecasting &amp; Hypothesis Centre—and displays the exact retrieval trace and source rows used.
        </p>
      </div>

      {/* Active Scenario Context & Grounded Lookup Bar */}
      <div className="rounded-xl border border-slate-200 bg-white p-4">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <div className="flex items-center gap-2 text-xs font-bold text-[#0A1E3D]">
              <Network className="h-4 w-4 text-[#0A2E65]" />
              <span>Active Scenario Context: {activePresetLabel}</span>
            </div>
            <div className="mt-1 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-slate-600 tabular-nums">
              <span>
                FX Shock:{" "}
                <strong className="font-mono text-[#0A1E3D]">
                  {driverDeltas.fxPct >= 0 ? "+" : ""}
                  {driverDeltas.fxPct.toFixed(1)}%
                </strong>
              </span>
              <span>
                Price Adj:{" "}
                <strong className="font-mono text-[#0A1E3D]">
                  {driverDeltas.pricePct >= 0 ? "+" : ""}
                  {driverDeltas.pricePct.toFixed(1)}%
                </strong>
              </span>
              {singlePeriodResult && (
                <span>
                  1M ML Δ:{" "}
                  <strong
                    className={`font-mono ${
                      singlePeriodResult.deltas.ml_predicted_profit_delta >= 0 ? "text-teal-700" : "text-red-600"
                    }`}
                  >
                    {singlePeriodResult.deltas.ml_predicted_profit_delta >= 0 ? "+" : ""}
                    {formatNGN(singlePeriodResult.deltas.ml_predicted_profit_delta, true)}
                  </strong>
                </span>
              )}
              {custom2030 && (
                <span>
                  Dec 2030 Sim:{" "}
                  <strong className="font-mono text-[#0A1E3D]">
                    {formatNGN(custom2030.dec_2030_operating_profit_ngn, true)}
                  </strong>
                </span>
              )}
              {customVerdict && (
                <span>
                  2030 Verdict: <strong className="text-[#0A2E65]">{customVerdict.category}</strong>
                </span>
              )}
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2 shrink-0">
            <button
              type="button"
              onClick={() => handleGroundScenario("fx")}
              className="rounded-lg bg-[#0A2E65] px-3 py-1.5 text-xs font-semibold text-white hover:bg-[#0D3B82] whitespace-nowrap"
            >
              Ground Scenario FX Shock
            </button>
            <button
              type="button"
              onClick={() => handleGroundScenario("forecast")}
              className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-semibold text-[#0A1E3D] hover:bg-slate-50 whitespace-nowrap"
            >
              Ground 2030 Verdict
            </button>
            <button
              type="button"
              onClick={() => setDrawerOpen(true)}
              className="inline-flex items-center gap-1 rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs font-semibold text-slate-700 hover:bg-slate-100 whitespace-nowrap"
            >
              <SlidersHorizontal className="h-3.5 w-3.5" />
              <span>Tune</span>
            </button>
          </div>
        </div>
      </div>

      <GraphicalRagPanel
        defaultLens="profit_intelligence"
        selectedMonth={selectedMonth}
        title="Interactive Financial Knowledge Graph & Evidence Explorer"
        externalTrigger={trigger}
      />
    </div>
  );
};

