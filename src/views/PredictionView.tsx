import React, { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { AlertTriangle, Cpu, Gauge, Layers, ShieldCheck, Sliders, TrendingUp } from "lucide-react";
import { api } from "../services/api";
import {
  FinancialRecord,
  ModelsSummaryResponse,
  PredictionInputs,
  PredictionResponse,
} from "../types/financial";
import { formatNGN, formatPct } from "../utils/format";
import { useScenario } from "../context/ScenarioContext";
import {
  getSliderFillStyle,
  TRIAD_TOOLTIP_ITEM_STYLE,
  TRIAD_TOOLTIP_LABEL_STYLE,
  TRIAD_TOOLTIP_STYLE,
  TriadSvgDefs,
} from "../utils/chartTheme";

interface PredictionViewProps {
  models: ModelsSummaryResponse | null;
  latestRecord: FinancialRecord | null;
  records?: FinancialRecord[];
  onNavigateToForecasting?: () => void;
}

type AnchorKey = "latest" | "t12_avg" | "peak";

interface ScaleMultipliers {
  unitsPct: number;
  pricePct: number;
  fxPct: number;
  mktPct: number;
  opexPct: number;
}

const DEFAULT_SCALES: ScaleMultipliers = {
  unitsPct: 0,
  pricePct: 0,
  fxPct: 0,
  mktPct: 0,
  opexPct: 0,
};

function buildAnchorInputs(
  anchor: AnchorKey,
  latestRecord: FinancialRecord | null,
  records: FinancialRecord[]
): PredictionInputs | null {
  if (!latestRecord && records.length === 0) return null;

  if (anchor === "t12_avg" && records.length > 0) {
    const slice = records.slice(-12);
    const mean = (fn: (r: FinancialRecord) => number) =>
      slice.reduce((acc, r) => acc + fn(r), 0) / slice.length;
    const latest = slice[slice.length - 1];
    return {
      USD_NGN_Planning_Rate: Number(mean((r) => r.USD_NGN_Planning_Rate).toFixed(2)),
      Inflation_YoY_Pct: Number(mean((r) => r.Inflation_YoY_Pct).toFixed(2)),
      Units_Sold: Math.round(mean((r) => r.Units_Sold)),
      Import_Cost_USD_Per_Unit: Number(mean((r) => r.Import_Cost_USD_Per_Unit).toFixed(2)),
      Avg_Selling_Price_NGN: Math.round(mean((r) => r.Avg_Selling_Price_NGN)),
      Marketing_Spend_NGN: Math.round(mean((r) => r.Marketing_Spend_NGN)),
      Other_Operating_Cost_NGN: Math.round(mean((r) => r.Other_Operating_Cost_NGN)),
      FX_Change_MoM_Pct: Number(mean((r) => r.FX_Change_MoM_Pct ?? 0).toFixed(2)),
      Quarter: latest.Quarter,
      Market_Notes: "Trailing 12-Month Historical Average Anchor",
      horizon_months: 1,
    };
  }

  if (anchor === "peak" && records.length > 0) {
    const peak = records.reduce(
      (best, r) => (r.Operating_Profit_NGN > best.Operating_Profit_NGN ? r : best),
      records[0]
    );
    return {
      USD_NGN_Planning_Rate: peak.USD_NGN_Planning_Rate,
      Inflation_YoY_Pct: Number(peak.Inflation_YoY_Pct.toFixed(2)),
      Units_Sold: peak.Units_Sold,
      Import_Cost_USD_Per_Unit: peak.Import_Cost_USD_Per_Unit,
      Avg_Selling_Price_NGN: peak.Avg_Selling_Price_NGN,
      Marketing_Spend_NGN: peak.Marketing_Spend_NGN,
      Other_Operating_Cost_NGN: peak.Other_Operating_Cost_NGN,
      FX_Change_MoM_Pct:
        peak.FX_Change_MoM_Pct !== null ? Number(peak.FX_Change_MoM_Pct.toFixed(2)) : 0,
      Quarter: peak.Quarter,
      Market_Notes: peak.Market_Notes || `Peak Operating Profit Month (${peak.Month_Label})`,
      horizon_months: 1,
    };
  }

  const ref = latestRecord || records[records.length - 1];
  return {
    USD_NGN_Planning_Rate: ref.USD_NGN_Planning_Rate,
    Inflation_YoY_Pct: Number(ref.Inflation_YoY_Pct.toFixed(2)),
    Units_Sold: ref.Units_Sold,
    Import_Cost_USD_Per_Unit: ref.Import_Cost_USD_Per_Unit,
    Avg_Selling_Price_NGN: ref.Avg_Selling_Price_NGN,
    Marketing_Spend_NGN: ref.Marketing_Spend_NGN,
    Other_Operating_Cost_NGN: ref.Other_Operating_Cost_NGN,
    FX_Change_MoM_Pct:
      ref.FX_Change_MoM_Pct !== null ? Number(ref.FX_Change_MoM_Pct.toFixed(2)) : 0,
    Quarter: ref.Quarter,
    Market_Notes: ref.Market_Notes || "",
    horizon_months: 1,
  };
}

export const PredictionView: React.FC<PredictionViewProps> = ({
  models,
  latestRecord,
  records = [],
  onNavigateToForecasting,
}) => {
  const {
    activePresetLabel,
    simulatedInputs: globalSimInputs,
    setExactSimulatedInputs,
    setDrawerOpen,
  } = useScenario();
  const [suiteTab, setSuiteTab] = useState<"driver" | "ex_ante">("driver");
  const [anchorKey, setAnchorKey] = useState<AnchorKey>("latest");
  const [scales, setScales] = useState<ScaleMultipliers>(DEFAULT_SCALES);
  const [horizonMonths, setHorizonMonths] = useState<number>(1);
  const [baseInputs, setBaseInputs] = useState<PredictionInputs | null>(null);

  const [inputs, setInputs] = useState<PredictionInputs>({
    USD_NGN_Planning_Rate: latestRecord?.USD_NGN_Planning_Rate ?? 1564.13,
    Inflation_YoY_Pct: latestRecord?.Inflation_YoY_Pct ?? 20.67,
    Units_Sold: latestRecord?.Units_Sold ?? 11700,
    Import_Cost_USD_Per_Unit: latestRecord?.Import_Cost_USD_Per_Unit ?? 10.12,
    Avg_Selling_Price_NGN: latestRecord?.Avg_Selling_Price_NGN ?? 25370,
    Marketing_Spend_NGN: latestRecord?.Marketing_Spend_NGN ?? 7689000,
    Other_Operating_Cost_NGN: latestRecord?.Other_Operating_Cost_NGN ?? 10037000,
    FX_Change_MoM_Pct: latestRecord?.FX_Change_MoM_Pct ?? 3.28,
    Quarter: latestRecord?.Quarter ?? "Q4",
    Market_Notes: "",
    horizon_months: 1,
  });

  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [predError, setPredError] = useState<string | null>(null);

  useEffect(() => {
    const built = buildAnchorInputs(anchorKey, latestRecord, records);
    if (built) {
      const withHorizon = { ...built, horizon_months: horizonMonths };
      setBaseInputs(built);
      setScales(DEFAULT_SCALES);
      setInputs(withHorizon);
      void runPrediction(withHorizon);
    }
  }, [latestRecord?.Month, records.length]);

  const applyAnchor = (nextAnchor: AnchorKey) => {
    setAnchorKey(nextAnchor);
    const built = buildAnchorInputs(nextAnchor, latestRecord, records);
    if (built) {
      const payload = { ...built, horizon_months: horizonMonths };
      setBaseInputs(built);
      setScales(DEFAULT_SCALES);
      setInputs(payload);
      void runPrediction(payload);
    }
  };

  const applyScaleChange = (field: keyof ScaleMultipliers, pct: number) => {
    const nextScales = { ...scales, [field]: pct };
    setScales(nextScales);
    const anchor = baseInputs || inputs;
    const nextInputs: PredictionInputs = {
      ...inputs,
      Units_Sold:
        field === "unitsPct" ? Math.max(1, Math.round(anchor.Units_Sold * (1 + pct / 100))) : inputs.Units_Sold,
      Avg_Selling_Price_NGN:
        field === "pricePct"
          ? Math.max(100, Math.round(anchor.Avg_Selling_Price_NGN * (1 + pct / 100)))
          : inputs.Avg_Selling_Price_NGN,
      USD_NGN_Planning_Rate:
        field === "fxPct"
          ? Math.max(10, Number((anchor.USD_NGN_Planning_Rate * (1 + pct / 100)).toFixed(2)))
          : inputs.USD_NGN_Planning_Rate,
      Marketing_Spend_NGN:
        field === "mktPct"
          ? Math.max(0, Math.round(anchor.Marketing_Spend_NGN * (1 + pct / 100)))
          : inputs.Marketing_Spend_NGN,
      Other_Operating_Cost_NGN:
        field === "opexPct"
          ? Math.max(0, Math.round(anchor.Other_Operating_Cost_NGN * (1 + pct / 100)))
          : inputs.Other_Operating_Cost_NGN,
      horizon_months: horizonMonths,
    };
    setInputs(nextInputs);
  };

  const runPrediction = async (payload: PredictionInputs) => {
    setPredicting(true);
    setPredError(null);
    try {
      const res = await api.predictOperatingProfit({
        ...payload,
        horizon_months: payload.horizon_months ?? horizonMonths,
      });
      setPrediction(res);
    } catch (err: unknown) {
      setPredError(err instanceof Error ? err.message : "Prediction failed.");
    } finally {
      setPredicting(false);
    }
  };

  const handleHorizonChange = (months: number) => {
    const clamped = Math.max(1, Math.min(60, Math.round(months || 1)));
    setHorizonMonths(clamped);
    const payload = { ...inputs, horizon_months: clamped };
    setInputs(payload);
    void runPrediction(payload);
  };

  // Auto-calculated dependent features preview
  const liveCalc = useMemo(() => {
    const impNgn = inputs.USD_NGN_Planning_Rate * inputs.Import_Cost_USD_Per_Unit;
    const cogs = inputs.Units_Sold * impNgn;
    const rev = inputs.Units_Sold * inputs.Avg_Selling_Price_NGN;
    const gp = rev - cogs;
    const gm = rev > 0 ? (gp / rev) * 100 : 0;
    const op = gp - inputs.Marketing_Spend_NGN - inputs.Other_Operating_Cost_NGN;
    const om = rev > 0 ? (op / rev) * 100 : 0;
    const unitSpread = inputs.Avg_Selling_Price_NGN - impNgn;
    const breakEvenUnits =
      unitSpread > 1 ? (inputs.Marketing_Spend_NGN + inputs.Other_Operating_Cost_NGN) / unitSpread : null;
    return { impNgn, cogs, rev, gp, gm, op, om, unitSpread, breakEvenUnits };
  }, [inputs]);

  const activeSuite = suiteTab === "driver" ? models?.driver_suite : models?.ex_ante_suite;
  const selectedModel = activeSuite?.selected_model;

  const historyPlusPredChart = useMemo(() => {
    if (!prediction) return [];
    const base = prediction.recent_history.map((h) => ({
      label: h.month_label,
      actual: h.operating_profit_ngn,
      predicted: null as number | null,
    }));
    base.push({
      label: "User Forecast",
      actual: null as unknown as number,
      predicted: prediction.ml_prediction.predicted_operating_profit_ngn,
    });
    return base;
  }, [prediction]);

  const ue = prediction?.measurable_unit_economics;
  const hs = prediction?.horizon_scaling;
  const sensitivities = prediction?.measurable_sensitivities || [];
  const warnings = prediction?.ml_prediction.out_of_range_warnings || [];

  return (
    <div className="space-y-6">
      {onNavigateToForecasting && (
        <div className="rounded-xl border border-slate-200 bg-white px-5 py-3.5 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div className="text-xs text-slate-700">
            <strong className="text-[#0A1E3D]">Multi-Year 2030 Projections &amp; Statistical Testing:</strong>{" "}
            Explore monthly forecasts through December 2030, rolling-origin backtesting, and formal hypothesis tests
            in the dedicated Forecasting Centre.
          </div>
          <button
            type="button"
            onClick={onNavigateToForecasting}
            className="triad-btn-primary inline-flex items-center gap-1.5 rounded-lg px-3.5 py-2 text-xs font-semibold text-white whitespace-nowrap shrink-0"
          >
            <TrendingUp className="h-3.5 w-3.5" />
            <span>Open 2030 Forecasting Centre</span>
          </button>
        </div>
      )}

      {/* Top Interactive Scalable & Measurable Prediction Studio */}
      <div className="grid grid-cols-1 gap-5 xl:grid-cols-12">
        {/* Left: Scalable Anchor Presets, % Multipliers, Horizon & Numeric Inputs */}
        <div className="rounded-xl border border-slate-200 bg-white p-5 xl:col-span-6 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 pb-3">
            <div className="flex items-center gap-2">
              <Cpu className="h-4 w-4 text-[#0A2E65]" />
              <h2 className="font-display text-sm font-bold text-[#0A1E3D]">
                Scalable Operating Profit Prediction Studio
              </h2>
            </div>
            <div className="flex flex-wrap items-center gap-1.5">
              <button
                type="button"
                onClick={() => {
                  const synced = { ...globalSimInputs, horizon_months: horizonMonths };
                  setInputs(synced);
                  void runPrediction(synced);
                }}
                className="rounded-md border border-teal-700/30 bg-teal-50 px-2.5 py-1 text-[11px] font-semibold text-teal-900 hover:bg-teal-100 whitespace-nowrap"
              >
                Load {activePresetLabel}
              </button>
              <button
                type="button"
                onClick={() => {
                  setExactSimulatedInputs(inputs);
                  setDrawerOpen(true);
                }}
                className="rounded-md border border-slate-300 bg-white px-2.5 py-1 text-[11px] font-semibold text-[#0A2E65] hover:bg-slate-50 whitespace-nowrap"
              >
                Push to Global Sim
              </button>
            </div>
          </div>

          {/* Baseline Anchor & Horizon Scaling Controls */}
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 rounded-lg border border-slate-200 bg-slate-50/70 p-3">
            <div>
              <div className="text-[11px] font-semibold text-slate-700 mb-1.5">
                Baseline Anchor Preset
              </div>
              <div className="flex flex-wrap gap-1">
                {[
                  { id: "latest" as const, label: "Latest Month" },
                  { id: "t12_avg" as const, label: "Trailing 12M Avg" },
                  { id: "peak" as const, label: "Peak Month" },
                ].map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => applyAnchor(item.id)}
                    className={`rounded-md px-2.5 py-1 text-[11px] font-semibold transition-all whitespace-nowrap ${
                      anchorKey === item.id
                        ? "bg-gradient-to-r from-[#0A2E65] to-[#1D63B8] text-white shadow-sm"
                        : "bg-white border border-slate-200 text-slate-600 hover:text-slate-900"
                    }`}
                  >
                    {item.label}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <div className="text-[11px] font-semibold text-slate-700 mb-1.5">
                Scalable Planning Horizon (Months)
              </div>
              <div className="flex flex-wrap items-center gap-1">
                {[1, 3, 6, 12].map((m) => (
                  <button
                    key={m}
                    type="button"
                    onClick={() => handleHorizonChange(m)}
                    className={`rounded-md px-2.5 py-1 text-[11px] font-semibold tabular-nums transition-all whitespace-nowrap ${
                      horizonMonths === m
                        ? "bg-gradient-to-r from-[#1D63B8] to-[#0D9488] text-white shadow-sm"
                        : "bg-white border border-slate-200 text-slate-600 hover:text-slate-900"
                    }`}
                  >
                    {m}M
                  </button>
                ))}
                <input
                  type="number"
                  min={1}
                  max={60}
                  value={horizonMonths}
                  onChange={(e) => handleHorizonChange(Number(e.target.value))}
                  aria-label="Custom Horizon Months"
                  className="w-14 rounded-md border border-slate-200 bg-white px-2 py-1 font-mono text-[11px] text-slate-900 tabular-nums focus:border-[#0A2E65] focus:outline-none"
                />
              </div>
            </div>
          </div>

          {/* Scalable % Multiplier Sliders */}
          <div className="rounded-lg border border-slate-200 bg-white p-3 space-y-2.5">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5 text-xs font-bold text-[#0A1E3D]">
                <Sliders className="h-3.5 w-3.5 text-[#0A2E65]" />
                <span>Scale Drivers Relative to Anchor (−50% to +200%)</span>
              </div>
              <button
                type="button"
                onClick={() => applyAnchor(anchorKey)}
                className="text-[11px] font-medium text-[#0A2E65] hover:underline whitespace-nowrap"
              >
                Reset Sliders
              </button>
            </div>
            <div className="grid grid-cols-1 gap-x-4 gap-y-2 sm:grid-cols-2">
              {[
                { key: "unitsPct" as const, label: "Units Sold Scale" },
                { key: "pricePct" as const, label: "Selling Price Scale" },
                { key: "fxPct" as const, label: "USD/NGN FX Scale" },
                { key: "mktPct" as const, label: "Marketing Spend Scale" },
                { key: "opexPct" as const, label: "Other OpEx Scale" },
              ].map((s) => (
                <div key={s.key} className="space-y-0.5">
                  <div className="flex items-center justify-between text-[11px]">
                    <span className="text-slate-600 font-medium">{s.label}</span>
                    <span className="font-mono font-bold text-[#0A1E3D] tabular-nums">
                      {scales[s.key] >= 0 ? `+${scales[s.key]}%` : `${scales[s.key]}%`}
                    </span>
                  </div>
                  <input
                    type="range"
                    min={-50}
                    max={200}
                    step={5}
                    value={scales[s.key]}
                    onChange={(e) => applyScaleChange(s.key, Number(e.target.value))}
                    style={getSliderFillStyle(scales[s.key], -50, 200)}
                    className="triad-slider w-full"
                  />
                </div>
              ))}
            </div>
          </div>

          <form
            onSubmit={(e) => {
              e.preventDefault();
              void runPrediction({ ...inputs, horizon_months: horizonMonths });
            }}
            className="space-y-4"
          >
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              <div>
                <label className="block text-xs font-medium text-slate-700">
                  USD/NGN Planning Rate (₦/$)
                </label>
                <input
                  type="number"
                  step="0.01"
                  required
                  value={inputs.USD_NGN_Planning_Rate}
                  onChange={(e) =>
                    setInputs({ ...inputs, USD_NGN_Planning_Rate: Number(e.target.value) })
                  }
                  className="mt-1 w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5 font-mono text-xs tabular-nums focus:border-[#0A2E65] focus:bg-white focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700">
                  Inflation YoY (%)
                </label>
                <input
                  type="number"
                  step="0.01"
                  required
                  value={inputs.Inflation_YoY_Pct}
                  onChange={(e) =>
                    setInputs({ ...inputs, Inflation_YoY_Pct: Number(e.target.value) })
                  }
                  className="mt-1 w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5 font-mono text-xs tabular-nums focus:border-[#0A2E65] focus:bg-white focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700">Units Sold</label>
                <input
                  type="number"
                  step="1"
                  required
                  value={inputs.Units_Sold}
                  onChange={(e) => setInputs({ ...inputs, Units_Sold: Number(e.target.value) })}
                  className="mt-1 w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5 font-mono text-xs tabular-nums focus:border-[#0A2E65] focus:bg-white focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700">
                  Import Cost (USD / Unit)
                </label>
                <input
                  type="number"
                  step="0.01"
                  required
                  value={inputs.Import_Cost_USD_Per_Unit}
                  onChange={(e) =>
                    setInputs({ ...inputs, Import_Cost_USD_Per_Unit: Number(e.target.value) })
                  }
                  className="mt-1 w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5 font-mono text-xs tabular-nums focus:border-[#0A2E65] focus:bg-white focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700">
                  Avg Selling Price (NGN / Unit)
                </label>
                <input
                  type="number"
                  step="10"
                  required
                  value={inputs.Avg_Selling_Price_NGN}
                  onChange={(e) =>
                    setInputs({ ...inputs, Avg_Selling_Price_NGN: Number(e.target.value) })
                  }
                  className="mt-1 w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5 font-mono text-xs tabular-nums focus:border-[#0A2E65] focus:bg-white focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700">
                  Marketing Spend (NGN)
                </label>
                <input
                  type="number"
                  step="10000"
                  required
                  value={inputs.Marketing_Spend_NGN}
                  onChange={(e) =>
                    setInputs({ ...inputs, Marketing_Spend_NGN: Number(e.target.value) })
                  }
                  className="mt-1 w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5 font-mono text-xs tabular-nums focus:border-[#0A2E65] focus:bg-white focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700">
                  Other Operating Cost (NGN)
                </label>
                <input
                  type="number"
                  step="10000"
                  required
                  value={inputs.Other_Operating_Cost_NGN}
                  onChange={(e) =>
                    setInputs({ ...inputs, Other_Operating_Cost_NGN: Number(e.target.value) })
                  }
                  className="mt-1 w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5 font-mono text-xs tabular-nums focus:border-[#0A2E65] focus:bg-white focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700">
                  FX Change MoM (%)
                </label>
                <input
                  type="number"
                  step="0.01"
                  value={inputs.FX_Change_MoM_Pct ?? 0}
                  onChange={(e) =>
                    setInputs({ ...inputs, FX_Change_MoM_Pct: Number(e.target.value) })
                  }
                  className="mt-1 w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5 font-mono text-xs tabular-nums focus:border-[#0A2E65] focus:bg-white focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700">Quarter</label>
                <select
                  value={inputs.Quarter}
                  onChange={(e) => setInputs({ ...inputs, Quarter: e.target.value })}
                  className="mt-1 w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs font-medium focus:border-[#0A2E65] focus:bg-white focus:outline-none"
                >
                  {["Q1", "Q2", "Q3", "Q4"].map((q) => (
                    <option key={q} value={q}>
                      {q}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700">
                  Market Notes (Optional)
                </label>
                <input
                  type="text"
                  value={inputs.Market_Notes}
                  onChange={(e) => setInputs({ ...inputs, Market_Notes: e.target.value })}
                  placeholder="e.g., Planned price adjustment"
                  className="mt-1 w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs focus:border-[#0A2E65] focus:bg-white focus:outline-none"
                />
              </div>
            </div>

            {/* Auto-Engineered Intermediate Preview */}
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
              <div className="text-[11px] font-semibold text-slate-600">
                Auto-Calculated Dependent Features (Pipeline Preview)
              </div>
              <div className="mt-2 grid grid-cols-2 gap-2 text-xs sm:grid-cols-3">
                <div>
                  <span className="text-slate-500">Import Cost/Unit: </span>
                  <span className="font-mono font-semibold text-slate-900 tabular-nums">
                    {formatNGN(liveCalc.impNgn)}
                  </span>
                </div>
                <div>
                  <span className="text-slate-500">COGS: </span>
                  <span className="font-mono font-semibold text-slate-900 tabular-nums">
                    {formatNGN(liveCalc.cogs, true)}
                  </span>
                </div>
                <div>
                  <span className="text-slate-500">Sales Revenue: </span>
                  <span className="font-mono font-semibold text-slate-900 tabular-nums">
                    {formatNGN(liveCalc.rev, true)}
                  </span>
                </div>
                <div>
                  <span className="text-slate-500">Gross Profit: </span>
                  <span className="font-mono font-semibold text-slate-900 tabular-nums">
                    {formatNGN(liveCalc.gp, true)}
                  </span>
                </div>
                <div>
                  <span className="text-slate-500">Gross Margin: </span>
                  <span className="font-mono font-semibold text-slate-900 tabular-nums">
                    {formatPct(liveCalc.gm)}
                  </span>
                </div>
                <div>
                  <span className="text-slate-500">Break-Even: </span>
                  <span className="font-mono font-semibold text-slate-900 tabular-nums">
                    {liveCalc.breakEvenUnits ? `${Math.round(liveCalc.breakEvenUnits).toLocaleString()} u` : "N/A"}
                  </span>
                </div>
              </div>
            </div>

            <button
              type="submit"
              disabled={predicting}
              className="triad-btn-primary w-full rounded-lg py-2.5 text-xs font-semibold text-white disabled:opacity-60"
            >
              {predicting ? "Running ML Prediction & SHAP..." : "Run Scalable ML Prediction"}
            </button>
          </form>

          {predError && (
            <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-xs text-red-700">
              {predError}
            </div>
          )}
        </div>

        {/* Right: Measurable ML Prediction, Prediction Intervals, Horizon Scaling, Unit Economics & SHAP */}
        <div className="rounded-xl border border-slate-200 bg-white p-5 xl:col-span-6 space-y-4">
          {prediction ? (
            <>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                <div className="rounded-xl bg-gradient-to-br from-[#072149] via-[#124A93] to-[#0D9488] p-4 text-white shadow-sm">
                  <div className="text-xs font-medium text-blue-100">
                    ML Predicted Operating Profit (1M Run-Rate)
                  </div>
                  <div className="mt-1.5 font-display text-2xl font-bold tabular-nums">
                    {formatNGN(prediction.ml_prediction.predicted_operating_profit_ngn, true)}
                  </div>
                  <div className="mt-1 text-[11px] text-blue-200 tabular-nums">
                    Exact: {formatNGN(prediction.ml_prediction.predicted_operating_profit_ngn)} ·{" "}
                    {formatPct(prediction.ml_prediction.predicted_operating_margin_pct)} margin
                  </div>
                  {prediction.ml_prediction.prediction_interval_95 && (
                    <div className="mt-1 text-[11px] text-blue-100/90 tabular-nums">
                      95% PI: {formatNGN(prediction.ml_prediction.prediction_interval_95[0], true)} to{" "}
                      {formatNGN(prediction.ml_prediction.prediction_interval_95[1], true)}
                    </div>
                  )}
                  <div className="mt-1.5 text-[11px] text-teal-300 tabular-nums">
                    Model: {prediction.ml_prediction.model_name}
                    {prediction.ml_prediction.ridge_extrapolation_ngn !== undefined &&
                      ` · Ridge Benchmark: ${formatNGN(prediction.ml_prediction.ridge_extrapolation_ngn, true)}`}
                  </div>
                </div>

                <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                  <div className="text-xs font-medium text-slate-600">
                    Accounting Formula Reconciliation
                  </div>
                  <div
                    className={`mt-1.5 font-display text-2xl font-bold tabular-nums ${
                      prediction.accounting_reconciliation.Accounting_Operating_Profit_NGN >= 0
                        ? "text-teal-800"
                        : "text-red-600"
                    }`}
                  >
                    {formatNGN(
                      prediction.accounting_reconciliation.Accounting_Operating_Profit_NGN,
                      true
                    )}
                  </div>
                  <div className="mt-1 text-[11px] text-slate-600 tabular-nums">
                    Gross Profit − Marketing − Other OpEx ·{" "}
                    {formatPct(
                      prediction.accounting_reconciliation.Accounting_Operating_Margin_Pct
                    )}{" "}
                    margin
                  </div>
                  <div className="mt-2 text-[11px] text-slate-500 tabular-nums">
                    ML vs. Accounting delta:{" "}
                    {formatNGN(prediction.accounting_reconciliation.ml_vs_accounting_gap_ngn, true)}
                  </div>
                </div>
              </div>

              {/* Out-of-Range Tree Extrapolation Warning */}
              {warnings.length > 0 && (
                <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs text-amber-900">
                  <div className="flex items-center gap-1.5 font-bold">
                    <AlertTriangle className="h-3.5 w-3.5 text-amber-700 shrink-0" />
                    <span>
                      Out-of-Range Scaled Inputs Detected ({warnings.length}) — Tree Clamping Notice
                    </span>
                  </div>
                  <p className="mt-1 text-[11px] leading-snug">
                    {prediction.ml_prediction.is_tree_model
                      ? "The selected tree ensemble cannot extrapolate beyond training min/max leaves. Compare against the Continuous Ridge Benchmark and exact Accounting Formula above."
                      : "Inputs extend outside historical min/max bounds; review uncertainty bands."}
                  </p>
                  <div className="mt-1.5 flex flex-wrap gap-x-4 gap-y-1 text-[11px] font-mono tabular-nums">
                    {warnings.map((w) => (
                      <span key={w.feature}>
                        {w.label}: {w.input_value.toLocaleString()} (hist {w.hist_min.toLocaleString()}–
                        {w.hist_max.toLocaleString()})
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Scalable Horizon Totals */}
              {hs && (
                <div className="rounded-lg border border-slate-200 bg-slate-50/80 p-3">
                  <div className="flex items-center justify-between text-xs font-bold text-[#0A1E3D]">
                    <span className="inline-flex items-center gap-1.5">
                      <Layers className="h-3.5 w-3.5 text-teal-700" />
                      Scaled {hs.horizon_months}-Month Planning Projection
                    </span>
                    <span className="font-mono text-[11px] text-slate-500 tabular-nums">
                      {hs.scaled_units_sold.toLocaleString()} units · {formatNGN(hs.scaled_revenue_ngn, true)} rev
                    </span>
                  </div>
                  <div className="mt-2 grid grid-cols-2 gap-2.5 sm:grid-cols-3 text-xs">
                    <div>
                      <div className="text-[11px] text-slate-500">ML {hs.horizon_months}M Profit</div>
                      <div className="font-mono font-bold text-[#0A1E3D] tabular-nums">
                        {formatNGN(hs.ml_scaled_profit_ngn, true)}
                      </div>
                    </div>
                    <div>
                      <div className="text-[11px] text-slate-500">95% Scaled Band</div>
                      <div className="font-mono font-semibold text-slate-700 tabular-nums">
                        {formatNGN(hs.ml_scaled_ci95_low_ngn, true)} –{" "}
                        {formatNGN(hs.ml_scaled_ci95_high_ngn, true)}
                      </div>
                    </div>
                    <div>
                      <div className="text-[11px] text-slate-500">Accounting {hs.horizon_months}M</div>
                      <div className="font-mono font-bold text-teal-800 tabular-nums">
                        {formatNGN(hs.accounting_scaled_profit_ngn, true)}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Measurable Unit Economics & Break-Even Strip */}
              {ue && (
                <div className="rounded-lg border border-slate-200 bg-white p-3">
                  <div className="flex items-center gap-1.5 text-xs font-bold text-[#0A1E3D] mb-2">
                    <Gauge className="h-3.5 w-3.5 text-[#0A2E65]" />
                    <span>Measurable Unit Economics &amp; Operating Leverage</span>
                  </div>
                  <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-3 text-xs">
                    <div className="rounded-md bg-slate-50 p-2">
                      <div className="text-[10px] text-slate-500">Unit Gross Spread</div>
                      <div className="font-mono font-bold text-slate-900 tabular-nums">
                        {formatNGN(ue.unit_spread_ngn)}/unit
                      </div>
                    </div>
                    <div className="rounded-md bg-slate-50 p-2">
                      <div className="text-[10px] text-slate-500">Price / Import Ratio</div>
                      <div className="font-mono font-bold text-slate-900 tabular-nums">
                        {ue.price_to_import_ratio.toFixed(2)}x
                      </div>
                    </div>
                    <div className="rounded-md bg-slate-50 p-2">
                      <div className="text-[10px] text-slate-500">Break-Even Volume</div>
                      <div className="font-mono font-bold text-slate-900 tabular-nums">
                        {ue.break_even_units !== null
                          ? `${ue.break_even_units.toLocaleString()} u`
                          : "Unprofitable Spread"}
                      </div>
                    </div>
                    <div className="rounded-md bg-slate-50 p-2">
                      <div className="text-[10px] text-slate-500">Margin of Safety</div>
                      <div
                        className={`font-mono font-bold tabular-nums ${
                          (ue.margin_of_safety_pct ?? 0) >= 0 ? "text-teal-700" : "text-red-600"
                        }`}
                      >
                        {ue.margin_of_safety_pct !== null ? `${ue.margin_of_safety_pct.toFixed(1)}%` : "—"}
                      </div>
                    </div>
                    <div className="rounded-md bg-slate-50 p-2">
                      <div className="text-[10px] text-slate-500">Marketing ROAS</div>
                      <div className="font-mono font-bold text-slate-900 tabular-nums">
                        {ue.marketing_roas !== null ? `${ue.marketing_roas.toFixed(2)}x` : "—"}
                      </div>
                    </div>
                    <div className="rounded-md bg-slate-50 p-2">
                      <div className="text-[10px] text-slate-500">OpEx % of Revenue</div>
                      <div className="font-mono font-bold text-slate-900 tabular-nums">
                        {ue.opex_to_revenue_pct.toFixed(1)}%
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Local SHAP Drivers */}
              <div>
                <div className="flex items-center justify-between">
                  <h3 className="text-xs font-bold text-[#0A1E3D]">
                    Individual Prediction SHAP Driver Contributions
                  </h3>
                  <span className="font-mono text-[11px] text-slate-500 tabular-nums">
                    Baseline: {formatNGN(prediction.local_explanation.baseline_prediction_ngn, true)}
                  </span>
                </div>
                <div className="mt-2 h-[180px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      layout="vertical"
                      data={prediction.local_explanation.contributions.slice(0, 7)}
                      margin={{ top: 4, right: 20, left: 10, bottom: 4 }}
                    >
                      <TriadSvgDefs idPrefix="pred-shap" />
                      <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                      <XAxis
                        type="number"
                        tick={{ fontSize: 10, fill: "#64748B" }}
                        tickFormatter={(v) => `₦${(v / 1e6).toFixed(0)}M`}
                      />
                      <YAxis
                        type="category"
                        dataKey="feature"
                        tick={{ fontSize: 10, fill: "#334155" }}
                        width={145}
                      />
                      <Tooltip
                        contentStyle={TRIAD_TOOLTIP_STYLE}
                        labelStyle={TRIAD_TOOLTIP_LABEL_STYLE}
                        itemStyle={TRIAD_TOOLTIP_ITEM_STYLE}
                        formatter={(val: any) => [formatNGN(Number(val)), "SHAP Impact"]}
                      />
                      <ReferenceLine x={0} stroke="#64748B" />
                      <Bar dataKey="shap_contribution_ngn" radius={[0, 4, 4, 0]}>
                        {prediction.local_explanation.contributions.slice(0, 7).map((entry) => (
                          <Cell
                            key={entry.feature}
                            fill={
                              entry.shap_contribution_ngn >= 0
                                ? "url(#pred-shap-bar-cobalt-teal-h)"
                                : "url(#pred-shap-bar-negative)"
                            }
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Trailing 12M History vs Forecast */}
              <div>
                <div className="mb-1.5 text-xs font-bold text-[#0A1E3D]">
                  Predicted Operating Profit vs. Trailing 12-Month Actuals
                </div>
                <div className="h-[155px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={historyPlusPredChart}>
                      <TriadSvgDefs idPrefix="pred-hist" />
                      <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                      <XAxis dataKey="label" tick={{ fontSize: 9, fill: "#64748B" }} />
                      <YAxis
                        tick={{ fontSize: 10, fill: "#64748B" }}
                        tickFormatter={(v) => `₦${(v / 1e6).toFixed(0)}M`}
                        width={52}
                      />
                      <Tooltip
                        contentStyle={TRIAD_TOOLTIP_STYLE}
                        labelStyle={TRIAD_TOOLTIP_LABEL_STYLE}
                        itemStyle={TRIAD_TOOLTIP_ITEM_STYLE}
                        formatter={(val: any) => formatNGN(Number(val))}
                      />
                      <Line
                        type="monotone"
                        dataKey="actual"
                        name="Historical Actual"
                        stroke="url(#pred-hist-stroke-navy-cobalt)"
                        strokeWidth={2.4}
                        dot={{ r: 3, fill: "#1D63B8" }}
                      />
                      <Line
                        type="monotone"
                        dataKey="predicted"
                        name="ML Prediction"
                        stroke="#0D9488"
                        strokeWidth={0}
                        dot={{ r: 6, fill: "#0D9488", stroke: "#FFFFFF", strokeWidth: 2 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </>
          ) : (
            <div className="py-12 text-center text-xs text-slate-500">
              Submit parameters to compute ML operating profit prediction and SHAP breakdown.
            </div>
          )}
        </div>
      </div>

      {/* Measurable +10% Local Driver Sensitivity Matrix */}
      {sensitivities.length > 0 && (
        <div className="rounded-xl border border-slate-200 bg-white p-5 space-y-4">
          <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h2 className="font-display text-sm font-bold text-[#0A1E3D]">
                Measurable +10% Driver Sensitivity Analysis
              </h2>
              <p className="text-xs text-slate-500">
                Exact Operating Profit delta (NGN &amp; %) from a ceteris-paribus +10% increase in each input lever
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-5 lg:grid-cols-12">
            <div className="h-[220px] lg:col-span-6">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={sensitivities} margin={{ top: 8, right: 16, left: 6, bottom: 8 }}>
                  <TriadSvgDefs idPrefix="pred-sens" />
                  <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                  <XAxis dataKey="label" tick={{ fontSize: 10, fill: "#475569" }} interval={0} angle={-12} textAnchor="end" height={46} />
                  <YAxis
                    tick={{ fontSize: 10, fill: "#64748B" }}
                    tickFormatter={(v) => `₦${(v / 1e6).toFixed(0)}M`}
                    width={54}
                  />
                  <Tooltip
                    contentStyle={TRIAD_TOOLTIP_STYLE}
                    labelStyle={TRIAD_TOOLTIP_LABEL_STYLE}
                    itemStyle={TRIAD_TOOLTIP_ITEM_STYLE}
                    formatter={(val: any) => formatNGN(Number(val))}
                  />
                  <Legend wrapperStyle={{ fontSize: "11px" }} />
                  <ReferenceLine y={0} stroke="#64748B" />
                  <Bar dataKey="accounting_delta_ngn" name="Accounting Delta (₦)" fill="url(#pred-sens-bar-navy-cobalt)" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="ml_delta_ngn" name="ML Model Delta (₦)" fill="url(#pred-sens-bar-cobalt-teal)" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="overflow-x-auto lg:col-span-6">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-200 text-slate-500">
                    <th className="py-2 pr-3 font-semibold">Input Lever (+10%)</th>
                    <th className="py-2 pr-3 text-right font-semibold">Accounting Δ (₦)</th>
                    <th className="py-2 pr-3 text-right font-semibold">Accounting Δ (%)</th>
                    <th className="py-2 pr-3 text-right font-semibold">ML Model Δ (₦)</th>
                    <th className="py-2 text-right font-semibold">ML Model Δ (%)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {sensitivities.map((s) => (
                    <tr key={s.lever}>
                      <td className="py-2 pr-3 font-semibold text-[#0A1E3D]">{s.label}</td>
                      <td
                        className={`py-2 pr-3 text-right font-mono tabular-nums ${
                          s.accounting_delta_ngn >= 0 ? "text-teal-700" : "text-red-600"
                        }`}
                      >
                        {s.accounting_delta_ngn >= 0 ? "+" : ""}
                        {formatNGN(s.accounting_delta_ngn, true)}
                      </td>
                      <td className="py-2 pr-3 text-right font-mono tabular-nums text-slate-600">
                        {s.accounting_delta_pct >= 0 ? "+" : ""}
                        {s.accounting_delta_pct.toFixed(1)}%
                      </td>
                      <td
                        className={`py-2 pr-3 text-right font-mono tabular-nums ${
                          s.ml_delta_ngn >= 0 ? "text-teal-700" : "text-red-600"
                        }`}
                      >
                        {s.ml_delta_ngn >= 0 ? "+" : ""}
                        {formatNGN(s.ml_delta_ngn, true)}
                      </td>
                      <td className="py-2 text-right font-mono tabular-nums text-slate-600">
                        {s.ml_delta_pct >= 0 ? "+" : ""}
                        {s.ml_delta_pct.toFixed(1)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Model Comparison & Chronological Validation */}
      <div className="rounded-xl border border-slate-200 bg-white p-5 space-y-4">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="font-display text-sm font-bold text-[#0A1E3D]">
              Chronological Model Evaluation &amp; Selection
            </h2>
            <p className="text-xs text-slate-500">
              {activeSuite
                ? `Train: ${activeSuite.train_period} (${activeSuite.train_size} mos) · Out-of-Time Test: ${activeSuite.test_period} (${activeSuite.test_size} mos)`
                : "Evaluating regression models chronologically"}
            </p>
          </div>
          <div className="flex rounded-lg bg-slate-100 p-1 text-xs font-medium">
            <button
              type="button"
              onClick={() => setSuiteTab("driver")}
              className={`rounded-md px-3 py-1 transition-colors ${
                suiteTab === "driver" ? "bg-white text-[#0A1E3D] shadow-sm" : "text-slate-600"
              }`}
            >
              Driver-Based Suite
            </button>
            <button
              type="button"
              onClick={() => setSuiteTab("ex_ante")}
              className={`rounded-md px-3 py-1 transition-colors ${
                suiteTab === "ex_ante" ? "bg-white text-[#0A1E3D] shadow-sm" : "text-slate-600"
              }`}
            >
              Ex-Ante Lagged Forecast
            </button>
          </div>
        </div>

        {activeSuite?.limitation_notice && (
          <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs text-amber-900">
            <strong>Model Evaluation Note:</strong> {activeSuite.limitation_notice}
          </div>
        )}

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-200 text-slate-500">
                <th className="py-2 pr-4 font-semibold">Algorithm</th>
                <th className="py-2 pr-4 text-right font-semibold">Test MAE</th>
                <th className="py-2 pr-4 text-right font-semibold">Test RMSE</th>
                <th className="py-2 pr-4 text-right font-semibold">Test R²</th>
                <th className="py-2 pr-4 text-right font-semibold">Test MAPE</th>
                <th className="py-2 pr-4 text-right font-semibold">Train CV RMSE</th>
                <th className="py-2 text-right font-semibold">Selection</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {activeSuite?.models.map((m) => {
                const isWinner = selectedModel?.key === m.key;
                return (
                  <tr key={m.key} className={isWinner ? "bg-teal-50/50" : ""}>
                    <td className="py-2.5 pr-4 font-semibold text-[#0A1E3D]">{m.name}</td>
                    <td className="py-2.5 pr-4 text-right font-mono tabular-nums">
                      {formatNGN(m.test_metrics.mae, true)}
                    </td>
                    <td className="py-2.5 pr-4 text-right font-mono font-semibold tabular-nums">
                      {formatNGN(m.test_metrics.rmse, true)}
                    </td>
                    <td className="py-2.5 pr-4 text-right font-mono tabular-nums">
                      {m.test_metrics.r2 ?? "—"}
                    </td>
                    <td className="py-2.5 pr-4 text-right font-mono tabular-nums">
                      {m.test_metrics.mape_pct !== null ? `${m.test_metrics.mape_pct}%` : "—"}
                    </td>
                    <td className="py-2.5 pr-4 text-right font-mono tabular-nums">
                      {m.cv_rmse ? formatNGN(m.cv_rmse, true) : "—"}
                    </td>
                    <td className="py-2.5 text-right font-semibold">
                      {isWinner ? (
                        <span className="text-teal-700">Selected (Best Out-of-Time)</span>
                      ) : (
                        <span className="text-slate-400">Evaluated</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Actual vs Predicted Out-of-Time + Residuals */}
        {selectedModel && (
          <div className="grid grid-cols-1 gap-5 pt-2 lg:grid-cols-2">
            <div>
              <div className="mb-1.5 text-xs font-bold text-[#0A1E3D]">
                Out-of-Time Actual vs. Predicted ({selectedModel.name})
              </div>
              <div className="h-[220px]">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={selectedModel.test_predictions}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                    <XAxis dataKey="month_label" tick={{ fontSize: 10, fill: "#64748B" }} />
                    <YAxis
                      tick={{ fontSize: 10, fill: "#64748B" }}
                      tickFormatter={(v) => `₦${(v / 1e6).toFixed(0)}M`}
                      width={52}
                    />
                    <Tooltip formatter={(val: any) => formatNGN(Number(val))} />
                    <Legend wrapperStyle={{ fontSize: "11px" }} />
                    <Line
                      type="monotone"
                      dataKey="actual"
                      name="Actual Operating Profit"
                      stroke="#0A2E65"
                      strokeWidth={2}
                    />
                    <Line
                      type="monotone"
                      dataKey="predicted"
                      name="Out-of-Time Prediction"
                      stroke="#0D9488"
                      strokeWidth={2}
                      strokeDasharray="4 4"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div>
              <div className="mb-1.5 text-xs font-bold text-[#0A1E3D]">
                Out-of-Time Prediction Residuals (Actual − Predicted)
              </div>
              <div className="h-[220px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={selectedModel.test_predictions}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                    <XAxis dataKey="month_label" tick={{ fontSize: 10, fill: "#64748B" }} />
                    <YAxis
                      tick={{ fontSize: 10, fill: "#64748B" }}
                      tickFormatter={(v) => `₦${(v / 1e6).toFixed(0)}M`}
                      width={52}
                    />
                    <Tooltip formatter={(val: any) => formatNGN(Number(val))} />
                    <ReferenceLine y={0} stroke="#64748B" />
                    <Bar dataKey="residual" name="Residual (NGN)">
                      {selectedModel.test_predictions.map((p) => (
                        <Cell key={p.month} fill={p.residual >= 0 ? "#0D9488" : "#D97706"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Critical Leakage Prevention Audit */}
      <div className="rounded-xl border border-slate-200 bg-white p-5">
        <div className="flex items-center gap-2">
          <ShieldCheck className="h-4 w-4 text-teal-700" />
          <h2 className="font-display text-sm font-bold text-[#0A1E3D]">
            Critical Leakage Prevention &amp; Ex-Ante Governance
          </h2>
        </div>
        <p className="mt-1 text-xs text-slate-600">
          The following period-end accounting variables are strictly excluded from all ML model predictors so the
          regression model learns underlying macro and commercial drivers rather than memorizing accounting identities:
        </p>
        <div className="mt-3 grid grid-cols-1 gap-2.5 sm:grid-cols-2 lg:grid-cols-3">
          {activeSuite?.leakage_audit?.map((item) => (
            <div key={item.feature} className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-xs">
              <div className="font-mono font-bold text-[#0A1E3D]">{item.feature}</div>
              <div className="mt-1 text-slate-600">{item.reason}</div>
              <div className="mt-1 text-[11px] font-medium text-amber-800">{item.availability}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
