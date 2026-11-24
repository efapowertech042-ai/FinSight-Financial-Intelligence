import React, { useEffect, useMemo, useState } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Download, Info, Moon, RotateCcw, ShieldCheck, SlidersHorizontal, Sun } from "lucide-react";
import { api } from "../services/api";
import { useScenario } from "../context/ScenarioContext";
import { Accounting5YrParams, Accounting5YrResponse } from "../types/financial";
import { formatNGN } from "../utils/format";
import { TriadSvgDefs, getSliderFillStyle } from "../utils/chartTheme";

interface Accounting5YrSimulatorProps {
  datasetName: string;
  onNavigateToEngineering?: () => void;
}

type PresetKey = "base" | "opt" | "stress" | "custom";

interface SliderConfig {
  key: keyof Accounting5YrParams;
  code: string;
  label: string;
  info: string;
  min: number;
  max: number;
  step: number;
  formatValue: (v: number) => string;
  minLabel: string;
  maxLabel: string;
}

const SLIDER_CONFIGS: SliderConfig[] = [
  {
    code: "A.",
    key: "annual_inflation_pct",
    label: "Annual Inflation",
    info: "Headline annual YoY inflation rate driving domestic OpEx escalation, pricing adjustments (C), and PPP exchange-rate pressure (G).",
    min: 0,
    max: 50,
    step: 0.1,
    formatValue: (v) => `${v.toFixed(2)}%`,
    minLabel: "0%",
    maxLabel: "50%",
  },
  {
    code: "B.",
    key: "unit_sales_growth_pct",
    label: "Unit Sales Growth",
    info: "Annualized compound growth rate in physical unit volume sold (seasonal monthly patterns derived from dataset history).",
    min: -20,
    max: 40,
    step: 0.5,
    formatValue: (v) => `${v > 0 ? "+" : ""}${v.toFixed(1)}%`,
    minLabel: "-20%",
    maxLabel: "+40%",
  },
  {
    code: "C.",
    key: "price_inflation_pass_through_pct",
    label: "Price Inflation Pass-Through",
    info: "Share of Annual Inflation (A) passed through directly into Avg Selling Price (NGN).",
    min: 0,
    max: 100,
    step: 1,
    formatValue: (v) => `${Math.round(v)}%`,
    minLabel: "0%",
    maxLabel: "100%",
  },
  {
    code: "D.",
    key: "fx_cost_pass_through_pct",
    label: "FX Cost Pass-Through",
    info: "Share of projected Naira FX depreciation passed into Avg Selling Price to defend unit gross spread.",
    min: 0,
    max: 100,
    step: 1,
    formatValue: (v) => `${Math.round(v)}%`,
    minLabel: "0%",
    maxLabel: "100%",
  },
  {
    code: "E.",
    key: "usd_import_cost_growth_pct",
    label: "USD Import Cost Growth",
    info: "Annualized change in foreign-currency supplier cost per unit (USD) before Naira FX conversion.",
    min: -10,
    max: 25,
    step: 0.5,
    formatValue: (v) => `${v > 0 ? "+" : ""}${v.toFixed(1)}%`,
    minLabel: "-10%",
    maxLabel: "+25%",
  },
  {
    code: "F.",
    key: "opex_inflation_sensitivity_pct",
    label: "OpEx Inflation Sensitivity",
    info: "Elasticity multiplier applying Annual Inflation (A) to Marketing Spend and Other Operating Overhead.",
    min: 0,
    max: 200,
    step: 5,
    formatValue: (v) => `${Math.round(v)}%`,
    minLabel: "0%",
    maxLabel: "200%",
  },
  {
    code: "G.",
    key: "fx_inflation_sensitivity_x",
    label: "FX Inflation Sensitivity",
    info: "Macro sensitivity scaling both FX depreciation drift and the FX-propagated 80% uncertainty band.",
    min: 0,
    max: 2.5,
    step: 0.1,
    formatValue: (v) => `${v.toFixed(1)}x`,
    minLabel: "0.0x",
    maxLabel: "2.5x",
  },
];

export const Accounting5YrSimulator: React.FC<Accounting5YrSimulatorProps> = ({
  datasetName,
  onNavigateToEngineering,
}) => {
  const { applyPreset } = useScenario();
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  const [granularity, setGranularity] = useState<"monthly" | "annual">("monthly");
  const [activePreset, setActivePreset] = useState<PresetKey>("base");
  const [params, setParams] = useState<Accounting5YrParams | null>(null);
  const [data, setData] = useState<Accounting5YrResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const isDark = theme === "dark";

  // Initial dataset-calibrated fetch
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    api
      .simulateAccounting5Yr({})
      .then((res) => {
        if (!cancelled) {
          setData(res);
          setParams(res.params);
          setActivePreset("base");
          setError(null);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to initialize 5-year simulator.");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [datasetName]);

  // Debounced live simulation whenever user adjusts A-G sliders
  useEffect(() => {
    if (!params) return;
    let cancelled = false;
    const timer = setTimeout(async () => {
      try {
        const res = await api.simulateAccounting5Yr({ ...params, horizon_months: 60 });
        if (!cancelled) {
          setData(res);
          setError(null);
        }
      } catch (err: unknown) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Simulation error.");
        }
      }
    }, 110);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [params]);

  const handleSelectPreset = (key: "base" | "opt" | "stress") => {
    if (!data?.presets) return;
    setActivePreset(key);
    setParams(data.presets[key]);
    if (key === "base") applyPreset("base");
    else if (key === "opt") applyPreset("optimistic");
    else applyPreset("fx_shock");
  };

  const handleSliderChange = (key: keyof Accounting5YrParams, value: number) => {
    if (!params) return;
    setActivePreset("custom");
    setParams({ ...params, [key]: value });
  };

  const chartData = useMemo(() => {
    if (!data) return [];
    return granularity === "monthly" ? data.monthly_series : data.annual_series;
  }, [data, granularity]);

  const minBandBound = useMemo(() => {
    if (!chartData.length) return -500_000_000;
    let minV = 0;
    for (const pt of chartData) {
      if (pt.historical_op !== null && pt.historical_op < minV) minV = pt.historical_op;
      if (pt.scenario_op !== null && pt.scenario_op < minV) minV = pt.scenario_op;
      if (pt.baseline_op !== null && pt.baseline_op < minV) minV = pt.baseline_op;
      if (pt.band_80 && pt.band_80[0] < minV) minV = pt.band_80[0];
    }
    return minV < 0 ? minV * 1.12 : -50_000_000;
  }, [chartData]);

  const exportHref = useMemo(() => {
    if (!params) return "/api/py/scenario/accounting-5yr/export";
    const qs = new URLSearchParams({
      annual_inflation_pct: String(params.annual_inflation_pct),
      unit_sales_growth_pct: String(params.unit_sales_growth_pct),
      price_inflation_pass_through_pct: String(params.price_inflation_pass_through_pct),
      fx_cost_pass_through_pct: String(params.fx_cost_pass_through_pct),
      usd_import_cost_growth_pct: String(params.usd_import_cost_growth_pct),
      opex_inflation_sensitivity_pct: String(params.opex_inflation_sensitivity_pct),
      fx_inflation_sensitivity_x: String(params.fx_inflation_sensitivity_x),
      horizon_months: "60",
    });
    return `/api/py/scenario/accounting-5yr/export?${qs.toString()}`;
  }, [params]);

  const formatShortNaira = (val: number) => {
    const abs = Math.abs(val);
    const sign = val < 0 ? "-" : "";
    if (abs >= 1e9) return `${sign}₦${(abs / 1e9).toFixed(2)}B`;
    if (abs >= 1e6) return `${sign}₦${(abs / 1e6).toFixed(2)}M`;
    if (abs >= 1e3) return `${sign}₦${(abs / 1e3).toFixed(1)}K`;
    return `${sign}₦${abs.toFixed(0)}`;
  };

  const renderNegativeDot = (props: {
    cx?: number;
    cy?: number;
    payload?: { negative_op?: number | null; month_label?: string };
    index?: number;
  }) => {
    const { cx, cy, payload, index } = props;
    if (
      payload?.negative_op === null ||
      payload?.negative_op === undefined ||
      cx === undefined ||
      cy === undefined
    ) {
      return <g key={`empty-${index}`} />;
    }
    return (
      <circle
        key={`neg-${payload.month_label}-${index}`}
        cx={cx}
        cy={cy}
        r={4}
        fill="#F43F5E"
        stroke={isDark ? "#07111F" : "#FFFFFF"}
        strokeWidth={1.5}
      />
    );
  };

  const anchorLabel =
    granularity === "monthly"
      ? data?.anchor_month_label
      : String(data?.annual_series.find((a) => a.period_type === "mixed")?.month_label || "");

  return (
    <div
      className={`rounded-2xl border p-5 transition-colors ${
        isDark
          ? "border-slate-800 bg-[#07111F] text-slate-100"
          : "border-slate-200 bg-white text-slate-900"
      }`}
    >
      {/* Top Cockpit Header */}
      <div
        className={`flex flex-col gap-3 border-b pb-4 lg:flex-row lg:items-center lg:justify-between ${
          isDark ? "border-slate-800/90" : "border-slate-200"
        }`}
      >
        <div>
          <h2
            className={`font-display text-base font-bold tracking-tight ${
              isDark ? "text-white" : "text-[#0A1E3D]"
            }`}
          >
            Predictive Analytics &amp; 5-Year Accounting Simulator
          </h2>
          <div
            className={`mt-1 flex flex-wrap items-center gap-x-2.5 gap-y-0.5 text-xs ${
              isDark ? "text-slate-400" : "text-slate-500"
            }`}
          >
            <span className={isDark ? "font-mono text-teal-400" : "font-mono font-semibold text-[#0A2E65]"}>
              {datasetName}
            </span>
            <span aria-hidden="true">·</span>
            <span className="tabular-nums">{data?.history_range_label || "Loading history..."}</span>
            <span aria-hidden="true">·</span>
            <span className="tabular-nums">
              {data ? `${data.historical_records} monthly observations` : "Calibrating baseline"}
            </span>
            {data && (
              <>
                <span aria-hidden="true">·</span>
                <span className="tabular-nums">
                  Horizon: {data.forecast_horizon_months} months ({data.summary.final_forecast_month})
                </span>
              </>
            )}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2 shrink-0">
          {/* Light / Dark Cockpit Switch */}
          <div
            className={`inline-flex items-center rounded-lg border p-0.5 text-xs font-semibold ${
              isDark ? "border-slate-700 bg-[#0C1A2E]" : "border-slate-200 bg-slate-100"
            }`}
          >
            <button
              type="button"
              onClick={() => setTheme("light")}
              className={`inline-flex items-center gap-1 rounded-md px-2.5 py-1 transition-colors whitespace-nowrap ${
                !isDark
                  ? "bg-white text-[#0A1E3D] shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <Sun className="h-3.5 w-3.5" />
              <span>Light</span>
            </button>
            <button
              type="button"
              onClick={() => setTheme("dark")}
              className={`inline-flex items-center gap-1 rounded-md px-2.5 py-1 transition-colors whitespace-nowrap ${
                isDark
                  ? "bg-gradient-to-r from-[#1D63B8] to-[#0D9488] text-white font-bold shadow-sm"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              <Moon className="h-3.5 w-3.5" />
              <span>Dark</span>
            </button>
          </div>

          {onNavigateToEngineering && (
            <button
              type="button"
              onClick={onNavigateToEngineering}
              className={`inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-semibold transition-colors whitespace-nowrap ${
                isDark
                  ? "border-emerald-700/60 bg-emerald-950/40 text-emerald-300 hover:bg-emerald-900/50"
                  : "border-emerald-200 bg-emerald-50 text-emerald-800 hover:bg-emerald-100"
              }`}
            >
              <ShieldCheck className="h-3.5 w-3.5" />
              <span>Data Audit (Clean)</span>
            </button>
          )}

          <a
            href={exportHref}
            className={`inline-flex items-center gap-1.5 rounded-lg px-3.5 py-1.5 text-xs font-bold transition-all whitespace-nowrap ${
              isDark
                ? "bg-gradient-to-r from-[#1D63B8] to-[#14B8A6] text-white hover:brightness-110 shadow-sm"
                : "triad-btn-primary text-white shadow-sm"
            }`}
          >
            <Download className="h-3.5 w-3.5" />
            <span>Export CSV</span>
          </a>
        </div>
      </div>

      {error && (
        <div className="mt-4 rounded-lg border border-red-400/50 bg-red-950/40 px-4 py-2.5 text-xs text-red-200">
          {error}
        </div>
      )}

      {/* 4 KPI Summary Cards */}
      <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <div
          className={`rounded-xl border p-4 ${
            isDark ? "border-slate-800 bg-[#0B1728]" : "border-slate-200 bg-slate-50/70"
          }`}
        >
          <div className={`text-[11px] font-medium ${isDark ? "text-slate-400" : "text-slate-500"}`}>
            Latest Monthly OP ({data?.summary.latest_actual_month || "Latest"})
          </div>
          <div
            className={`mt-1.5 font-mono text-xl font-bold tabular-nums ${
              isDark ? "text-white" : "text-[#0A1E3D]"
            }`}
          >
            {data ? formatShortNaira(data.summary.latest_actual_op_ngn) : "—"}
          </div>
        </div>

        <div
          className={`rounded-xl border p-4 ${
            isDark ? "border-slate-800 bg-[#0B1728]" : "border-slate-200 bg-slate-50/70"
          }`}
        >
          <div className={`text-[11px] font-medium ${isDark ? "text-slate-400" : "text-slate-500"}`}>
            Forecast Monthly OP ({data?.summary.final_forecast_month || "Year +5"})
          </div>
          <div className="mt-1.5 flex items-baseline gap-2 tabular-nums">
            <span
              className={`font-mono text-xl font-bold ${
                data && data.summary.final_forecast_op_ngn < 0
                  ? "text-rose-400"
                  : isDark
                    ? "text-teal-300"
                    : "text-teal-700"
              }`}
            >
              {data ? formatShortNaira(data.summary.final_forecast_op_ngn) : "—"}
            </span>
            {data && Math.abs(data.summary.final_op_delta_ngn) > 1 && (
              <span
                className={`font-mono text-xs font-semibold ${
                  data.summary.final_op_delta_ngn >= 0
                    ? isDark
                      ? "text-emerald-400"
                      : "text-emerald-700"
                    : "text-rose-400"
                }`}
              >
                ({data.summary.final_op_delta_ngn >= 0 ? "+" : ""}
                {formatShortNaira(data.summary.final_op_delta_ngn)})
              </span>
            )}
          </div>
        </div>

        <div
          className={`rounded-xl border p-4 ${
            isDark ? "border-slate-800 bg-[#0B1728]" : "border-slate-200 bg-slate-50/70"
          }`}
        >
          <div className={`text-[11px] font-medium ${isDark ? "text-slate-400" : "text-slate-500"}`}>
            5-Yr Cumulative OP
          </div>
          <div className="mt-1.5 flex flex-wrap items-baseline gap-2 tabular-nums">
            <span
              className={`font-mono text-xl font-bold ${
                isDark ? "text-white" : "text-[#0A1E3D]"
              }`}
            >
              {data ? formatShortNaira(data.summary.cumulative_5yr_op_ngn) : "—"}
            </span>
            {data && (
              <span
                className={`font-mono text-xs font-semibold ${
                  data.summary.cumulative_5yr_delta_ngn >= 0
                    ? isDark
                      ? "text-emerald-400"
                      : "text-emerald-700"
                    : "text-rose-400"
                }`}
              >
                ({data.summary.cumulative_5yr_delta_ngn >= 0 ? "+" : ""}
                {formatShortNaira(data.summary.cumulative_5yr_delta_ngn)})
              </span>
            )}
          </div>
        </div>

        <div
          className={`rounded-xl border p-4 ${
            isDark ? "border-slate-800 bg-[#0B1728]" : "border-slate-200 bg-slate-50/70"
          }`}
        >
          <div className={`text-[11px] font-medium ${isDark ? "text-slate-400" : "text-slate-500"}`}>
            5-Yr Operating Margin
          </div>
          <div className="mt-1.5 flex items-baseline gap-2 tabular-nums">
            <span
              className={`font-mono text-xl font-bold ${
                data && data.summary.operating_margin_5yr_pct < 10
                  ? "text-amber-400"
                  : isDark
                    ? "text-white"
                    : "text-[#0A1E3D]"
              }`}
            >
              {data ? `${data.summary.operating_margin_5yr_pct.toFixed(2)}%` : "—"}
            </span>
            {data && (
              <span className={`text-[11px] ${isDark ? "text-slate-400" : "text-slate-500"}`}>
                Base: {data.summary.baseline_margin_5yr_pct.toFixed(1)}%
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Main 2-Column Cockpit: Left Trajectory Chart + Right Live Scenario Controls */}
      <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-12 items-stretch">
        {/* Left Chart Card */}
        <div
          className={`flex flex-col rounded-xl border p-5 lg:col-span-8 ${
            isDark ? "border-slate-800 bg-[#0B1728]" : "border-slate-200 bg-white"
          }`}
        >
          <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
            <div>
              <h3
                className={`font-display text-sm font-bold ${
                  isDark ? "text-white" : "text-[#0A1E3D]"
                }`}
              >
                Business Operating Profit Forecast ({data?.full_range_label || "Historical + 5-Yr Horizon"})
              </h3>
              <p className={`mt-0.5 text-xs ${isDark ? "text-slate-400" : "text-slate-500"}`}>
                Accounting-linked projection (Revenue − COGS − Marketing − Overhead) · Negative-profit periods highlighted below zero line
              </p>
            </div>

            <div
              className={`inline-flex shrink-0 rounded-lg border p-0.5 text-xs font-semibold ${
                isDark ? "border-slate-700 bg-[#07111F]" : "border-slate-200 bg-slate-100"
              }`}
            >
              <button
                type="button"
                onClick={() => setGranularity("monthly")}
                className={`rounded-md px-3 py-1 transition-colors whitespace-nowrap ${
                  granularity === "monthly"
                    ? isDark
                      ? "bg-gradient-to-r from-[#1D63B8] to-[#0D9488] text-white font-bold shadow-sm"
                      : "bg-gradient-to-r from-[#0A2E65] via-[#1D63B8] to-[#0D9488] text-white"
                    : isDark
                      ? "text-slate-300 hover:text-white"
                      : "text-slate-600 hover:text-slate-900"
                }`}
              >
                Monthly
              </button>
              <button
                type="button"
                onClick={() => setGranularity("annual")}
                className={`rounded-md px-3 py-1 transition-colors whitespace-nowrap ${
                  granularity === "annual"
                    ? isDark
                      ? "bg-gradient-to-r from-[#1D63B8] to-[#0D9488] text-white font-bold shadow-sm"
                      : "bg-gradient-to-r from-[#0A2E65] via-[#1D63B8] to-[#0D9488] text-white"
                    : isDark
                      ? "text-slate-300 hover:text-white"
                      : "text-slate-600 hover:text-slate-900"
                }`}
              >
                Annual
              </button>
            </div>
          </div>

          {/* Explicit Legend Row */}
          <div
            className={`mt-4 flex flex-wrap items-center gap-x-3 gap-y-1.5 text-[11px] ${
              isDark ? "text-slate-300" : "text-slate-600"
            }`}
          >
            <span className="inline-flex items-center gap-1.5">
              <span
                className="inline-block h-0.5 w-4"
                style={{ backgroundColor: isDark ? "#38BDF8" : "#0A2E65" }}
              />
              <span>Historical OP</span>
            </span>
            <span className="inline-flex items-center gap-1.5">
              <span className="inline-block h-0.5 w-4 border-t-2 border-dashed border-slate-400" />
              <span>Model Baseline</span>
            </span>
            <span className="inline-flex items-center gap-1.5">
              <span
                className="inline-block h-0.5 w-4 border-t-2 border-dotted"
                style={{ borderColor: isDark ? "#10B981" : "#0D9488" }}
              />
              <span>User-Adjusted Scenario</span>
            </span>
            <span className="inline-flex items-center gap-1.5">
              <span
                className="inline-block h-2.5 w-3.5 rounded-sm"
                style={{ backgroundColor: isDark ? "rgba(20,184,166,0.28)" : "rgba(13,148,136,0.22)" }}
              />
              <span>FX-Propagated 80% Band</span>
            </span>
            <span className="inline-flex items-center gap-1.5">
              <span className="inline-block h-2.5 w-2.5 rounded-full bg-rose-500" />
              <span>Negative Profit Period</span>
            </span>
          </div>

          {/* Chart Viewport */}
          <div className="mt-3 flex-1 min-h-[390px]">
            {loading && !data ? (
              <div
                className={`flex h-full min-h-[390px] items-center justify-center rounded-lg border text-xs ${
                  isDark
                    ? "border-slate-800 bg-[#07111F]/60 text-slate-400"
                    : "border-slate-200 bg-slate-50 text-slate-500"
                }`}
              >
                Running 5-year accounting &amp; FX propagation simulation...
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={400}>
                <ComposedChart data={chartData} margin={{ top: 18, right: 18, left: 6, bottom: 8 }}>
                  <TriadSvgDefs idPrefix="sim5yr" />
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke={isDark ? "#1E293B" : "#E2E8F0"}
                    vertical={false}
                  />
                  <XAxis
                    dataKey="month_label"
                    tick={{ fontSize: 10, fill: isDark ? "#94A3B8" : "#64748B" }}
                    minTickGap={28}
                    axisLine={{ stroke: isDark ? "#334155" : "#CBD5E1" }}
                    tickLine={false}
                  />
                  <YAxis
                    tick={{ fontSize: 10, fill: isDark ? "#94A3B8" : "#64748B" }}
                    tickFormatter={(v) => formatNGN(Number(v), true)}
                    width={64}
                    axisLine={false}
                    tickLine={false}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: isDark ? "rgba(7, 17, 31, 0.96)" : "rgba(10, 23, 45, 0.96)",
                      borderColor: isDark ? "rgba(56, 189, 248, 0.3)" : "rgba(29, 99, 184, 0.35)",
                      borderRadius: "10px",
                      fontSize: "12px",
                      color: "#F8FAFC",
                      boxShadow: "0 12px 28px -6px rgba(2, 6, 23, 0.55)",
                    }}
                    labelStyle={{ color: "#94A3B8", fontWeight: 600 }}
                    itemStyle={{ color: "#F8FAFC" }}
                    formatter={(val: unknown, name: unknown) => {
                      if (Array.isArray(val)) {
                        return [
                          `${formatNGN(Number(val[0]), true)} to ${formatNGN(Number(val[1]), true)}`,
                          "FX-Propagated 80% Band",
                        ];
                      }
                      return [formatNGN(Number(val)), String(name)];
                    }}
                  />
                  {/* Negative-profit territory subtle shading */}
                  <ReferenceArea
                    y1={minBandBound}
                    y2={0}
                    fill="#F43F5E"
                    fillOpacity={isDark ? 0.08 : 0.05}
                  />
                  <ReferenceLine
                    y={0}
                    stroke="#F43F5E"
                    strokeWidth={1.3}
                    label={({ viewBox }: { viewBox?: { x?: number; y?: number } }) => {
                      const vx = viewBox?.x ?? 0;
                      const vy = viewBox?.y ?? 0;
                      return (
                        <text
                          x={vx + 8}
                          y={vy - 6}
                          fill="#F43F5E"
                          fontSize={10}
                          fontWeight={600}
                        >
                          Break-Even (₦0)
                        </text>
                      );
                    }}
                  />
                  {anchorLabel && (
                    <ReferenceLine
                      x={anchorLabel}
                      stroke={isDark ? "#38BDF8" : "#1D63B8"}
                      strokeDasharray="4 4"
                      strokeOpacity={0.65}
                    />
                  )}
                  <Area
                    type="monotone"
                    dataKey="band_80"
                    name="FX-Propagated 80% Band"
                    stroke="none"
                    fill={isDark ? "url(#sim5yr-band-dark)" : "url(#sim5yr-band-light)"}
                    fillOpacity={1}
                    connectNulls={false}
                    isAnimationActive={false}
                  />
                  <Area
                    type="monotone"
                    dataKey="historical_op"
                    name="Historical OP"
                    stroke={isDark ? "url(#sim5yr-stroke-dark-hist)" : "url(#sim5yr-stroke-navy-cobalt)"}
                    fill={isDark ? "url(#sim5yr-area-cobalt)" : "url(#sim5yr-area-navy)"}
                    fillOpacity={1}
                    strokeWidth={2.2}
                    dot={renderNegativeDot}
                    connectNulls={false}
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="baseline_op"
                    name="Model Baseline"
                    stroke="#94A3B8"
                    strokeWidth={1.7}
                    strokeDasharray="5 4"
                    dot={false}
                    connectNulls={false}
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="scenario_op"
                    name="User-Adjusted Scenario"
                    stroke={
                      data && data.summary.forecast_negative_months > 0
                        ? "url(#sim5yr-stroke-dark-stress)"
                        : isDark
                          ? "url(#sim5yr-stroke-dark-sim)"
                          : "url(#sim5yr-stroke-cobalt-teal)"
                    }
                    strokeWidth={2.5}
                    strokeDasharray="3 3"
                    dot={renderNegativeDot}
                    connectNulls={false}
                    isAnimationActive={false}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            )}
          </div>

          {data && (
            <div
              className={`mt-3 flex flex-wrap items-center justify-between gap-2 border-t pt-3 text-[11px] tabular-nums ${
                isDark ? "border-slate-800 text-slate-400" : "border-slate-100 text-slate-500"
              }`}
            >
              <span>
                Historical negative-profit months: <strong className="text-rose-400">{data.summary.historical_negative_months}</strong>
                {" · "}Simulated deficit months:{" "}
                <strong className={data.summary.forecast_negative_months > 0 ? "text-rose-400" : "text-teal-400"}>
                  {data.summary.forecast_negative_months}
                </strong>
                {" · "}80% FX lower-band breaches below ₦0:{" "}
                <strong className={data.summary.band_breach_months > 0 ? "text-amber-400" : "text-teal-400"}>
                  {data.summary.band_breach_months} mos
                </strong>
              </span>
              <span>Base Case 5-Yr Cumulative: {formatShortNaira(data.summary.cumulative_5yr_baseline_op_ngn)}</span>
            </div>
          )}
        </div>

        {/* Right Controls Card: LIVE SCENARIO CONTROLS */}
        <div
          className={`flex flex-col rounded-xl border p-5 lg:col-span-4 ${
            isDark ? "border-slate-800 bg-[#0B1728]" : "border-slate-200 bg-slate-50/60"
          }`}
        >
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <SlidersHorizontal className={`h-4 w-4 ${isDark ? "text-teal-400" : "text-[#0A2E65]"}`} />
              <h3
                className={`text-xs font-bold tracking-wider uppercase ${
                  isDark ? "text-white" : "text-[#0A1E3D]"
                }`}
              >
                Live Scenario Controls
              </h3>
            </div>
            <button
              type="button"
              onClick={() => handleSelectPreset("base")}
              className={`inline-flex items-center gap-1 rounded-md border px-2.5 py-1 text-[11px] font-semibold transition-colors whitespace-nowrap ${
                isDark
                  ? "border-slate-700 bg-[#07111F] text-slate-300 hover:bg-slate-800"
                  : "border-slate-200 bg-white text-slate-700 hover:bg-slate-100"
              }`}
            >
              <RotateCcw className="h-3 w-3" />
              <span>Reset</span>
            </button>
          </div>

          {/* 4 Preset Tabs: Base | Opt | Stress | Custom */}
          <div
            className={`mt-3 grid grid-cols-4 gap-1 rounded-lg border p-1 text-xs font-semibold ${
              isDark ? "border-slate-800 bg-[#07111F]" : "border-slate-200 bg-white"
            }`}
          >
            {(
              [
                { id: "base", label: "Base" },
                { id: "opt", label: "Opt" },
                { id: "stress", label: "Stress" },
                { id: "custom", label: "Custom" },
              ] as const
            ).map((tab) => {
              const selected = activePreset === tab.id;
              return (
                <button
                  key={tab.id}
                  type="button"
                  disabled={tab.id === "custom" && activePreset !== "custom"}
                  onClick={() => {
                    if (tab.id !== "custom") handleSelectPreset(tab.id);
                  }}
                  className={`rounded-md py-1.5 text-center transition-colors whitespace-nowrap ${
                    selected
                      ? isDark
                        ? "bg-gradient-to-r from-[#1D63B8] to-[#0D9488] text-white font-bold shadow-sm"
                        : "bg-gradient-to-r from-[#0A2E65] via-[#1D63B8] to-[#0D9488] text-white"
                      : tab.id === "custom"
                        ? isDark
                          ? "text-slate-500 cursor-default"
                          : "text-slate-400 cursor-default"
                        : isDark
                          ? "text-slate-300 hover:bg-slate-800/80 hover:text-white"
                          : "text-slate-600 hover:bg-slate-100"
                  }`}
                >
                  {tab.label}
                </button>
              );
            })}
          </div>

          <p className={`mt-3 text-[11px] leading-relaxed ${isDark ? "text-slate-400" : "text-slate-500"}`}>
            Adjust demand growth (B), pass-through levers (C, D), supplier USD cost (E), or OpEx sensitivity (F) to watch operating profit respond live.
          </p>

          {/* 7 Interactive Sliders (A through G) */}
          <div className="mt-4 flex-1 space-y-3.5">
            {params &&
              SLIDER_CONFIGS.map((cfg) => {
                const val = params[cfg.key];
                return (
                  <div key={cfg.key} className="space-y-1">
                    <div className="flex items-center justify-between gap-2 text-xs">
                      <label
                        htmlFor={`sim-${cfg.key}`}
                        className={`inline-flex items-center gap-1 font-medium ${
                          isDark ? "text-slate-200" : "text-slate-700"
                        }`}
                      >
                        <span>
                          <span className={isDark ? "text-teal-400 font-semibold" : "text-[#0A2E65] font-semibold"}>
                            {cfg.code}
                          </span>{" "}
                          {cfg.label}
                        </span>
                        <span title={cfg.info} className="cursor-help opacity-60 hover:opacity-100">
                          <Info className="h-3 w-3" />
                        </span>
                      </label>
                      <span
                        className={`font-mono text-xs font-bold tabular-nums ${
                          isDark ? "text-teal-300" : "text-[#0A2E65]"
                        }`}
                      >
                        {cfg.formatValue(val)}
                      </span>
                    </div>

                    <input
                      id={`sim-${cfg.key}`}
                      type="range"
                      min={cfg.min}
                      max={cfg.max}
                      step={cfg.step}
                      value={val}
                      style={getSliderFillStyle(val, cfg.min, cfg.max)}
                      onChange={(e) => handleSliderChange(cfg.key, Number(e.target.value))}
                      className={`w-full ${isDark ? "triad-slider-dark" : "triad-slider"}`}
                    />

                    <div
                      className={`flex justify-between text-[10px] font-mono tabular-nums ${
                        isDark ? "text-slate-500" : "text-slate-400"
                      }`}
                    >
                      <span>{cfg.minLabel}</span>
                      <span>{cfg.maxLabel}</span>
                    </div>
                  </div>
                );
              })}
          </div>
        </div>
      </div>
    </div>
  );
};
