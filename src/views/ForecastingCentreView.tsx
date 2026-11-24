import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  AlertTriangle,
  Award,
  BookOpen,
  CheckCircle2,
  Download,
  FileSpreadsheet,
  FlaskConical,
  RefreshCw,
  ShieldCheck,
  SlidersHorizontal,
  TrendingUp,
} from "lucide-react";
import { api } from "../services/api";
import { SCENARIO_PRESETS, useScenario } from "../context/ScenarioContext";
import {
  CustomScenarioVerdict,
  ForecastScenariosBundle,
  ForecastingCentreResponse,
  HypothesisTestingBundle,
  OperationalVerdict,
} from "../types/financial";
import { formatCompactNumber, formatNGN, formatPct } from "../utils/format";
import {
  getSliderFillStyle,
  TRIAD_TOOLTIP_ITEM_STYLE,
  TRIAD_TOOLTIP_LABEL_STYLE,
  TRIAD_TOOLTIP_STYLE,
  TriadSvgDefs,
} from "../utils/chartTheme";

interface ForecastingCentreViewProps {
  datasetName: string;
  recordCount: number;
}

type PrimaryMetricKey =
  | "operating_profit_ngn"
  | "sales_revenue_ngn"
  | "gross_profit_ngn"
  | "operating_expenses_ngn"
  | "operating_margin_pct"
  | "units_sold"
  | "profit_per_unit_ngn";

const METRIC_OPTIONS: { key: PrimaryMetricKey; label: string; isCurrency: boolean; isPct?: boolean }[] = [
  { key: "operating_profit_ngn", label: "Operating Profit (NGN)", isCurrency: true },
  { key: "sales_revenue_ngn", label: "Sales Revenue (NGN)", isCurrency: true },
  { key: "gross_profit_ngn", label: "Gross Profit (NGN)", isCurrency: true },
  { key: "operating_expenses_ngn", label: "Total Operating Expenses (NGN)", isCurrency: true },
  { key: "operating_margin_pct", label: "Operating Margin (%)", isCurrency: false, isPct: true },
  { key: "units_sold", label: "Units Sold", isCurrency: false },
  { key: "profit_per_unit_ngn", label: "Operating Profit / Unit (NGN)", isCurrency: true },
];


export const ForecastingCentreView: React.FC<ForecastingCentreViewProps> = ({
  datasetName,
  recordCount,
}) => {
  const {
    presetId,
    activePresetLabel,
    isScenarioActive,
    forecastAssumptions: customAssumptions,
    setForecastAssumptions: setCustomAssumptions,
    applyPreset,
    resetScenario,
    forecastScenarios: ctxScenarios,
    customVerdict: ctxCustomVerdict,
    forecastLoading: simLoading,
    setDrawerOpen,
  } = useScenario();

  const [data, setData] = useState<ForecastingCentreResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Chart & Table View Controls
  const [selectedMetric, setSelectedMetric] = useState<PrimaryMetricKey>("operating_profit_ngn");
  const [chartWindow, setChartWindow] = useState<"full" | "recent">("full");
  const [showSimOverlay, setShowSimOverlay] = useState<boolean>(true);
  const [showEnvelope, setShowEnvelope] = useState<boolean>(false);
  const [summaryTab, setSummaryTab] = useState<"annual" | "quarterly" | "scenario_diff" | "risk">("annual");

  // Hypothesis Testing Lab State
  const [alpha, setAlpha] = useState<number>(0.05);
  const [splitYear, setSplitYear] = useState<number>(2023);
  const [hypBundle, setHypBundle] = useState<HypothesisTestingBundle | null>(null);
  const [hypLoading, setHypLoading] = useState<boolean>(false);

  // Fallback local scenarios/verdict before context populates
  const [initialScenarios, setInitialScenarios] = useState<ForecastScenariosBundle | null>(null);
  const [verdictState, setVerdictState] = useState<OperationalVerdict | null>(null);
  const [initialCustomVerdict, setInitialCustomVerdict] = useState<CustomScenarioVerdict | null>(null);

  // Feature Dictionary toggle
  const [showAllFeatures, setShowAllFeatures] = useState<boolean>(false);

  const scenariosBundle = ctxScenarios || initialScenarios;
  const activeCustomVerdict = ctxCustomVerdict || initialCustomVerdict;

  const loadCentre = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.getForecastingCentre();
      setData(res);
      if (res.hypothesis_testing) {
        setHypBundle(res.hypothesis_testing);
        setAlpha(res.hypothesis_testing.alpha);
        setSplitYear(res.hypothesis_testing.split_year);
      }
      if (res.scenarios) {
        setInitialScenarios(res.scenarios);
      }
      if (res.verdict) {
        setVerdictState(res.verdict);
      }
      if (res.custom_verdict) {
        setInitialCustomVerdict(res.custom_verdict);
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to load 2030 Forecasting Centre.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadCentre();
  }, [loadCentre, datasetName, recordCount]);

  const handleRunHypotheses = async (nextAlpha: number, nextSplit: number) => {
    setAlpha(nextAlpha);
    setSplitYear(nextSplit);
    setHypLoading(true);
    try {
      const res = await api.runHypothesisTests(nextAlpha, nextSplit);
      setHypBundle(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to run hypothesis tests.");
    } finally {
      setHypLoading(false);
    }
  };

  const selectedMetricMeta = useMemo(
    () => METRIC_OPTIONS.find((m) => m.key === selectedMetric) || METRIC_OPTIONS[0],
    [selectedMetric]
  );

  const supportsSimOverlay = useMemo(
    () =>
      selectedMetric === "operating_profit_ngn" ||
      selectedMetric === "sales_revenue_ngn" ||
      selectedMetric === "operating_expenses_ngn" ||
      selectedMetric === "operating_margin_pct",
    [selectedMetric]
  );

  // Build unified Historical + Forecast + Live Simulated Scenario chart data
  const timelineChartData = useMemo(() => {
    if (!data?.historical_months || !data?.forecast_months) return [];
    const hist =
      chartWindow === "recent" ? data.historical_months.slice(-48) : data.historical_months;
    const lastHist = hist[hist.length - 1];
    const simMap = new Map(
      (scenariosBundle?.monthly_comparison || []).map((row) => [row.month, row])
    );

    const points = hist.map((m) => ({
      month: m.month,
      month_label: m.month_label,
      historicalVal: m[selectedMetric] as number,
      forecastVal: null as number | null,
      simulatedVal: null as number | null,
      optimisticVal: null as number | null,
      stressVal: null as number | null,
      ciRange: null as [number, number] | null,
      lower95: null as number | null,
      upper95: null as number | null,
      isRisk: m.is_risk_period,
    }));

    // Connect forecast and simulated lines smoothly from last actual point
    if (lastHist && points.length > 0) {
      const anchorVal = lastHist[selectedMetric] as number;
      points[points.length - 1].forecastVal = anchorVal;
      if (supportsSimOverlay) {
        points[points.length - 1].simulatedVal = anchorVal;
        if (selectedMetric === "operating_profit_ngn") {
          points[points.length - 1].optimisticVal = anchorVal;
          points[points.length - 1].stressVal = anchorVal;
        }
      }
    }

    data.forecast_months.forEach((m) => {
      const val = m[selectedMetric] as number;
      const isOp = selectedMetric === "operating_profit_ngn";
      const simRow = simMap.get(m.month);

      let simVal: number | null = null;
      if (simRow && supportsSimOverlay) {
        if (selectedMetric === "operating_profit_ngn") simVal = simRow.custom;
        else if (selectedMetric === "sales_revenue_ngn") simVal = simRow.custom_revenue ?? null;
        else if (selectedMetric === "operating_expenses_ngn") simVal = simRow.custom_opex ?? null;
        else if (selectedMetric === "operating_margin_pct") simVal = simRow.custom_margin ?? null;
      }

      points.push({
        month: m.month,
        month_label: m.month_label,
        historicalVal: null as unknown as number,
        forecastVal: val,
        simulatedVal: simVal,
        optimisticVal: isOp && simRow ? simRow.optimistic : null,
        stressVal: isOp && simRow ? simRow.stress : null,
        ciRange:
          isOp && m.lower_95_ngn !== null && m.upper_95_ngn !== null
            ? [m.lower_95_ngn, m.upper_95_ngn]
            : null,
        lower95: isOp ? m.lower_95_ngn : null,
        upper95: isOp ? m.upper_95_ngn : null,
        isRisk: m.is_risk_period,
      });
    });

    return points;
  }, [data, selectedMetric, chartWindow, scenariosBundle, supportsSimOverlay]);

  // Transform 2026-2030 lever attribution waterfall into floating [low, high] bars
  const leverWaterfallData = useMemo(() => {
    const steps = scenariosBundle?.lever_attribution_waterfall || [];
    let running = 0;
    return steps.map((w) => {
      if (w.type === "baseline") {
        running = w.value;
        return { step: w.step, range: [0, w.value], value: w.value, color: "url(#fc-wf-bar-navy-cobalt)" };
      }
      if (w.type === "total") {
        return {
          step: w.step,
          range: [0, w.value],
          value: w.value,
          color: w.value >= 0 ? "url(#fc-wf-bar-cobalt-teal)" : "url(#fc-wf-bar-negative)",
        };
      }
      const prev = running;
      running += w.value;
      return {
        step: w.step,
        range: [Math.min(prev, running), Math.max(prev, running)],
        value: w.value,
        color: w.value >= 0 ? "url(#fc-wf-bar-cobalt-teal)" : "url(#fc-wf-bar-negative)",
      };
    });
  }, [scenariosBundle]);

  const forecastStartLabel = data?.forecast_months?.[0]?.month_label || "";

  const riskMonthsList = useMemo(
    () => (data?.forecast_months || []).filter((m) => m.is_negative_profit || m.is_risk_period),
    [data]
  );

  const availableSplitYears = useMemo(() => {
    if (!data?.annual_summaries) return [2020, 2021, 2022, 2023, 2024];
    const histYears = data.annual_summaries
      .filter((a) => a.historical_months > 0)
      .map((a) => a.year);
    if (histYears.length <= 2) return histYears;
    return histYears.slice(1, -1);
  }, [data]);

  const handleDownloadPdf = async () => {
    if (!data?.executive_summary || !verdictState) return;
    const [{ default: jsPDF }, { default: autoTable }] = await Promise.all([
      import("jspdf"),
      import("jspdf-autotable"),
    ]);
    const doc = new jsPDF({ orientation: "portrait", unit: "pt", format: "a4" });
    const exec = data.executive_summary;

    doc.setFontSize(15);
    doc.setTextColor(10, 30, 61);
    doc.text("FinSight BI — 2030 Operational Forecasting & Validation Report", 40, 42);

    doc.setFontSize(9);
    doc.setTextColor(80, 95, 115);
    doc.text(
      `Dataset: ${datasetName} | Historical: ${data.dataset_summary?.historical_start} to ${data.dataset_summary?.historical_end} | Forecast: through ${data.dataset_summary?.forecast_end}`,
      40,
      58
    );

    autoTable(doc, {
      startY: 74,
      head: [["2030 Executive Outlook Summary", "Value"]],
      body: [
        ["Verdict Category", verdictState.category],
        ["Latest Actual Operating Profit", `${exec.latest_actual_month}: ${formatNGN(exec.latest_actual_operating_profit_ngn)}`],
        ["December 2030 Projected Profit", `${formatNGN(exec.dec_2030_forecast_operating_profit_ngn)} (${formatPct(exec.forecast_growth_to_2030_pct, 2, true)} vs latest)`],
        ["Dec 2030 95% Prediction Interval", `${formatNGN(exec.dec_2030_lower_95_ngn)} to ${formatNGN(exec.dec_2030_upper_95_ngn)}`],
        ["Cumulative 2026–2030 Profit", formatNGN(exec.projected_cumulative_operating_profit_ngn)],
        ["Projected Operating Margin", formatPct(exec.forecast_operating_margin_pct)],
        ["Champion Forecasting Model", `${exec.champion_model_name} (Holdout RMSE: ${formatNGN(exec.champion_holdout_rmse)})`],
        ["Negative / Risk Months (2026–2030)", `${exec.forecast_negative_months} negative / ${exec.forecast_risk_months} risk`],
      ],
      theme: "grid",
      headStyles: { fillColor: [10, 46, 101] },
      styles: { fontSize: 8.5 },
    });

    if (data.leaderboard?.length) {
      autoTable(doc, {
        startY: (doc as any).lastAutoTable.finalY + 16,
        head: [["Model", "Holdout MAE", "Holdout RMSE", "MAPE %", "Backtest RMSE", "95% Cov %", "Status"]],
        body: data.leaderboard.map((m) => [
          m.name,
          formatNGN(m.mae, true),
          formatNGN(m.rmse, true),
          m.mape_pct !== null ? `${m.mape_pct}%` : "—",
          formatNGN(m.backtest_mean_rmse, true),
          `${m.pi_coverage_95_pct}%`,
          m.status,
        ]),
        theme: "grid",
        headStyles: { fillColor: [13, 148, 136] },
        styles: { fontSize: 8 },
      });
    }

    doc.addPage();
    doc.setFontSize(13);
    doc.setTextColor(10, 30, 61);
    doc.text("Hypothesis Testing & Annual 2026–2030 Trajectory", 40, 42);

    if (hypBundle?.tests?.length) {
      autoTable(doc, {
        startY: 56,
        head: [["Hypothesis", "Test", "Statistic", "p-value", "Decision"]],
        body: hypBundle.tests.map((t) => [
          t.title,
          t.test_used,
          t.statistic !== null ? `${t.statistic_name}: ${t.statistic}` : "—",
          t.p_value !== null ? String(t.p_value) : "—",
          t.decision,
        ]),
        theme: "grid",
        headStyles: { fillColor: [10, 46, 101] },
        styles: { fontSize: 8 },
      });
    }

    if (data.annual_summaries?.length) {
      const recentAnnual = data.annual_summaries.filter((a) => a.year >= 2022);
      autoTable(doc, {
        startY: ((doc as any).lastAutoTable?.finalY || 140) + 16,
        head: [["Year", "Scope", "Operating Profit", "Sales Revenue", "Op Margin %", "YoY %"]],
        body: recentAnnual.map((a) => [
          String(a.year),
          a.period_scope,
          formatNGN(a.total_operating_profit_ngn, true),
          formatNGN(a.total_sales_revenue_ngn, true),
          formatPct(a.weighted_operating_margin_pct),
          a.yoy_operating_profit_pct !== null ? formatPct(a.yoy_operating_profit_pct, 1, true) : "—",
        ]),
        theme: "grid",
        headStyles: { fillColor: [13, 148, 136] },
        styles: { fontSize: 8 },
      });
    }

    const yEnd = ((doc as any).lastAutoTable?.finalY || 320) + 20;
    doc.setFontSize(10);
    doc.setTextColor(15, 23, 42);
    doc.text("Recommended Management Interventions:", 40, yEnd);
    doc.setFontSize(8.5);
    verdictState.recommended_interventions.forEach((line, idx) => {
      const wrapped = doc.splitTextToSize(`${idx + 1}. ${line}`, 510);
      doc.text(wrapped, 40, yEnd + 16 + idx * 24);
    });

    doc.save(`FinSight_2030_Forecast_Validation_${datasetName.replace(/\.[^/.]+$/, "")}.pdf`);
  };

  if (loading) {
    return (
      <div className="space-y-5">
        <div className="h-24 animate-pulse rounded-xl border border-slate-200 bg-white" />
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {[0, 1, 2, 3, 4, 5].map((i) => (
            <div key={i} className="h-28 animate-pulse rounded-xl border border-slate-200 bg-white" />
          ))}
        </div>
        <div className="h-80 animate-pulse rounded-xl border border-slate-200 bg-white" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-xl border border-red-200 bg-red-50 p-6 text-sm text-red-800 space-y-3">
        <div className="font-semibold">Unable to load 2030 Operational Forecasting Centre</div>
        <div className="text-xs">{error}</div>
        <button
          type="button"
          onClick={() => void loadCentre()}
          className="inline-flex items-center gap-1.5 rounded-lg border border-red-300 bg-white px-3.5 py-2 text-xs font-semibold text-red-800 hover:bg-red-100"
        >
          <RefreshCw className="h-3.5 w-3.5" />
          <span>Retry Forecasting Engine</span>
        </button>
      </div>
    );
  }

  if (!data || !data.sufficient_data || !data.executive_summary || !verdictState) {
    return (
      <div className="rounded-xl border border-amber-200 bg-amber-50 p-6 text-sm text-amber-900 space-y-2">
        <div className="font-bold">Insufficient Historical Observations for 2030 Multi-Horizon Forecasting</div>
        <p className="text-xs leading-relaxed">
          {data?.limitation_notice ||
            "Upload a dataset with at least 24 chronological monthly observations to unlock rolling-origin validation, hypothesis testing, and forecasting through December 2030."}
        </p>
      </div>
    );
  }

  const exec = data.executive_summary;
  const ds = data.dataset_summary!;
  const val = data.validation!;

  const verdictStyle =
    verdictState.category === "Strong Positive Outlook"
      ? { bg: "bg-emerald-50/70", border: "border-emerald-200", text: "text-emerald-900", accent: "text-emerald-700" }
      : verdictState.category === "Positive but Requires Monitoring"
        ? { bg: "bg-teal-50/70", border: "border-teal-200", text: "text-teal-900", accent: "text-teal-700" }
        : verdictState.category === "Stable / Mixed Outlook"
          ? { bg: "bg-slate-50", border: "border-slate-300", text: "text-slate-900", accent: "text-[#0A2E65]" }
          : { bg: "bg-amber-50/70", border: "border-amber-200", text: "text-amber-900", accent: "text-amber-700" };

  const customSummary = scenariosBundle?.cases.custom.summary;
  const baseSummary = scenariosBundle?.cases.base.summary;
  const simDecDelta =
    customSummary && baseSummary
      ? customSummary.dec_2030_operating_profit_ngn - baseSummary.dec_2030_operating_profit_ngn
      : 0;
  const simCumDelta =
    customSummary && baseSummary
      ? customSummary.cumulative_profit_ngn - baseSummary.cumulative_profit_ngn
      : 0;
  const simMarginDelta =
    customSummary && baseSummary && customSummary.cumulative_margin_pct !== undefined && baseSummary.cumulative_margin_pct !== undefined
      ? customSummary.cumulative_margin_pct - baseSummary.cumulative_margin_pct
      : 0;

  return (
    <div className="space-y-6">
      {/* Top Analytical Control & Export Strip */}
      <div className="rounded-xl border border-slate-200 bg-white p-5 flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div className="space-y-1">
          <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
            <span>Dataset: {datasetName}</span>
            <span aria-hidden="true">·</span>
            <span>
              Historical: {ds.historical_start} to {ds.historical_end} ({ds.historical_records} mos)
            </span>
            <span aria-hidden="true">·</span>
            <span>
              Forecast Horizon: {ds.forecast_start} to {ds.forecast_end} ({ds.forecast_horizon_months} mos)
            </span>
          </div>
          <h2 className="font-display text-base font-bold text-[#0A1E3D]">
            Operational Forecasting through December 2030, Interactive Simulator &amp; Validation
          </h2>
          <p className="text-xs text-slate-600 max-w-4xl leading-relaxed">{ds.horizon_note}</p>
        </div>

        <div className="flex flex-wrap items-center gap-2 shrink-0">
          <a
            href="#scenario-simulator-2030"
            className="inline-flex items-center gap-1.5 rounded-lg border border-teal-700/30 bg-teal-50 px-3 py-2 text-xs font-semibold text-teal-900 hover:bg-teal-100 whitespace-nowrap"
          >
            <SlidersHorizontal className="h-3.5 w-3.5 text-teal-700" />
            <span>2030 Simulator</span>
          </a>
          <a
            href="/api/py/forecasting/export/csv"
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-semibold text-slate-800 hover:bg-slate-50 whitespace-nowrap"
          >
            <FileSpreadsheet className="h-3.5 w-3.5 text-teal-700" />
            <span>Forecast CSV</span>
          </a>
          <a
            href="/api/py/forecasting/export/validation-csv"
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-semibold text-slate-800 hover:bg-slate-50 whitespace-nowrap"
          >
            <Award className="h-3.5 w-3.5 text-[#0A2E65]" />
            <span>Validation CSV</span>
          </a>
          <button
            type="button"
            onClick={handleDownloadPdf}
            className="triad-btn-primary inline-flex items-center gap-1.5 rounded-lg px-4 py-2 text-xs font-semibold text-white whitespace-nowrap"
          >
            <Download className="h-3.5 w-3.5" />
            <span>PDF Report</span>
          </button>
        </div>
      </div>

      {/* Section A: Executive Forecast Summary Cards (showing Base + Live Active Scenario Deltas) */}
      <section aria-label="Executive Forecast Summary" className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
        <div className="rounded-xl bg-gradient-to-br from-[#072149] via-[#0E3A7A] to-[#1D63B8] p-5 text-white shadow-sm">
          <div className="text-xs font-medium text-blue-100">
            Latest Actual Operating Profit ({exec.latest_actual_month})
          </div>
          <div className="mt-1.5 font-display text-2xl font-bold tabular-nums">
            {formatNGN(exec.latest_actual_operating_profit_ngn, true)}
          </div>
          <div className="mt-1 text-xs text-blue-100/90 tabular-nums">
            Exact: {formatNGN(exec.latest_actual_operating_profit_ngn)} · Historical loss months:{" "}
            {ds.historical_negative_profit_months}
          </div>
        </div>

        <div className="rounded-xl bg-gradient-to-br from-[#134287] via-[#13699C] to-[#0D9488] p-5 text-white shadow-sm">
          <div className="flex items-center justify-between gap-2 text-xs font-medium text-teal-100">
            <span>December 2030 Forecast Operating Profit</span>
            {isScenarioActive && (
              <span className="font-mono font-bold text-amber-200 whitespace-nowrap">
                {activePresetLabel}
              </span>
            )}
          </div>
          <div className="mt-1.5 flex items-baseline gap-2.5 flex-wrap">
            <span className="font-display text-2xl font-bold tabular-nums">
              {formatNGN(
                isScenarioActive && customSummary
                  ? customSummary.dec_2030_operating_profit_ngn
                  : exec.dec_2030_forecast_operating_profit_ngn,
                true
              )}
            </span>
            {isScenarioActive && customSummary && (
              <span
                className={`font-mono text-xs font-bold tabular-nums ${
                  simDecDelta >= 0 ? "text-emerald-200" : "text-amber-200"
                }`}
              >
                ({simDecDelta >= 0 ? "+" : ""}
                {formatNGN(simDecDelta, true)} vs Base)
              </span>
            )}
          </div>
          <div className="mt-1 text-xs text-teal-100 tabular-nums">
            Base: {formatNGN(exec.dec_2030_forecast_operating_profit_ngn, true)} · 95% PI:{" "}
            {formatNGN(exec.dec_2030_lower_95_ngn, true)} to {formatNGN(exec.dec_2030_upper_95_ngn, true)}
          </div>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-5">
          <div className="text-xs font-medium text-slate-500">
            Cumulative Projected Profit ({ds.forecast_start.slice(0, 7)} – Dec 2030)
          </div>
          <div className="mt-1.5 flex items-baseline gap-2 flex-wrap">
            <span className="font-display text-2xl font-bold text-[#0A1E3D] tabular-nums">
              {formatNGN(
                isScenarioActive && customSummary
                  ? customSummary.cumulative_profit_ngn
                  : exec.projected_cumulative_operating_profit_ngn,
                true
              )}
            </span>
            {isScenarioActive && customSummary && (
              <span
                className={`font-mono text-xs font-bold tabular-nums ${
                  simCumDelta >= 0 ? "text-teal-700" : "text-red-600"
                }`}
              >
                {simCumDelta >= 0 ? "+" : ""}
                {formatNGN(simCumDelta, true)}
              </span>
            )}
          </div>
          <div className="mt-1 text-xs text-slate-600 tabular-nums">
            Base: {formatNGN(exec.projected_cumulative_operating_profit_ngn, true)} across{" "}
            {ds.forecast_horizon_months} forecast months
          </div>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-5">
          <div className="text-xs font-medium text-slate-500">
            Projected 2026–2030 Operating Margin
          </div>
          <div className="mt-1.5 flex items-baseline gap-2 flex-wrap">
            <span className="font-display text-2xl font-bold text-teal-800 tabular-nums">
              {formatPct(
                isScenarioActive && customSummary?.cumulative_margin_pct !== undefined
                  ? customSummary.cumulative_margin_pct
                  : exec.forecast_operating_margin_pct
              )}
            </span>
            {isScenarioActive && (
              <span
                className={`font-mono text-xs font-bold tabular-nums ${
                  simMarginDelta >= 0 ? "text-teal-700" : "text-red-600"
                }`}
              >
                ({formatPct(simMarginDelta, 2, true)} vs Base)
              </span>
            )}
          </div>
          <div className="mt-1 text-xs text-slate-600 tabular-nums">
            Base Model Margin: {formatPct(exec.forecast_operating_margin_pct)}
          </div>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-5">
          <div className="text-xs font-medium text-slate-500">
            Champion Model (Untouched {val.holdout_size}M Holdout)
          </div>
          <div className="mt-1.5 font-display text-lg font-bold text-[#0A1E3D]">
            {exec.champion_model_name}
          </div>
          <div className="mt-1 text-xs text-slate-600 tabular-nums">
            Holdout RMSE: {formatNGN(exec.champion_holdout_rmse, true)} · MAE:{" "}
            {formatNGN(exec.champion_holdout_mae, true)} · MAPE:{" "}
            {exec.champion_holdout_mape_pct !== null ? `${exec.champion_holdout_mape_pct}%` : "—"}
          </div>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-5">
          <div className="text-xs font-medium text-slate-500">
            Dec 2030 Uncertainty &amp; Scenario Risk Months
          </div>
          <div className="mt-1.5 font-display text-2xl font-bold text-amber-700 tabular-nums">
            ±{exec.forecast_uncertainty_95_pct.toFixed(1)}%
          </div>
          <div className="mt-1 text-xs text-slate-600 tabular-nums">
            Base loss/risk: {exec.forecast_negative_months}/{exec.forecast_risk_months}m ·{" "}
            Simulated:{" "}
            <strong
              className={
                (customSummary?.negative_months || 0) > 0 ? "text-red-600" : "text-teal-700"
              }
            >
              {customSummary?.negative_months ?? exec.forecast_negative_months}/
              {customSummary?.risk_months ?? exec.forecast_risk_months}m
            </strong>
          </div>
        </div>
      </section>

      {/* Section B: Historical Actuals vs Predicted Baseline & Live Simulated 2030 Trajectory */}
      <section className="rounded-xl border border-slate-200 bg-white p-5 space-y-4">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between border-b border-slate-100 pb-4">
          <div>
            <h3 className="font-display text-sm font-bold text-[#0A1E3D]">
              Historical Actuals vs. Champion Forecast &amp; Active Scenario Trajectory (through Dec 2030)
            </h3>
            <p className="text-xs text-slate-500">
              Separating verified historical records ({ds.historical_start.slice(0, 7)}–
              {ds.historical_end.slice(0, 7)}) from {exec.champion_model_name} baseline and your live{" "}
              <strong className="text-[#0A1E3D]">{activePresetLabel}</strong> simulation.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {supportsSimOverlay && (
              <button
                type="button"
                onClick={() => setShowSimOverlay((v) => !v)}
                className={`rounded-lg border px-2.5 py-1.5 text-xs font-semibold transition-colors whitespace-nowrap ${
                  showSimOverlay
                    ? "border-blue-600 bg-blue-50 text-blue-800"
                    : "border-slate-300 bg-white text-slate-600 hover:bg-slate-50"
                }`}
              >
                {showSimOverlay ? "Simulated Overlay: On" : "Simulated Overlay: Off"}
              </button>
            )}

            {selectedMetric === "operating_profit_ngn" && (
              <button
                type="button"
                onClick={() => setShowEnvelope((v) => !v)}
                className={`rounded-lg border px-2.5 py-1.5 text-xs font-semibold transition-colors whitespace-nowrap ${
                  showEnvelope
                    ? "border-amber-600 bg-amber-50 text-amber-800"
                    : "border-slate-300 bg-white text-slate-600 hover:bg-slate-50"
                }`}
              >
                {showEnvelope ? "Opt / Stress Envelope: On" : "Opt / Stress Envelope"}
              </button>
            )}

            <select
              aria-label="Select forecast target KPI"
              value={selectedMetric}
              onChange={(e) => setSelectedMetric(e.target.value as PrimaryMetricKey)}
              className="rounded-lg border border-slate-300 bg-slate-50 px-3 py-1.5 text-xs font-semibold text-slate-800 focus:border-[#0A2E65] focus:bg-white focus:outline-none"
            >
              {METRIC_OPTIONS.map((opt) => (
                <option key={opt.key} value={opt.key}>
                  {opt.label}
                </option>
              ))}
            </select>

            <div className="flex items-center gap-1 rounded-lg bg-slate-100 p-1 text-xs font-medium">
              <button
                type="button"
                onClick={() => setChartWindow("full")}
                className={`rounded-md px-2.5 py-1 transition-colors whitespace-nowrap ${
                  chartWindow === "full" ? "bg-white text-[#0A1E3D] shadow-sm" : "text-slate-600"
                }`}
              >
                Full (2016–2030)
              </button>
              <button
                type="button"
                onClick={() => setChartWindow("recent")}
                className={`rounded-md px-2.5 py-1 transition-colors whitespace-nowrap ${
                  chartWindow === "recent" ? "bg-white text-[#0A1E3D] shadow-sm" : "text-slate-600"
                }`}
              >
                Recent + Forecast
              </button>
            </div>
          </div>
        </div>

        <div className="h-[340px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={timelineChartData} margin={{ top: 10, right: 20, left: 6, bottom: 4 }}>
              <TriadSvgDefs idPrefix="fc-main" />
              <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
              <XAxis
                dataKey="month_label"
                tick={{ fontSize: 10, fill: "#64748B" }}
                minTickGap={26}
              />
              <YAxis
                tick={{ fontSize: 10, fill: "#64748B" }}
                width={62}
                tickFormatter={(v) => {
                  const num = Number(v);
                  if (selectedMetricMeta.isPct) return `${num.toFixed(0)}%`;
                  if (!selectedMetricMeta.isCurrency) return formatCompactNumber(num);
                  return formatNGN(num, true);
                }}
              />
              <Tooltip
                contentStyle={TRIAD_TOOLTIP_STYLE}
                labelStyle={TRIAD_TOOLTIP_LABEL_STYLE}
                itemStyle={TRIAD_TOOLTIP_ITEM_STYLE}
                formatter={(val: any, name: any) => {
                  if (Array.isArray(val)) {
                    return [`${formatNGN(Number(val[0]), true)} to ${formatNGN(Number(val[1]), true)}`, "95% Prediction Interval"];
                  }
                  const num = Number(val);
                  if (selectedMetricMeta.isPct) return [formatPct(num), name];
                  if (!selectedMetricMeta.isCurrency) return [num.toLocaleString(), name];
                  return [formatNGN(num), name];
                }}
              />
              <Legend wrapperStyle={{ fontSize: "11px" }} />
              <ReferenceLine y={0} stroke="#94A3B8" strokeDasharray="2 2" />
              {forecastStartLabel && (
                <ReferenceLine
                  x={forecastStartLabel}
                  stroke="#1D63B8"
                  strokeDasharray="4 4"
                  label={{
                    value: "Forecast Start",
                    position: "top",
                    fill: "#0A2E65",
                    fontSize: 10,
                  }}
                />
              )}
              {selectedMetric === "operating_profit_ngn" && (
                <Area
                  type="monotone"
                  dataKey="ciRange"
                  name="95% Prediction Interval"
                  stroke="none"
                  fill="url(#fc-main-band-light)"
                  fillOpacity={1}
                  connectNulls={false}
                />
              )}
              <Line
                type="monotone"
                dataKey="historicalVal"
                name="Historical Actual"
                stroke="url(#fc-main-stroke-navy-cobalt)"
                strokeWidth={2.4}
                dot={false}
                connectNulls={false}
              />
              <Line
                type="monotone"
                dataKey="forecastVal"
                name="2030 Model Baseline"
                stroke="url(#fc-main-stroke-cobalt-teal)"
                strokeWidth={2.4}
                strokeDasharray="6 3"
                dot={false}
                connectNulls={false}
              />
              {showSimOverlay && supportsSimOverlay && (
                <Line
                  type="monotone"
                  dataKey="simulatedVal"
                  name={`Simulated (${activePresetLabel})`}
                  stroke={simCumDelta < 0 ? "#E11D48" : "#0D9488"}
                  strokeWidth={2.6}
                  dot={false}
                  connectNulls={false}
                />
              )}
              {showEnvelope && selectedMetric === "operating_profit_ngn" && (
                <>
                  <Line
                    type="monotone"
                    dataKey="optimisticVal"
                    name="Optimistic Benchmark"
                    stroke="#0D9488"
                    strokeWidth={1.5}
                    strokeDasharray="3 3"
                    dot={false}
                    connectNulls={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="stressVal"
                    name="Stress Benchmark"
                    stroke="#E11D48"
                    strokeWidth={1.5}
                    strokeDasharray="3 3"
                    dot={false}
                    connectNulls={false}
                  />
                </>
              )}
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        {/* Annual / Quarterly / Scenario Diff / Risk Summary Tabs */}
        <div className="pt-2 space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex flex-wrap items-center gap-1 rounded-lg bg-slate-100 p-1 text-xs font-medium">
              <button
                type="button"
                onClick={() => setSummaryTab("annual")}
                className={`rounded-md px-3 py-1 transition-colors whitespace-nowrap ${
                  summaryTab === "annual" ? "bg-white text-[#0A1E3D] shadow-sm" : "text-slate-600"
                }`}
              >
                Annual Totals &amp; YoY (2016–2030)
              </button>
              <button
                type="button"
                onClick={() => setSummaryTab("scenario_diff")}
                className={`rounded-md px-3 py-1 transition-colors whitespace-nowrap ${
                  summaryTab === "scenario_diff" ? "bg-white text-[#0A1E3D] shadow-sm" : "text-slate-600"
                }`}
              >
                Scenario Year Diff (2026–2030)
              </button>
              <button
                type="button"
                onClick={() => setSummaryTab("quarterly")}
                className={`rounded-md px-3 py-1 transition-colors whitespace-nowrap ${
                  summaryTab === "quarterly" ? "bg-white text-[#0A1E3D] shadow-sm" : "text-slate-600"
                }`}
              >
                Quarterly Forecast (Q4 2026–Q4 2030)
              </button>
              <button
                type="button"
                onClick={() => setSummaryTab("risk")}
                className={`rounded-md px-3 py-1 transition-colors whitespace-nowrap ${
                  summaryTab === "risk" ? "bg-white text-[#0A1E3D] shadow-sm" : "text-slate-600"
                }`}
              >
                Negative &amp; Risk Periods ({riskMonthsList.length})
              </button>
            </div>
            <span className="text-xs text-slate-500">
              Secondary KPIs modeled independently via validated Exponential Smoothing / Holt-Winters
            </span>
          </div>

          {summaryTab === "annual" && data.annual_summaries && (
            <div className="overflow-x-auto max-h-[300px]">
              <table className="w-full text-left text-xs">
                <thead className="sticky top-0 bg-white">
                  <tr className="border-b border-slate-200 text-slate-500">
                    <th className="py-2 pr-4 font-semibold">Year</th>
                    <th className="py-2 pr-4 font-semibold">Scope</th>
                    <th className="py-2 pr-4 text-right font-semibold">Operating Profit</th>
                    <th className="py-2 pr-4 text-right font-semibold">Sales Revenue</th>
                    <th className="py-2 pr-4 text-right font-semibold">Gross Profit</th>
                    <th className="py-2 pr-4 text-right font-semibold">Total OpEx</th>
                    <th className="py-2 pr-4 text-right font-semibold">Op Margin</th>
                    <th className="py-2 text-right font-semibold">YoY Profit Δ</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {data.annual_summaries.map((yr) => (
                    <tr
                      key={yr.year}
                      className={yr.period_scope === "Forecast" ? "bg-teal-50/35" : ""}
                    >
                      <td className="py-2 pr-4 font-mono font-bold text-[#0A1E3D] tabular-nums">
                        {yr.year}
                      </td>
                      <td className="py-2 pr-4 text-slate-600">
                        {yr.period_scope} ({yr.months_count}m)
                      </td>
                      <td className="py-2 pr-4 text-right font-mono font-semibold text-[#0A1E3D] tabular-nums">
                        {formatNGN(yr.total_operating_profit_ngn, true)}
                      </td>
                      <td className="py-2 pr-4 text-right font-mono tabular-nums">
                        {formatNGN(yr.total_sales_revenue_ngn, true)}
                      </td>
                      <td className="py-2 pr-4 text-right font-mono tabular-nums">
                        {formatNGN(yr.total_gross_profit_ngn, true)}
                      </td>
                      <td className="py-2 pr-4 text-right font-mono tabular-nums">
                        {formatNGN(yr.total_operating_expenses_ngn, true)}
                      </td>
                      <td className="py-2 pr-4 text-right font-mono tabular-nums">
                        {formatPct(yr.weighted_operating_margin_pct)}
                      </td>
                      <td
                        className={`py-2 text-right font-mono font-semibold tabular-nums ${
                          (yr.yoy_operating_profit_pct ?? 0) >= 0 ? "text-teal-700" : "text-amber-700"
                        }`}
                      >
                        {yr.yoy_operating_profit_pct !== null
                          ? formatPct(yr.yoy_operating_profit_pct, 1, true)
                          : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {summaryTab === "scenario_diff" && scenariosBundle?.annual_comparison && (
            <div className="overflow-x-auto max-h-[300px]">
              <table className="w-full text-left text-xs">
                <thead className="sticky top-0 bg-white">
                  <tr className="border-b border-slate-200 text-slate-500">
                    <th className="py-2 pr-3 font-semibold">Forecast Year</th>
                    <th className="py-2 pr-3 text-right font-semibold">Base Profit</th>
                    <th className="py-2 pr-3 text-right font-semibold">Simulated Profit</th>
                    <th className="py-2 pr-3 text-right font-semibold">Scenario Delta</th>
                    <th className="py-2 pr-3 text-right font-semibold">Simulated Margin</th>
                    <th className="py-2 pr-3 text-right font-semibold">Stress Floor</th>
                    <th className="py-2 text-right font-semibold">Sim Loss / Risk Mos</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {scenariosBundle.annual_comparison.map((row) => (
                    <tr key={row.year} className="hover:bg-slate-50/70">
                      <td className="py-2 pr-3 font-mono font-bold text-[#0A1E3D] tabular-nums">
                        {row.year} ({row.months}m)
                      </td>
                      <td className="py-2 pr-3 text-right font-mono tabular-nums">
                        {formatNGN(row.base_profit_ngn, true)}
                      </td>
                      <td className="py-2 pr-3 text-right font-mono font-semibold text-[#0A1E3D] tabular-nums">
                        {formatNGN(row.custom_profit_ngn, true)}
                      </td>
                      <td
                        className={`py-2 pr-3 text-right font-mono font-bold tabular-nums ${
                          row.delta_profit_ngn >= 0 ? "text-teal-700" : "text-red-600"
                        }`}
                      >
                        {row.delta_profit_ngn >= 0 ? "+" : ""}
                        {formatNGN(row.delta_profit_ngn, true)}
                      </td>
                      <td className="py-2 pr-3 text-right font-mono tabular-nums">
                        {formatPct(row.custom_margin_pct)}{" "}
                        <span className="text-slate-400">
                          (vs {formatPct(row.base_margin_pct)})
                        </span>
                      </td>
                      <td className="py-2 pr-3 text-right font-mono text-red-700 tabular-nums">
                        {formatNGN(row.stress_profit_ngn, true)}
                      </td>
                      <td className="py-2 text-right font-mono tabular-nums">
                        <span
                          className={
                            row.custom_negative_months > 0 ? "text-red-600 font-bold" : "text-teal-700"
                          }
                        >
                          {row.custom_negative_months}
                        </span>{" "}
                        / {row.custom_risk_months}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {summaryTab === "quarterly" && data.quarterly_forecasts && (
            <div className="overflow-x-auto max-h-[300px]">
              <table className="w-full text-left text-xs">
                <thead className="sticky top-0 bg-white">
                  <tr className="border-b border-slate-200 text-slate-500">
                    <th className="py-2 pr-4 font-semibold">Quarter</th>
                    <th className="py-2 pr-4 text-right font-semibold">Months</th>
                    <th className="py-2 pr-4 text-right font-semibold">Projected Profit</th>
                    <th className="py-2 pr-4 text-right font-semibold">Projected Revenue</th>
                    <th className="py-2 pr-4 text-right font-semibold">Op Margin</th>
                    <th className="py-2 text-right font-semibold">95% Profit Range (Sum)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {data.quarterly_forecasts.map((q) => (
                    <tr key={q.year_quarter}>
                      <td className="py-2 pr-4 font-mono font-semibold text-[#0A1E3D]">
                        {q.year_quarter}
                      </td>
                      <td className="py-2 pr-4 text-right font-mono tabular-nums">{q.months}</td>
                      <td className="py-2 pr-4 text-right font-mono font-semibold text-teal-800 tabular-nums">
                        {formatNGN(q.total_operating_profit_ngn, true)}
                      </td>
                      <td className="py-2 pr-4 text-right font-mono tabular-nums">
                        {formatNGN(q.total_sales_revenue_ngn, true)}
                      </td>
                      <td className="py-2 pr-4 text-right font-mono tabular-nums">
                        {formatPct(q.avg_operating_margin_pct)}
                      </td>
                      <td className="py-2 text-right font-mono text-slate-600 tabular-nums">
                        {formatNGN(q.lower_95_sum_ngn, true)} to {formatNGN(q.upper_95_sum_ngn, true)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {summaryTab === "risk" && (
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 text-xs text-slate-700">
              {riskMonthsList.length === 0 ? (
                <div className="flex items-center gap-2 text-teal-800 font-medium">
                  <CheckCircle2 className="h-4 w-4 shrink-0" />
                  <span>
                    Zero negative-profit or low-profit risk periods (&lt;50% of trailing 12M profit or negative
                    95% lower bound) were flagged in the Base Case forecast through December 2030. Under Stress Case
                    assumptions or Custom Scenario shocks, inspect the Simulator below.
                  </span>
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead>
                      <tr className="border-b border-slate-200 text-slate-500">
                        <th className="py-1.5 pr-4 font-semibold">Month</th>
                        <th className="py-1.5 pr-4 text-right font-semibold">Baseline Profit</th>
                        <th className="py-1.5 pr-4 text-right font-semibold">Lower 95% Bound</th>
                        <th className="py-1.5 text-right font-semibold">Flag</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-200">
                      {riskMonthsList.map((rm) => (
                        <tr key={rm.month}>
                          <td className="py-1.5 pr-4 font-medium text-[#0A1E3D]">{rm.month_label}</td>
                          <td className="py-1.5 pr-4 text-right font-mono tabular-nums">
                            {formatNGN(rm.operating_profit_ngn, true)}
                          </td>
                          <td className="py-1.5 pr-4 text-right font-mono tabular-nums">
                            {formatNGN(rm.lower_95_ngn, true)}
                          </td>
                          <td className="py-1.5 text-right font-semibold text-amber-700">
                            {rm.is_negative_profit ? "Projected Loss" : "Downside Interval Risk"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </div>
      </section>

      {/* Flagship Section C: Live Interactive 2030 Scenario Simulator (Synchronized Across All Pages) */}
      {scenariosBundle && (
        <section
          id="scenario-simulator-2030"
          className="rounded-xl border border-slate-200 bg-white p-5 space-y-5"
        >
          <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between border-b border-slate-100 pb-4">
            <div className="flex items-start gap-2.5">
              <SlidersHorizontal className="h-4 w-4 text-[#0A2E65] mt-0.5 shrink-0" />
              <div>
                <div className="flex flex-wrap items-center gap-2">
                  <h3 className="font-display text-sm font-bold text-[#0A1E3D]">
                    Interactive 2030 Scenario Simulator &amp; Lever Attribution
                  </h3>
                  <span className="text-[11px] font-semibold text-teal-700">
                    Active: {activePresetLabel}
                    {simLoading ? " (Updating...)" : ""}
                  </span>
                </div>
                <p className="text-xs text-slate-500">
                  Every slider and strategic preset updates the 2016–2030 trajectory above, decomposes cumulative profit drivers via exact accounting waterfall, and synchronizes across all application pages.
                </p>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-1.5">
              {SCENARIO_PRESETS.map((p) => (
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
                onClick={() => setDrawerOpen(true)}
                className="rounded-lg border border-teal-700/30 bg-teal-50 px-2.5 py-1.5 text-xs font-semibold text-teal-900 hover:bg-teal-100 whitespace-nowrap"
              >
                Open Quick Dock
              </button>
              <button
                type="button"
                onClick={resetScenario}
                className="rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-xs font-semibold text-slate-700 hover:bg-slate-50 whitespace-nowrap"
              >
                Reset
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-5 xl:grid-cols-12">
            {/* Live Interactive Sliders + Numeric Steppers */}
            <div className="xl:col-span-4 space-y-3.5 rounded-xl border border-slate-200 bg-slate-50 p-4">
              <div className="flex items-center justify-between">
                <div className="text-xs font-bold text-[#0A1E3D]">
                  2026–2030 Multi-Year Operational Levers
                </div>
                <span className="text-[10px] font-semibold text-teal-700">Live Auto-Sync</span>
              </div>

              <div>
                <div className="flex items-center justify-between text-xs gap-2">
                  <label htmlFor="rev-growth-slider" className="font-medium text-slate-700">
                    Annual Revenue Growth Adj
                  </label>
                  <span className="font-mono font-semibold tabular-nums text-[#0A1E3D]">
                    {customAssumptions.revenue_growth_adj_pp >= 0 ? "+" : ""}
                    {customAssumptions.revenue_growth_adj_pp.toFixed(1)} pp/yr
                  </span>
                </div>
                <input
                  id="rev-growth-slider"
                  type="range"
                  min={-10}
                  max={12}
                  step={0.5}
                  value={customAssumptions.revenue_growth_adj_pp}
                  onChange={(e) =>
                    setCustomAssumptions({
                      ...customAssumptions,
                      revenue_growth_adj_pp: Number(e.target.value),
                    })
                  }
                  style={getSliderFillStyle(customAssumptions.revenue_growth_adj_pp, -10, 12)}
                  className="triad-slider mt-1 w-full"
                />
              </div>

              <div>
                <div className="flex items-center justify-between text-xs gap-2">
                  <label htmlFor="opex-growth-slider" className="font-medium text-slate-700">
                    Annual OpEx Growth Adj
                  </label>
                  <span className="font-mono font-semibold tabular-nums text-[#0A1E3D]">
                    {customAssumptions.opex_growth_adj_pp >= 0 ? "+" : ""}
                    {customAssumptions.opex_growth_adj_pp.toFixed(1)} pp/yr
                  </span>
                </div>
                <input
                  id="opex-growth-slider"
                  type="range"
                  min={-8}
                  max={14}
                  step={0.5}
                  value={customAssumptions.opex_growth_adj_pp}
                  onChange={(e) =>
                    setCustomAssumptions({
                      ...customAssumptions,
                      opex_growth_adj_pp: Number(e.target.value),
                    })
                  }
                  style={getSliderFillStyle(customAssumptions.opex_growth_adj_pp, -8, 14)}
                  className="triad-slider mt-1 w-full"
                />
              </div>

              <div>
                <div className="flex items-center justify-between text-xs gap-2">
                  <label htmlFor="margin-eff-slider" className="font-medium text-slate-700">
                    Margin Efficiency Shift
                  </label>
                  <span className="font-mono font-semibold tabular-nums text-[#0A1E3D]">
                    {customAssumptions.margin_efficiency_adj_pp >= 0 ? "+" : ""}
                    {customAssumptions.margin_efficiency_adj_pp.toFixed(1)} pp
                  </span>
                </div>
                <input
                  id="margin-eff-slider"
                  type="range"
                  min={-6}
                  max={6}
                  step={0.5}
                  value={customAssumptions.margin_efficiency_adj_pp}
                  onChange={(e) =>
                    setCustomAssumptions({
                      ...customAssumptions,
                      margin_efficiency_adj_pp: Number(e.target.value),
                    })
                  }
                  style={getSliderFillStyle(customAssumptions.margin_efficiency_adj_pp, -6, 6)}
                  className="triad-slider mt-1 w-full"
                />
              </div>

              <div>
                <div className="flex items-center justify-between text-xs gap-2">
                  <label htmlFor="fx-cost-slider" className="font-medium text-slate-700">
                    FX / Import Cost Shock
                  </label>
                  <span className="font-mono font-semibold tabular-nums text-[#0A1E3D]">
                    {customAssumptions.fx_cost_pressure_adj_pct >= 0 ? "+" : ""}
                    {customAssumptions.fx_cost_pressure_adj_pct.toFixed(1)}%
                  </span>
                </div>
                <input
                  id="fx-cost-slider"
                  type="range"
                  min={-10}
                  max={25}
                  step={1}
                  value={customAssumptions.fx_cost_pressure_adj_pct}
                  onChange={(e) =>
                    setCustomAssumptions({
                      ...customAssumptions,
                      fx_cost_pressure_adj_pct: Number(e.target.value),
                    })
                  }
                  style={getSliderFillStyle(customAssumptions.fx_cost_pressure_adj_pct, -10, 25)}
                  className="triad-slider mt-1 w-full"
                />
              </div>

              <div>
                <div className="flex items-center justify-between text-xs gap-2">
                  <label htmlFor="vol-prod-slider" className="font-medium text-slate-700">
                    Volume &amp; Productivity Shift
                  </label>
                  <span className="font-mono font-semibold tabular-nums text-[#0A1E3D]">
                    {customAssumptions.volume_productivity_adj_pct >= 0 ? "+" : ""}
                    {customAssumptions.volume_productivity_adj_pct.toFixed(1)}%
                  </span>
                </div>
                <input
                  id="vol-prod-slider"
                  type="range"
                  min={-15}
                  max={20}
                  step={1}
                  value={customAssumptions.volume_productivity_adj_pct}
                  onChange={(e) =>
                    setCustomAssumptions({
                      ...customAssumptions,
                      volume_productivity_adj_pct: Number(e.target.value),
                    })
                  }
                  style={getSliderFillStyle(customAssumptions.volume_productivity_adj_pct, -15, 20)}
                  className="triad-slider mt-1 w-full"
                />
              </div>

              {activeCustomVerdict && (
                <div className="rounded-lg border border-slate-200 bg-white p-3 space-y-1 text-xs">
                  <div className="text-[11px] font-semibold text-slate-500">
                    Simulated Scenario Verdict
                  </div>
                  <div className="font-display font-bold text-[#0A1E3D]">
                    {activeCustomVerdict.category}
                  </div>
                  <div className="font-mono text-[11px] tabular-nums text-slate-600">
                    Dec 2030 Δ:{" "}
                    <strong
                      className={
                        activeCustomVerdict.delta_vs_base_dec_2030_ngn >= 0
                          ? "text-teal-700"
                          : "text-red-600"
                      }
                    >
                      {activeCustomVerdict.delta_vs_base_dec_2030_ngn >= 0 ? "+" : ""}
                      {formatNGN(activeCustomVerdict.delta_vs_base_dec_2030_ngn, true)}
                    </strong>
                  </div>
                </div>
              )}
            </div>

            {/* Scenario Comparison Chart + Lever Attribution Waterfall */}
            <div className="xl:col-span-8 space-y-5">
              <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                <div className="space-y-1.5">
                  <div className="text-xs font-bold text-[#0A1E3D]">
                    2026–2030 Monthly Scenario Trajectories (Operating Profit)
                  </div>
                  <div className="h-[225px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={scenariosBundle.monthly_comparison}>
                        <TriadSvgDefs idPrefix="fc-scen" />
                        <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                        <XAxis dataKey="month_label" tick={{ fontSize: 10, fill: "#64748B" }} minTickGap={24} />
                        <YAxis
                          tick={{ fontSize: 10, fill: "#64748B" }}
                          tickFormatter={(v) => `₦${(Number(v) / 1e6).toFixed(0)}M`}
                          width={54}
                        />
                        <Tooltip
                          contentStyle={TRIAD_TOOLTIP_STYLE}
                          labelStyle={TRIAD_TOOLTIP_LABEL_STYLE}
                          itemStyle={TRIAD_TOOLTIP_ITEM_STYLE}
                          formatter={(val: any) => formatNGN(Number(val))}
                        />
                        <Legend wrapperStyle={{ fontSize: "10.5px" }} />
                        <ReferenceLine y={0} stroke="#94A3B8" strokeDasharray="3 3" />
                        <Line type="monotone" dataKey="optimistic" name="Optimistic" stroke="url(#fc-scen-stroke-cobalt-teal)" strokeWidth={2} dot={false} />
                        <Line type="monotone" dataKey="base" name="Base (Model)" stroke="url(#fc-scen-stroke-navy-cobalt)" strokeWidth={2.2} dot={false} />
                        <Line type="monotone" dataKey="custom" name="Simulated" stroke="url(#fc-scen-stroke-full)" strokeWidth={2.6} dot={false} />
                        <Line type="monotone" dataKey="stress" name="Stress Shock" stroke="#E11D48" strokeWidth={1.8} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="space-y-1.5">
                  <div className="text-xs font-bold text-[#0A1E3D]">
                    2026–2030 Cumulative Profit Lever Bridge (Base → Simulated)
                  </div>
                  <div className="h-[225px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={leverWaterfallData} margin={{ top: 8, right: 12, left: 0, bottom: 10 }}>
                        <TriadSvgDefs idPrefix="fc-wf" />
                        <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                        <XAxis dataKey="step" tick={{ fontSize: 9, fill: "#334155" }} interval={0} />
                        <YAxis
                          tick={{ fontSize: 10, fill: "#64748B" }}
                          tickFormatter={(v) => formatNGN(Number(v), true)}
                          width={56}
                        />
                        <Tooltip
                          contentStyle={TRIAD_TOOLTIP_STYLE}
                          labelStyle={TRIAD_TOOLTIP_LABEL_STYLE}
                          itemStyle={TRIAD_TOOLTIP_ITEM_STYLE}
                          formatter={(_val: any, _name: any, item: any) => [
                            formatNGN(Number(item?.payload?.value || 0)),
                            item?.payload?.step,
                          ]}
                        />
                        <ReferenceLine y={0} stroke="#64748B" />
                        <Bar dataKey="range" radius={[4, 4, 4, 4]}>
                          {leverWaterfallData.map((entry) => (
                            <Cell key={entry.step} fill={entry.color} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="border-b border-slate-200 text-slate-500">
                      <th className="py-2 pr-3 font-semibold">Scenario</th>
                      <th className="py-2 pr-3 text-right font-semibold">Dec 2030 Profit</th>
                      <th className="py-2 pr-3 text-right font-semibold">2030 Total Profit</th>
                      <th className="py-2 pr-3 text-right font-semibold">2030 Margin</th>
                      <th className="py-2 pr-3 text-right font-semibold">2026–2030 Cumulative</th>
                      <th className="py-2 pr-3 text-right font-semibold">Δ vs Base</th>
                      <th className="py-2 text-right font-semibold">Loss Mos</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {(["base", "optimistic", "stress", "custom"] as const).map((k) => {
                      const c = scenariosBundle.cases[k];
                      const diffCum =
                        c.summary.cumulative_profit_ngn -
                        scenariosBundle.cases.base.summary.cumulative_profit_ngn;
                      return (
                        <tr key={k} className={k === "custom" ? "bg-blue-50/50 font-medium" : ""}>
                          <td className="py-2 pr-3 font-semibold text-[#0A1E3D]">
                            {k === "custom" ? `Simulated (${activePresetLabel})` : c.label}
                          </td>
                          <td className="py-2 pr-3 text-right font-mono tabular-nums">
                            {formatNGN(c.summary.dec_2030_operating_profit_ngn, true)}
                          </td>
                          <td className="py-2 pr-3 text-right font-mono tabular-nums">
                            {formatNGN(c.summary.year_2030_total_profit_ngn, true)}
                          </td>
                          <td className="py-2 pr-3 text-right font-mono tabular-nums">
                            {formatPct(c.summary.year_2030_operating_margin_pct)}
                          </td>
                          <td className="py-2 pr-3 text-right font-mono font-semibold text-[#0A1E3D] tabular-nums">
                            {formatNGN(c.summary.cumulative_profit_ngn, true)}
                          </td>
                          <td
                            className={`py-2 pr-3 text-right font-mono font-semibold tabular-nums ${
                              k === "base"
                                ? "text-slate-400"
                                : diffCum >= 0
                                  ? "text-teal-700"
                                  : "text-red-600"
                            }`}
                          >
                            {k === "base"
                              ? "Baseline"
                              : `${diffCum >= 0 ? "+" : ""}${formatNGN(diffCum, true)}`}
                          </td>
                          <td
                            className={`py-2 text-right font-mono font-semibold tabular-nums ${
                              c.summary.negative_months > 0 ? "text-red-600" : "text-teal-700"
                            }`}
                          >
                            {c.summary.negative_months}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* Operational Indicators Trajectory: Improving vs Deteriorating */}
      {data.indicator_trends && (
        <section className="rounded-xl border border-slate-200 bg-white p-5 space-y-3">
          <div>
            <h3 className="font-display text-sm font-bold text-[#0A1E3D]">
              Operational Indicators Trajectory (Trailing 12M vs. Prior 12M Actuals)
            </h3>
            <p className="text-xs text-slate-500">
              Empirical comparison identifying which operational indicators are improving, stable, or deteriorating
              entering the 2026–2030 forecast horizon.
            </p>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500">
                  <th className="py-2 pr-4 font-semibold">Operational Indicator</th>
                  <th className="py-2 pr-4 text-right font-semibold">Prior 12M Avg</th>
                  <th className="py-2 pr-4 text-right font-semibold">Trailing 12M Avg</th>
                  <th className="py-2 pr-4 text-right font-semibold">YoY Change</th>
                  <th className="py-2 text-right font-semibold">Direction</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {data.indicator_trends.map((ind) => {
                  const isPct = ind.indicator.includes("%");
                  const isUnits = ind.indicator.includes("Unit Volume");
                  const fmt = (v: number) =>
                    isPct ? formatPct(v) : isUnits ? Math.round(v).toLocaleString() : formatNGN(v, true);
                  return (
                    <tr key={ind.indicator}>
                      <td className="py-2.5 pr-4 font-medium text-[#0A1E3D]">{ind.indicator}</td>
                      <td className="py-2.5 pr-4 text-right font-mono tabular-nums text-slate-600">
                        {fmt(ind.prior_12m_avg)}
                      </td>
                      <td className="py-2.5 pr-4 text-right font-mono font-semibold tabular-nums text-[#0A1E3D]">
                        {fmt(ind.trailing_12m_avg)}
                      </td>
                      <td className="py-2.5 pr-4 text-right font-mono tabular-nums">
                        {formatPct(ind.yoy_change_pct, 2, true)}
                      </td>
                      <td
                        className={`py-2.5 text-right font-semibold ${
                          ind.status === "Improving"
                            ? "text-teal-700"
                            : ind.status === "Deteriorating"
                              ? "text-amber-700"
                              : "text-slate-600"
                        }`}
                      >
                        {ind.status}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* Sections C, D & E: Model Leaderboard, Chronological Validation & Residual Diagnostics */}
      <section className="rounded-xl border border-slate-200 bg-white p-5 space-y-5">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between border-b border-slate-100 pb-3">
          <div>
            <h3 className="font-display text-sm font-bold text-[#0A1E3D]">
              Chronological Validation Leaderboard &amp; Rolling-Origin Backtesting
            </h3>
            <p className="text-xs text-slate-500">
              Untouched Holdout: {val.holdout_period} ({val.holdout_size} mos) · Rolling Backtest:{" "}
              {val.backtest_origins_count} expanding origins ({val.backtest_horizon_months}M horizon each)
            </p>
          </div>
          <div className="text-xs font-medium text-teal-800">
            Champion: {val.champion_name} (Selected via lowest composite out-of-sample RMSE)
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-200 text-slate-500">
                <th className="py-2 pr-3 font-semibold">Model Name</th>
                <th className="py-2 pr-3 font-semibold">Family</th>
                <th className="py-2 pr-3 text-right font-semibold">MAE</th>
                <th className="py-2 pr-3 text-right font-semibold">RMSE</th>
                <th className="py-2 pr-3 text-right font-semibold">MAPE</th>
                <th className="py-2 pr-3 text-right font-semibold">R²</th>
                <th className="py-2 pr-3 text-right font-semibold">Mean Error / Bias</th>
                <th className="py-2 pr-3 text-right font-semibold">Backtest RMSE</th>
                <th className="py-2 pr-3 text-right font-semibold">Stability</th>
                <th className="py-2 pr-3 text-right font-semibold">95% Cov</th>
                <th className="py-2 text-right font-semibold">Role</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {data.leaderboard?.map((m) => (
                <tr key={m.key} className={m.status === "Champion" ? "bg-teal-50/50" : ""}>
                  <td className="py-2.5 pr-3 font-semibold text-[#0A1E3D] whitespace-nowrap">{m.name}</td>
                  <td className="py-2.5 pr-3 text-slate-500 whitespace-nowrap">{m.family}</td>
                  <td className="py-2.5 pr-3 text-right font-mono tabular-nums">
                    {formatNGN(m.mae, true)}
                  </td>
                  <td className="py-2.5 pr-3 text-right font-mono font-semibold text-[#0A1E3D] tabular-nums">
                    {formatNGN(m.rmse, true)}
                  </td>
                  <td className="py-2.5 pr-3 text-right font-mono tabular-nums">
                    {m.mape_pct !== null ? `${m.mape_pct}%` : "—"}
                  </td>
                  <td className="py-2.5 pr-3 text-right font-mono tabular-nums">
                    {m.r2 !== null ? m.r2.toFixed(3) : "—"}
                  </td>
                  <td className="py-2.5 pr-3 text-right font-mono tabular-nums whitespace-nowrap">
                    {formatNGN(m.mean_error, true)} ({m.forecast_bias})
                  </td>
                  <td className="py-2.5 pr-3 text-right font-mono tabular-nums">
                    {formatNGN(m.backtest_mean_rmse, true)}
                  </td>
                  <td className="py-2.5 pr-3 text-right font-mono tabular-nums">
                    {m.stability_score_pct.toFixed(1)}%
                  </td>
                  <td className="py-2.5 pr-3 text-right font-mono tabular-nums">
                    {m.pi_coverage_95_pct.toFixed(1)}%
                  </td>
                  <td
                    className={`py-2.5 text-right font-semibold whitespace-nowrap ${
                      m.status === "Champion" ? "text-teal-700" : "text-slate-400"
                    }`}
                  >
                    {m.status}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Holdout Actual vs Predicted + Residual Distribution & ACF */}
        <div className="grid grid-cols-1 gap-5 lg:grid-cols-3 pt-2">
          <div className="space-y-2">
            <div className="text-xs font-bold text-[#0A1E3D]">
              Holdout Actual vs. Champion &amp; Naive ({val.holdout_size}M)
            </div>
            <div className="h-[210px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={val.holdout_series}>
                  <TriadSvgDefs idPrefix="fc-val" />
                  <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                  <XAxis dataKey="month_label" tick={{ fontSize: 9, fill: "#64748B" }} minTickGap={16} />
                  <YAxis
                    tick={{ fontSize: 10, fill: "#64748B" }}
                    tickFormatter={(v) => `₦${(Number(v) / 1e6).toFixed(0)}M`}
                    width={52}
                  />
                  <Tooltip
                    contentStyle={TRIAD_TOOLTIP_STYLE}
                    labelStyle={TRIAD_TOOLTIP_LABEL_STYLE}
                    itemStyle={TRIAD_TOOLTIP_ITEM_STYLE}
                    formatter={(v: any) => formatNGN(Number(v))}
                  />
                  <Legend wrapperStyle={{ fontSize: "10px" }} />
                  <Line type="monotone" dataKey="actual" name="Actual" stroke="url(#fc-val-stroke-navy-cobalt)" strokeWidth={2.2} dot={{ r: 2, fill: "#0A2E65" }} />
                  <Line
                    type="monotone"
                    dataKey="predicted"
                    name="Champion"
                    stroke="url(#fc-val-stroke-cobalt-teal)"
                    strokeWidth={2.2}
                    strokeDasharray="4 3"
                    dot={{ r: 2, fill: "#0D9488" }}
                  />
                  <Line
                    type="monotone"
                    dataKey="naive"
                    name="Naive"
                    stroke="#94A3B8"
                    strokeWidth={1.5}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="space-y-2">
            <div className="text-xs font-bold text-[#0A1E3D]">Holdout Residual Distribution</div>
            <div className="h-[210px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={val.residual_histogram}>
                  <TriadSvgDefs idPrefix="fc-res" />
                  <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                  <XAxis dataKey="bin" tick={{ fontSize: 9, fill: "#64748B" }} />
                  <YAxis tick={{ fontSize: 10, fill: "#64748B" }} allowDecimals={false} width={30} />
                  <Tooltip
                    contentStyle={TRIAD_TOOLTIP_STYLE}
                    labelStyle={TRIAD_TOOLTIP_LABEL_STYLE}
                    itemStyle={TRIAD_TOOLTIP_ITEM_STYLE}
                  />
                  <Bar dataKey="count" name="Months" fill="url(#fc-res-bar-navy-cobalt)" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <div className="text-xs font-bold text-[#0A1E3D]">Residual Autocorrelation (ACF)</div>
              <span className="font-mono text-[11px] text-slate-500 tabular-nums">
                DW: {val.durbin_watson ?? "—"} · LB p: {val.ljung_box_p_value ?? "—"}
              </span>
            </div>
            <div className="h-[210px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={val.residual_acf}>
                  <TriadSvgDefs idPrefix="fc-acf" />
                  <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                  <XAxis dataKey="lag" tick={{ fontSize: 10, fill: "#64748B" }} tickFormatter={(l) => `Lag ${l}`} />
                  <YAxis domain={[-1, 1]} tick={{ fontSize: 10, fill: "#64748B" }} width={36} />
                  <Tooltip
                    contentStyle={TRIAD_TOOLTIP_STYLE}
                    labelStyle={TRIAD_TOOLTIP_LABEL_STYLE}
                    itemStyle={TRIAD_TOOLTIP_ITEM_STYLE}
                  />
                  <ReferenceLine y={0} stroke="#64748B" />
                  {val.residual_acf[0] && (
                    <>
                      <ReferenceLine y={val.residual_acf[0].bound} stroke="#D97706" strokeDasharray="3 3" />
                      <ReferenceLine y={-val.residual_acf[0].bound} stroke="#D97706" strokeDasharray="3 3" />
                    </>
                  )}
                  <Bar dataKey="acf" name="ACF" radius={[3, 3, 0, 0]}>
                    {val.residual_acf.map((item) => (
                      <Cell
                        key={item.lag}
                        fill={Math.abs(item.acf) > item.bound ? "#D97706" : "url(#fc-acf-bar-cobalt-teal)"}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-xs text-slate-700">
          <strong>Bias &amp; Error Diagnostics:</strong> {val.bias_explanation} Durbin-Watson statistic is{" "}
          <span className="font-mono font-semibold">{val.durbin_watson ?? "N/A"}</span> and Ljung-Box white-noise
          test p-value is <span className="font-mono font-semibold">{val.ljung_box_p_value ?? "N/A"}</span>.
        </div>
      </section>

      {/* Section F: Hypothesis Testing Laboratory */}
      <section className="rounded-xl border border-slate-200 bg-white p-5 space-y-4">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between border-b border-slate-100 pb-3">
          <div className="flex items-start gap-2.5">
            <FlaskConical className="h-4 w-4 text-[#0A2E65] mt-0.5 shrink-0" />
            <div>
              <h3 className="font-display text-sm font-bold text-[#0A1E3D]">
                Statistical Hypothesis Testing Laboratory
              </h3>
              <p className="text-xs text-slate-500">
                Formal hypothesis tests with automatic assumption checks and non-parametric fallbacks. Statistical
                significance does not imply causation.
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3 text-xs">
            <div className="flex items-center gap-1.5">
              <span className="font-medium text-slate-600">Significance α:</span>
              <div className="flex rounded-lg bg-slate-100 p-1">
                {[0.01, 0.05, 0.1].map((a) => (
                  <button
                    key={a}
                    type="button"
                    disabled={hypLoading}
                    onClick={() => void handleRunHypotheses(a, splitYear)}
                    className={`rounded-md px-2.5 py-1 font-mono font-semibold transition-colors whitespace-nowrap ${
                      alpha === a ? "bg-white text-[#0A1E3D] shadow-sm" : "text-slate-600"
                    }`}
                  >
                    {a.toFixed(2)}
                  </button>
                ))}
              </div>
            </div>

            <div className="flex items-center gap-1.5">
              <label htmlFor="split-year-select" className="font-medium text-slate-600">
                Regime Split:
              </label>
              <select
                id="split-year-select"
                value={splitYear}
                disabled={hypLoading}
                onChange={(e) => void handleRunHypotheses(alpha, Number(e.target.value))}
                className="rounded-lg border border-slate-300 bg-slate-50 px-2.5 py-1.5 font-mono text-xs font-semibold text-slate-800"
              >
                {availableSplitYears.map((yr) => (
                  <option key={yr} value={yr}>
                    Pre-{yr} vs {yr}+
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {hypBundle?.tests.map((test) => (
            <div key={test.id} className="rounded-xl border border-slate-200 bg-slate-50/50 p-4 space-y-2.5">
              <div className="flex items-start justify-between gap-2">
                <h4 className="font-display text-xs font-bold text-[#0A1E3D]">{test.title}</h4>
                <span
                  className={`text-xs font-bold whitespace-nowrap ${
                    test.decision === "Reject H0"
                      ? "text-teal-700"
                      : test.decision === "Fail to Reject H0"
                        ? "text-slate-600"
                        : "text-amber-700"
                  }`}
                >
                  {test.decision}
                </span>
              </div>

              <div className="text-[11px] text-slate-600 space-y-1">
                <div>
                  <strong>Null (H₀):</strong> {test.null_hypothesis}
                </div>
                <div>
                  <strong>Alternative (H₁):</strong> {test.alternative_hypothesis}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2 border-y border-slate-200 py-2 text-[11px]">
                <div>
                  <span className="text-slate-500">Test: </span>
                  <span className="font-semibold text-slate-800">{test.test_used}</span>
                </div>
                <div className="text-right font-mono tabular-nums">
                  <span className="text-slate-500">{test.statistic_name}: </span>
                  <span className="font-semibold text-[#0A1E3D]">{test.statistic ?? "—"}</span>
                  <span className="text-slate-400"> · </span>
                  <span className="text-slate-500">p = </span>
                  <span className="font-bold text-[#0A1E3D]">
                    {test.p_value !== null ? (test.p_value < 0.0001 ? "<0.0001" : test.p_value.toFixed(4)) : "—"}
                  </span>
                </div>
                <div className="col-span-2 text-slate-600 font-mono tabular-nums">
                  Effect / CI: {test.confidence_interval}
                </div>
              </div>

              <div className="text-[11px] text-slate-700 leading-relaxed">
                <strong>Business Interpretation:</strong> {test.business_interpretation}
              </div>
              <div className="text-[11px] text-slate-500 leading-relaxed">
                <strong>Assumption Check:</strong> {test.assumption_checks}
              </div>
            </div>
          ))}
        </div>

        {hypBundle?.methodology_note && (
          <div className="text-xs text-slate-500">{hypBundle.methodology_note}</div>
        )}
      </section>

      {/* Section G: Feature Dictionary & Driver Importance */}
      <section className="grid grid-cols-1 gap-6 xl:grid-cols-12">
        <div className="rounded-xl border border-slate-200 bg-white p-5 xl:col-span-5 space-y-3">
          <div>
            <h3 className="font-display text-sm font-bold text-[#0A1E3D]">
              Time-Series Feature Driver Importance
            </h3>
            <p className="text-xs text-slate-500">
              Engineered lag, rolling &amp; seasonal drivers in {data.driver_importance_model}. Reflects predictive
              signal, not guaranteed causal control.
            </p>
          </div>
          <div className="h-[290px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                layout="vertical"
                data={(data.driver_importance || []).slice(0, 10)}
                margin={{ top: 4, right: 20, left: 6, bottom: 4 }}
              >
                <TriadSvgDefs idPrefix="fc-drv" />
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
                  width={138}
                />
                <Tooltip
                  contentStyle={TRIAD_TOOLTIP_STYLE}
                  labelStyle={TRIAD_TOOLTIP_LABEL_STYLE}
                  itemStyle={TRIAD_TOOLTIP_ITEM_STYLE}
                  formatter={(val: any) => [`${Number(val).toFixed(2)}%`, "Importance Share"]}
                />
                <Bar dataKey="share_pct" name="Share %" fill="url(#fc-drv-bar-triad-h)" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-5 xl:col-span-7 space-y-3">
          <div className="flex items-center justify-between gap-2">
            <div>
              <h3 className="font-display text-sm font-bold text-[#0A1E3D]">
                Leak-Free Feature Engineering Dictionary ({data.feature_dictionary?.length || 0} Predictors)
              </h3>
              <p className="text-xs text-slate-500">
                Strict point-in-time lagged and deterministic calendar features computed strictly prior to period t.
              </p>
            </div>
            <button
              type="button"
              onClick={() => setShowAllFeatures((v) => !v)}
              className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-semibold text-slate-700 hover:bg-slate-50 whitespace-nowrap"
            >
              {showAllFeatures ? "Show Top 8" : "Show All 16"}
            </button>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500">
                  <th className="py-2 pr-3 font-semibold">Feature</th>
                  <th className="py-2 pr-3 font-semibold">Category</th>
                  <th className="py-2 pr-3 font-semibold">Formula</th>
                  <th className="py-2 font-semibold">Operational Relevance</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {(showAllFeatures
                  ? data.feature_dictionary || []
                  : (data.feature_dictionary || []).slice(0, 8)
                ).map((f) => (
                  <tr key={f.name}>
                    <td className="py-2 pr-3 font-mono font-semibold text-[#0A1E3D] whitespace-nowrap">
                      {f.name}
                    </td>
                    <td className="py-2 pr-3 text-slate-500 whitespace-nowrap">{f.category}</td>
                    <td className="py-2 pr-3 font-mono text-[11px] text-slate-700">{f.formula}</td>
                    <td className="py-2 text-slate-600">{f.business_relevance}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* Section I: 2030 Operational Outlook & Management Verdict (Base Model + Active Simulated Scenario) */}
      <section className={`rounded-xl border ${verdictStyle.border} ${verdictStyle.bg} p-6 space-y-5`}>
        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between border-b border-slate-200/80 pb-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-500">
              <TrendingUp className="h-3.5 w-3.5" />
              <span>2030 Operational Outlook &amp; Verdict</span>
            </div>
            <h3 className={`font-display text-lg font-bold ${verdictStyle.accent}`}>
              Base Model: {verdictState.category}
            </h3>
            <p className="text-xs text-slate-700 leading-relaxed max-w-4xl">{verdictState.headline}</p>
          </div>

          {isScenarioActive && activeCustomVerdict && (
            <div className="rounded-xl border border-blue-200 bg-white px-4 py-3 text-xs space-y-1 lg:max-w-md shrink-0">
              <div className="text-[11px] font-bold uppercase tracking-wider text-blue-700">
                Active Simulated Scenario ({activePresetLabel})
              </div>
              <div className="font-display text-sm font-bold text-[#0A1E3D]">
                {activeCustomVerdict.category}
              </div>
              <p className="text-slate-600 leading-relaxed">{activeCustomVerdict.headline}</p>
            </div>
          )}
        </div>

        {/* Transparent Threshold Checklist */}
        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <div className="mb-2 flex items-center justify-between gap-2">
            <span className="text-xs font-bold text-[#0A1E3D]">
              Threshold Evaluation Matrix ({isScenarioActive && activeCustomVerdict ? "Active Simulated Scenario vs. Governance Rules" : "Base Model Governance"})
            </span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500">
                  <th className="py-1.5 pr-4 font-semibold">Criterion</th>
                  <th className="py-1.5 pr-4 font-semibold">Required Threshold</th>
                  <th className="py-1.5 pr-4 text-right font-semibold">Observed Result</th>
                  <th className="py-1.5 text-right font-semibold">Evaluation</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {(isScenarioActive && activeCustomVerdict
                  ? activeCustomVerdict.thresholds_evaluated
                  : verdictState.thresholds_evaluated
                ).map((th) => (
                  <tr key={th.rule}>
                    <td className="py-2 pr-4 font-medium text-[#0A1E3D]">{th.rule}</td>
                    <td className="py-2 pr-4 text-slate-600">{th.threshold}</td>
                    <td className="py-2 pr-4 text-right font-mono font-semibold tabular-nums">
                      {th.observed}
                    </td>
                    <td
                      className={`py-2 text-right font-semibold ${
                        th.passed ? "text-teal-700" : "text-amber-700"
                      }`}
                    >
                      {th.passed ? "Passed" : "Monitor"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <div className="rounded-xl border border-slate-200 bg-white p-4 space-y-2.5 text-xs text-slate-700">
            <div className="font-bold text-[#0A1E3D]">Trajectory, Efficiency &amp; Cost Pressures</div>
            <p>
              <strong>Annual Trajectory:</strong> {verdictState.annual_trajectory_summary}
            </p>
            <p>
              <strong>Operational Efficiency:</strong> {verdictState.operational_efficiency}
            </p>
            <p>
              <strong>Major Cost Pressures:</strong> {verdictState.major_cost_pressures}
            </p>
            <p>
              <strong>Forecast Uncertainty:</strong> {verdictState.forecast_uncertainty_summary}
            </p>
            <p className="text-slate-500">{verdictState.verdict_rationale}</p>
          </div>

          <div className="rounded-xl border border-slate-200 bg-white p-4 space-y-3 text-xs text-slate-700">
            <div>
              <div className="flex items-center gap-1.5 font-bold text-amber-800">
                <AlertTriangle className="h-3.5 w-3.5" />
                <span>Principal Downside Risks</span>
              </div>
              <ul className="mt-1.5 list-disc pl-4 space-y-1">
                {verdictState.principal_risks.map((r) => (
                  <li key={r}>{r}</li>
                ))}
              </ul>
            </div>

            <div>
              <div className="flex items-center gap-1.5 font-bold text-teal-800">
                <ShieldCheck className="h-3.5 w-3.5" />
                <span>Recommended Management Interventions</span>
              </div>
              <ul className="mt-1.5 list-disc pl-4 space-y-1">
                {verdictState.recommended_interventions.map((rec) => (
                  <li key={rec}>{rec}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* Section 12: How the Forecasting System Works */}
      <section className="rounded-xl border border-slate-200 bg-white p-5 space-y-4">
        <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
          <BookOpen className="h-4 w-4 text-[#0A2E65]" />
          <h3 className="font-display text-sm font-bold text-[#0A1E3D]">
            How the 2030 Forecasting System Works (Methodology &amp; Governance)
          </h3>
        </div>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-3 text-xs text-slate-600 leading-relaxed">
          <div className="space-y-1">
            <div className="font-bold text-[#0A1E3D]">01. Data Preparation &amp; Accounting Audit</div>
            <p>
              Ingests historical monthly records, validates numeric types and chronological ordering, and recalculates
              COGS, Sales Revenue, Gross Profit, and Operating Profit directly from raw drivers before modeling.
            </p>
          </div>
          <div className="space-y-1">
            <div className="font-bold text-[#0A1E3D]">02. Leak-Free Feature Engineering</div>
            <p>
              Constructs 16 strictly backward-looking predictors (lags 1–12, rolling 3M/6M/12M averages, volatility,
              harmonics, and lagged margins). Same-period revenue/cost identities are excluded from recursive ML predictors.
            </p>
          </div>
          <div className="space-y-1">
            <div className="font-bold text-[#0A1E3D]">03. Multi-Model Suite &amp; Selection</div>
            <p>
              Evaluates 11 statistical baselines (Naive, Seasonal Naive, SES, Holt Damped, Holt-Winters, AR) and ML
              regressors (Linear, Ridge, Random Forest, Gradient Boosting, XGBoost) without pre-judging the winner.
            </p>
          </div>
          <div className="space-y-1">
            <div className="font-bold text-[#0A1E3D]">04. Chronological Validation &amp; Backtesting</div>
            <p>
              Scores every candidate on an untouched {val.holdout_size}-month chronological holdout plus{" "}
              {val.backtest_origins_count} expanding-window rolling origins to measure MAE, RMSE, MAPE, bias, and stability.
            </p>
          </div>
          <div className="space-y-1">
            <div className="font-bold text-[#0A1E3D]">05. Hypothesis Testing &amp; Confounding Checks</div>
            <p>
              Runs Mann-Kendall trend, ANOVA/Kruskal-Wallis, Spearman/Pearson with cost-ratio controls, Mann-Whitney
              period shifts, and Harvey-Leybourne-Newbold Diebold-Mariano tests against user-configurable α.
            </p>
          </div>
          <div className="space-y-1">
            <div className="font-bold text-[#0A1E3D]">06. Horizon-Scaled Uncertainty &amp; Verdict</div>
            <p>
              Calibrates 80%/95% prediction intervals from empirical holdout errors widening with forecast horizon, and
              separates ML baseline projections from user-defined Optimistic/Stress/Custom scenario assumptions.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};
