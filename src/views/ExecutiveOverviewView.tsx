import React, { useId, useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  ArrowDownRight,
  ArrowRight,
  ArrowUpRight,
  BarChart2,
  BrainCircuit,
  DollarSign,
  Percent,
  TrendingUp,
} from "lucide-react";
import {
  AnomalyReport,
  DatasetState,
  ExplainabilityResponse,
  ModelsSummaryResponse,
} from "../types/financial";
import { formatCompactNumber, formatFX, formatNGN, formatPct } from "../utils/format";
import { SectionId } from "../components/Sidebar";
import { useScenario } from "../context/ScenarioContext";
import {
  TRIAD_TOOLTIP_ITEM_STYLE,
  TRIAD_TOOLTIP_LABEL_STYLE,
  TRIAD_TOOLTIP_STYLE,
  TriadSvgDefs,
  getSliderFillStyle,
} from "../utils/chartTheme";

interface ExecutiveOverviewViewProps {
  dataset: DatasetState;
  models: ModelsSummaryResponse | null;
  explainability: ExplainabilityResponse | null;
  anomalies: AnomalyReport | null;
  onNavigate: (section: SectionId) => void;
  onSelectMonthForRag: (monthIso: string) => void;
}

function MiniSparkline({
  values,
  startColor = "#60A5FA",
  endColor = "#2DD4BF",
}: {
  values: number[];
  startColor?: string;
  endColor?: string;
}) {
  const uid = useId().replace(/:/g, "");
  if (!values.length) return null;
  const sliced = values.slice(-16);
  const min = Math.min(...sliced);
  const max = Math.max(...sliced);
  const span = max - min || 1;
  const coords = sliced.map((v, i, arr) => {
    const x = 2 + (i / Math.max(1, arr.length - 1)) * 70;
    const y = 26 - ((v - min) / span) * 20;
    return { x, y };
  });
  const pts = coords.map((c) => `${c.x.toFixed(1)},${c.y.toFixed(1)}`);
  const last = coords[coords.length - 1];
  const areaPts = [`${coords[0].x.toFixed(1)},28`, ...pts, `${last.x.toFixed(1)},28`];

  return (
    <svg width="76" height="30" viewBox="0 0 76 30" className="shrink-0" aria-hidden="true">
      <defs>
        <linearGradient id={`spk-stroke-${uid}`} x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor={startColor} />
          <stop offset="100%" stopColor={endColor} />
        </linearGradient>
        <linearGradient id={`spk-fill-${uid}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={endColor} stopOpacity={0.34} />
          <stop offset="100%" stopColor={startColor} stopOpacity={0.02} />
        </linearGradient>
      </defs>
      <polygon points={areaPts.join(" ")} fill={`url(#spk-fill-${uid})`} />
      <polyline
        fill="none"
        stroke={`url(#spk-stroke-${uid})`}
        strokeWidth="2.1"
        strokeLinecap="round"
        strokeLinejoin="round"
        points={pts.join(" ")}
      />
      <circle cx={last.x} cy={last.y} r="2.6" fill={endColor} stroke="#FFFFFF" strokeWidth="1" />
    </svg>
  );
}

export const ExecutiveOverviewView: React.FC<ExecutiveOverviewViewProps> = ({
  dataset,
  models,
  explainability,
  anomalies,
  onNavigate,
  onSelectMonthForRag,
}) => {
  const { kpis, records, quarterly_summary, factor_impacts } = dataset;
  const {
    activePresetLabel,
    isScenarioActive,
    driverDeltas,
    setDriverDeltas,
    forecastScenarios,
    singlePeriodResult,
    setDrawerOpen,
  } = useScenario();
  const [quickTab, setQuickTab] = useState<"forecast" | "whatif">("forecast");
  const [selectedQuarterWidget, setSelectedQuarterWidget] = useState<string>("Q4");

  const revenueSeries = useMemo(() => records.map((r) => r.Sales_Revenue_NGN), [records]);
  const opProfitSeries = useMemo(() => records.map((r) => r.Operating_Profit_NGN), [records]);
  const grossProfitSeries = useMemo(() => records.map((r) => r.Gross_Profit_NGN), [records]);
  const gmSeries = useMemo(() => records.map((r) => r.Gross_Margin_Pct), [records]);

  // Profitability donut breakdown across the 3-color relational triad
  const totalOpEx = (kpis.total_marketing_spend || 0) + (kpis.total_other_operating_cost || 0);
  const donutData = [
    { name: "Operating Profit", value: Math.max(0, kpis.total_operating_profit), color: "#0D9488", gradId: "url(#exec-donut-bar-cobalt-teal)" },
    { name: "Marketing & OpEx", value: Math.max(0, totalOpEx), color: "#1D63B8", gradId: "url(#exec-donut-bar-navy-cobalt)" },
    { name: "COGS (Import)", value: Math.max(0, kpis.total_import_costs), color: "#0A2E65", gradId: "#0A2E65" },
  ];

  // Quick Forecast widget summary from selected out-of-time model + quarter average
  const selectedModel = models?.driver_suite?.selected_model;
  const lastTestPred = selectedModel?.test_predictions?.[selectedModel.test_predictions.length - 1];
  const quarterStats = quarterly_summary.find((q) => q.quarter === selectedQuarterWidget);
  const baselineQuarterOp = quarterStats?.avg_operating_profit ?? (dataset.latest_record?.Operating_Profit_NGN || 0);
  const quickSimulatedOp =
    singlePeriodResult?.simulated.accounting.Accounting_Operating_Profit_NGN ?? baselineQuarterOp;
  const dec2030CustomOp = forecastScenarios?.cases.custom.summary.dec_2030_operating_profit_ngn;

  const topDriver = explainability?.global_importance?.[0];
  const secondDriver = explainability?.global_importance?.[1];
  const latestRec = dataset.latest_record;

  const renderDelta = (val: number | null | undefined, suffix = "%") => {
    if (val === null || val === undefined) return <span className="text-xs opacity-75">N/A</span>;
    const positive = val >= 0;
    return (
      <span className="inline-flex items-center gap-0.5 text-xs font-semibold tabular-nums">
        {positive ? <ArrowUpRight className="h-3.5 w-3.5" /> : <ArrowDownRight className="h-3.5 w-3.5" />}
        {positive ? "+" : ""}
        {val.toFixed(1)}
        {suffix}
      </span>
    );
  };

  return (
    <div className="space-y-5">
      {/* Row 1: 4 Hero KPI Cards progressing cleanly across the 3-color relational triad */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {/* Card 1: Sovereign Navy -> Cobalt - Total Sales Revenue */}
        <div
          className="rounded-xl bg-gradient-to-br from-[#072149] via-[#0A2E65] to-[#154A96] p-5 text-white shadow-sm transition-shadow hover:shadow-md"
          title={formatNGN(kpis.total_sales_revenue)}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs font-medium text-blue-100">
              <BarChart2 className="h-4 w-4 text-sky-300" />
              <span>Total Sales Revenue</span>
            </div>
          </div>
          <div className="mt-3 flex items-end justify-between gap-2">
            <div>
              <div className="font-display text-2xl font-bold tracking-tight tabular-nums">
                {formatNGN(kpis.total_sales_revenue, true)}
              </div>
              <div className="mt-1 flex items-center gap-1.5 whitespace-nowrap text-teal-200">
                {renderDelta(kpis.deltas.revenue_pct)}
                <span className="text-[11px] text-blue-200/80">vs. 1st half</span>
              </div>
            </div>
            <MiniSparkline values={revenueSeries} startColor="#60A5FA" endColor="#38BDF8" />
          </div>
        </div>

        {/* Card 2: Navy-to-Cobalt Bridge - Total Gross Profit */}
        <div
          className="rounded-xl bg-gradient-to-br from-[#0A2E65] via-[#144B96] to-[#1D63B8] p-5 text-white shadow-sm transition-shadow hover:shadow-md"
          title={formatNGN(kpis.total_gross_profit)}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs font-medium text-blue-100">
              <DollarSign className="h-4 w-4 text-sky-200" />
              <span>Total Gross Profit</span>
            </div>
          </div>
          <div className="mt-3 flex items-end justify-between gap-2">
            <div>
              <div className="font-display text-2xl font-bold tracking-tight tabular-nums">
                {formatNGN(kpis.total_gross_profit, true)}
              </div>
              <div className="mt-1 flex items-center gap-1.5 whitespace-nowrap text-teal-200">
                {renderDelta(kpis.deltas.gross_profit_pct)}
                <span className="text-[11px] text-blue-100/80">Net of COGS</span>
              </div>
            </div>
            <MiniSparkline values={grossProfitSeries} startColor="#93C5FD" endColor="#2DD4BF" />
          </div>
        </div>

        {/* Card 3: Cobalt-to-Teal Bridge - Total Operating Profit */}
        <div
          className="rounded-xl bg-gradient-to-br from-[#124B8E] via-[#136B8E] to-[#0D9488] p-5 text-white shadow-sm transition-shadow hover:shadow-md"
          title={formatNGN(kpis.total_operating_profit)}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs font-medium text-teal-50">
              <TrendingUp className="h-4 w-4 text-teal-200" />
              <span>Total Operating Profit</span>
            </div>
          </div>
          <div className="mt-3 flex items-end justify-between gap-2">
            <div>
              <div className="font-display text-2xl font-bold tracking-tight tabular-nums">
                {formatNGN(kpis.total_operating_profit, true)}
              </div>
              <div className="mt-1 flex items-center gap-1.5 whitespace-nowrap text-teal-100">
                {renderDelta(kpis.deltas.operating_profit_pct)}
                <span className="text-[11px] text-teal-100/80">Target EBIT</span>
              </div>
            </div>
            <MiniSparkline values={opProfitSeries} startColor="#38BDF8" endColor="#5EEAD4" />
          </div>
        </div>

        {/* Card 4: Deep Emerald-Teal Anchor - Gross Margin % */}
        <div className="rounded-xl bg-gradient-to-br from-[#0A5A60] via-[#0D7D77] to-[#14B8A6] p-5 text-white shadow-sm transition-shadow hover:shadow-md">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs font-medium text-teal-50">
              <Percent className="h-4 w-4 text-white" />
              <span>Gross Margin (%)</span>
            </div>
          </div>
          <div className="mt-3 flex items-end justify-between gap-2">
            <div>
              <div className="font-display text-2xl font-bold tracking-tight tabular-nums">
                {formatPct(kpis.gross_margin_pct, 1)}
              </div>
              <div className="mt-1 flex items-center gap-1.5 whitespace-nowrap text-white">
                {renderDelta(kpis.deltas.gross_margin_pp, " pp")}
                <span className="text-[11px] text-teal-50/90">Weighted</span>
              </div>
            </div>
            <MiniSparkline values={gmSeries} startColor="#99F6E4" endColor="#FFFFFF" />
          </div>
        </div>
      </div>

      {/* Row 1B: Secondary 4-Metric Tabular KPI Strip (completes all 8 required KPIs cleanly) */}
      <div className="grid grid-cols-2 gap-4 rounded-xl border border-slate-200 bg-white px-5 py-3.5 lg:grid-cols-4">
        <div title={kpis.total_units_sold.toLocaleString()}>
          <div className="text-xs font-medium text-slate-500">Total Units Sold</div>
          <div className="mt-1 flex items-baseline gap-2">
            <span className="font-display text-lg font-bold text-[#0A1E3D] tabular-nums">
              {formatCompactNumber(kpis.total_units_sold)}
            </span>
            <span className="text-xs font-medium text-emerald-700 tabular-nums">
              {kpis.deltas.units_pct !== undefined && kpis.deltas.units_pct !== null
                ? `${kpis.deltas.units_pct >= 0 ? "+" : ""}${kpis.deltas.units_pct}%`
                : ""}
            </span>
          </div>
        </div>

        <div>
          <div className="text-xs font-medium text-slate-500">Avg Operating Margin</div>
          <div className="mt-1 flex items-baseline gap-2">
            <span className="font-display text-lg font-bold text-[#0A1E3D] tabular-nums">
              {formatPct(kpis.avg_operating_margin_pct, 2)}
            </span>
            <span className="text-xs text-slate-500 tabular-nums">
              ({formatPct(kpis.weighted_operating_margin_pct, 1)} weighted)
            </span>
          </div>
        </div>

        <div title={formatNGN(kpis.total_marketing_spend)}>
          <div className="text-xs font-medium text-slate-500">Total Marketing Expenditure</div>
          <div className="mt-1 flex items-baseline gap-2">
            <span className="font-display text-lg font-bold text-[#0A1E3D] tabular-nums">
              {formatNGN(kpis.total_marketing_spend, true)}
            </span>
            <span className="text-xs text-slate-500 tabular-nums">
              {kpis.total_sales_revenue > 0
                ? `${((kpis.total_marketing_spend / kpis.total_sales_revenue) * 100).toFixed(1)}% of rev`
                : ""}
            </span>
          </div>
        </div>

        <div title={formatNGN(kpis.total_import_costs)}>
          <div className="text-xs font-medium text-slate-500">Total Import Costs (COGS)</div>
          <div className="mt-1 flex items-baseline gap-2">
            <span className="font-display text-lg font-bold text-[#0A1E3D] tabular-nums">
              {formatNGN(kpis.total_import_costs, true)}
            </span>
            <span className="text-xs text-slate-500 tabular-nums">
              Avg FX {formatFX(kpis.avg_usd_ngn_rate)}
            </span>
          </div>
        </div>
      </div>

      {/* Row 2: Sales & Profit Trend + Profitability Overview Donut + Key Metrics + Prediction Right Rail */}
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-12">
        {/* Sales & Profit Trend (5 cols) */}
        <div className="rounded-xl border border-slate-200 bg-white p-5 xl:col-span-5">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="font-display text-sm font-bold text-[#0A1E3D]">
                Sales, Gross &amp; Operating Profit Trend
              </h2>
              <p className="text-xs text-slate-500">
                Click any month to inspect its evidence in Graphical RAG
              </p>
            </div>
          </div>
          <div className="mt-3 h-[240px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart
                data={records}
                margin={{ top: 8, right: 10, left: 0, bottom: 0 }}
                onClick={(state: any) => {
                  const m = state?.activePayload?.[0]?.payload?.Month;
                  if (m) onSelectMonthForRag(m);
                }}
              >
                <TriadSvgDefs idPrefix="exec-trend" />
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" vertical={false} />
                <XAxis dataKey="Month_Label" tick={{ fontSize: 10, fill: "#64748B" }} minTickGap={26} />
                <YAxis
                  tick={{ fontSize: 10, fill: "#64748B" }}
                  tickFormatter={(v) => `₦${(v / 1e6).toFixed(0)}M`}
                  width={54}
                />
                <Tooltip
                  contentStyle={TRIAD_TOOLTIP_STYLE}
                  labelStyle={TRIAD_TOOLTIP_LABEL_STYLE}
                  itemStyle={TRIAD_TOOLTIP_ITEM_STYLE}
                  formatter={(val: any, name: any) => [formatNGN(Number(val)), name]}
                />
                <Legend wrapperStyle={{ fontSize: "11px" }} />
                <Area
                  type="monotone"
                  dataKey="Sales_Revenue_NGN"
                  name="Sales Revenue"
                  stroke="#0A2E65"
                  fill="url(#exec-trend-area-navy)"
                  fillOpacity={1}
                  strokeWidth={2.2}
                  dot={false}
                  activeDot={{ r: 4, fill: "#0A2E65", stroke: "#FFFFFF", strokeWidth: 1.5 }}
                />
                <Area
                  type="monotone"
                  dataKey="Gross_Profit_NGN"
                  name="Gross Profit"
                  stroke="#1D63B8"
                  fill="url(#exec-trend-area-cobalt)"
                  fillOpacity={1}
                  strokeWidth={2.1}
                  dot={false}
                  activeDot={{ r: 4, fill: "#1D63B8", stroke: "#FFFFFF", strokeWidth: 1.5 }}
                />
                <Area
                  type="monotone"
                  dataKey="Operating_Profit_NGN"
                  name="Operating Profit"
                  stroke="#0D9488"
                  fill="url(#exec-trend-area-teal)"
                  fillOpacity={1}
                  strokeWidth={2.3}
                  dot={false}
                  activeDot={{ r: 4.5, fill: "#0D9488", stroke: "#FFFFFF", strokeWidth: 1.5 }}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Profitability Overview Donut (3 cols) */}
        <div className="rounded-xl border border-slate-200 bg-white p-5 xl:col-span-3">
          <h2 className="font-display text-sm font-bold text-[#0A1E3D]">Profitability Overview</h2>
          <p className="text-xs text-slate-500">Revenue allocation across COGS, OpEx &amp; EBIT</p>
          <div className="relative mt-2 h-[175px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <TriadSvgDefs idPrefix="exec-donut" />
                <Pie
                  data={donutData}
                  dataKey="value"
                  nameKey="name"
                  innerRadius={52}
                  outerRadius={74}
                  paddingAngle={2.5}
                >
                  {donutData.map((entry) => (
                    <Cell key={entry.name} fill={entry.gradId} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={TRIAD_TOOLTIP_STYLE}
                  itemStyle={TRIAD_TOOLTIP_ITEM_STYLE}
                  formatter={(val: any) => formatNGN(Number(val))}
                />
              </PieChart>
            </ResponsiveContainer>
            <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
              <span className="font-display text-lg font-bold text-[#0A1E3D] tabular-nums">
                {formatPct(kpis.gross_margin_pct, 1)}
              </span>
              <span className="text-[10px] font-medium text-slate-500">Gross Margin</span>
            </div>
          </div>
          <div className="mt-2 space-y-1.5 text-xs">
            {donutData.map((d) => (
              <div key={d.name} className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-slate-600">
                  <span
                    className="h-2.5 w-2.5 rounded-sm shrink-0"
                    style={{ backgroundColor: d.color }}
                  />
                  <span>{d.name}</span>
                </div>
                <span className="font-mono font-semibold text-slate-900 tabular-nums">
                  {formatNGN(d.value, true)}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Key Metrics Stack + Right Rail Prediction Panel (4 cols) */}
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:col-span-4 xl:grid-cols-2">
          {/* Key Metrics Card */}
          <div className="flex flex-col justify-between rounded-xl border border-slate-200 bg-white p-5">
            <div>
              <h2 className="font-display text-sm font-bold text-[#0A1E3D]">Key Metrics</h2>
              <div className="mt-4 space-y-4">
                <div>
                  <div className="text-xs text-slate-500">Latest Monthly Operating Profit</div>
                  <div className="mt-0.5 font-display text-lg font-bold text-[#0A1E3D] tabular-nums">
                    {formatNGN(latestRec?.Operating_Profit_NGN, true)}
                  </div>
                  <div className="text-[11px] text-slate-500">{latestRec?.Month_Label}</div>
                </div>
                <div className="border-t border-slate-100 pt-3">
                  <div className="text-xs text-slate-500">Latest Operating Margin</div>
                  <div className="mt-0.5 font-display text-lg font-bold text-teal-700 tabular-nums">
                    {formatPct(latestRec?.Operating_Margin_Pct, 2)}
                  </div>
                </div>
                <div className="border-t border-slate-100 pt-3">
                  <div className="text-xs text-slate-500">Avg YoY Inflation</div>
                  <div className="mt-0.5 font-display text-lg font-bold text-amber-700 tabular-nums">
                    {formatPct(kpis.avg_inflation_pct, 2)}
                  </div>
                </div>
              </div>
            </div>
            <button
              type="button"
              onClick={() => onNavigate("performance")}
              className="mt-4 inline-flex items-center gap-1 text-xs font-semibold text-[#0A2E65] hover:underline"
            >
              <span>Deep performance view</span>
              <ArrowRight className="h-3.5 w-3.5" />
            </button>
          </div>

          {/* Right Rail: Operating Profit Prediction Card */}
          <div className="flex flex-col overflow-hidden rounded-xl border border-slate-200 bg-white">
            <div className="bg-gradient-to-r from-[#0A2E65] via-[#1D63B8] to-[#0D9488] px-4 py-3.5 text-white">
              <div className="flex items-center gap-2">
                <BrainCircuit className="h-4 w-4 text-teal-200" />
                <span className="font-display text-xs font-bold leading-tight">
                  Operating Profit Prediction
                </span>
              </div>
            </div>
            <div className="flex flex-1 flex-col justify-between p-4">
              <div>
                <div className="flex rounded-lg bg-slate-100 p-0.5 text-[11px] font-semibold">
                  <button
                    type="button"
                    onClick={() => setQuickTab("forecast")}
                    className={`flex-1 rounded-md py-1 transition-colors ${
                      quickTab === "forecast" ? "bg-white text-[#0A1E3D] shadow-sm" : "text-slate-600"
                    }`}
                  >
                    Forecast
                  </button>
                  <button
                    type="button"
                    onClick={() => setQuickTab("whatif")}
                    className={`flex-1 whitespace-nowrap rounded-md py-1 transition-colors ${
                      quickTab === "whatif" ? "bg-white text-[#0A1E3D] shadow-sm" : "text-slate-600"
                    }`}
                  >
                    FX Sim
                  </button>
                </div>

                {quickTab === "forecast" ? (
                  <div className="mt-3 space-y-2.5">
                    <div>
                      <label className="text-[11px] font-medium text-slate-500">
                        Quarter Benchmark
                      </label>
                      <select
                        value={selectedQuarterWidget}
                        onChange={(e) => setSelectedQuarterWidget(e.target.value)}
                        className="mt-1 w-full rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1.5 text-xs font-medium text-slate-800"
                      >
                        {["Q1", "Q2", "Q3", "Q4"].map((q) => (
                          <option key={q} value={q}>
                            {q} Historical Avg
                          </option>
                        ))}
                      </select>
                    </div>
                    <div className="rounded-lg bg-slate-50 p-3">
                      <div className="text-[11px] text-slate-500">
                        Latest Out-of-Time ML Prediction
                      </div>
                      <div className="mt-0.5 font-display text-lg font-bold text-[#0A1E3D] tabular-nums">
                        {formatNGN(lastTestPred?.predicted ?? baselineQuarterOp, true)}
                      </div>
                      <div className="mt-0.5 text-[11px] text-slate-600 tabular-nums">
                        {selectedModel?.name || "Regression"} · {selectedQuarterWidget} Avg:{" "}
                        {formatNGN(baselineQuarterOp, true)}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="mt-3 space-y-2.5">
                    <div>
                      <div className="flex justify-between text-[11px] text-slate-600">
                        <span>Global FX Shock</span>
                        <span className="font-mono font-semibold tabular-nums">
                          {driverDeltas.fxPct >= 0 ? "+" : ""}
                          {driverDeltas.fxPct.toFixed(0)}%
                        </span>
                      </div>
                      <input
                        type="range"
                        min="-20"
                        max="35"
                        step="1"
                        value={driverDeltas.fxPct}
                        style={getSliderFillStyle(driverDeltas.fxPct, -20, 35)}
                        onChange={(e) =>
                          setDriverDeltas({
                            ...driverDeltas,
                            fxPct: Number(e.target.value),
                          })
                        }
                        className="triad-slider mt-1 w-full"
                      />
                    </div>
                    <div className="rounded-lg bg-slate-50 p-3 space-y-1">
                      <div className="text-[11px] text-slate-500">
                        Simulated Latest-Month Profit ({activePresetLabel})
                      </div>
                      <div
                        className={`font-display text-lg font-bold tabular-nums ${
                          quickSimulatedOp >= 0 ? "text-[#0A1E3D]" : "text-red-600"
                        }`}
                      >
                        {formatNGN(quickSimulatedOp, true)}
                      </div>
                      {dec2030CustomOp !== undefined && (
                        <div className="text-[11px] text-slate-600 tabular-nums">
                          Dec 2030 Target:{" "}
                          <strong className="text-[#0A2E65]">
                            {formatNGN(dec2030CustomOp, true)}
                          </strong>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>

              <div className="mt-3 grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => setDrawerOpen(true)}
                  className="rounded-lg border border-slate-300 bg-white py-2 text-[11px] font-semibold text-[#0A1E3D] hover:bg-slate-50 transition-colors"
                >
                  {isScenarioActive ? "Edit Scenario" : "Open Simulator"}
                </button>
                <button
                  type="button"
                  onClick={() => onNavigate("forecasting")}
                  className="triad-btn-primary rounded-lg py-2 text-[11px] font-semibold text-white"
                >
                  2030 Forecast
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Row 3: Quarterly Financial Comparison + Impact of Key Factors */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        <div className="rounded-xl border border-slate-200 bg-white p-5 lg:col-span-5">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="font-display text-sm font-bold text-[#0A1E3D]">
                Quarterly Financial Comparison
              </h2>
              <p className="text-xs text-slate-500">
                Average monthly Revenue vs. Operating Profit by quarter
              </p>
            </div>
          </div>
          <div className="mt-3 h-[215px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={quarterly_summary} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                <TriadSvgDefs idPrefix="exec-qtr" />
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" vertical={false} />
                <XAxis dataKey="quarter" tick={{ fontSize: 11, fill: "#334155" }} />
                <YAxis
                  tick={{ fontSize: 10, fill: "#64748B" }}
                  tickFormatter={(v) => `₦${(v / 1e6).toFixed(0)}M`}
                  width={54}
                />
                <Tooltip
                  contentStyle={TRIAD_TOOLTIP_STYLE}
                  labelStyle={TRIAD_TOOLTIP_LABEL_STYLE}
                  itemStyle={TRIAD_TOOLTIP_ITEM_STYLE}
                  formatter={(val: any, name: any) => [formatNGN(Number(val)), name]}
                />
                <Legend wrapperStyle={{ fontSize: "11px" }} />
                <Bar
                  dataKey="avg_revenue"
                  name="Avg Monthly Revenue"
                  fill="url(#exec-qtr-bar-navy-cobalt)"
                  radius={[4, 4, 0, 0]}
                />
                <Bar
                  dataKey="avg_operating_profit"
                  name="Avg Operating Profit"
                  fill="url(#exec-qtr-bar-cobalt-teal)"
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-5 lg:col-span-7">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="font-display text-sm font-bold text-[#0A1E3D]">
                Impact of Key Factors (Correlation with Revenue &amp; Operating Profit)
              </h2>
              <p className="text-xs text-slate-500">
                Pearson correlation coefficients calculated across filtered observations
              </p>
            </div>
          </div>
          <div className="mt-3 h-[215px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={factor_impacts} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                <TriadSvgDefs idPrefix="exec-corr" />
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" vertical={false} />
                <XAxis dataKey="factor" tick={{ fontSize: 10, fill: "#334155" }} interval={0} />
                <YAxis domain={[-1, 1]} tick={{ fontSize: 10, fill: "#64748B" }} width={40} />
                <Tooltip
                  contentStyle={TRIAD_TOOLTIP_STYLE}
                  labelStyle={TRIAD_TOOLTIP_LABEL_STYLE}
                  itemStyle={TRIAD_TOOLTIP_ITEM_STYLE}
                  formatter={(val: any, name: any) => [Number(val).toFixed(3), name]}
                />
                <Legend wrapperStyle={{ fontSize: "11px" }} />
                <Bar
                  dataKey="corr_revenue"
                  name="Corr vs. Revenue"
                  fill="url(#exec-corr-bar-navy-cobalt)"
                  radius={[4, 4, 0, 0]}
                />
                <Bar
                  dataKey="corr_operating_profit"
                  name="Corr vs. Operating Profit"
                  fill="url(#exec-corr-bar-cobalt-teal)"
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Row 4: Evidence-Backed Market Insights + Prediction & Anomaly Summary */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        <div className="rounded-xl border border-slate-200 bg-white p-5 lg:col-span-7">
          <div className="flex items-center justify-between">
            <h2 className="font-display text-sm font-bold text-[#0A1E3D]">
              Evidence-Grounded Market Insights
            </h2>
            <button
              type="button"
              onClick={() => onNavigate("rag")}
              className="text-xs font-semibold text-[#0A2E65] hover:underline"
            >
              Explore Graphical RAG
            </button>
          </div>
          <ul className="mt-3 space-y-2.5 text-xs leading-relaxed text-slate-700">
            <li className="flex items-start gap-2">
              <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-[#0D9488] shrink-0" />
              <span>
                <strong className="text-slate-900">Top Predictive SHAP Drivers:</strong>{" "}
                {topDriver && secondDriver
                  ? `${topDriver.feature} (${topDriver.share_pct}% SHAP share) and ${secondDriver.feature} (${secondDriver.share_pct}% share) account for the largest model contribution to Operating Profit.`
                  : "Train model summary loading."}
              </span>
            </li>
            <li className="flex items-start gap-2">
              <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-amber-500 shrink-0" />
              <span>
                <strong className="text-slate-900">Macro &amp; FX Risk:</strong>{" "}
                {anomalies?.analytical_findings?.find((f) => f.id === "fx-cost-pressure")?.finding ||
                  "FX rate surges pass directly into local unit import costs."}
              </span>
            </li>
            <li className="flex items-start gap-2">
              <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-[#1D4ED8] shrink-0" />
              <span>
                <strong className="text-slate-900">Anomaly Surveillance:</strong>{" "}
                {anomalies
                  ? `${anomalies.summary.total_anomalies} anomalous periods flagged (${anomalies.summary.high_severity} high severity), notably Jun 2023 FX regime shift and Feb–Mar 2024 margin pressure.`
                  : "Analyzing historical series."}
              </span>
            </li>
          </ul>
        </div>

        <div className="rounded-xl border border-slate-200 bg-gradient-to-br from-[#0A2E65]/[0.04] via-[#1D63B8]/[0.05] to-[#0D9488]/[0.09] p-5 lg:col-span-5 flex flex-col justify-between">
          <div>
            <h2 className="font-display text-sm font-bold text-[#0A1E3D]">
              Model &amp; Data Integrity Status
            </h2>
            <div className="mt-3 flex items-start gap-3">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-[#0A2E65] via-[#1D63B8] to-[#0D9488] text-white font-display text-sm font-bold shadow-sm">
                ML
              </div>
              <div>
                <div className="text-sm font-bold text-[#0A1E3D]">
                  {selectedModel?.name || "Out-of-Time Regression"}
                </div>
                <p className="mt-0.5 text-xs text-slate-700">
                  Chronological test MAPE:{" "}
                  <strong className="font-mono tabular-nums">
                    {selectedModel?.test_metrics?.mape_pct ?? "—"}%
                  </strong>{" "}
                  · Records verified:{" "}
                  <strong className="font-mono tabular-nums">
                    {dataset.quality_summary.records_processed}
                  </strong>{" "}
                  · Material calc errors:{" "}
                  <strong className="font-mono tabular-nums">
                    {dataset.quality_summary.material_discrepancies_count}
                  </strong>
                </p>
              </div>
            </div>
          </div>
          <div className="mt-4 flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => onNavigate("explainability")}
              className="rounded-lg border border-[#1D63B8]/30 bg-white px-3 py-1.5 text-xs font-semibold text-[#0A2E65] hover:bg-blue-50/50 transition-colors"
            >
              Inspect SHAP Explainability
            </button>
            <button
              type="button"
              onClick={() => onNavigate("anomalies")}
              className="triad-btn-primary rounded-lg px-3 py-1.5 text-xs font-semibold text-white"
            >
              Review Flagged Anomalies
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
