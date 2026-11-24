import React, { useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { Search, SlidersHorizontal } from "lucide-react";
import { DatasetState, FinancialRecord } from "../types/financial";
import { formatFX, formatNGN, formatPct } from "../utils/format";
import { GraphicalRagPanel } from "../components/GraphicalRagPanel";
import { useScenario } from "../context/ScenarioContext";
import {
  TRIAD_TOOLTIP_ITEM_STYLE,
  TRIAD_TOOLTIP_LABEL_STYLE,
  TRIAD_TOOLTIP_STYLE,
  TriadSvgDefs,
} from "../utils/chartTheme";

interface FinancialPerformanceViewProps {
  dataset: DatasetState;
}

export const FinancialPerformanceView: React.FC<FinancialPerformanceViewProps> = ({ dataset }) => {
  const { records, profit_distribution, quarterly_summary } = dataset;
  const {
    activePresetLabel,
    isScenarioActive,
    singlePeriodResult,
    setAnchorRecord,
    setDrawerOpen,
  } = useScenario();
  const [search, setSearch] = useState("");
  const [selectedRecord, setSelectedRecord] = useState<FinancialRecord | null>(null);
  const [ragLens, setRagLens] = useState<"fx_intelligence" | "cost_intelligence" | "sales_intelligence">("fx_intelligence");

  const filteredLedger = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return [...records].reverse();
    return [...records]
      .reverse()
      .filter(
        (r) =>
          r.Month_Label.toLowerCase().includes(q) ||
          r.Quarter.toLowerCase().includes(q) ||
          (r.Market_Notes && r.Market_Notes.toLowerCase().includes(q))
      );
  }, [records, search]);

  const histMonthlyAvgOp = useMemo(
    () =>
      records.length
        ? records.reduce((s, r) => s + r.Operating_Profit_NGN, 0) / records.length
        : 0,
    [records]
  );
  const histMonthlyAvgRev = useMemo(
    () =>
      records.length
        ? records.reduce((s, r) => s + r.Sales_Revenue_NGN, 0) / records.length
        : 0,
    [records]
  );

  return (
    <div className="space-y-6">
      {singlePeriodResult && (
        <div className="rounded-xl border border-slate-200 bg-white p-4 flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div>
            <div className="text-xs font-bold text-[#0A1E3D]">
              Scenario Run-Rate Benchmark ({activePresetLabel}) vs. Filtered Historical Average
            </div>
            <div className="mt-1 flex flex-wrap gap-x-5 gap-y-1 text-xs text-slate-600 tabular-nums">
              <span>
                Filtered Monthly Avg Revenue:{" "}
                <strong className="font-mono text-[#0A1E3D]">{formatNGN(histMonthlyAvgRev, true)}</strong>
              </span>
              <span>
                Filtered Monthly Avg Profit:{" "}
                <strong className="font-mono text-[#0A1E3D]">{formatNGN(histMonthlyAvgOp, true)}</strong>
              </span>
              <span>
                Simulated Monthly Profit:{" "}
                <strong
                  className={`font-mono ${
                    singlePeriodResult.simulated.accounting.Accounting_Operating_Profit_NGN >= histMonthlyAvgOp
                      ? "text-teal-700"
                      : "text-amber-700"
                  }`}
                >
                  {formatNGN(singlePeriodResult.simulated.accounting.Accounting_Operating_Profit_NGN, true)}
                </strong>{" "}
                ({formatPct(singlePeriodResult.simulated.accounting.Accounting_Operating_Margin_Pct, 1)} margin)
              </span>
            </div>
          </div>
          <button
            type="button"
            onClick={() => setDrawerOpen(true)}
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-300 bg-slate-50 px-3 py-1.5 text-xs font-semibold text-[#0A1E3D] hover:bg-slate-100 whitespace-nowrap self-start md:self-auto"
          >
            <SlidersHorizontal className="h-3.5 w-3.5 text-teal-700" />
            <span>{isScenarioActive ? "Modify Scenario Levers" : "Simulate Shock"}</span>
          </button>
        </div>
      )}

      {/* 1. Revenue, Gross Profit & Operating Profit + Operating Margin Development */}
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-12">
        <div className="rounded-xl border border-slate-200 bg-white p-5 lg:col-span-7">
          <h2 className="font-display text-sm font-bold text-[#0A1E3D]">
            Monthly Revenue, Gross Profit &amp; Operating Profit
          </h2>
          <p className="mt-0.5 text-xs text-slate-500">
            Tracks top-line revenue expansion against gross spread and net operating profitability.
          </p>
          <div className="mt-3 h-[260px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={records} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                <TriadSvgDefs idPrefix="fp-rev" />
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis dataKey="Month_Label" tick={{ fontSize: 10, fill: "#64748B" }} minTickGap={24} />
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
                  fill="url(#fp-rev-area-navy)"
                  fillOpacity={1}
                  strokeWidth={2.2}
                />
                <Area
                  type="monotone"
                  dataKey="Gross_Profit_NGN"
                  name="Gross Profit"
                  stroke="#1D63B8"
                  fill="url(#fp-rev-area-cobalt)"
                  fillOpacity={1}
                  strokeWidth={2.2}
                />
                <Area
                  type="monotone"
                  dataKey="Operating_Profit_NGN"
                  name="Operating Profit"
                  stroke="#0D9488"
                  fill="url(#fp-rev-area-teal)"
                  fillOpacity={1}
                  strokeWidth={2.4}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-5 lg:col-span-5">
          <h2 className="font-display text-sm font-bold text-[#0A1E3D]">
            Gross &amp; Operating Margin Development (%)
          </h2>
          <p className="mt-0.5 text-xs text-slate-500">
            Highlights margin stability (2016–2022) and the 2023–2024 FX compression trough.
          </p>
          <div className="mt-3 h-[260px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={records} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                <TriadSvgDefs idPrefix="fp-mrg" />
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis dataKey="Month_Label" tick={{ fontSize: 10, fill: "#64748B" }} minTickGap={24} />
                <YAxis tick={{ fontSize: 10, fill: "#64748B" }} tickFormatter={(v) => `${v}%`} width={44} />
                <Tooltip
                  contentStyle={TRIAD_TOOLTIP_STYLE}
                  labelStyle={TRIAD_TOOLTIP_LABEL_STYLE}
                  itemStyle={TRIAD_TOOLTIP_ITEM_STYLE}
                  formatter={(val: any, name: any) => [`${Number(val).toFixed(2)}%`, name]}
                />
                <Legend wrapperStyle={{ fontSize: "11px" }} />
                <Line
                  type="monotone"
                  dataKey="Gross_Margin_Pct"
                  name="Gross Margin %"
                  stroke="url(#fp-mrg-stroke-navy-cobalt)"
                  strokeWidth={2.2}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="Operating_Margin_Pct"
                  name="Operating Margin %"
                  stroke="url(#fp-mrg-stroke-cobalt-teal)"
                  strokeWidth={2.2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* 2. Macro & Cost Correlations: FX vs Profit, Inflation vs Profit, Marketing vs Profit, Import Cost vs Profit */}
      <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
        <div className="rounded-xl border border-slate-200 bg-white p-5">
          <h2 className="font-display text-sm font-bold text-[#0A1E3D]">
            FX Rate (USD/NGN) vs. Operating Profit
          </h2>
          <p className="mt-0.5 text-xs text-slate-500">
            Shows how Naira depreciation regimes reshaped pricing and nominal profit levels.
          </p>
          <div className="mt-3 h-[220px]">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 10, right: 16, left: 0, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis
                  type="number"
                  dataKey="USD_NGN_Planning_Rate"
                  name="USD/NGN"
                  tick={{ fontSize: 10, fill: "#64748B" }}
                  tickFormatter={(v) => `₦${v}`}
                />
                <YAxis
                  type="number"
                  dataKey="Operating_Profit_NGN"
                  name="Operating Profit"
                  tick={{ fontSize: 10, fill: "#64748B" }}
                  tickFormatter={(v) => `₦${(v / 1e6).toFixed(0)}M`}
                  width={54}
                />
                <ZAxis range={[36, 36]} />
                <Tooltip
                  contentStyle={TRIAD_TOOLTIP_STYLE}
                  labelStyle={TRIAD_TOOLTIP_LABEL_STYLE}
                  itemStyle={TRIAD_TOOLTIP_ITEM_STYLE}
                  cursor={{ strokeDasharray: "3 3" }}
                  formatter={(val: any, name: any) => [
                    name === "USD/NGN" ? formatFX(Number(val)) : formatNGN(Number(val)),
                    name,
                  ]}
                />
                <Scatter name="Monthly Observation" data={records} fill="#0A2E65" fillOpacity={0.75} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-5">
          <h2 className="font-display text-sm font-bold text-[#0A1E3D]">
            Inflation YoY (%) vs. Operating Margin (%)
          </h2>
          <p className="mt-0.5 text-xs text-slate-500">
            Illustrates operating margin sensitivity during high-inflation periods (&gt;25% YoY).
          </p>
          <div className="mt-3 h-[220px]">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 10, right: 16, left: 0, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis
                  type="number"
                  dataKey="Inflation_YoY_Pct"
                  name="Inflation YoY"
                  tick={{ fontSize: 10, fill: "#64748B" }}
                  tickFormatter={(v) => `${v}%`}
                />
                <YAxis
                  type="number"
                  dataKey="Operating_Margin_Pct"
                  name="Operating Margin"
                  tick={{ fontSize: 10, fill: "#64748B" }}
                  tickFormatter={(v) => `${v}%`}
                  width={44}
                />
                <ZAxis range={[36, 36]} />
                <Tooltip
                  contentStyle={TRIAD_TOOLTIP_STYLE}
                  labelStyle={TRIAD_TOOLTIP_LABEL_STYLE}
                  itemStyle={TRIAD_TOOLTIP_ITEM_STYLE}
                  formatter={(val: any, name: any) => [`${Number(val).toFixed(2)}%`, name]}
                />
                <Scatter name="Monthly Observation" data={records} fill="#1D63B8" fillOpacity={0.75} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-5">
          <h2 className="font-display text-sm font-bold text-[#0A1E3D]">
            Marketing Expenditure vs. Operating Profit
          </h2>
          <p className="mt-0.5 text-xs text-slate-500">
            Tracks promotional spend efficiency and Q4 seasonal campaign scaling.
          </p>
          <div className="mt-3 h-[220px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={records} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                <TriadSvgDefs idPrefix="fp-mkt" />
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis dataKey="Month_Label" tick={{ fontSize: 10, fill: "#64748B" }} minTickGap={24} />
                <YAxis
                  yAxisId="left"
                  tick={{ fontSize: 10, fill: "#64748B" }}
                  tickFormatter={(v) => `₦${(v / 1e6).toFixed(0)}M`}
                  width={54}
                />
                <YAxis
                  yAxisId="right"
                  orientation="right"
                  tick={{ fontSize: 10, fill: "#64748B" }}
                  tickFormatter={(v) => `₦${(v / 1e6).toFixed(1)}M`}
                  width={54}
                />
                <Tooltip
                  contentStyle={TRIAD_TOOLTIP_STYLE}
                  labelStyle={TRIAD_TOOLTIP_LABEL_STYLE}
                  itemStyle={TRIAD_TOOLTIP_ITEM_STYLE}
                  formatter={(val: any, name: any) => [formatNGN(Number(val)), name]}
                />
                <Legend wrapperStyle={{ fontSize: "11px" }} />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="Operating_Profit_NGN"
                  name="Operating Profit"
                  stroke="url(#fp-mkt-stroke-cobalt-teal)"
                  strokeWidth={2.2}
                  dot={false}
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="Marketing_Spend_NGN"
                  name="Marketing Spend"
                  stroke="url(#fp-mkt-stroke-navy-cobalt)"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-5">
          <h2 className="font-display text-sm font-bold text-[#0A1E3D]">
            Import Cost (COGS) &amp; Profit Distribution
          </h2>
          <p className="mt-0.5 text-xs text-slate-500">
            Frequency distribution of monthly Operating Profit across the selected period.
          </p>
          <div className="mt-3 h-[220px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={profit_distribution} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                <TriadSvgDefs idPrefix="fp-dist" />
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis dataKey="range" tick={{ fontSize: 9, fill: "#475569" }} interval={0} />
                <YAxis tick={{ fontSize: 10, fill: "#64748B" }} width={34} />
                <Tooltip
                  contentStyle={TRIAD_TOOLTIP_STYLE}
                  labelStyle={TRIAD_TOOLTIP_LABEL_STYLE}
                  itemStyle={TRIAD_TOOLTIP_ITEM_STYLE}
                />
                <Bar dataKey="count" name="Months" fill="url(#fp-dist-bar-triad-v)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* 3. Quarterly Table Summary */}
      <div className="rounded-xl border border-slate-200 bg-white p-5">
        <h2 className="font-display text-sm font-bold text-[#0A1E3D]">
          Quarterly Performance Breakdown
        </h2>
        <div className="mt-3 overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-200 text-slate-500">
                <th className="py-2 pr-4 font-semibold">Quarter</th>
                <th className="py-2 pr-4 text-right font-semibold">Months</th>
                <th className="py-2 pr-4 text-right font-semibold">Total Revenue</th>
                <th className="py-2 pr-4 text-right font-semibold">Total Gross Profit</th>
                <th className="py-2 pr-4 text-right font-semibold">Total Operating Profit</th>
                <th className="py-2 pr-4 text-right font-semibold">Avg Gross Margin</th>
                <th className="py-2 pr-4 text-right font-semibold">Avg Operating Margin</th>
                <th className="py-2 text-right font-semibold">Total Units</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {quarterly_summary.map((q) => (
                <tr key={q.quarter} className="hover:bg-slate-50/80">
                  <td className="py-2.5 pr-4 font-semibold text-[#0A1E3D]">{q.quarter}</td>
                  <td className="py-2.5 pr-4 text-right font-mono tabular-nums">{q.months_count}</td>
                  <td className="py-2.5 pr-4 text-right font-mono tabular-nums">
                    {formatNGN(q.total_revenue)}
                  </td>
                  <td className="py-2.5 pr-4 text-right font-mono tabular-nums">
                    {formatNGN(q.total_gross_profit)}
                  </td>
                  <td className="py-2.5 pr-4 text-right font-mono font-semibold text-teal-700 tabular-nums">
                    {formatNGN(q.total_operating_profit)}
                  </td>
                  <td className="py-2.5 pr-4 text-right font-mono tabular-nums">
                    {formatPct(q.avg_gross_margin_pct)}
                  </td>
                  <td className="py-2.5 pr-4 text-right font-mono tabular-nums">
                    {formatPct(q.avg_operating_margin_pct)}
                  </td>
                  <td className="py-2.5 text-right font-mono tabular-nums">
                    {q.total_units.toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* 4. Interactive Monthly Ledger with Drill-Down */}
      <div className="rounded-xl border border-slate-200 bg-white p-5">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="font-display text-sm font-bold text-[#0A1E3D]">
              Monthly Financial Ledger (Recalculated &amp; Verified)
            </h2>
            <p className="text-xs text-slate-500">
              Select any row to inspect unit economics and trigger Graphical RAG evidence retrieval.
            </p>
          </div>
          <div className="relative w-full sm:w-64">
            <Search className="absolute left-3 top-2.5 h-3.5 w-3.5 text-slate-400" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Filter month, quarter, or note..."
              className="w-full rounded-lg border border-slate-200 bg-slate-50 pl-8 pr-3 py-1.5 text-xs focus:border-[#0A2E65] focus:bg-white focus:outline-none"
            />
          </div>
        </div>

        <div className="mt-3 max-h-[340px] overflow-auto">
          <table className="w-full text-left text-xs">
            <thead className="sticky top-0 bg-slate-50 border-b border-slate-200 text-slate-600">
              <tr>
                <th className="py-2 px-3 font-semibold">Month</th>
                <th className="py-2 px-3 font-semibold">Qtr</th>
                <th className="py-2 px-3 text-right font-semibold">USD/NGN</th>
                <th className="py-2 px-3 text-right font-semibold">Units</th>
                <th className="py-2 px-3 text-right font-semibold">Price (₦)</th>
                <th className="py-2 px-3 text-right font-semibold">Revenue</th>
                <th className="py-2 px-3 text-right font-semibold">COGS</th>
                <th className="py-2 px-3 text-right font-semibold">Operating Profit</th>
                <th className="py-2 px-3 text-right font-semibold">Op Margin</th>
                <th className="py-2 px-3 font-semibold">Market Notes</th>
                <th className="py-2 px-3 text-right font-semibold">Scenario</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {filteredLedger.map((r) => (
                <tr
                  key={r.Month}
                  onClick={() => setSelectedRecord(r)}
                  className={`cursor-pointer transition-colors hover:bg-blue-50/50 ${
                    selectedRecord?.Month === r.Month ? "bg-blue-50" : ""
                  }`}
                >
                  <td className="py-2 px-3 font-medium text-[#0A1E3D] whitespace-nowrap">
                    {r.Month_Label}
                  </td>
                  <td className="py-2 px-3 text-slate-600">{r.Quarter}</td>
                  <td className="py-2 px-3 text-right font-mono tabular-nums">
                    {r.USD_NGN_Planning_Rate.toFixed(2)}
                  </td>
                  <td className="py-2 px-3 text-right font-mono tabular-nums">
                    {r.Units_Sold.toLocaleString()}
                  </td>
                  <td className="py-2 px-3 text-right font-mono tabular-nums">
                    {r.Avg_Selling_Price_NGN.toLocaleString()}
                  </td>
                  <td className="py-2 px-3 text-right font-mono tabular-nums">
                    {formatNGN(r.Sales_Revenue_NGN, true)}
                  </td>
                  <td className="py-2 px-3 text-right font-mono tabular-nums">
                    {formatNGN(r.COGS_NGN, true)}
                  </td>
                  <td
                    className={`py-2 px-3 text-right font-mono font-semibold tabular-nums ${
                      r.Operating_Profit_NGN < 0 ? "text-red-600" : "text-teal-700"
                    }`}
                  >
                    {formatNGN(r.Operating_Profit_NGN, true)}
                  </td>
                  <td
                    className={`py-2 px-3 text-right font-mono tabular-nums ${
                      r.Operating_Margin_Pct < 0 ? "text-red-600" : "text-slate-700"
                    }`}
                  >
                    {formatPct(r.Operating_Margin_Pct)}
                  </td>
                  <td className="py-2 px-3 text-slate-600 max-w-[180px] truncate">
                    {r.Market_Notes || "—"}
                  </td>
                  <td className="py-2 px-3 text-right">
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        setAnchorRecord(r);
                        setDrawerOpen(true);
                      }}
                      className="rounded border border-slate-300 bg-white px-2 py-0.5 text-[11px] font-semibold text-[#0A2E65] hover:bg-slate-100 whitespace-nowrap"
                    >
                      Anchor Sim
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* 5. Integrated Graphical RAG Lens */}
      <div className="space-y-2">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="text-xs font-semibold text-slate-600">
            Graphical RAG Performance Lens {selectedRecord ? `· Focused on ${selectedRecord.Month_Label}` : ""}
          </div>
          <div className="flex gap-1 rounded-lg bg-slate-200/70 p-1">
            {[
              { id: "fx_intelligence", label: "FX Intelligence" },
              { id: "cost_intelligence", label: "Cost Intelligence" },
              { id: "sales_intelligence", label: "Sales Intelligence" },
            ].map((l) => (
              <button
                key={l.id}
                type="button"
                onClick={() => setRagLens(l.id as any)}
                className={`rounded-md px-2.5 py-1 text-xs font-medium ${
                  ragLens === l.id ? "bg-white text-[#0A1E3D] shadow-sm" : "text-slate-600"
                }`}
              >
                {l.label}
              </button>
            ))}
          </div>
        </div>
        <GraphicalRagPanel
          defaultLens={ragLens}
          selectedMonth={selectedRecord?.Month || null}
          title="Financial Performance Graphical RAG Panel"
        />
      </div>
    </div>
  );
};
