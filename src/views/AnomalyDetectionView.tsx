import React, { useEffect, useState } from "react";
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
import { AlertTriangle, SearchCheck, SlidersHorizontal } from "lucide-react";
import { AnomalyItem, AnomalyReport } from "../types/financial";
import { formatNGN } from "../utils/format";
import { GraphicalRagPanel } from "../components/GraphicalRagPanel";
import { useScenario } from "../context/ScenarioContext";
import {
  TRIAD_TOOLTIP_ITEM_STYLE,
  TRIAD_TOOLTIP_LABEL_STYLE,
  TRIAD_TOOLTIP_STYLE,
  TriadSvgDefs,
} from "../utils/chartTheme";

interface AnomalyDetectionViewProps {
  data: AnomalyReport | null;
}

export const AnomalyDetectionView: React.FC<AnomalyDetectionViewProps> = ({ data }) => {
  const {
    activePresetLabel,
    singlePeriodResult,
    forecastScenarios,
    setDrawerOpen,
  } = useScenario();
  const [selectedAnomaly, setSelectedAnomaly] = useState<AnomalyItem | null>(null);
  const [severityFilter, setSeverityFilter] = useState<string>("ALL");

  useEffect(() => {
    if (data?.anomalies?.length && !selectedAnomaly) {
      setSelectedAnomaly(data.anomalies[0]);
    }
  }, [data]);

  if (!data) {
    return (
      <div className="rounded-xl border border-slate-200 bg-white p-8 text-center text-sm text-slate-500">
        Loading Isolation Forest and time-series anomaly diagnostics...
      </div>
    );
  }

  const filteredAnomalies =
    severityFilter === "ALL"
      ? data.anomalies
      : data.anomalies.filter((a) => a.severity === severityFilter);

  const formatMetricVal = (variable: string, val: number) => {
    if (variable.includes("Pct")) return `${val.toFixed(2)}%`;
    return formatNGN(val, true);
  };

  // Empirical distribution parameters from historical actual_operating_profit series
  const profitSeries = data.time_series.map((t) => t.actual_operating_profit);
  const nObs = profitSeries.length;
  const meanProfit = nObs > 0 ? profitSeries.reduce((s, v) => s + v, 0) / nObs : 0;
  const stdProfit =
    nObs > 1
      ? Math.sqrt(profitSeries.reduce((s, v) => s + Math.pow(v - meanProfit, 2), 0) / (nObs - 1))
      : 0;
  const lower2Sigma = meanProfit - 2 * stdProfit;
  const upper2Sigma = meanProfit + 2 * stdProfit;

  const simSingleProfit = singlePeriodResult?.simulated.accounting.Accounting_Operating_Profit_NGN ?? null;
  const simZScore = simSingleProfit !== null && stdProfit > 0 ? (simSingleProfit - meanProfit) / stdProfit : null;
  const custom2030 = forecastScenarios?.cases?.custom?.summary;

  return (
    <div className="space-y-6">
      {/* Active Scenario Anomaly Stress Benchmark */}
      <div className="rounded-xl border border-slate-200 bg-white p-4">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <div className="text-xs font-bold text-[#0A1E3D]">
              Active Scenario Anomaly &amp; Tail-Risk Benchmark ({activePresetLabel})
            </div>
            <div className="mt-1 flex flex-wrap items-center gap-x-5 gap-y-1 text-[11px] text-slate-600 tabular-nums">
              <span>
                Historical Normal Band (μ ± 2σ):{" "}
                <strong className="font-mono text-[#0A1E3D]">
                  {formatNGN(lower2Sigma, true)} to {formatNGN(upper2Sigma, true)}
                </strong>
              </span>
              {simSingleProfit !== null && (
                <span>
                  1M Simulated Profit:{" "}
                  <strong className="font-mono text-[#0A1E3D]">{formatNGN(simSingleProfit, true)}</strong>
                  {simZScore !== null && (
                    <span
                      className={`ml-1.5 font-mono font-bold ${
                        Math.abs(simZScore) >= 3
                          ? "text-red-600"
                          : Math.abs(simZScore) >= 2
                            ? "text-amber-700"
                            : "text-teal-700"
                      }`}
                    >
                      (z = {simZScore >= 0 ? "+" : ""}
                      {simZScore.toFixed(2)} · {Math.abs(simZScore) >= 2 ? "Outside 2σ Band" : "Within Normal Band"})
                    </span>
                  )}
                </span>
              )}
              {custom2030 && (
                <span>
                  2026–2030 Simulated Deficit / Stress Months:{" "}
                  <strong className="font-mono text-[#0A1E3D]">
                    {custom2030.negative_months} Deficit / {custom2030.risk_months} Margin-Risk
                  </strong>
                </span>
              )}
            </div>
          </div>
          <button
            type="button"
            onClick={() => setDrawerOpen(true)}
            className="inline-flex items-center gap-1.5 self-start rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs font-semibold text-[#0A1E3D] hover:bg-slate-100 whitespace-nowrap lg:self-auto"
          >
            <SlidersHorizontal className="h-3.5 w-3.5 text-[#0A2E65]" />
            <span>Stress Test Scenario</span>
          </button>
        </div>
      </div>

      {/* KPI Summary Strip */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <div className="text-xs font-medium text-slate-500">Anomalous Months Flagged</div>
          <div className="mt-1 font-display text-2xl font-bold text-[#0A1E3D] tabular-nums">
            {data.summary.total_anomalies}
          </div>
          <div className="mt-0.5 text-[11px] text-slate-500">
            Isolation Forest + Z-Score + IQR
          </div>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <div className="text-xs font-medium text-slate-500">High Severity Events</div>
          <div className="mt-1 font-display text-2xl font-bold text-red-600 tabular-nums">
            {data.summary.high_severity}
          </div>
          <div className="mt-0.5 text-[11px] text-slate-500">|z| &ge; 3.0 or multi-method shock</div>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <div className="text-xs font-medium text-slate-500">Medium Severity Events</div>
          <div className="mt-1 font-display text-2xl font-bold text-amber-700 tabular-nums">
            {data.summary.medium_severity}
          </div>
          <div className="mt-0.5 text-[11px] text-slate-500">2.0 &le; |z| &lt; 3.0</div>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <div className="text-xs font-medium text-slate-500">Low Severity Deviations</div>
          <div className="mt-1 font-display text-2xl font-bold text-[#0D9488] tabular-nums">
            {data.summary.low_severity}
          </div>
          <div className="mt-0.5 text-[11px] text-slate-500">Single-method threshold</div>
        </div>
      </div>

      {/* Time-Series Anomaly Markers & Residuals */}
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-12">
        <div className="rounded-xl border border-slate-200 bg-white p-5 lg:col-span-8">
          <h2 className="font-display text-sm font-bold text-[#0A1E3D]">
            Actual vs. Expected Operating Profit &amp; Anomaly Markers
          </h2>
          <p className="mt-0.5 text-xs text-slate-500">
            Red/Amber markers indicate periods flagged by Isolation Forest, IQR, or Time-Series deviation.
          </p>
          <div className="mt-3 h-[260px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data.time_series} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                <TriadSvgDefs idPrefix="anom-ts" />
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis dataKey="month_label" tick={{ fontSize: 10, fill: "#64748B" }} minTickGap={24} />
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
                <Line
                  type="monotone"
                  dataKey="expected_operating_profit"
                  name="Expected Operating Profit"
                  stroke="url(#anom-ts-stroke-cobalt-teal)"
                  strokeDasharray="4 4"
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="actual_operating_profit"
                  name="Actual Operating Profit"
                  stroke="url(#anom-ts-stroke-navy-cobalt)"
                  strokeWidth={2.2}
                  dot={(props: any) => {
                    const { cx, cy, payload } = props;
                    if (!payload?.is_anomaly || cx === undefined || cy === undefined) return <g key={props.index} />;
                    return (
                      <circle
                        key={`anom-${payload.month}`}
                        cx={cx}
                        cy={cy}
                        r={payload.severity === "High" ? 5 : 3.5}
                        fill={payload.severity === "High" ? "#DC2626" : "#D97706"}
                        stroke="#FFFFFF"
                        strokeWidth={1.2}
                      />
                    );
                  }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Severity & Feature Breakdown */}
        <div className="rounded-xl border border-slate-200 bg-white p-5 lg:col-span-4 space-y-4">
          <div>
            <h2 className="font-display text-sm font-bold text-[#0A1E3D]">
              Severity &amp; Variable Breakdown
            </h2>
            <div className="mt-2 h-[110px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data.severity_distribution}>
                  <TriadSvgDefs idPrefix="anom-sev" />
                  <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                  <XAxis dataKey="severity" tick={{ fontSize: 10, fill: "#334155" }} />
                  <YAxis tick={{ fontSize: 10, fill: "#64748B" }} width={28} />
                  <Tooltip
                    contentStyle={TRIAD_TOOLTIP_STYLE}
                    labelStyle={TRIAD_TOOLTIP_LABEL_STYLE}
                    itemStyle={TRIAD_TOOLTIP_ITEM_STYLE}
                  />
                  <Bar dataKey="count" name="Anomalies" radius={[4, 4, 0, 0]}>
                    {data.severity_distribution.map((s) => (
                      <Cell
                        key={s.severity}
                        fill={
                          s.severity === "High"
                            ? "url(#anom-sev-bar-negative)"
                            : s.severity === "Medium"
                              ? "#D97706"
                              : "url(#anom-sev-bar-cobalt-teal)"
                        }
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="border-t border-slate-100 pt-3">
            <div className="text-xs font-semibold text-slate-700">Primary Variable Triggered</div>
            <div className="mt-2 space-y-1.5 text-xs">
              {data.feature_breakdown.map((fb) => (
                <div key={fb.variable} className="flex items-center justify-between">
                  <span className="font-mono text-[11px] text-slate-700">{fb.variable}</span>
                  <span className="font-mono font-semibold text-[#0A1E3D] tabular-nums">
                    {fb.count}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Residual Time-Series Bar */}
      <div className="rounded-xl border border-slate-200 bg-white p-5">
        <h2 className="font-display text-sm font-bold text-[#0A1E3D]">
          Time-Series Residual Deviations (Actual − Expected Operating Profit)
        </h2>
        <div className="mt-3 h-[170px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data.time_series}>
              <TriadSvgDefs idPrefix="anom-res" />
              <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
              <XAxis dataKey="month_label" tick={{ fontSize: 9, fill: "#64748B" }} minTickGap={24} />
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
              <ReferenceLine y={0} stroke="#64748B" />
              <Bar dataKey="residual" name="Residual" radius={[2, 2, 0, 0]}>
                {data.time_series.map((t) => (
                  <Cell
                    key={t.month}
                    fill={t.residual >= 0 ? "url(#anom-res-bar-cobalt-teal)" : "url(#anom-res-bar-negative)"}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Anomalies Table + Detailed Inspector */}
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-12">
        <div className="rounded-xl border border-slate-200 bg-white p-5 lg:col-span-6">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-amber-600" />
              <h2 className="font-display text-sm font-bold text-[#0A1E3D]">
                Detected Anomalies ({filteredAnomalies.length})
              </h2>
            </div>
            <div className="flex gap-1 rounded-lg bg-slate-100 p-0.5 text-[11px] font-medium">
              {["ALL", "High", "Medium", "Low"].map((sev) => (
                <button
                  key={sev}
                  type="button"
                  onClick={() => setSeverityFilter(sev)}
                  className={`rounded-md px-2 py-1 ${
                    severityFilter === sev ? "bg-white text-[#0A1E3D] shadow-sm" : "text-slate-600"
                  }`}
                >
                  {sev}
                </button>
              ))}
            </div>
          </div>

          <div className="mt-3 max-h-[380px] overflow-auto divide-y divide-slate-100">
            {filteredAnomalies.map((a) => {
              const active = selectedAnomaly?.month === a.month && selectedAnomaly?.variable_affected === a.variable_affected;
              return (
                <button
                  key={`${a.month}-${a.variable_affected}`}
                  type="button"
                  onClick={() => setSelectedAnomaly(a)}
                  className={`w-full px-3 py-3 text-left transition-colors hover:bg-slate-50 ${
                    active ? "bg-blue-50/70" : ""
                  }`}
                >
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-bold text-[#0A1E3D]">
                      {a.month_label} ({a.quarter}) · {a.variable_label}
                    </span>
                    <span
                      className={`font-semibold ${
                        a.severity === "High"
                          ? "text-red-600"
                          : a.severity === "Medium"
                            ? "text-amber-700"
                            : "text-teal-700"
                      }`}
                    >
                      {a.severity}
                    </span>
                  </div>
                  <div className="mt-1 flex items-center justify-between text-[11px] text-slate-600 tabular-nums">
                    <span>
                      Observed: {formatMetricVal(a.variable_affected, a.observed_value)} vs. Ref:{" "}
                      {formatMetricVal(a.variable_affected, a.expected_value)}
                    </span>
                    <span className="font-mono font-semibold">
                      {a.deviation_pct >= 0 ? "+" : ""}
                      {a.deviation_pct.toFixed(1)}%
                    </span>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* 8-Point Anomaly Inspector */}
        <div className="rounded-xl border border-slate-200 bg-white p-5 lg:col-span-6">
          {selectedAnomaly ? (
            <div className="space-y-3 text-xs">
              <div className="flex items-start justify-between border-b border-slate-100 pb-3">
                <div>
                  <div className="text-[11px] font-medium text-slate-500">
                    {selectedAnomaly.month} · {selectedAnomaly.methods_triggered.join(" · ")}
                  </div>
                  <h3 className="font-display text-base font-bold text-[#0A1E3D]">
                    {selectedAnomaly.month_label}: {selectedAnomaly.variable_label}
                  </h3>
                </div>
                <span
                  className={`font-bold ${
                    selectedAnomaly.severity === "High"
                      ? "text-red-600"
                      : selectedAnomaly.severity === "Medium"
                        ? "text-amber-700"
                        : "text-teal-700"
                  }`}
                >
                  {selectedAnomaly.severity} Severity
                </span>
              </div>

              <div className="grid grid-cols-3 gap-2 rounded-lg bg-slate-50 p-3">
                <div>
                  <div className="text-[11px] text-slate-500">Observed Value</div>
                  <div className="font-mono font-bold text-slate-900 tabular-nums">
                    {formatMetricVal(selectedAnomaly.variable_affected, selectedAnomaly.observed_value)}
                  </div>
                </div>
                <div>
                  <div className="text-[11px] text-slate-500">Expected Reference</div>
                  <div className="font-mono font-bold text-slate-900 tabular-nums">
                    {formatMetricVal(selectedAnomaly.variable_affected, selectedAnomaly.expected_value)}
                  </div>
                </div>
                <div>
                  <div className="text-[11px] text-slate-500">Deviation Magnitude</div>
                  <div className="font-mono font-bold text-amber-800 tabular-nums">
                    {selectedAnomaly.deviation_pct >= 0 ? "+" : ""}
                    {selectedAnomaly.deviation_pct.toFixed(1)}%
                  </div>
                </div>
              </div>

              <div>
                <div className="font-semibold text-slate-700">Supporting Variables</div>
                <div className="mt-1 grid grid-cols-2 gap-1.5 rounded-lg border border-slate-200 p-2.5 font-mono text-[11px] tabular-nums">
                  <div>FX Rate: ₦{selectedAnomaly.supporting_variables.USD_NGN_Planning_Rate}</div>
                  <div>FX MoM: {selectedAnomaly.supporting_variables.FX_Change_MoM_Pct}%</div>
                  <div>Inflation: {selectedAnomaly.supporting_variables.Inflation_YoY_Pct}%</div>
                  <div>Units: {selectedAnomaly.supporting_variables.Units_Sold.toLocaleString()}</div>
                  <div>Gross Margin: {selectedAnomaly.supporting_variables.Gross_Margin_Pct}%</div>
                  <div>Op Margin: {selectedAnomaly.supporting_variables.Operating_Margin_Pct}%</div>
                </div>
              </div>

              <div>
                <div className="font-semibold text-slate-700">Possible Business Interpretation</div>
                <p className="mt-0.5 text-slate-600 leading-relaxed">
                  {selectedAnomaly.interpretation}
                </p>
              </div>

              <div>
                <div className="font-semibold text-slate-700">Recommended Investigation</div>
                <p className="mt-0.5 text-slate-600 leading-relaxed">
                  {selectedAnomaly.recommended_investigation}
                </p>
              </div>

              <div className="rounded-lg border border-slate-200 bg-slate-50 p-2.5 text-[11px] text-slate-700">
                <strong>Statistical vs. Business Problem Distinction:</strong>{" "}
                {selectedAnomaly.statistical_vs_business_note}
              </div>
            </div>
          ) : (
            <div className="py-12 text-center text-xs text-slate-500">
              Select an anomaly from the list to inspect full evidence.
            </div>
          )}
        </div>
      </div>

      {/* Evidence-Based Analytical Findings (6 Core Questions) */}
      <div className="rounded-xl border border-slate-200 bg-white p-5">
        <div className="flex items-center gap-2">
          <SearchCheck className="h-4 w-4 text-[#0A2E65]" />
          <h2 className="font-display text-sm font-bold text-[#0A1E3D]">
            Evidence-Based Analytical Findings
          </h2>
        </div>
        <div className="mt-3 grid grid-cols-1 gap-3 md:grid-cols-2">
          {data.analytical_findings.map((f) => (
            <div key={f.id} className="rounded-lg border border-slate-200 bg-slate-50/70 p-3.5 text-xs">
              <div className="font-bold text-[#0A1E3D]">{f.question}</div>
              <p className="mt-1 leading-relaxed text-slate-700">{f.finding}</p>
              {f.evidence_months.length > 0 && (
                <div className="mt-1.5 text-[11px] font-medium text-teal-800">
                  Key periods: {f.evidence_months.join(" · ")}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      <GraphicalRagPanel
        defaultLens="anomaly_intelligence"
        selectedMonth={selectedAnomaly?.month || null}
        title="Anomaly Investigation Graphical RAG"
      />
    </div>
  );
};
