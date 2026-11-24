import React from "react";
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
import { Activity, Download, FileSpreadsheet, FileText, SlidersHorizontal } from "lucide-react";
import {
  AnomalyReport,
  DatasetState,
  ExplainabilityResponse,
  ModelsSummaryResponse,
  MonitoringReport,
} from "../types/financial";
import { formatNGN, formatPct } from "../utils/format";
import { useScenario } from "../context/ScenarioContext";

interface ReportsMonitoringViewProps {
  dataset: DatasetState;
  models: ModelsSummaryResponse | null;
  explainability: ExplainabilityResponse | null;
  anomalies: AnomalyReport | null;
  monitoring: MonitoringReport | null;
}

export const ReportsMonitoringView: React.FC<ReportsMonitoringViewProps> = ({
  dataset,
  models,
  explainability,
  anomalies,
  monitoring,
}) => {
  const {
    activePresetLabel,
    driverDeltas,
    forecastAssumptions,
    forecastScenarios,
    customVerdict,
    singlePeriodResult,
    setDrawerOpen,
  } = useScenario();

  const custom2030 = forecastScenarios?.cases?.custom?.summary;

  const handleDownloadPdf = async () => {
    const [{ default: jsPDF }, { default: autoTable }] = await Promise.all([
      import("jspdf"),
      import("jspdf-autotable"),
    ]);
    const doc = new jsPDF({ orientation: "portrait", unit: "pt", format: "a4" });
    doc.setFontSize(16);
    doc.setTextColor(10, 30, 61);
    doc.text("FinSight Financial Intelligence — Executive Analytical Report", 40, 44);

    doc.setFontSize(9.5);
    doc.setTextColor(80, 95, 115);
    doc.text(
      `Dataset: ${dataset.dataset_name} | Observations: ${dataset.kpis.period_count} months | Generated: ${new Date().toISOString().slice(0, 10)}`,
      40,
      62
    );

    autoTable(doc, {
      startY: 78,
      head: [["Executive KPI", "Value"]],
      body: [
        ["Total Sales Revenue", formatNGN(dataset.kpis.total_sales_revenue)],
        ["Total Operating Profit", formatNGN(dataset.kpis.total_operating_profit)],
        ["Average Operating Margin", formatPct(dataset.kpis.avg_operating_margin_pct)],
        ["Total Gross Profit", formatNGN(dataset.kpis.total_gross_profit)],
        ["Weighted Gross Margin", formatPct(dataset.kpis.gross_margin_pct)],
        ["Total Units Sold", dataset.kpis.total_units_sold.toLocaleString()],
        ["Total Marketing Spend", formatNGN(dataset.kpis.total_marketing_spend)],
        ["Total Import Costs (COGS)", formatNGN(dataset.kpis.total_import_costs)],
      ],
      theme: "grid",
      headStyles: { fillColor: [10, 46, 101] },
      styles: { fontSize: 8.5 },
    });

    const yAfterKpi = (doc as any).lastAutoTable.finalY + 18;
    if (models?.driver_suite?.models?.length) {
      autoTable(doc, {
        startY: yAfterKpi,
        head: [["ML Algorithm (Out-of-Time)", "Test MAE", "Test RMSE", "Test R²", "MAPE %"]],
        body: models.driver_suite.models.map((m) => [
          m.name,
          formatNGN(m.test_metrics.mae),
          formatNGN(m.test_metrics.rmse),
          String(m.test_metrics.r2 ?? "—"),
          m.test_metrics.mape_pct !== null ? `${m.test_metrics.mape_pct}%` : "—",
        ]),
        theme: "grid",
        headStyles: { fillColor: [13, 148, 136] },
        styles: { fontSize: 8.5 },
      });
    }

    const yAfterModels = (doc as any).lastAutoTable.finalY + 18;
    if (explainability?.global_importance?.length) {
      autoTable(doc, {
        startY: yAfterModels,
        head: [["Top SHAP Feature", "SHAP Share %", "Mean |SHAP| (NGN)", "Direction"]],
        body: explainability.global_importance.slice(0, 6).map((g) => [
          g.feature,
          `${g.share_pct}%`,
          formatNGN(g.mean_abs_shap),
          g.direction,
        ]),
        theme: "grid",
        headStyles: { fillColor: [10, 46, 101] },
        styles: { fontSize: 8 },
      });
    }

    doc.addPage();
    doc.setFontSize(13);
    doc.setTextColor(10, 30, 61);
    doc.text("Detected Anomalies & Business Recommendations", 40, 44);

    if (anomalies?.anomalies?.length) {
      autoTable(doc, {
        startY: 60,
        head: [["Month", "Variable Affected", "Severity", "Deviation %", "Interpretation"]],
        body: anomalies.anomalies.slice(0, 12).map((a) => [
          a.month_label,
          a.variable_label,
          a.severity,
          `${a.deviation_pct >= 0 ? "+" : ""}${a.deviation_pct.toFixed(1)}%`,
          a.interpretation.slice(0, 110),
        ]),
        theme: "grid",
        headStyles: { fillColor: [217, 119, 6] },
        styles: { fontSize: 7.5 },
      });
    }

    const yEnd = ((doc as any).lastAutoTable?.finalY || 120) + 22;
    doc.setFontSize(10);
    doc.setTextColor(15, 23, 42);
    doc.text("Management Recommendations:", 40, yEnd);
    doc.setFontSize(8.5);
    const recs = [
      "1. Align pricing adjustment cadence with USD/NGN movements (>5% MoM shifts) to avoid gross margin compression.",
      "2. Monitor unit price-to-landed import cost ratio closely; protect operating margin floor above 25%.",
      "3. Audit Q4 promotional pricing discounts to ensure volume spikes translate into net operating profit expansion.",
      "4. Retrain operating profit prediction models quarterly as new post-stabilization months close.",
    ];
    recs.forEach((line, idx) => {
      doc.text(line, 40, yEnd + 16 + idx * 14);
    });

    if (custom2030 || singlePeriodResult) {
      const ySim = yEnd + 16 + recs.length * 14 + 18;
      autoTable(doc, {
        startY: ySim,
        head: [[`Active Scenario Summary (${activePresetLabel})`, "Simulated Metric"]],
        body: [
          ...(singlePeriodResult
            ? [
                [
                  "1-Month Simulated ML Operating Profit",
                  `${formatNGN(singlePeriodResult.simulated.ml_predicted_operating_profit_ngn)} (Delta: ${
                    singlePeriodResult.deltas.ml_predicted_profit_delta >= 0 ? "+" : ""
                  }${formatNGN(singlePeriodResult.deltas.ml_predicted_profit_delta)})`,
                ],
                [
                  "1-Month Driver Shocks (FX / Price / Units)",
                  `FX ${driverDeltas.fxPct >= 0 ? "+" : ""}${driverDeltas.fxPct.toFixed(1)}% | Price ${
                    driverDeltas.pricePct >= 0 ? "+" : ""
                  }${driverDeltas.pricePct.toFixed(1)}% | Units ${driverDeltas.unitsPct >= 0 ? "+" : ""}${driverDeltas.unitsPct.toFixed(1)}%`,
                ],
              ]
            : []),
          ...(custom2030
            ? [
                ["2030 Simulated Dec Monthly Operating Profit", formatNGN(custom2030.dec_2030_operating_profit_ngn)],
                ["2026–2030 Cumulative Simulated Profit", formatNGN(custom2030.cumulative_profit_ngn)],
                [
                  "2026–2030 Simulated Avg Operating Margin",
                  `${(custom2030.cumulative_margin_pct ?? custom2030.year_2030_operating_margin_pct).toFixed(2)}%`,
                ],
              ]
            : []),
          ...(customVerdict ? [["Simulated 2030 Governance Verdict", `${customVerdict.category} — ${customVerdict.headline}`]] : []),
        ],
        theme: "grid",
        headStyles: { fillColor: [10, 30, 61] },
        styles: { fontSize: 8 },
      });
    }

    doc.save(`FinSight_Executive_Report_${dataset.dataset_name.replace(/\.[^/.]+$/, "")}.pdf`);
  };

  return (
    <div className="space-y-6">
      {/* Export Controls */}
      <div className="rounded-xl border border-slate-200 bg-white p-5 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-start gap-3">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-[#0A2E65]/10 text-[#0A2E65]">
            <FileText className="h-5 w-5" />
          </div>
          <div>
            <h2 className="font-display text-sm font-bold text-[#0A1E3D]">
              Executive Financial &amp; ML Analytical Reports
            </h2>
            <p className="mt-0.5 text-xs text-slate-600">
              Export the full engineered dataset (CSV) or generate a structured multi-section PDF report containing KPIs,
              chronological model metrics, SHAP drivers, anomalies, and business recommendations.
            </p>
          </div>
        </div>
        <div className="flex flex-wrap gap-2.5 shrink-0">
          <a
            href="/api/py/export/csv"
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3.5 py-2 text-xs font-semibold text-slate-800 hover:bg-slate-50 whitespace-nowrap"
          >
            <FileSpreadsheet className="h-3.5 w-3.5 text-teal-700" />
            <span>Download CSV Dataset</span>
          </a>
          <button
            type="button"
            onClick={handleDownloadPdf}
            className="inline-flex items-center gap-1.5 rounded-lg bg-[#0A2E65] px-4 py-2 text-xs font-semibold text-white hover:bg-[#0D3B82] whitespace-nowrap"
          >
            <Download className="h-3.5 w-3.5" />
            <span>Download PDF Report</span>
          </button>
        </div>
      </div>

      {/* Active Scenario Governance & Drift Readiness */}
      <div className="rounded-xl border border-slate-200 bg-white p-5">
        <div className="flex flex-col gap-3 border-b border-slate-100 pb-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h3 className="font-display text-sm font-bold text-[#0A1E3D]">
              Active Scenario Governance &amp; Drift Readiness ({activePresetLabel})
            </h3>
            <p className="mt-0.5 text-xs text-slate-500">
              Compares active simulated shocks against historical feature drift thresholds and is included in PDF exports.
            </p>
          </div>
          <button
            type="button"
            onClick={() => setDrawerOpen(true)}
            className="inline-flex items-center gap-1.5 self-start rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs font-semibold text-[#0A1E3D] hover:bg-slate-100 whitespace-nowrap sm:self-auto"
          >
            <SlidersHorizontal className="h-3.5 w-3.5 text-[#0A2E65]" />
            <span>Tune Levers</span>
          </button>
        </div>

        <div className="mt-3 grid grid-cols-1 gap-3 md:grid-cols-4">
          <div className="rounded-lg border border-slate-200 bg-slate-50/70 p-3">
            <div className="text-[11px] font-medium text-slate-500">Simulated FX / Units / Price</div>
            <div className="mt-1 font-mono text-xs font-bold text-[#0A1E3D] tabular-nums">
              {driverDeltas.fxPct >= 0 ? "+" : ""}
              {driverDeltas.fxPct.toFixed(1)}% FX · {driverDeltas.unitsPct >= 0 ? "+" : ""}
              {driverDeltas.unitsPct.toFixed(1)}% Units · {driverDeltas.pricePct >= 0 ? "+" : ""}
              {driverDeltas.pricePct.toFixed(1)}% Price
            </div>
            <div className="mt-1 text-[11px] text-slate-500 tabular-nums">
              2030 Rev Adj: {forecastAssumptions.revenue_growth_adj_pp >= 0 ? "+" : ""}
              {forecastAssumptions.revenue_growth_adj_pp.toFixed(1)} pp/yr
            </div>
          </div>

          <div className="rounded-lg border border-slate-200 bg-slate-50/70 p-3">
            <div className="text-[11px] font-medium text-slate-500">1-Month ML Profit Impact</div>
            <div
              className={`mt-1 font-mono text-sm font-bold tabular-nums ${
                (singlePeriodResult?.deltas.ml_predicted_profit_delta ?? 0) >= 0 ? "text-teal-700" : "text-red-600"
              }`}
            >
              {singlePeriodResult
                ? `${singlePeriodResult.deltas.ml_predicted_profit_delta >= 0 ? "+" : ""}${formatNGN(
                    singlePeriodResult.deltas.ml_predicted_profit_delta,
                    true
                  )}`
                : "Pending"}
            </div>
            <div className="mt-1 text-[11px] text-slate-500 tabular-nums">
              {singlePeriodResult
                ? `Simulated: ${formatNGN(singlePeriodResult.simulated.ml_predicted_operating_profit_ngn, true)}`
                : "Loading scenario baseline"}
            </div>
          </div>

          <div className="rounded-lg border border-slate-200 bg-slate-50/70 p-3">
            <div className="text-[11px] font-medium text-slate-500">2030 Cumulative Profit</div>
            <div className="mt-1 font-mono text-sm font-bold text-[#0A1E3D] tabular-nums">
              {custom2030 ? formatNGN(custom2030.cumulative_profit_ngn, true) : "Pending"}
            </div>
            <div className="mt-1 text-[11px] text-slate-500 tabular-nums">
              {custom2030
                ? `Avg Margin: ${(custom2030.cumulative_margin_pct ?? custom2030.year_2030_operating_margin_pct).toFixed(1)}%`
                : "2026–2030 Horizon"}
            </div>
          </div>

          <div className="rounded-lg border border-slate-200 bg-slate-50/70 p-3">
            <div className="text-[11px] font-medium text-slate-500">2030 Simulated Verdict</div>
            <div className="mt-1 text-xs font-bold text-[#0A2E65]">
              {customVerdict ? customVerdict.category : "Base Governance"}
            </div>
            <div className="mt-1 text-[11px] text-slate-500">
              {monitoring
                ? `${monitoring.feature_drift.filter((d) => d.status === "High Drift").length} historical High Drift features`
                : "Drift check active"}
            </div>
          </div>
        </div>
      </div>

      {/* Model Monitoring & Drift Diagnostics */}
      {monitoring && (
        <>
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
            <div className="rounded-xl border border-slate-200 bg-white p-4">
              <div className="text-xs font-medium text-slate-500">Active Production Model</div>
              <div className="mt-1 font-display text-base font-bold text-[#0A1E3D]">
                {monitoring.selected_model}
              </div>
              <div className="mt-1 text-xs text-slate-600 tabular-nums">
                Out-of-Time RMSE: {formatNGN(monitoring.test_metrics?.rmse, true)} · MAPE:{" "}
                {monitoring.test_metrics?.mape_pct ?? "—"}%
              </div>
            </div>

            <div className="rounded-xl border border-slate-200 bg-white p-4">
              <div className="text-xs font-medium text-slate-500">Data Quality Health</div>
              <div className="mt-1 font-display text-base font-bold text-teal-700 tabular-nums">
                {monitoring.data_quality.records_processed} Verified Months
              </div>
              <div className="mt-1 text-xs text-slate-600 tabular-nums">
                Missing: {monitoring.data_quality.missing_values_total} · Material errors:{" "}
                {monitoring.data_quality.material_discrepancies}
              </div>
            </div>

            <div className="rounded-xl border border-slate-200 bg-white p-4">
              <div className="text-xs font-medium text-slate-500">Retraining Assessment</div>
              <div
                className={`mt-1 font-display text-base font-bold ${
                  monitoring.retraining_assessment.recommended ? "text-amber-700" : "text-emerald-700"
                }`}
              >
                {monitoring.retraining_assessment.recommended
                  ? "Periodic Retraining Recommended"
                  : "Nominal Performance"}
              </div>
              <div className="mt-1 text-xs text-slate-600">
                {monitoring.retraining_assessment.rationale}
              </div>
            </div>
          </div>

          {/* Residual Error Pattern Over Full History */}
          <div className="rounded-xl border border-slate-200 bg-white p-5">
            <div className="flex items-center gap-2">
              <Activity className="h-4 w-4 text-[#0A2E65]" />
              <h3 className="font-display text-sm font-bold text-[#0A1E3D]">
                Chronological Model Residual Pattern (Train + Out-of-Time Test)
              </h3>
            </div>
            <div className="mt-3 h-[220px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={monitoring.residual_series}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                  <XAxis dataKey="month_label" tick={{ fontSize: 9, fill: "#64748B" }} minTickGap={24} />
                  <YAxis
                    tick={{ fontSize: 10, fill: "#64748B" }}
                    tickFormatter={(v) => `₦${(v / 1e6).toFixed(0)}M`}
                    width={52}
                  />
                  <Tooltip formatter={(val: any) => formatNGN(Number(val))} />
                  <ReferenceLine y={0} stroke="#64748B" />
                  <Bar dataKey="residual" name="Prediction Error (Actual - Pred)">
                    {monitoring.residual_series.map((r) => (
                      <Cell key={r.month} fill={r.residual >= 0 ? "#0D9488" : "#D97706"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Feature Drift Monitor */}
          <div className="rounded-xl border border-slate-200 bg-white p-5">
            <h3 className="font-display text-sm font-bold text-[#0A1E3D]">
              Feature Drift Monitoring (Early 70% Reference Window vs. Recent 30% Window)
            </h3>
            <div className="mt-3 overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-200 text-slate-500">
                    <th className="py-2 pr-4 font-semibold">Feature</th>
                    <th className="py-2 pr-4 text-right font-semibold">Reference Mean</th>
                    <th className="py-2 pr-4 text-right font-semibold">Recent Mean</th>
                    <th className="py-2 pr-4 text-right font-semibold">% Shift</th>
                    <th className="py-2 pr-4 text-right font-semibold">Std. Z-Shift</th>
                    <th className="py-2 text-right font-semibold">Drift Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {monitoring.feature_drift.map((d) => (
                    <tr key={d.feature}>
                      <td className="py-2.5 pr-4 font-mono font-semibold text-[#0A1E3D]">{d.feature}</td>
                      <td className="py-2.5 pr-4 text-right font-mono tabular-nums">
                        {d.reference_mean.toLocaleString()}
                      </td>
                      <td className="py-2.5 pr-4 text-right font-mono tabular-nums">
                        {d.recent_mean.toLocaleString()}
                      </td>
                      <td className="py-2.5 pr-4 text-right font-mono tabular-nums">
                        {d.pct_change >= 0 ? "+" : ""}
                        {d.pct_change.toFixed(1)}%
                      </td>
                      <td className="py-2.5 pr-4 text-right font-mono tabular-nums">
                        {d.standardized_shift.toFixed(2)}
                      </td>
                      <td className="py-2.5 text-right font-semibold">
                        <span
                          className={
                            d.status === "High Drift"
                              ? "text-red-600"
                              : d.status === "Moderate Drift"
                                ? "text-amber-700"
                                : "text-teal-700"
                          }
                        >
                          {d.status}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
};
