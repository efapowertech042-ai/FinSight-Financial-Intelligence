import React, { useEffect, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ArrowRight, Database, GitBranch, Network, Send, Table2 } from "lucide-react";
import { api } from "../services/api";
import { GraphicalRagResponse } from "../types/financial";
import { formatNGN } from "../utils/format";

interface GraphicalRagPanelProps {
  defaultLens?:
    | "profit_intelligence"
    | "fx_intelligence"
    | "cost_intelligence"
    | "sales_intelligence"
    | "anomaly_intelligence"
    | "forecast_intelligence";
  selectedMonth?: string | null;
  compact?: boolean;
  title?: string;
  externalTrigger?: {
    question: string;
    lens?:
      | "profit_intelligence"
      | "fx_intelligence"
      | "cost_intelligence"
      | "sales_intelligence"
      | "anomaly_intelligence"
      | "forecast_intelligence";
    seq: number;
  } | null;
}

const LENS_OPTIONS = [
  { id: "profit_intelligence", label: "Operating Profit" },
  { id: "fx_intelligence", label: "FX" },
  { id: "cost_intelligence", label: "Costs & COGS" },
  { id: "sales_intelligence", label: "Sales" },
  { id: "anomaly_intelligence", label: "Anomalies" },
  { id: "forecast_intelligence", label: "2030 Forecast" },
] as const;

const SUGGESTED_QUESTIONS = [
  "Why did operating profit change in Jun 2023 during FX depreciation?",
  "What is the 2030 forecast verdict and hypothesis p-value?",
  "Which months had the lowest operating profit or negative margin?",
  "How did FX movements affect import costs and profitability?",
  "Which variables contributed most to operating profit?",
  "Which months experienced unusual performance?",
];

export const GraphicalRagPanel: React.FC<GraphicalRagPanelProps> = ({
  defaultLens = "profit_intelligence",
  selectedMonth = null,
  compact = false,
  title = "Graphical RAG Evidence Engine",
  externalTrigger = null,
}) => {
  const [lens, setLens] = useState<string>(defaultLens);
  const [question, setQuestion] = useState<string>("");
  const [data, setData] = useState<GraphicalRagResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const runQuery = async (q: string, targetLens: string, monthOverride?: string | null) => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.askGraphicalRag({
        question: q,
        lens: targetLens,
        selected_month: monthOverride ?? selectedMonth,
      });
      setData(res);
      if (q.trim().length >= 4 && res.intent) {
        setLens(res.intent);
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to retrieve evidence.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    setLens(defaultLens);
    runQuery("", defaultLens, selectedMonth);
  }, [defaultLens, selectedMonth]);

  useEffect(() => {
    if (!externalTrigger || externalTrigger.seq <= 0) return;
    const nextLens = externalTrigger.lens || lens;
    setLens(nextLens);
    setQuestion(externalTrigger.question);
    runQuery(externalTrigger.question, nextLens, selectedMonth);
  }, [externalTrigger?.seq]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    runQuery(question, lens);
  };

  const formatAxisVal = (val: number, key: string) => {
    if (key.includes("Pct") || key.includes("margin")) return `${val.toFixed(0)}%`;
    if (key.includes("Units")) return `${(val / 1000).toFixed(0)}K`;
    if (Math.abs(val) >= 1e6) return `₦${(val / 1e6).toFixed(0)}M`;
    return `₦${val.toLocaleString()}`;
  };

  return (
    <section className="rounded-xl border border-slate-200 bg-white p-5">
      <div className="flex flex-col gap-3 border-b border-slate-100 pb-4 lg:flex-row lg:items-center lg:justify-between">
        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[#0A2E65]/10 text-[#0A2E65]">
            <Network className="h-4 w-4" />
          </div>
          <div>
            <h3 className="font-display text-sm font-bold text-[#0A1E3D]">{title}</h3>
            <p className="text-xs text-slate-500">
              Multi-store dataset lookup, causal evidence graph, and step-by-step retrieval audit
              {data?.synthesis_mode ? ` · ${data.synthesis_mode}` : ""}
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-1 rounded-lg bg-slate-100 p-1">
          {LENS_OPTIONS.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => {
                setLens(item.id);
                runQuery(question, item.id);
              }}
              className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors whitespace-nowrap ${
                lens === item.id
                  ? "bg-white text-[#0A1E3D] shadow-sm"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>

      {/* Query input & Quick analytical prompts */}
      <form onSubmit={handleSubmit} className="mt-4 flex flex-col gap-2 sm:flex-row">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask about any month (Jun 2023), Market Notes keyword (depreciation), 2030 forecast, or hypothesis p-value..."
          className="flex-1 rounded-lg border border-slate-200 bg-slate-50 px-3.5 py-2 text-sm text-slate-900 placeholder:text-slate-400 focus:border-[#0A2E65] focus:bg-white focus:outline-none"
        />
        <button
          type="submit"
          disabled={loading}
          className="inline-flex items-center justify-center gap-1.5 rounded-lg bg-[#0A2E65] px-4 py-2 text-xs font-semibold text-white hover:bg-[#0D3B82] transition-colors disabled:opacity-60 whitespace-nowrap"
        >
          <Send className="h-3.5 w-3.5" />
          <span>{loading ? "Retrieving..." : "Retrieve & Explain"}</span>
        </button>
      </form>

      <div className="mt-2.5 flex flex-wrap gap-1.5">
        {SUGGESTED_QUESTIONS.map((sq) => (
          <button
            key={sq}
            type="button"
            onClick={() => {
              setQuestion(sq);
              runQuery(sq, lens);
            }}
            className="rounded-md border border-slate-200 bg-slate-50 px-2.5 py-1 text-[11px] font-medium text-slate-600 hover:border-[#0A2E65]/40 hover:bg-white hover:text-[#0A1E3D] transition-colors text-left"
          >
            {sq}
          </button>
        ))}
      </div>

      {error && (
        <div className="mt-4 rounded-lg border border-red-200 bg-red-50 p-3 text-xs text-red-700">
          {error}
        </div>
      )}

      {data && !loading && (
        <div className="mt-5 space-y-5">
          {/* Data Provenance & Stores Consulted Strip */}
          {data.data_provenance && (
            <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3.5 py-2.5 text-[11px] text-slate-600">
              <div className="flex flex-wrap items-center gap-2">
                <Database className="h-3.5 w-3.5 text-[#0A2E65]" />
                <span>
                  Source: <strong className="text-[#0A1E3D]">{data.data_provenance.dataset_name}</strong>
                </span>
                <span aria-hidden="true">·</span>
                <span className="tabular-nums">
                  {data.data_provenance.records_scanned} months scanned (
                  {data.data_provenance.date_start.slice(0, 7)} to{" "}
                  {data.data_provenance.date_end.slice(0, 7)})
                </span>
              </div>
              <div className="text-[11px] text-slate-500">
                Stores:{" "}
                <span className="font-medium text-slate-700">
                  {data.data_provenance.stores_consulted.join(" · ")}
                </span>
              </div>
            </div>
          )}

          {/* 5-Step Retrieval & Process Explanation Trace */}
          {data.retrieval_process && data.retrieval_process.length > 0 && (
            <div className="rounded-lg border border-slate-200 bg-white p-4">
              <div className="mb-3 flex items-center gap-2 text-xs font-bold text-[#0A1E3D]">
                <GitBranch className="h-3.5 w-3.5 text-teal-700" />
                <span>How Graphical RAG Executed Lookup &amp; Reasoning (Process Trace)</span>
              </div>
              <div className="grid grid-cols-1 gap-2.5 md:grid-cols-5">
                {data.retrieval_process.map((step) => (
                  <div
                    key={step.step}
                    className="rounded-lg border border-slate-200 bg-slate-50/70 p-2.5"
                  >
                    <div className="flex items-center gap-1.5">
                      <span className="font-mono text-[10px] font-bold text-[#0A2E65] tabular-nums">
                        0{step.step}
                      </span>
                      <span className="text-[11px] font-bold text-[#0A1E3D] leading-tight">
                        {step.stage}
                      </span>
                    </div>
                    <p className="mt-1.5 text-[11px] leading-snug text-slate-600">{step.detail}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Knowledge Retrieval Graph Flow */}
          <div className="rounded-lg border border-slate-200 bg-slate-50/70 p-4">
            <div className="mb-2.5 text-xs font-semibold text-slate-700">
              Retrieved Evidence Graph Path
            </div>
            <div className="flex flex-wrap items-center gap-2">
              {data.graph.nodes.map((node, idx) => (
                <React.Fragment key={node.id}>
                  <div className="rounded-lg border border-slate-200 bg-white px-3 py-2">
                    <div className="text-[11px] font-medium text-slate-500">{node.role}</div>
                    <div className="text-xs font-bold text-[#0A1E3D]">{node.label}</div>
                    {node.metric && (
                      <div className="mt-0.5 font-mono text-[11px] text-teal-700 tabular-nums">
                        {node.metric}
                      </div>
                    )}
                  </div>
                  {idx < data.graph.nodes.length - 1 && (
                    <div className="flex items-center gap-1 text-slate-400">
                      <ArrowRight className="h-3.5 w-3.5" />
                    </div>
                  )}
                </React.Fragment>
              ))}
            </div>
          </div>

          <div className={`grid gap-5 ${compact ? "lg:grid-cols-1" : "lg:grid-cols-12"}`}>
            {/* Chart */}
            <div className={compact ? "" : "lg:col-span-7"}>
              <div className="mb-2 text-xs font-semibold text-slate-700">{data.chart.title}</div>
              <div className="h-[260px] w-full rounded-lg border border-slate-100 bg-white p-2">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={data.chart.data} margin={{ top: 8, right: 16, left: 6, bottom: 4 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                    <XAxis
                      dataKey={data.chart.x_key}
                      tick={{ fontSize: 10, fill: "#64748B" }}
                      minTickGap={24}
                    />
                    <YAxis
                      yAxisId="left"
                      tick={{ fontSize: 10, fill: "#64748B" }}
                      tickFormatter={(v) => formatAxisVal(v, data.chart.series[0]?.key || "")}
                      width={58}
                    />
                    {data.chart.series.some((s) => s.axis === "right") && (
                      <YAxis
                        yAxisId="right"
                        orientation="right"
                        tick={{ fontSize: 10, fill: "#64748B" }}
                        tickFormatter={(v) =>
                          formatAxisVal(
                            v,
                            data.chart.series.find((s) => s.axis === "right")?.key || ""
                          )
                        }
                        width={54}
                      />
                    )}
                    <Tooltip
                      contentStyle={{
                        fontSize: "12px",
                        borderRadius: "8px",
                        border: "1px solid #E2E8F0",
                      }}
                      formatter={(val: any, name: any) => [
                        typeof val === "number"
                          ? String(name).includes("%")
                            ? `${val.toFixed(2)}%`
                            : String(name).includes("Units")
                              ? val.toLocaleString()
                              : formatNGN(val)
                          : val,
                        name,
                      ]}
                    />
                    <Legend wrapperStyle={{ fontSize: "11px" }} />
                    {data.chart.series.map((s) => (
                      <Line
                        key={s.key}
                        yAxisId={s.axis}
                        type="monotone"
                        dataKey={s.key}
                        name={s.label}
                        stroke={s.color}
                        strokeWidth={2}
                        dot={false}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Retrieved Evidence & Grounded Explanation */}
            <div className={`${compact ? "" : "lg:col-span-5"} flex flex-col gap-3`}>
              <div className="rounded-lg border border-slate-200 bg-slate-50 p-3.5">
                <div className="text-xs font-bold text-[#0A1E3D]">Analytical Explanation</div>
                <p className="mt-1 text-xs leading-relaxed text-slate-700">
                  {data.llm_narrative || data.analytical_explanation}
                </p>
                <div className="mt-2.5 text-xs font-bold text-teal-800">Business Implications</div>
                <p className="mt-0.5 text-xs leading-relaxed text-slate-700">
                  {data.business_implications}
                </p>
              </div>

              <div className="space-y-2">
                <div className="text-xs font-semibold text-slate-700">
                  Retrieved Supporting Evidence ({data.evidence.length})
                </div>
                {data.evidence.slice(0, 4).map((ev, i) => (
                  <div
                    key={`${ev.month}-${i}`}
                    className="rounded-lg border border-slate-200 bg-white px-3 py-2.5"
                  >
                    <div className="flex items-center justify-between gap-2 text-[11px] text-slate-500">
                      <span className="font-semibold text-[#0A2E65]">{ev.type}</span>
                      <span className="font-mono tabular-nums">{ev.month_label}</span>
                    </div>
                    <div className="mt-0.5 text-xs font-semibold text-slate-900">
                      {ev.headline}
                    </div>
                    <div className="mt-0.5 text-[11px] leading-snug text-slate-600">
                      {ev.detail}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Exact Looked-Up Original Dataset Rows */}
          {data.matched_records && data.matched_records.length > 0 && (
            <div className="rounded-lg border border-slate-200 bg-white p-4">
              <div className="mb-2.5 flex items-center justify-between gap-2">
                <div className="flex items-center gap-2 text-xs font-bold text-[#0A1E3D]">
                  <Table2 className="h-3.5 w-3.5 text-[#0A2E65]" />
                  <span>Matched Dataset Ledger Rows (Exact Source Lookup)</span>
                </div>
                <span className="text-[11px] text-slate-500 tabular-nums">
                  {data.matched_records.length} anchor observations retrieved
                </span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-[11px]">
                  <thead>
                    <tr className="border-b border-slate-200 text-slate-500">
                      <th className="py-2 pr-3 font-semibold whitespace-nowrap">Month</th>
                      <th className="py-2 pr-3 font-semibold">Match Criterion</th>
                      <th className="py-2 pr-3 text-right font-semibold whitespace-nowrap">USD/NGN</th>
                      <th className="py-2 pr-3 text-right font-semibold whitespace-nowrap">Units</th>
                      <th className="py-2 pr-3 text-right font-semibold whitespace-nowrap">Price</th>
                      <th className="py-2 pr-3 text-right font-semibold whitespace-nowrap">Revenue</th>
                      <th className="py-2 pr-3 text-right font-semibold whitespace-nowrap">COGS</th>
                      <th className="py-2 pr-3 text-right font-semibold whitespace-nowrap">Op. Profit</th>
                      <th className="py-2 pr-3 text-right font-semibold whitespace-nowrap">Margin</th>
                      <th className="py-2 font-semibold">Market Notes</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {(compact ? data.matched_records.slice(0, 3) : data.matched_records).map((row) => (
                      <tr key={row.month} className="hover:bg-slate-50/70">
                        <td className="py-2 pr-3 font-mono font-semibold text-[#0A1E3D] whitespace-nowrap">
                          {row.month_label}
                        </td>
                        <td className="py-2 pr-3 text-slate-600">{row.match_reason}</td>
                        <td className="py-2 pr-3 text-right font-mono tabular-nums whitespace-nowrap">
                          ₦{row.usd_ngn_rate.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                        </td>
                        <td className="py-2 pr-3 text-right font-mono tabular-nums whitespace-nowrap">
                          {row.units_sold.toLocaleString()}
                        </td>
                        <td className="py-2 pr-3 text-right font-mono tabular-nums whitespace-nowrap">
                          ₦{row.avg_selling_price_ngn.toLocaleString()}
                        </td>
                        <td className="py-2 pr-3 text-right font-mono tabular-nums whitespace-nowrap">
                          {formatNGN(row.sales_revenue_ngn, true)}
                        </td>
                        <td className="py-2 pr-3 text-right font-mono tabular-nums whitespace-nowrap">
                          {formatNGN(row.cogs_ngn, true)}
                        </td>
                        <td
                          className={`py-2 pr-3 text-right font-mono font-semibold tabular-nums whitespace-nowrap ${
                            row.operating_profit_ngn >= 0 ? "text-teal-700" : "text-red-600"
                          }`}
                        >
                          {formatNGN(row.operating_profit_ngn, true)}
                        </td>
                        <td className="py-2 pr-3 text-right font-mono tabular-nums whitespace-nowrap">
                          {row.operating_margin_pct.toFixed(1)}%
                        </td>
                        <td className="py-2 text-slate-600 max-w-[220px]">
                          {row.market_notes || "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <div className="border-t border-slate-100 pt-2.5 text-[11px] text-slate-500">
            <span className="font-semibold text-slate-600">Data &amp; Model Limitations:</span>{" "}
            {data.limitations}
          </div>
        </div>
      )}
    </section>
  );
};
