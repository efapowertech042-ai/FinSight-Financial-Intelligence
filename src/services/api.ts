import {
  Accounting5YrParams,
  Accounting5YrResponse,
  AnomalyReport,
  CustomScenarioVerdict,
  DatasetState,
  ExplainabilityResponse,
  FeaturesIntelligenceResponse,
  ForecastScenarioAssumptions,
  ForecastScenariosBundle,
  ForecastingCentreResponse,
  GraphicalRagResponse,
  HypothesisTestingBundle,
  ModelsSummaryResponse,
  MonitoringReport,
  OperationalVerdict,
  PredictionInputs,
  PredictionResponse,
  ScenarioResponse,
} from "../types/financial";

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function tryParseJson(raw: string): unknown | null {
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function isHtmlPayload(raw: string, contentType: string): boolean {
  const trimmed = raw.trimStart().toLowerCase();
  return (
    contentType.toLowerCase().includes("text/html") ||
    trimmed.startsWith("<!doctype") ||
    trimmed.startsWith("<html")
  );
}

async function requestJson<T>(url: string, options?: RequestInit): Promise<T> {
  const method = (options?.method || "GET").toUpperCase();
  const maxAttempts = method === "GET" ? 20 : 6;

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const res = await fetch(url, options);
    const contentType = res.headers.get("content-type") || "";
    const rawText = await res.text();
    const parsed = rawText ? tryParseJson(rawText) : null;
    const htmlReturned = isHtmlPayload(rawText, contentType);

    const isWarmingUp =
      res.status === 503 &&
      parsed !== null &&
      typeof parsed === "object" &&
      "warming_up" in parsed &&
      Boolean((parsed as { warming_up?: boolean }).warming_up);

    const isTransientProxyHtml =
      htmlReturned && (res.status === 502 || res.status === 503 || res.status === 504 || res.status >= 500);

    if ((isWarmingUp || isTransientProxyHtml) && attempt < maxAttempts - 1) {
      await sleep(Math.min(4000, 1200 + attempt * 400));
      continue;
    }

    if (!res.ok) {
      let detail = `Request failed (${res.status})`;
      if (parsed && typeof parsed === "object") {
        const obj = parsed as { detail?: string; error?: string };
        detail = obj.detail || obj.error || detail;
      } else if (htmlReturned) {
        detail = `Analytical service returned an unexpected HTML response (HTTP ${res.status}). Please retry in a moment.`;
      } else if (rawText.trim()) {
        detail = rawText.trim().slice(0, 220);
      }
      throw new Error(detail);
    }

    if (parsed === null) {
      if (htmlReturned) {
        throw new Error(
          "Analytical API returned HTML instead of JSON. Ensure the FinSight backend server is running and retry."
        );
      }
      throw new Error("Analytical API returned an invalid JSON payload.");
    }

    return parsed as T;
  }

  throw new Error("Analytical backend did not become ready in time. Please retry.");
}

export const api = {
  getDatasetState(params?: { startDate?: string; endDate?: string; quarter?: string }): Promise<DatasetState> {
    const qs = new URLSearchParams();
    if (params?.startDate) qs.set("start_date", params.startDate);
    if (params?.endDate) qs.set("end_date", params.endDate);
    if (params?.quarter && params.quarter !== "ALL") qs.set("quarter", params.quarter);
    const suffix = qs.toString() ? `?${qs.toString()}` : "";
    return requestJson<DatasetState>(`/api/py/dataset/state${suffix}`);
  },

  async uploadDataset(file: File): Promise<{ status: string; dataset_name: string }> {
    const form = new FormData();
    form.append("file", file);
    return requestJson<{ status: string; dataset_name: string }>("/api/py/dataset/upload", {
      method: "POST",
      body: form,
    });
  },

  resetDataset(): Promise<{ status: string; dataset_name: string }> {
    return requestJson<{ status: string; dataset_name: string }>("/api/py/dataset/reset", {
      method: "POST",
    });
  },

  getFeatureIntelligence(): Promise<FeaturesIntelligenceResponse> {
    return requestJson<FeaturesIntelligenceResponse>("/api/py/features/intelligence");
  },

  getModelsSummary(): Promise<ModelsSummaryResponse> {
    return requestJson<ModelsSummaryResponse>("/api/py/models/summary");
  },

  predictOperatingProfit(inputs: PredictionInputs): Promise<PredictionResponse> {
    return requestJson<PredictionResponse>("/api/py/models/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(inputs),
    });
  },

  getExplainability(): Promise<ExplainabilityResponse> {
    return requestJson<ExplainabilityResponse>("/api/py/explainability");
  },

  getAnomalies(): Promise<AnomalyReport> {
    return requestJson<AnomalyReport>("/api/py/anomalies");
  },

  simulateScenario(baseline: PredictionInputs, simulated: PredictionInputs): Promise<ScenarioResponse> {
    return requestJson<ScenarioResponse>("/api/py/scenario/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ baseline, simulated }),
    });
  },

  askGraphicalRag(payload: {
    question: string;
    lens: string;
    selected_month?: string | null;
  }): Promise<GraphicalRagResponse> {
    return requestJson<GraphicalRagResponse>("/api/rag/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  },

  getMonitoring(): Promise<MonitoringReport> {
    return requestJson<MonitoringReport>("/api/py/monitoring");
  },

  getForecastingCentre(): Promise<ForecastingCentreResponse> {
    return requestJson<ForecastingCentreResponse>("/api/py/forecasting/centre");
  },

  runHypothesisTests(alpha: number, splitYear: number): Promise<HypothesisTestingBundle> {
    return requestJson<HypothesisTestingBundle>("/api/py/forecasting/hypotheses", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ alpha, split_year: splitYear }),
    });
  },

  simulateForecastScenario(
    assumptions: ForecastScenarioAssumptions
  ): Promise<{
    scenarios: ForecastScenariosBundle;
    verdict: OperationalVerdict;
    custom_verdict: CustomScenarioVerdict;
  }> {
    return requestJson<{
      scenarios: ForecastScenariosBundle;
      verdict: OperationalVerdict;
      custom_verdict: CustomScenarioVerdict;
    }>("/api/py/forecasting/scenario", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(assumptions),
    });
  },

  simulateAccounting5Yr(
    params?: Partial<Accounting5YrParams> & { horizon_months?: number }
  ): Promise<Accounting5YrResponse> {
    return requestJson<Accounting5YrResponse>("/api/py/scenario/accounting-5yr", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params || {}),
    });
  },
};
