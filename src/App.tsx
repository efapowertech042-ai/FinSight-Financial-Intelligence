import React, { Suspense, lazy, useCallback, useEffect, useState } from "react";
import { api } from "./services/api";
import {
  AnomalyReport,
  DatasetState,
  ExplainabilityResponse,
  FeaturesIntelligenceResponse,
  ModelsSummaryResponse,
  MonitoringReport,
} from "./types/financial";
import { SectionId, Sidebar } from "./components/Sidebar";
import { TopHeader } from "./components/TopHeader";
import { ExecutiveOverviewView } from "./views/ExecutiveOverviewView";
import { ScenarioProvider, useScenario } from "./context/ScenarioContext";
import { GlobalScenarioBar } from "./components/GlobalScenarioBar";
import { GlobalScenarioDrawer } from "./components/GlobalScenarioDrawer";

const FinancialPerformanceView = lazy(() =>
  import("./views/FinancialPerformanceView").then((m) => ({ default: m.FinancialPerformanceView }))
);
const DataEngineeringView = lazy(() =>
  import("./views/DataEngineeringView").then((m) => ({ default: m.DataEngineeringView }))
);
const PredictionView = lazy(() =>
  import("./views/PredictionView").then((m) => ({ default: m.PredictionView }))
);
const ExplainabilityView = lazy(() =>
  import("./views/ExplainabilityView").then((m) => ({ default: m.ExplainabilityView }))
);
const GraphicalRagView = lazy(() =>
  import("./views/GraphicalRagView").then((m) => ({ default: m.GraphicalRagView }))
);
const AnomalyDetectionView = lazy(() =>
  import("./views/AnomalyDetectionView").then((m) => ({ default: m.AnomalyDetectionView }))
);
const ScenarioSimulationView = lazy(() =>
  import("./views/ScenarioSimulationView").then((m) => ({ default: m.ScenarioSimulationView }))
);
const ReportsMonitoringView = lazy(() =>
  import("./views/ReportsMonitoringView").then((m) => ({ default: m.ReportsMonitoringView }))
);
const ForecastingCentreView = lazy(() =>
  import("./views/ForecastingCentreView").then((m) => ({ default: m.ForecastingCentreView }))
);

const SectionFallback: React.FC = () => (
  <div className="space-y-4">
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
      {[0, 1, 2, 3].map((idx) => (
        <div key={idx} className="h-24 animate-pulse rounded-xl border border-slate-200 bg-white" />
      ))}
    </div>
    <div className="h-72 animate-pulse rounded-xl border border-slate-200 bg-white" />
  </div>
);

const SECTION_TITLES: Record<SectionId, { title: string; subtitle: string }> = {
  executive: {
    title: "Financial Performance Dashboard",
    subtitle: "From Financial Data to Predictive Intelligence",
  },
  performance: {
    title: "Financial Performance & Macro Analytics",
    subtitle: "Revenue, margin compression, FX sensitivity, and monthly ledger drill-down",
  },
  engineering: {
    title: "Data Engineering & Feature Intelligence",
    subtitle: "Automated formula recalculation, integrity verification, and dependency graph",
  },
  prediction: {
    title: "Operating Profit Prediction Engine",
    subtitle: "Leakage-free chronological ML forecasting and accounting reconciliation",
  },
  forecasting: {
    title: "2030 Forecasting & Predictive Intelligence",
    subtitle: "Long-horizon forecasting through December 2030, hypothesis testing, and model validation",
  },
  explainability: {
    title: "Model Explainability & SHAP Intelligence",
    subtitle: "Global SHAP contributions, permutation importance, and dependence diagnostics",
  },
  rag: {
    title: "Graphical RAG Intelligence",
    subtitle: "Retrieval-grounded explanations backed by historical financial evidence",
  },
  anomalies: {
    title: "Anomaly Detection & Analytical Breakdown",
    subtitle: "Isolation Forest, Z-score, IQR, and time-series deviation diagnostics",
  },
  scenario: {
    title: "Scenario Simulation & What-If Analysis",
    subtitle: "Interactive macro/commercial stress testing and profit bridge waterfall",
  },
  reports: {
    title: "Reports & Model Monitoring",
    subtitle: "Downloadable CSV/PDF reports, feature drift, and retraining governance",
  },
};

export function App() {
  return <AppShell />;
}

function AppShell() {
  const [activeSection, setActiveSection] = useState<SectionId>("executive");
  const [mobileOpen, setMobileOpen] = useState(false);

  // Global filters
  const [quarterFilter, setQuarterFilter] = useState<string>("ALL");
  const [startDateFilter, setStartDateFilter] = useState<string>("");
  const [endDateFilter, setEndDateFilter] = useState<string>("");
  const [ragFocusMonth, setRagFocusMonth] = useState<string | null>(null);

  // Backend state
  const [datasetState, setDatasetState] = useState<DatasetState | null>(null);
  const [featureIntel, setFeatureIntel] = useState<FeaturesIntelligenceResponse | null>(null);
  const [modelsSummary, setModelsSummary] = useState<ModelsSummaryResponse | null>(null);
  const [explainability, setExplainability] = useState<ExplainabilityResponse | null>(null);
  const [anomalies, setAnomalies] = useState<AnomalyReport | null>(null);
  const [monitoring, setMonitoring] = useState<MonitoringReport | null>(null);

  const [loading, setLoading] = useState<boolean>(true);
  const [uploading, setUploading] = useState<boolean>(false);
  const [errorBanner, setErrorBanner] = useState<string | null>(null);

  const fetchFilteredDataset = useCallback(async () => {
    try {
      const ds = await api.getDatasetState({
        startDate: startDateFilter || undefined,
        endDate: endDateFilter || undefined,
        quarter: quarterFilter,
      });
      setDatasetState(ds);
    } catch (err: unknown) {
      setErrorBanner(err instanceof Error ? err.message : "Failed to load filtered dataset.");
    }
  }, [startDateFilter, endDateFilter, quarterFilter]);

  const fetchAllAnalytics = useCallback(async () => {
    setLoading(true);
    setErrorBanner(null);
    try {
      const [ds, fi, ms, ex, an, mon] = await Promise.all([
        api.getDatasetState({
          startDate: startDateFilter || undefined,
          endDate: endDateFilter || undefined,
          quarter: quarterFilter,
        }),
        api.getFeatureIntelligence(),
        api.getModelsSummary(),
        api.getExplainability(),
        api.getAnomalies(),
        api.getMonitoring(),
      ]);
      setDatasetState(ds);
      setFeatureIntel(fi);
      setModelsSummary(ms);
      setExplainability(ex);
      setAnomalies(an);
      setMonitoring(mon);
      const rangeStart = ds.available_date_range?.start || "";
      const rangeEnd = ds.available_date_range?.end || "";
      setStartDateFilter(rangeStart);
      setEndDateFilter(rangeEnd);
    } catch (err: unknown) {
      setErrorBanner(err instanceof Error ? err.message : "Failed to initialize analytical engine.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAllAnalytics();
  }, [fetchAllAnalytics]);

  useEffect(() => {
    if (!loading) {
      fetchFilteredDataset();
    }
  }, [quarterFilter, startDateFilter, endDateFilter]);

  const handleUploadFile = async (file: File) => {
    setUploading(true);
    setErrorBanner(null);
    try {
      await api.uploadDataset(file);
      setStartDateFilter("");
      setEndDateFilter("");
      setQuarterFilter("ALL");
      await fetchAllAnalytics();
    } catch (err: unknown) {
      setErrorBanner(err instanceof Error ? err.message : "Failed to process uploaded dataset.");
    } finally {
      setUploading(false);
    }
  };

  const handleResetDataset = async () => {
    setUploading(true);
    setErrorBanner(null);
    try {
      await api.resetDataset();
      setStartDateFilter("");
      setEndDateFilter("");
      setQuarterFilter("ALL");
      await fetchAllAnalytics();
    } catch (err: unknown) {
      setErrorBanner(err instanceof Error ? err.message : "Failed to restore default dataset.");
    } finally {
      setUploading(false);
    }
  };

  const handleSelectMonthForRag = (monthIso: string) => {
    setRagFocusMonth(monthIso);
    setActiveSection("rag");
  };

  const datasetKey = `${datasetState?.dataset_name || "default"}_${datasetState?.quality_summary?.sha256_short || ""}`;

  return (
    <ScenarioProvider latestRecord={datasetState?.latest_record || null} datasetKey={datasetKey}>
      <AppWorkspace
        activeSection={activeSection}
        setActiveSection={setActiveSection}
        mobileOpen={mobileOpen}
        setMobileOpen={setMobileOpen}
        quarterFilter={quarterFilter}
        setQuarterFilter={setQuarterFilter}
        startDateFilter={startDateFilter}
        setStartDateFilter={setStartDateFilter}
        endDateFilter={endDateFilter}
        setEndDateFilter={setEndDateFilter}
        ragFocusMonth={ragFocusMonth}
        datasetState={datasetState}
        featureIntel={featureIntel}
        modelsSummary={modelsSummary}
        explainability={explainability}
        anomalies={anomalies}
        monitoring={monitoring}
        loading={loading}
        uploading={uploading}
        errorBanner={errorBanner}
        fetchAllAnalytics={fetchAllAnalytics}
        handleUploadFile={handleUploadFile}
        handleResetDataset={handleResetDataset}
        handleSelectMonthForRag={handleSelectMonthForRag}
      />
    </ScenarioProvider>
  );
}

interface AppWorkspaceProps {
  activeSection: SectionId;
  setActiveSection: (id: SectionId) => void;
  mobileOpen: boolean;
  setMobileOpen: (open: boolean) => void;
  quarterFilter: string;
  setQuarterFilter: (q: string) => void;
  startDateFilter: string;
  setStartDateFilter: (d: string) => void;
  endDateFilter: string;
  setEndDateFilter: (d: string) => void;
  ragFocusMonth: string | null;
  datasetState: DatasetState | null;
  featureIntel: FeaturesIntelligenceResponse | null;
  modelsSummary: ModelsSummaryResponse | null;
  explainability: ExplainabilityResponse | null;
  anomalies: AnomalyReport | null;
  monitoring: MonitoringReport | null;
  loading: boolean;
  uploading: boolean;
  errorBanner: string | null;
  fetchAllAnalytics: () => Promise<void>;
  handleUploadFile: (file: File) => Promise<void>;
  handleResetDataset: () => Promise<void>;
  handleSelectMonthForRag: (monthIso: string) => void;
}

const AppWorkspace: React.FC<AppWorkspaceProps> = ({
  activeSection,
  setActiveSection,
  mobileOpen,
  setMobileOpen,
  quarterFilter,
  setQuarterFilter,
  startDateFilter,
  setStartDateFilter,
  endDateFilter,
  setEndDateFilter,
  ragFocusMonth,
  datasetState,
  featureIntel,
  modelsSummary,
  explainability,
  anomalies,
  monitoring,
  loading,
  uploading,
  errorBanner,
  fetchAllAnalytics,
  handleUploadFile,
  handleResetDataset,
  handleSelectMonthForRag,
}) => {
  const { setDrawerOpen, activePresetLabel, isScenarioActive } = useScenario();
  const headerMeta = SECTION_TITLES[activeSection];

  return (
    <div className="flex min-h-screen bg-[#F3F6FA] text-slate-900">
      <Sidebar
        activeSection={activeSection}
        onSelectSection={setActiveSection}
        mobileOpen={mobileOpen}
        onCloseMobile={() => setMobileOpen(false)}
        onOpenScenarioDrawer={() => setDrawerOpen(true)}
        activeScenarioLabel={activePresetLabel}
        isScenarioActive={isScenarioActive}
      />

      <div className="flex min-w-0 flex-1 flex-col">
        <TopHeader
          title={headerMeta.title}
          subtitle={headerMeta.subtitle}
          datasetName={datasetState?.dataset_name || "default_financial_data.csv"}
          recordCount={datasetState?.kpis.period_count ?? 0}
          dateRange={datasetState?.available_date_range || { start: null, end: null }}
          quarterFilter={quarterFilter}
          onQuarterChange={setQuarterFilter}
          startDateFilter={startDateFilter}
          endDateFilter={endDateFilter}
          onStartDateChange={setStartDateFilter}
          onEndDateChange={setEndDateFilter}
          onResetFilters={() => {
            setQuarterFilter("ALL");
            setStartDateFilter(datasetState?.available_date_range?.start || "");
            setEndDateFilter(datasetState?.available_date_range?.end || "");
          }}
          onUploadFile={handleUploadFile}
          onResetDataset={handleResetDataset}
          uploading={uploading}
          onOpenMobileMenu={() => setMobileOpen(true)}
        />

        <main className="flex-1 px-5 py-6 lg:px-8">
          {errorBanner && (
            <div className="mb-5 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-xs font-medium text-red-800">
              <span>{errorBanner}</span>
              <button
                type="button"
                onClick={() => void fetchAllAnalytics()}
                className="rounded-lg border border-red-300 bg-white px-3 py-1 text-xs font-semibold text-red-800 hover:bg-red-100 whitespace-nowrap"
              >
                Retry Connection
              </button>
            </div>
          )}

          {loading || !datasetState ? (
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
              {[0, 1, 2, 3].map((idx) => (
                <div
                  key={idx}
                  className="h-28 animate-pulse rounded-xl border border-slate-200 bg-white"
                />
              ))}
            </div>
          ) : (
            <>
              <GlobalScenarioBar activeSection={activeSection} onNavigate={setActiveSection} />

              <Suspense fallback={<SectionFallback />}>
                {activeSection === "executive" && (
                  <ExecutiveOverviewView
                    dataset={datasetState}
                    models={modelsSummary}
                    explainability={explainability}
                    anomalies={anomalies}
                    onNavigate={setActiveSection}
                    onSelectMonthForRag={handleSelectMonthForRag}
                  />
                )}
                {activeSection === "performance" && (
                  <FinancialPerformanceView dataset={datasetState} />
                )}
                {activeSection === "engineering" && (
                  <DataEngineeringView data={featureIntel} />
                )}
                {activeSection === "prediction" && (
                  <PredictionView
                    models={modelsSummary}
                    latestRecord={datasetState.latest_record}
                    records={datasetState.records}
                    onNavigateToForecasting={() => setActiveSection("forecasting")}
                  />
                )}
                {activeSection === "forecasting" && (
                  <ForecastingCentreView
                    datasetName={datasetState.dataset_name}
                    recordCount={datasetState.quality_summary.records_processed}
                  />
                )}
                {activeSection === "explainability" && (
                  <ExplainabilityView data={explainability} />
                )}
                {activeSection === "rag" && (
                  <GraphicalRagView selectedMonth={ragFocusMonth} />
                )}
                {activeSection === "anomalies" && (
                  <AnomalyDetectionView data={anomalies} />
                )}
                {activeSection === "scenario" && (
                  <ScenarioSimulationView
                    latestRecord={datasetState.latest_record}
                    datasetName={datasetState.dataset_name}
                    onNavigateToForecasting={() => setActiveSection("forecasting")}
                    onNavigateToEngineering={() => setActiveSection("engineering")}
                  />
                )}
                {activeSection === "reports" && (
                  <ReportsMonitoringView
                    dataset={datasetState}
                    models={modelsSummary}
                    explainability={explainability}
                    anomalies={anomalies}
                    monitoring={monitoring}
                  />
                )}
              </Suspense>
            </>
          )}
        </main>
      </div>

      <GlobalScenarioDrawer onNavigate={setActiveSection} />
    </div>
  );
}

export default App;
