import React from "react";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  BrainCircuit,
  Cpu,
  Database,
  FileText,
  LayoutDashboard,
  Network,
  SlidersHorizontal,
  TrendingUp,
} from "lucide-react";

export type SectionId =
  | "executive"
  | "performance"
  | "engineering"
  | "prediction"
  | "forecasting"
  | "explainability"
  | "rag"
  | "anomalies"
  | "scenario"
  | "reports";

interface NavItem {
  id: SectionId;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
}

const NAV_ITEMS: NavItem[] = [
  { id: "executive", label: "Executive Overview", icon: LayoutDashboard },
  { id: "performance", label: "Financial Performance", icon: BarChart3 },
  { id: "engineering", label: "Data & Features", icon: Database },
  { id: "prediction", label: "Profit Prediction", icon: Cpu },
  { id: "forecasting", label: "2030 Forecasting Centre", icon: TrendingUp },
  { id: "explainability", label: "Model Explainability", icon: BrainCircuit },
  { id: "rag", label: "Graphical RAG", icon: Network },
  { id: "anomalies", label: "Anomaly Detection", icon: AlertTriangle },
  { id: "scenario", label: "Scenario Simulation", icon: SlidersHorizontal },
  { id: "reports", label: "Reports & Monitoring", icon: FileText },
];

interface SidebarProps {
  activeSection: SectionId;
  onSelectSection: (id: SectionId) => void;
  mobileOpen: boolean;
  onCloseMobile: () => void;
  onOpenScenarioDrawer?: () => void;
  activeScenarioLabel?: string;
  isScenarioActive?: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeSection,
  onSelectSection,
  mobileOpen,
  onCloseMobile,
  onOpenScenarioDrawer,
  activeScenarioLabel = "Base Model",
  isScenarioActive = false,
}) => {
  return (
    <>
      {mobileOpen && (
        <div
          className="fixed inset-0 z-30 bg-slate-950/50 lg:hidden"
          onClick={onCloseMobile}
        />
      )}
      <aside
        className={`fixed inset-y-0 left-0 z-40 flex w-[256px] flex-col justify-between bg-[#081B33] text-slate-200 transition-transform duration-200 lg:static lg:translate-x-0 ${
          mobileOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div>
          {/* Brand Logo Lockup */}
          <div className="flex items-center gap-3 px-5 pt-6 pb-5 border-b border-white/10">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-[#0A2E65] via-[#1D63B8] to-[#0D9488] border border-white/15 shadow-sm">
              <Activity className="h-5 w-5 text-white" />
            </div>
            <div className="min-w-0">
              <div className="font-display text-base font-bold tracking-tight text-white leading-tight">
                FinSight BI
              </div>
              <div className="text-[11px] font-medium text-slate-400">
                Predictive Intelligence
              </div>
            </div>
          </div>

          <nav className="mt-3 space-y-0.5 px-3" aria-label="Primary Navigation">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              const isActive = activeSection === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => {
                    onSelectSection(item.id);
                    onCloseMobile();
                  }}
                  className={`flex w-full items-center gap-3 rounded-lg px-3.5 py-2.5 text-left text-[13.5px] font-medium transition-all whitespace-nowrap ${
                    isActive
                      ? "bg-gradient-to-r from-[#0A2E65] via-[#154E96] to-[#0D7D82] text-white shadow-sm"
                      : "text-slate-300 hover:bg-white/5 hover:text-white"
                  }`}
                >
                  <Icon
                    className={`h-4 w-4 shrink-0 ${
                      isActive ? "text-teal-300" : "text-slate-400"
                    }`}
                  />
                  <span className="truncate">{item.label}</span>
                </button>
              );
            })}
          </nav>

          {onOpenScenarioDrawer && (
            <div className="mt-4 px-3">
              <button
                type="button"
                onClick={() => {
                  onOpenScenarioDrawer();
                  onCloseMobile();
                }}
                className={`flex w-full items-center justify-between gap-2 rounded-lg border px-3 py-2.5 text-left transition-colors ${
                  isScenarioActive
                    ? "border-teal-400/50 bg-[#0D9488]/20 text-white"
                    : "border-white/15 bg-white/5 text-slate-200 hover:bg-white/10"
                }`}
              >
                <div className="min-w-0">
                  <div className="text-[11px] font-semibold text-teal-300">
                    Scenario Simulator
                  </div>
                  <div className="truncate text-xs font-bold text-white">
                    {activeScenarioLabel}
                  </div>
                </div>
                <SlidersHorizontal className="h-4 w-4 shrink-0 text-teal-300" />
              </button>
            </div>
          )}
        </div>

        {/* Bottom Visual Anchor stepping across the 3-color relational triad */}
        <div className="px-5 py-5 border-t border-white/10">
          <div className="mb-3 flex items-end gap-1.5 h-10" aria-hidden="true">
            <div className="w-2.5 h-3 rounded-t bg-[#0A2E65]" />
            <div className="w-2.5 h-5 rounded-t bg-[#124A8F]" />
            <div className="w-2.5 h-6 rounded-t bg-[#1D63B8]" />
            <div className="w-2.5 h-8 rounded-t bg-[#1380A0]" />
            <div className="w-2.5 h-10 rounded-t bg-gradient-to-t from-[#0D9488] to-[#2DD4BF]" />
          </div>
          <p className="text-xs font-semibold text-white leading-snug">
            From Financial Data to Predictive Intelligence.
          </p>
          <p className="mt-1 text-[11px] text-slate-400 leading-snug">
            Better decisions backed by ML, SHAP &amp; Graphical RAG.
          </p>
        </div>
      </aside>
    </>
  );
};
