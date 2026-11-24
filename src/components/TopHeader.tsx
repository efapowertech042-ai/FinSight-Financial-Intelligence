import React, { useRef } from "react";
import { Menu, RotateCcw, Upload } from "lucide-react";

interface TopHeaderProps {
  title: string;
  subtitle: string;
  datasetName: string;
  recordCount: number;
  dateRange: { start: string | null; end: string | null };
  quarterFilter: string;
  onQuarterChange: (q: string) => void;
  startDateFilter: string;
  endDateFilter: string;
  onStartDateChange: (d: string) => void;
  onEndDateChange: (d: string) => void;
  onResetFilters: () => void;
  onUploadFile: (file: File) => void;
  onResetDataset: () => void;
  uploading: boolean;
  onOpenMobileMenu: () => void;
}

export const TopHeader: React.FC<TopHeaderProps> = ({
  title,
  subtitle,
  datasetName,
  recordCount,
  dateRange,
  quarterFilter,
  onQuarterChange,
  startDateFilter,
  endDateFilter,
  onStartDateChange,
  onEndDateChange,
  onResetFilters,
  onUploadFile,
  onResetDataset,
  uploading,
  onOpenMobileMenu,
}) => {
  const fileRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) {
      onUploadFile(f);
      e.target.value = "";
    }
  };

  const rangeStartYm = dateRange.start ? dateRange.start.slice(0, 7) : "";
  const rangeEndYm = dateRange.end ? dateRange.end.slice(0, 7) : "";
  const startYm = startDateFilter ? startDateFilter.slice(0, 7) : rangeStartYm;
  const endYm = endDateFilter ? endDateFilter.slice(0, 7) : rangeEndYm;
  const hasCustomFilter =
    quarterFilter !== "ALL" ||
    (Boolean(rangeStartYm) && startYm !== rangeStartYm) ||
    (Boolean(rangeEndYm) && endYm !== rangeEndYm);

  return (
    <header className="triad-top-rule border-b border-slate-200 bg-white px-5 py-4 lg:px-8">
      <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
        <div className="flex items-start gap-3">
          <button
            type="button"
            onClick={onOpenMobileMenu}
            className="mt-0.5 rounded-lg border border-slate-200 p-2 text-slate-600 hover:bg-slate-50 lg:hidden"
            aria-label="Open Navigation"
          >
            <Menu className="h-5 w-5" />
          </button>
          <div>
            <h1 className="font-display text-xl font-bold tracking-tight text-[#0A1E3D] sm:text-2xl">
              {title}
            </h1>
            <div className="mt-0.5 flex flex-wrap items-center gap-2 text-xs text-slate-500">
              <span>{subtitle}</span>
              <span aria-hidden="true">·</span>
              <span className="font-medium text-slate-700">{datasetName}</span>
              <span aria-hidden="true">·</span>
              <span className="tabular-nums">{recordCount} months</span>
              {dateRange.start && dateRange.end && (
                <>
                  <span aria-hidden="true">·</span>
                  <span className="tabular-nums">
                    {rangeStartYm} to {rangeEndYm}
                  </span>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Filters + Upload Action */}
        <div className="flex flex-wrap items-center gap-2.5">
          <div className="flex items-center gap-1.5 rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1.5 text-xs">
            <label htmlFor="filter-start" className="text-slate-500 font-medium">From</label>
            <input
              id="filter-start"
              type="month"
              min={rangeStartYm || undefined}
              max={endYm || rangeEndYm || undefined}
              value={startYm}
              onChange={(e) => onStartDateChange(e.target.value ? `${e.target.value}-01` : dateRange.start || "")}
              className="bg-transparent font-mono text-xs text-slate-800 focus:outline-none"
            />
            <span className="text-slate-300">|</span>
            <label htmlFor="filter-end" className="text-slate-500 font-medium">To</label>
            <input
              id="filter-end"
              type="month"
              min={startYm || rangeStartYm || undefined}
              max={rangeEndYm || undefined}
              value={endYm}
              onChange={(e) => onEndDateChange(e.target.value ? `${e.target.value}-01` : dateRange.end || "")}
              className="bg-transparent font-mono text-xs text-slate-800 focus:outline-none"
            />
          </div>

          <div className="flex items-center gap-0.5 rounded-lg bg-slate-100 p-1">
            {["ALL", "Q1", "Q2", "Q3", "Q4"].map((q) => (
              <button
                key={q}
                type="button"
                onClick={() => onQuarterChange(q)}
                className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors whitespace-nowrap ${
                  quarterFilter === q
                    ? "bg-white text-[#0A1E3D] shadow-sm"
                    : "text-slate-600 hover:text-slate-900"
                }`}
              >
                {q === "ALL" ? "All Qtrs" : q}
              </button>
            ))}
          </div>

          {hasCustomFilter && (
            <button
              type="button"
              onClick={onResetFilters}
              className="rounded-lg border border-slate-200 px-2.5 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50 whitespace-nowrap"
            >
              Reset Filters
            </button>
          )}

          <input
            ref={fileRef}
            type="file"
            accept=".csv,.xlsx,.xls"
            onChange={handleFileChange}
            className="hidden"
          />
          <button
            type="button"
            disabled={uploading}
            onClick={() => fileRef.current?.click()}
            className="triad-btn-primary inline-flex items-center gap-1.5 rounded-lg px-3.5 py-2 text-xs font-semibold text-white shadow-sm disabled:opacity-60 whitespace-nowrap shrink-0"
          >
            <Upload className="h-3.5 w-3.5" />
            <span>{uploading ? "Processing..." : "Upload CSV / Excel"}</span>
          </button>

          {datasetName !== "default_financial_data.csv" && (
            <button
              type="button"
              onClick={onResetDataset}
              className="inline-flex items-center gap-1 rounded-lg border border-slate-200 px-2.5 py-2 text-xs font-medium text-slate-700 hover:bg-slate-50 whitespace-nowrap"
              title="Restore default 10-year financial dataset"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              <span>Restore Default</span>
            </button>
          )}
        </div>
      </div>
    </header>
  );
};
