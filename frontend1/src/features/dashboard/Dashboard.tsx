import { useCallback, useEffect, useMemo, useState } from "react";
import MechanicalEye from "../../components/eye/MechanicalEye";
import Card from "../../components/ui/Card";
import Button from "../../components/ui/Button";
import { getMe, type UserResponse } from "../../services/auth";
import { listPatients, type PatientResponse } from "../../services/patients";
import { listReports, type ReportResponse } from "../../services/reports";
import { severityFromStage } from "../screening/mockAnalysis";

type ReportWithPriority = ReportResponse & { priority_score: 1 | 2 | 3 | 4 | 5 };

function formatPercent(confidence: number) {
  return `${(confidence * 100).toFixed(1)}%`;
}

function formatCompact(n: number) {
  return new Intl.NumberFormat(undefined, { notation: "compact" }).format(n);
}

function sameLocalDay(a: Date, b: Date) {
  return (
    a.getFullYear() === b.getFullYear() &&
    a.getMonth() === b.getMonth() &&
    a.getDate() === b.getDate()
  );
}

function SvgBars(props: { labels: string[]; values: number[]; yUnit?: string }) {
  const { labels, values, yUnit } = props;
  const width = 600;
  const height = 200;
  const axisLeft = 46;
  const axisBottom = 38;
  const paddingRight = 14;
  const paddingTop = 14;
  const plotW = width - axisLeft - paddingRight;
  const plotH = height - paddingTop - axisBottom;

  const max = Math.max(1, ...values);
  const barW = plotW / values.length;

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="w-full h-72 text-on-surface-variant"
    >
      <defs>
        <linearGradient id="barGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="rgb(251,130,253)" stopOpacity="0.95" />
          <stop offset="100%" stopColor="rgb(148,34,156)" stopOpacity="0.75" />
        </linearGradient>
      </defs>

      {/* grid + y ticks */}
      {[0, 0.25, 0.5, 0.75, 1].map((t) => {
        const y = paddingTop + plotH * (1 - t);
        const v = Math.round(max * t);
        return (
          <g key={t}>
            <line
              x1={axisLeft}
              x2={width - paddingRight}
              y1={y}
              y2={y}
              stroke="currentColor"
              strokeOpacity="0.16"
              strokeWidth="1"
            />
            <text
              x={axisLeft - 10}
              y={y + 4}
              textAnchor="end"
              fontSize="16"
              fill="currentColor"
              fillOpacity="0.8"
            >
              {v}
            </text>
          </g>
        );
      })}

      {/* axes */}
      <line
        x1={axisLeft}
        x2={axisLeft}
        y1={paddingTop}
        y2={paddingTop + plotH}
        stroke="currentColor"
        strokeOpacity="0.35"
        strokeWidth="1"
      />
      <line
        x1={axisLeft}
        x2={width - paddingRight}
        y1={paddingTop + plotH}
        y2={paddingTop + plotH}
        stroke="currentColor"
        strokeOpacity="0.35"
        strokeWidth="1"
      />

      {values.map((v, i) => {
        const h = (plotH * v) / max;
        const x = axisLeft + i * barW + 10;
        const y = paddingTop + plotH - h;
        const w = Math.max(10, barW - 20);
        return (
          <g key={labels[i] ?? i}>
            <rect
              x={x}
              y={y}
              width={w}
              height={h}
              rx="10"
              fill="url(#barGrad)"
              opacity={0.9}
            />
            <text
              x={x + w / 2}
              y={y - 8}
              textAnchor="middle"
              fontSize="16"
              fontWeight="700"
              fill="currentColor"
              fillOpacity="0.9"
            >
              {v}
            </text>
            <text
              x={x + w / 2}
              y={height - 14}
              textAnchor="middle"
              fontSize="16"
              fill="currentColor"
              fillOpacity="0.85"
            >
              {labels[i]}
            </text>
          </g>
        );
      })}

      {yUnit && (
        <text
          x={axisLeft}
          y={12}
          textAnchor="start"
          fontSize="16"
          fill="currentColor"
          fillOpacity="0.75"
        >
          {yUnit}
        </text>
      )}
    </svg>
  );
}

function SvgLine(props: { values: number[]; yUnit?: string; xStartLabel?: string; xEndLabel?: string }) {
  const width = 620;
  const height = 200;
  const axisLeft = 46;
  const axisBottom = 38;
  const paddingRight = 14;
  const paddingTop = 14;
  const plotW = width - axisLeft - paddingRight;
  const plotH = height - paddingTop - axisBottom;

  const max = Math.max(1, ...props.values);
  const min = Math.min(0, ...props.values);
  const span = Math.max(1, max - min);
  const stepX = plotW / Math.max(1, props.values.length - 1);

  const points = props.values
    .map((v, i) => {
      const x = axisLeft + i * stepX;
      const y = paddingTop + (plotH * (max - v)) / span;
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="w-full h-72 text-on-surface-variant"
    >
      <defs>
        <linearGradient id="lineGrad" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="rgb(148,34,156)" stopOpacity="0.55" />
          <stop offset="60%" stopColor="rgb(251,130,253)" stopOpacity="0.9" />
          <stop offset="100%" stopColor="rgb(251,130,253)" stopOpacity="0.35" />
        </linearGradient>
      </defs>
      {[0, 0.25, 0.5, 0.75, 1].map((t) => {
        const y = paddingTop + plotH * (1 - t);
        const v = Math.round(max * t);
        return (
          <g key={t}>
            <line
              x1={axisLeft}
              x2={width - paddingRight}
              y1={y}
              y2={y}
              stroke="currentColor"
              strokeOpacity="0.16"
              strokeWidth="1"
            />
            <text
              x={axisLeft - 10}
              y={y + 4}
              textAnchor="end"
              fontSize="16"
              fill="currentColor"
              fillOpacity="0.8"
            >
              {v}
            </text>
          </g>
        );
      })}
      <line
        x1={axisLeft}
        x2={axisLeft}
        y1={paddingTop}
        y2={paddingTop + plotH}
        stroke="currentColor"
        strokeOpacity="0.35"
        strokeWidth="1"
      />
      <line
        x1={axisLeft}
        x2={width - paddingRight}
        y1={paddingTop + plotH}
        y2={paddingTop + plotH}
        stroke="currentColor"
        strokeOpacity="0.35"
        strokeWidth="1"
      />

      <polyline
        points={points}
        fill="none"
        stroke="url(#lineGrad)"
        strokeWidth="4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {props.values.map((v, i) => {
        const x = axisLeft + i * stepX;
        const y = paddingTop + (plotH * (max - v)) / span;
        return (
          <g key={i}>
            <circle cx={x} cy={y} r="4.5" fill="rgb(251,130,253)" opacity={0.9} />
          </g>
        );
      })}

      <text
        x={axisLeft}
        y={height - 12}
        textAnchor="start"
        fontSize="16"
        fill="currentColor"
        fillOpacity="0.8"
      >
        {props.xStartLabel ?? ""}
      </text>
      <text
        x={width - paddingRight}
        y={height - 12}
        textAnchor="end"
        fontSize="16"
        fill="currentColor"
        fillOpacity="0.8"
      >
        {props.xEndLabel ?? ""}
      </text>
      {props.yUnit && (
        <text
          x={axisLeft}
          y={12}
          textAnchor="start"
          fontSize="16"
          fill="currentColor"
          fillOpacity="0.75"
        >
          {props.yUnit}
        </text>
      )}
    </svg>
  );
}

function StatCard(props: { label: string; value: string; hint?: string; icon: string }) {
  return (
    <Card className="p-6 space-y-3 shadow-2xl shadow-black/10 hover:shadow-black/20 transition-shadow">
      <div className="flex items-center justify-between">
        <span className="text-on-surface-variant font-medium">{props.label}</span>
        <span className="material-symbols-outlined text-primary text-xl">{props.icon}</span>
      </div>
      <div className="flex items-baseline gap-2">
        <span className="text-3xl font-headline font-bold">{props.value}</span>
        {props.hint && <span className="text-on-surface-variant text-sm">{props.hint}</span>}
      </div>
    </Card>
  );
}

export default function Dashboard() {
  const [user, setUser] = useState<UserResponse | null>(null);
  const [patients, setPatients] = useState<PatientResponse[]>([]);
  const [reports, setReports] = useState<ReportWithPriority[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setError(null);
    setIsLoading(true);
    const [meRes, patientsRes, reportsRes] = await Promise.allSettled([
      getMe(),
      listPatients(),
      listReports(),
    ]);

    if (meRes.status === "fulfilled") setUser(meRes.value);
    if (patientsRes.status === "fulfilled") setPatients(patientsRes.value);
    if (reportsRes.status === "fulfilled") {
      const sorted = reportsRes.value
        .slice()
        .sort((a, b) => +new Date(b.created_at) - +new Date(a.created_at))
        .map((r) => ({ ...r, priority_score: severityFromStage(r.prediction) as 1 | 2 | 3 | 4 | 5 }));
      setReports(sorted);
    }

    if (
      patientsRes.status === "rejected" ||
      reportsRes.status === "rejected"
    ) {
      setError("Failed to load dashboard data.");
    }

    setIsLoading(false);
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const stats = useMemo(() => {
    const now = new Date();
    const todayReports = reports.filter((r) => sameLocalDay(new Date(r.created_at), now));
    const avgConfidence =
      reports.length === 0
        ? 0
        : reports.reduce((acc, r) => acc + r.confidence, 0) / reports.length;
    const highPriority = reports.filter((r) => r.priority_score >= 4).length;
    const lastReportAt = reports[0]?.created_at ? new Date(reports[0].created_at) : null;

    return {
      patientCount: patients.length,
      reportCount: reports.length,
      todayCount: todayReports.length,
      avgConfidence,
      highPriority,
      lastReportAt,
    };
  }, [patients.length, reports]);

  const severityDist = useMemo(() => {
    const stages = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"] as const;
    const counts = stages.map((s) => reports.filter((r) => r.prediction === s).length);
    return { stages, counts };
  }, [reports]);

  const confidenceHistogram = useMemo(() => {
    const bins = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01];
    const labels = ["70", "75", "80", "85", "90", "95"];
    const counts = new Array(labels.length).fill(0) as number[];
    for (const r of reports) {
      const c = r.confidence;
      let idx = labels.length - 1;
      for (let i = 0; i < labels.length; i++) {
        if (c >= bins[i]! && c < bins[i + 1]!) {
          idx = i;
          break;
        }
      }
      counts[idx]! += 1;
    }
    return { labels, counts };
  }, [reports]);

  const reportsLast14Days = useMemo(() => {
    const days = 14;
    const today = new Date();
    const series: number[] = [];
    for (let i = days - 1; i >= 0; i--) {
      const d = new Date(today);
      d.setDate(today.getDate() - i);
      const count = reports.filter((r) => sameLocalDay(new Date(r.created_at), d)).length;
      series.push(count);
    }
    return series;
  }, [reports]);

  const recent = useMemo(() => reports.slice(0, 8), [reports]);
  const patientNameById = useMemo(() => new Map(patients.map((p) => [p.id, p.name])), [patients]);

  return (
    <main className="min-h-screen pt-24 pb-32 px-6 md:px-12 max-w-7xl mx-auto">
      <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-4 mb-10">
        <div>
          <h1 className="text-4xl font-headline font-extrabold text-on-surface tracking-tight">
            {user ? `Welcome, ${user.name}` : "Dashboard"}
          </h1>
        </div>
        <div className="flex gap-3">
          <Button variant="secondary" icon="refresh" onClick={load} disabled={isLoading}>
            Refresh
          </Button>
        </div>
      </div>

      {error && (
        <Card className="p-5 mb-8 border border-error/25 bg-error-container/20">
          <div className="text-error font-semibold">{error}</div>
          <div className="text-on-surface-variant text-sm mt-1">
            Make sure the backend is running and you&apos;re logged in.
          </div>
        </Card>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-4 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-1 gap-6">
          <StatCard label="Patients" value={formatCompact(stats.patientCount)} icon="group" />
          <StatCard label="Total Reports" value={formatCompact(stats.reportCount)} icon="summarize" />
          <StatCard label="Reports Today" value={formatCompact(stats.todayCount)} icon="today" />
          <StatCard
            label="High Priority"
            value={formatCompact(stats.highPriority)}
            hint="(4–5)"
            icon="priority_high"
          />
          <StatCard
            label="Avg Confidence"
            value={`${(stats.avgConfidence * 100).toFixed(1)}%`}
            icon="analytics"
          />
          <StatCard
            label="Last Analysis"
            value={stats.lastReportAt ? stats.lastReportAt.toLocaleTimeString() : "—"}
            hint={stats.lastReportAt ? stats.lastReportAt.toLocaleDateString() : undefined}
            icon="schedule"
          />
        </div>

        <div className="lg:col-span-5 flex flex-col items-center justify-center">
          <div className="w-full flex items-center justify-center py-6">
            <MechanicalEye />
          </div>
          <Card className="w-full p-6">
            <div className="flex items-center justify-between mb-3">
              <div className="font-headline font-bold">Reports (14 days)</div>
              <div className="text-xs text-on-surface-variant">
                {isLoading ? "Loading…" : `${reportsLast14Days.reduce((a, b) => a + b, 0)} total`}
              </div>
            </div>
            <div className="flex items-center justify-between text-xs text-on-surface-variant mb-3">
              <span>X: Days (old → new)</span>
              <span>Y: Reports (count)</span>
            </div>
            <SvgLine
              values={reportsLast14Days}
              yUnit="Reports"
              xStartLabel="14d ago"
              xEndLabel="Today"
            />
          </Card>
        </div>

        <div className="lg:col-span-3 space-y-6">
          <Card className="p-6">
            <div className="flex items-center justify-between mb-3">
              <div className="font-headline font-bold">Severity Distribution</div>
              <div className="text-xs text-on-surface-variant">{reports.length} reports</div>
            </div>
            <div className="flex items-center justify-between text-xs text-on-surface-variant mb-3">
              <span>X: DR stage</span>
              <span>Y: Reports (count)</span>
            </div>
            <SvgBars labels={["0", "1", "2", "3", "4"]} values={severityDist.counts} yUnit="Reports" />
            <div className="grid grid-cols-5 gap-2 mt-2 text-xs text-on-surface-variant">
              {["No DR", "Mild", "Moderate", "Severe", "Prolif."].map((t) => (
                <div key={t} className="text-center">
                  {t}
                </div>
              ))}
            </div>
          </Card>

          <Card className="p-6">
            <div className="flex items-center justify-between mb-3">
              <div className="font-headline font-bold">Confidence Histogram</div>
              <div className="text-xs text-on-surface-variant">(% bins)</div>
            </div>
            <div className="flex items-center justify-between text-xs text-on-surface-variant mb-3">
              <span>X: Confidence (%)</span>
              <span>Y: Reports (count)</span>
            </div>
            <SvgBars labels={confidenceHistogram.labels} values={confidenceHistogram.counts} yUnit="Reports" />
          </Card>
        </div>
      </div>

      <div className="mt-10 grid grid-cols-1 lg:grid-cols-12 gap-6">
        <Card className="lg:col-span-12 p-6 overflow-hidden">
          <div className="flex items-center justify-between mb-4">
            <h2 className="font-headline font-bold text-xl">Recent Reports</h2>
            <div className="text-xs text-on-surface-variant">
              {isLoading ? "Loading…" : "Newest first"}
            </div>
          </div>
          <div className="overflow-x-auto no-scrollbar">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="text-xs uppercase tracking-wider text-on-surface-variant">
                  <th className="py-3 pr-4">Patient</th>
                  <th className="py-3 pr-4">Prediction</th>
                  <th className="py-3 pr-4 text-center">Confidence</th>
                  <th className="py-3 pr-4 text-center">Priority</th>
                  <th className="py-3 text-right">Time</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-outline-variant/10">
                {recent.map((r) => (
                  <tr key={r.id} className="text-sm hover:bg-surface-container-high transition-colors">
                    <td className="py-4 pr-4 font-semibold">
                      {patientNameById.get(r.patient_id) ?? `Patient #${r.patient_id}`}
                    </td>
                    <td className="py-4 pr-4 text-on-surface-variant">{r.prediction}</td>
                    <td className="py-4 pr-4 text-center text-on-surface-variant">{formatPercent(r.confidence)}</td>
                    <td className="py-4 pr-4 text-center font-bold text-primary">{r.priority_score}</td>
                    <td className="py-4 text-right text-on-surface-variant">
                      {new Date(r.created_at).toLocaleString()}
                    </td>
                  </tr>
                ))}
                {!isLoading && recent.length === 0 && (
                  <tr>
                    <td colSpan={5} className="py-10 text-center text-on-surface-variant">
                      No reports yet. Run a screening to populate the dashboard.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </main>
  );
}
