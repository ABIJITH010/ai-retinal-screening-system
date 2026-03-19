import { useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent } from "react";
import Button from "../../components/ui/Button";
import Card from "../../components/ui/Card";
import { API_BASE_URL } from "../../utils/constants";
import {
  createPatient,
  listPatients,
  type PatientResponse,
} from "../../services/patients";
import {
  createReport,
  type ReportResponse,
} from "../../services/reports";
import { severityFromStage, stageDescription } from "./mockAnalysis";

type ReportRow = ReportResponse & { priority_score: 1 | 2 | 3 | 4 | 5 };

const BACKEND_ORIGIN = API_BASE_URL.replace(/\/api\/?$/, "");

function resolveBackendUrl(pathOrUrl: string) {
  if (!pathOrUrl) return "";
  if (
    pathOrUrl.startsWith("http://") ||
    pathOrUrl.startsWith("https://") ||
    pathOrUrl.startsWith("data:")
  ) {
    return pathOrUrl;
  }
  const normalized = pathOrUrl.replace(/\\/g, "/").replace(/^\/+/, "");
  return `${BACKEND_ORIGIN}/${normalized}`;
}

function toReportRow(report: ReportResponse): ReportRow {
  return { ...report, priority_score: severityFromStage(report.prediction) };
}

function readFileAsDataUrl(file: File) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error("Failed to read file"));
    reader.readAsDataURL(file);
  });
}

function formatPercent(confidence: number) {
  return `${(confidence * 100).toFixed(1)}%`;
}

export default function Screening() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const printFrameRef = useRef<HTMLIFrameElement | null>(null);

  const [patients, setPatients] = useState<PatientResponse[]>([]);
  const [patientsLoading, setPatientsLoading] = useState(false);
  const [patientsError, setPatientsError] = useState<string | null>(null);

  const [mode, setMode] = useState<"existing" | "new">("existing");
  const [selectedPatientId, setSelectedPatientId] = useState<number | null>(null);

  const [newName, setNewName] = useState("");
  const [newAge, setNewAge] = useState<number>(35);
  const [newGender, setNewGender] = useState("Male");
  const [newPhone, setNewPhone] = useState("");
  const [newAddress, setNewAddress] = useState("");
  const [createError, setCreateError] = useState<string | null>(null);
  const [createLoading, setCreateLoading] = useState(false);

  const [file, setFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);

  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [report, setReport] = useState<ReportRow | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);

  const patientsById = useMemo(
    () => new Map(patients.map((p) => [p.id, p])),
    [patients],
  );
  const selectedPatient = selectedPatientId
    ? patientsById.get(selectedPatientId)
    : undefined;

  async function refreshPatients() {
    setPatientsError(null);
    setPatientsLoading(true);
    try {
      const data = await listPatients();
      setPatients(data);
      if (!selectedPatientId && data.length > 0) setSelectedPatientId(data[0]!.id);
    } catch (err: unknown) {
      const message =
        err && typeof err === "object" && "message" in err
          ? String((err as { message: unknown }).message)
          : "Failed to load patients";
      setPatientsError(message);
    } finally {
      setPatientsLoading(false);
    }
  }

  useEffect(() => {
    refreshPatients();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function onCreatePatient(e: FormEvent) {
    e.preventDefault();
    setCreateError(null);
    setCreateLoading(true);
    try {
      const patient = await createPatient({
        name: newName.trim(),
        age: Number(newAge),
        gender: newGender,
        phone: newPhone.trim(),
        address: newAddress.trim(),
      });
      setPatients([patient, ...patients]);
      setSelectedPatientId(patient.id);
      setMode("existing");
      setNewName("");
      setNewPhone("");
      setNewAddress("");
    } catch (err: unknown) {
      const message =
        err && typeof err === "object" && "message" in err
          ? String((err as { message: unknown }).message)
          : "Failed to create patient";
      setCreateError(message);
    } finally {
      setCreateLoading(false);
    }
  }

  async function onPickFile(picked: File | null) {
    setAnalysisError(null);
    setReport(null);
    if (!picked) {
      setFile(null);
      setImagePreview(null);
      return;
    }
    setFile(picked);
    const dataUrl = await readFileAsDataUrl(picked);
    setImagePreview(dataUrl);
  }

  async function onRunAnalysis() {
    setAnalysisError(null);
    if (!selectedPatientId) return setAnalysisError("Please select (or create) a patient first.");
    if (!file) return setAnalysisError("Please upload an image first.");

    setIsAnalyzing(true);
    try {
      const created = await createReport({ patientId: selectedPatientId, file });
      setReport(toReportRow(created));
    } catch {
      setAnalysisError("Failed to run analysis");
    } finally {
      setIsAnalyzing(false);
    }
  }

  function onReset() {
    setFile(null);
    setImagePreview(null);
    setReport(null);
    setAnalysisError(null);
    setDragActive(false);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  function onExportPdf() {
    if (!report || !selectedPatient) return;
    const frame = printFrameRef.current;
    if (!frame) return;

    const imageUrl = resolveBackendUrl(report.image_url);
    const heatmapUrl = resolveBackendUrl(report.heatmap_url);
    const createdAt = new Date(report.created_at).toLocaleString();

    const html = `<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Retina Max Report</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: Arial, sans-serif; margin: 18px; color: #111827; }
    h1 { margin: 0 0 6px; font-size: 20px; }
    .muted { color: #6b7280; font-size: 12px; }
    .grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:12px; }
    .card { border:1px solid #e5e7eb; border-radius:10px; padding:12px; }
    .row { display:grid; grid-template-columns:120px 1fr; gap:8px; margin:4px 0; font-size: 12px; }
    .k { color:#6b7280; }
    img { width: 100%; border-radius: 10px; border: 1px solid #e5e7eb; }
    @media print { body { margin: 0; } }
  </style>
</head>
<body>
  <h1>Retina Max - Diabetic Retinopathy Report</h1>
  <div class="muted">Generated: ${createdAt}</div>
  <div class="grid">
    <div class="card">
      <div style="font-weight:700; margin-bottom:6px;">Patient</div>
      <div class="row"><div class="k">ID</div><div>${selectedPatient.id}</div></div>
      <div class="row"><div class="k">Name</div><div>${selectedPatient.name}</div></div>
      <div class="row"><div class="k">Age</div><div>${selectedPatient.age}</div></div>
      <div class="row"><div class="k">Gender</div><div>${selectedPatient.gender}</div></div>
      <div class="row"><div class="k">Phone</div><div>${selectedPatient.phone}</div></div>
      <div class="row"><div class="k">Address</div><div>${selectedPatient.address}</div></div>
    </div>
    <div class="card">
      <div style="font-weight:700; margin-bottom:6px;">Report</div>
      <div class="row"><div class="k">Report ID</div><div>${report.id}</div></div>
      <div class="row"><div class="k">Prediction</div><div><b>${report.prediction}</b></div></div>
      <div class="row"><div class="k">Confidence</div><div>${formatPercent(report.confidence)}</div></div>
      <div class="row"><div class="k">Priority</div><div>${report.priority_score}/5</div></div>
      <div class="row"><div class="k">Summary</div><div>${stageDescription(report.prediction)}</div></div>
    </div>
  </div>
  <div class="grid">
    <div class="card">
      <div style="font-weight:700; margin-bottom:6px;">Input Image</div>
      <img src="${imageUrl}" />
    </div>
    <div class="card">
      <div style="font-weight:700; margin-bottom:6px;">Heatmap</div>
      <img src="${heatmapUrl}" />
    </div>
  </div>
</body>
</html>`;

    frame.onload = async () => {
      const win = frame.contentWindow;
      const doc = frame.contentDocument;
      if (!win || !doc) return;
      const images = Array.from(doc.images);
      await Promise.all(
        images.map(
          (img) =>
            img.complete
              ? Promise.resolve()
              : new Promise<void>((resolve) => {
                  img.onload = () => resolve();
                  img.onerror = () => resolve();
                }),
        ),
      );
      win.focus();
      win.print();
    };

    frame.srcdoc = html;
  }

  const resolvedHeatmap = report ? resolveBackendUrl(report.heatmap_url) : "";
  const confidencePct = report ? Math.min(100, Math.max(0, report.confidence * 100)) : 0;

  const inputClass =
    "block w-full px-4 py-3 bg-surface-container-lowest border border-outline/10 rounded-xl font-body text-on-surface focus:ring-1 focus:ring-primary/40 focus:border-transparent transition-all outline-none";

  return (
    <main className="max-w-7xl mx-auto px-6 pt-24 pb-32">
      <iframe ref={printFrameRef} title="print" className="absolute w-0 h-0 opacity-0 pointer-events-none" />

      <div className="mb-10">
        <h1 className="font-headline text-4xl font-extrabold tracking-tight mb-2">Patient Screening</h1>
        <p className="text-on-surface-variant max-w-2xl">
          Select a patient, upload a fundus image, run analysis, and export a PDF report.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-7 space-y-6">
          <Card className="p-6 shadow-2xl shadow-black/20">
            <div className="flex items-center justify-between gap-3 mb-5">
              <h3 className="font-headline font-semibold text-lg">Patient</h3>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => setMode("existing")}
                  className={`px-3 py-1.5 rounded-lg text-xs font-bold border transition-colors ${
                    mode === "existing"
                      ? "bg-primary/20 border-primary/30 text-primary"
                      : "border-outline-variant/30 text-on-surface-variant hover:bg-surface-container-high"
                  }`}
                >
                  Existing
                </button>
                <button
                  type="button"
                  onClick={() => setMode("new")}
                  className={`px-3 py-1.5 rounded-lg text-xs font-bold border transition-colors ${
                    mode === "new"
                      ? "bg-primary/20 border-primary/30 text-primary"
                      : "border-outline-variant/30 text-on-surface-variant hover:bg-surface-container-high"
                  }`}
                >
                  New
                </button>
              </div>
            </div>

            {patientsError && (
              <div className="rounded-xl bg-error-container/30 text-error px-4 py-3 text-sm mb-4">
                {patientsError}
              </div>
            )}

            {mode === "existing" ? (
              <div className="grid grid-cols-1 md:grid-cols-12 gap-4 items-end">
                <div className="md:col-span-8">
                  <label className="block text-sm font-label text-on-surface-variant mb-2">
                    Select patient
                  </label>
                  <div className="relative">
                    <select
                      className="appearance-none w-full bg-surface-container-low border border-outline/10 rounded-xl px-4 py-3 pr-10 font-label text-on-surface-variant focus:ring-1 focus:ring-primary/40 outline-none"
                      value={selectedPatientId ?? ""}
                      onChange={(e) => setSelectedPatientId(Number(e.target.value))}
                      disabled={patientsLoading || patients.length === 0}
                    >
                      {patients.length === 0 ? (
                        <option value="">No patients yet</option>
                      ) : (
                        patients.map((p) => (
                          <option key={p.id} value={p.id}>
                            {p.name} (#{p.id})
                          </option>
                        ))
                      )}
                    </select>
                    <div className="absolute inset-y-0 right-0 pr-4 flex items-center pointer-events-none">
                      <span className="material-symbols-outlined text-outline">expand_more</span>
                    </div>
                  </div>
                </div>
                <div className="md:col-span-4">
                  <Button
                    type="button"
                    variant="secondary"
                    className="w-full"
                    icon="refresh"
                    disabled={patientsLoading}
                    onClick={refreshPatients}
                  >
                    Refresh
                  </Button>
                </div>
              </div>
            ) : (
              <form onSubmit={onCreatePatient} className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-label text-on-surface-variant mb-2">
                      Name
                    </label>
                    <input className={inputClass} value={newName} onChange={(e) => setNewName(e.target.value)} required />
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-label text-on-surface-variant mb-2">
                        Age
                      </label>
                      <input className={inputClass} type="number" min={0} value={newAge} onChange={(e) => setNewAge(Number(e.target.value))} required />
                    </div>
                    <div>
                      <label className="block text-sm font-label text-on-surface-variant mb-2">
                        Gender
                      </label>
                      <select className={inputClass} value={newGender} onChange={(e) => setNewGender(e.target.value)}>
                        <option>Male</option>
                        <option>Female</option>
                        <option>Other</option>
                      </select>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-label text-on-surface-variant mb-2">Phone</label>
                    <input className={inputClass} value={newPhone} onChange={(e) => setNewPhone(e.target.value)} required />
                  </div>
                  <div>
                    <label className="block text-sm font-label text-on-surface-variant mb-2">Address</label>
                    <input className={inputClass} value={newAddress} onChange={(e) => setNewAddress(e.target.value)} required />
                  </div>
                </div>

                {createError && <div className="rounded-xl bg-error-container/30 text-error px-4 py-3 text-sm">{createError}</div>}

                <div className="flex gap-3 justify-end pt-1">
                  <Button type="button" variant="ghost" onClick={() => setMode("existing")}>Cancel</Button>
                  <Button type="submit" disabled={createLoading} icon="person_add">{createLoading ? "Creating..." : "Create Patient"}</Button>
                </div>
              </form>
            )}
          </Card>

          <section
            className={`rounded-xl p-8 flex flex-col items-center justify-center border-2 border-dashed transition-all duration-300 cursor-pointer h-[280px] ${
              dragActive ? "bg-primary/10 border-primary/60" : "bg-surface-container-low border-outline-variant/20 hover:border-primary/40"
            }`}
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
            onDragLeave={() => setDragActive(false)}
            onDrop={(e) => { e.preventDefault(); setDragActive(false); void onPickFile(e.dataTransfer.files?.[0] ?? null); }}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => onPickFile(e.target.files?.[0] ?? null)}
            />
            <div className="bg-surface-container-high w-16 h-16 rounded-full flex items-center justify-center mb-4">
              <span className="material-symbols-outlined text-primary text-3xl">upload_file</span>
            </div>
            <h3 className="font-headline text-xl font-bold mb-2">Upload Retinal Image</h3>
            <p className="text-on-surface-variant text-sm text-center max-w-xs mb-5">Drag & drop or browse. PNG/JPG supported.</p>
            <button type="button" className="bg-gradient-to-r from-primary to-primary-container text-on-primary-container font-bold px-6 py-2.5 rounded-lg transition-transform active:scale-95 shadow-lg shadow-primary/10 hover:scale-[1.02]">
              Browse Files
            </button>
            {file && <div className="mt-4 text-xs text-on-surface-variant">Selected: <span className="font-semibold">{file.name}</span></div>}
          </section>

          <section className="bg-surface-container-low rounded-xl overflow-hidden relative shadow-2xl shadow-black/10">
            <div className="p-6 border-b border-outline-variant/10 flex justify-between items-center">
              <h3 className="font-headline font-semibold text-lg">Input Preview</h3>
              <span className="text-xs font-medium text-on-surface-variant px-2 py-1 bg-surface-container rounded-md">
                Patient: {selectedPatient ? `#${selectedPatient.id}` : "—"}
              </span>
            </div>
            <div className="aspect-video w-full bg-surface-container-lowest relative overflow-hidden flex items-center justify-center">
              {imagePreview ? (
                <>
                  <img
                    alt="Uploaded fundus photograph"
                    className={`w-full h-full object-cover transition-all duration-500 ${isAnalyzing ? "opacity-70 blur-[1px]" : "opacity-100"}`}
                    src={imagePreview}
                  />
                  {isAnalyzing && (
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="flex items-center gap-3 px-4 py-2 rounded-full bg-black/40 backdrop-blur-md border border-white/10">
                        <div className="w-5 h-5 rounded-full border-2 border-primary border-t-transparent animate-spin" />
                        <span className="text-sm font-bold text-white">Analyzing…</span>
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div className="text-on-surface-variant text-sm">No image selected</div>
              )}
            </div>
            <div className="p-6 flex justify-end gap-3">
              <Button type="button" variant="ghost" onClick={onReset} disabled={isAnalyzing}>Clear</Button>
              <Button type="button" onClick={onRunAnalysis} disabled={isAnalyzing || !selectedPatientId || !file} icon="auto_awesome">
                {isAnalyzing ? "Analyzing..." : "Run Analysis"}
              </Button>
            </div>
            {analysisError && (
              <div className="px-6 pb-6 -mt-3">
                <div className="rounded-xl bg-error-container/30 text-error px-4 py-3 text-sm">{analysisError}</div>
              </div>
            )}
          </section>
        </div>

        <div className="lg:col-span-5 space-y-6">
          <section className="bg-surface-container-low rounded-xl p-8 relative overflow-hidden h-full flex flex-col shadow-2xl shadow-black/20 lg:sticky lg:top-24">
            <div className="absolute top-0 right-0 p-4">
              <span className="material-symbols-outlined text-primary/10 text-7xl select-none">clinical_notes</span>
            </div>

            <h3 className="font-headline text-2xl font-bold mb-7 flex items-center gap-2">
              Analysis Results
              <span className={`w-2 h-2 rounded-full ${report ? "bg-primary animate-pulse" : "bg-outline/40"}`} />
            </h3>

            {!report ? (
              <div className="text-on-surface-variant text-sm leading-relaxed">
                Upload an image and run analysis to see results.
              </div>
            ) : (
              <div className="space-y-7 flex-grow transition-all duration-300">
                <div className="space-y-2">
                  <span className="text-xs uppercase tracking-widest text-on-surface-variant font-bold">Prediction</span>
                  <div className="bg-tertiary-container/10 border border-tertiary-container/20 rounded-xl p-5">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="text-2xl font-bold text-tertiary">{report.prediction}</h4>
                      <span className="text-xs font-bold px-2 py-1 rounded-md bg-surface-container-high text-on-surface-variant">
                        Priority {report.priority_score}/5
                      </span>
                    </div>
                    <p className="text-sm text-on-surface-variant leading-relaxed">{stageDescription(report.prediction)}</p>
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="flex justify-between items-end">
                    <span className="text-xs uppercase tracking-widest text-on-surface-variant font-bold">Confidence Score</span>
                    <span className="text-3xl font-headline font-extrabold text-primary tracking-tighter">
                      {confidencePct.toFixed(1)}<span className="text-lg">%</span>
                    </span>
                  </div>
                  <div className="h-1.5 w-full bg-surface-container-high rounded-full overflow-hidden">
                    <div className="h-full bg-primary rounded-full transition-[width] duration-700 ease-out" style={{ width: `${confidencePct}%` }} />
                  </div>
                </div>

                <div className="space-y-2 pt-2">
                  <span className="text-xs uppercase tracking-widest text-on-surface-variant font-bold">Heatmap (uploads/)</span>
                  <div className="aspect-square w-full rounded-xl bg-surface-container-lowest relative overflow-hidden">
                    {resolvedHeatmap ? (
                      <img alt="Heatmap visualization" className="w-full h-full object-cover opacity-90" src={resolvedHeatmap} />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center text-on-surface-variant text-sm">No heatmap available</div>
                    )}
                    <div className="absolute inset-0 bg-gradient-to-t from-surface-container-lowest via-transparent to-transparent" />
                  </div>
                </div>
              </div>
            )}

            <div className="mt-8 pt-6 border-t border-outline-variant/10 flex gap-4">
              <button
                type="button"
                className="flex-1 flex items-center justify-center gap-2 py-3 bg-surface-container-high rounded-lg text-sm font-bold hover:text-primary transition-colors disabled:opacity-50 disabled:hover:text-on-surface-variant"
                disabled={!report || !selectedPatient}
                onClick={onExportPdf}
              >
                <span className="material-symbols-outlined text-sm">print</span>
                Export PDF
              </button>
              <button
                type="button"
                className="flex-1 flex items-center justify-center gap-2 py-3 bg-surface-container-high rounded-lg text-sm font-bold hover:text-primary transition-colors"
                onClick={onReset}
              >
                <span className="material-symbols-outlined text-sm">restart_alt</span>
                New Scan
              </button>
            </div>
          </section>
        </div>
      </div>
    </main>
  );
}
