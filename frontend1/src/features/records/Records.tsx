import Card from "../../components/ui/Card";

const patients = [
  { initials: "EH", name: "Elias Thorne", pid: "#RET-9921", date: "Oct 24, 2023", result: "Stage II DR", resultClass: "bg-tertiary-container/20 text-tertiary-dim", confidence: 98.2 },
  { initials: "SC", name: "Selena Chao", pid: "#RET-8412", date: "Oct 22, 2023", result: "Healthy", resultClass: "bg-primary-container/20 text-primary", confidence: 99.8 },
  { initials: "MK", name: "Marcus Knight", pid: "#RET-1055", date: "Oct 21, 2023", result: "Severe Glaucoma", resultClass: "bg-error-container/20 text-error", confidence: 94.5 },
  { initials: "JL", name: "Julianne Lowe", pid: "#RET-7220", date: "Oct 20, 2023", result: "Mild ARMD", resultClass: "bg-tertiary-container/20 text-tertiary-dim", confidence: 87.1 },
];

export default function Records() {
  return (
    <main className="pt-24 pb-32 px-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-10">
        <h1 className="font-headline text-4xl font-extrabold tracking-tight text-on-surface mb-2">
          Records
        </h1>
        <p className="font-body text-on-surface-variant text-lg">
          Centralized historical patient screening database and AI diagnostics.
        </p>
      </div>

      {/* Filter & Search */}
      <div className="grid grid-cols-1 md:grid-cols-12 gap-6 mb-8">
        <div className="md:col-span-8">
          <div className="relative group">
            <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
              <span className="material-symbols-outlined text-outline">search</span>
            </div>
            <input
              className="block w-full pl-12 pr-4 py-4 bg-surface-container-lowest border border-outline/10 rounded-xl font-body text-on-surface focus:ring-1 focus:ring-primary/40 focus:border-transparent transition-all outline-none"
              placeholder="Search by patient name, ID or specialist..."
              type="text"
            />
          </div>
        </div>
        <div className="md:col-span-4 flex gap-4">
          <div className="relative w-full">
            <select className="appearance-none w-full bg-surface-container-low border-none rounded-xl px-4 py-4 pr-10 font-label text-on-surface-variant focus:ring-1 focus:ring-primary/40 outline-none">
              <option>All Severities</option>
              <option>Critical</option>
              <option>Moderate</option>
              <option>Stable</option>
            </select>
            <div className="absolute inset-y-0 right-0 pr-4 flex items-center pointer-events-none">
              <span className="material-symbols-outlined text-outline">filter_list</span>
            </div>
          </div>
        </div>
      </div>

      {/* Data Table */}
      <Card className="overflow-hidden shadow-2xl">
        <div className="overflow-x-auto no-scrollbar">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-surface-container-high">
                {["Patient Name", "Date", "Result", "Confidence", "Actions"].map((h, i) => (
                  <th
                    key={h}
                    className={`px-6 py-5 font-headline text-sm font-semibold text-on-surface-variant uppercase tracking-wider ${
                      i === 3 ? "text-center" : i === 4 ? "text-right" : ""
                    }`}
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-outline-variant/10">
              {patients.map((p) => (
                <tr key={p.pid} className="hover:bg-surface-container-highest transition-colors">
                  <td className="px-6 py-6">
                    <div className="flex items-center gap-4">
                      <div className="w-10 h-10 rounded-full bg-surface-container-highest flex items-center justify-center text-primary font-bold">
                        {p.initials}
                      </div>
                      <div>
                        <div className="font-body font-semibold text-on-surface">{p.name}</div>
                        <div className="text-xs text-on-surface-variant">ID: {p.pid}</div>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-6 font-body text-on-surface-variant">{p.date}</td>
                  <td className="px-6 py-6">
                    <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-bold ${p.resultClass}`}>
                      {p.result}
                    </span>
                  </td>
                  <td className="px-6 py-6 text-center">
                    <div className="flex flex-col items-center gap-1">
                      <span className="font-headline font-bold text-primary">{p.confidence}%</span>
                      <div className="w-20 h-1 bg-surface-container rounded-full overflow-hidden">
                        <div className="bg-primary h-full" style={{ width: `${p.confidence}%` }} />
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-6 text-right">
                    <div className="flex justify-end gap-3">
                      <button className="p-2 text-on-surface-variant hover:text-primary transition-colors">
                        <span className="material-symbols-outlined">visibility</span>
                      </button>
                      <button className="p-2 text-on-surface-variant hover:text-primary transition-colors">
                        <span className="material-symbols-outlined">download</span>
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        <div className="px-6 py-4 bg-surface-container-low flex items-center justify-between border-t border-outline-variant/10">
          <span className="font-label text-sm text-on-surface-variant">Showing 4 of 1,240 records</span>
          <div className="flex items-center gap-2">
            <button className="p-2 rounded-lg bg-surface-container-high text-on-surface-variant hover:bg-primary/20 hover:text-primary transition-all">
              <span className="material-symbols-outlined">chevron_left</span>
            </button>
            <button className="px-4 py-2 rounded-lg bg-primary-container text-on-primary-container font-bold text-sm">1</button>
            <button className="px-4 py-2 rounded-lg bg-surface-container-high text-on-surface-variant font-bold text-sm hover:bg-surface-container-highest">2</button>
            <button className="px-4 py-2 rounded-lg bg-surface-container-high text-on-surface-variant font-bold text-sm hover:bg-surface-container-highest">3</button>
            <button className="p-2 rounded-lg bg-surface-container-high text-on-surface-variant hover:bg-primary/20 hover:text-primary transition-all">
              <span className="material-symbols-outlined">chevron_right</span>
            </button>
          </div>
        </div>
      </Card>

      {/* Bottom Section */}
      <div className="mt-12 grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Historical Trends */}
        <div className="lg:col-span-2 bg-surface-container-low rounded-xl p-8">
          <h3 className="font-headline text-xl font-bold mb-6 flex items-center gap-2">
            <span className="material-symbols-outlined text-primary">insights</span>
            Historical Trends
          </h3>
          <div className="aspect-video w-full relative">
            <img
              className="w-full h-full object-cover rounded-xl opacity-40 mix-blend-screen"
              alt="Data visualization"
              src="https://lh3.googleusercontent.com/aida-public/AB6AXuDdgXisNdZBUc3-pXO3xx7mmbB5JnWU_uPPSr6x0m8hXUC2XQSIYQ8SjClSAnA8uN_UWao6wb03_QchsK7zUN0-k08twIlAdzEFnQiXip69Xc7trE3RWnPiB5AIM4Chxzn14wIUYHWgpoNLdHQKc-S-_RT70sYtJmgfPyI_cUw99T6zKxNW5GEmozMKuhPzhn0s3ZcbTrkrL4zs8_kaO6nDNTwiwwUtMP80avSqJZ2VxYdRhA5vwYKHoMvoxDFNcdWyLupMOqF6HKfn"
            />
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center">
                <div className="text-4xl font-headline font-extrabold text-primary mb-2">+12.4%</div>
                <div className="text-on-surface-variant font-body">Diagnostic Accuracy Growth</div>
              </div>
            </div>
          </div>
        </div>

        {/* Export */}
        <div className="bg-gradient-to-br from-primary-container/10 to-transparent border border-primary/10 rounded-xl p-8">
          <h3 className="font-headline text-xl font-bold mb-4">Export Batch</h3>
          <p className="font-body text-on-surface-variant mb-6 text-sm">
            Generate comprehensive reports for the selected time range (Oct 1 - Oct 31).
          </p>
          <div className="space-y-4">
            <button className="w-full py-3 bg-primary text-on-primary font-bold rounded-xl flex items-center justify-center gap-2 transition-transform active:scale-95">
              <span className="material-symbols-outlined">picture_as_pdf</span>
              Generate PDF Report
            </button>
            <button className="w-full py-3 bg-surface-container-highest text-on-surface font-bold rounded-xl flex items-center justify-center gap-2 hover:bg-surface-container transition-colors">
              <span className="material-symbols-outlined">csv</span>
              Export to CSV
            </button>
          </div>
        </div>
      </div>
    </main>
  );
}
