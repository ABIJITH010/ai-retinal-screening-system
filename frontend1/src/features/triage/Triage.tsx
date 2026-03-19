const triageColumns = [
  {
    title: "Critical",
    count: "03",
    barColor: "bg-error",
    badgeClass: "bg-error-container text-on-error-container",
    cards: [
      {
        label: "Imminent Threat",
        labelColor: "text-error",
        name: "Elias Thorne",
        pid: "#RE-9021",
        insight: "Bilateral papilledema with hemorrhaging. High intracranial pressure detected.",
        timeAgo: "2m ago",
        footer: (
          <div className="flex -space-x-2">
            <div className="w-8 h-8 rounded-full bg-surface-container-highest flex items-center justify-center outline outline-2 outline-surface-container-low">
              <span className="material-symbols-outlined text-sm">visibility</span>
            </div>
            <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center outline outline-2 outline-surface-container-low">
              <span className="material-symbols-outlined text-sm text-primary">psychiatry</span>
            </div>
          </div>
        ),
      },
      {
        label: "Urgent Surgery",
        labelColor: "text-error",
        name: "Sarah Jenkins",
        pid: "#RE-4458",
        insight: "Macula-on retinal detachment. Immediate intervention required to save vision.",
        timeAgo: "14m ago",
        footer: (
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-error" style={{ fontVariationSettings: "'FILL' 1" }}>
              priority_high
            </span>
            <span className="text-xs font-bold text-error">Tier 1</span>
          </div>
        ),
      },
    ],
  },
  {
    title: "High Risk",
    count: "05",
    barColor: "bg-tertiary",
    badgeClass: "bg-tertiary-container text-on-tertiary-container",
    cards: [
      {
        label: "Monitoring Needed",
        labelColor: "text-tertiary",
        name: "Marcus Vane",
        pid: "#RE-1120",
        insight: "Severe Non-proliferative Diabetic Retinopathy. Neovascularization progression.",
        timeAgo: "45m ago",
        footer: (
          <div className="bg-primary-container/10 px-3 py-1 rounded-full flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-primary animate-pulse" />
            <span className="text-[10px] font-bold text-primary">AI ANALYZING</span>
          </div>
        ),
      },
      {
        label: "Acute Glaucoma",
        labelColor: "text-tertiary",
        name: "Elena Rossi",
        pid: "#RE-8871",
        insight: "Elevated IOP markers and optic disc cupping. Immediate pressure check required.",
        timeAgo: "1h ago",
        footer: (
          <div className="flex -space-x-1">
            <img
              className="w-6 h-6 rounded-full outline outline-2 outline-surface-container-low"
              alt="Doctor avatar"
              src="https://lh3.googleusercontent.com/aida-public/AB6AXuDPQe5CpqdmJ8xCGsFPM3Ue_61svU0U1RC4_2zBaYYFlZl6WQAAzgmvTbmFdpqifbppUxz7ei7tpZBT7bYDP4UTYiDiQCHiBN4uMv7HuyJRZspYwXx_1jUqb9mHsyXgomgMc260X8hnhBmPMe3T69KesSkAVw9-wo563kBB4yLoWENlzBDkavDTeu_JthtOhpru_7TVzwMCsw3DUsPoEPa9Z3HrPZWhLvpdqyiLJx5QPWgC246MbA5R289kmEKJKHObdG4GtWg_IyBg"
            />
            <div className="w-6 h-6 rounded-full bg-surface-container-highest flex items-center justify-center outline outline-2 outline-surface-container-low text-[8px] font-bold">
              +2
            </div>
          </div>
        ),
      },
    ],
  },
  {
    title: "Moderate",
    count: "12",
    barColor: "bg-primary",
    badgeClass: "bg-primary-container text-on-primary-container",
    cards: [
      {
        label: "Follow-Up",
        labelColor: "text-primary",
        name: "David Chen",
        pid: "#RE-3329",
        insight: "Mild AMD progression noted in left eye. Schedule review within 48 hours.",
        timeAgo: "3h ago",
        footer: (
          <span className="text-xs text-on-surface-variant font-label flex items-center gap-1">
            <span className="material-symbols-outlined text-sm">schedule</span>
            Next: 3:30 PM
          </span>
        ),
      },
    ],
  },
  {
    title: "Stable",
    count: "24",
    barColor: "bg-outline",
    badgeClass: "bg-surface-container-high text-on-surface-variant",
    cards: [
      {
        label: "Routine Clear",
        labelColor: "text-on-surface-variant",
        name: "Linda Thompson",
        pid: "#RE-0092",
        insight: "No acute findings. Return for annual screening as planned.",
        timeAgo: "Yesterday",
        footer: (
          <div className="flex items-center gap-1 text-primary text-xs font-bold">
            <span className="material-symbols-outlined text-sm" style={{ fontVariationSettings: "'FILL' 1" }}>
              check_circle
            </span>
            ARCHIVED
          </div>
        ),
      },
    ],
  },
];

export default function Triage() {
  return (
    <main className="pt-24 pb-32 px-6 md:px-12 max-w-[1600px] mx-auto">
      {/* Header */}
      <div className="mb-10 flex flex-col md:flex-row md:items-end justify-between gap-6">
        <div>
          <h1 className="font-headline text-4xl font-extrabold text-on-surface tracking-tight mb-2">
            Triage Queue
          </h1>
          <p className="text-on-surface-variant font-body">
            Real-time patient prioritization based on AI-detected retinal pathologies.
          </p>
        </div>
        <div className="flex gap-3">
          <button className="bg-surface-container-high px-5 py-2.5 rounded-xl text-on-surface font-label flex items-center gap-2 hover:bg-surface-container-highest transition-colors">
            <span className="material-symbols-outlined text-xl">filter_list</span>
            Refine View
          </button>
          <button className="bg-gradient-to-r from-primary to-primary-container px-6 py-2.5 rounded-xl text-on-primary-container font-bold font-label flex items-center gap-2 shadow-lg shadow-primary/10">
            <span className="material-symbols-outlined text-xl">add</span>
            Manual Entry
          </button>
        </div>
      </div>

      {/* Kanban Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6 items-start">
        {triageColumns.map((col) => (
          <div key={col.title} className="flex flex-col gap-4">
            {/* Column header */}
            <div className="flex items-center justify-between px-2 mb-2">
              <div className="flex items-center gap-2">
                <div className={`w-2 h-6 ${col.barColor} rounded-full`} />
                <h2 className="font-headline text-lg font-bold text-on-surface">{col.title}</h2>
              </div>
              <span className={`${col.badgeClass} px-2 py-0.5 rounded-md text-xs font-bold`}>
                {col.count}
              </span>
            </div>

            {/* Cards */}
            <div className={`space-y-4 ${col.title === "Stable" ? "opacity-70" : ""}`}>
              {col.cards.map((card) => (
                <div
                  key={card.pid}
                  className="bg-surface-container-low p-5 rounded-xl hover:bg-surface-container-high transition-all duration-300 group"
                >
                  <div className="flex justify-between items-start mb-4">
                    <div className={`text-xs font-label ${card.labelColor} uppercase tracking-widest font-bold`}>
                      {card.label}
                    </div>
                    <span className="material-symbols-outlined text-on-surface-variant group-hover:text-primary transition-colors cursor-pointer">
                      more_vert
                    </span>
                  </div>
                  <h3 className="font-headline text-xl font-semibold mb-1">{card.name}</h3>
                  <p className="text-xs text-on-surface-variant font-label mb-4">ID: {card.pid}</p>
                  <div className="bg-surface-container-lowest p-3 rounded-lg mb-4 outline outline-1 outline-outline/10">
                    {card.label !== "Routine Clear" && (
                      <p className="text-xs text-on-surface-variant mb-1">AI INSIGHT</p>
                    )}
                    <p className="text-sm font-body">{card.insight}</p>
                  </div>
                  <div className="flex items-center justify-between mt-6 pt-4">
                    {card.footer}
                    <span className="text-xs text-on-surface-variant font-label">{card.timeAgo}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </main>
  );
}
