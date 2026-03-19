import Card from "../../components/ui/Card";

const notifications = [
  {
    id: 1,
    title: "High-risk patient detected",
    message: "Critical pathology detected in Case #RE-9021. Immediate review required.",
    timestamp: "2 minutes ago",
    unread: true,
  },
  {
    id: 2,
    title: "Daily Screening Report ready",
    message: "Your daily summary for Oct 24, 2023 is now available for download.",
    timestamp: "1 hour ago",
    unread: false,
  },
  {
    id: 3,
    title: "System Update Complete",
    message: "EfficientNet model version 2.4 has been successfully deployed.",
    timestamp: "5 hours ago",
    unread: false,
  },
  {
    id: 4,
    title: "New Triage Entry",
    message: "Patient Sarah Jenkins has been added to the Critical queue.",
    timestamp: "Yesterday",
    unread: false,
  },
];

export default function Notifications() {
  return (
    <main className="min-h-screen pt-24 pb-32 px-6 md:px-12 max-w-4xl mx-auto">
      <div className="flex justify-between items-center mb-10">
        <div>
          <h1 className="text-4xl font-headline font-extrabold text-on-surface tracking-tight">
            Notifications
          </h1>
          <p className="text-on-surface-variant text-lg">
            Stay updated with real-time AI alerts and system status.
          </p>
        </div>
        <button className="text-sm font-bold text-primary hover:text-primary-container transition-colors">
          Mark all as read
        </button>
      </div>

      <div className="space-y-4">
        {notifications.map((item) => (
          <Card key={item.id} className="p-5 hover:bg-surface-container-high transition-all cursor-pointer group relative">
            <div className="flex justify-between items-start">
              <div className="flex gap-4">
                <div className={`mt-1.5 w-2 h-2 rounded-full flex-shrink-0 ${item.unread ? 'bg-primary animate-pulse' : 'bg-transparent'}`} />
                <div>
                  <h3 className={`font-headline font-bold mb-1 ${item.unread ? 'text-on-surface' : 'text-on-surface-variant'}`}>
                    {item.title}
                  </h3>
                  <p className="text-on-surface-variant text-sm leading-relaxed mb-2">
                    {item.message}
                  </p>
                  <span className="text-xs text-outline font-medium">{item.timestamp}</span>
                </div>
              </div>
              <button className="text-outline hover:text-primary transition-colors opacity-0 group-hover:opacity-100">
                <span className="material-symbols-outlined text-xl">delete</span>
              </button>
            </div>
          </Card>
        ))}
      </div>
    </main>
  );
}
