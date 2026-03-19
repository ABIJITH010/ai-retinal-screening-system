import Card from "../../components/ui/Card";
import Button from "../../components/ui/Button";

export default function Profile() {
  const activity = [
    { label: "Patients Screened", value: "1,240" },
    { label: "Critical Cases identified", value: "84" },
    { label: "Accuracy Score", value: "99.2%" },
  ];

  return (
    <main className="min-h-screen pt-24 pb-32 px-6 md:px-12 max-w-5xl mx-auto">
      <div className="mb-10">
        <h1 className="text-4xl font-headline font-extrabold text-on-surface tracking-tight">
          Profile
        </h1>
        <p className="text-on-surface-variant text-lg">
          Manage your account information and view performance metrics.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Profile Card */}
        <div className="lg:col-span-2 space-y-8">
          <Card className="p-8">
            <div className="flex flex-col md:flex-row items-center gap-8 mb-10">
              <div className="w-32 h-32 rounded-full bg-surface-container-highest border-4 border-primary/20 flex items-center justify-center text-primary overflow-hidden relative group">
                <span className="material-symbols-outlined text-6xl group-hover:opacity-20 transition-opacity">account_circle</span>
                <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer bg-black/40">
                  <span className="material-symbols-outlined text-on-surface">photo_camera</span>
                </div>
              </div>
              <div className="text-center md:text-left">
                <h2 className="text-3xl font-headline font-bold text-on-surface">Dr. Aris Thorne</h2>
                <p className="text-primary font-bold tracking-widest uppercase text-xs mb-4">Senior Ophthalmologist</p>
                <div className="flex flex-wrap justify-center md:justify-start gap-4">
                  <div className="bg-surface-container px-3 py-1 rounded-full text-xs text-on-surface-variant flex items-center gap-1">
                    <span className="material-symbols-outlined text-sm">location_on</span>
                    Central Eye Hospital
                  </div>
                  <div className="bg-surface-container px-3 py-1 rounded-full text-xs text-on-surface-variant flex items-center gap-1">
                    <span className="material-symbols-outlined text-sm">verified</span>
                    Board Certified
                  </div>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
              <div className="space-y-2">
                <label className="text-xs uppercase tracking-widest text-on-surface-variant font-bold">Email Address</label>
                <div className="bg-surface-container-lowest border border-outline/10 rounded-xl px-4 py-3 text-on-surface font-body">
                  a.thorne@retinamax.ai
                </div>
              </div>
              <div className="space-y-2">
                <label className="text-xs uppercase tracking-widest text-on-surface-variant font-bold">Phone Number</label>
                <div className="bg-surface-container-lowest border border-outline/10 rounded-xl px-4 py-3 text-on-surface font-body">
                  +1 (555) 902-1440
                </div>
              </div>
            </div>

            <Button className="w-full md:w-auto px-8" icon="edit">
              Edit Profile
            </Button>
          </Card>
        </div>

        {/* Activity Summary */}
        <div className="space-y-6">
          <Card className="p-8 bg-gradient-to-br from-primary-container/10 to-transparent border border-primary/10">
            <h3 className="font-headline font-bold mb-6 flex items-center gap-2">
              <span className="material-symbols-outlined text-primary">insights</span>
              Performance
            </h3>
            <div className="space-y-6">
              {activity.map((stat) => (
                <div key={stat.label}>
                  <p className="text-xs text-on-surface-variant uppercase tracking-widest font-bold mb-1">{stat.label}</p>
                  <p className="text-2xl font-headline font-extrabold text-on-surface">{stat.value}</p>
                </div>
              ))}
            </div>
          </Card>

          <Card className="p-6">
            <h3 className="font-headline font-bold mb-4 text-sm">Account Security</h3>
            <button className="text-on-surface-variant hover:text-primary text-sm flex items-center gap-2 transition-colors">
              <span className="material-symbols-outlined text-sm">lock</span>
              Change password
            </button>
          </Card>
        </div>
      </div>
    </main>
  );
}
