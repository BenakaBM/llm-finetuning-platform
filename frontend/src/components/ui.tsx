import { ReactNode } from "react";
import { Loader2 } from "lucide-react";

export function StatCard({ label, value, hint }: { label: string; value: ReactNode; hint?: string }) {
  return (
    <div className="card">
      <p className="text-xs uppercase tracking-wide text-slate-500">{label}</p>
      <p className="text-2xl font-semibold mt-2">{value}</p>
      {hint && <p className="text-xs text-slate-500 mt-1">{hint}</p>}
    </div>
  );
}

export function StatusBadge({ status }: { status: string }) {
  const map: Record<string, string> = {
    COMPLETED: "bg-good/15 text-good",
    RUNNING: "bg-accent/15 text-accent",
    PREPARING: "bg-accent/15 text-accent",
    PENDING: "bg-slate-500/15 text-slate-300",
    FAILED: "bg-bad/15 text-bad",
    CANCELLED: "bg-slate-500/15 text-slate-400",
  };
  return <span className={`badge ${map[status] ?? "bg-slate-500/15 text-slate-300"}`}>{status}</span>;
}

export function EmptyState({ title, description }: { title: string; description: string }) {
  return (
    <div className="card text-center py-14">
      <p className="text-sm font-medium text-slate-200">{title}</p>
      <p className="text-sm text-slate-500 mt-1">{description}</p>
    </div>
  );
}

export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-slate-500 py-8 justify-center">
      <Loader2 size={16} className="animate-spin" />
      {label}
    </div>
  );
}

export function PageHeader({ title, description, action }: { title: string; description?: string; action?: ReactNode }) {
  return (
    <div className="flex items-start justify-between mb-6">
      <div>
        <h1 className="text-xl font-semibold text-slate-50">{title}</h1>
        {description && <p className="text-sm text-slate-500 mt-1">{description}</p>}
      </div>
      {action}
    </div>
  );
}
