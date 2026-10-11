import { useQuery } from "@tanstack/react-query";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import { api } from "../services/api";
import { PageHeader, StatCard, StatusBadge, Loading, EmptyState } from "../components/ui";

export default function DashboardPage() {
  const { data, isLoading } = useQuery({
    queryKey: ["dashboard-summary"],
    queryFn: async () => (await api.get("/dashboard/summary")).data,
  });

  if (isLoading) return <Loading label="Loading dashboard…" />;

  const trend = (data?.evaluation_trend ?? []).map((t: any, i: number) => ({
    index: i + 1,
    rouge1: t.scores?.rouge1 ?? null,
    bleu: t.scores?.bleu ?? null,
  }));

  return (
    <div>
      <PageHeader title="Dashboard" description="Overview of your datasets, models, and experiments." />

      <div className="grid grid-cols-4 gap-4 mb-6">
        <StatCard label="Datasets" value={data?.counts?.datasets ?? 0} />
        <StatCard label="Models" value={data?.counts?.models ?? 0} />
        <StatCard label="Active Jobs" value={data?.counts?.active_jobs ?? 0} />
        <StatCard label="Experiments" value={data?.counts?.experiments ?? 0} />
      </div>

      <div className="grid grid-cols-3 gap-4">
        <div className="col-span-2 card">
          <p className="text-sm font-medium mb-4">Evaluation Trend (ROUGE-1 / BLEU)</p>
          {trend.length > 0 ? (
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={trend}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1a2130" />
                <XAxis dataKey="index" stroke="#64748b" fontSize={12} />
                <YAxis stroke="#64748b" fontSize={12} />
                <Tooltip contentStyle={{ background: "#141a24", border: "1px solid #26303f" }} />
                <Line type="monotone" dataKey="rouge1" stroke="#4f8cff" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="bleu" stroke="#3ecf8e" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-sm text-slate-500 py-16 text-center">
              No completed evaluations yet. Run a fine-tuning job to see trends here.
            </p>
          )}
        </div>

        <div className="card">
          <p className="text-sm font-medium mb-4">Best Evaluation Result</p>
          {data?.best_evaluation ? (
            <ul className="space-y-2">
              {Object.entries(data.best_evaluation).map(([k, v]) => (
                <li key={k} className="flex justify-between text-sm">
                  <span className="text-slate-400">{k}</span>
                  <span className="font-medium">{typeof v === "number" ? v.toFixed(3) : String(v)}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-slate-500">No evaluations yet.</p>
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 mt-6">
        <div className="card">
          <p className="text-sm font-medium mb-3">Recent Experiments</p>
          {data?.recent_experiments?.length ? (
            <ul className="space-y-2">
              {data.recent_experiments.map((e: any) => (
                <li key={e.id} className="flex items-center justify-between text-sm">
                  <span className="text-slate-300 truncate">{e.name}</span>
                  <StatusBadge status={e.status} />
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-slate-500">No experiments yet.</p>
          )}
        </div>

        <div className="card">
          <p className="text-sm font-medium mb-3">Latest Training Jobs</p>
          {data?.recent_jobs?.length ? (
            <ul className="space-y-2">
              {data.recent_jobs.map((j: any) => (
                <li key={j.id} className="flex items-center justify-between text-sm">
                  <span className="text-slate-300 truncate">{j.base_model_id}</span>
                  <StatusBadge status={j.status} />
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-slate-500">No training jobs yet.</p>
          )}
        </div>
      </div>
    </div>
  );
}
