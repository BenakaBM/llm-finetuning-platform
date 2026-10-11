import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../services/api";
import { Experiment } from "../types";
import { PageHeader, StatusBadge, Loading, EmptyState } from "../components/ui";

export default function ExperimentsPage() {
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [comparison, setComparison] = useState<any | null>(null);

  const { data: experiments, isLoading } = useQuery<Experiment[]>({
    queryKey: ["experiments"],
    queryFn: async () => (await api.get("/experiments")).data,
  });

  const compareMutation = useMutation({
    mutationFn: async () => (await api.post("/experiments/compare", selectedIds)).data,
    onSuccess: setComparison,
  });

  function toggle(id: string) {
    setSelectedIds((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : prev.length < 4 ? [...prev, id] : prev));
  }

  return (
    <div>
      <PageHeader
        title="Experiments"
        description="Every training run automatically becomes a tracked experiment."
        action={
          <button className="btn-primary" disabled={selectedIds.length < 2} onClick={() => compareMutation.mutate()}>
            Compare selected ({selectedIds.length})
          </button>
        }
      />

      {isLoading ? (
        <Loading />
      ) : experiments && experiments.length > 0 ? (
        <div className="card p-0 overflow-hidden mb-6">
          <table className="table-shell">
            <thead>
              <tr className="[&>th]:px-4">
                <th></th>
                <th>Name</th>
                <th>Model</th>
                <th>Method</th>
                <th>Status</th>
                <th>Duration</th>
              </tr>
            </thead>
            <tbody>
              {experiments.map((e) => (
                <tr key={e.id} className="hover:bg-base-800">
                  <td className="px-4">
                    <input type="checkbox" checked={selectedIds.includes(e.id)} onChange={() => toggle(e.id)} />
                  </td>
                  <td className="px-4">
                    <Link to={`/experiments/${e.id}`} className="text-slate-200 hover:text-accent">{e.name}</Link>
                  </td>
                  <td className="px-4 text-slate-400">{e.base_model_id}</td>
                  <td className="px-4 text-slate-400">{e.fine_tuning_method ?? "—"}</td>
                  <td className="px-4"><StatusBadge status={e.status} /></td>
                  <td className="px-4 text-slate-400">{e.duration_seconds ? `${e.duration_seconds.toFixed(1)}s` : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <EmptyState title="No experiments yet" description="Start a fine-tuning job to create your first experiment." />
      )}

      {comparison && (
        <div className="card">
          <p className="text-sm font-medium mb-4">Comparison</p>
          <table className="table-shell">
            <thead>
              <tr>
                <th>Experiment</th>
                <th>Method</th>
                <th>ROUGE-1</th>
                <th>BLEU</th>
                <th>Exact Match</th>
                <th>Avg Latency</th>
              </tr>
            </thead>
            <tbody>
              {comparison.rows.map((r: any) => (
                <tr key={r.experiment_id} className={r.experiment_id === comparison.best_experiment_id ? "bg-good/5" : ""}>
                  <td>{r.name} {r.experiment_id === comparison.best_experiment_id && <span className="badge bg-good/15 text-good ml-1">best</span>}</td>
                  <td>{r.method}</td>
                  <td>{r.scores?.rouge1?.toFixed(3) ?? "—"}</td>
                  <td>{r.scores?.bleu?.toFixed(3) ?? "—"}</td>
                  <td>{r.scores?.exact_match?.toFixed(3) ?? "—"}</td>
                  <td>{r.latency_ms_avg ? `${r.latency_ms_avg.toFixed(0)}ms` : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
