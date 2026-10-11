import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import { api } from "../services/api";
import { PageHeader, Loading, StatusBadge } from "../components/ui";

export default function ExperimentDetailPage() {
  const { experimentId } = useParams<{ experimentId: string }>();

  const { data, isLoading } = useQuery({
    queryKey: ["experiment-detail", experimentId],
    queryFn: async () => (await api.get(`/experiments/${experimentId}`)).data,
    refetchInterval: 6000,
  });

  if (isLoading || !data) return <Loading label="Loading experiment…" />;

  const { experiment, training_curves, comparison, sample_predictions } = data;

  return (
    <div>
      <PageHeader
        title={experiment.name}
        description={`${experiment.base_model_id} · ${experiment.fine_tuning_method ?? "N/A"}`}
        action={
          experiment.report_path && (
            <a className="btn-secondary" href={`/storage/experiments/${experiment.id}/report.html`} target="_blank" rel="noreferrer">
              Download Report
            </a>
          )
        }
      />

      <div className="grid grid-cols-4 gap-4 mb-6">
        <div className="card"><p className="text-xs text-slate-500">Status</p><div className="mt-2"><StatusBadge status={experiment.status} /></div></div>
        <div className="card"><p className="text-xs text-slate-500">Duration</p><p className="text-lg font-semibold mt-2">{experiment.duration_seconds ? `${experiment.duration_seconds.toFixed(1)}s` : "—"}</p></div>
        <div className="card"><p className="text-xs text-slate-500">Seed</p><p className="text-lg font-semibold mt-2">{experiment.hyperparameters?.seed ?? "—"}</p></div>
        <div className="card"><p className="text-xs text-slate-500">Epochs</p><p className="text-lg font-semibold mt-2">{experiment.hyperparameters?.epochs ?? "—"}</p></div>
      </div>

      <div className="card mb-6">
        <p className="text-sm font-medium mb-4">Training Curves</p>
        {training_curves?.length > 0 ? (
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={training_curves}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1a2130" />
              <XAxis dataKey="step" stroke="#64748b" fontSize={12} />
              <YAxis stroke="#64748b" fontSize={12} />
              <Tooltip contentStyle={{ background: "#141a24", border: "1px solid #26303f" }} />
              <Line type="monotone" dataKey="train_loss" stroke="#4f8cff" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="eval_loss" stroke="#3ecf8e" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-sm text-slate-500 py-12 text-center">No training curves recorded.</p>
        )}
      </div>

      <div className="card mb-6">
        <p className="text-sm font-medium mb-4">Base vs Fine-Tuned Comparison</p>
        {comparison?.length > 0 ? (
          <table className="table-shell">
            <thead>
              <tr><th>Metric</th><th>Base</th><th>Fine-Tuned</th><th>Improvement</th></tr>
            </thead>
            <tbody>
              {comparison.map((row: any) => (
                <tr key={row.metric}>
                  <td>{row.metric}</td>
                  <td>{row.base}</td>
                  <td>{row.fine_tuned}</td>
                  <td className={row.improvement_pct >= 0 ? "text-good" : "text-bad"}>
                    {row.improvement_pct >= 0 ? "+" : ""}{row.improvement_pct}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="text-sm text-slate-500 py-8 text-center">Evaluation still running or not yet available.</p>
        )}
      </div>

      <div className="card">
        <p className="text-sm font-medium mb-4">Sample Predictions</p>
        <div className="space-y-4">
          {(sample_predictions ?? []).slice(0, 5).map((s: any, i: number) => (
            <div key={i} className="bg-base-850 rounded-lg p-3 text-xs space-y-1">
              <p className="text-slate-500">Prompt</p>
              <p className="text-slate-300 whitespace-pre-wrap">{s.prompt}</p>
              <p className="text-slate-500 mt-2">Reference</p>
              <p className="text-good/80 whitespace-pre-wrap">{s.reference}</p>
              <p className="text-slate-500 mt-2">Prediction</p>
              <p className="text-accent/90 whitespace-pre-wrap">{s.prediction}</p>
            </div>
          ))}
          {(!sample_predictions || sample_predictions.length === 0) && (
            <p className="text-sm text-slate-500">No sample predictions yet.</p>
          )}
        </div>
      </div>
    </div>
  );
}
