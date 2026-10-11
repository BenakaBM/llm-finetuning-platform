import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from "recharts";
import { api } from "../services/api";
import { TrainingJob, TrainingMetricPoint } from "../types";
import { PageHeader, StatusBadge, Loading } from "../components/ui";

export default function TrainingJobDetailPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();
  const [live, setLive] = useState<{ status: string; current_step: number; current_epoch: number; train_loss: number | null; eval_loss: number | null } | null>(null);

  const { data: job, refetch: refetchJob } = useQuery<TrainingJob>({
    queryKey: ["training-job", jobId],
    queryFn: async () => (await api.get(`/training/jobs/${jobId}`)).data,
    refetchInterval: 5000,
  });

  const { data: metrics } = useQuery<TrainingMetricPoint[]>({
    queryKey: ["training-metrics", jobId],
    queryFn: async () => (await api.get(`/training/jobs/${jobId}/metrics`)).data,
    refetchInterval: 4000,
  });

  useEffect(() => {
    if (!jobId) return;
    const token = localStorage.getItem("access_token");
    const source = new EventSource(`/api/training/jobs/${jobId}/stream`);
    source.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      setLive(payload);
      if (["COMPLETED", "FAILED", "CANCELLED"].includes(payload.status)) {
        refetchJob();
        source.close();
      }
    };
    source.onerror = () => source.close();
    return () => source.close();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId]);

  async function handleCancel() {
    await api.post(`/training/jobs/${jobId}/cancel`);
    refetchJob();
  }

  async function handleDelete() {
    const isActive = ["PENDING", "PREPARING", "RUNNING"].includes(status);
    const message = isActive
      ? "This job is still running. Deleting it will forcibly stop it and remove its history. Continue?"
      : "Delete this training job and its history?";
    if (!window.confirm(message)) return;
    await api.delete(`/training/jobs/${jobId}`);
    navigate("/training-jobs");
  }

  if (!job) return <Loading label="Loading job…" />;

  const status = live?.status ?? job.status;
  const chartData = (metrics ?? []).map((m) => ({ step: m.step, train_loss: m.train_loss, eval_loss: m.eval_loss, lr: m.learning_rate }));

  return (
    <div>
      <PageHeader
        title={job.base_model_id}
        description="Live training monitoring"
        action={
          <div className="flex gap-2">
            {["RUNNING", "PREPARING", "PENDING"].includes(status) && (
              <button className="btn-secondary" onClick={handleCancel}>Cancel job</button>
            )}
            <button className="btn-secondary hover:text-bad hover:border-bad/50" onClick={handleDelete}>Delete job</button>
          </div>
        }
      />

      <div className="grid grid-cols-5 gap-4 mb-6">
        <div className="card"><p className="text-xs text-slate-500">Status</p><div className="mt-2"><StatusBadge status={status} /></div></div>
        <div className="card"><p className="text-xs text-slate-500">Epoch</p><p className="text-lg font-semibold mt-2">{(live?.current_epoch ?? job.current_epoch).toFixed?.(2) ?? live?.current_epoch ?? job.current_epoch}</p></div>
        <div className="card"><p className="text-xs text-slate-500">Step</p><p className="text-lg font-semibold mt-2">{live?.current_step ?? job.current_step}</p></div>
        <div className="card"><p className="text-xs text-slate-500">Train Loss</p><p className="text-lg font-semibold mt-2">{live?.train_loss?.toFixed?.(4) ?? "—"}</p></div>
        <div className="card"><p className="text-xs text-slate-500">Eval Loss</p><p className="text-lg font-semibold mt-2">{live?.eval_loss?.toFixed?.(4) ?? "—"}</p></div>
      </div>

      {job.error_message && (
        <div className="card border-bad/40 bg-bad/5 mb-6">
          <p className="text-sm text-bad font-medium mb-1">Training failed</p>
          <p className="text-xs text-slate-400 font-mono">{job.error_message}</p>
        </div>
      )}

      <div className="card mb-6">
        <p className="text-sm font-medium mb-4">Training / Validation Loss</p>
        {chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1a2130" />
              <XAxis dataKey="step" stroke="#64748b" fontSize={12} />
              <YAxis stroke="#64748b" fontSize={12} />
              <Tooltip contentStyle={{ background: "#141a24", border: "1px solid #26303f" }} />
              <Legend />
              <Line type="monotone" dataKey="train_loss" name="Train Loss" stroke="#4f8cff" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="eval_loss" name="Eval Loss" stroke="#3ecf8e" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-sm text-slate-500 text-center py-16">No metrics recorded yet.</p>
        )}
      </div>

      <div className="card">
        <p className="text-sm font-medium mb-3">Hardware</p>
        <pre className="text-xs text-slate-400 bg-base-850 rounded-lg p-3 overflow-x-auto">
          {JSON.stringify(job.hardware_info, null, 2)}
        </pre>
      </div>
    </div>
  );
}
