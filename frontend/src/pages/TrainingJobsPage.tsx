import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { Trash2 } from "lucide-react";
import { api } from "../services/api";
import { TrainingJob } from "../types";
import { PageHeader, StatusBadge, Loading, EmptyState } from "../components/ui";

export default function TrainingJobsPage() {
  const queryClient = useQueryClient();

  const { data: jobs, isLoading } = useQuery<TrainingJob[]>({
    queryKey: ["training-jobs"],
    queryFn: async () => (await api.get("/training/jobs")).data,
    refetchInterval: 5000,
  });

  const deleteMutation = useMutation({
    mutationFn: async (jobId: string) => api.delete(`/training/jobs/${jobId}`),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["training-jobs"] }),
  });

  function handleDelete(e: React.MouseEvent, job: TrainingJob) {
    e.preventDefault();
    e.stopPropagation();
    const isActive = ["PENDING", "PREPARING", "RUNNING"].includes(job.status);
    const message = isActive
      ? "This job is still running. Deleting it will forcibly stop it and remove its history. Continue?"
      : "Delete this training job and its history?";
    if (window.confirm(message)) {
      deleteMutation.mutate(job.id);
    }
  }

  return (
    <div>
      <PageHeader title="Training Jobs" description="Monitor active and past fine-tuning runs." />
      {isLoading ? (
        <Loading />
      ) : jobs && jobs.length > 0 ? (
        <div className="card p-0 overflow-hidden">
          <table className="table-shell">
            <thead>
              <tr className="[&>th]:px-4">
                <th>Model</th>
                <th>Status</th>
                <th>Step</th>
                <th>Epoch</th>
                <th>Started</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {jobs.map((j) => (
                <tr key={j.id} className="hover:bg-base-800">
                  <td className="px-4">
                    <Link to={`/training-jobs/${j.id}`} className="text-slate-200 hover:text-accent">
                      {j.base_model_id}
                    </Link>
                  </td>
                  <td className="px-4"><StatusBadge status={j.status} /></td>
                  <td className="px-4">{j.current_step}</td>
                  <td className="px-4">{j.current_epoch}</td>
                  <td className="px-4 text-slate-500">{j.started_at ? new Date(j.started_at).toLocaleString() : "—"}</td>
                  <td className="px-4">
                    <button
                      onClick={(e) => handleDelete(e, j)}
                      disabled={deleteMutation.isPending}
                      title="Delete job"
                      className="text-slate-500 hover:text-bad transition-colors p-1"
                    >
                      <Trash2 size={15} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <EmptyState title="No training jobs yet" description="Launch one from the Fine-Tuning page." />
      )}
    </div>
  );
}
