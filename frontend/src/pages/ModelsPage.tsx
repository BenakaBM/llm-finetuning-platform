import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../services/api";
import { RegisteredModel } from "../types";
import { PageHeader, Loading, EmptyState } from "../components/ui";

export default function ModelsPage() {
  const queryClient = useQueryClient();
  const [baseModelId, setBaseModelId] = useState("");
  const [name, setName] = useState("");

  const { data: models, isLoading } = useQuery<RegisteredModel[]>({
    queryKey: ["models"],
    queryFn: async () => (await api.get("/models")).data,
  });

  const registerMutation = useMutation({
    mutationFn: async () => (await api.post("/models", { name, base_model_id: baseModelId })).data,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["models"] });
      setName("");
      setBaseModelId("");
    },
  });

  return (
    <div>
      <PageHeader title="Models" description="Registry of base models and fine-tuned versions." />

      <div className="card mb-6">
        <p className="text-sm font-medium mb-3">Register a Hugging Face model</p>
        <div className="grid grid-cols-3 gap-3 items-end">
          <div>
            <label className="label">Display name</label>
            <input className="input" value={name} onChange={(e) => setName(e.target.value)} placeholder="TinyLlama Chat" />
          </div>
          <div>
            <label className="label">HF model ID</label>
            <input
              className="input"
              value={baseModelId}
              onChange={(e) => setBaseModelId(e.target.value)}
              placeholder="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            />
          </div>
          <button
            className="btn-primary"
            disabled={!name || !baseModelId || registerMutation.isPending}
            onClick={() => registerMutation.mutate()}
          >
            {registerMutation.isPending ? "Registering…" : "Register model"}
          </button>
        </div>
        <p className="text-xs text-slate-500 mt-2">
          Weights are not downloaded until the model is actually used for training or inference.
        </p>
      </div>

      {isLoading ? (
        <Loading />
      ) : models && models.length > 0 ? (
        <div className="grid grid-cols-2 gap-4">
          {models.map((m) => (
            <div key={m.id} className="card">
              <p className="text-sm font-medium">{m.name}</p>
              <p className="text-xs text-slate-500 mb-3">{m.base_model_id}</p>
              {m.versions.length > 0 ? (
                <table className="table-shell">
                  <thead>
                    <tr>
                      <th>Version</th>
                      <th>Method</th>
                      <th>Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {m.versions.map((v) => (
                      <tr key={v.id}>
                        <td>{v.version_label}</td>
                        <td>{v.training_method ?? "base"}</td>
                        <td>
                          {v.evaluation_scores?.rouge1 ? v.evaluation_scores.rouge1.toFixed(3) : "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <p className="text-xs text-slate-500">No fine-tuned versions yet.</p>
              )}
            </div>
          ))}
        </div>
      ) : (
        <EmptyState title="No models registered" description="Register a Hugging Face model above, or start a fine-tuning job." />
      )}
    </div>
  );
}
