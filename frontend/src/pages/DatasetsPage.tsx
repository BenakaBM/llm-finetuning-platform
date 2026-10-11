import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Upload } from "lucide-react";
import { api } from "../services/api";
import { Dataset, DatasetVersion } from "../types";
import { PageHeader, Loading, EmptyState } from "../components/ui";

export default function DatasetsPage() {
  const queryClient = useQueryClient();
  const [name, setName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [selected, setSelected] = useState<Dataset | null>(null);
  const [error, setError] = useState<string | null>(null);

  const { data: datasets, isLoading } = useQuery<Dataset[]>({
    queryKey: ["datasets"],
    queryFn: async () => (await api.get("/datasets")).data,
  });

  const { data: versions } = useQuery<DatasetVersion[]>({
    queryKey: ["dataset-versions", selected?.id],
    queryFn: async () => (await api.get(`/datasets/${selected!.id}/versions`)).data,
    enabled: !!selected,
  });

  const uploadMutation = useMutation({
    mutationFn: async () => {
      if (!file) throw new Error("Choose a file first.");
      const form = new FormData();
      form.append("file", file);
      form.append("name", name || file.name);
      const res = await api.post("/datasets", form, { headers: { "Content-Type": "multipart/form-data" } });
      return res.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["datasets"] });
      setFile(null);
      setName("");
      setError(null);
    },
    onError: (err: any) => setError(err?.response?.data?.error?.message ?? "Upload failed."),
  });

  return (
    <div>
      <PageHeader title="Datasets" description="Upload, validate, and version instruction-tuning datasets." />

      <div className="grid grid-cols-3 gap-6">
        <div className="col-span-1 space-y-4">
          <div className="card">
            <p className="text-sm font-medium mb-3">Upload dataset</p>
            {error && <p className="text-xs text-bad mb-2">{error}</p>}
            <label className="label">Name</label>
            <input className="input mb-3" value={name} onChange={(e) => setName(e.target.value)} placeholder="my-dataset" />
            <label className="label">File (.json / .jsonl / .csv / .parquet)</label>
            <input
              className="input mb-3"
              type="file"
              accept=".json,.jsonl,.csv,.parquet"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
            <button
              className="btn-primary w-full"
              onClick={() => uploadMutation.mutate()}
              disabled={uploadMutation.isPending || !file}
            >
              <Upload size={14} />
              {uploadMutation.isPending ? "Uploading…" : "Upload & validate"}
            </button>
          </div>

          {isLoading ? (
            <Loading />
          ) : datasets && datasets.length > 0 ? (
            <div className="card p-0 divide-y divide-base-800">
              {datasets.map((d) => (
                <button
                  key={d.id}
                  onClick={() => setSelected(d)}
                  className={`w-full text-left px-4 py-3 text-sm hover:bg-base-800 transition-colors ${
                    selected?.id === d.id ? "bg-base-800" : ""
                  }`}
                >
                  <p className="font-medium text-slate-200">{d.name}</p>
                  <p className="text-xs text-slate-500">{d.record_count} records · {d.dataset_format}</p>
                </button>
              ))}
            </div>
          ) : (
            <EmptyState title="No datasets yet" description="Upload a JSON/JSONL/CSV/Parquet file to get started." />
          )}
        </div>

        <div className="col-span-2">
          {!selected ? (
            <EmptyState title="Select a dataset" description="Choose a dataset on the left to see its statistics and quality report." />
          ) : (
            <div className="space-y-4">
              <div className="card">
                <p className="text-sm font-medium mb-1">{selected.name}</p>
                <p className="text-xs text-slate-500">{selected.description || "No description."}</p>
              </div>

              {versions?.map((v) => (
                <div key={v.id} className="card">
                  <div className="flex items-center justify-between mb-4">
                    <p className="text-sm font-medium">Version {v.version_number}</p>
                    <span className={`badge ${v.quality_report.score >= 80 ? "bg-good/15 text-good" : v.quality_report.score >= 60 ? "bg-warn/15 text-warn" : "bg-bad/15 text-bad"}`}>
                      Quality Score: {v.quality_report.score}/100
                    </span>
                  </div>
                  <div className="grid grid-cols-3 gap-4 mb-4">
                    {Object.entries(v.statistics).map(([k, val]) => (
                      <div key={k}>
                        <p className="text-xs text-slate-500 capitalize">{k.replace(/_/g, " ")}</p>
                        <p className="text-sm font-medium">{String(val)}</p>
                      </div>
                    ))}
                  </div>
                  {v.quality_report.warnings.length > 0 && (
                    <div className="bg-warn/10 border border-warn/30 rounded-lg p-3">
                      <p className="text-xs font-medium text-warn mb-1">Warnings</p>
                      <ul className="text-xs text-slate-300 space-y-0.5 list-disc list-inside">
                        {v.quality_report.warnings.map((w, i) => (
                          <li key={i}>{w}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
