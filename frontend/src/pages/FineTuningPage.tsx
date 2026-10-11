import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { api } from "../services/api";
import { Dataset, DatasetVersion } from "../types";
import { PageHeader } from "../components/ui";

const STEPS = ["Dataset", "Base Model", "Training Method", "Hyperparameters", "Review & Start"];

const PRESET_MODELS = [
  "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "sshleifer/tiny-gpt2",
  "microsoft/phi-2",
  "Qwen/Qwen2.5-0.5B-Instruct",
];

export default function FineTuningPage() {
  const navigate = useNavigate();
  const [step, setStep] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const [datasetId, setDatasetId] = useState("");
  const [versionId, setVersionId] = useState("");
  const [baseModelId, setBaseModelId] = useState(PRESET_MODELS[0]);
  const [customModel, setCustomModel] = useState("");
  const [experimentName, setExperimentName] = useState("");
  const [method, setMethod] = useState<"LORA" | "QLORA">("LORA");
  const [maxTrainSamples, setMaxTrainSamples] = useState<number | "">("");

  const [config, setConfig] = useState({
    learning_rate: 2e-4,
    batch_size: 2,
    gradient_accumulation_steps: 4,
    epochs: 1,
    max_seq_length: 512,
    warmup_ratio: 0.03,
    weight_decay: 0.0,
    optimizer: "adamw_torch",
    scheduler: "cosine",
    seed: 42,
    lora_rank: 8,
    lora_alpha: 16,
    lora_dropout: 0.05,
    lora_target_modules: ["q_proj", "v_proj"],
    use_4bit: false,
    quant_type: "nf4",
    compute_dtype: "float16",
    double_quant: true,
  });

  const { data: datasets } = useQuery<Dataset[]>({
    queryKey: ["datasets"],
    queryFn: async () => (await api.get("/datasets")).data,
  });
  const { data: versions } = useQuery<DatasetVersion[]>({
    queryKey: ["dataset-versions", datasetId],
    queryFn: async () => (await api.get(`/datasets/${datasetId}/versions`)).data,
    enabled: !!datasetId,
  });

  const startMutation = useMutation({
    mutationFn: async () => {
      const resolvedModel = customModel.trim() || baseModelId;
      const res = await api.post(
        "/training/jobs",
        {
          dataset_version_id: versionId,
          base_model_id: resolvedModel,
          experiment_name: experimentName || `${resolvedModel.split("/").pop()}-run`,
          config: { ...config, method, use_4bit: method === "QLORA" },
          max_train_samples: maxTrainSamples === "" ? null : maxTrainSamples,
        },
        { params: maxTrainSamples === "" ? {} : { max_train_samples: maxTrainSamples } }
      );
      return res.data;
    },
    onSuccess: (job) => navigate(`/training-jobs/${job.id}`),
    onError: (err: any) => setError(err?.response?.data?.error?.message ?? "Failed to start training job."),
  });

  const canProceed = [!!versionId, !!(customModel.trim() || baseModelId), true, true, true][step];

  return (
    <div>
      <PageHeader title="Fine-Tuning" description="Configure and launch a LoRA/QLoRA fine-tuning job." />

      <div className="flex items-center gap-2 mb-6">
        {STEPS.map((s, i) => (
          <div key={s} className="flex items-center gap-2">
            <div
              className={`h-7 w-7 rounded-full flex items-center justify-center text-xs font-medium ${
                i === step ? "bg-accent text-white" : i < step ? "bg-good/20 text-good" : "bg-base-800 text-slate-500"
              }`}
            >
              {i + 1}
            </div>
            <span className={`text-sm ${i === step ? "text-slate-100" : "text-slate-500"}`}>{s}</span>
            {i < STEPS.length - 1 && <div className="w-8 h-px bg-base-700 mx-1" />}
          </div>
        ))}
      </div>

      <div className="card max-w-2xl">
        {error && <p className="text-sm text-bad mb-4">{error}</p>}

        {step === 0 && (
          <div className="space-y-4">
            <div>
              <label className="label">Dataset</label>
              <select className="input" value={datasetId} onChange={(e) => { setDatasetId(e.target.value); setVersionId(""); }}>
                <option value="">Select a dataset…</option>
                {datasets?.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name} ({d.record_count} records)
                  </option>
                ))}
              </select>
            </div>
            {datasetId && (
              <div>
                <label className="label">Version</label>
                <select className="input" value={versionId} onChange={(e) => setVersionId(e.target.value)}>
                  <option value="">Select a version…</option>
                  {versions?.map((v) => (
                    <option key={v.id} value={v.id}>
                      v{v.version_number} — quality {v.quality_report.score}/100
                    </option>
                  ))}
                </select>
              </div>
            )}
          </div>
        )}

        {step === 1 && (
          <div className="space-y-4">
            <div>
              <label className="label">Base model (preset)</label>
              <select className="input" value={baseModelId} onChange={(e) => setBaseModelId(e.target.value)}>
                {PRESET_MODELS.map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="label">Or specify a custom Hugging Face model ID</label>
              <input className="input" value={customModel} onChange={(e) => setCustomModel(e.target.value)} placeholder="org/model-name" />
            </div>
            <p className="text-xs text-slate-500">
              For CPU / no-GPU testing, use <code className="text-accent">sshleifer/tiny-gpt2</code>.
            </p>
          </div>
        )}

        {step === 2 && (
          <div className="space-y-4">
            <label className="label">Training method</label>
            <div className="grid grid-cols-2 gap-3">
              {(["LORA", "QLORA"] as const).map((m) => (
                <button
                  key={m}
                  onClick={() => setMethod(m)}
                  className={`card text-left ${method === m ? "border-accent ring-1 ring-accent" : ""}`}
                >
                  <p className="text-sm font-medium">{m}</p>
                  <p className="text-xs text-slate-500 mt-1">
                    {m === "LORA" ? "Low-rank adapters, fp16/fp32 base weights." : "4-bit quantized base weights + LoRA adapters (requires CUDA GPU)."}
                  </p>
                </button>
              ))}
            </div>
          </div>
        )}

        {step === 3 && (
          <div className="grid grid-cols-2 gap-4">
            {[
              ["learning_rate", "Learning Rate"],
              ["batch_size", "Batch Size"],
              ["gradient_accumulation_steps", "Gradient Accumulation"],
              ["epochs", "Epochs"],
              ["max_seq_length", "Max Sequence Length"],
              ["warmup_ratio", "Warmup Ratio"],
              ["weight_decay", "Weight Decay"],
              ["seed", "Seed"],
              ["lora_rank", "LoRA Rank"],
              ["lora_alpha", "LoRA Alpha"],
              ["lora_dropout", "LoRA Dropout"],
            ].map(([key, label]) => (
              <div key={key}>
                <label className="label">{label}</label>
                <input
                  className="input"
                  type="number"
                  step="any"
                  value={(config as any)[key]}
                  onChange={(e) => setConfig({ ...config, [key]: parseFloat(e.target.value) })}
                />
              </div>
            ))}
            <div>
              <label className="label">Max train samples (CPU demo cap, optional)</label>
              <input
                className="input"
                type="number"
                value={maxTrainSamples}
                onChange={(e) => setMaxTrainSamples(e.target.value === "" ? "" : parseInt(e.target.value))}
                placeholder="e.g. 20"
              />
            </div>
          </div>
        )}

        {step === 4 && (
          <div className="space-y-4">
            <div>
              <label className="label">Experiment name</label>
              <input className="input" value={experimentName} onChange={(e) => setExperimentName(e.target.value)} placeholder="my-first-lora-run" />
            </div>
            <div className="bg-base-850 rounded-lg p-3 text-xs text-slate-400 space-y-1">
              <p>Model: <span className="text-slate-200">{customModel.trim() || baseModelId}</span></p>
              <p>Method: <span className="text-slate-200">{method}</span></p>
              <p>Epochs: <span className="text-slate-200">{config.epochs}</span> · LR: <span className="text-slate-200">{config.learning_rate}</span></p>
              <p>LoRA rank/alpha: <span className="text-slate-200">{config.lora_rank}/{config.lora_alpha}</span></p>
            </div>
          </div>
        )}

        <div className="flex justify-between mt-6">
          <button className="btn-secondary" disabled={step === 0} onClick={() => setStep((s) => s - 1)}>
            Back
          </button>
          {step < STEPS.length - 1 ? (
            <button className="btn-primary" disabled={!canProceed} onClick={() => setStep((s) => s + 1)}>
              Next
            </button>
          ) : (
            <button className="btn-primary" disabled={startMutation.isPending} onClick={() => startMutation.mutate()}>
              {startMutation.isPending ? "Starting…" : "Start Training"}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
