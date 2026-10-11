import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Send } from "lucide-react";
import { api } from "../services/api";
import { RegisteredModel } from "../types";
import { PageHeader } from "../components/ui";

interface GenResult {
  response: string;
  input_tokens: number;
  output_tokens: number;
  latency_ms: number;
}

function ModelPanel({ label }: { label: string }) {
  const { data: models } = useQuery<RegisteredModel[]>({
    queryKey: ["models"],
    queryFn: async () => (await api.get("/models")).data,
  });

  const [modelVersionId, setModelVersionId] = useState("");
  const [baseModelId, setBaseModelId] = useState("");
  const [result, setResult] = useState<GenResult | null>(null);

  const allVersions = (models ?? []).flatMap((m) =>
    m.versions.map((v) => ({ id: v.id, label: `${m.name} · ${v.version_label}` }))
  );

  const generateMutation = useMutation({
    mutationFn: async (payload: { prompt: string; temperature: number; top_p: number; max_tokens: number }) => {
      const body: any = { ...payload };
      if (modelVersionId) body.model_version_id = modelVersionId;
      else body.base_model_id = baseModelId;
      const res = await api.post("/playground/generate", body);
      return res.data as GenResult;
    },
    onSuccess: setResult,
  });

  const [prompt, setPrompt] = useState("");
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(0.9);
  const [maxTokens, setMaxTokens] = useState(128);

  return (
    <div className="card space-y-3">
      <p className="text-sm font-medium">{label}</p>

      <select className="input" value={modelVersionId} onChange={(e) => { setModelVersionId(e.target.value); setBaseModelId(""); }}>
        <option value="">— select a fine-tuned version —</option>
        {allVersions.map((v) => <option key={v.id} value={v.id}>{v.label}</option>)}
      </select>
      <input
        className="input"
        placeholder="…or a base HF model ID, e.g. sshleifer/tiny-gpt2"
        value={baseModelId}
        onChange={(e) => { setBaseModelId(e.target.value); setModelVersionId(""); }}
      />

      <div className="grid grid-cols-3 gap-2">
        <div>
          <label className="label">Temp</label>
          <input className="input" type="number" step="0.1" value={temperature} onChange={(e) => setTemperature(parseFloat(e.target.value))} />
        </div>
        <div>
          <label className="label">Top P</label>
          <input className="input" type="number" step="0.05" value={topP} onChange={(e) => setTopP(parseFloat(e.target.value))} />
        </div>
        <div>
          <label className="label">Max Tokens</label>
          <input className="input" type="number" value={maxTokens} onChange={(e) => setMaxTokens(parseInt(e.target.value))} />
        </div>
      </div>

      {result && (
        <div className="bg-base-850 rounded-lg p-3 text-sm whitespace-pre-wrap">
          {result.response}
          <p className="text-xs text-slate-500 mt-2">
            {result.input_tokens} in / {result.output_tokens} out · {result.latency_ms.toFixed(0)}ms
          </p>
        </div>
      )}

      <div className="flex gap-2">
        <input className="input" placeholder="Enter a prompt…" value={prompt} onChange={(e) => setPrompt(e.target.value)} />
        <button
          className="btn-primary"
          disabled={!prompt || (!modelVersionId && !baseModelId) || generateMutation.isPending}
          onClick={() => generateMutation.mutate({ prompt, temperature, top_p: topP, max_tokens: maxTokens })}
        >
          <Send size={14} />
        </button>
      </div>
    </div>
  );
}

export default function PlaygroundPage() {
  return (
    <div>
      <PageHeader title="Model Playground" description="Test a model directly, or compare two side by side." />
      <div className="grid grid-cols-2 gap-6">
        <ModelPanel label="Model A" />
        <ModelPanel label="Model B" />
      </div>
    </div>
  );
}
