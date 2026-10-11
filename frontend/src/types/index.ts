export interface Dataset {
  id: string;
  name: string;
  description: string | null;
  file_format: string;
  dataset_format: "instruction" | "conversational";
  record_count: number;
  created_at: string;
}

export interface DatasetVersion {
  id: string;
  dataset_id: string;
  version_number: number;
  statistics: Record<string, number | string>;
  quality_report: { score: number; warnings: string[] };
  created_at: string;
}

export interface ModelVersion {
  id: string;
  version_label: string;
  training_method: string | null;
  evaluation_scores: Record<string, number>;
  artifact_path: string | null;
}

export interface RegisteredModel {
  id: string;
  name: string;
  base_model_id: string;
  architecture: string | null;
  parameter_count: string | null;
  context_length: number | null;
  versions: ModelVersion[];
}

export type JobStatus =
  | "PENDING"
  | "PREPARING"
  | "RUNNING"
  | "COMPLETED"
  | "FAILED"
  | "CANCELLED";

export interface TrainingJob {
  id: string;
  status: JobStatus;
  base_model_id: string;
  current_epoch: number;
  current_step: number;
  total_steps: number;
  output_dir: string | null;
  error_message: string | null;
  hardware_info: Record<string, unknown>;
  started_at: string | null;
  completed_at: string | null;
  created_at: string;
}

export interface TrainingMetricPoint {
  step: number;
  epoch: number;
  train_loss: number | null;
  eval_loss: number | null;
  learning_rate: number | null;
  recorded_at: string;
}

export interface Experiment {
  id: string;
  name: string;
  base_model_id: string;
  fine_tuning_method: string | null;
  hyperparameters: Record<string, unknown>;
  hardware: Record<string, unknown>;
  duration_seconds: number | null;
  status: string;
  report_path: string | null;
  created_at: string;
}

export interface ComparisonRow {
  metric: string;
  base: number;
  fine_tuned: number;
  improvement_pct: number | null;
}
