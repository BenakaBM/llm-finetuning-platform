# Fine-Tuning (LoRA / QLoRA)

## What actually happens

`app/ml/training/trainer.py` performs real supervised fine-tuning using
Hugging Face `Transformers` + `PEFT`:

1. Load the base causal LM and tokenizer from the Hugging Face Hub
   (cached locally after first download).
2. Wrap the model with a `peft.LoraConfig` (`get_peft_model`) so only a
   small number of low-rank adapter parameters are trainable.
3. If QLoRA is selected and a CUDA GPU is available, the base model is
   loaded in 4-bit precision via `BitsAndBytesConfig` (`bnb_4bit_quant_type`,
   `bnb_4bit_compute_dtype`, `bnb_4bit_use_double_quant` are all exposed as
   configuration options).
4. Records are formatted into a single prompt/response string
   (`### Instruction: ... ### Response: ...`) and tokenized.
5. A standard `transformers.Trainer` runs real gradient descent, logging
   `loss`, `eval_loss`, `learning_rate`, `step`, and `epoch` at every
   logging step via a `TrainerCallback`, which is what powers the live
   training charts in the UI.
6. The resulting LoRA adapter (not the full base model) is saved to
   `storage/checkpoints/<job_id>/final/` — this keeps checkpoints small
   even for large base models.

## LoRA vs QLoRA

| | LoRA | QLoRA |
|---|---|---|
| Base weights | fp16/fp32 | 4-bit (nf4/fp4) |
| Requires GPU | No (slow on CPU) | Yes |
| Memory footprint | Higher | Lower |
| Use case | Small/medium models, CPU demo | Larger models on limited GPU memory |

## Running on CPU (no GPU)

Set `base_model_id` to a tiny model (e.g. `sshleifer/tiny-gpt2`) and pass
`max_train_samples` (exposed in the Fine-Tuning UI and the
`POST /api/training/jobs` query param) to cap the training set size. This
keeps a *real* training run — same code path, same PEFT/LoRA machinery —
finishing in well under a minute. `scripts/demo.py` does exactly this.

## Resource validation

Before a job is queued, `app/ml/training/hardware.py` estimates GPU
memory requirements from the selected batch size, sequence length, and
quantization mode, and compares against the detected GPU's memory. If
infeasible, the API returns a `409 RESOURCE_ERROR` with concrete
suggestions (enable QLoRA, reduce batch size, reduce sequence length,
increase gradient accumulation instead) rather than letting the job start
and crash out-of-memory.

## Job lifecycle

```
PENDING -> PREPARING -> RUNNING -> COMPLETED
                              \-> FAILED
                              \-> CANCELLED
```

Cancellation is cooperative: the worker checks the job's status at the
end of every epoch (`should_stop` callback) and stops training early if
the job has been marked `CANCELLED`.
