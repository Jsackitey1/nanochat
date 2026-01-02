
E&M Tutor Micro-LLM Implementation Plan
Goal
Create and fine-tune a small LLM (nanochat) to act as an E&M Tutor that can explain concepts, solve problems step-by-step, and perform unit checks. Scope: Option B (Electrostatics + Gauss + Potential + Capacitors).

User Review Required
IMPORTANT

Compute Constraints: Training a 561M parameter model on a Mac CPU is not feasible for a 4-hour "speedrun". We will use a "micro" configuration (depth=4, reduced context) similar to 
dev/runcpu.sh
 to demonstrate the mechanics of the pipeline. The resulting model will be "dumb" but will effectively show the training process.

Proposed Changes
Dataset Creation
[NEW] 
em_tutor_dataset_builder.py
Identify 60-100 problem types covering:
Coulomb's Law (1D, 2D)
Electric Field (Point charges, continuous lines/rings/disks)
Gauss's Law (Sphere, Cylinder, Plane symmetries)
Electric Potential (V from E, E from V, point charges)
Capacitance (Parallel plate, spherical, cylindrical, energy, dielectrics)
Script to validate and write JSONL files.
Generation Strategy: We will use a combination of template-based generation and synthetic data generation (using the agent's capabilities) to populate em_tutor_train.jsonl (200 examples) and em_tutor_test.jsonl (50 examples).
Baseline Training
[NEW] 
speedrun_micro_local.sh
Based on 
dev/runcpu.sh
.
Reduces model depth to 4 or 6 layers.
Reduces sequence length to 512 or 1024.
Uses uv sync --extra cpu.
Runs the full pipeline: Tokenizer -> Base Train -> Mid Train -> SFT (Baseline).
Fine-Tuning Script
[NEW] 
scripts/em_finetune.py
Clones 
scripts/chat_sft.py
.
Modifications:
source defaults to the output of the baseline SFT (or we can start from 'mid' and mix E&M data, but "fine-tuning the baseline" implies a second stage).
Decision: We will perform a second SFT stage starting from the Baseline SFT model to specialize it.
train_ds: Uses CustomJSON to load ONLY em_tutor_train.jsonl (or a mixture with 90% E&M, 10% chat to prevent catastrophic forgetting, but for a simple demo, pure E&M is fine).
target_examples_per_step: Adjusted for CPU training.
Evaluation
[NEW] 
em_eval.py
Script to run inference on em_tutor_test.jsonl.
Scores:
Valid Format (Did it verify units?)
Correct Answer (Exact number matching within tolerance).
Hallucinations (Manual check or heuristic).
Verification Plan
Automated Tests
Run em_tutor_dataset_builder.py to ensure valid JSONL generation.
Run speedrun_micro_local.sh and ensure it completes without error (even if loss is high).
Run scripts/em_finetune.py and ensure it saves a new checkpoint.
Manual Verification
Inspect generated em_tutor_train.jsonl for quality.
Chat with the final model using 
scripts/chat_cli.py
 to see if it attempts to solve an E&M problem (even if poorly).
