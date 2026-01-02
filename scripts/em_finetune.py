"""
Finetune the chat model on E&M Physics problems.
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import wandb
import torch
import torch.distributed as dist
from contextlib import nullcontext

from nanochat.common import compute_init, compute_cleanup, get_base_dir, print0, DummyWandb, autodetect_device_type
from nanochat.checkpoint_manager import load_model, save_checkpoint
from nanochat.engine import Engine
from scripts.chat_eval import run_chat_eval

from tasks.common import TaskMixture
from tasks.customjson import CustomJSON

# -----------------------------------------------------------------------------
# E&M SFT Hyperparameters
run = "dummy"
source = "sft" # Start from the general SFT model
model_tag = "d4" # Matches the depth used in speedrun_micro_local.sh
step = None
device_type = ""
dtype = "bfloat16"
device_batch_size = 4
num_epochs = 3 
num_iterations = -1
target_examples_per_step = 16
unembedding_lr = 1e-4 # Lower LR for fine-tuning
embedding_lr = 1e-4
matrix_lr = 1e-4
weight_decay = 0.01
init_lr_frac = 1.0 # Constant LR or small warmup
eval_every = 10
eval_steps = 10
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
ptdtype = torch.float32 if dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

# wandb logging init
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-em", name=run, config={})

# Load the model directly from the SFT checkpoint
# We need to find the latest checkpoint in the d4 directory if not specified
# But load_model handles source="sft" by looking in chatsft_checkpoints/d{depth}
model, tokenizer, meta = load_model(source, device, phase="train", model_tag=model_tag, step=step)
engine = Engine(model, tokenizer)

# -----------------------------------------------------------------------------
# Dataset
train_path = "em_tutor_train.jsonl"
test_path = "em_tutor_test.jsonl"

train_ds = CustomJSON(filepath=train_path)
val_ds = CustomJSON(filepath=test_path)

print0(f"Loaded {len(train_ds)} training examples and {len(val_ds)} validation examples.")

# -----------------------------------------------------------------------------
# DataLoader (Simplified from chat_sft.py)

def sft_data_generator(dataset, batch_size):
    pad_token_id = tokenizer.encode_special("<|assistant_end|>")
    
    def collate_and_yield(batch):
        nrows = len(batch)
        ncols = max(len(ids) for ids, mask in batch) - 1
        inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
        targets = torch.full((nrows, ncols), -1, dtype=torch.long)
        for i, (ids, mask) in enumerate(batch):
            n = len(ids)
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            inputs[i, :n-1] = ids_tensor[:-1]
            row_targets = ids_tensor[1:]
            mask_tensor = torch.tensor(mask[1:], dtype=torch.long)
            row_targets[mask_tensor == 0] = -1
            targets[i, :n-1] = row_targets
        inputs = inputs.to(device)
        targets = targets.to(device)
        return inputs, targets

    batch = []
    # Infinite loop for generator
    while True:
        # Shuffle for training
        indices = list(range(len(dataset)))
        import random
        random.shuffle(indices)
        
        for i in indices:
            if i % ddp_world_size != ddp_rank: continue
            
            doc = dataset[i]
            # CustomJSON returns the doc dict directly. 
            # tokenizer.render_conversation_from_json matches CustomJSON format?
            # CustomJSON yields dicts with "prompt" and "response" (or "messages").
            # Does tokenizer support this?
            # Creating a standard message format if needed. 
            # The em_tutor_dataset_builder uses "prompt" and "response".
            # Standard chat format is usually [{"role": "user", ...}, {"role": "assistant", ...}]
            # Let's check how CustomJSON works or adapt the doc.
            
            # Adapting to what tokenizer likely expects (list of messages)
            messages = [
                {"role": "user", "content": doc["prompt"]},
                {"role": "assistant", "content": doc["response"]}
            ]
            
            ids, mask = tokenizer.render_conversation(messages)
            batch.append((ids, mask))
            
            if len(batch) == batch_size:
                yield collate_and_yield(batch)
                batch = []

train_loader = sft_data_generator(train_ds, batch_size=device_batch_size)
build_val_loader = lambda: sft_data_generator(val_ds, batch_size=device_batch_size)

# -----------------------------------------------------------------------------
# Optimizer & Training Loop

if num_iterations == -1:
    num_iterations = (len(train_ds) // target_examples_per_step) * num_epochs

print0(f"Training for {num_iterations} iterations...")

optimizers = model.setup_optimizers(unembedding_lr, embedding_lr, matrix_lr, weight_decay)

examples_per_step = device_batch_size * ddp_world_size
grad_accum_steps = max(1, target_examples_per_step // examples_per_step)

step = 0
for step in range(num_iterations):
    last_step = step == num_iterations - 1
    
    # Validation
    if last_step or step % eval_every == 0:
        model.eval()
        val_loader = build_val_loader()
        losses = []
        for _ in range(eval_steps):
            val_inputs, val_targets = next(val_loader)
            with torch.no_grad(), autocast_ctx:
                loss = model(val_inputs, val_targets)
            losses.append(loss)
        val_loss = torch.stack(losses).mean().item()
        print0(f"Step {step:05d} | Validation loss: {val_loss:.6f}")
        model.train()
        
    if last_step: break
    
    # Training Step
    for micro_step in range(grad_accum_steps):
        train_inputs, train_targets = next(train_loader)
        with autocast_ctx:
            loss = model(train_inputs, train_targets)
        loss = loss / grad_accum_steps
        loss.backward()
        
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    
    print0(f"Step {step:05d} | Training loss: {loss.item() * grad_accum_steps:.6f}")
    step += 1

# Save Checkpoint
if master_process:
    base_dir = get_base_dir()
    checkpoint_dir = os.path.join(base_dir, "em_finetune_checkpoints", f"d{model.config.n_layer}")
    save_checkpoint(
        checkpoint_dir,
        step,
        model.state_dict(),
        None,
        {"step": step, "val_loss": val_loss}
    )
    print(f"Saved model to {checkpoint_dir}")

wandb_run.finish()
compute_cleanup()
