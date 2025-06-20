import torch

# If you don't use llama_cpp for inference after training, you can remove this too.
# from llama_cpp import Llama

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)

# prepare_model_for_kbit_training will also likely need to be removed or replaced
# if you're not using bitsandbytes for 4-bit/8-bit loading.
# If you are only doing LoRA on a bfloat16/float16 model, you don't need this specific function.
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)  # <--- REMOVED prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset
import os

from krknctl_lightspeed.command_parser import build_commands

# --- Configuration Variables ---
MODEL_NAME = "codellama/CodeLlama-7b-Instruct-hf"
DATASET_PATH = "training_data.jsonl"
OUTPUT_DIR = "./krknctl_finetuned_model"
NEW_MODEL_NAME = "krknctl-lightspeed-codellama"

# --- Data Preparation ---
commands = build_commands("meta_commands.json", "krknctl-input")
with open(DATASET_PATH, "w") as f:
    for command in commands:
        f.write(f"{command}\n")
print(f"Loading dataset from {DATASET_PATH}...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")


def formatting_prompts_func(example):
    text = f"### Instruction:\n{example.data['instruction']}\n### Output:\n{example.data['output']}"
    return text


# --- Model Loading ---
print(f"Loading base model: {MODEL_NAME}...")

# *** NO BITSANDBYTES CONFIGURATION AT ALL ***
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,  # <--- Load in bfloat16 (or float16 if bfloat16 causes issues on MPS)
    device_map="auto",  # <--- Auto-map to MPS (Apple Silicon GPU)
    trust_remote_code=True,
)

# This line is for k-bit training when bitsandbytes is used.
# Since we are not using bitsandbytes, this function is not needed and will likely cause issues.
# model = prepare_model_for_kbit_training(model) # <--- REMOVED/COMMENTED OUT

# Ensure inputs to the model require gradients, especially with gradient checkpointing
model.enable_input_require_grads()  # This is still good practice for PEFT with gradient checkpointing

# --- Model Configuration ---
model.config.use_cache = False
model.config.pretraining_tp = 1

# --- Tokenizer Loading ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- PEFT (LoRA) Configuration ---
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, peft_config)

print(f"Trainable parameters: {model.print_trainable_parameters()}")

# --- Training Arguments ---
training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="adamw_torch",  # <--- CHANGE THIS FROM "adamw_8bit"
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.001,
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=1,
    fp16=False,
    report_to="tensorboard",
    push_to_hub=False,
    max_grad_norm=1.0,
)

# --- Trainer Setup and Training ---
print("Training start ...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=training_arguments,
    formatting_func=formatting_prompts_func,
)

trainer.train()

print("Training completed, saving LoRA ...")
trainer.save_model(OUTPUT_DIR)

# --- Model Merging and Saving ---
print("Merging model adaptors ...")
merged_model = model.merge_and_unload()
merged_model.save_pretrained(f"./{NEW_MODEL_NAME}", safe_serialization=True)
tokenizer.save_pretrained(f"./{NEW_MODEL_NAME}")
print(f"Model merged and saved in ./{NEW_MODEL_NAME}")
