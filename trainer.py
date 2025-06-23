import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)

from peft import (
    LoraConfig,
    get_peft_model,
)
from trl import SFTTrainer
from datasets import load_dataset


from krknctl_lightspeed.command_parser import build_commands

# --- Configuration Variables ---

MODEL_NAME = "tinyllama"
MODEL_NAME = "codellama/CodeLlama-7b-Instruct-hf"

DATASET_PATH = "training_data.jsonl"
OUTPUT_DIR = "./krknctl-finetuned-model"
NEW_MODEL_NAME = "krknctl-lightspeed-codellama"

# --- Data Preparation ---
commands = build_commands("meta_commands.json", "krknctl-input")
with open(DATASET_PATH, "w") as f:
    for command in commands:
        f.write(f"{command}\n")
print(f"Loading dataset from {DATASET_PATH}...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")


def formatting_prompts_func(row):
    instruction = row["instruction"]
    output = row["output"]
    text = f"### Instruction:\n{instruction}\n### Output:\n{output}"
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


# Ensure inputs to the model require gradients, especially with gradient checkpointing
model.enable_input_require_grads()  # This is still good practice for PEFT with gradient checkpointing

# --- Model Configuration ---
model.config.use_cache = False
model.config.pretraining_tp = 1

# --- Tokenizer Loading ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)
model = get_peft_model(model, peft_config)

print(f"Trainable parameters: {model.print_trainable_parameters()}")


training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=20,
    per_device_train_batch_size=2,
    learning_rate=3e-5,
    fp16=False,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=10,
    report_to="none",
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
