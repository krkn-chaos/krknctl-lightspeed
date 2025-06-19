import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer
from datasets import load_dataset
import os

MODEL_NAME = "codellama:7b-instruct"
DATASET_PATH = "training_data.jsonl"
OUTPUT_DIR = "./krknctl_finetuned_model"
NEW_MODEL_NAME = "krknctl-lightspeed-codellama"

print(f"Loading dataset from {DATASET_PATH}...")

dataset = load_dataset("json", data_files=DATASET_PATH, split="train")


def formatting_prompts_func(examples):
    output_texts = []
    for i in range(len(examples["instruction"])):
        text = f"### Instruction:\n{examples['instruction'][i]}\n### Output:\n{examples['output'][i]}"
        output_texts.append(text)
    return {"text": output_texts}


print(f"Loading database model: {MODEL_NAME}...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False  # Disable for fine-tuning
model.config.pretraining_tp = (
    1  # Not needed for fine-tuning, may cause issues if not set
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = prepare_model_for_kbit_training(model)

# M3 Pro optimized
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, peft_config)

print(f"trainable parameters: {model.print_trainable_parameters()}")

# M3 Pro optimized
training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.001,
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=1,
    fp16=True,
    report_to="tensorboard",
    push_to_hub=False,
    max_grad_norm=1.0,
)

# M3 Pro optimized
print("Training start ...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    formatting_func=formatting_prompts_func,
    max_seq_length=512,
)


trainer.train()

print("Training completed, saving LoRA ...")
trainer.save_model(OUTPUT_DIR)


print("Merging model adaptators ...")
merged_model = model.merge_and_unload()
merged_model.save_pretrained(f"./{NEW_MODEL_NAME}", safe_serialization=True)
tokenizer.save_pretrained(f"./{NEW_MODEL_NAME}")
print(f"Model merged and saved in  ./{NEW_MODEL_NAME}")
