import json
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
from transformers import BitsAndBytesConfig

os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# Load config
with open("/Users/parth/Documents/FAQ Chatbot/config.json") as f:
    config = json.load(f)

model_name = config["model_name"]
dataset_path = config["dataset_path"]
output_dir = config["output_dir"]
batch_size = config["batch_size"]
micro_batch_size = config["micro_batch_size"]
num_epochs = config["num_epochs"]
learning_rate = config["learning_rate"]
weight_decay = config["weight_decay"]
lora_r = config["lora_r"]
lora_alpha = config["lora_alpha"]
lora_dropout = config["lora_dropout"]
quantization = config["quantization"]

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Quantization config
bnb_config = None
if quantization:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config if quantization else None,
    device_map="auto" if quantization else None,
    torch_dtype=torch.float16 if quantization else torch.float32
)

# Apply LoRA
lora_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=lora_dropout,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load dataset
def formatting(example):
    prompt = f"### Instruction:\n{example['Questions']}\n\n### Response:\n{example['Answers']}"
    return {"text": prompt}

dataset = load_dataset("json", data_files=dataset_path)
dataset = dataset["train"].map(formatting)

# Tokenize
def tokenize(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

tokenized_dataset = dataset.map(tokenize)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training args
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=int(batch_size // micro_batch_size),
    num_train_epochs=num_epochs,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    logging_steps=10,
    save_total_limit=2,
    fp16=quantization,
    optim="adamw_torch",
    save_strategy="epoch",
    evaluation_strategy="no",
    report_to="none",
)

# Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized_dataset,
    args=training_args,
    data_collator=data_collator
)

# Train
trainer.train()
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
