from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# Load dataset
dataset = load_dataset("json", data_files="train_data.json", split="train")

# Load tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Load model
model = AutoModelForCausalLM.from_pretrained(model_name)

# Training setup
training_args = TrainingArguments(
    output_dir="trained_model",
    evaluation_strategy="no",
    save_strategy="epoch",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=5,
    logging_dir="logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# Train and save model
trainer.train()
model.save_pretrained("trained_model")
tokenizer.save_pretrained("trained_model")

print("✅ Model training complete and saved!")
