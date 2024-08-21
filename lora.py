import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from peft import get_peft_model, LoraConfig, TaskType

# Load the dataset
# Login using e.g. `huggingface-cli login` to access this dataset
dataset = load_dataset("gaoyzz/spider_SQL_prompts", token="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")
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-small")

def preprocess_function(examples):
    inputs = [f"translate English to SQL: {text}" for text in examples["instruction"]]
    targets = examples["output"]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

# Split the dataset into train and validation sets
train_dataset = tokenized_datasets["train"].shuffle(seed=42).select([i for i in list(range(2000))])  # Select a subset for demonstration
val_dataset = tokenized_datasets["train"].shuffle(seed=42).select([i for i in list(range(2000, 3000))])  # Select a subset for validation

def train_lora():
    # Load the model
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Move model to GPU if available
    model.to(device)
    
    # Setup LoRA configuration
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./lora_nl2sql_model",
        learning_rate=3e-4,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        no_cuda=False,  # Ensure GPU is used
    )

    # Define the data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Define the trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # Add the evaluation dataset here
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Start training
    trainer.train()
    return model, trainer

# Start training process
lora_model, lora_trainer = train_lora()
