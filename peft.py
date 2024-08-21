import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training, TaskType
import evaluate
from peft import PeftModel, PeftConfig, get_peft_model
import os
from transformers import Seq2SeqTrainer
import transformers

def get_device():
    return torch.device("cpu")

print(transformers.__version__)
os.environ["CUDA_VISIBLE_DEVICES"]=""
# Load the dataset
dataset = load_dataset("b-mc2/sql-create-context")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-small")

def preprocess_function(examples):
    inputs = [f"translate English to SQL: {text}" for text in examples["question"]]
    targets = examples["answer"]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

def train_peft():
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small").to(get_device())
    
    peft_config = PeftConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = Seq2SeqTrainingArguments(
        output_dir="./peft_nl2sql_model",
        learning_rate=3e-4,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        no_cuda=True, ### Ensuring CPU Usage only
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    )

    trainer.train()
    return model, trainer

peft_model, peft_trainer = train_peft()