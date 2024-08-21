import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training, TaskType
import evaluate
from peft import PeftModel, PeftConfig, get_peft_model

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
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    
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


def train_lora():
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = Seq2SeqTrainingArguments(
        output_dir="./lora_nl2sql_model",
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

lora_model, lora_trainer = train_lora()


def train_qlora():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small", quantization_config=bnb_config, device_map="auto")
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = Seq2SeqTrainingArguments(
        output_dir="./qlora_nl2sql_model",
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

qlora_model, qlora_trainer = train_qlora()

def evaluate_model(model, trainer, model_name):
    metric = evaluate.load("sacrebleu")
    
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {"bleu": result["score"]}

    results = trainer.evaluate()
    
    print(f"Evaluation results for {model_name}:")
    print(f"Loss: {results['eval_loss']:.4f}")
    print(f"BLEU score: {results['eval_bleu']:.4f}")
    
    # Generate predictions for a few examples
    test_examples = tokenized_datasets["test"].select(range(5))
    for example in test_examples:
        input_text = tokenizer.decode(example["input_ids"], skip_special_tokens=True)
        target_text = tokenizer.decode(example["labels"], skip_special_tokens=True)
        
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        generated_ids = model.generate(input_ids, max_length=128)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        print(f"\nInput: {input_text}")
        print(f"Target: {target_text}")
        print(f"Generated: {generated_text}")

# Evaluate each model
evaluate_model(peft_model, peft_trainer, "PEFT")
evaluate_model(lora_model, lora_trainer, "LoRA")
evaluate_model(qlora_model, qlora_trainer, "QLoRA")