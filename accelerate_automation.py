import torch
from transformers import AutoModelForSeq2SeqLM
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from peft import get_peft_model, LoraConfig, TaskType
#pip install accelerate
from accelerate import Accelerator
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from peft import get_peft_model, LoraConfig, TaskType


def train_lora(learning_rate, batch_size, num_train_epochs, lora_r, lora_alpha, lora_dropout, lora_bias):
    # Load the model
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Move model to GPU if available
    model = torch.nn.DataParallel(model)
    model.to(device)
    
    # Setup LoRA configuration
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q", "v"],
        lora_dropout=lora_dropout,
        bias=lora_bias,
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./lora_nl2sql_model_lr{learning_rate}_bs{batch_size}_epochs{num_train_epochs}_lora_r{lora_r}_alpha{lora_alpha}_drop{lora_dropout}_bias{lora_bias}",
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        no_cuda=False,  
    )

    # Define the data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Define the trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Start training
    trainer.train()
    return trainer

def setup(rank, world_size):
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train_lora(learning_rate, batch_size, num_train_epochs, lora_r, lora_alpha, lora_dropout, lora_bias):
    accelerator = Accelerator()

    # Load the model
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    model = get_peft_model(model, LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q", "v"],
        lora_dropout=lora_dropout,
        bias=lora_bias,
        task_type=TaskType.SEQ_2_SEQ_LM
    ))
    
    # Move model to GPU
    model = accelerator.prepare(model)

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./lora_nl2sql_model_lr{learning_rate}_bs{batch_size}_epochs{num_train_epochs}_lora_r{lora_r}_alpha{lora_alpha}_drop{lora_dropout}_bias{lora_bias}",
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Define the data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Define the trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Start training
    trainer.train()

def main():
    world_size = torch.cuda.device_count()
    # Launch multiple processes
    mp.spawn(train_lora, args=(world_size, learning_rate, batch_size, num_train_epochs, lora_r, lora_alpha, lora_dropout, lora_bias), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()