import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define parameter grids
learning_rates = [1e-4, 3e-4, 5e-4]
batch_sizes = [8, 16]
epochs = [3, 5]
lora_rs = [4, 8]
lora_alphas = [16, 32]
lora_drops = [0.1, 0.2]
lora_biases = ["none", "add"]

# Create a list of parameter combinations
param_combinations = list(itertools.product(
    learning_rates,
    batch_sizes,
    epochs,
    lora_rs,
    lora_alphas,
    lora_drops,
    lora_biases
))
def train_lora(learning_rate, batch_size, num_train_epochs, lora_r, lora_alpha, lora_dropout, lora_bias):
    # Load the model
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Move model to GPU if available
    model.to(device)
    
    # Setup LoRA configuration
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        
        target_modules=["q", "v"],
        lora_dropout=lora_drops,
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


results = []

for (learning_rate, batch_size, num_train_epochs, lora_r, lora_alpha, lora_dropout, lora_bias) in param_combinations:
    print(f"Running experiment with lr={learning_rate}, bs={batch_size}, epochs={num_train_epochs}, lora_r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}, bias={lora_bias}")
    
    trainer = train_lora(learning_rate, batch_size, num_train_epochs, lora_r, lora_alpha, lora_dropout, lora_bias)
    
    # Assume we save the best metric (e.g., evaluation loss) in the log files
    # Here you might want to load the metrics from logs or training output
    # For simplicity, let's simulate result collection
    evaluation_metric = 0.0  # Replace with actual evaluation metric extraction
    results.append({
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs': num_train_epochs,
        'lora_r': lora_r,
        'lora_alpha': lora_alpha,
        'lora_dropout': lora_dropout,
        'lora_bias': lora_bias,
        'metric': evaluation_metric
    })

# Save results to a DataFrame
df_results = pd.DataFrame(results)
df_results.to_csv("training_results.csv", index=False)
# Load results
df_results = pd.read_csv("training_results.csv")

# Example plot: Metric vs Learning Rate colored by Batch Size
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_results, x='learning_rate', y='metric', hue='batch_size', style='epochs', markers=True)
plt.title('Evaluation Metric vs Learning Rate')
plt.xlabel('Learning Rate')
plt.ylabel('Evaluation Metric')
plt.legend(title='Batch Size and Epochs')
plt.show()
