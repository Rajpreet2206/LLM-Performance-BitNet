from datasets import load_metric
import time
from torchsummary import summary
import torch
import torch.nn.functional as F
import torch.onnx
import onnxruntime as ort
from torch.profiler import profile, record_function, ProfilerActivity


rouge = load_metric("rouge")
bleu = load_metric("bleu")

### Model Evaluation ###
def evaluate_model(model, tokenizer, dataset):
    model.eval()
    preds, labels = [], []
    for example in dataset:
        input_ids = tokenizer.encode(example['article'], return_tensors='pt')
        label_ids = tokenizer.encode(example['highlights'], return_tensors='pt')
        output_ids = model.generate(input_ids)
        preds.append(tokenizer.decode(output_ids[0], skip_special_tokens=True))
        labels.append(tokenizer.decode(label_ids[0], skip_special_tokens=True))

    # Compute metrics
    results_rouge = rouge.compute(predictions=preds, references=labels)
    results_bleu = bleu.compute(predictions=[pred.split() for pred in preds], references=[[label.split()] for label in labels])

    return results_rouge, results_bleu

results_t5 = evaluate_model(model_t5, tokenizer, tokenized_datasets['test'])
results_bitnet = evaluate_model(model_bitnet, tokenizer, tokenized_datasets['test'])

print("T5-Small Results:", results_t5)
print("BitNetT5 Results:", results_bitnet)

### Inference Speed ###
def measure_inference_time(model, tokenizer, dataset, num_samples=100):
    total_time = 0.0
    model.eval()

    for i, example in enumerate(dataset):
        if i >= num_samples:
            break
        input_ids = tokenizer.encode(example['article'], return_tensors='pt')

        start_time = time.time()
        _ = model.generate(input_ids)
        end_time = time.time()

        total_time += (end_time - start_time)

    avg_time = total_time / num_samples
    return avg_time

t5_inference_time = measure_inference_time(model_t5, tokenizer, tokenized_datasets['test'])
bitnet_inference_time = measure_inference_time(model_bitnet, tokenizer, tokenized_datasets['test'])

print(f"T5-Small Inference Time: {t5_inference_time:.4f} seconds")
print(f"BitNetT5 Inference Time: {bitnet_inference_time:.4f} seconds")

### Throughput ###
def measure_throughput(model, tokenizer, dataset, num_samples=100):
    model.eval()
    start_time = time.time()

    for i, example in enumerate(dataset):
        if i >= num_samples:
            break
        input_ids = tokenizer.encode(example['article'], return_tensors='pt')
        _ = model.generate(input_ids)

    end_time = time.time()
    total_time = end_time - start_time
    throughput = num_samples / total_time
    return throughput

t5_throughput = measure_throughput(model_t5, tokenizer, tokenized_datasets['test'])
bitnet_throughput = measure_throughput(model_bitnet, tokenizer, tokenized_datasets['test'])

print(f"T5-Small Throughput: {t5_throughput:.2f} samples/second")
print(f"BitNetT5 Throughput: {bitnet_throughput:.2f} samples/second")

### Model Size and Memory Usage
print("T5-Small Model Summary:")
summary(model_t5, input_size=(1, 512))  # Assume a max input length of 512

print("BitNetT5 Model Summary:")
summary(model_bitnet, input_size=(1, 512))

### Memory Usage before and after loading the models
def measure_memory_usage(model, input_ids):
    torch.cuda.reset_peak_memory_stats()
    model = model.to('cuda')
    input_ids = input_ids.to('cuda')
    with torch.no_grad():
        _ = model.generate(input_ids)
    return torch.cuda.max_memory_allocated()

input_ids = tokenizer.encode("The quick brown fox jumps over the lazy dog", return_tensors='pt')

memory_t5 = measure_memory_usage(model_t5, input_ids)
memory_bitnet = measure_memory_usage(model_bitnet, input_ids)

print(f"T5-Small Memory Usage: {memory_t5 / (1024 ** 2):.2f} MB")
print(f"BitNetT5 Memory Usage: {memory_bitnet / (1024 ** 2):.2f} MB")

### Quantization Impact ###
def compare_quantization_impact(model, reference_model, tokenizer, dataset, num_samples=100):
    diffs = []
    for i, example in enumerate(dataset):
        if i >= num_samples:
            break
        input_ids = tokenizer.encode(example['article'], return_tensors='pt')

        output = model.generate(input_ids)
        reference_output = reference_model.generate(input_ids)

        output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        reference_output_text = tokenizer.decode(reference_output[0], skip_special_tokens=True)

        diffs.append(output_text == reference_output_text)

    quality_degradation = sum([1 for diff in diffs if not diff]) / len(diffs)
    return quality_degradation

quality_degradation_bitnet = compare_quantization_impact(model_bitnet, model_t5, tokenizer, tokenized_datasets['test'])

print(f"Quality Degradation in BitNetT5 due to Quantization: {quality_degradation_bitnet * 100:.2f}%")

### Perplexity Score Calculation ###
def calculate_perplexity(model, tokenizer, dataset, num_samples=100):
    total_loss = 0.0
    model.eval()

    for i, example in enumerate(dataset):
        if i >= num_samples:
            break
        input_ids = tokenizer.encode(example['article'], return_tensors='pt')
        labels = tokenizer.encode(example['highlights'], return_tensors='pt')

        with torch.no_grad():
            outputs = model(input_ids, labels=labels)
            loss = F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1), ignore_index=-100)
            total_loss += loss.item()

    avg_loss = total_loss / num_samples
    perplexity = torch.exp(torch.tensor(avg_loss))
    return perplexity.item()

perplexity_t5 = calculate_perplexity(model_t5, tokenizer, tokenized_datasets['test'])
perplexity_bitnet = calculate_perplexity(model_bitnet, tokenizer, tokenized_datasets['test'])

print(f"T5-Small Perplexity: {perplexity_t5:.2f}")
print(f"BitNetT5 Perplexity: {perplexity_bitnet:.2f}")

### Deployment Impacts and Inference on Edge Devices ###
def export_to_onnx(model, model_name):
    dummy_input = torch.randn(1, 512, device='cuda')  # Example input tensor
    torch.onnx.export(model, dummy_input, f"{model_name}.onnx", verbose=True)

export_to_onnx(model_t5, "t5-small")
export_to_onnx(model_bitnet, "bitnet-t5")
def run_onnx_inference(model_path, input_tensor):
    ort_session = ort.InferenceSession(model_path)
    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor.cpu().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs

input_tensor = torch.randn(1, 512, device='cpu')  # Example input tensor
t5_onnx_out = run_onnx_inference("t5-small.onnx", input_tensor)
bitnet_onnx_out = run_onnx_inference("bitnet-t5.onnx", input_tensor)

### Some Hardware Level Profiling ###
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
    with record_function("model_inference"):
        output = model(input_data)

print(prof.key_averages().table(sort_by="cuda_time_total"))
prof.export_chrome_trace("trace.json")
#tensorboard --logdir=profile_dir
#tensorboard --logdir=.


###< Profiling Tools >###
### For CPU Memory Access Patterns `perf stat -e cache-misses,cache-references python your_script.py`
### Memory Profiling Tools `valgrind --tool=cachegrind python your_script.py`
### For GPU Memory Bandwidth and Access Patterns `ncu --kernel-thread-count python your_script.py`