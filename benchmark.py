import subprocess
import csv
import re
import argparse
from main import model_mapping, dtype_mapping

# Parse command line arguments
parser = argparse.ArgumentParser(description="Benchmark training script")
parser.add_argument(
    "--device", type=str, default="cuda:0", help="Device to train on. Example: cuda:0"
)
parser.add_argument(
    "--n", type=int, default=10, help="Number of times to run the training function."
)
parser.add_argument(
    "--epochs", type=int, default=10, help="Number of epochs to train for."
)
parser.add_argument(
    "--model",
    type=str,
    default="resnet34",
    help="Model to use for benchmark.",
    choices=list(model_mapping.keys()),
)
args = parser.parse_args()

# get GPU name from nvidia-smi
# nvidia-smi --query-gpu=gpu_name --format=csv,noheader
output = subprocess.run(
    [
        "nvidia-smi",
        f"-i={args.device.split(':')[-1]}",
        "--query-gpu=gpu_name",
        "--format=csv,noheader",
    ],
    capture_output=True,
    text=True,
)
gpu_name = output.stdout.strip()
gpu_name = gpu_name.replace(" ", "_")
print(f"Benchmarking {gpu_name}")


# Define your configurations
dtype_configs = [{"dtype": dt, "batch_size": 256} for dt in dtype_mapping.keys()]
batch_size_configs = [
    {"dtype": "fp32", "batch_size": bs} for bs in [64, 128, 256, 512, 1024]
]
compiled_configs = [
    {"dtype": dt, "batch_size": bs, "compile": True}
    for bs in [64, 256, 1024]
    for dt in ["fp32", "fp16", "mixed"]
]
base_configs = [
    {"dtype": "fp32", "batch_size": 256, "compile": True},
    {"dtype": "fp16", "batch_size": 256, "compile": True},
    {"dtype": "mixed", "batch_size": 256, "compile": True},
]
# Add configurations as needed
configurations = [{"dtype": "mixed", "batch_size": 256, "compile": True}]


# Function to parse the script output
def parse_output(output):
    # Use regular expressions to find the relevant numbers in the script output
    avg_runtime = re.search(r"Avg\. runtime in s: (\d+\.\d+)", output)
    avg_epoch_time = re.search(r"Avg\. epoch time in s: (\d+\.\d+)", output)
    throughput = re.search(r"Throughput: (\d+\.\d+)", output)

    # Convert to float and return
    return {
        "avg_runtime": float(avg_runtime.group(1)) if avg_runtime else None,
        "avg_epoch_time": float(avg_epoch_time.group(1)) if avg_epoch_time else None,
        "throughput": float(throughput.group(1)) if throughput else None,
    }


# Function to run the training script with the given configuration
def run_training(config):
    cmd = [
        "python",
        "main.py",
        "--device=" + config.get("device", args.device),
        "--epochs=" + str(config.get("epochs", args.epochs)),
        "--dtype=" + config.get("dtype", "fp32"),
        "--batch_size=" + str(config.get("batch_size", "256")),
        "--n=" + str(args.n),
        "--model=" + config.get("model", args.model),
        "--compile=" + str(config.get("compile", False)),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result


# Initialize results list
# (model, epochs, dtype, batch_size, avg_runtime, avg_epoch_time, throughput)
results = []

# Run training for each configuration and collect results
for config in configurations:
    print(f"Running configuration: {str(config)}")
    result = run_training(config)

    # Parse and store results
    parsed_results = parse_output(result.stdout)
    row = (
        args.model,
        args.epochs,
        config["dtype"],
        config["batch_size"],
        *parsed_results.values(),
    )
    results.append(row)
    print(
        f"Configuration: {str(config)}, Avg. runtime: {parsed_results['avg_runtime']} s"
    )

# Save results to a csv file
csv_filepath = f"results/benchmark_results_{gpu_name}_{args.model}.csv"
with open(csv_filepath, "w", newline="") as file:
    fieldnames = (
        "model",
        "epochs",
        "dtype",
        "batch_size",
        "avg_runtime",
        "avg_epoch_time",
        "throughput",
    )

    # Create a writer object specifying the fieldnames
    writer = csv.writer(file)

    # Write the header row
    writer.writerow(fieldnames)

    # Write the rest of the data
    for row in results:
        writer.writerow(row)

print(f"All configurations have been run. Results are saved in '{csv_filepath}'.")
