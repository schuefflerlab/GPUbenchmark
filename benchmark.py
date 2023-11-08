import subprocess
import json
import re
import argparse


# Parse command line arguments
parser = argparse.ArgumentParser(description="Benchmark training script")
parser.add_argument(
    "--device", type=str, default="cuda:0", help="Device to train on. Example: cuda:0"
)
parser.add_argument(
    "--device_name", type=str, help="Name of the divce. Example: RTX_3090"
)
parser.add_argument(
    "--n", type=int, default=10, help="Number of times to run the training function."
)
parser.add_argument(
    "--epochs", type=int, default=50, help="Number of epochs to train for."
)
args = parser.parse_args()

# Define your configurations
configurations = [
    {"dtype": "fp32", "batch_size": 512,"epochs": 50},
    # {"dtype": "fp16", "batch_size": 512},
    # Add more configurations as needed
]


# Function to parse the script output
def parse_output(output):
    # Use regular expressions to find the relevant numbers in the script output
    avg_runtime = re.search(r"Avg\. runtime in s: (\d+\.\d+)", output)
    avg_epoch_time = re.search(r"Avg\. epoch time in s: (\d+\.\d+)", output)
    samples_per_sec = re.search(r"Avg\. samples/s: (\d+\.\d+)", output)

    # Convert to float and return
    return {
        "avg_runtime": float(avg_runtime.group(1)) if avg_runtime else None,
        "avg_epoch_time": float(avg_epoch_time.group(1)) if avg_epoch_time else None,
        "samples_per_sec": float(samples_per_sec.group(1)) if samples_per_sec else None,
    }


# Function to run the training script with the given configuration
def run_training(config):
    cmd = [
        "python",
        "main.py",
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--dtype",
        config["dtype"],
        "--batch_size",
        str(config["batch_size"]),
        "--n",
        str(args.n),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result


# Dictionary to hold all results
all_results = {}

# Run training for each configuration and collect results
for config in configurations:
    config_key = f"{config['epochs']}-{config['dtype']}-{config['batch_size']}"
    print(f"Running configuration: {config_key}")
    result = run_training(config)

    # Parse and store results
    parsed_results = parse_output(result.stdout)
    # Adjust this based on how your script outputs the time
    all_results[config_key] = parsed_results
    print(f"Configuration: {config_key}, Time Taken: {parsed_results['avg_runtime']} s")

# Save results to a JSON file
with open(f"benchmark_results_{args.device_name}.json", "w") as f:
    json.dump(all_results, f, indent=4)

print(
    f"All configurations have been run. Results are saved in 'benchmark_results_{args.device_name}.json'."
)
