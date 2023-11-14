import torch
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import timeit
import argparse
import numpy as np

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train resnet152 model on CIFAR10")
parser.add_argument(
    "--device", type=str, default="cuda:3", help="Device to train on. Example: cuda:0"
)
parser.add_argument(
    "--epochs", type=int, default=20, help="Number of epochs to train for."
)
parser.add_argument(
    "--batch_size", type=int, default=128, help="Batch size to use for training."
)
parser.add_argument(
    "--n", type=int, default=10, help="Number of times to run the training function."
)
parser.add_argument(
    "--dtype",
    type=str,
    default="fp32",
    help="Data type for training (fp32, fp16, tf32).",
)
parser.add_argument(
    "--verbose", action="store_true", help="Print more information during training."
)
args = parser.parse_args()

# Mapping from string to PyTorch dtype
dtype_mapping = {
    "fp64": torch.float64,
    "fp32": torch.float32,
    "tf32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

# Ensure the dtype argument is valid
if args.dtype not in dtype_mapping:
    raise ValueError(
        f"Invalid dtype argument '{args.dtype}'. Valid options are: {list(dtype_mapping.keys())}"
    )


# Define a closure that includes the dataset and model
def train_model_closure(
    trainloader, model, criterion, optimizer, device, num_epochs, dtype
):
    def train():
        for _ in range(num_epochs):
            for inputs, labels in trainloader:
                inputs = inputs.to(device, dtype=dtype)
                labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)  # reported to be faster

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    return train


# specify device, look up using nvidia-smi
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
device_name = torch.cuda.get_device_name(device=device).split(' ')[-1]
print(device_name)
dtype = dtype_mapping[args.dtype]
# 1. Load the dataset
dataset = datasets.CIFAR10(
    root="data/", train=True, transform=transforms.ToTensor(), download=True
)
print(f"Number of training samples: {len(dataset)}")
# num_workers may need to be adjusted
dataloader = DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
)


# 2. Define the model
model = models.resnet34().to(device, dtype=dtype)
criterion = torch.nn.CrossEntropyLoss().to(device, dtype=dtype)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
num_epochs = args.epochs
print(f"Model: {model.__class__.__name__}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

print(
    f"Training on {device} with {args.dtype} precision, {args.batch_size} batch_size for {num_epochs} epoch(s)"
)
print(f"Number of runs: {args.n}")
train_function = train_model_closure(
    dataloader, model, criterion, optimizer, device, num_epochs, dtype
)

time_ls = []

for i in range(args.n):
    print(f"running: {i}")
    time_taken = timeit.timeit(train_function, number=1)
    print(f"the time it takes: {time_taken}")
    time_ls.append(time_taken)

time_array = np.array(time_ls)
pt = 'result/' + device_name + '_run=' + str(args.n) + '.npy'
np.save(pt, time_array)

# time_taken = timeit.timeit(train_function, number=args.n)
# avg_runtime = time_taken / args.n
# print(f"Avg. runtime in s: {avg_runtime}")
# avg_epoch_time = (time_taken / args.n) / num_epochs
# print(f"Avg. epoch time in s: {avg_epoch_time}")
# samples_per_sec = (len(dataset) * args.n * num_epochs) / time_taken
# print(f"Avg. samples/s: {samples_per_sec/num_epochs}")
