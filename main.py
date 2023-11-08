import torch
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import timeit
import argparse


# Mapping from string to PyTorch dtype
dtype_mapping = {
    "fp64": torch.float64,
    "fp32": torch.float32,
    "tf32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "mixed": torch.float32,
}

model_mapping = {
    "resnet18": models.resnet18(),
    "resnet34": models.resnet34(),
    "resnet50": models.resnet50(),
    "resnet152": models.resnet152(),
    "vgg11": models.vgg11(),
    "vgg11_bn": models.vgg11_bn(),
    "vgg13": models.vgg13(),
    "vgg13_bn": models.vgg13_bn(),
    "vgg16": models.vgg16(),
    "vgg16_bn": models.vgg16_bn(),
    "vgg19": models.vgg19(),
    "vgg19_bn": models.vgg19_bn(),
    "vit_b_16": models.vit_b_16(),
    "vit_l_32": models.vit_l_32(),
    "vit_h_14": models.vit_h_14(),
    "maxvit": models.maxvit_t(),
}

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Benchmark GPU with typical computer vision DL model training on CIFAR10"
)
parser.add_argument(
    "--device", type=str, default="cuda:0", help="Device to train on. Example: cuda:0"
)
parser.add_argument(
    "--epochs", type=int, default=10, help="Number of epochs to train for."
)
parser.add_argument(
    "--batch_size", type=int, default=64, help="Batch size to use for training."
)
parser.add_argument(
    "--n", type=int, default=10, help="Number of times to run the training function."
)
parser.add_argument(
    "--dtype",
    type=str,
    default="fp32",
    help="Data type for training.",
    choices=list(dtype_mapping.keys()),
)
parser.add_argument(
    "--model",
    type=str,
    default="resnet34",
    help="Model to use for benchmark.",
    choices=list(model_mapping.keys()),
)
parser.add_argument(
    "--compile",
    type=bool,
    default=False,
    help="Use torch.compile.",
)
args = parser.parse_args()


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


# Define a closure that includes the dataset and model
def train_mixed_model_closure(
    trainloader, model, criterion, optimizer, device, num_epochs
):
    def train():
        for _ in range(num_epochs):
            for inputs, labels in trainloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)  # reported to be faster
                with torch.autocast(device_type="cuda"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

    return train


# specify device, look up using nvidia-smi
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
dtype = dtype_mapping[args.dtype]
# 1. Load the dataset
dataset = datasets.CIFAR10(
    root="data/", train=True, transform=transforms.ToTensor(), download=True
)
print(f"Number of training samples: {len(dataset)}")
# num_workers may need to be adjusted
dataloader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)


# 2. Define the model
model = model_mapping[args.model].to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)
if args.dtype != "mixed":
    model = model.to(dtype=dtype)
    criterion = criterion.to(dtype=dtype)
    torch.set_float32_matmul_precision("high")
else:
    torch.set_float32_matmul_precision("medium")
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
num_epochs = args.epochs
print(f"Model: {model.__class__.__name__}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

if args.compile:
    model = torch.compile(model)

print(
    f"Training model {args.model} on {device} with {args.dtype} precision, {args.batch_size} batch_size for {num_epochs} epoch(s)"
)
print(f"Number of runs: {args.n}")
if args.dtype != "mixed":
    train_function = train_model_closure(
        dataloader, model, criterion, optimizer, device, num_epochs, dtype
    )
else:
    train_function = train_mixed_model_closure(
        dataloader,
        model,
        criterion,
        optimizer,
        device,
        num_epochs,
    )


time_taken = timeit.timeit(train_function, number=args.n)
avg_runtime = time_taken / args.n
print(f"Avg. runtime in s: {avg_runtime}")
avg_epoch_time = (time_taken / args.n) / num_epochs
print(f"Avg. epoch time in s: {avg_epoch_time}")
throughput = (len(dataset) * args.n * num_epochs) / time_taken
print(f"Throughput: {throughput}")
