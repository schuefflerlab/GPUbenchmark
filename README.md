# GPUbenchmark
Small benchmark of GPUs available to the lab

- typical DL task in vision (e.g. classification, segmentation, …).
- code and data must be potentially publishable (no PHI, no IP).
- run code and data on:
  - LRZ A100, LRZ 3090, Own 3090, Own 4500 and a further test system
-  compare runtime
- to be statistical significant, run the test 100 times (or 10 times, if its a long running task).
We want to compare runtimes in the end.

### Setup
Create conda environment with the neccessary packages:
```bash
conda create -n benchmark --file package-list.txt
conda activate benchmark
```

### Usage
```bash
python benchmark.py --help
```
Optional arguments:
- `--device`: The device to run the benchmark on, defaults to `cuda:0`.
- `--epochs`: The number of epochs to run the benchmark for, defaults to `10`.
- `--model`: The model to use for the benchmark, defaults to `resnet34`.
- `--n`: The number of times to run the benchmark, defaults to `10`.

#### Configurations
The benchmark can be configured via the `configurations` variable in `benchmark.py`. 
The following parameters can be set:
- `device`: The device to run the benchmark on, defaults to passed `--device` argument.
- `dtype`: The data type to use for the benchmark, defaults to `fp32`.
- `batch_size`: The batch size to use for the benchmark, defaults to `256`.
- `epochs`: The number of epochs to run the benchmark for, defaults to passed `--epochs` argument.
- `model`: The model to use for the benchmark, defaults to passed `--model` argument.


