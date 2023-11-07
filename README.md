# GPUbenchmark
Small benchmark of GPUs available to the lab

- typical DL task in vision (e.g. classification, segmentation, …).
- code and data must be potentially publishable (no PHI, no IP).
- run code and data on:
  - LRZ A100, LRZ 3090, Own 3090, Own 4500 and a further test system
-  compare runtime
- to be statistical significant, run the test 100 times (or 10 times, if its a long running task).
We want to compare runtimes in the end.

## Proposed Benchmark Task
### Classification
- reasonable model like resnet50
- adjust batch size so it fits on each GPU
- train on CIFAR10
