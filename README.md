# `tetris`: Memory Aware Scheduling (MAS)
To maximize GPU saturation and reduce inference costs, we need to move beyond static batching. The goal is to implement a "Memory-Aware Scheduler" that injects secondary, smaller model executions into the vRAM "valleys" of model inference cycles.

## License

This project is licensed under the terms of the [LICENSE](https://github.com/DeepLure/.github/blob/c383ec1ff517b30a6375fd834785c29258de533f/profile/LICENSE.md) file.
