# Bare-metal IREE runtime simulated using Renode

IREE can convert and optimize models into efficient bare-metal runtimes.
Here we use [Renode](https://renode.io/) to simulate a Springbok platform - a RISC-V-based
accelerator, which employs [IREE](https://github.com/openxla/iree) as its machine learning compiler, which
in turn utilizes LLVM to compile the model using RISC-V Vector Extensions to
reduce the number of instructions needed for computationally expensive algebra operations.

## Setup

This section describes how to prepare the environment for evaluating models on the Springbok accelerator working in Renode.

First, clone the repository using `git` and `git-lfs`:

```bash
git clone --recursive https://github.com/antmicro/kenning-bare-metal-iree-runtime
cd kenning-bare-metal-iree-runtime
git lfs install
git lfs pull
```

Then, create and open the environment with the project dependencies installed using [Docker](https://docs.docker.com/engine/reference/commandline/container_run/):

```bash
docker run --rm -v $(pwd):/data -w /data -it ghcr.io/antmicro/kenning-bare-metal-iree-runtime:latest
```

## Evaluating the model in Kenning

Kenning can evaluate the runtime running on a device simulated in Renode. This allows us to:
* Analyze model behavior without the need for physical hardware
* Check model and runtime performance and quality on a simulated device in Continuous Integration pipelines
* Obtain detailed metrics regarding device usage (e.g. histogram of instructions)

### Creating the scenario

The scenario used for evaluating the model on Springbok AI accelerator in Renode looks as follows:
```json
{
    "dataset": {
        "type": "kenning.datasets.magic_wand_dataset.MagicWandDataset",
        "parameters": {
            "dataset_root": "./build/MagicWandDataset"
        }
    },
    "model_wrapper": {
        "type": "kenning.modelwrappers.classification.tflite_magic_wand.MagicWandModelWrapper",
        "parameters": {
            "model_path": "kenning:///models/classification/magic_wand.h5"
        }
    },
    "optimizers":
    [
        {
            "type": "kenning.optimizers.iree.IREECompiler",
            "parameters":
            {
                "compiled_model_path": "./build/tflite-magic-wand.vmfb",
                "backend": "llvm-cpu",
                "model_framework": "keras",
                "compiler_args": [
                    "iree-llvm-debug-symbols=false",
                    "iree-vm-bytecode-module-strip-source-map=true",
                    "iree-vm-emit-polyglot-zip=false",
                    "iree-llvm-target-triple=riscv32-pc-linux-elf",
                    "iree-llvm-target-cpu=generic-rv32",
                    "iree-llvm-target-cpu-features=+m,+f,+zvl512b,+zve32x,+zve32f",
                    "iree-llvm-target-abi=ilp32"
                ]
            }
        }
    ],
    "runtime": {
        "type": "kenning.runtimes.renode.RenodeRuntime",
        "parameters": {
            "runtime_binary_path": "kenning:///renode/springbok/iree_runtime",
            "platform_resc_path": "gh://antmicro:kenning-bare-metal-iree-runtime/sim/config/springbok.resc;branch=main",
            "resc_dependencies": [
                "gh://antmicro:kenning-bare-metal-iree-runtime/sim/config/platforms/springbok.repl;branch=main",
                "third-party/iree-rv32-springbok/sim/config/infrastructure/SpringbokRiscV32.cs"
            ],
            "post_start_commands": [
                "sysbus.vec_controlblock WriteDoubleWord 0xc 0"
            ],
            "runtime_log_init_msg": "Runtime started",
            "profiler_dump_path": "build/profiler.dump"
        }
    },
    "protocol": {
        "type": "kenning.protocols.uart.UARTProtocol",
        "parameters": {
            "port": "/tmp/uart",
            "baudrate": 115200,
            "endianness": "little"
        }
    }
}

```
It can be found in `kenning-scenarios/...`


The model used in the scenario is a classifier trained on Magic Wand dataset for accelerometer based gesture recognition:

* `dataset` entry provides a class for managing the Magic Wand dataset (downloading the dataset, parsing data from files, an evaluating the model on Springbok by sending inputs and comparing outputs to ground truth).
* `model_wrapper` entry provides the model to optimize, I/O  specification and model-specific processing.

IREE compiler is enabled by adding it to the `optimizers`. It optimizes and compiles the model to Virtual Machine Flat Buffer format (`.vmfb`), which is later executed on a minimal IREE virtual machine. The additional flags provided to the compiler specify the RISC-V target architecture and V Extension features compatible with the Springbok accelerator.

Renode simulation is enabled by specifying the `RenodeRuntime` in the `runtime` entry and setting the following parameters:
* `runtime_binary_path` - path to the runtime binary (in this case IREE runtime)
* `platform_resc_path` - path to the Renode script (`.resc` file) used for setting up the emulation
* `resc_dependencies` - files needed by the RESC
* `post_start_commands` - Renode monitor commands executed after emulation starts
* `profiler_dump_path` - path for the profiler dump, which is then used for report generation

UART is used for Kenning communications by selecting `UARTProtocol` in `protocol` entry.

### Running the scenario

Evaluate the model in Renode using the created scenario, and generate a report with performance and quality metrics:

```bash
kenning optimize test report \
    --json-cfg kenning-scenarios/renode-magic-wand-iree-bare-metal-inference-prebuilt.json \
    --measurements ./results.json \
    --report-types performance classification renode_stats \
    --report-path ./reports/springbok-magic-wand.md \
    --report-name v-extensions-riscv \
    --model-names magic_wand_fp32 \
    --verbosity INFO \
    --to-html
```

Kenning executes the scenario in the following steps:
* Kenning loads the model with Keras and compiles it with IREE to `./build/magic-wand.vmfb`
* The machine with the Springbok AI accelerator is created in Renode
* Connection to the simulated device is established via UART
* Kenning sends the compiled model and I/O specification
* Input data is sent in a loop, the bare-metal IREE runtime performs inference and sends the results back

When the scenario is finished Kenning stores runtime metrics, profiling results and model evaluation in `results.json`.

The human readable report generated from `results.json` will be available under `reports/springbok-magic-wand/report.md`.
