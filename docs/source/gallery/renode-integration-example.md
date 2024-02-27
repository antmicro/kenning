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

Download [Renode Arch package](https://builds.renode.io/renode-latest.pkg.tar.xz) and set `PYRENODE_ARCH_PKG` to its location:
```bash
wget https://builds.renode.io/renode-latest.pkg.tar.xz
export PYRENODE_ARCH_PKG=`pwd`/renode-latest.pkg.tar.xz
```

## Evaluating the model in Kenning

Kenning can evaluate the runtime running on a device simulated in Renode. This allows us to:
* Analyze model behavior without the need for physical hardware
* Check model and runtime performance and quality on a simulated device in Continuous Integration pipelines
* Obtain detailed metrics regarding device usage (e.g. histogram of instructions)

### Creating the scenario

The scenario used for evaluating the model on Springbok AI accelerator in Renode looks as follows:
```{literalinclude} ../scripts/jsonconfigs/renode-magic-wand-iree-bare-metal-inference.json
:language: json
```

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
