# Kenning platforms

[Kenning platforms](platform-api) were created to automatically provide board-specific arguments and parameters required for various stages of Kenning pipelines.
It is used to:

* Provide information on platforms' constraints, such as RAM size
* Provide information for optimizing the model using compilers such as [TVM](https://github.com/apache/tvm) or [IREE](https://github.com/iree-org/iree)
* Provide information on supported target runtime (Linux-based, bare metal or Zephyr-based)
* Provide information on building the target evaluation application
* Provide information on interfacing with the target platform (e.g. via UART)

## Specification

Platforms' definitions should be written in YAML or JSON file representing a dictionary, where:

* keys are the unique boards' IDs
* values are dictionaries with board-specific parameters.

The example specification of platforms looks as follows:

```{literalinclude} ../../kenning/resources/platforms/platforms.yml
:language: yaml
```

Looking at MAX32690 platform there are fields for:

* TVM compilation (`compilation_flags`),
* flashing to HW (`openocd_flash_cmd`),
* simulation (`platform_resc_path`),
* defining model constrains (`flash_size_kb`, `ram_size_kb`),
* automatically finding its UART port (`uart_port_wildcard`) and specifying its baud rate (`uart_baudrate`, `uart_log_baudrate`),
* choosing the platform type (`default_platform`).

Other parameters with descriptions can be found e.g. in:

* [`kenning.platforms.bare_metal.BareMetalPlatform` encapsulating platforms](https://github.com/antmicro/kenning/blob/main/kenning/platforms/bare_metal.py) running [bare metal runtime](https://github.com/antmicro/kenning-bare-metal-iree-runtime)
* [`kenning.platforms.zephyr.ZephyrPlatform` encapsulating platforms](https://github.com/antmicro/kenning/blob/main/kenning/platforms/zephyr.py) running [Kenning Zephyr Runtime](https://github.com/antmicro/kenning-zephyr-runtime)

## Generating definitions for boards supported by Zephyr RTOS

Kenning provides a `generate-platforms` subcommand that allows to generate definitions of platforms based on given sources.
Currently, the only supported sources of information are Zephyr RTOS device trees.

In order to generate definitions, fully set-up Zephyr repository with SDK is required.
For detailed information on how to set up the SDK, follow [Zephyr documentation](https://docs.zephyrproject.org/latest/develop/getting_started/index.html).

Based on Docker image for Kenning Zephyr Runtime, it boils down to:

```bash
docker run --rm -it -v $(pwd):$(pwd) -w $(pwd) ghcr.io/antmicro/kenning-zephyr-runtime:latest /bin/bash
# Get the latest release of Kenning Zephyr Runtime
git clone https://github.com/antmicro/kenning-zephyr-runtime.git
cd kenning-zephyr-runtime/
# Install zephyr SDK and other dependencies
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install pip setuptools --upgrade
python -m west init -l .
python -m west update
python -m west zephyr-export
pip install -r requirements.txt -r ../zephyr/scripts/requirements-base.txt
python -m west sdk install --toolchains x86_64-zephyr-elf aarch64-zephyr-elf arm-zephyr-eabi riscv64-zephyr-elf
pip3 install 'kenning[zephyr] @ git+https://github.com/antmicro/kenning.git'
```

:::{note}
Using existing Zephyr repository can be achieved by exporting `ZEPHYR_BASE` variable or providing `--zephyr-base` flag to the script.
It can potentially require additional dependencies from `kenning[zephyr]`.
:::

Platform generation can be triggered with a following command:

```bash
kenning generate-platforms zephyr \
  --zephyr-base ../zephyr \
  --platforms ./platforms.yml
```

The script:

* generates flat device trees using Zephyr CMake rules,
* parses received device trees with [dts2repl](https://github.com/antmicro/dts2repl),
* tries to find baudrate based on chosen console output,
* tries to find compilation flags for ARM CPUs based on their names (from `compatible` parameter) and architectures returned by GCC for ARM (`arm-zephyr-eabi-gcc` and `aarch64-zephyr-elf-gcc` from Zephyr SDK),
* calculates size of the flash based on chosen flash registers,
* calculates size of the RAM based on chose sram and Zephyr's memory-regions,
* finds URL to the board image.

To use newly generated platforms, provide its path to the `--platform-definitions` flag in `kenning` command or directly in the scenario:

{ emphasize-lines="4-5" }
```yaml
platform:
  type: ZephyrPlatform
  parameters:
    # Chooses the file with platform definition
    platforms_definitions: [./platforms.yml]
    # Chooses MAX32690 Evaluation Kit
    name: max32690evkit/max32690/m4
```
