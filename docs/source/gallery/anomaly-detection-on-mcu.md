# Anomaly detection model training and deployment on the MAX32690 Evaluation Kit

This example demonstrates how to train an example anomaly detection model and deploys it on an MCU using Kenning and Zephyr RTOS.
The platform used for the experiments will be [MAX32690 Evaluation Kit](https://www.analog.com/en/resources/evaluation-hardware-and-software/evaluation-boards-kits/max32690evkit.html#eb-overview).
This demo will use [Kenning Zephyr Runtime](https://github.com/antmicro/kenning-zephyr-runtime) and [Zephyr RTOS](https://www.zephyrproject.org/) for execution of model on hardware.

## Prepare an environment for development

This example uses the pre-built Docker image from [Kenning Zephyr Runtime](https://github.com/antmicro/kenning-zephyr-runtime).
To get started, create a workspace directory:

```bash
mkdir zephyr-workspace && cd zephyr-workspace
```

Secondly, create a Docker container with the necessary environment:

```bash
docker run --rm -it -v $(pwd):$(pwd) -w $(pwd) ghcr.io/antmicro/kenning-zephyr-runtime:latest /bin/bash
```

In case the MAX32690 Evaluation Kit is connected to the desktop PC using MAX32625PICO debugger, the Docker container needs to be started in privileged mode, with UART and ACM devices forwarded to it.
Assuming that the programmer is available as `/dev/ttyACM0`, and UART as `/dev/ttyUSB0` in the currently running system, run:

```bash
docker run --privileged --device /dev/ttyACM0 --device /dev/ttyUSB0 --rm -it -v $(pwd):$(pwd) -w $(pwd) ghcr.io/antmicro/kenning-zephyr-runtime:latest /bin/bash
```

Then, in the Docker container clone the Kenning Zephyr Runtime repository and install the latest Zephyr SDK:

```bash
git clone https://github.com/antmicro/kenning-zephyr-runtime
cd kenning-zephyr-runtime/
./scripts/prepare_zephyr_env.sh
./scripts/prepare_modules.sh
source .venv/bin/activate
```

An environment configured this way will allow working with Kenning and Zephyr RTOS.
For more step-by-step instructions on how to set up the environment locally, see [Kenning Zephyr Runtime build instructions](https://github.com/antmicro/kenning-zephyr-runtime/tree/main#building-the-project).

## Install software necessary for flashing the MAX32690 Evaluation Kit

To flash the MAX32690 Evaluation Kit, a Maxim Micros SDK is needed.

Within the Docker image, run:

```bash
mkdir -p /usr/share/applications
wget -O ./MaximMicrosSDK_linux.run https://github.com/analogdevicesinc/msdk/releases/download/v2024_10/MaximMicrosSDK_linux.run
chmod +x ./MaximMicrosSDK_linux.run
./MaximMicrosSDK_linux.run install
```

Now follow along, answering prompts in the installation process.
Once everything is installed successfully, it will be possible to flash the device with the evaluation app from Kenning Zephyr Runtime.

:::{note}
During installation, the script may notify that it was unable to install libncurses5.

In such case, answer `Ignore` and proceed with the installation - it is not needed for this demo.
:::

## Train the sample anomaly detection model (optional)

:::{note}
The pretrained model is available at `https://dl.antmicro.com/kenning/models/anomaly_detection/vae_cats.pth`.
In Kenning, such files are obtainable using the `kenning://` scheme, e.g. `kenning:///models/anomaly_detection/vae_cats.pth`.

To skip the training step, run:

```bash
wget https://dl.antmicro.com/kenning/models/anomaly_detection/vae_cats.pth
wget https://dl.antmicro.com/kenning/models/anomaly_detection/vae_cats.pth.json
```

And proceed to the next section.
:::

Once the environment is set up, the sample model can be trained.
In this demo, a [Variational AutoEncoder (VAE)] will be used.
In Kenning, there is a [PytorchAnomalyDetectionVAE](https://github.com/antmicro/kenning/blob/main/kenning/modelwrappers/anomaly_detection/vae.py) `ModelWrapper` encapsulating the model.

As for dataset, [Controlled Anomalies Time Series (CATS)](https://data.niaid.nih.gov/resources?id=zenodo_7646896) will be used.
It provides telemetry readings of a simulated complex dynamical system with external stimuli.
It provides a nice set of time series for sensors with anomalies.

```bash
kenning train test \
    --dataset-cls kenning.datasets.anomaly_detection_dataset.AnomalyDetectionDataset \
    --dataset-root ./dataset/ \
    --csv-file https://zenodo.org/records/8338435/files/data.csv \
    --modelwrapper-cls kenning.modelwrappers.anomaly_detection.vae.PyTorchAnomalyDetectionVAE \
    --model-path ./vae_cats.pth \
    --batch-size 256 --learning-rate 0.00002 --num-epochs 4 \
    --logdir output/vae_train \
    --verbosity INFO \
    --measurements workspace/vae-torch-native.json \
    --inference-batch-size 256 \
    --batch-norm --loss-beta 0.2 --loss-capacity 0.1
```

The command above:

* Downloads the CATS dataset from https://zenodo.org/records/8338435/files/data.csv
* Creates `AnomalyDetectionDataset` class with dataset data taken from the downloaded CSV
* Creates `PyTorchAnomalyDetectionVAE` model wrapper encapsulating VAE model, providing necessary methods for input preprocessing, output postprocessing, model training and more
* Trains the model using given batch size, learning rate, number of epochs and other exposed training parameters for the model
* Evaluates the model in the end, providing evaluation results in `./workspace/vae-torch-native.json`
* Saves trained model in `./vae_cats.pth` file.

## Deploy the VAE model with TensorFlow Lite Micro

Once the VAE model is trained and available under `./vae_cats.pth`, it can be compiled with `kenning optimize` command and deployed on device using either TensorFlow Lite Micro or microTVM.
This section focuses on TensorFlow Lite Micro.

First, compile the model using `kenning optimize`:

```bash
kenning optimize --cfg ./kenning-scenarios/zephyr-tflite-vae-inference-max32690.yaml
```

It runs the following scenario:

```{literalinclude} ../scripts/configs/zephyr-tflite-vae-inference-max32690.yaml
:language: yaml
```

This will create a `./workspace/vae_cats.tflite` file with an optimized model.

Now, to test the model in simulation or an actual hardware, the evaluation app needs to be compiled.
This can be done with `west build` command with `tflite.conf` configuration for the selected board.
However, this will build TensorFlow Lite Micro runtime with a limited set of operators.
To get the runtime with only necessary set of operators for a given model, it is possible to provide the model that was just created.

To build the evaluation application, run:

```bash
west build -p always -b max32690evkit/max32690/m4 app -- \
      -DEXTRA_CONF_FILE=tflite.conf \
      -DCONFIG_KENNING_MODEL_PATH=\"./workspace/vae_cats.tflite\"
```

Once the model and evaluation app are ready, it is possible to simulate the board in Renode with:

```bash
kenning test --cfg ./kenning-scenarios/zephyr-tflite-vae-inference-max32690-renode.yaml --measurements workspace/vae-tflite-renode.json --verbosity INFO
```

:::{note}
The only difference between `./kenning-scenarios/zephyr-tflite-vae-inference-max32690.yaml` and `./kenning-scenarios/zephyr-tflite-vae-inference-max32690-renode.yaml` is which runtime and protocol is commented out.
Other parts are the same.
:::

The produced `workspace/vae-tflite.json` is a file with raw measurements regarding the model's performance and predictions.

It can be parsed and rendered into a Markdown-based or HTML-based report using the `kenning report` command:

```bash
kenning report --measurements workspace/vae-tflite-renode.json --report-path reports/vae-tflite-renode/report.md --to-html
```

Lastly, the model can be evaluated on actual MAX32690 Evaluation Kit.
First, flash the board with an evaluation app:

```bash
/root/MaximSDK/Tools/OpenOCD/openocd \
    -s /root/MaximSDK/Tools/OpenOCD/scripts/ \
    -c 'source [find interface/cmsis-dap.cfg]' \
    -c 'source [find target/max32690.cfg]' \
    -c 'init' -c 'targets' -c 'reset init' \
    -c 'flash write_image erase ./build/zephyr/zephyr.hex' \
    -c 'reset run' \
    -c 'shutdown'
```

In the end, logs from the flashing process should look as follows to ensure successful flashing:

```
Open On-Chip Debugger (Analog Devices 0.12.0-1.0.0-7)  OpenOCD 0.12.0 (2023-09-27-07:53)
Licensed under GNU GPL v2
Report bugs to <processor.tools.support@analog.com>
Info : CMSIS-DAP: SWD supported
Info : CMSIS-DAP: Atomic commands supported
Info : CMSIS-DAP: Test domain timer supported
Info : CMSIS-DAP: FW Version = 2.0.0
Info : CMSIS-DAP: Serial# = 0409170272c2c8d800000000000000000000000097969906
Info : CMSIS-DAP: Interface Initialised (SWD)
Info : SWCLK/TCK = 1 SWDIO/TMS = 1 TDI = 0 TDO = 0 nTRST = 0 nRESET = 1
Info : CMSIS-DAP: Interface ready
Info : clock speed 2000 kHz
Info : SWD DPIDR 0x2ba01477
Info : [max32xxx.cpu] Cortex-M4 r0p1 processor detected
Info : [max32xxx.cpu] target has 6 breakpoints, 4 watchpoints
Info : starting gdb server for max32xxx.cpu on 3333
Info : Listening on port 3333 for gdb connections
    TargetName         Type       Endian TapName            State
--  ------------------ ---------- ------ ------------------ ------------
 0* max32xxx.cpu       cortex_m   little max32xxx.cpu       running

[max32xxx.cpu] halted due to debug-request, current mode: Thread
xPSR: 0x61000000 pc: 0x10016372 psp: 0x2000afa0
Info : SWD DPIDR 0x2ba01477
[max32xxx.cpu] halted due to debug-request, current mode: Thread
xPSR: 0x01000000 pc: 0x0000fff4 msp: 0x2000a900
auto erase enabled
wrote 131072 bytes from file ./build/zephyr/zephyr.hex in 3.210366s (39.871 KiB/s)

Info : SWD DPIDR 0x2ba01477
shutdown command invoked
```

If `./build/zephyr/zephyr.hex` is successfully written to the device, the model can be tested directly on hardware platform.

To do so, let's use a single-command approach, where `kenning optimize test report` are invoked all at once:

```bash
kenning optimize test report --cfg ./kenning-scenarios/zephyr-tflite-vae-inference-max32690.yaml \
    --measurements workspace/vae-tflite-hw.json \
    --report-path reports/vae-tflite-hw/report.md --to-html \
    --verbosity INFO
```

The `workspace/vae-tflite-hw.json` will hold the collected performance and quality data, and `reports/vae-tflite-hw/report/report.html` will demonstrate the work of the model on hardware.

## Deploy the VAE model with microTVM

With microTVM, the deployment looks similar, similarly as the scenario.
The only difference are used optimizers and runtimes:

```yaml
# ...
optimizers:
    - type: kenning.optimizers.tvm.TVMCompiler
      parameters:
          compiled_model_path: ./workspace/vae.tvm.graph_data
          model_framework: onnx
          target: zephyr
          target_attrs: -keys=arm_cpu,cpu -device=arm_cpu -march=armv7e-m -mcpu=cortex-m4 -model=max32690
# ...

runtime:
    type: kenning.runtimes.tvm.TVMRuntime
    parameters:
        save_model_path: ./workspace/vae.tvm.graph_data

# ...
```

To begin evaluation, run compilation of the evaluation app using microTVM as the runtime:

```bash
west build -p always -b max32690evkit/max32690/m4 app -- -DEXTRA_CONF_FILE='tvm.conf;boards/max32690evkit_max32690_m4.conf' -DKENNING_MODEL_PATH=`realpath ./vae_cats.pth`
```

:::{note}
The `./vae_cats.pth` here is used for similar reasons as in TensorFlow Lite Micro - to provide a minimal set of operations to run models.
The weights of the model are provided separately along with its architecture during evaluation, allowing to test various models without reflashing as long as now new types of layers appear.
:::

After this, run:

```bash
kenning optimize test report --cfg ./kenning-scenarios/zephyr-tvm-vae-inference-max32690-renode.yaml \
    --measurements workspace/vae-tvm-renode.json \
    --report-path reports/vae-tvm-renode/report.md --to-html \
    --verbosity INFO
```

This performs all actions at once - model optimization, model evaluation and report generation.

To test the model on hardware, first flash the device with microTVM-based app:

```bash
/root/MaximSDK/Tools/OpenOCD/openocd \
    -s /root/MaximSDK/Tools/OpenOCD/scripts/ \
    -c 'source [find interface/cmsis-dap.cfg]' \
    -c 'source [find target/max32690.cfg]' \
    -c 'init' -c 'targets' -c 'reset init' \
    -c 'flash write_image erase ./build/zephyr/zephyr.hex' \
    -c 'reset run' \
    -c 'shutdown'
```

And run testing on device (`optimize` is not necessary, since compilation was done before simulation in Renode):

```bash
kenning optimize test report --cfg ./kenning-scenarios/zephyr-tvm-vae-inference-max32690.yaml \
    --measurements workspace/vae-tvm-hw.json \
    --report-path reports/vae-tvm-hw/report.md --to-html \
    --verbosity INFO
```
