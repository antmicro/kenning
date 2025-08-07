# Generating anomaly detection models for the MAX32690 Evaluation Kit with AutoML

This example demonstrates how to find anomaly detection models and evaluate them on a simulated MCU using Kenning, Zephyr RTOS and Renode.
The models are generated with the [Auto-PyTorch](https://github.com/antmicro/auto-pytorch) AutoML framework.
The platform used in this example is the [MAX32690 Evaluation Kit](https://www.analog.com/en/resources/evaluation-hardware-and-software/evaluation-boards-kits/max32690evkit.html#eb-overview).
The demo uses [Kenning Zephyr Runtime](https://github.com/antmicro/kenning-zephyr-runtime) and [Zephyr RTOS](https://www.zephyrproject.org/) for execution of the model on simulated hardware.

## Prepare an environment for development

Assuming `git` and `docker` are available in the system, first let's clone the repository:

```bash
git clone https://github.com/antmicro/kenning-zephyr-runtime-example-app.git sample-app
```

Then, let's build the Docker image based on `ghcr.io/antmicro/kenning-zephyr-runtime:latest` for quicker environment setup:

```bash
docker build -t kenning-automl ./sample-app/environments
```

After successful build of the image, run:

```bash
docker run --rm -it --name automl -w $(realpath sample-app) -v $(pwd):$(pwd) kenning-automl:latest bash
```

Then, in the Docker container, initialize the Zephyr application and Kenning Zephyr Runtime as follows:

```bash
west init -l app
west update
west zephyr-export
pushd ./kenning-zephyr-runtime
./scripts/prepare_zephyr_env.sh
./scripts/prepare_modules.sh
popd
```

:::{note}
To make sure you have the latest version of the Kenning with AutoML features, run:

```bash
pip3 install "kenning[iree,tvm,torch,anomaly_detection,auto_pytorch,tensorflow,tflite,reports,renode,uart] @ git+https://github.com/antmicro/kenning.git"
```
:::

With the configured environment, you can now run the AutoML flow.

To create a `workspace` directory where intermediate results of command executed further will be stored, run:

```bash
mkdir -p workspace
```

:::{note}
For more step-by-step instructions on how to set up the environment locally, see [Kenning Zephyr Runtime build instructions](https://github.com/antmicro/kenning-zephyr-runtime/tree/main#user-content-building-the-project).
:::

## Run AutoML flow

The AutoML flow can be configured using the YAML below.

```{literalinclude} ../scripts/configs/automl-scenario.yml
:language: yaml
:emphasize-lines: 5-28
```

The introduced `automl` entry provides an implementation of the AutoML flow, as well as its parameters.
The main parameters affecting the model search process are:

* `time_limit` - determines how much time the AutoML algorithm will spend looking for the models
* `application_size` - determines how much of the space is consumed by the app excluding the model.
  By taking into account the provided size of the application ran on the board and the size of RAM (received from the platform class), the flow will automatically reject (before training) all models that do not fit into available space.
  This ensures than time is not wasted on models that cannot be run on the hardware (or its simulation).
* `use_models` - provides models for the algorithm to take into account.
  Models provided in the list contribute to the search space of available configurations, providing their hyperparameters with acceptable ranges and structure definition.
  It is possible to override default ranges specifying them right after chosen model:
  ```yaml
  use_models:
    - PyTorchAnomalyDetectionVAE:
        encoder_neuron_list:
          list_range: [4, 15]
          item_range: [6, 128]
        dropout_rate:
          item_range: [0.1, 0.4]
        output_activation:
          enum: [tanh]
  ```
  The list of AutoML specific options is available in [Defining arguments for classes section](defining-arguments-for-core-classes).

:::{note}
To use the microTVM runtime, change the optimizer to:

```yaml
optimizers:
- type: TVMCompiler
  parameters:
    compiled_model_path: ./workspace/vae.graph_data
```

Depending on runtime selection, application size may vary greatly, so to generate models with correct size, make sure to adjust this value.
:::

To run the full flow, use this command:

```bash
kenning automl optimize test report \
  --cfg ./kenning-zephyr-runtime/kenning-scenarios/renode-zephyr-auto-tflite-automl-vae-max32690.yml \
  --report-path ./workspace/automl-report/report.md \
  --allow-failures --to-html --ver INFO \
  --skip-general-information
```

The command above:

* Runs an AutoML search for models for the amount of time specified in the `time_limit`
* Optimizes best-performing using given optimization pipeline
* Runs evaluation of the compiled models in Renode simulation
* Generates a full comparison report for the models so that user can pick the best one (located in `workspace/automl-report/report/report.html`)
* Generates optimized models (`vae.<id>.tflite`), their AutoML-derived configuration (`automl_conf_<id>.yml`) and IO specification files for Kenning (`vae.<id>.tflite.json`) under `workspace/automl-results/`.

## Run AutoML flow with quantization

The AutoML flow can also take into account the model size after quantization.
To make it possible, a scenario has to define at least one Optimizer which will quantize the model:

```{literalinclude} ../scripts/configs/automl-cnn-scenario.yml save-as=automl_quantization.yml
:name: automl-quantization-scenario
:language: yaml
:emphasize-lines: 31-38
```

In order to avoid training models that will not fit into available space, the flow triggers quantization on initialized models and rejects ones that are too large.
If model fits in the memory, the flow proceeds with training of default (non-quantized) models.
In the end, it quantizes and evaluates the best models based on the training process.
Such flow can be run with the following command, assuming [the scenario](automl-quantization-scenario) is saved as `automl_quantization.yml`:

```bash
kenning automl optimize test report \
  --cfg ./automl_quantization.yml \
  --report-path ./workspace/automl-report/report.md \
  --allow-failures --to-html --ver INFO \
  --skip-general-information
```

## Run sample app with a chosen model

Once the best model is selected (e.g. `workspace/automl-results/vae.0.tflite`), let's compile the sample app with it:

```bash
west build \
  -p always \
  -b max32690evkit/max32690/m4 app -- \
  -DEXTRA_CONF_FILE="tflite.conf" \
  -DCONFIG_KENNING_MODEL_PATH=\"$(realpath workspace/automl-results/vae.0.tflite)\"
west build -t board-repl
```

:::{note}
To use microTVM runtime, change `-DEXTRA_CONF_FILE` to `"tvm.conf"` and `-DCONFIG_KENNING_MODEL_PATH` to a chosen model compiled with TVM (usually with `.graph_data` extension).
:::

In the end, the app with the model can be simulated with:

```bash test-skip
python3 kenning-zephyr-runtime/scripts/run_renode.py --no-kcomm
```

To end the simulation, press `Ctrl+C`.
