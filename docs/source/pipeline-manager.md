# Visualizing Kenning data flows with Pipeline Manager

[Pipeline Manager](https://github.com/antmicro/kenning-pipeline-manager) is a GUI tool that helps visualize and edit data flows. This chapter describes how to setup and use the manager with Kenning's graphs

![](img/pipeline-manager-visualisation.png)

## Installing Pipeline Manager

Kenning requires extra dependencies to run integration with Pipeline Manager. To install them run:
```bash
pip install "kenning[pipeline_manager] @ git+https://github.com/antmicro/kenning.git"
```

To use Pipeline Manager, clone the repository:

```bash
git clone https://github.com/antmicro/kenning-pipeline-manager.git
cd kenning-pipeline-manager
```

And follow installation requirements present in [Pipeline Manager README](https://github.com/antmicro/kenning-pipeline-manager).

After this, build the server application for Pipeline Manager with:

```bash
./build server-app
```

## Running Pipeline Manager with Kenning

Firstly, in the Pipeline Manager project start the server with:

```bash
./run
```

The server now waits for Kenning application to connect.

Secondly, start the Kenning pipeline manager with:

```bash
python3 -m kenning.scenarios.pipeline_manager_client [OPTIONS]
```

Where possible options are:

* `--spec-type` - the type of Kenning scenarios to run, can be either `pipeline` (for [optimization and deployment pipeline](json-scenarios)) or `flow` (for creating [runtime scenarios](kenning-flow)).
  By default it is `pipeline`
* `--file-path` - the file to either store pipeline optimization measurements, or flow's runtime data.
* `--host` - the address of the Pipeline Manager server, default `127.0.0.1`
* `--port` - the port of the Pipeline Manager server, default `9000`
* `--verbosity` - verbosity of the logs

When the Pipeline Manager is started, the editing of graph can begin - adding or removing nodes and connections, editing node options, etc. The following commands are available when working with the manager:
* `Load file` - Loads JSON describing Kenning scenario to Pipeline Manager
* `Validate` - Validates and returns the information whether the scenario is valid (for example it will return error when two optimizers in the chain are incompatible with each other)
* `Run` - Creates and runs the optimization pipeline or [Kenning runtime flow](kenning-flow).
  The results of the run are stored in the output file provided as a client argument.
* `Save file` - Saves the JSON scenario of Kenning to a specified path.

More information regarding information how to work with Pipeline Manager are available in the [Pipeline Manager documentation](https://antmicro.github.io/kenning-pipeline-manager/introduction.html)
