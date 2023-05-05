# Visualizing Kenning data flows with Pipeline Manager

[Pipeline Manager](https://github.com/antmicro/kenning-pipeline-manager) is a GUI tool that helps visualize and edit data flows. This chapter describes how to setup and use the manager with Kenning's graphs

![](img/pipeline-manager-visualisation.png)

## Installing Pipeline Manager

Kenning requires extra dependencies to run integration with Pipeline Manager. To install them run: 
```bash
pip install "kenning[pipeline_manager] @ git+https://github.com/antmicro/kenning.git"
```

The following script will download, setup and run the Pipeline Manager server:
```bash
git clone https://github.com/antmicro/kenning-pipeline-manager.git
cd kenning-pipeline-manager
pip install -r requirements.txt
./build server-app
./run
```

## Running Pipeline Manager with Kenning

The server can be started with the following command:
```bash
./run
```

Kenning client can be started with the following command:
```bash
python3 -m kenning.scenarios.pipeline_manager_client --file-path <FILE_PATH>
```
`file-path` allows to define path where the output of a Pipeline Manager command will be stored.

Optional client arguments:
* `spec-type` - There are two graph formats that are defined within the Kenning - [Optimization pipelines](json-scenarios) and [KenningFlows](kenning-flow). The `pipeline` option will allow you to create and edit the optimization pipelines, while `flow` will allow for handling the KenningFlows. Default is `pipeline`
* `host` - Address of Pipeline Manager server
* `port` - Port of  Pipeline Manager server

When the Pipeline Manager is started, the editing of graph can begin - adding or removing nodes and connections, editing node options, etc. The following commands are available when working with the manager:
* `Load file` - Loads JSON describing Kenning dataflow to Pipeline Manager
* `Validate` - Kenning will parse and return the information whether the dataflow is valid (for example it will return error when two optimizer in the chain are incompatible with each other)
* `Run` - Kenning will create and run the dataflow that is in the Pipeline Manager. The results of the run is stored in the output file provided as a client argument
* `Save file` - Exports graph and saves the JSON in the output path

More information regarding information how to work with Pipeline Manager are available in the [project's documentation](https://antmicro.github.io/kenning-pipeline-manager/introduction.html)