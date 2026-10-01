/*
 * Copyright (c) 2022-2025 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { toPng } from 'html-to-image';
import EditorManager from '../core/EditorManager';
import NotificationHandler from '../core/notifications';
import { brokenImage } from '../../../resources/broken_image.js';

interface SaveConfiguration {
    readonly?: boolean;
    hideHud?: boolean;
    position?: boolean;
    hideLayers?: boolean;
    graph?: boolean;
    minify?: boolean;
    graphName?: string | null;
    saveName: string;

    getSave: () => object;

    saveCallback: () => void;
    saveCallbackCustomFormat?: (blob: string) => void;

    reset(): void;
}

const saveBlob = (blob: Blob, filename: string) => {
    const linkElement = document.createElement('a');
    linkElement.href = window.URL.createObjectURL(blob);
    linkElement.download = filename;
    linkElement.click();
};

export const saveSpecificationConfiguration: SaveConfiguration = {
    graph: false,
    minify: false,
    saveName: 'specification',

    getSave() {
        const editorManager = EditorManager.getEditorManagerInstance();
        const specification = editorManager.saveSpecification();
        const dataflow = editorManager.saveDataflow();
        specification.graphs ??= [];

        if (this.graph) {
        // Match graph IDs with subgraphId in node specification
            dataflow.graphs.forEach((graph: any) => {
                dataflow.graphs.forEach((g: any) => {
                    const checkedId = graph.id;
                    const subgraphNode = g.nodes.find((n: any) => n.subgraph === checkedId);
                    if (subgraphNode !== undefined) {
                        const nodeSpecification = specification.nodes?.find(
                            (n: any) => n.name === subgraphNode.name,
                        );
                        if (nodeSpecification) {
                            // eslint-disable-next-line no-param-reassign
                            graph.id = nodeSpecification.subgraphId;
                            subgraphNode.subgraph = nodeSpecification.subgraphId;
                        }
                    }

                    const relatedGraphNode = g.nodes.find((n: any) =>
                        n.relatedGraphs?.some((entry: any) => entry.id === checkedId),
                    );
                    const graphButton = g.nodes.find((n: any) =>
                        n.properties?.some(
                            (prop: any) => prop.type === 'button-graph' && prop.default === checkedId,
                        ),
                    );

                    if (subgraphNode || relatedGraphNode || graphButton) {
                        if (Array.isArray(graph.nodes) && graph.nodes.length) {
                            specification.graphs = specification.graphs.filter(
                                (oldGraph: any) => oldGraph.id !== graph.id,
                            );
                            specification.graphs.push(graph);
                        } else {
                        // Remove empty graphs
                            if (subgraphNode) {
                                const nodesIndex = specification.nodes
                                    .map((n: any) => n.name)
                                    .indexOf(subgraphNode.name);
                                const graphsIndex = g.nodes
                                    .map((n: any) => n.name)
                                    .indexOf(subgraphNode.name);
                                delete specification.nodes[nodesIndex].subgraphId;
                                // eslint-disable-next-line no-param-reassign
                                delete g.nodes[graphsIndex].subgraph;
                            }
                            if (relatedGraphNode) {
                                const nodesIndex = specification.nodes
                                    .map((n: any) => n.name)
                                    .indexOf(relatedGraphNode.name);
                                const graphsIndex = g.nodes
                                    .map((n: any) => n.name)
                                    .indexOf(relatedGraphNode.name);
                                if (relatedGraphNode.relatedGraphs.length > 1) {
                                    specification.nodes[nodesIndex].relatedGraphs =
                                    relatedGraphNode.relatedGraphs.filter(
                                        (entry: any) => entry.id !== checkedId,
                                    );
                                    // eslint-disable-next-line no-param-reassign
                                    g.nodes[graphsIndex].relatedGraphs =
                                    relatedGraphNode.relatedGraphs.filter(
                                        (entry: any) => entry.id !== checkedId,
                                    );
                                } else {
                                    delete specification.nodes[nodesIndex].relatedGraphs;
                                    // eslint-disable-next-line no-param-reassign
                                    delete g.nodes[graphsIndex].relatedGraphs;
                                }
                            }
                            if (graphButton) {
                                const nodesIndex = specification.nodes
                                    .map((n: any) => n.name)
                                    .indexOf(graphButton.name);
                                const graphsIndex = g.nodes
                                    .map((n: any) => n.name)
                                    .indexOf(graphButton.name);
                                specification.nodes[nodesIndex].properties =
                                graphButton.properties.filter(
                                    (prop: any) =>
                                        !(
                                            prop.type === 'button-graph' &&
                                            prop.default === checkedId
                                        ),
                                );
                                // eslint-disable-next-line no-param-reassign
                                g.nodes[graphsIndex].properties = graphButton.properties.filter(
                                    (prop: any) =>
                                        !(prop.type === 'button-graph' && prop.default === checkedId),
                                );
                            }
                        }
                    }
                });
            });
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            dataflow.graphs.forEach((graph: any) => {
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                const prevIndex = specification.graphs.map((g: any) => g.id).indexOf(graph.id);
                const [index, remove] = prevIndex !== -1
                    ? [prevIndex, 1]
                    : [specification.graphs.length, 0];
                specification.graphs.splice(index, remove, graph);
            });

            specification.graphs = specification.graphs.filter(
                (graph: any) => Array.isArray(graph.nodes) && graph.nodes.length,
            );

            if (specification.graphs.values().some((g:any) => g.id === dataflow.entryGraph)) {
                specification.entryGraph = dataflow.entryGraph;
            } else {
                specification.entryGraph = specification.graphs[0]?.id;
            }
        }

        Object.entries(dataflow.metadata).forEach(([key, value]) => {
            if (specification.metadata === undefined) {
                specification.metadata = {};
            }
            if (value !== undefined) {
                specification.metadata[key] = value;
            }
        });

        if (this.minify && specification.nodes) {
            const usedNames = EditorManager.getUsedNames([
                ...dataflow.graphs,
                ...specification.graphs,
            ]);
            specification.nodes =
                EditorManager.minifySpecificationNodes(specification.nodes, usedNames);
            if (!specification.nodes.length) delete specification.nodes;
        }
        return specification;
    },

    saveCallback() {
        const specification = this.getSave();
        const blob = new Blob(
            [
                JSON.stringify(
                    specification,
                    null,
                    this.minify ? 0 : 4,
                ),
            ],
            { type: 'application/json' },
        );
        saveBlob(blob, this.saveName);
        NotificationHandler.showToast('info', 'Specification saved');
    },

    reset() {
        this.graph = false;
        this.minify = false;
        this.saveName = 'specification';
    },
};

export const saveGraphConfiguration: SaveConfiguration = {
    readonly: false,
    hideHud: false,
    position: false,
    hideLayers: false,
    graphName: null,
    saveName: 'save',

    getSave() {
        const editorManager = EditorManager.getEditorManagerInstance();
        return editorManager.saveDataflow({
            readonly: !!this.readonly,
            hideHud: !!this.hideHud,
            position: !!this.position,
            hideLayers: !!this.hideLayers,
            graphName: this.graphName,
        });
    },

    saveCallback() {
        const dataflow = this.getSave();
        const blob = new Blob([JSON.stringify(dataflow, null, 4)], {
            type: 'application/json',
        });
        saveBlob(blob, this.saveName);
        NotificationHandler.showToast('info', 'Dataflow saved');
    },

    saveCallbackCustomFormat(blob: string) {
        const saveElement = document.createElement('a');
        let mimeType;
        if (typeof blob === 'string') {
            mimeType = 'application/octet-stream';
            saveElement.href = `data:${mimeType};base64,${blob}`;
        } else {
            mimeType = 'application/json';
            saveElement.href = window.URL.createObjectURL(
                new Blob(
                    [JSON.stringify(blob)],
                    { type: mimeType }),
            );
        }
        saveElement.download = this.saveName;
        saveElement.click();
        NotificationHandler.showToast('info', `File saved successfully: ${this.saveName}`);
    },

    reset() {
        this.readonly = false;
        this.hideHud = false;
        this.position = false;
        this.graphName = null;
        this.saveName = 'save';
    },
};

export const exportGraph = {
    width: 1920,
    height: 1080,
    maxWidth: 16384,
    maxHeight: 16384,
    saveName: 'dataflow',

    exportCallback() {
        // Get editor with data flow
        const nodeEditor: HTMLElement = document.querySelector('.inner-editor')!;
        // Exclude node palette
        const filter = (node: any) => !node.classList?.contains('baklava-node-palette');

        if (this.width < 0) {
            NotificationHandler.showToast('error', `Negative width. The image could not be saved.`);
            return;
        }
        if (this.height < 0) {
            NotificationHandler.showToast('error', `Negative height. The image could not be saved.`);
            return;
        }
        if (this.width > this.maxWidth) {
            NotificationHandler.showToast('warning', `Invalid width. The maximum value ${this.maxWidth} was used instead.`);
            this.width = this.maxWidth;
        }
        if (this.height > this.maxHeight) {
            NotificationHandler.showToast('warning', `Invalid height. The maximum value ${this.maxHeight} was used instead.`);
            this.height = this.maxHeight;
        }

        toPng(nodeEditor, {
            filter,
            imagePlaceholder: brokenImage,
            canvasWidth: this.width,
            canvasHeight: this.height,
        })
            .then((dataUrl) => {
                const downloadLink = document.createElement('a');
                downloadLink.download = this.saveName;
                downloadLink.href = dataUrl;
                downloadLink.dataset.downloadurl = [
                    dataUrl,
                    downloadLink.download,
                    downloadLink.href,
                ].join(':');
                document.body.appendChild(downloadLink);
                downloadLink.click();
                document.body.removeChild(downloadLink);
            })
            .catch((error) => {
                NotificationHandler.showToast('error', `Export to PNG failed: ${error}`);
            });
    },

    reset() {
        this.width = 1920;
        this.height = 1080;
        this.saveName = 'save';
    },
};
