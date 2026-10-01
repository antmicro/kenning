/*
 * Copyright (c) 2022-2024 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* eslint-disable max-classes-per-file */
import {
    reactive, Ref, ref, watch,
} from 'vue';
import { Graph, AbstractNode } from '@baklavajs/core';

import { v4 as uuidv4 } from 'uuid';

import {
    ICommandHandler, ICommand,
    useViewModel,
} from '@baklavajs/renderer-vue';
import notifyEvents from '../custom/notifyEvents.js';

export const suppressingHistory: Ref<boolean> = ref(false);
const transactionId: Ref<string> = ref('');

export interface IHistory {
    max_steps: number;
}

export class Step {
    type: string;

    transactionId: string;

    // I need this to be of basically any type, as
    // it may need to receive objects, arrays of objects, other steps
    /* eslint-disable @typescript-eslint/no-explicit-any */
    topic: any;

    constructor(type: string, topic: any, tid: string = uuidv4()) {
        this.type = type;
        this.topic = topic;
        this.transactionId = tid;
    }

    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    add(graph: Ref<Graph>) {
        throw new Error(`Method add has thrown an error for topic: ${this.topic}`);
    }

    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    remove(graph: Ref<Graph>) {
        throw new Error(`Method remove has thrown an error for topic: ${this.topic}`);
    }

    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    edit(graph: Ref<Graph>) {
        throw new Error(`Method edit has thrown an error for topic: ${this.topic}`);
    }
}

class SubgraphDestroyedStep extends Step {
    // it holds id of graph node and state of graph
    graphTuple: Array<any> = [];

    constructor(type: string, topic: any, tid: string = uuidv4()) {
        if (tid === '') tid = uuidv4(); // eslint-disable-line no-param-reassign
        super(type, topic, tid);
    }

    add(graph: Ref<Graph>) {
        if (this.graphTuple[0] === undefined) {
            return;
        }
        const node:any = graph.value.findNodeById(this.topic);
        if (node === undefined) {
            return;
        }
        const graphState = this.graphTuple[1];

        const { editor } = graph.value;
        const newGraph = this.graphTuple[0];
        newGraph.editor = editor;
        newGraph.load(graphState);
        editor.registerGraph(newGraph);

        node.subgraph = newGraph;
        newGraph.graphNode = node;
        node.updateExposedInterfaces(undefined, undefined, true);
    }

    remove(graph: Ref<Graph>) {
        const node : any = graph.value.findNodeById(this.topic);
        const { subgraph } = node;
        if (subgraph !== undefined) {
            this.graphTuple = [subgraph, subgraph.save()];
            subgraph.destroy?.();
            node.updateExposedInterfaces(undefined, undefined, true);
        }
    }
}

class NodeStep extends Step {
    nodeTuple: Array<any> = [];

    constructor(type: string, topic: any, tid: string = uuidv4()) {
        if (tid === '') tid = uuidv4(); // eslint-disable-line no-param-reassign
        super(type, topic, tid);
    }

    add(graph: Ref<Graph>) {
        if (this.nodeTuple[0] !== undefined) {
            const n = graph.value.addNode(this.nodeTuple[0]);
            n.load(this.nodeTuple[1]);
            if ((<any>graph.value).graphNode !== undefined) {
                (<any>graph.value).graphNode.updateExposedInterfaces();
            }
        }
    }

    remove(graph: Ref<Graph>) {
        const node : any = graph.value.nodes.find((n) => n.id === this.topic);
        if (node !== undefined) {
            this.nodeTuple = [node, node.save()];
            graph.value.removeNode(node);
        }
    }

    edit(graph: Ref<Graph>) {
        if (this.nodeTuple[0] !== undefined) {
            this.nodeTuple[0] = graph.value.nodes.find((n) => n.id === this.nodeTuple[0].id);
            if (this.nodeTuple[0] === undefined) {
                return;
            }
            this.nodeTuple[0] = this.nodeTuple[0] as AbstractNode;
            // save node connections
            const interfaces = [
                ...Object.values(this.nodeTuple[0].inputs),
                ...Object.values(this.nodeTuple[0].outputs),
            ];
            const connections = graph.value.connections.filter(
                (c) => interfaces.includes(c.from) || interfaces.includes(c.to) ||
                    interfaces.some((i: any) =>
                        i.bus?.stubs?.some((s: any) => [c.to, c.from].includes(s),
                        )));

            const save = this.nodeTuple[0].save();
            // remove the current version of the node
            graph.value.removeNode(this.nodeTuple[0]);

            // save the current version and load the previous save
            const n = graph.value.addNode(this.nodeTuple[0]);
            n.load(this.nodeTuple[1], true);
            this.nodeTuple = [this.nodeTuple[0], save];

            // don't save bus interface stubs
            [
                ...Object.values(n.inputs),
                ...Object.values(n.outputs),
            // eslint-disable-next-line no-param-reassign
            ].filter((intf: any) => intf.bus).forEach((intf: any) => delete intf.bus.stubs);

            // restore connections
            connections.forEach((conn) => {
                // if it was a stub, connect to parent
                const to = (<any>conn.to).parent ?? conn.to;
                const from = (<any>conn.from).parent ?? conn.from;
                const offset = (<any>conn.to).offset ?? (<any>conn.from).offset;
                const stubId = (<any>conn.to).offset ? conn.to.id : conn.from.id;
                const stubSide = (<any>conn.to).offset ?
                    (<any>conn.to).side : (<any>conn.from).side;
                // eslint-disable-next-line max-len
                const locc = (<any>graph.value).addConnection(from, to, offset, stubId, stubSide);
                let pos = 0;
                ((<any>conn).anchors ?? []).forEach((anchor: any) => {
                    (<any>graph.value).addAnchor(anchor, locc, pos);
                    pos += 1;
                });
                locc.id = conn.id;
            });
        }
    }
}

class NodeSpecStep extends Step {
    // It is used to store node specification
    specTuple:Array<any> = [];

    constructor(type: string, topic: any, tid: string = uuidv4()) {
        if (tid === '') tid = uuidv4(); // eslint-disable-line no-param-reassign
        super(type, topic, tid);
    }

    updateSpec(specification: any) {
        const nodeType = this.specTuple[0];
        const editorManager = this.specTuple[3];

        if (specification?.simpleInherited) {
            delete specification.simpleInherited; // eslint-disable-line no-param-reassign
        }
        if (specification?.extending) {
            delete specification.extending; // eslint-disable-line no-param-reassign
        }

        editorManager
            .replaceNodeSpecificationInEditor(nodeType, specification);
    }

    remove(graph: Ref<Graph>) {
        // restore node specification
        const specification = this.specTuple[2];

        this.updateSpec(specification);
    }

    add(graph: Ref<Graph>) {
        // restore node specification
        const specification = this.specTuple[1];

        this.updateSpec(specification);
    }
}

class MultipleSteps extends Step {
    // Array of node steps
    steps: Array<Step> = [];

    constructor(type: string, topic: any, tid: string = uuidv4()) {
        if (tid === '') tid = uuidv4(); // eslint-disable-line no-param-reassign
        super(type, topic, tid);
    }

    remove(graph: Ref<Graph>) {
        this.steps.forEach((step) => {
            step.remove(graph);
        });
    }

    add(graph: Ref<Graph>) {
        this.steps.forEach((step) => {
            step.add(graph);
        });
    }

    edit(graph: Ref<Graph>) {
        this.steps.forEach((step) => {
            step.edit(graph);
        });
    }
}

class ConnectionStep extends Step {
    conn: any = undefined;

    constructor(type: string, topic: any, tid: string = uuidv4()) {
        if (tid === '') tid = uuidv4(); // eslint-disable-line no-param-reassign
        super(type, topic, tid);
    }

    add(graph: Ref<Graph>) {
        if (this.conn !== undefined) {
            // The object of the interfaces itself has changed and despite
            // having all the same fields, it will not assign the connection
            // correctly. That's why it is necessary to extract the nodeId
            // from what we have and find the interface in said node manually

            const fromNode = graph.value.findNodeById(this.conn.from.nodeId);
            const toNode = graph.value.findNodeById(this.conn.to.nodeId);
            if (!fromNode || !toNode) return;

            let fromIntf = this.conn.from;
            if (fromIntf.offset) {
                fromIntf = this.conn.from.parent;
            }
            const from = [
                ...Object.values(fromNode.inputs),
                ...Object.values(fromNode.outputs),
            ].filter(
                (iface) => iface.port,
            ).find((iface) => iface.id === fromIntf.id);

            let toIntf = this.conn.to;
            if (toIntf.offset) {
                toIntf = this.conn.to.parent;
            }
            const to = [
                ...Object.values(toNode.inputs),
                ...Object.values(toNode.outputs),
            ].filter(
                (iface) => iface.port,
            ).find((iface) => iface.id === toIntf.id);

            if (!from || !to) return;
            const stubPos = this.conn.to.offset ?? this.conn.from.offset;
            const stubId = this.conn.to.offset ? this.conn.to.id : this.conn.from.id;
            const stubSide = this.conn.to.offset ? this.conn.to.side : this.conn.from.side;

            const connAdded = (<any>graph).value.addConnection(from, to, stubPos, stubId, stubSide);
            if (connAdded === undefined) {
                return;
            }
            connAdded.id = this.conn.id;
        }
    }

    remove(graph: Ref<Graph>) {
        const connToRemove = graph.value.connections.find(
            (n) => n.from.id === this.conn.from.id &&
                n.to.id === this.conn.to.id);
        if (connToRemove !== undefined) {
            this.conn = connToRemove;
            graph.value.removeConnection(connToRemove);
        }
    }
}

class AnchorStep extends Step {
    anchor: any = undefined;

    prevPosition: any = undefined;

    constructor(type: string, topic: any, tid: string = uuidv4()) {
        if (tid === '') tid = uuidv4(); // eslint-disable-line no-param-reassign
        super(type, topic, tid);
    }

    add(graph: Ref<Graph>) {
        if (this.anchor !== undefined) {
            const conn = graph.value.connections.find(
                (n) => n.from.id === this.anchor[0].from.id &&
                n.to.id === this.anchor[0].to.id &&
                n.id === this.anchor[0].id,
            );
            if (conn === undefined) return;
            if ((<any>conn).anchors === undefined) (<any>conn).anchors = [];
            (<any>conn).anchors.splice(
                this.anchor[2], 0, this.anchor[1],
            );
        }
    }

    remove(graph: Ref<Graph>) {
        if (this.anchor !== undefined) {
            const conn = graph.value.connections.find(
                (n) => n.from.id === this.anchor[0].from.id && n.to.id === this.anchor[0].to.id,
            );
            if (conn !== undefined) (<any>conn).anchors.splice(this.anchor[2], 1);
        }
    }

    edit(graph: Ref<Graph>) {
        if (this.anchor !== undefined) {
            const conn = graph.value.connections.find(
                (n) => n.from.id === this.anchor[0].from.id && n.to.id === this.anchor[0].to.id,
            );
            if (conn === undefined) return;
            const prevX = this.anchor[1].x;
            const prevY = this.anchor[1].y;
            this.anchor[1].x = this.prevPosition.x;
            this.anchor[1].y = this.prevPosition.y;
            this.prevPosition.x = prevX;
            this.prevPosition.y = prevY;
            (<any>conn).anchors.splice(
                this.anchor[2], 1, this.anchor[1],
            );
        }
    }
}
class InterfaceStep extends Step {
    intf: any = undefined;

    externalName = '';

    editor: any = undefined;

    exposed = true;

    constructor(type: string, topic: any, tid: string = uuidv4()) {
        if (tid === '') tid = uuidv4(); // eslint-disable-line no-param-reassign
        super(type, topic, tid);
    }

    edit(graph: Ref<Graph>) {
        if (this.intf !== undefined) {
            if (this.exposed) {
                this.editor.privatizeInterface(graph.value.id, this.intf);
            } else {
                this.editor.exposeInterface(graph.value.id, this.intf, this.externalName);
            }
            this.exposed = !this.exposed;
        }
    }
}

class GroupStep extends Step {
    group: any = undefined;

    g: any = undefined;

    constructor(type: string, topic: any, tid: string = uuidv4()) {
        if (tid === '') tid = uuidv4(); // eslint-disable-line no-param-reassign
        super(type, topic, tid);
    }

    remove(graph: Ref<Graph>) {
        const toRem = (<any>graph.value).groups.find((g: any) => g.id === this.group.id);
        (<any>graph.value).removeGroup(toRem);
        this.g = { ...toRem };
        this.g.nodes = [...toRem.nodes];
        this.group = undefined;
    }

    add(graph: Ref<Graph>) {
        this.group = (<any>graph.value).addGroup(
            this.g.name,
            this.g.color,
            this.g.nodes,
            this.g.id,
        );
    }

    edit(graph: Ref<Graph>) {
        const index = (<any>graph.value).groups.findIndex((g: any) => g.id === this.group.id);
        if (index === -1) return;
        const tmp = { ...(<any>graph.value).groups[index] };
        tmp.nodes = [...tmp.nodes];
        // eslint-disable-next-line no-param-reassign
        (<any>graph.value).groups[index] = this.g;
        this.g = tmp;
        this.group = (<any>graph.value).groups[index];
    }
}

export function suppressHistoryLogging(value: boolean) {
    suppressingHistory.value = value;
}

export function startTransaction(id: string = uuidv4()) {
    if (transactionId.value !== '') return undefined;
    transactionId.value = id;
    return id;
}

export function commitTransaction() {
    transactionId.value = '';
}

export function useHistory(graph: Ref<any>, commandHandler: ICommandHandler): IHistory {
    const token = Symbol('CustomHistoryToken');
    const maxSteps = 200;
    const history: Map<string, Step[]> = new Map<string, Step[]>();
    const undoneHistory: Map<string, Step[]> = new Map<string, Step[]>();
    let currentId = 'ThisShouldNotAppearInHistoryMaps';
    let oldId = 'ThisShouldNotAppearInHistoryMaps';
    let justCleared = false;
    let warningCallback: () => void;

    const clearHistory = (callback = () => undefined) => {
        history.forEach((arr, key) => arr.splice(0, arr.length));
        undoneHistory.forEach((arr, key) => arr.splice(0, arr.length));
        justCleared = true;
        warningCallback = callback;
    };
    const unsubscribeFromGraphEvents = (g: any, tok : symbol) => {
        g.events.addNode.unsubscribe(tok);
        g.events.removeNode.unsubscribe(tok);
        g.events.editNode.unsubscribe(tok);
        g.events.dragNodes.unsubscribe(tok);
        g.events.addConnection.unsubscribe(tok);
        g.events.removeConnection.unsubscribe(tok);
        g.events.addAnchor.unsubscribe(tok);
        g.events.removeAnchor.unsubscribe(tok);
        g.events.editAnchor.unsubscribe(tok);
        g.events.exposeInterface.unsubscribe(tok);
        g.events.privatizeInterface.unsubscribe(tok);
        g.events.addGroup.unsubscribe(tok);
        g.events.removeGroup.unsubscribe(tok);
        g.events.editGroup.unsubscribe(tok);
        notifyEvents.subgraphDestroyed.unsubscribe(tok);
        notifyEvents.specificationUpdate.unsubscribe(tok);
    };

    // Switch all the events to any new graph that's displayed
    const graphSwitch = (newGraph : any, oldGraph: any, copyStateStack = false) => {
        if (oldGraph) {
            unsubscribeFromGraphEvents(oldGraph, token);
        }
        if (newGraph) {
            oldId = currentId;
            currentId = newGraph.id;
            if (history.get(currentId) === undefined) history.set(currentId, []);
            if (undoneHistory.get(currentId) === undefined) undoneHistory.set(currentId, []);
            if (copyStateStack) {
                const undoneItem = undoneHistory.get(oldId);
                if (undoneItem) undoneHistory.set(currentId, undoneItem);
                const historyItem = history.get(oldId);
                if (historyItem) history.set(currentId, historyItem);
            }

            const newId = newGraph.id;
            notifyEvents.subgraphDestroyed.subscribe(token, (data:any) => {
                const { node, subgraph } = data;

                if (!suppressingHistory.value) {
                    const historyItem = history.get(newId);
                    if (!historyItem) return;
                    const step = new SubgraphDestroyedStep('rem', node.id.toString(), transactionId.value);
                    historyItem.push(step);
                    step.graphTuple = [subgraph, subgraph.save()];
                    subgraph.destroy?.();
                    undoneHistory.set(newId, []);
                }
            });
            notifyEvents.specificationUpdate.subscribe(token, (data:any) => {
                const {
                    nodeType, specification, newSpecification, editorManager,
                } = data;

                if (!suppressingHistory.value) {
                    const historyItem = history.get(newId);
                    if (!historyItem) return;
                    const step = new NodeSpecStep('add', nodeType, transactionId.value);
                    step.specTuple = [
                        nodeType,
                        newSpecification,
                        specification,
                        editorManager,
                    ];
                    historyItem.push(step);
                    undoneHistory.set(newId, []);
                }
            });

            newGraph.events.addGroup.subscribe(token, (group: any) => {
                if (!suppressingHistory.value) {
                    const historyItem = history.get(newId);
                    if (!historyItem) return;
                    const step = new GroupStep('add', group.id.toString(), transactionId.value);
                    historyItem.push(step);
                    step.group = group;
                    const g = { ...group };
                    g.nodes = [...g.nodes];
                    step.g = g;
                    undoneHistory.set(newId, []);
                }
            });
            newGraph.events.removeGroup.subscribe(token, (group: any) => {
                if (!suppressingHistory.value) {
                    const historyItem = history.get(newId);
                    if (!historyItem) return;
                    const step = new GroupStep('rem', group.id.toString(), transactionId.value);
                    historyItem.push(step);
                    step.group = group;
                    const g = { ...group };
                    g.nodes = [...g.nodes];
                    step.g = g;
                    undoneHistory.set(newId, []);
                }
            });
            newGraph.events.editGroup.subscribe(token, (group: any) => {
                if (!suppressingHistory.value) {
                    const historyItem = history.get(newId);
                    if (!historyItem) return;
                    const step = new GroupStep('edit', group.id.toString(), transactionId.value);
                    historyItem.push(step);
                    step.group = group;
                    const g = { ...group };
                    g.nodes = [...g.nodes];
                    step.g = g;
                    undoneHistory.set(newId, []);
                }
            });
            newGraph.events.addNode.subscribe(token, (node : any) => {
                if (!suppressingHistory.value) {
                    const historyItem = history.get(newId);
                    if (!historyItem) return;
                    historyItem.push(new NodeStep('add', node.id.toString(), transactionId.value));
                    undoneHistory.set(newId, []);
                }
            });
            newGraph.events.removeNode.subscribe(token, (node : any) => {
                if (!suppressingHistory.value) {
                    const historyItem = history.get(newId);
                    if (!historyItem) return;
                    const step = new NodeStep('rem', node.id.toString(), transactionId.value);
                    historyItem.push(step);
                    step.nodeTuple = [node, node.save()];
                    undoneHistory.set(newId, []);
                }
            });
            newGraph.events.editNode.subscribe(token, (node : any) => {
                if (!suppressingHistory.value) {
                    const historyItem = history.get(newId);
                    if (!historyItem) return;
                    const step = new NodeStep('edit', node.id.toString(), transactionId.value);
                    historyItem.push(step);
                    step.nodeTuple = [node, node.save()];
                    undoneHistory.set(newId, []);
                }
            });
            newGraph.events.dragNodes.subscribe(token, (nodes : any) => {
                if (!suppressingHistory.value) {
                    const historyItem = history.get(newId);
                    if (!historyItem) return;
                    const step = new MultipleSteps('edit', nodes[0].id.toString(), transactionId.value);
                    historyItem.push(step);
                    nodes.forEach((node:any) => {
                        const nodeTuple = [node, node.save()];
                        const nodeStep = new NodeStep('edit', node.id.toString(), transactionId.value);
                        nodeStep.nodeTuple = nodeTuple;
                        step.steps.push(nodeStep);
                    });
                    undoneHistory.set(newId, []);
                }
            });
            newGraph.events.addConnection.subscribe(token, (conn : any) => {
                if (!suppressingHistory.value) {
                    const historyItem = history.get(newId);
                    if (!historyItem) return;
                    const step = new ConnectionStep('add', conn.id.toString(), transactionId.value);
                    historyItem.push(step);
                    step.conn = conn;
                    undoneHistory.set(newId, []);
                }
            });
            newGraph.events.removeConnection.subscribe(token, (conn : any) => {
                if (!suppressingHistory.value) {
                    const inTransaction = transactionId.value !== '';
                    if (!inTransaction) startTransaction();
                    const historyItem = history.get(newId);
                    if (!historyItem) return;
                    (conn.anchors ?? []).reverse().forEach((_: any, idx: number) => {
                        newGraph.removeAnchor(conn, idx);
                    });
                    const step = new ConnectionStep('rem', conn.id.toString(), transactionId.value);
                    historyItem.push(step);
                    step.conn = conn;
                    undoneHistory.set(newId, []);
                    if (!inTransaction) commitTransaction();
                }
            });
            newGraph.events.addAnchor.subscribe(token, (tuple: any) => {
                if (!suppressingHistory.value) {
                    const historyItem = history.get(newId);
                    if (!historyItem) return;
                    const idx = tuple[1];
                    const conn = tuple[0];
                    const step = new AnchorStep('add', conn.anchors[idx].id.toString(), transactionId.value);
                    historyItem.push(step);
                    step.anchor = [conn, conn.anchors[idx], idx];
                    undoneHistory.set(newId, []);
                }
            });
            newGraph.events.editAnchor.subscribe(token, (tuple: any) => {
                if (!suppressingHistory.value) {
                    const historyItem = history.get(newId);
                    if (!historyItem) return;
                    const idx = tuple[1];
                    const conn = tuple[0];
                    const prevPos = tuple[3];
                    const step = new AnchorStep('edit', conn.anchors[idx].id.toString(), transactionId.value);
                    historyItem.push(step);
                    step.anchor = [conn, conn.anchors[idx], idx];
                    step.prevPosition = prevPos;
                    undoneHistory.set(newId, []);
                }
            });
            newGraph.events.removeAnchor.subscribe(token, (tuple: any) => {
                if (!suppressingHistory.value) {
                    const historyItem = history.get(newId);
                    if (!historyItem) return;
                    const idx = tuple[1];
                    const conn = tuple[0];
                    const step = new AnchorStep('rem', conn.anchors[idx].id.toString(), transactionId.value);
                    historyItem.push(step);
                    step.anchor = [conn, conn.anchors[idx], idx];
                    undoneHistory.set(newId, []);
                }
            });
            newGraph.events.exposeInterface.subscribe(token, (tuple: any) => {
                if (!suppressingHistory.value) {
                    const historyItem = history.get(newId);
                    if (!historyItem) return;
                    const intf = tuple[0];
                    const editor = tuple[1];
                    const step = new InterfaceStep('edit', intf.id.toString(), transactionId.value);
                    step.externalName = intf.externalName;
                    step.intf = intf;
                    step.editor = editor;
                    step.exposed = true;
                    historyItem.push(step);
                    undoneHistory.set(newId, []);
                }
            });
            newGraph.events.privatizeInterface.subscribe(token, (tuple: any) => {
                if (!suppressingHistory.value) {
                    const historyItem = history.get(newId);
                    if (!historyItem) return;
                    const intf = tuple[0];
                    const editor = tuple[1];
                    const step = new InterfaceStep('edit', intf.id.toString(), transactionId.value);
                    step.externalName = intf.externalName;
                    step.intf = intf;
                    step.editor = editor;
                    step.exposed = false;
                    historyItem.push(step);
                    undoneHistory.set(newId, []);
                }
            });
        }
    };

    watch(graph, (newGraph, oldGraph) => graphSwitch(newGraph, oldGraph), { flush: 'post', immediate: true });

    const singleStepTransaction = (mainHistory: Step[], auxiliaryHistory:Step[]) => {
        const step : Step | undefined = mainHistory.pop();
        if (step === undefined) return;
        suppressingHistory.value = true;
        if (step.type === 'add') {
            step.type = 'rem';
            step.remove(graph);
        } else if (step.type === 'rem') {
            step.type = 'add';
            step.add(graph);
        } else if (step.type === 'edit') {
            step.type = 'edit';
            step.edit(graph);
        }
        auxiliaryHistory.push(step);
        if (
            mainHistory.length > 0 &&
            mainHistory[mainHistory.length - 1].transactionId === step.transactionId
        ) singleStepTransaction(mainHistory, auxiliaryHistory);
        suppressingHistory.value = false;
        graph.value.selectedNodes.splice(0, graph.value.selectedNodes.length);
    };

    commandHandler.registerCommand<ICommand<void>>('undo', {
        canExecute: () => true,
        execute: () => {
            const historyItem = history.get(currentId);
            if (historyItem && historyItem.length !== 0) {
                const undoneItem = undoneHistory.get(currentId);
                if (historyItem && undoneItem) singleStepTransaction(historyItem, undoneItem);
            }
            if (historyItem && historyItem.length === 0 && justCleared) {
                justCleared = false;
                if (warningCallback) {
                    warningCallback();
                }
            }
        },
    });

    commandHandler.registerCommand<ICommand<void>>('redo', {
        canExecute: () => true,
        execute: () => {
            const historyItem = history.get(currentId);
            const undoneItem = undoneHistory.get(currentId);
            if (historyItem && undoneItem && undoneItem.length !== 0) {
                singleStepTransaction(undoneItem, historyItem);
            }
            if (undoneItem?.length === 0 && justCleared) {
                justCleared = false;
                if (warningCallback) {
                    warningCallback();
                }
            }
        },
    });
    commandHandler.registerCommand<ICommand<void>>('START_TRANSACTION', {
        canExecute: () => transactionId.value === '',
        execute: () => startTransaction,
    });
    commandHandler.registerCommand<ICommand<void>>('COMMIT_TRANSACTION', {
        canExecute: () => transactionId.value !== '',
        execute: () => commitTransaction,
    });

    commandHandler.registerHotkey(['Control', 'z'], 'undo');
    commandHandler.registerHotkey(['Control', 'y'], 'redo');

    return reactive({
        max_steps: maxSteps,
        graphSwitch,
        unsubscribeFromGraphEvents,
        clearHistory,
    });
}
