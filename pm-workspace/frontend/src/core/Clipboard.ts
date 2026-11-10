/*
 * Copyright (c) 2022-2024 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
The baklavajs functionality was fully copied to this file, as the original
implementation is not exported and cannot be integrated with custom history
as the Commands have been replaced.

The original source is slightly adjusted to integrate with the customized history
and the new command handler
*/

import {
    computed, reactive, Ref, ref,
} from 'vue';
import { v4 as uuidv4 } from 'uuid';
import {
    AbstractNode,
    INodeState,
    IConnectionState,
    Connection,
    NodeInterface,
} from '@baklavajs/core';
import {
    ICommandHandler, ICommand, useViewModel,
} from '@baklavajs/renderer-vue';
import {
    startTransaction, commitTransaction,
} from './History.ts';
import { removeNode } from '../custom/CustomNode.js';

export const COPY_COMMAND = 'COPY';
export const DELETE_COMMAND = 'DELETE';
export const UNHIGHLIGHT_COMMAND = 'UNHIGHLIGHT';
export const PASTE_COMMAND = 'PASTE';
export const UNPASTE_COMMAND = 'UNPASTE';
export const CLEAR_CLIPBOARD_COMMAND = 'CLEAR_CLIPBOARD';

/* eslint-disable  @typescript-eslint/no-explicit-any */
interface PMNodeState extends INodeState<any, any> {
    name: string;
    subgraph?: string;
    graphState?: any;
    interfaces: any[];
}

export interface IClipboard {
    isEmpty: boolean;
}

export function useClipboard(
    displayedGraph: any,
    editor: Ref<any>,
    commandHandler: ICommandHandler,
): IClipboard {
    const token = Symbol('ClipboardToken');

    const nodeBuffer = ref('');
    const connectionBuffer = ref('');

    const consecutivePasteNumber = ref(0);

    const isEmpty = computed(() => !nodeBuffer.value);

    const clear = () => {
        nodeBuffer.value = '';
        connectionBuffer.value = '';
    };

    const copy = () => {
        // find all connections from and to the selected nodes
        const interfacesOfSelectedNodes = displayedGraph.value.selectedNodes.flatMap((n : any) => [
            ...Object.values(n.inputs),
            ...Object.values(n.outputs),
        ]);

        const connections = displayedGraph.value.connections
            .filter(
                (conn : any) => interfacesOfSelectedNodes.includes(conn.from) ||
                                interfacesOfSelectedNodes.includes(conn.to),
            )
            .map((conn : any) => ({ from: conn.from.id, to: conn.to.id } as IConnectionState));

        connectionBuffer.value = JSON.stringify(connections);
        nodeBuffer.value = JSON.stringify(displayedGraph.value.selectedNodes.map(
            (n : any) => n.save(),
        ));

        consecutivePasteNumber.value = 0;
    };

    const del = () => {
        const { viewModel } = useViewModel() as { viewModel: any };
        if (viewModel.value.editor.readonly) return;
        startTransaction();
        displayedGraph.value.selectedNodes.forEach((node : any) => {
            removeNode(node);
        });
        commitTransaction();
    };

    const findInterface = (
        nodes: AbstractNode[],
        id: string,
        io?: 'input' | 'output',
    ): NodeInterface<any> | undefined => {
        for (let i = 0; i < nodes.length; i += 1) {
            let intf: NodeInterface<any> | undefined;
            if (!io || io === 'input') {
                intf = Object.values(nodes[i].inputs).find((iface) => iface.id === id);
            }
            if (!intf && (!io || io === 'output')) {
                intf = Object.values(nodes[i].outputs).find((iface) => iface.id === id);
            }
            if (intf) {
                return intf;
            }
        }
        return undefined;
    };

    const paste = () => {
        const { viewModel } = useViewModel() as { viewModel: any };
        if (isEmpty.value || viewModel.value.editor.readonly) {
            return;
        }
        startTransaction();
        const movementStep = (<any>viewModel.value).movementStep ?? 1;

        // Map old IDs to new IDs
        const idmap = new Map<string, string>();

        const parsedNodeBuffer = JSON.parse(nodeBuffer.value) as PMNodeState[];
        const parsedConnectionBuffer = JSON.parse(connectionBuffer.value) as IConnectionState[];

        const newNodes: AbstractNode[] = [];
        const newConnections: Connection[] = [];

        const graph = displayedGraph.value;

        commandHandler.executeCommand<ICommand<void>>('START_TRANSACTION');

        for (let i = 0; i < parsedNodeBuffer.length; i += 1) {
            const nodeType = editor.value.nodeTypes.get(parsedNodeBuffer[i].name);

            if (!nodeType) {
                return;
            }
            /* eslint-disable-next-line new-cap */
            let copiedNode = new nodeType.type();

            newNodes.push(copiedNode);

            copiedNode.hooks.beforeLoad.subscribe(token, (nodeState: any) => {
                const ns = nodeState as any;
                if (ns.position) {
                    consecutivePasteNumber.value += 1;

                    ns.position.x += consecutivePasteNumber.value * Math.max(40, movementStep);
                    ns.position.y += consecutivePasteNumber.value * Math.max(40, movementStep);
                }
                if (parsedNodeBuffer[i].graphState !== undefined) {
                    parsedNodeBuffer[i].graphState.nodes.forEach((n: any) => {
                        /* eslint-disable-next-line no-param-reassign */
                        if (n.type !== undefined) n.name = n.type;
                    });
                }
                copiedNode.hooks.beforeLoad.unsubscribe(token);
                return ns;
            });

            copiedNode = graph.addNode(copiedNode);

            const mapNewId = (obj: { id: string; [key: string]: any; }) => {
                /* eslint-disable no-param-reassign */
                const newId = uuidv4();
                idmap.set(obj.id, newId);
                obj.id = newId;
            };

            const assignNewIds = (node: PMNodeState, curGraph: any) => {
                /* eslint-disable no-param-reassign */

                // New node id
                mapNewId(node);

                if (node.graphState !== undefined) {
                    mapNewId(node.graphState);

                    node.graphState.nodes.forEach((subNode: PMNodeState) => {
                        assignNewIds(subNode, undefined);
                    });

                    // If it is a subgraph node, then some interfaces have to have the same IDs
                    // as the ones in the subgraph
                    node.interfaces.forEach((intf: any) => {
                        const mapped = idmap.get(intf.id);
                        if (mapped === undefined) {
                            mapNewId(intf);
                        } else {
                            intf.id = mapped;
                        }
                        // If the node has any external interfaces, then their names have
                        // to be resolved as they cannot conflict with the existing ones.
                        if (intf.externalName !== undefined && curGraph !== undefined) {
                            intf.externalName = graph.resolveNewExposedName(intf.externalName);
                        }
                    });

                    node.graphState.connections.forEach((conn: any) => {
                        if (
                            idmap.get(conn.from) === undefined ||
                            idmap.get(conn.to) === undefined
                        ) {
                            throw new Error(
                                'Error when executing copy and paste. ' +
                                `Connection from interface ${conn.from} to ${conn.to} is invalid`,
                            );
                        }
                        conn.from = idmap.get(conn.from);
                        conn.to = idmap.get(conn.to);
                    });
                } else {
                    // If it is a regular node, then interfaces need new IDs.
                    node.interfaces.forEach((intf: any) => {
                        mapNewId(intf);

                        // If the node has any external interfaces, then their names have
                        // to be resolved as they cannot conflict with the existing ones.
                        if (intf.externalName !== undefined && curGraph !== undefined) {
                            intf.externalName = graph.resolveNewExposedName(intf.externalName);
                        }
                    });
                }
            };
            const updateInterfaces = (node: any) => {
                if (node.subgraph !== undefined) {
                    const { subgraph } = node;
                    subgraph.nodes.forEach((subNode: PMNodeState) => {
                        updateInterfaces(subNode);
                    });
                    if (subgraph.graphNode === undefined) {
                        return;
                    }

                    subgraph.graphNode.updateExposedInterfaces(undefined, undefined, true);
                }
            };

            assignNewIds(parsedNodeBuffer[i], graph);
            copiedNode.load({ ...parsedNodeBuffer[i], id: copiedNode.id });
            updateInterfaces(copiedNode);

            // If the pasted graph was inside of a graph node, then the graph node has to
            // have its exposed interfaces refreshed
            if (displayedGraph.value.graphNode !== undefined) {
                displayedGraph.value.graphNode.updateExposedInterfaces();
            }
        }

        for (let i = 0; i < parsedConnectionBuffer.length; i += 1) {
            const fromId = idmap.get(parsedConnectionBuffer[i].from);
            const toId = idmap.get(parsedConnectionBuffer[i].to);
            if (fromId && toId) {
                const fromIntf = findInterface(newNodes, fromId, 'output');
                const toIntf = findInterface(newNodes, toId, 'input');
                if (fromIntf && toIntf) {
                    const newConnection = graph.addConnection(fromIntf, toIntf);
                    if (newConnection) {
                        newConnections.push(newConnection);
                    }
                }
            }
        }

        commitTransaction();

        /* eslint-disable-next-line consistent-return */
        return {
            newNodes,
            newConnections,
        };
    };

    commandHandler.registerCommand(DELETE_COMMAND, {
        canExecute: () => displayedGraph.value.selectedNodes.length > 0,
        execute: del,
    });
    commandHandler.registerHotkey(['Delete'], DELETE_COMMAND);
    commandHandler.registerCommand(UNHIGHLIGHT_COMMAND, {
        canExecute: () => true,
        execute: () => {
            displayedGraph.value.selectedNodes = []; // eslint-disable-line no-param-reassign
        },
    });
    commandHandler.registerHotkey(['Escape'], UNHIGHLIGHT_COMMAND);
    commandHandler.registerCommand(COPY_COMMAND, {
        canExecute: () => true,
        execute: copy,
    });
    commandHandler.registerHotkey(['Control', 'c'], COPY_COMMAND);
    commandHandler.registerCommand(PASTE_COMMAND, {
        canExecute: () => !isEmpty.value,
        execute: paste,
    });
    commandHandler.registerHotkey(['Control', 'z'], UNPASTE_COMMAND);
    commandHandler.registerCommand(UNPASTE_COMMAND, {
        canExecute: () => true,
        execute: () => { consecutivePasteNumber.value -= 1; },
    });
    commandHandler.registerHotkey(['Control', 'v'], PASTE_COMMAND);
    commandHandler.registerCommand(CLEAR_CLIPBOARD_COMMAND, {
        canExecute: () => true,
        execute: clear,
    });

    return reactive({ isEmpty });
}
