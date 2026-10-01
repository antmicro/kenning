/*
 * Copyright (c) 2022-2024 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Implements custom version of baklava's Graph object
 */

import { DummyConnection, Connection } from '@baklavajs/core';
import { v4 as uuidv4 } from 'uuid';
import { BaklavaEvent } from '@baklavajs/events';
import { toRaw } from 'vue';
import { useViewModel } from '@baklavajs/renderer-vue';
import { startTransaction, commitTransaction } from '../core/History.ts';
import { updateInterfacePosition } from './CustomNode.js';
import GraphTemplate from './CustomGraphTemplate.js';

/* eslint-disable no-param-reassign */
/* eslint-disable no-underscore-dangle */
export default function createPipelineManagerGraph(graph) {
    // Add an event for adding an anchor to the graph
    graph.events.addAnchor = new BaklavaEvent();
    graph.events.removeAnchor = new BaklavaEvent();

    // Add an event for editing node
    graph.events.editNode = new BaklavaEvent();
    graph.disableDestroy = false;

    // Add event for moving anchors.
    graph.events.editAnchor = new BaklavaEvent();
    // Add event for dragging multiple nodes
    graph.events.dragNodes = new BaklavaEvent();

    // Add an event for editing node
    graph.events.exposeInterface = new BaklavaEvent();
    graph.events.privatizeInterface = new BaklavaEvent();

    // group events
    graph.events.addGroup = new BaklavaEvent();
    graph.events.removeGroup = new BaklavaEvent();
    graph.events.editGroup = new BaklavaEvent();

    // Graph node that represents the graph itself. Root graph does not have a node graph assigned.
    graph.graphNode = undefined;
    graph.groups = [];

    graph._idToNode = {};

    // A value that indicates whether we should save the graph or not
    graph.toSave = true;
    // A value that indicate whether graph was loaded from specification or not
    graph.specGraph = false;

    graph.setToSave = function setToSave(save) {
        this.toSave = save;
    };

    graph.dragNodes = function dragNodes(nodes) {
        this.events.dragNodes.emit(nodes);
    };

    graph.setSpecGraph = function setSpecGraph(spec) {
        this.specGraph = spec;

        this.nodes.forEach((n) => {
            if (n?.subgraph !== undefined) {
                n.subgraph.setSpecGraph(spec);
            }
        });
    };

    graph.addGroup = function addGroup(name, color, nodeIds, id = uuidv4()) {
        this.groups.push({
            id,
            color,
            name,
            nodes: nodeIds,
        });
        this.events.addGroup.emit(this.groups[this.groups.length - 1]);
        return this.groups[this.groups.length - 1];
    };
    graph.removeGroup = function removeGroup(group) {
        if (!this.groups.includes(group)) {
            return;
        }
        this.events.removeGroup.emit(group);
        this.groups.splice(this.groups.indexOf(group), 1);
    };
    graph.ungroupNode = function ungroupNode(node) {
        const associated = this.groups?.filter((g) => g.nodes.includes(node.id));
        const groupsToRemove = [];
        associated.forEach((g) => {
            const index = g.nodes.indexOf(node.id);
            if (g.nodes.length > 2) {
                this.events.editGroup.emit(g);
                g.nodes.splice(index, 1);
            } else {
                groupsToRemove.push(g);
            }
        });
        groupsToRemove.forEach((g) => this.removeGroup(g));
    };

    graph.refreshConnections = function refreshConnections() {
        const { viewModel } = useViewModel();
        // get connections from selected nodes
        const connectionsToUpdate = this.connections.filter(
            (conn) => this.selectedNodes.some((n) => n.id === conn.from.nodeId
                || n.id === conn.to.nodeId),
        );

        connectionsToUpdate.forEach((conn) => {
            viewModel.value.connectionRenderer.connectionRefresh(
                conn,
            );
        });
    };

    graph.checkConnection = function checkConnection(from, to) {
        if (!from || !to) {
            return { connectionAllowed: false, error: 'Invalid from and to references.' };
        }

        const fromNode = this.findNodeById(from.nodeId);
        const toNode = this.findNodeById(to.nodeId);

        if (fromNode && toNode && fromNode === toNode && !this.editor.allowLoopbacks) {
            return { connectionAllowed: false, error: 'Loopbacks are not allowed.' };
        }

        // reverse connection so that 'from' is input and 'to' is output
        if (
            (from.direction === 'input' && to.direction === 'output') ||
            (from.direction === 'input' && to.direction === 'inout') ||
            (from.direction === 'inout' && to.direction === 'output')
        ) {
            const tmp = from;
            from = to;
            to = tmp;
        }

        if (from.isInput && from.direction !== 'inout') {
            return {
                connectionAllowed: false,
                error: 'Connections are only allowed from output or inout interfaces.',
            };
        }

        if (!to.isInput) {
            return {
                connectionAllowed: false,
                error: 'Connections are only allowed to input or inout interfaces.',
            };
        }

        if (this.connections.some((c) => c.from === from && c.to === to)) {
            return { connectionAllowed: false, error: 'Duplicate connections are not allowed.' };
        }

        if (from.maxConnectionsCount > 0 && from.connectionCount + 1 > from.maxConnectionsCount) {
            return {
                connectionAllowed: false,
                error: `Too many connections from an input interface '${from.id}', maximum of '${from.maxConnectionsCount}' are allowed.`,
            };
        }

        if (
            (to.maxConnectionsCount === 0 || to.maxConnectionsCount === undefined) &&
            to.connectionCount > 0
        ) {
            return {
                connectionAllowed: false,
                error: `By default only one connection to an input interface '${to.id}' allowed.`,
            };
        }

        if (to.maxConnectionsCount > 0 && to.connectionCount + 1 > to.maxConnectionsCount) {
            return {
                connectionAllowed: false,
                error: `Too many connections to an output interface '${to.id}', maximum of '${to.maxConnectionsCount}' are allowed.`,
            };
        }

        if (from.type && to.type) {
            const fromTypes =
                typeof from.type === 'string' || from.type instanceof String
                    ? [from.type]
                    : from.type;
            const toTypes =
                typeof to.type === 'string' || to.type instanceof String ? [to.type] : to.type;

            const commonType = fromTypes.find((t) => toTypes.includes(t));

            if (commonType === undefined) {
                return {
                    connectionAllowed: false,
                    error: `No common types between interfaces. Interface '${from.id}' supports types '${fromTypes}' and interface '${to.id}' supports types '${toTypes}'.`,
                };
            }
        }

        if (this.events.checkConnection.emit({ from, to }).prevented) {
            return {
                connectionAllowed: false,
                error: `Connection between an input interface '${from.id}' and an output interface '${to.id}' was prevented`,
            };
        }

        const hookResults = this.hooks.checkConnection.execute({ from, to });
        if (hookResults.some((hr) => !hr.connectionAllowed)) {
            return {
                connectionAllowed: false,
                errors: `Connection between an input interface '${from.id}' and an output interface '${to.id}' was prevented`,
            };
        }

        // List of connections that are removed once the dummyConnection is created
        const connectionsInDanger = [];
        return {
            connectionAllowed: true,
            dummyConnection: new DummyConnection(from, to),
            connectionsInDanger,
        };
    };

    /**
     * Adds an anchor to the connection and creates an event that
     * the anchor was added, which is used by history system.
     *
     * @param anchor anchor to be added
     * @param connection connection to which the anchor is added
     * @param position position of the anchor in the connection
     */
    graph.addAnchor = function addAnchor(anchor, connection, position, legacy = undefined) {
        const anchorToAdd = {
            x: anchor.x,
            y: anchor.y,
            index: position,
            id: uuidv4(),
            legacy,
        };
        if (connection.anchors === undefined) connection.anchors = [];

        connection.anchors.push(anchorToAdd);
        graph.events.addAnchor.emit([connection, connection.anchors.length - 1]);
        return connection.anchors[connection.anchors.length - 1];
    };

    const createStub = (intf, offset = undefined, stubID = undefined, stubSide = undefined) => {
        if (intf.bus?.type !== 'twoSided') {
            stubSide = undefined;
        }
        const maxConnReached = (intf.maxConnectionsCount &&
            intf.maxConnectionsCount < intf.connectionCount + 1);
        const stub = {};
        stub.nodeId = intf.nodeId;
        stub.id = stubID ?? uuidv4();
        stub.direction = intf.direction;
        stub.isInput = intf.isInput;
        stub.side = stubSide ?? intf.side;
        stub.maxConnectionsCount = 1;
        stub.connectionCount = maxConnReached ? 1 : 0;
        stub.type = intf.type;
        stub.sidePosition = 0;
        stub.offset = offset ?? ((intf.bus?.size ?? 0) / 2);
        stub.parent = intf;
        return stub;
    };

    // Replaces given instance of a node with a node of type `newNodeName`
    // All properties that are common preserve their values
    // All connections that were connected to the interfaces that are common
    // for those two nodes are preserved as well.
    graph.replaceNode = function replaceNode(oldNode, newNodeName, copyInterfaces = false) {
        const oldPosition = oldNode.position;
        const newNode = this.editor.nodeTypes.get(newNodeName);
        const newNodeInstance = new newNode.type(); // eslint-disable-line new-cap

        // Restoring a custom title of the node
        if (oldNode.title !== oldNode.type) {
            newNodeInstance.title = oldNode.title;
        }

        // Restoring width and height
        newNodeInstance.width = oldNode.width;
        newNodeInstance.height = oldNode?.height;

        // Restoring properties and interfaces
        if (copyInterfaces) {
            Object.entries({ ...oldNode.inputs, ...oldNode.outputs }).forEach(([name, intf]) => {
                if (
                    !Object.prototype.hasOwnProperty.call(newNodeInstance.inputs, name) &&
                    !Object.prototype.hasOwnProperty.call(newNodeInstance.outputs, name)
                ) {
                    if (intf.direction === undefined) {
                        newNodeInstance.addInput(name, intf);
                    } else {
                        newNodeInstance.addInterface(intf.direction, name, intf);
                    }
                }
            });
        } else {
            // Reset interfaces positions
            Object.entries({ ...oldNode.inputs, ...oldNode.outputs }).forEach(([_name, intf]) => {
                intf.sidePosition = 0;
            });
            Object.entries({ ...oldNode.inputs, ...oldNode.outputs }).forEach(([name, intf]) => {
                if (intf.direction !== undefined) {
                    if (Object.prototype.hasOwnProperty.call(newNodeInstance.inputs, name)) {
                        updateInterfacePosition(
                            newNodeInstance,
                            newNodeInstance.inputs[name],
                            intf.side,
                            intf.sidePosition,
                            false,
                            false,
                        );
                        newNodeInstance.inputs[name]
                            .id = intf.id;
                        newNodeInstance.inputs[name]
                            .externalName = newNodeInstance
                                .inputs[name]?.externalName ?? intf?.externalName;
                        newNodeInstance.inputs[name]
                            .maxConnectionsCount = newNodeInstance
                                .inputs[name]?.maxConnectionsCount ?? intf?.maxConnectionsCount;
                    }
                    if (Object.prototype.hasOwnProperty.call(newNodeInstance.outputs, name)) {
                        updateInterfacePosition(
                            newNodeInstance,
                            newNodeInstance.outputs[name],
                            intf.side,
                            intf.sidePosition,
                            false,
                            false,
                        );
                        newNodeInstance.outputs[name]
                            .id = intf.id;
                        newNodeInstance.outputs[name]
                            .externalName = newNodeInstance
                                .outputs[name]?.externalName ?? intf?.externalName;
                        newNodeInstance.outputs[name]
                            .maxConnectionsCount = newNodeInstance
                                .outputs[name]?.maxConnectionsCount ?? intf?.maxConnectionsCount;
                    }
                    // If the new node has the same property as it could be overridden
                } else if (
                    Object.prototype.hasOwnProperty.call(newNodeInstance.inputs, name)
                ) {
                    if (newNodeInstance.inputs[name].componentName === intf.componentName) {
                        newNodeInstance.inputs[name].value = intf.value;
                    }
                    newNodeInstance.inputs[name]
                        .id = intf.id;
                    newNodeInstance.inputs[name]
                        .externalName = newNodeInstance
                            .inputs[name]?.externalName ?? intf?.externalName;
                }
            });
        }

        // Restoring connections
        const interfaces = [...Object.values(oldNode.inputs), ...Object.values(oldNode.outputs)];
        const connections = this.connections.filter(
            (c) => interfaces.includes(c.from) || interfaces.includes(c.to) ||
                interfaces.some((i) => i.bus?.stubs?.some((s) => [c.to, c.from].includes(s))),
        );

        connections.forEach((conn) => {
            this.removeConnection(conn, false);
        });

        const connectionsToRestore = [];

        Object.entries({ ...oldNode.inputs, ...oldNode.outputs }).forEach(([name, intf]) => {
            if (intf.direction === undefined) return;

            // Rewiring connections to new interfaces
            connections.forEach((conn) => {
                [newNodeInstance.inputs, newNodeInstance.outputs].forEach((arr) => {
                    if (Object.prototype.hasOwnProperty.call(arr, name)) {
                        const checkAndReadd = (ii, curi) => {
                            if (conn.from === ii || conn.to === ii) {
                                const isFrom = conn.from === ii;
                                const fromNode = isFrom ? curi : conn.from;
                                const toNode = isFrom ? conn.to : curi;

                                if (this.checkConnection(fromNode, toNode).connectionAllowed) {
                                    const newConn = new Connection(fromNode, toNode);
                                    newConn.anchors = conn.anchors;
                                    connectionsToRestore.push(newConn);
                                    return true;
                                }
                            }
                            return false;
                        };
                        checkAndReadd(intf, arr[name]);
                        if (intf.bus?.size && intf.bus?.stubs) {
                            arr[name].bus.stubs = arr[name].bus.stubs ?? [];
                            intf.bus.stubs.forEach((stub) => {
                                const scopy =
                                    createStub(arr[name], stub.offset, stub.id, stub.side);
                                arr[name].bus.stubs.push(scopy);
                                if (!checkAndReadd(stub, arr[name].bus.stubs.at(-1))) {
                                    arr[name].bus.stubs.pop();
                                }
                            });
                        }
                    }
                });
            });
        });

        newNodeInstance.position = oldPosition;

        startTransaction();

        const oldGroups = this.groups.filter((g) => g.nodes.includes(oldNode.id));
        oldGroups.forEach((g) => {
            g.nodes[g.nodes.findIndex((id) => id === oldNode.id)] = newNodeInstance.id;
        });
        this.removeNode(oldNode);
        this.addNode(newNodeInstance);

        connectionsToRestore.forEach((conn) => {
            this.internalAddConnection(conn);
        });

        commitTransaction();

        const color = this.editor.getNodeColor(oldNode);

        if (color) {
            this.editor.setNodeColor(newNodeInstance.id, color);
        }
        const res = this.nodes.find((n) => n.id === newNodeInstance.id);

        return res;
    };

    graph.addNode = function addNode(node, graphLoadingState, nodeId) {
        if (this.events.beforeAddNode.emit(node).prevented) {
            return;
        }
        this.nodeEvents.addTarget(node.events);
        this.nodeHooks.addTarget(node.hooks);
        node.registerGraph(this);

        if (node.template !== undefined) {
            const newState = JSON.parse(JSON.stringify(node.template.save()));
            newState.id ??= uuidv4();
            node.template = new GraphTemplate(newState, this.editor);
        }

        this._nodes.push(node);
        // when adding the node to the array, it will be made reactive by Vue.
        // However, our current reference is the non-reactive version.
        // Therefore, we need to get the reactive version from the array.
        node = this.nodes.find((n) => n.id === node.id);
        node.onPlaced(graphLoadingState, nodeId);
        node.events.propertyEdit.subscribe(this, (tuple) => {
            const curNode = this.nodes.find((n) => n.id === tuple[0].id);
            this.editNode(curNode);
        });
        this.events.addNode.emit(node);
        this._idToNode[node.id] = node;
        return node; // eslint-disable-line consistent-return
    };

    graph.getNode = function getNode(nodeId) {
        if (this._idToNode[nodeId] === undefined) {
            const node = this._nodes.find((n) => n.id === nodeId);
            if (node) {
                this._idToNode[node.id] = node;
            }
        }
        return this._idToNode[nodeId];
    };

    graph.destroy = function destroy() {
        // Needed as a workaround for in-built baklava graph handling.
        // For example when changing graphs using switchGraphs the destroy
        // mechanism has to be restricted.
        if (this.disableDestroy) {
            return;
        }
        for (let i = this.connections.length - 1; i >= 0; i -= 1) {
            this.removeConnection(this.connections[i]);
        }
        for (let i = this.nodes.length - 1; i >= 0; i -= 1) {
            this.removeNode(this.nodes[i], false);
        }
        if (this.editor) {
            this.editor.unregisterGraph?.(this);
        }
        if (this.hooks) {
            Object.values(this.hooks).forEach((hook) => {
                hook.listeners.forEach((l) => {
                    hook.unsubscribe(l);
                });
            });
        }
        Object.values(this.events).forEach((event) => {
            event.listeners.forEach((l) => {
                event.unsubscribe(l);
            });
        });
        Object.values(this.nodeEvents).forEach((event) => {
            event.listeners.forEach((l) => {
                event.unsubscribe(l);
            });
        });
        if (this.graphNode) {
            this.graphNode.subgraph = undefined;
        }
        this.graphNode = undefined;
        this.editor = undefined;
        this.groups = undefined;
        this.nodes.length = 0;
        this.connections.length = 0;
        if (this.selectedNodes) {
            this.selectedNodes.length = 0;
        }
        this._idToNode = {};
        this._nodes = [];
    };
    graph.removeConnection = function removeConnection(connection, deleteOrphanStubs = true) {
        if (!this.connections.includes(connection)) {
            return;
        }
        if (this.events.beforeRemoveConnection.emit(connection).prevented) {
            return;
        }
        // Handling bus endpoints
        Object.values([connection.to, connection.from]).forEach((intf) => {
            if (intf.parent) {
                const p = intf.parent;
                if (deleteOrphanStubs) {
                    p.bus.stubs.splice(p.bus.stubs.indexOf(intf), 1);
                } else {
                    intf.isOrphaned = true;
                }
            }
        });
        connection.destruct();
        this._connections.splice(this.connections.indexOf(connection), 1);
        this.events.removeConnection.emit(connection);
        this.connectionEvents.removeTarget(connection.events);
    };

    // eslint-disable-next-line no-unused-vars
    graph.addConnection = function addConnection(from, to, offset, stubID, stubSide) {
        if (!to) return undefined;
        if (!from) return undefined;
        // the target interface is a bus, we have to create a connection point
        [to, from] = [to, from].map((intf) => {
            if (!intf.bus?.type) {
                return intf;
            }
            if (intf.bus.stubs === undefined) {
                intf.bus.stubs = [];
            }
            const stub = createStub(intf, offset, stubID, stubSide);

            return stub;
        });
        const checkConnectionResult = this.checkConnection(from, to);
        if (!checkConnectionResult.connectionAllowed) {
            return undefined;
        }
        // if allowed, push back stub
        [to, from].forEach((intf) => {
            if (intf.parent) {
                intf.parent.connectionCount += 1;
                intf.parent.bus.stubs.push(intf);
            }
        });

        if (this.events.beforeAddConnection.emit({ from, to }).prevented) {
            return undefined;
        }

        checkConnectionResult.connectionsInDanger.forEach((connectionToRemove) => {
            const instance = this.connections.find(
                (conn) => conn.id === connectionToRemove.id,
            );
            if (instance) {
                this.removeConnection(instance);
            }
        });

        const conn = new Connection(
            toRaw(checkConnectionResult.dummyConnection.from),
            toRaw(checkConnectionResult.dummyConnection.to),
        );
        this.internalAddConnection(conn);
        return conn;
    };
    graph.findNodeInterface = function findNodeInterface(id) {
        let result;
        const findStub = (intf, id_) => {
            if (!intf.bus?.size) {
                return undefined;
            }
            return intf.bus.stubs?.find((stub) => stub.id === id_);
        };
        const findId = (arr) => Object.keys(arr)?.forEach((k) => {
            const nodeInput = arr[k];
            if (nodeInput.id === id) {
                result = nodeInput;
            } else {
                const s = findStub(arr[k], id);
                if (s) {
                    result = s;
                }
            }
        });
        this.nodes.forEach((node) => {
            if (result) return;
            findId(node.inputs);
            if (result) return;
            findId(node.outputs);
        });
        return result;
    };
    graph.load = function load(state, loadAll = false) {
        const errors = [];

        // Clear current state
        for (let i = this.connections.length - 1; i >= 0; i -= 1) {
            this.removeConnection(this.connections[i]);
        }
        for (let i = this.nodes.length - 1; i >= 0; i -= 1) {
            this.removeNode(this.nodes[i], false);
        }

        // Load state
        this.id = state.id ?? uuidv4();
        this.name = state.name ?? undefined;

        const specOrGenerated = (mapping, id) => {
            if (!mapping) return id;
            return mapping.has(id) ? `'${mapping.get(id)}'` : `'${id}' (generated)`;
        };

        const nodesErrors = [];
        const connectedIds = state.connections.flatMap((c) => [c.to, c.from]);
        state.nodes.forEach((n) => {
            const nodeInformation = this.editor.nodeTypes.get(n.name);

            if (!nodeInformation) {
                nodesErrors.push(`Node type ${n.name} is not registered`);
            } else {
                const node = new nodeInformation.type(); // eslint-disable-line new-cap

                // The node state may not have an id, so we it has to be assigned manually
                // if needed
                n.id ??= uuidv4();
                // eslint-disable-next-line prefer-destructuring
                let graphLoadingState = state.graphLoadingState;
                if (!graphLoadingState) {
                    const interfacesEntries = n.interfaces
                        .filter((intf) => intf.id)
                        .map((intf) => [intf.name, intf.id]);
                    graphLoadingState = {
                        newToSpecNodeIds: new Map([[n.id, n.id]]),
                        newInterfaceIds: new Map([[n.id, new Map(interfacesEntries)]]),
                        newToSpecConnIds: new Map(),
                        newToSpecIntfIds: new Map(),
                    };
                }
                graphLoadingState.loadAll = loadAll;

                if (n.subgraph && node.template) {
                    // Remove new graphs except current one since they become dangling after
                    // CustomGraphNode.load
                    const graphsBefore = new Set([...this.editor.graphs].map(toRaw));

                    this.addNode(node, graphLoadingState, n.id);

                    if (!loadAll) {
                        [...this.editor.graphs]
                            .map(toRaw)
                            .filter((g) => !graphsBefore.has(g))
                            .filter((g) => g !== node.subgraph)
                            .forEach((g) => {
                                this.editor.unregisterGraph(g);
                                g.destroy?.();
                            });
                    }
                } else {
                    this.addNode(node, state.graphLoadingState, n.id);
                }
                const nodeErrors = node.load(n);
                if (Array.isArray(nodeErrors) && nodeErrors.length) {
                    nodesErrors.push(`Node '${node.type}' with id ${specOrGenerated(state.graphLoadingState?.newToSpecNodeIds, n.id)} is invalid:`, ...nodeErrors.map((e) => `    ${e}`));
                }
                // check for bus stubs not connected to anything
                n.interfaces?.filter((i) => i.bus).forEach((intf) => {
                    intf.bus?.stubs?.forEach((stub) => {
                        if (!connectedIds.includes(stub.id)) {
                            errors.push(`Interface bus stub ${stub.id} in interface ${intf.id} is not connected to anything`);
                        }
                    });
                });
            }
        });
        if (nodesErrors.length) errors.push('Node errors:', ...nodesErrors.map((e) => `    ${e}`));

        // Assigning ids to connections that do not have them
        state.connections.forEach((c) => {
            c.id ??= uuidv4();
        });

        const connectionsErrors = [];
        state.connections.forEach((c) => {
            const connectionErrors = [];
            const fromIf = this.findNodeInterface(c.from);
            const toIf = this.findNodeInterface(c.to);
            const otherExists = state.connections.some(
                (conn) => conn.id === c.id && (conn.from !== c.from || conn.to !== c.to),
            );

            if (otherExists) connectionErrors.push(`ID is already taken.`);

            const reportIfMissing = (direction, intf, id) => {
                if (intf) return;
                connectionErrors.push(`Missing ${direction} ${specOrGenerated(state.graphLoadingState?.newToSpecIntfIds, id)}`);
            };

            reportIfMissing('src', fromIf, c.from);
            reportIfMissing('dst', toIf, c.to);

            if (connectionErrors.length) {
                const reportIfFound = (direction, intf) => {
                    if (!intf) return;
                    const node = this.nodes.find(({ id }) => id === intf.nodeId);
                    const intfIdLabel =
                        specOrGenerated(state.graphLoadingState?.newToSpecIntfIds, intf.id);
                    const nodeIdLabel =
                        specOrGenerated(state.graphLoadingState?.newToSpecNodeIds, node.id);
                    connectionErrors.push(`Found   ${direction} ${intfIdLabel} '${intf.name}' in node '${node.type}' with id ${nodeIdLabel}`);
                };

                reportIfFound('src', fromIf);
                reportIfFound('dst', toIf);

                const connIdLabel =
                    specOrGenerated(state.graphLoadingState?.newToSpecConnIds, c.id);
                connectionsErrors.push(`Connection ${connIdLabel} is invalid:`, ...connectionErrors.map((e) => `    ${e}`));
            } else {
                // Manually adding connections instead of using `addConnection` from baklavajs
                // as we want to get a feedback message from `checkConnection` function
                // which is suppressed in baklavajs functionality
                const checkConnectionResult = this.checkConnection(fromIf, toIf);
                if (!checkConnectionResult.connectionAllowed) {
                    errors.push(
                        `Could not create connection of id '${c.id}'. ${checkConnectionResult.error}`,
                    );
                } else {
                    checkConnectionResult.connectionsInDanger.forEach((connectionToRemove) => {
                        const instance = this.connections.find(
                            (conn) => conn.id === connectionToRemove.id,
                        );
                        if (instance) {
                            this.removeConnection(instance);
                        }
                    });

                    const conn = new Connection(
                        checkConnectionResult.dummyConnection.from,
                        checkConnectionResult.dummyConnection.to,
                    );

                    if (c.anchors?.some((anchor) => !anchor.index)) {
                        // legacy anchors
                        c.anchors?.forEach((anchor, index) => {
                            graph.addAnchor(anchor, conn, index, true);
                        });
                    } else {
                        c.anchors?.forEach((anchor) => {
                            graph.addAnchor(anchor, conn, anchor.index);
                        });
                    }
                    this.internalAddConnection(conn);
                }
            }
        });

        if (connectionsErrors.length) errors.push('Connection errors:', ...connectionsErrors.map((e) => `    ${e}`));

        this.hooks.load.execute(state);
        return errors;
    };

    graph.size = function size() {
        const ignoredNodeSet = this.editor.getIgnoredNodes(this.id);

        const nodes = this.nodes.filter((n) => !ignoredNodeSet.has(n.layer));

        const sizes = nodes.map((node) => {
            const HTMLelement = document.getElementById(node.id);
            return {
                width: HTMLelement.offsetWidth,
                height: HTMLelement.offsetHeight,
                position: this.editor.getNodeDisplayedPosition(node),
            };
        });
        const margin = 100;

        const rightmostX = Math.max(...sizes.map((node) => node.position.x + node.width)) + margin;
        const leftmostX = Math.min(...sizes.map((node) => node.position.x)) - margin;

        const bottommostY =
            Math.max(...sizes.map((node) => node.position.y + node.height)) + margin;
        const topmostY = Math.min(...sizes.map((node) => node.position.y)) - margin;

        const graphWidth = rightmostX - leftmostX;
        const graphHeight = bottommostY - topmostY;

        return {
            graphHeight,
            graphWidth,
            rightmostX,
            leftmostX,
            bottommostY,
            topmostY,
        };
    };

    graph.save = function save() {
        const state = {
            id: this.id,
            name: this.name !== '' ? this.name : undefined,
            nodes: this.nodes.map((n) => n.save()),
            groups: this.groups,
            connections: this.connections.map((c) => ({
                id: c.id,
                from: c.from.id,
                to: c.to.id,
                // eslint-disable-next-line no-unused-vars
                anchors: c.anchors?.map((a) => ({ x: a.x, y: a.y, index: a.index })),
            })),
            scaling: this.scaling,
            panning: this.panning,
        };
        return this.hooks.save.execute(state);
    };

    graph.removeSelectedNodes = function removeSelectedNodes() {
        for (let i = this.selectedNodes.length - 1; i >= 0; i -= 1) {
            this.removeNode(this.selectedNodes[i]);
        }
    };

    graph.removeNode = function removeNode(node, updateInterfaces = true) {
        if (this.nodes.includes(node)) {
            if (this.events.beforeRemoveNode.emit(node).prevented) {
                return;
            }
            const interfaces = [...Object.values(node.inputs), ...Object.values(node.outputs)];
            this.connections
                .filter((c) =>
                    interfaces.includes(c.from) || interfaces.includes(c.to) ||
                        interfaces.some((i) =>
                            i.bus?.stubs?.some((s) => [c.to, c.from].includes(s))),
                )
                .forEach((c) => this.removeConnection(c));
            this.ungroupNode(node);
            this._nodes.splice(this.nodes.indexOf(node), 1);
            this.events.removeNode.emit(node);
            node.events.propertyEdit.unsubscribe(this);
            node.onDestroy();
            this.nodeEvents.removeTarget(node.events);
            this.nodeHooks.removeTarget(node.hooks);
            if (this.graphNode !== undefined && updateInterfaces) {
                const removedInterfaces =
                    [...Object.values(node.inputs), ...Object.values(node.outputs)];
                const keptInputs = Object.values(this.graphNode.inputs)
                    .filter((intf) => !removedInterfaces.some((r) => r.externalName === intf.name));
                const keptOutputs = Object.values(this.graphNode.outputs)
                    .filter((intf) => !removedInterfaces.some((r) => r.externalName === intf.name));
                this.graphNode.updateGraphNodeInterfaces(
                    keptOutputs ?? [], keptInputs ?? [], true);
            }

            if (node.subgraph) {
                node.subgraph.destroy();
            }
            if (this._idToNode[node.id]) {
                delete this._idToNode[node.id];
            }
        }
    };

    graph.removeNodeOnly = function removeNodeOnly(node) {
        this._nodes.splice(this.nodes.indexOf(node), 1);
        this.events.removeNode.emit(node);
        node.events.propertyEdit.unsubscribe(this);
        node.onDestroy();
        this.nodeEvents.removeTarget(node);
        this.nodeHooks.removeTarget(node);
    };
    graph.removeAnchor = function removeAnchor(connection, index) {
        if (connection.anchors === undefined) connection.anchors = [];
        graph.events.removeAnchor.emit([connection, index]);
        connection.anchors.splice(index, 1);
        connection.anchors.slice(index).forEach((a) => { a.index = -1; });
    };

    graph.editAnchor = function editAnchor(connection, index, position, prevPos) {
        if (connection.anchors === undefined) connection.anchors = [];
        graph.events.editAnchor.emit([connection, index, position, prevPos]);
    };
    graph.editNode = function editNode(node) {
        this.events.editNode.emit(node);
    };

    graph.obtainExposedNames = function obtainExposedNames() {
        const exposedNames = [];
        this._nodes.forEach((node) => {
            Object.values({ ...node.inputs, ...node.outputs }).forEach((intf) => {
                if (intf.externalName !== undefined) {
                    exposedNames.push(intf.externalName);
                }
            });
        });
        return exposedNames;
    };

    graph.isIncorrectExternalName = function isIncorrectExternalName(name, exposedNames) {
        if (this.graphNode === undefined) return false;

        const sameExposedNames = exposedNames.filter((n) => n === name).length;

        // Extract interface names from graph node
        const interfaces = [
            ...Object.keys(this.graphNode.inputs),
            ...Object.keys(this.graphNode.outputs),
        ].map((intf) => intf.split('_'))
            .filter((intf) => intf[0] !== 'property')
            .map((intf) => intf[1]);

        const sameInterfaceNames = interfaces.filter((n) => n === name).length;
        return name.length === 0 || sameExposedNames !== 0 || sameInterfaceNames !== 0;
    };

    graph.resolveNewExposedName = function resolveNewExposedName(name) {
        const exposedNames = this.obtainExposedNames();

        // Check if the external name is taken and add a suffix if it is
        let suffix = 1;
        let tmpName = name;
        while (this.isIncorrectExternalName(tmpName, exposedNames)) {
            tmpName = `${name}_${suffix}`;
            suffix += 1;
        }
        return tmpName;
    };

    return graph;
}
