/*
 * Copyright (c) 2024 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

// TODO: Define NodeInterface type
type NodeInterface = any; // eslint-disable-line @typescript-eslint/no-explicit-any

interface registeredInterface {
    sharedInterface: NodeInterface,
    sharedInterfaceGraphId: string,
    /**
     * List of ids of graphNodes that use the exposed interfaces.
     * The order of the ids determines the order of the graph nodes on a path,
     * from the oldest to the newest.
     */
    graphIds: string[],
}

export class InterfaceRegistry {
    private exposedInterfaces = new Map<string, registeredInterface>();

    /**
     * Clears the registry.
     */
    clearRegistry() {
        this.exposedInterfaces.clear();
    }

    /**
     * Stores graph id of a shared interface in the registry, so that when
     * privatizing an interface we can remove it from all the graphs that use it.
     * If the interface is not registered, an error is thrown.
     *
     * @param {string} intfId Id of the interface.
     * @param {string} graphId Id of a graph that has the registered interface
     */
    pushGraphIdToRegistry(intfId: string, graphId: string) {
        const sharedInterface = this.exposedInterfaces.get(intfId);
        if (!sharedInterface) {
            throw new Error(`Interface with id ${intfId} not found in the register.`);
        }

        sharedInterface.graphIds.push(graphId);
    }

    /**
     * Seamlessly wraps passed `intf`, so that the properties `maxConnectionsCount`,
     * `connectionCount` and `type` are shared between those two interfaces.
     *
     * @param {NodeInterface} intf Interface for which the wrapper is created.
     */
    createSharedInterface(intf: NodeInterface) {
        const sharedInterface = this.exposedInterfaces.get(intf.id);
        if (!sharedInterface) {
            throw new Error(`'Interface with id ${intf.id} not found in the register.`);
        }

        Object.defineProperty(intf, 'maxConnectionsCount', {
            get() {
                return sharedInterface.sharedInterface.maxConnectionsCount;
            },
            set(value: number) {
                sharedInterface.sharedInterface.maxConnectionsCount = value;
            },
        });

        Object.defineProperty(intf, 'connectionCount', {
            get() {
                return sharedInterface.sharedInterface.connectionCount;
            },
            set(value: number) {
                sharedInterface.sharedInterface.connectionCount = value;
            },
        });

        Object.defineProperty(intf, 'type', {
            get() {
                return sharedInterface.sharedInterface.type;
            },
        });
        // eslint-disable-next-line no-param-reassign
        intf.bus = intf.bus || {};

        Object.defineProperty(intf.bus, 'size', {
            get() {
                return sharedInterface.sharedInterface.bus?.size;
            },
            set(value: number) {
                sharedInterface.sharedInterface.bus ||= {};
                sharedInterface.sharedInterface.bus.size = value;
            },
        });

        Object.defineProperty(intf.bus, 'type', {
            get() {
                return sharedInterface.sharedInterface.bus?.type;
            },
        });
    }

    /**
     * Checks if the interface of given id is registered.
     *
     * @param {string} intfId Id of the interface to check.
     * @returns {boolean} True if the interface is registered, false otherwise.
     */
    isRegistered(intfId: string): boolean {
        return this.exposedInterfaces.has(intfId);
    }

    /**
     * Returns the registered interface of given id. If the interface is not registered,
     * an error is thrown.
     *
     * @param {string} intfId Id of the interface to get.
     */
    getRegisteredInterface(intfId: string) {
        if (!this.exposedInterfaces.has(intfId)) {
            throw new Error(`Interface of id ${intfId} is not registered.`);
        }
        return this.exposedInterfaces.get(intfId);
    }

    /**
     * Deletes the registered interface of given id. If the interface is not registered,
     * an error is thrown.
     *
     * @param {string} intfId Id of the interface to delete
     */
    deleteRegisteredInterface(intfId: string) {
        if (!this.exposedInterfaces.has(intfId)) {
            throw new Error(`Interface of id ${intfId} is not registered.`);
        }
        return this.exposedInterfaces.delete(intfId);
    }

    /**
     * Registers an interface in the registry, so that it can be shared between
     * multiple graph nodes. If the interface is already registered, an error is thrown.
     *
     * @param {NodeInterface} intf Interface to register.
     */
    registerInterface(intf: NodeInterface, graphId: string) {
        if (this.exposedInterfaces.has(intf.id)) {
            throw new Error(`Trying to register an interface of id ${intf.id}, but it is already registered.`);
        }

        this.exposedInterfaces.set(intf.id, {
            sharedInterface: intf,
            sharedInterfaceGraphId: graphId,
            graphIds: [],
        });
    }
}

export const ir = new InterfaceRegistry();
