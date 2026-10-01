/*
 * Copyright (c) 2022-2024 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* eslint-disable max-classes-per-file */

import { NodeInterface, Node } from '@baklavajs/core';
import { BaklavaEvent } from '@baklavajs/events';

import { useViewModel } from '@baklavajs/renderer-vue';
import { updateInterfacePosition } from '../custom/CustomNode.js';
import {
    applySidePositions,
    parseInterfaces,
    validateInterfaceGroups,
    generateProperties,
} from './interfaceParser.js';

import CheckboxInterface from '../interfaces/CheckboxInterface.js';
import HexInterface from '../interfaces/HexInterface.js';
import InputInterface from '../interfaces/InputInterface.js';
import IntegerInterface from '../interfaces/IntegerInterface.js';
import ListInterface from '../interfaces/ListInterface.js';
import NumberInterface from '../interfaces/NumberInterface.js';
import SelectInterface from '../interfaces/SelectInterface.js';
import SliderInterface from '../interfaces/SliderInterface.js';
import TextAreaInterface from '../interfaces/TextAreaInterface.js';
import TextInterface from '../interfaces/TextInterface.js';
import ButtonInterface from '../interfaces/ButtonInterface.js';
import GraphTemplate from '../custom/CustomGraphTemplate.js';

import globalProperties from '../globalProperties.ts';

import { ir } from './interfaceRegistry.ts';

import { terminalStore } from './stores.js';

/**
 * Test list values against dtype type and
 * returns array with elements that mismatch
 * with type
 *
 * @param {list} values - list of values in property list
 * @param {*} dtype - a type to test values element against
 * @returns List elements that are not of type dtype
 */
function validateList(values, dtype) {
    return values.filter((val) => {
        // Accept both 'integer' and 'number' for dtype 'integer'.
        if (dtype === 'integer') {
            return typeof val !== 'number' || !Number.isInteger(val);
        }
        return typeof val !== dtype;// eslint-disable-line valid-typeof
    });
}

/**
 * @param properties coming from the specification
 * @returns object that can be used to create properties or an array of errors
 * if any occurred.
 */
export function parseProperties(properties) {
    const parsedProperties = {};
    const usedNames = new Set();
    const errors = [];

    properties.forEach((prop) => {
        const newProp = prop;

        if (newProp.group !== undefined && Array.isArray(newProp.group)) {
            const parsedGroup = parseProperties(newProp.group);
            if (Array.isArray(parsedGroup) && parsedGroup.length) {
                errors.push(...parsedGroup);
            }

            Object.entries(parsedGroup).forEach(([pgroupname]) => {
                if (usedNames.has(pgroupname)) {
                    const realname = pgroupname.slice(pgroupname.indexOf('_') + 1);
                    errors.push(
                        `Property named '${realname}' in a group property '${newProp.name}' is a duplicate.`,
                    );
                }
                usedNames.add(pgroupname);
            });

            newProp.group = parsedGroup; // eslint-disable-line no-param-reassign
        }

        if (usedNames.has(`property_${newProp.name}`)) {
            errors.push(
                `Property named '${newProp.name}' is a duplicate.`,
            );
        }

        if (globalProperties.softLoad) {
            // validate property type
            if (newProp.type === 'select') {
                const items = newProp.values;
                const defaultValue = newProp.default;

                if (Array.isArray(items)) {
                    if (defaultValue !== 'constant' && !items.map(String).includes(String(defaultValue))) {
                        errors.push(
                            `Default value for '${newProp.name}' mismatch. Expected '${items}' (type: ${typeof items}), got '${defaultValue}' (type: ${typeof defaultValue}).`,
                        );

                        newProp.default = items.at(0);
                    }
                }
            } else if (newProp.type === 'list') {
                const { dtype } = newProp;
                const items = newProp.default;

                const mismatchedElements = validateList(items, dtype);

                if (mismatchedElements.length > 0) {
                    errors.push(
                        `Property '${newProp.name}' default value mismatch. ` +
                        `Items: '${mismatchedElements.join(' ')}' are not of '${dtype}' dtype.` +
                        `Items are of type: ${mismatchedElements.map((val) => typeof val).join(', ')} ` +
                        `Removing them from list.`,
                    );
                }

                const filteredItems = items.filter((val) =>
                    mismatchedElements.indexOf(val) === -1);

                newProp.default = filteredItems;
            }
        }

        parsedProperties[`property_${newProp.name}`] = { ...newProp };
        usedNames.add(`property_${newProp.name}`);
    });

    if (errors.length && !globalProperties.softLoad) {
        return errors;
    }
    if (errors.length) {
        terminalStore.addParsed('Error during parse properties', errors.toString());
    }

    return parsedProperties;
}

/**
 * Creates a property based on its specification.
 *
 * @param p property specification with name and type
 * @param hidden whether the property should be hidden
 *
 * @returns property object of a given type
 * */
export function newProperty(p, hidden = false) {
    const propName = p.name;
    const propType = p.type;
    let propDef = p.default;
    let intf;

    switch (propType) {
        case 'constant':
            intf = new TextInterface(propName, propDef);
            break;
        case 'text':
            intf = new InputInterface(propName, propDef, p.readonly);
            break;
        case 'multiline':
            intf = new TextAreaInterface(propName, propDef, p.readonly);
            break;
        case 'number':
            intf = new NumberInterface(propName, propDef, p.min, p.max, p.readonly);
            break;
        case 'integer':
            intf = new IntegerInterface(
                propName,
                propDef,
                p.min,
                p.max,
                p.readonly,
                p.dynamicCounter,
            );
            break;
        case 'hex':
            intf = new HexInterface(
                propName,
                propDef.toLowerCase(),
                p.min ? p.min : undefined,
                p.max ? p.max : undefined,
                p.readonly,
            );
            break;
        case 'select': {
            const it = p.values.map((element) => element.toString());
            intf = new SelectInterface(propName, propDef, it, p.readonly);
        } break;
        case 'bool':
            intf = new CheckboxInterface(propName, propDef, p.readonly);
            break;
        case 'slider':
            if (propDef === undefined) {
                propDef = p.min;
            }
            intf = new SliderInterface(propName, propDef, p.min, p.max, p.step, p.readonly);
            break;
        case 'list':
            if (propDef === null) {
                propDef = [];
            }
            intf = new ListInterface(propName, propDef, p.dtype, p.readonly);
            break;
        case 'button-url':
            intf = new ButtonInterface(propName, () => window.open(intf.value, '_blank'), propDef, 'button-url');
            break;
        case 'button-api':
            intf = new ButtonInterface(
                propName,
                () => intf.events.updated.emit(['button_click', { id: intf.id, value: intf.value }]),
                propDef,
                'button-api');
            break;
        case 'button-graph':
            intf = new ButtonInterface(
                propName,
                () => intf.events.updated.emit(intf.value),
                propDef,
                'button-graph');
            break;
        default:
            /* eslint-disable no-console */
            console.error(propType, ' input type is not recognized.');
    }
    if (intf !== undefined) {
        intf.hidden = hidden;
        intf.type = propType;
        intf.default = propDef;
        intf.hideOnDefault = p.hideOnDefault;
        if (p.inherited !== undefined) {
            intf.inherited = p.inherited;
        }

        if (p.interfaceMaxConnectionsCount !== undefined) {
            intf.interfaceMaxConnectionsCount = p.interfaceMaxConnectionsCount;
        }

        if (p.interfaceType !== undefined) {
            intf.interfaceType = p.interfaceType;
        }
    }

    return intf;
}

/**
 * @param properties that are validated and parsed. The format
 * should be the same as the one returned by parseProperties.
 * @returns created properties
 */
export function createProperties(properties) {
    const tempProperties = {};

    Object.entries(properties).forEach(([pname, p]) => {
        if (p.group !== undefined) {
            tempProperties[pname] = (() => {
                const groupedProperty = newProperty(p);
                groupedProperty.group = Object.keys(p.group);
                return groupedProperty;
            });
            Object.entries(p.group).forEach(([pgroupname, pgroup]) => {
                tempProperties[pgroupname] = () => {
                    const groupedProperty = newProperty(pgroup);
                    groupedProperty.groupProperty = tempProperties[pname];
                    return groupedProperty;
                };
            });
        } else {
            tempProperties[pname] = () => newProperty(p);
        }
    });
    return tempProperties;
}

/**
 * Function check if passed value is of a type, specified in propType.
 *
 * @param propType property type which value is going to be check against
 * @param value value which type is going to be checked
 * @returns true when types match, false otherwise
 */
function checkType(propType, value) {
    switch (propType) {
        case 'constant':
        case 'select':
            return true;
        case 'text':
        case 'multiline':
        case 'button-url':
        case 'button-graph':
        case 'hex':
            return typeof value === 'string';
        case 'number':
        case 'integer':
        case 'slider':
            return typeof value === 'number';
        case 'bool':
            return typeof value === 'boolean';
        case 'list':
            return Array.isArray(value);
        case 'button-api':
            return typeof value === 'object';
        default:
            return false;
    }
}

/**
 * Function performs sanity checking on parsed state before loading it
 * into the editor. It should throw explicit errors if any discrepancy is detected.
 *
 * @param parsedState that is passed to node to load
 * @param inputs inputs of the node
 * @param outputs outputs of the node
 * @returns list of errors.
 */
function detectDiscrepancies(parsedState, inputs, outputs) {
    let errors = [];

    // Checking for existence of interfaces defined
    Object.keys({
        ...parsedState.inputs,
        ...parsedState.outputs,
    }).forEach((ioName) => {
        const name = ioName.slice(ioName.indexOf('_') + 1);
        const direction = ioName.slice(0, ioName.indexOf('_'));
        if (
            !Object.prototype.hasOwnProperty.call(inputs, ioName) &&
            !Object.prototype.hasOwnProperty.call(outputs, ioName)
        ) {
            if (direction === 'property') {
                errors.push(`Property named '${name}' not found in specification!`);
            } else {
                errors.push(`Interface named '${name}' of direction '${direction}' not found in specification!`);
            }
        } else if (direction === 'property') {
            // Verifying property type defined in the node and the value passed
            const parsedValue = parsedState.inputs[ioName].value;
            const propertyType = inputs[ioName].type;
            if (!checkType(propertyType, parsedValue)) {
                errors.push(`Property '${name}' type mismatch. ${propertyType} expected, ${typeof parsedValue} found.`);
            } else if (propertyType === 'select') {
                const { items } = inputs[ioName];
                if (Array.isArray(items)) {
                    if (!items.map(String).includes(String(parsedValue))) {
                        errors.push(
                            `Property '${name}' value mismatch. ${parsedValue} (type: ${typeof parsedValue}) not found in ${JSON.stringify(items)}`,
                        );
                    }
                } else if (String(parsedValue) !== String(items)) {
                    errors.push(
                        `Property '${name}' value mismatch. Expected '${items}' (type: ${typeof items}), found '${parsedValue}' (type: ${typeof parsedValue}).`,
                    );
                }
            } else if (propertyType === 'list') {
                const { dtype } = inputs[ioName];

                const mismatchedElements = validateList(parsedValue, dtype);

                if (mismatchedElements.length > 0) {
                    errors.push(
                        `Property '${name}' value mismatch. ` +
                        `Items: '${mismatchedElements.join(' ')}' are not of '${dtype}' dtype.` +
                        `Items are of type: ${mismatchedElements.map((val) => typeof val).join(', ')}`,
                    );
                }
            }
        }
    });

    // Checking for existence of interface groups
    Object.keys(parsedState.enabledInterfaceGroups).forEach((groupName) => {
        if (
            !Object.prototype.hasOwnProperty.call(inputs, groupName) &&
            !Object.prototype.hasOwnProperty.call(outputs, groupName)
        ) {
            const direction = groupName.slice(0, groupName.indexOf('_'));
            const name = groupName.slice(groupName.indexOf('_') + 1);

            errors.push(`Interface group named '${name}' of direction '${direction}' not found in specification!`);
        }
    });

    if (errors && errors.length) {
        return errors;
    }

    errors = validateInterfaceGroups(
        Object.keys(parsedState.enabledInterfaceGroups),
        inputs,
        outputs,
    );

    return errors;
}

/**
 * Function performs sanity checking on parsed state before loading it
 * into the editor and the tries to resolve found discrepancies.
 *
 * @param parsedState that is passed to node to load
 * @param inputs inputs of the node
 * @param outputs outputs of the node
 * @returns parsed state.
 */
function resolveDiscrepancies(parsedState, inputs, outputs) {
    const errors = [];

    const InterfacesToRemove = [];
    const PropertiesToRemove = [];

    const newState = parsedState;

    const checkProperty = (ioName) => {
        const name = ioName.slice(ioName.indexOf('_') + 1);

        const parsedValue = parsedState.inputs[ioName].value;
        const propertyType = inputs[ioName].type;
        const defaultValue = inputs[ioName].default;

        if (propertyType === 'select') {
            const { items } = inputs[ioName];
            if (Array.isArray(items)) {
                if (!items.map(String).includes(String(parsedValue))) {
                    errors.push(
                        `Value for '${name}' mismatch. Expected '${items}' (type: ${typeof items}), got '${parsedValue}' (type: ${typeof parsedValue}),falling back to default value`,
                    );

                    newState.inputs[ioName].value = defaultValue;
                }
            }
        } else if (propertyType === 'list') {
            const { dtype } = inputs[ioName];

            const mismatchedElements = validateList(parsedValue, dtype);

            if (mismatchedElements.length > 0) {
                errors.push(
                    `Property '${name}' value mismatch. ` +
                    `Items: '${mismatchedElements.join(' ')}' are not of '${dtype}' dtype.` +
                    `Items are of type: ${mismatchedElements.map((val) => typeof val).join(', ')} ` +
                    `Removing them from list.`,
                );
            }

            const filteredValue = parsedValue.filter((val) =>
                mismatchedElements.indexOf(val) === -1);

            newState.inputs[ioName].value = filteredValue;
        } else if (!checkType(propertyType, parsedValue)) {
            errors.push(`Property '${name}' type mismatch. ${propertyType} expected, ${typeof parsedValue} found, falling back to default value.`);
            newState.inputs[ioName].value = defaultValue;
        }
    };

    // Checking for existence of interfaces defined
    Object.keys({
        ...parsedState.inputs,
        ...parsedState.outputs,
    }).forEach((ioName) => {
        const name = ioName.slice(ioName.indexOf('_') + 1);
        const direction = ioName.slice(0, ioName.indexOf('_'));
        if (
            !Object.prototype.hasOwnProperty.call(inputs, ioName) &&
            !Object.prototype.hasOwnProperty.call(outputs, ioName)
        ) {
            if (direction === 'property') {
                PropertiesToRemove.push(ioName);
                errors.push(`Property named '${name}' not found in specification!`);
            } else {
                InterfacesToRemove.push(ioName);
                errors.push(`Interface named '${name}' of direction '${direction}' not found in specification!`);
            }
        } else if (direction === 'property') {
            checkProperty(ioName);
        }
    });

    const stateInputs = Object.entries(parsedState.inputs);
    const stateOutputs = Object.entries(parsedState.outputs);

    const parsedInputs = stateInputs.filter(([key, _]) =>
        InterfacesToRemove.indexOf(key) === -1 && PropertiesToRemove.indexOf(key) === -1);

    const parsedOutputs = stateOutputs.filter(([key, _]) =>
        InterfacesToRemove.indexOf(key) === -1);

    newState.inputs = Object.fromEntries(parsedInputs);
    newState.outputs = Object.fromEntries(parsedOutputs);

    if (errors.length) {
        terminalStore.addParsed('Error during discrepancies resolvement', errors.toString());
    }

    return newState;
}

/**
 * @param state state to be loaded. Should be a valid dataflow
 * @returns state that can be given to baklavajs, or an array of errors if any occurred
 */
function parseNodeState(state) {
    const newState = JSON.parse(JSON.stringify(state));

    if (newState.interfaces !== undefined) {
        const out = parseInterfaces(newState.interfaces, [], []);
        if (Array.isArray(out) && out.length) {
            return out;
        }

        const { inputs, outputs } = out;
        newState.inputs = inputs;
        newState.outputs = outputs;

        delete newState.interfaces;
    }

    if (newState.properties !== undefined) {
        const out = parseProperties(newState.properties);
        if (Array.isArray(out) && out.length) {
            return out;
        }

        newState.inputs = { ...newState.inputs, ...out };
        delete newState.properties;
    }

    if (newState.enabledInterfaceGroups !== undefined) {
        const interfaceGroups = {};
        newState.enabledInterfaceGroups.forEach((intf) => {
            interfaceGroups[`${intf.direction}_${intf.name}`] = { ...intf };
        });
        newState.enabledInterfaceGroups = interfaceGroups;
    } else {
        newState.enabledInterfaceGroups = {};
    }

    newState.title = newState.instanceName ?? '';
    delete newState.instanceName;

    newState.parsed = true;
    return newState;
}

export class CustomNode extends Node {
    inputs = {};

    outputs = {};

    type = undefined;

    views = [];

    constructor(
        name,
        layer,
        inputs,
        outputs,
        twoColumn,
        description = '',
        nodeExtends = [],
        nodeExtending = [],
        nodeSiblings = [],
        width = 0,
        simpleInherited = [],
        relatedGraphs = undefined,
        height = undefined,
    ) {
        super();

        this.description = description;
        this.extends = nodeExtends;
        this.extending = nodeExtending;
        this.siblings = nodeSiblings;
        this.layer = layer;
        this.title = name;
        this.twoColumn = twoColumn;
        this.type = name;
        this.width = width;
        this.height = height;
        this.relatedGraphs = relatedGraphs;
        this.simpleInherited = simpleInherited;
        this.views = [];
        this.internalPosition = undefined;

        this.events.propertyEdit = new BaklavaEvent();

        Object.keys(inputs).forEach((k) => {
            const intf = inputs[k]();
            this.addInput(k, intf);
        });

        Object.keys(outputs).forEach((k) => {
            const intf = outputs[k]();
            this.addOutput(k, intf);
        });
    }

    set position(pos) {
        try {
            const { viewModel } = useViewModel();
            const { editor } = viewModel.value;
            const curName = editor.currentView;
            const entry = this.views?.find((v) => v.name === curName);

            if (entry === undefined) {
                this.internalPosition = pos;
            } else {
                entry.internalPosition = pos;
            }
        } catch {
            this.internalPosition = pos;
        }
    }

    get position() {
        try {
            const { viewModel } = useViewModel();
            const { editor } = viewModel.value;
            const curName = editor.currentView;
            const entry = this.views?.find((v) => v.name === curName);
            return entry ? entry.position : this.internalPosition;
        } catch {
            return this.internalPosition;
        }
    }

    /**
     * Function for updating dynamic interfaces of the node based on the property passed.
     *
     * @param {*} prop property that is responsible for creating dynamic interfaces.
     * It must have name and value properties.
     */
    updateDynamicInterfaces(prop) {
        const interfaces = [];
        const { value } = prop;

        // Ensure prop.name is defined and is a string.
        if (typeof prop.name !== 'string') {
            throw new Error('Property \'name\' is undefined or not a string in updateDynamicInterfaces.');
        }

        // The interface metadata has to be obtained from the specification of the property
        const propertyInput = this.inputs[`property_${prop.name}`];
        const interfaceType = propertyInput?.interfaceType;
        const interfaceMaxConnectionsCount = propertyInput?.interfaceMaxConnectionsCount;

        // Direction is obtained from property name
        // Format: {name} {direction}
        const propName = prop.name.split(' ');

        // Ensure propName has enough elements
        if (propName.length < 2) {
            throw new Error(
                'Property name is too short to extract direction in updateDynamicInterfaces.' +
                `Property name: ${prop.name}`,
            );
        }

        const direction = propName.pop();
        const interfaceName = propName.join(' ');

        const occupied = {
            left: [], right: [], top: [], bottom: [],
        };

        const stateIOs = { ...this.inputs, ...this.outputs };

        // Assigning sides and sides Positions to interfaces
        Object.entries(stateIOs).forEach(([ioName, ioState]) => {
            if (ioName.startsWith('property_')) return;
            occupied[ioState.side].push(ioState.sidePosition);
        });

        for (let i = 0; i < value; i += 1) {
            const ioName = `${interfaceName}[${i}]`;
            const directionIoName = `${direction}_${ioName}`;

            const intf = {
                name: ioName,
                direction,
            };

            const container = direction === 'output' ? this.outputs : this.inputs;

            if (directionIoName in container) {
                intf.externalName = container[directionIoName].externalName;
                intf.side = container[directionIoName].side;
                intf.sidePosition = container[directionIoName].sidePosition;
            }

            if (
                !Object.prototype.hasOwnProperty.call(intf, 'sidePosition') &&
                !Object.prototype.hasOwnProperty.call(intf, 'side')
            ) {
                const side = direction === 'output' ? 'right' : 'left';
                let firstUnoccupied = occupied[side].sort((a, b) => a - b).findIndex(
                    (sidePosition, index) => sidePosition !== index,
                );

                if (firstUnoccupied === -1) {
                    if (occupied[side].length === 0) {
                        firstUnoccupied = 0;
                    } else {
                        firstUnoccupied = Math.max(...occupied[side]) + 1;
                    }
                }

                intf.sidePosition = firstUnoccupied;
                intf.side = side;
                intf.type = interfaceType;
                intf.maxConnectionCount = interfaceMaxConnectionsCount;

                occupied[intf.side].push(firstUnoccupied);
            }

            interfaces.push(intf);
        }

        // Remove extra interfaces if value of the property gets decreased.
        const container = direction === 'output' ? this.outputs : this.inputs;
        Object.keys(container).forEach((key) => {
            if (!key.startsWith(`${direction}_${interfaceName}[`)) {
                return;
            }

            const match = key.match(/\[(\d+)\]$/);
            if (!match) {
                return;
            }

            const idx = parseInt(match[1], 10);
            if (idx < value) {
                return;
            }

            if (direction === 'output') {
                this.removeOutput(key);
            } else {
                this.removeInput(key);
            }
        });

        const out = parseInterfaces(interfaces, [], []);
        if (Array.isArray(out) && out.length) {
            throw new Error(`Internal error, node ${this.type} invalid. Reason: ${out.join(' ')}`);
        }
        const { inputs: newInputs, outputs: newOutputs } = out;

        // Add new interfaces if they do not exist.
        Object.entries(direction === 'output' ? newOutputs : newInputs).forEach(([key, intf]) => {
            if ((key in container)) {
                return;
            }

            const baklavaIntf = new NodeInterface(key);
            Object.assign(baklavaIntf, intf);
            if (direction === 'output') {
                this.addOutput(key, baklavaIntf);
            } else {
                this.addInput(key, baklavaIntf);
            }
        });

        // Finding a reactive reference of `this` and using it to bind
        // the function to the node instance, so that the changes are
        // reflected in the editor
        const node = this.graph.nodes.find((n) => n.id === this.id);
        const reactiveUpdate = this.updateInterfaces.bind(node);
        reactiveUpdate(newInputs, newOutputs, false, [`${direction}_${interfaceName}`]);
    }

    toggleInterfaceGroup(intf, visible) {
        // If the interface is visible and is being disabled

        if (!intf.hidden && !visible) {
            const connections = this.graphInstance.connections.filter(
                (c) => c.from === intf || c.to === intf,
            );
            connections.forEach((c) => {
                this.graphInstance.removeConnection(c);
            });
        }

        // checking if there is an interface with the same side position
        if (visible) {
            updateInterfacePosition(this, intf, intf.side);
        }
        // It may also need a new sidePosition
        intf.hidden = !visible; // eslint-disable-line no-param-reassign
    }

    save() {
        const savedState = super.save();
        const newProperties = [];
        const newInterfaces = [];
        const enabledInterfaceGroups = [];

        Object.entries({ ...this.inputs, ...this.outputs }).forEach((io) => {
            const [ioName, ioState] = io;

            if (ioState.port) {
                if (!ioState.hidden) {
                    if (ioState.interfaces) {
                        // Enabled interface groups
                        enabledInterfaceGroups.push({
                            name: ioName.slice(ioState.direction.length + 1),
                            direction: ioState.direction,
                        });
                    }
                    const busStubs = ioState.bus?.stubs?.map((stub) => ({
                        id: stub.id,
                        offset: Math.floor(stub.offset),
                        ...(ioState.bus?.type === 'twoSided' && { side: stub.side }),
                    }));
                    const bus = {
                        size: ioState.bus?.size,
                        type: ioState.bus?.type,
                        stubs: busStubs,
                    };
                    const isBusEmpty = (!bus.size || !bus.type
                        || busStubs.length === 0);

                    newInterfaces.push({
                        name: ioName.slice(ioState.direction.length + 1),
                        externalName: ioState.externalName,
                        id: ioState.id,
                        direction: ioState.direction,
                        side: ioState.side,
                        ...(!ioState.bus && { sidePosition: ioState.sidePosition }),
                        ...(!isBusEmpty && { bus }),
                    });
                }
            } else {
                newProperties.push({
                    name: ioName.slice('property'.length + 1),
                    externalName: ioState.externalName,
                    id: ioState.id,
                    value: ioState.value === undefined ? null : ioState.value,
                    dynamicCounter: ioState.dynamicCounter,
                    hidden: ioState.hidden,
                });
            }
        });

        delete savedState.inputs;
        delete savedState.outputs;
        savedState.interfaces = newInterfaces;
        savedState.properties = newProperties;
        savedState.enabledInterfaceGroups = enabledInterfaceGroups;
        savedState.relatedGraphs = this.relatedGraphs;
        savedState.views = this.views;

        savedState.name = savedState.type;
        delete savedState.type;

        savedState.instanceName = savedState.title === '' ? undefined : savedState.title;
        delete savedState.title;

        savedState.height = this.height;

        if (savedState.position) {
            savedState.position = { ...savedState.position };
        }
        return savedState;
    }

    updateInterfaceKey(type, intf, oldKey, newKey) {
        const afterEvent = type === 'output' ? this.events.addOutput : this.events.addInput;
        const ioObject = type === 'output' ? this.outputs[oldKey] : this.inputs[oldKey];

        // remove old key
        ioObject.events.setValue.unsubscribe(this);
        if (type === 'output') {
            delete this.outputs[oldKey];
        } else {
            delete this.inputs[oldKey];
        }
        afterEvent.emit(ioObject);

        // add a new key
        if (type === 'output') {
            this.outputs[newKey] = intf;
        } else {
            this.inputs[newKey] = intf;
        }
        this.initializeIntf(type, newKey, intf);
        afterEvent.emit(intf);
    }

    /**
     * Function used to update interfaces of a node when loading a dataflow
     * in a development mode.
     *
     * @param {object} stateInputs newInputs of the node
     * @param {object} stateOutputs newOutputs of the node
     * @param {boolean} updateInterfaces determines what to do if an interface in either
     * @param {string[]} include prefixes of names of interfaces that are to be removed. Other
     * interfaces are left untouched. If set to undefined then all interfaces are updated.
     * `stateInputs` or `stateOutputs` already exists in the node. If set to `true`, the
     * interface will be updated with the new values, otherwise it will be left untouched.
     */
    updateInterfaces(stateInputs, stateOutputs, updateInterfaces = true, include = undefined) {
        const errors = [];
        // Updating interfaces of a graph node
        Object.entries(this.inputs).forEach(([k, intf]) => {
            // Process only interfaces, not properties
            if (intf.direction === undefined) return;
            if (
                !Object.keys(stateInputs).includes(k) &&
                (include === undefined || include.some((prefix) => k.startsWith(prefix)))
            ) {
                errors.push(
                    `Interface '${intf.name}' of direction '${intf.direction}' ` +
                    `removed as it was not found in the dataflow.`,
                );

                // The interface might have to be privatzed
                this.graph.editor.privatizeInterface(this.graph.id, intf);

                this.removeInput(k);
            }
        });
        Object.entries(stateInputs).forEach(([idA, intfA]) => {
            if (intfA.direction === undefined) return;
            const foundIntf = Object.entries(this.inputs).find(
                ([idB, intfB]) => idB === idA && intfB.direction === intfA.direction,
            );
            if (foundIntf === undefined) {
                const baklavaIntf = new NodeInterface(idA);
                errors.push(
                    `Interface '${intfA.name}' of direction '${intfA.direction}' ` +
                    `created as it was not found in the specification.`,
                );
                Object.assign(baklavaIntf, intfA);
                this.addInterface(baklavaIntf.direction, idA, baklavaIntf);
            } else if (updateInterfaces) {
                Object.assign(foundIntf[1], intfA);
            }
        });

        Object.entries(this.outputs).forEach(([k, intf]) => {
            // Process only interfaces, not properties
            if (intf.direction === undefined) return;
            if (!Object.keys(stateOutputs).includes(k) &&
                (include === undefined || include.some((prefix) => k.startsWith(prefix)))
            ) {
                errors.push(
                    `Interface '${intf.name}' of direction '${intf.direction}' ` +
                    `removed as it was not found in the dataflow.`,
                );

                // The interface might have to be privatzed
                this.graph.editor.privatizeInterface(this.graph.id, intf);

                this.removeOutput(k);
            }
        });
        Object.entries(stateOutputs).forEach(([idA, intfA]) => {
            const foundIntf = Object.entries(this.outputs).find(
                ([idB, intfB]) => idB === idA && intfB.direction === intfA.direction,
            );
            if (foundIntf === undefined) {
                const baklavaIntf = new NodeInterface(idA);
                errors.push(
                    `Interface '${intfA.name}' of direction '${intfA.direction}' ` +
                    `created as it was not found in the specification.`,
                );
                Object.assign(baklavaIntf, intfA);
                this.addInterface(baklavaIntf.direction, idA, baklavaIntf);
            } else if (updateInterfaces) {
                Object.assign(foundIntf[1], intfA);
            }
        });
        return errors;
    }

    updateProperties(stateProperties) {
        const errors = [];
        // Updating properties of a graph node
        Object.entries(this.inputs).forEach(([k, prop]) => {
            // Process only properties, not interfaces
            if (prop.direction !== undefined) return;
            if (!Object.keys(stateProperties).includes(k)) {
                errors.push(
                    `Property '${prop.name}' ` +
                    `removed as it was not found in the dataflow.`,
                );
                this.removeInput(k);
            }
        });
        Object.entries(stateProperties).forEach(([idA, propA]) => {
            if (propA.direction !== undefined) return;
            const foundProp = Object.entries(this.inputs).find(
                ([idB]) => idB === idA,
            );
            if (foundProp === undefined) {
                const baklavaProp = new InputInterface(
                    propA.name,
                    propA.value,
                );
                errors.push(
                    `Property '${propA.name}' ` +
                    `created as it was not found in the specification.`,
                );
                Object.assign(baklavaProp, propA);
                this.addInput(idA, baklavaProp);
            }
        });
        return errors;
    }

    load(state) {
        let parsedState;

        // `parsed` determines whether the state was already parsed before loading
        // This is caused by the fact that `load` can be used both to load a state
        // from a dataflow and from an instance of a node
        if (Object.prototype.hasOwnProperty.call(state, 'parsed') && state.parsed) {
            parsedState = state;
        } else {
            parsedState = parseNodeState(state);

            if (Array.isArray(parsedState) && parsedState.length) {
                return parsedState;
            }
        }

        let errors = [];

        Object.entries(parsedState.inputs).forEach(([name, intf]) => {
            if (!name.startsWith('property_')) return;
            if (intf.dynamicCounter) {
                this.updateDynamicInterfaces(intf);
            }
        });

        if (!globalProperties.softLoad) {
            errors = detectDiscrepancies(parsedState, this.inputs, this.outputs);
            if (Array.isArray(errors) && errors.length) {
                return errors;
            }
        } else {
            parsedState = resolveDiscrepancies(parsedState, this.inputs, this.outputs);
        }

        super.load(parsedState);

        this.height = parsedState?.height;

        // Disabling default interface groups if the node has its own state
        if (Object.keys(parsedState.enabledInterfaceGroups).length) {
            Object.entries({ ...this.inputs, ...this.outputs }).forEach(([, intf]) => {
                // If this is an interfaces group
                if (intf.interfaces !== undefined) {
                    intf.hidden = true; // eslint-disable-line no-param-reassign
                }
            });
        }

        if (parsedState.views) {
            this.views = parsedState.views;
        }

        // Enabling interface groups
        Object.entries(parsedState.enabledInterfaceGroups).forEach(
            ([groupName, groupState]) => {
                if (groupState.direction === 'input' || groupState.direction === 'inout') {
                    this.inputs[groupName].hidden = false;
                } else if (groupState.direction === 'output') {
                    this.outputs[groupName].hidden = false;
                }
            },
        );

        const occupied = {
            left: [], right: [], top: [], bottom: [],
        };

        const stateios = { ...parsedState.inputs, ...parsedState.outputs };

        // Assigning sides and sides Positions to interfaces
        Object.entries(stateios).forEach(([ioName, ioState]) => {
            let intf;
            if (ioState.direction === 'input' || ioState.direction === 'inout') {
                if (!(ioName in this.inputs)) {
                    const baklavaIntf = new NodeInterface(ioName);
                    Object.assign(baklavaIntf, ioState);
                    this.addInput(ioName, baklavaIntf);
                }
                intf = this.inputs[ioName];
                intf.side = ioState.side;
                intf.sidePosition = ioState.sidePosition;
                intf.externalName = ioState.externalName;
                intf.direction = ioState.direction;
                occupied[ioState.side].push(ioState.sidePosition);
            } else if (ioState.direction === 'output') {
                if (!(ioName in this.outputs)) {
                    const baklavaIntf = new NodeInterface(ioName);
                    Object.assign(baklavaIntf, ioState);
                    this.addOutput(ioName, baklavaIntf);
                }
                intf = this.outputs[ioName];
                intf.side = ioState.side;
                intf.sidePosition = ioState.sidePosition;
                intf.externalName = ioState.externalName;
                intf.direction = ioState.direction;
                occupied[ioState.side].push(ioState.sidePosition);
            } else if (ioState.direction === undefined) {
                if (!(ioName in this.inputs)) {
                    const baklavaIntf = new NodeInterface(ioName);
                    Object.assign(baklavaIntf, ioState);
                    this.addOutput(ioName, baklavaIntf);
                }
                intf = this.inputs[ioName];
                intf.externalName = ioState.externalName;
                if (ioState.hidden !== undefined) this.inputs[ioName].hidden = ioState.hidden;
            }
            if (ioState.bus?.stubs && intf) {
                intf.bus = intf.bus ?? {};
                intf.bus.stubs = ioState.bus.stubs;
                // eslint-disable-next-line no-restricted-syntax, guard-for-in
                for (const stubId in intf.bus.stubs) {
                    const stub = intf.bus.stubs[stubId];
                    const side = intf.bus.type === 'twoSided' ? stub.side : intf.side;
                    stub.nodeId = intf.nodeId;
                    stub.direction = intf.direction;
                    stub.maxConnectionsCount = 1;
                    stub.type = intf.type;
                    stub.sidePosition = 0;
                    stub.side = side ?? intf.side;
                    stub.parent = intf;
                    stub.isInput = intf.isInput;
                }
            }
        });

        const refreshSidePositions = (entries) => {
            // When state provided in the graph is incomplete, e.g. it misses
            // an interface, we allow it.
            // This, however, requires from us that we make sure that newly added
            // interfaces (not present in parsedState) are not on conflicting positions
            Object.entries(entries).forEach(([ioName, ioState]) => {
                if (ioName.startsWith('property_')) return;
                // if interface was explicitly defined in the graph file, skip it
                if (ioName in stateios) return;
                // otherwise, if the interface was implicitly created but it does not
                // cover existing interface, skip it
                if (!occupied[ioState.side].includes(ioState.sidePosition)) return;
                // if the positions are clashing, pick first available max position on
                // given side
                const maxposition = Math.max(...occupied[ioState.side]);
                ioState.sidePosition = maxposition + 1; // eslint-disable-line no-param-reassign
                occupied[ioState.side].push(maxposition + 1);
            });
        };

        refreshSidePositions(this.inputs);
        refreshSidePositions(this.outputs);

        // Default position should be undefined instead of (0, 0) so that it can be set
        // by autolayout
        if (state.position === undefined) {
            this.position = undefined;
        }

        this.views = state?.views;

        this.relatedGraphs = state.relatedGraphs;
        return errors;
    }

    onPlaced() {
        super.onPlaced();
        const externalRequest = ([method, params]) => this.graphInstance
            ?.editor
            ?.editorManager
            ?.externalApplicationManager
            ?.request(method, params);
        const goToGraph = (graphid) => this.graphInstance
            ?.editor.switchToRelatedGraph(graphid);
        Object.entries(this.inputs)
            .filter(([name, _]) => name.startsWith('property_'))
            .filter(([_, intf]) => intf.componentName === 'ButtonInterface')
            .forEach(([_, intf]) => (intf.type === 'button-graph' ? intf.events.updated.subscribe(this, goToGraph) : intf.events.updated.subscribe(this, externalRequest)),
            );
        Object.entries(this.inputs)
            .filter(([name, _]) => name.startsWith('property_'))
            .filter(([_, intf]) => intf.port === false)
            .forEach(([_, intf]) => (
                intf.events.beforeSetValue.subscribe(this, (args) => {
                    this.events.propertyEdit.emit([this, args, intf]);
                })
            ));
    }

    onDestroy() {
        [...Object.values(this.inputs), ...Object.values(this.outputs)].forEach((io) => {
            Object.values(io.events).forEach((event) => {
                // We need to unsubscribe from all events to avoid memory leaks
                // On token mismatch, the event will not be unsubscribed
                event.unsubscribe(this);
                event.unsubscribe(io);
            });
        });
    }
}

/**
 * @param parsedInterfaces that are validated and parsed. The format
 * should be the same as the one returned by parseInterfaces.
 * @returns created interfaces
 */
export const createBaklavaInterfaces = (parsedInterfaces) => {
    function createBaklavaInterface(intf) {
        return () => {
            const baklavaIntf = new NodeInterface(intf.name);
            Object.assign(baklavaIntf, intf);
            return baklavaIntf;
        };
    }

    // Creating interfaces for baklavajs
    const inputs = Object.fromEntries(
        Object.entries(parsedInterfaces.inputs).map(
            ([n, intf]) => [n, createBaklavaInterface(intf)],
        ),
    );

    const outputs = Object.fromEntries(
        Object.entries(parsedInterfaces.outputs).map(
            ([n, intf]) => [n, createBaklavaInterface(intf)],
        ),
    );

    return [inputs, outputs];
};

/**
 * Class factory that creates a class for a custom Node that is described by the arguments.
 * It can be later registered so that the user can use it and save the editor.
 * `inputs`, `properties` and `outputs` formats are described in the documentation.
 *
 * @param {string} name Name of the block that is stored when saving
 * @param {string} layer Layer of the node
 * @param {*} interfaces List of interfaces in the block (input, output and inout)
 * @param {*} properties List of properties of the block
 * @param {*} interfaceGroups Object describing groups of interfaces
 * @param {*} defaultInterfaceGroups Object describing groups of interfaces that
 * are enabled by default
 * @param {boolean} twoColumn type of layout of the nodes
 * @returns Node based class is successful, otherwise an array of errors is returned.
 */
export function CustomNodeFactory(
    name,
    layer,
    interfaces,
    properties,
    interfaceGroups,
    defaultInterfaceGroups,
    twoColumn,
    description = '',
    nodeExtends = [],
    nodeExtending = [],
    nodeSiblings = [],
    width = 0,
    simpleInherited = [],
    height = undefined,
) {
    const generatedProperties = generateProperties(interfaces);
    if (!generatedProperties.success) {
        return generatedProperties.value.map((error) => `Node ${name} invalid. ${error}`);
    }

    const parsedInterfaces = parseInterfaces(interfaces, interfaceGroups, defaultInterfaceGroups);
    // If parsedInterfaces returns an array, it is an array of errors
    if (Array.isArray(parsedInterfaces) && parsedInterfaces.length) {
        return parsedInterfaces.map((error) => `Node ${name} invalid. ${error}`);
    }
    const [inputs, newOutputs] = createBaklavaInterfaces(parsedInterfaces);

    const parsedProperties = parseProperties([...properties, ...generatedProperties.value]);
    // If parsedProperties returns an array, it is an array of errors
    if (Array.isArray(parsedProperties) && parsedProperties.length) {
        return parsedProperties.map((error) => `Node ${name} invalid. ${error}`);
    }
    const createdProperties = createProperties(parsedProperties);

    const newInputs = {
        ...inputs,
        ...createdProperties,
    };

    return class extends CustomNode {
        constructor() {
            super(
                name,
                layer,
                newInputs,
                newOutputs,
                twoColumn,
                description,
                nodeExtends,
                nodeExtending,
                nodeSiblings,
                width,
                simpleInherited,
                undefined,
                height,
            );
        }
    };
}

/**
 * Function checks for duplicate interface names in the subgraph.
 *
 * An error message is returned for each duplicate name.
 *
 * @param interfaces External interfaces in the subgraph
 *
 * @returns List of new interfaces and errors
 */
function checkInterfaceNames(interfaces) {
    const countedIntfNames = Object.create(null);
    const externalInterfaces = [];
    interfaces?.forEach(
        ([, intf]) => {
            countedIntfNames[intf.externalName] = (countedIntfNames[intf.externalName] ?? 0) + 1;
            if (countedIntfNames[intf.externalName] === 1) {
                externalInterfaces.push(intf);
            } else {
                externalInterfaces.push(`Interface '${intf.externalName}' is repeated ${countedIntfNames[intf.externalName]} times.`);
            }
        },
    );
    return externalInterfaces;
}

function updateInterfaceRegistry(intf, graphId) {
    // It may happen that the registered interface has the same id, but is a reference,
    // for example when dealing with history or clipboard.
    if (
        ir.isRegistered(intf.id) &&
        ir.getRegisteredInterface(intf.id).sharedInterface !== intf &&
        ir.getRegisteredInterface(intf.id).sharedInterface.id === intf.id &&
        ir.getRegisteredInterface(intf.id).sharedInterfaceGraphId === graphId
    ) {
        ir.deleteRegisteredInterface(intf.id);
    }

    if (!ir.isRegistered(intf.id)) {
        ir.registerInterface(intf, graphId);
    }
}

/**
 * Function looks for graph node interfaces based on the nodes inside of it.
 *
 * It finds all interfaces that have `externalName` property set and registers
 * them in the InterfaceRegistry, if they are not already registered, so that
 * they their state can be easily shared and synchronized.
 *
 * If there are any errors, they are returned as an array of strings.
 * If the operation was successful, the new inputs and outputs are returned.
 *
 * @param nodes Nodes of the subgraph
 * @param inputs List of inputs of the subgraph
 * @param outputs List of outputs of the subgraph
 *
 * @returns List of errors or new inputs and outputs
 */
export function updateSubgraphInterfaces(nodes, inputs = [], outputs = []) {
    // Interfaces that are not connected to any other interface
    const INTERFACE_PREFIXES = ['input_', 'inout_', 'output_'];

    const exposedIntf = [
        ...nodes.map((node) => Object.entries({ ...node.inputs, ...node.outputs })).flat(),
    ].filter(([key]) => INTERFACE_PREFIXES.some((prefix) => key.startsWith(prefix)))
        .filter(
            ([, intf]) => intf.externalName,
        );

    // Filter out repeated external names
    const externalInterfaces = checkInterfaceNames(exposedIntf);
    const errorMessages = externalInterfaces.filter((n) => typeof n === 'string');
    if (errorMessages.length) {
        return errorMessages;
    }

    const checkInterfaceSide = (side) => {
        if (side === 'top') {
            return 'left';
        }
        if (side === 'bottom') {
            return 'right';
        }

        return side;
    };

    // Create new inputs and outputs
    const newInterfaces = [];
    externalInterfaces.forEach((intf) => {
        const graphId = nodes[0].graph.id;
        updateInterfaceRegistry(intf, graphId);

        const container = intf.direction === 'output' ? outputs : inputs;
        const idx = container.findIndex((x) => x.id === intf.id);
        if (idx === -1) {
            // Graph node interface should not inherit some properties that
            // are node-specific, they will be accessed using InterfaceRegistry

            newInterfaces.push({
                name: intf.externalName,
                id: intf.id,
                externalName: undefined,
                side: checkInterfaceSide(intf.side),
                direction: intf.direction,
                sidePosition: undefined,
            });
        } else {
            container[idx].name = intf.externalName;
            newInterfaces.push(container[idx]);
        }
    });

    const newInterfacesPositionsOrErrors = applySidePositions(newInterfaces, {});
    if (Array.isArray(newInterfacesPositionsOrErrors) && newInterfacesPositionsOrErrors.length) {
        return newInterfacesPositionsOrErrors;
    }

    return {
        inputs: Object.values(newInterfacesPositionsOrErrors.inputs),
        outputs: Object.values(newInterfacesPositionsOrErrors.outputs),
    };
}

/**
 * Function looks for graph node properties based on the nodes inside of it.
 *
 * If there are any errors, they are returned as an array of strings.
 * If the operation was successful, the new properties are returned.
 *
 * @param nodes Nodes of the subgraph
 *
 * @returns List of errors or new properties
 */
export function updateSubgraphProperties(nodes) {
    const exposedProperties = nodes.map((node) => Object.entries(node.inputs)).flat()
        .filter(([key]) => key.startsWith('property_'))
        .filter(([, prop]) => prop.externalName);

    // Filter out repeated external names
    const externalProperties = checkInterfaceNames(exposedProperties);
    const errorMessages = externalProperties?.filter((n) => typeof n === 'string');
    if (errorMessages.length) {
        return errorMessages;
    }

    const newProperties = [];
    externalProperties.forEach((property) => {
        const graphId = nodes[0].graph.id;
        updateInterfaceRegistry(property, graphId);

        newProperties.push({
            name: property.externalName,
            id: property.id,
            externalName: undefined,
            type: property.type,
            value: property.value,
            description: property.description,
            default: property.default,
            min: property.min,
            max: property.max,
            step: property.step,
            values: property.items,
            dtype: property.dtype,
            override: property.override,
            readonly: property.readonly,
        });
    });
    return newProperties;
}

/**
 * Function creating a graph template as defined in specification
 *
 * @param nodes Nodes of the subgraph
 * @param connections Connections inside the subgraph
 * @param name Default name that will be displayed in editor
 * @param editor PipelineManagerEditor instance
 * @returns Graph template that will be used to define the subgraph node
 */
export function GraphFactory(nodes, connections, name, editor) {
    const parsedState = nodes.map((node) => parseNodeState(node));
    const errorMessages = parsedState.filter((n) => typeof n === 'string');
    if (errorMessages.length) {
        return errorMessages;
    }

    const state = {
        name,
        nodes: parsedState,
        connections,
        inputs: [],
        outputs: [],
    };

    return new GraphTemplate(state, editor);
}
