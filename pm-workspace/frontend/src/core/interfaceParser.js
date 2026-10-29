/*
 * Copyright (c) 2022-2024 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

const defaultDirection = 'inout';

/**
 * Returns a prepared interface that is passed to baklava constructor
 *
 * @param io configuration of the interface
 * @param hidden whether th interface should be hidden. For example groups of interfaces
 * are hidden by default
 * @param {*} name custom name for the interface that should be used instead of the one coming
 * from `io`
 * @returns baklava interface constructor
 */
function createInterface(io, hidden, name = undefined) {
    const intf = {};
    Object.assign(intf, io);

    intf.name = name ?? io.name;
    if (intf.type !== undefined) {
        intf.type = typeof io.type === 'string' || io.type instanceof String ? [io.type] : io.type;
    }
    intf.componentName = 'NodeInterface';
    intf.hidden = hidden;

    // Readonly values used for detecting whether there were any changes to the interface
    intf.originalSide = intf.side;
    intf.originalSidePosition = intf.sidePosition;
    return intf;
}

/**
 * Parses and validates interfaces passed in specification
 *
 * @param interfaces list of interfaces from specification that is going to be parsed
 * @param interfaceGroup determines whether `interfaces` are interface groups. If true the
 * additionally field `.interfaces` is parsed.
 * @returns parsed interfaces that can be passed to baklavajs if the interfaces were valid.
 * Otherwise an array of errors is returned.
 */
function parseSingleInterfaces(interfaces, interfaceGroup = false) {
    const errors = [];
    const tempParsed = {
        input: {},
        inout: {},
        output: {},
    };

    interfaces.forEach((io) => {
        // Omitting interfaces that are marked as dynamic, as they are generated
        // using 'updateDynamicInterfaces function
        if (Object.prototype.hasOwnProperty.call(io, 'dynamic')) return;

        // Copy the interface to avoid modifying the original object
        const tempIO = JSON.parse(JSON.stringify(io));

        if (tempIO?.x !== undefined) {
            tempIO.x = Math.max(Math.min(tempIO.x / 100.0, 1.0), 0.0);
        }
        if (tempIO?.y !== undefined) {
            tempIO.y = Math.max(Math.min(tempIO.y / 100.0, 1.0), 0.0);
        }

        const direction = io.direction ?? defaultDirection;
        tempIO.direction = direction;

        if (io.array !== undefined) {
            const [left, right] = io.array;

            for (let j = left; j < right; j += 1) {
                const name = `${io.name}[${j}]`;

                if (tempParsed[direction][name] !== undefined) {
                    errors.push(
                        `Interface named '${name}' of direction '${direction}' is a duplicate.`,
                    );
                }
                tempIO.externalName = io.externalName ? `${io.externalName}[${j}]` : undefined;

                // Copy the interface to avoid modifying the assigned object
                tempParsed[direction][name] = JSON.parse(JSON.stringify(tempIO));
            }
        } else {
            if (tempParsed[direction][io.name] !== undefined) {
                errors.push(
                    `Interface named '${io.name}' of direction '${direction}' is a duplicate.`,
                );
            }

            tempParsed[direction][io.name] = tempIO;
        }

        if (interfaceGroup) {
            const newInterfaces = [];

            tempIO.interfaces.forEach((buildingIO) => {
                const bdirection = buildingIO.direction ?? defaultDirection;
                if (buildingIO.array !== undefined) {
                    const [left, right] = buildingIO.array;

                    for (let j = left; j < right; j += 1) {
                        const name = `${bdirection}_${buildingIO.name}[${j}]`;
                        newInterfaces.push(name);
                    }
                } else {
                    const name = `${bdirection}_${buildingIO.name}`;
                    newInterfaces.push(name);
                }
            });
            tempIO.interfaces = newInterfaces; // eslint-disable-line no-param-reassign
        }
    });

    // Removing inout with duplicate names
    const filteredTempInouts = Object.fromEntries(
        Object.entries(tempParsed.inout).filter(([name, state]) => {
            const direction = state.direction ?? defaultDirection;

            const tempParsedInterfaces = [...Object.values(tempParsed.input),
                ...Object.values(tempParsed.output)];

            const duplicate = tempParsedInterfaces.find((intf) => intf.name === name
                        && intf.side === state.side);

            if (duplicate) {
                errors.push(
                    `Interface named '${name}' of direction '${direction}' ` +
                        `is a duplicate. There already exists an input or output of this name.`,
                );
            }
            return !duplicate;
        }),
    );

    tempParsed.inout = filteredTempInouts;
    tempParsed.input = { ...tempParsed.input, ...tempParsed.inout };
    delete tempParsed.inout;

    tempParsed.input = Object.fromEntries(
        Object.entries(tempParsed.input).map(([name, state]) => [
            `${state.direction}_${name}`,
            state,
        ]),
    );
    tempParsed.output = Object.fromEntries(
        Object.entries(tempParsed.output).map(([name, state]) => [
            `${state.direction}_${name}`,
            state,
        ]),
    );

    if (errors.length) {
        return errors;
    }

    return tempParsed;
}

/**
 * Checks whether interface groups that are in enabledInterfaceGroup
 * can be enabled at the same time
 * @param {array} enabledInterfaceGroups array of names of enabled interface groups
 * @param {*} inputs inputs of the node
 * @param {*} outputs outputs of the node
 * @returns list tuples that have information about conflicting interfaces.
 */
export function validateInterfaceGroupsNames(enabledInterfaceGroups, inputs, outputs) {
    const errors = [];
    // Checking for integrity of interface groups
    const usedInterfaces = new Set();

    enabledInterfaceGroups.forEach((name) => {
        const interfaces = inputs[name]?.interfaces ?? outputs[name]?.interfaces;
        const groupDirection = name.slice(0, name.indexOf('_'));
        const groupName = name.slice(name.indexOf('_') + 1);

        interfaces.forEach((intfName) => {
            if (usedInterfaces.has(intfName)) {
                const intfDirection = intfName.slice(0, intfName.indexOf('_'));
                const parsedIntfName = intfName.slice(intfName.indexOf('_') + 1);

                errors.push([parsedIntfName, intfDirection, groupName, groupDirection]);
            } else {
                usedInterfaces.add(intfName);
            }
        });
    });
    return errors;
}

/**
 * Checks whether interface groups that are in enabledInterfaceGroup
 * can be enabled at the same time
 * @param {array} enabledInterfaceGroups array of names of enabled interface groups
 * @param {*} inputs inputs of the node
 * @param {*} outputs outputs of the node
 * @returns list of explicit errors.
 */
export function validateInterfaceGroups(enabledInterfaceGroups, inputs, outputs) {
    const errors = validateInterfaceGroupsNames(enabledInterfaceGroups, inputs, outputs);
    const errorMessages = [];

    errors.forEach(([parsedIntfName, intfDirection, groupName, groupDirection]) => {
        errorMessages.push(
            `Interface of name '${parsedIntfName}' and direction '${intfDirection}' has been reused ` +
                `by interface group named '${groupName}' of direction '${groupDirection}'. ` +
                `Make sure your interface groups are disjoint.`,
        );
    });

    return errorMessages;
}

/**
 *
 * @param {*} inputs inputs of the node
 * @param {*} outputs outputs of the node
 * @returns inputs and outputs of the nodes with `sidePositions` assigned if
 * there were no errors found. Otherwise a list of errors is returned.
 */
export function applySidePositions(inputs, outputs) {
    const tempParsedSides = {
        left: {},
        right: {},
        top: {},
        bottom: {},
    };

    const errors = [];

    // Dividing interfaces into left and right sides
    Object.entries({ ...inputs, ...outputs }).forEach(([name, intf]) => {
        if (intf.side === 'right' || (intf.side === undefined && intf.direction === 'output')) {
            tempParsedSides.right[name] = { ...intf };
            tempParsedSides.right[name].side = 'right';
        } else if (intf.side === 'left' || (intf.side === undefined && intf.direction !== 'output')) {
            tempParsedSides.left[name] = { ...intf };
            tempParsedSides.left[name].side = 'left';
        } else if (intf.side === 'top') {
            tempParsedSides.top[name] = { ...intf };
            tempParsedSides.top[name].side = 'top';
        } else if (intf.side === 'bottom') {
            tempParsedSides.bottom[name] = { ...intf };
            tempParsedSides.bottom[name].side = 'bottom';
        }
    });

    const stripName = (name) => name.slice(name.indexOf('_') + 1);

    // validating and setting sidePositions
    const occupiedInputSidePositions = new Set();
    const occupiedOutputSidePositions = new Set();

    Object.entries(tempParsedSides.left).forEach(([name, intf]) => {
        if (intf.sidePosition !== undefined) {
            if (occupiedInputSidePositions.has(intf.sidePosition)) {
                errors.push(
                    `Interface named '${stripName(name)}' of direction '${intf.direction}' has ` +
                        `invalid sidePosition value '${intf.sidePosition}'. ` +
                        `There already exists an input or output with this sidePosition.`,
                );
            }
            occupiedInputSidePositions.add(intf.sidePosition);
        }
    });

    Object.entries(tempParsedSides.right).forEach(([name, intf]) => {
        if (intf.sidePosition !== undefined) {
            if (occupiedOutputSidePositions.has(intf.sidePosition)) {
                errors.push(
                    `Interface named '${stripName(name)}' of direction '${intf.direction}' has ` +
                        `invalid sidePosition value '${intf.sidePosition}'. ` +
                        `There already exists an input or output with this sidePosition.`,
                );
            }
            occupiedOutputSidePositions.add(intf.sidePosition);
        }
    });

    if (errors.length) {
        return errors;
    }

    let leftSidePositionIndex = 0;
    let rightSidePositionIndex = 0;

    const getLeftSidePos = (intf) => {
        if (intf.sidePosition === undefined) {
            while (occupiedInputSidePositions.has(leftSidePositionIndex)) {
                leftSidePositionIndex += 1;
            }
            occupiedInputSidePositions.add(leftSidePositionIndex);
            return leftSidePositionIndex;
        }
        return intf.sidePosition;
    };

    const getRightSidePos = (intf) => {
        if (intf.sidePosition === undefined) {
            while (occupiedOutputSidePositions.has(rightSidePositionIndex)) {
                rightSidePositionIndex += 1;
            }
            occupiedOutputSidePositions.add(rightSidePositionIndex);
            return rightSidePositionIndex;
        }
        return intf.sidePosition;
    };

    Object.entries(tempParsedSides.left).forEach(([, intf]) => {
        intf.sidePosition = getLeftSidePos(intf); // eslint-disable-line no-param-reassign,max-len
    });

    Object.entries(tempParsedSides.right).forEach(([, intf]) => {
        intf.sidePosition = getRightSidePos(intf); // eslint-disable-line no-param-reassign,max-len
    });

    Object.entries(tempParsedSides.top).forEach(([, intf]) => {
        intf.sidePosition = 0; // eslint-disable-line no-param-reassign,max-len
    });

    Object.entries(tempParsedSides.bottom).forEach(([, intf]) => {
        intf.sidePosition = 0; // eslint-disable-line no-param-reassign,max-len
    });

    return {
        inputs: Object.fromEntries(
            Object.entries({
                ...tempParsedSides.top,
                ...tempParsedSides.bottom,
                ...tempParsedSides.left,
                ...tempParsedSides.right,
            })
                .filter(([, intf]) => intf.direction !== 'output'),
        ),
        outputs: Object.fromEntries(
            Object.entries({
                ...tempParsedSides.top,
                ...tempParsedSides.bottom,
                ...tempParsedSides.left,
                ...tempParsedSides.right,
            })
                .filter(([, intf]) => intf.direction === 'output'),
        ),
    };
}

/**
 * The function reads provided `parsedInterfaces` and looks for interfaces with `dynamic` attribute.
 * For such interfaces, a dedicated property is created that controls the number interface
 * instances. The properties are returned and should be included in node creation.
 *
 * @param {Object} interfaces List of interfaces.
 * @return Object with two properties, success and value. If success is true, value contains
 * an array of properties that should be included in the node. Otherwise, value contains an array
 * of errors.
 */
export function generateProperties(interfaces) {
    const errors = [];
    const properties = [];
    interfaces.forEach(
        (intf) => {
            if (Object.prototype.hasOwnProperty.call(intf, 'dynamic')) {
                if (
                    Array.isArray(intf.dynamic) &&
                    intf.dynamic.length === 2 &&
                    Number.isInteger(intf.dynamic[0]) &&
                    Number.isInteger(intf.dynamic[1])
                ) {
                    // Property should have limits specified by 'intf.dynamic' value
                    properties.push({
                        name: `${intf.name} ${intf.direction}`,
                        type: 'integer',
                        min: intf.dynamic[0],
                        max: intf.dynamic[1],
                        default: intf.dynamic[0],
                        // The type of dynamic interfaces is stored as `interfaceType`
                        interfaceType: intf.type,
                        interfaceMaxConnectionCount: intf.maxConnectionCount,
                        dynamicCounter: true,
                    });
                } else if (intf.dynamic === true) {
                    // Property should not have limits
                    properties.push({
                        name: `${intf.name} ${intf.direction}`,
                        type: 'integer',
                        min: 0,
                        default: 0,
                        // The type of dynamic interfaces is stored as `interfaceType`
                        interfaceType: intf.type,
                        interfaceMaxConnectionCount: intf.maxConnectionCount,
                        dynamicCounter: true,
                    });
                } else {
                    errors.push(
                        `Interface '${intf.name}' has invalid 'dynamic' attribute. ` +
                        'It should be either a boolean or an array with two integer elements.',
                    );
                }
            }
        },
    );

    if (errors.length) {
        return { success: false, value: errors };
    }
    return { success: true, value: properties };
}

/**
 * @param {*} interfaces List of interfaces in the block (input, output and inout)
 * @param {*} interfaceGroups Object describing groups of interfaces'
 * @param {*} defaultInterfaceGroups Object describing groups of interfaces that are enabled
 * @returns object that has inputs and outputs key if parsing was successful,
 * a list of errors otherwise.
 */
export function parseInterfaces(
    interfaces,
    interfaceGroups,
    defaultInterfaceGroups,
) {
    let errors = [];

    // Parsing single interfaces first
    const tempParsed = parseSingleInterfaces(interfaces);

    // If parseSingleInterfaces returns an array, it is an array of errors
    if (Array.isArray(tempParsed) && tempParsed.length) {
        return tempParsed;
    }

    // Checking for integrity of interface groups
    interfaceGroups.forEach((intfG) => {
        const directionG = intfG.direction ?? defaultDirection;
        intfG.interfaces.forEach((intf) => {
            const direction = intf.direction ?? defaultDirection;
            if (intf.array !== undefined) {
                const [left, right] = intf.array;

                for (let j = left; j < right; j += 1) {
                    const name = `${direction}_${intf.name}[${j}]`;
                    if (
                        !Object.keys({ ...tempParsed.input, ...tempParsed.output }).includes(name)
                    ) {
                        errors.push(
                            `Interface named '${intf.name}[${j}]' of direction '${direction}' ` +
                                `used for interface group '${intfG.name}' of direction ` +
                                `'${directionG}' does not exist.`,
                        );
                    }
                }
            } else {
                const name = `${direction}_${intf.name}`;
                if (!Object.keys({ ...tempParsed.input, ...tempParsed.output }).includes(name)) {
                    errors.push(
                        `Interface named '${intf.name}' of direction '${direction}' ` +
                            `used for interface group '${intfG.name}' of direction ` +
                            `'${directionG}' does not exist.`,
                    );
                }
            }
        });
    });

    if (errors.length) {
        return errors;
    }

    const tempParsedGroups = parseSingleInterfaces(interfaceGroups, true);
    // If parseSingleInterfaces returns an array, it is an array of errors
    if (Array.isArray(tempParsedGroups) && tempParsedGroups.length) {
        return tempParsedGroups;
    }

    // All interfaces that create some interfaces groups
    const interfacesCreatingGroups = new Set();
    Object.values({
        ...tempParsedGroups.input,
        ...tempParsedGroups.output,
    }).forEach((state) => {
        state.interfaces.forEach((intf) => interfacesCreatingGroups.add(intf));
    });

    // Detecting integrity of enabled interface groups
    const enabledInterfaceGroupsNames = defaultInterfaceGroups.map(
        (group) => `${group.direction ?? defaultDirection}_${group.name}`,
    );

    errors = validateInterfaceGroups(
        enabledInterfaceGroupsNames,
        { ...tempParsedGroups.input, ...tempParsed.input },
        { ...tempParsedGroups.output, ...tempParsed.output },
    );

    if (errors.length) {
        return errors;
    }

    // Interfaces that belong to groups are removed as they should not have side
    // positions applied as they are never rendered
    tempParsed.input = Object.fromEntries(Object.entries(tempParsed.input).filter(
        ([name]) => !interfacesCreatingGroups.has(name),
    ));

    tempParsed.output = Object.fromEntries(Object.entries(tempParsed.output).filter(
        ([name]) => !interfacesCreatingGroups.has(name),
    ));

    const parsedSides = applySidePositions(
        { ...tempParsed.input, ...tempParsedGroups.input },
        { ...tempParsed.output, ...tempParsedGroups.output },
    );
    if (Array.isArray(parsedSides) && parsedSides.length) {
        return parsedSides;
    }

    const stripName = (name) => name.slice(name.indexOf('_') + 1);

    const createdInterfaces = {
        inputs: {},
        outputs: {},
    };

    // Filtering single interfaces that are part of interface groups
    // Those interfaces are removed as they are never rendered
    // This is only used when parsing a specification format
    Object.entries(parsedSides.inputs).forEach(([name, intf]) => {
        // It is an interface group
        if (intf.interfaces !== undefined) {
            // Adding interfaces groups, hidden by default
            createdInterfaces.inputs[name] = createInterface(
                intf,
                !enabledInterfaceGroupsNames.includes(name),
                stripName(name),
            );
        } else {
            createdInterfaces.inputs[name] = createInterface(
                intf,
                false,
                stripName(name),
            );
        }
    });

    Object.entries(parsedSides.outputs).forEach(([name, intf]) => {
        // It is an interface group
        if (intf.interfaces !== undefined) {
            // Adding interfaces groups, hidden by default
            createdInterfaces.outputs[name] = createInterface(
                intf,
                !enabledInterfaceGroupsNames.includes(name),
                stripName(name),
            );
        } else {
            createdInterfaces.outputs[name] = createInterface(
                intf,
                false,
                stripName(name),
            );
        }
    });

    return createdInterfaces;
}
