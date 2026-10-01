/*
 * Copyright (c) 2022-2024 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */

/* eslint-disable class-methods-use-this */
export default class BaklavaInterfaceTypes {
    types = new Map();

    defaultStyle = {
        interfaceConnectionPattern: 'solid',
        interfaceConnectionColor: '#FFFFFF', // $white
        interfaceColor: '#00E58D', // $green
    };

    /**
     * Initialize Interface types instance used to manager styles of interfaces and connections
     * and validate adding connections.
     */
    constructor(viewPlugin) {
        viewPlugin.hooks.renderInterface.subscribe(this, ({ intf, el }) => {
            if (intf.type) {
                const types = this.normalizeType(intf.type);
                const firstType = types.find((t) => this.types[t]?.interfaceColor !== undefined);

                if (firstType !== undefined) {
                    const color = this.types[firstType].interfaceColor;
                    const arrows = el.querySelectorAll('.__port:not(.greyedout_arrow)');
                    // iterate through all in case interface bus is used
                    arrows.forEach((arrow) => {
                        // eslint-disable-next-line no-param-reassign
                        if (arrow !== null) arrow.style.backgroundColor = color;
                        else {
                            const greyArrow = el.querySelector('.__port');
                            if (greyArrow !== null) {
                                greyArrow.style.backgroundColor =
                                    getComputedStyle(greyArrow).getPropertyValue('$gray-500');
                            }
                        }
                    });
                    const buses = el.querySelectorAll('.__port-bus:not(.greyedout_arrow)');
                    buses.forEach((bus) => {
                        // eslint-disable-next-line no-param-reassign
                        if (bus !== null) bus.style.borderColor = color;
                    });
                }
            }

            return { intf, el };
        });
    }

    normalizeType(type) {
        return typeof type === 'string' || type instanceof String ? [type] : type;
    }

    /**
     * Returns connection style for a given from and to interfaces.
     * It takes style of a common type of those interfaces and completes its missing values
     * with default ones. If there are multiple common types, a default style is returned.
     *
     * If there is no `to` interface then a style for `from` interface is returned.
     * Again with completed missing values.
     *
     * @param {Interface} from connection source
     * @param {Interface} to connection target
     * @returns style for a defined connection
     */
    getConnectionStyle(from, to) {
        const fromTypes = this.normalizeType(from?.type);
        const toTypes = this.normalizeType(to?.type);

        if (from?.type === undefined) {
            return this.defaultStyle;
        }

        if (to?.type === undefined) {
            const firstType = fromTypes.find((t) => this.types[t] !== undefined);
            return { ...this.defaultStyle, ...this.types[firstType] };
        }

        const commonTypes = fromTypes.filter((t) => toTypes.includes(t));
        if (Array.isArray(commonTypes) && commonTypes.length > 1) {
            return this.defaultStyle;
        }

        const firstType = commonTypes.find((t) => this.types[t] !== undefined);
        return { ...this.defaultStyle, ...this.types[firstType] };
    }

    /**
     * Function that reads all nodes in the specification and creates `NodeInterfaceType` objects
     * for their inputs' and outputs' types so that a simple validation based on those
     * types can be performed.
     *
     * The read interface types are stored in `interfaceTypes` object which is returned by
     * this function
     * @param {*} metadata metadata containing information about styling
     * @returns read interface types
     */
    readInterfaceTypes(metadata) {
        this.types = {};
        if (metadata?.interfaces) {
            Object.entries(metadata.interfaces).forEach(([type, io]) => {
                this.types[type] = { ...io, name: type };
            });
        }
    }
}
