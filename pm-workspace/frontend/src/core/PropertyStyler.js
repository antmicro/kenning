/*
 * Copyright (c) 2022-2026 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */

import { validateCssObject, getCssString } from './cssValidation.ts';

/**
 * Class that allows for dynamic modification of css styles of properties.
 */
/* eslint-disable class-methods-use-this */
export default class BaklavaPropertyStyler {
    styles = new Map();

    defaultStyle = {
        show: true,
    };

    editor = undefined;

    /**
     * Initialize Property types instance used to manager styles of properties.
     */
    constructor(viewPlugin) {
        this.editor = viewPlugin.editor;
        viewPlugin.hooks.renderInterface.subscribe(this, ({ intf, el }) => {
            if (intf.changedStyle !== undefined) {
                Object.entries(intf.changedStyle).forEach(([p, _]) => el.style.removeProperty(p));
                // eslint-disable-next-line no-param-reassign
                delete intf.changedStyle;
            }
            // eslint-disable-next-line no-underscore-dangle
            const node = this.editor._graph.getNode(intf.nodeId);
            if (!node) return { intf, el };
            const styleName = this.editor.getNodeInstanceStyle(node);
            const style = this.styles[styleName];
            if (!style) return { intf, el };

            const property = intf;
            if (property.type && !intf.port) {
                // eslint-disable-next-line prefer-destructuring
                const type = property.type;
                const firstType = style[type] ?? style['*'];
                if (firstType !== undefined) {
                    if (!el) return { intf, el };
                    Object.entries(firstType).forEach(([p, v]) => el.style.setProperty(p, v, 'important'));
                    // eslint-disable-next-line no-param-reassign
                    intf.changedStyle = firstType;
                }
            }
            return { intf, el };
        });
    }

    /**
     * Function that reads all nodes in the specification and creates objects
     * for each style, then for each style a specific property type eg. 'text'
     * has its' CSS customization defined.
     *
     * The read interface types are stored in `styles` object which is returned by
     * this function
     * @param {*} metadata metadata containing information about styling
     * @returns read property types
     */
    readPropertyStyles(metadata) {
        this.styles = {};
        if (!metadata?.styles) return this.styles;
        Object.entries(metadata?.styles).filter(([_, style]) => style?.properties)
            .forEach(([styleName, style]) => {
                const io = style.properties;
                this.styles[styleName] = { ...io };
                Object.values(io).forEach((e) => {
                    if (!io[e]) return;
                    if (!validateCssObject(io[e])) {
                        throw new Error(`invalid CSS provided in property-specific styling:\n${getCssString(io[e])}`);
                    }
                });
            });
        return this.styles;
    }
}
