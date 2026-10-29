/*
 * Copyright (c) 2022-2026 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */

import { validateCssObject, getCssString } from './cssValidation.ts';

/**
 * Class that allows for dynamic modification of css styles of interfaces.
 * Analogous to ./InterfaceTypes but allows for any css modifications.
 * Old InterfaceTypes is kept for connection modifications and legacy support.
 */
/* eslint-disable class-methods-use-this */
export default class BaklavaInterfaceStyler {
    styles = new Map();

    defaultStyle = {
        show: true,
    };

    editor = undefined;

    /**
     * Initialize Interface styler object instance used to manager styles of interfaces.
     */
    constructor(viewPlugin) {
        this.editor = viewPlugin.editor;
        viewPlugin.hooks.renderInterface.subscribe(this, ({ intf, el }) => {
            if (!intf.port) return { intf, el };
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

            if (intf.type && intf.port) {
                // eslint-disable-next-line prefer-destructuring
                const type = intf.type;
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
     * Function that reads all interface styles from metadata and creates an object
     * for each style.
     *
     * The read interface types are stored in `styles` object which is returned by
     * this function
     * @param {*} metadata metadata containing information about styling
     * @returns read interface types
     */
    readInterfaceStyles(metadata) {
        this.styles = {};
        if (!metadata?.styles) return this.styles;
        Object.entries(metadata?.styles).filter(([_, style]) => style?.interfaces)
            .forEach(([styleName, style]) => {
                const io = style.interfaces;
                this.styles[styleName] = { ...io };
                Object.values(io).forEach((e) => {
                    if (!io[e]) return;
                    if (!validateCssObject(io[e])) {
                        throw new Error(`invalid CSS provided in interface-specific styling:\n${getCssString(io[e])}`);
                    }
                });
            });
        return this.styles;
    }
}
