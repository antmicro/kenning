/*
 * Copyright (c) 2025 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { reactive } from 'vue';

export type NodeDataConfiguration = {
    name: string,
    category: string,
    layer?: string,
    description?: string,
    color?: string,
    isLayerInherited?: boolean,
    isCategoryInherited?: boolean,
    isShaped?: boolean
}

export type PropertyConfiguration = {
    name: string,
    type: string,
    default: string | number | boolean | null,
    min?: number,
    max?: number,
    inherited?: boolean,
    values?:any,
    step?: number,
    readonly?: boolean,
    dtype?: any,
    group?: string
};

export type InterfaceConfiguration = {
    name: string,
    type?: string | string[],
    side?: string,
    direction: string,
    maxConnectionsCount?: number,
    inherited?: boolean,
    array?: []
}

export type NodeRect = {
    x: number,
    y: number,
    width: number,
    height: number
};

export type ConfigurationState = {
    editedType?: string,
    nodeData: NodeDataConfiguration,
    properties: PropertyConfiguration[],
    interfaces: InterfaceConfiguration[],
    nodeRect?: NodeRect
    success?: boolean,
    pill?: object,
    extends?: any,
    subgraphId?: string,
}

/**
 * Configuration state for the configuration menu.
 * This state is reactive and can be accessed and modified from any component.
 * It is used to control the visibility of the configuration menu and its options.
 */
export const menuState = reactive({
    configurationMenu: {
        visible: false,
        addNode: false,
        placeNode: false,
        duplicateNode: false,
    },
    groupMenu: false,
    propertyMenu: false,
    interfaceMenu: false,
    propertyListMenu: false,
    interfaceListMenu: false,
    addingPositionedInterface: false,
    layerMenu: false,
});

/**
 * Configuration state for the node data.
 * It is used to store the node data that is being configured.
 */
export const configurationState: ConfigurationState = reactive({
    nodeData: {
        name: 'Custom Node',
        category: 'Default category',
        layer: '',
        description: '',
        color: '',

    },
    properties: [],
    interfaces: [],
    success: false,
});

/*
 * Event bus for the YAML node editor.
 * It is used to check whether the editor state has changed.
 */
export const editorEventBus = new EventTarget();
