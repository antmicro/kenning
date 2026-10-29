/*
 * Copyright (c) 2022-2025 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Module contains palette types.
 */

import {
    type Graph,
    type IBaklavaViewModel,
    type INodeTypeInformation,
    type IViewSettings,
} from 'baklavajs';
import { type ComputedRef, type Component } from 'vue';

/**
 * `LinkData` entry
 */
export interface NodeURL {
    icon: string,
    name: string,
    url: string,
}

export type DragCallback = (position: { x: number, y: number }) => void;

export interface IEntryData {
    /** Name of the entry. */
    title: string,
    /** Icon of the entry. */
    icon?: string | { component: Component, props: object, classes?: string[] },
    /** URLs of the entry. */
    URLs?: NodeURL[],

    // Events
    onDrag?: DragCallback,
    onClick?: () => void,
    onContextMenu?: (...args: any) => void,
}

export interface IEntryComputedData {
    active?: boolean,
    showChildren?:boolean,
    items?: object[],
}

/*
 * Virtual entry.
 * This is the entry that is computed from a certain source, e.g. node types or graphs.
 */

/**
 * Virtual leaf entry.
 */
export interface IVEntryLeaf<T extends IEntryData = IEntryData> {
    id: string,
    data: T,
    computed?: ComputedRef<IEntryComputedData>,
}

/**
 * Virtual non-leaf entry.
 */
export interface IVEntryInternal<T extends IEntryData = IEntryData> extends IVEntryLeaf<T> {
    showChildren?: boolean,
    children: IVEntryLeaf<T>[],
}

/**
 * Virtual entry.
 */
export type IVEntry<T extends IEntryData = IEntryData> = IVEntryLeaf<T> | IVEntryInternal<T>;

/*
 * Entry.
 * Usually reactive, contains toggle and search information.
 */

/**
 * Leaf entry.
 */
export interface IEntryLeaf<T extends IEntryData = IEntryData> {
    id: string,
    data: T,
    computed?: IEntryComputedData,
    /** Displayed name. */
    titleAnnotated?: string,
    /** Whether current entry should be displayed. */
    show: boolean,
}

/**
 * Non-leaf entry.
 */
export interface IEntryInternal<T extends IEntryData = IEntryData> extends IEntryLeaf<T> {
    /**
     * Whether children elements should be displayed.
     */
    showChildren: boolean,
    children: (IEntryLeaf<T> | IEntryInternal<T>)[],
}

/**
 * Entry.
 */
export type IEntry<T extends IEntryData = IEntryData> = IEntryLeaf<T> | IEntryInternal<T>;

/*
 * Type utils.
 */

export const isInternalV = <T extends IEntryData>(entry: IVEntry<T>): entry is IVEntryInternal<T> => 'children' in entry;

export const isInternal = <T extends IEntryData>(
    entry: IEntry<T>,
): entry is IEntryInternal<T> => 'children' in entry;

export const toLeaf = <T extends IEntryData>(
    entry: IEntryInternal<T>,
): IEntryLeaf<T> => {
    const typedEntry = entry as Omit<Omit<IEntryInternal<T>, 'showChildren'>, 'children'> & Partial<typeof entry>;
    delete typedEntry.children;
    delete typedEntry.showChildren;
    return typedEntry as IEntryLeaf<T>;
};

export const toInternal = <T extends IEntryData>(
    entry: IEntryLeaf<T>, showChildren: boolean,
): IEntryInternal<T> => {
    const typedEntry = entry as IEntryInternal<T>;
    typedEntry.children = [];
    typedEntry.showChildren = showChildren;
    return typedEntry;
};

/* baklavajs shims */

export interface CustomNodeTypeInformation extends INodeTypeInformation {
    isCategory: boolean,
    subgraph?: Graph,
}

export interface CustomViewSettings extends IViewSettings {
    newNodeType?: boolean,
    newGraphNode?: boolean,
    editableNodeTypes?: boolean,
    showIds?: boolean,
}

export interface CustomViewModel extends IBaklavaViewModel {
    collapseSidebar?: boolean,
    settings: CustomViewSettings,
    cache: Record<string, string>,
}

export interface CustomGraph extends Graph {
    name: string,
    graphNode: CustomNodeTypeInformation,
}
