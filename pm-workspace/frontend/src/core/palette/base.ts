/*
 * Copyright (c) 2022-2025 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Module contains base palette implementation.
 */

import {
    watch,
    reactive,
    type Ref,
    type Reactive,
    computed,
    ComputedRef,
} from 'vue';
import fuzzysort from 'fuzzysort';
import {
    isInternal,
    isInternalV,
    toInternal,
    toLeaf,
    type IVEntryInternal,
    type IEntry,
    type IEntryData,
    type IVEntry,
} from './types';

export default function usePalette<T extends IEntryData>(
    vEntries: Ref<IVEntry<T>[]>,
    searchValueRef?: Ref<string>,
    comparator?: (a: IVEntry<T> | IEntry<T>, b: IVEntry | IEntry<T>) => number,
    defaultCollapse?: boolean,
    showNotMatchedNodes?: ComputedRef<boolean>,
): Reactive<IEntry<T>[]> {
    const showChildren = (vEntry: IVEntryInternal<T>) => vEntry.showChildren ?? !defaultCollapse;

    type IVEntryPrepared = IVEntry<T> & Fuzzysort.Prepared;

    const collectVEntries = (
        currentVEntries: IVEntry<T>[],
    ): [string, IVEntryPrepared][] => currentVEntries
        // Add 'target' value
        .map((e) => (
            [e.id, { ...e, ...fuzzysort.prepare(e.data.title) }] as [string, IVEntryPrepared]
        ))
        // Collect children
        .concat(currentVEntries.flatMap(
            (e): [string, IVEntryPrepared][] => (isInternalV(e) ? collectVEntries(e.children) : []),
        ));

    const vEntryMap = computed(() => new Map(collectVEntries(vEntries.value)));

    type ComputedType = typeof vEntries.value[number]['computed'];
    type IEntryRef = IEntry<T> & { computed: ComputedType };

    /**
     * Creates new entries from virtual entries.
     */
    const gatherNewEntries = (entries: IVEntry<T>[]): IEntryRef[] => entries
        .map((vEntry) => {
            const base = {
                id: vEntry.id,
                data: vEntry.data,
                computed: vEntry.computed,
                titleAnnotated: undefined,
                show: true,
            };

            if (!isInternalV(vEntry)) return base;
            return {
                ...base,
                showChildren: showChildren(vEntry),
                children: gatherNewEntries(vEntry.children),
            };
        }).sort(comparator);

    /**
     * Adds, modifies, removes, re-orders entries based on virtual entries.
     */
    const entryUpdate = (
        currentVEntries: IVEntry<T>[],
        currentEntries: Reactive<IEntry<T>[]>,
    ) => {
        // Update entries that stay
        const currentVEntryMap = new Map(currentVEntries.map((vEntry) => [vEntry.id, vEntry]));
        const updatedEntriesIds = new Set();
        const updatedEntries = currentEntries
            .flatMap((entry) => {
                // Check if should stay
                const vEntry = currentVEntryMap.get(entry.id);
                if (!vEntry) return [];
                updatedEntriesIds.add(entry.id);

                // Update data
                Object.assign(entry.data, vEntry.data);

                // Update children
                if (isInternalV(vEntry)) {
                    // Add/modify
                    const internalEntry = isInternal(entry)
                        ? entry
                        : toInternal(entry, showChildren(vEntry));

                    entryUpdate((vEntry as IVEntryInternal<T>).children, internalEntry.children);
                } else if (isInternal(entry) && !isInternalV(vEntry)) {
                    // Remove
                    toLeaf(entry);
                }
                return [entry];
            });

        // Create new entries
        const newVEntries = currentVEntries.filter(({ id }) => !updatedEntriesIds.has(id));
        const newEntries = gatherNewEntries(newVEntries);

        currentEntries.splice(0, currentEntries.length, ...updatedEntries, ...reactive(newEntries));
        (currentEntries as IEntry<T>[]).sort(comparator);
    };

    type SearchResult = {
        result: Fuzzysort.Result,
        title: string,
    }

    const applySearch = (
        searchMap: Map<string, SearchResult>,
        currentEntries: Reactive<IEntry<T>[]>,
        threshold: number,
        override = false,
    ): boolean => currentEntries.reduce((acc, entry) => {
        /* eslint-disable no-param-reassign */

        // Retrieve search info
        const {
            result: searchResult,
            title: titleAnnotated,
        } = searchMap.get(entry.data.title) ?? {};
        const matched = !!searchResult && searchResult.score > threshold;

        // Update title
        entry.titleAnnotated = titleAnnotated;

        const shouldShow = matched || override;

        if (!isInternal(entry)) {
            entry.show = shouldShow;
            return acc || shouldShow;
        }

        // Update children
        const someChildren = applySearch(
            searchMap,
            entry.children,
            threshold,
            override || shouldShow,
        );

        const internalShouldShow = shouldShow || someChildren;
        entry.show = internalShouldShow;
        entry.showChildren = internalShouldShow;

        return acc || entry.show;
        /* eslint-enable no-param-reassign */
    }, false);

    const resetSearch = (currentEntries: Reactive<IEntry<T>[]>) => {
        currentEntries.forEach((entry) => {
            /* eslint-disable no-param-reassign */
            entry.titleAnnotated = undefined;

            entry.show = true;
            if (!isInternal(entry)) return;

            const vEntry = vEntryMap.value.get(entry.id) as IVEntryInternal<T>;
            entry.showChildren = showChildren(vEntry);
            resetSearch(entry.children);
            /* eslint-enable no-param-reassign */
        });
    };

    const searchUpdate = (
        searchValue: string,
        entries: Reactive<IEntry<T>[]>,
        threshold = -50,
    ) => {
        if (searchValue.length > 1) {
            const targets = Array.from(vEntryMap.value.values());
            const searchResults = fuzzysort.go(searchValue, targets, { threshold });
            const searchEntries = searchResults.map((result) => [
                (result as unknown as { data: { title: string } }).data.title,
                // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
                { result, title: fuzzysort.highlight(result, '<span>', '</span>')! },
            ] as [string, SearchResult]);
            const showNodes = showNotMatchedNodes !== undefined ? showNotMatchedNodes.value : false;
            applySearch(new Map(searchEntries), entries, threshold, showNodes);
        } else {
            resetSearch(entries);
        }
    };

    const entries = reactive<IEntry<T>[]>([]);
    entryUpdate(vEntries.value, entries);
    watch(vEntries, (vEntriesValue) => entryUpdate(vEntriesValue, entries));
    if (searchValueRef !== undefined) {
        watch(searchValueRef, (searchValue) => searchUpdate(searchValue, entries));
    }

    return entries;
}
