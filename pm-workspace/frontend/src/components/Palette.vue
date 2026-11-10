<!--
Copyright (c) 2022-2025 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!--
Implements left sidebar containing available nodes and graphs.
-->

<template>
    <div
        ref="paletteRef"
        class="baklava-node-palette export-hidden"
        :class="{'hidden-navbar': $isMobile}"
    >
        <div class="search-bar">
            <div class="palette-title">
                <div
                    v-for="name in Object.values(visibleTabs)"
                    :key="name"
                    :class="['tab', { '--active': name === currentTab }]"
                    @click="() => { currentTab = name }"
                >
                    <span>{{ name }}</span>
                </div>
            </div>
            <div class="__entry_search">
                <Magnifier class="__title-icon" :color="'gray'" />
                <input class="palette-search" v-model="paletteSearch" placeholder="Search" />
            </div>
        </div>
        <PaletteSection
            :entries="currentEntries"
            :palette="paletteRef!"
            :sectionNames="sectionNames"
            :oneNodeList="oneNodeList"
        />
    </div>
</template>

<script lang="ts" setup>
import { useViewModel } from 'baklavajs';
import {
    computed, ref, useTemplateRef,
} from 'vue';
import Magnifier from '../icons/Magnifier.vue';
import PaletteSection from './PaletteSection.vue';
import { useNodePalette, createNodeListPalette } from '../core/palette/node.ts';
import useGraphPalette from '../core/palette/graph.ts';

const { viewModel } = useViewModel();
const nodeLists = computed(
    () => viewModel.value.editor?.nodeLists ?? Map());

const oneNodeList = computed(() => nodeLists.value?.size === 1);

const Tabs = computed(() => ({
    nodes: 'Nodes',
    nodeLists: (oneNodeList.value ? nodeLists.value?.keys().next().value : 'Node Lists'),
    graphsTree: 'Graphs',
}));

const visibilities = computed(() => ({
    nodes: true,
    nodeLists: [...(nodeLists.value?.keys() ?? [])].length !== 0,
    graphsTree: true,
}));

const visibleTabs = computed(() =>
    Object.fromEntries(Object.entries(Tabs.value).filter(([key, _]) => visibilities.value[key])),
);

// update nodeListPalette in reaction to changed specification

const paletteRef = useTemplateRef('paletteRef');
const currentTab = ref(Tabs.value.nodes);
const paletteSearch = ref('');

const sectionNames = computed(() => {
    if (currentTab.value !== Tabs.value.nodeLists
        || nodeLists.value == null
        || nodeLists?.value.values().length === 0) {
        return undefined;
    }
    return [...nodeLists.value.keys()];
});

const nodePalette = useNodePalette(paletteSearch);
const nodeListPalette = createNodeListPalette(paletteSearch, sectionNames);
const graphPalette = useGraphPalette(paletteSearch, { tree: true });

const paletteEntries = computed(() => ({
    [Tabs.value.nodes]: nodePalette,
    [Tabs.value.nodeLists]: nodeListPalette,
    [Tabs.value.graphsTree]: graphPalette,
}));

const currentEntries = computed(() => paletteEntries.value[currentTab.value]);

</script>

<style lang="scss" scoped>
.baklava-node-palette {
    top: 0;
    width: auto;
    min-width: min(300px, 50vw);
    max-width: min(450px, 50vw);
    background: $gray-700;
    user-select: none;
    border-right: 1px solid #393939;

    // default baklavjs height - terminal panel height
    height: calc(100% - $navbar-height - $terminal-container-height);
    padding: 0;
    z-index: 3;

    display: flex;
    flex-direction: column;
    overflow: visible;

    &.hidden-navbar {
        top: $navbar-height;
        transform: translate(-450px, 0px);
    }

    & > .search-bar {
        & > .palette-title {
            display: flex;
            margin: 0;

            border-bottom: 1px solid #393939;

            height: 3em;

            & .tab {
                cursor: pointer;
                display: flex;

                align-items: center;
                justify-content: center;

                height: 100%;
                width: 100%;

                &.--active {
                    background-color: var(--baklava-node-color-background);
                }

                &:not(.--active) {
                    color: var(--baklava-control-color-hover);
                }

                &:hover {
                    color: $green;
                }
            }

        }

        & > .__entry_search {
            display: flex;
            align-items: center;

            .__title-icon {
                margin-left: 10px;
            }

            font-size: $fs-small;

            gap: 1em;
            height: 4em;

            border-bottom: 1px solid #393939;

            background-color: $gray-700;
            cursor: auto;
            padding-right: 0;
            overflow: visible;

            & > .palette-search {
                height: 100%;
                width: 100%;
                color: $white;
                border: none;
                background-color: $gray-700;

                padding-left: 1em;

                &:focus {
                    outline: 1px solid $green;
                }

                &::placeholder {
                    opacity: 0.5;
                }
            }
        }
    }

    h1 {
        margin: 0;
    }
}
</style>
