<!--
Copyright (c) 2026 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!--
Popup menu for choosing which graphs to save.
-->

<template>
    <div>
        <div>Graphs to save:</div>
        <div class="graph_list">
            <PaletteEntry
                v-for="entry in currentEntries"
                :key="entry.id"
                :entry="entry"
                :clickable_arrow="true"
            />
        </div>
    </div>
</template>

<script>
import {
    defineComponent, computed, ref,
} from 'vue';

import PaletteEntry from '../PaletteEntry.vue';
import usePalette from '../../core/palette/base.ts';

export default defineComponent({
    components: {
        PaletteEntry,
    },
    props: {
        graphs: {
            type: Object,
            required: true,
        },
    },
    setup(props) {
        const setToSaveInGraph = (v, graph) => {
            graph.setToSave(v);

            graph.nodes.forEach((node) => {
                if (node.subgraph !== undefined) {
                    setToSaveInGraph(v, node.subgraph);
                }
            });
        };

        const gatherGraphs = (
            graphs,
            visited = new Set(),
        ) => graphs.flatMap((graph) => {
            if (graph.specGraph) return [];
            if (visited.has(graph.id)) return [];
            visited.add(graph.id);

            const titleName = [graph.name, graph.graphNode?.title, 'Anonymous'].find((title) => title && title !== '');
            const title = `${titleName}`;

            const computedValue = computed(() => ({
                active: graph.toSave,
                showChildren: !graph.toSave,
            }));

            const baseEntry = {
                id: graph.id,
                data: {
                    title,
                    onClick: () => {
                        setToSaveInGraph(!graph.toSave, graph);
                    },
                },
                computed: computedValue,
            };

            const subgraphs = graph.nodes
                .map((node) => node.subgraph)
                .filter((subgraph) => subgraph !== undefined);

            return [{ ...baseEntry, children: gatherGraphs(subgraphs, visited) }];
        },
        );

        const entries = computed(() => gatherGraphs(Array.from(props.graphs)));

        const searchRef = ref('');
        const currentEntries = computed(() => usePalette(entries, searchRef, undefined, false));

        return { currentEntries };
    },
});
</script>

<style lang="css">

.graph_list
{
    overflow-y: auto;
    max-height:300px;
    margin-left: 0px;
    margin-right: 25px;
    margin-top: 5px;
    margin-bottom: 10px;
}

</style>
