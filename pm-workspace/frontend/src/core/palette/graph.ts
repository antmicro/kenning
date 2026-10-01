/*
 * Copyright (c) 2022-2025 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Module contains graph palette entries constructor.
 */

import { type Graph, useViewModel } from 'baklavajs';
import {
    computed,
    type Reactive,
    type Ref,
} from 'vue';
import PipelineManagerEditor from '../../custom/Editor';
import {
    type CustomGraph,
    type CustomNodeTypeInformation,
    type CustomViewModel,
    type IEntry,
    type IVEntry,
} from './types';
import usePalette from './base';

export default function useGraphPalette(
    nameFilterRef: Ref<string>, opts: { tree: boolean } = { tree: false },
): Reactive<IEntry[]> {
    const { viewModel } = useViewModel();
    const editor = viewModel.value.editor as unknown as PipelineManagerEditor;

    const gatherGraphs = (
        graphs: Graph[],
        showIds: boolean,
        visited = new Set<Graph>(),
    ): IVEntry[] => graphs.flatMap((graph) => {
        if (visited.has(graph)) return [];
        visited.add(graph);

        const customGraph = graph as CustomGraph;

        // Title
        const titleName = [customGraph.name, customGraph.graphNode?.title, 'Anonymous'].find((title) => title && title !== '');
        const titleId = showIds ? `(${customGraph.id})` : '';
        const title = `${titleName} ${titleId}`;

        const computedValue = computed(() => ({
            active: viewModel.value.displayedGraph === graph,
        }));

        const baseEntry = {
            id: customGraph.id,
            data: {
                title,
                onClick: () => editor.switchToRelatedGraph(customGraph.id, false),
            },
            computed: computedValue,
        };

        if (!opts.tree) return [baseEntry];

        // Add children
        const subgraphs = customGraph.nodes
            .map((node) => (node as unknown as CustomNodeTypeInformation).subgraph)
            .filter((subgraph) => subgraph !== undefined);

        return [{ ...baseEntry, children: gatherGraphs(subgraphs, showIds, visited) }];
    });

    const entries = computed(() => gatherGraphs(
        Array.from(viewModel.value.editor.graphs).filter((g:any) => g?.specGraph === undefined
        || !g.specGraph),
        Boolean((viewModel.value as CustomViewModel).settings.showIds),
    ));
    return usePalette(entries, nameFilterRef, undefined, false);
}
