/*
 * Copyright (c) 2026 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Module contains graph palette entries constructor.
 */

import { type Graph, useViewModel } from 'baklavajs';
import {
    computed,
    ref,
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

export default function useSpecGraphPalette(
    nameFilterRef: Ref<string>,
): Reactive<IEntry[]> {
    const { viewModel } = useViewModel();
    const editor = viewModel.value.editor as unknown as PipelineManagerEditor;

    const gatherGraphs = (
        graphs: any[],
        showIds: boolean,
        visited = new Set<Graph>(),
    ): IVEntry[] => {
        if (graphs === undefined) {
            return [];
        }

        return graphs.flatMap((graph) => {
            if (visited.has(graph)) return [];
            visited.add(graph);

            const specGraph = graph.spec;

            // Title
            const titleName = [specGraph.name, specGraph.graphNode?.title, 'Anonymous'].find((title) => title && title !== '');
            const titleId = showIds ? `(${specGraph.id})` : '';
            const title = `${titleName} ${titleId}`;

            const computedValue = computed(() => ({
                active: viewModel.value.displayedGraph === specGraph,
            }));

            const baseEntry = {
                id: specGraph.id,
                data: {
                    title,
                    onClick: () => editor.loadPreloadedGraph(graph),
                },
                computed: computedValue,
            };

            return [baseEntry];
        });
    };

    const entries = computed(() => gatherGraphs(
        (viewModel.value.editor as unknown as PipelineManagerEditor).preloadedGraphs,
        Boolean((viewModel.value as CustomViewModel).settings.showIds),
    ));
    return usePalette(entries, nameFilterRef, undefined, true);
}
