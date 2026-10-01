/*
 * Copyright (c) 2022-2025 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Module contains node palette entries constructor.
 */

import {
    computed,
    type ComputedRef,
    type Reactive,
    type Ref,
    watch,
    reactive,
    ref,
} from 'vue';
import {
    AbstractNode,
    IBaklavaViewModel,
    useViewModel,
} from 'baklavajs';
import jq from '@michaelhomer/jqjs';
import fuzzysort from 'fuzzysort';
import Cross from '../../icons/Cross.vue';
import PipelineManagerEditor from '../../custom/Editor';
import {
    type CustomNodeTypeInformation,
    type CustomViewModel,
    type DragCallback,
    type IEntry,
    type IEntryComputedData,
    type IEntryData,
    type IVEntry,
    type NodeURL,
} from './types';
import usePalette from './base';
import EditorManager, {
    DEFAULT_CUSTOM_NODE_CATEGORY,
    DEFAULT_CUSTOM_NODE_TYPE,
    DEFAULT_GRAPH_NODE_CATEGORY,
    DEFAULT_GRAPH_NODE_TYPE,
} from '../EditorManager';
import { configurationState, menuState } from '../nodeCreation/ConfigurationState';
import Bin from '../../icons/Bin.vue';
import { prepareNodeForDuplication } from '../nodeCreation/Configuration';

function placeNode(viewModel: IBaklavaViewModel, instance: AbstractNode, x: number, y: number) {
    type IAbstractNode = {
        position: { x: number; y: number };
    }
    /* eslint-disable no-param-reassign */
    (instance as unknown as IAbstractNode).position = { x, y };

    viewModel.displayedGraph.addNode(instance);
}

type IEntryDataWithNodeType = IEntryData & {
    nodeType: CustomNodeTypeInformation,
    onDrag: DragCallback,
};
type IEntryDataNode = IEntryData | IEntryDataWithNodeType;

function onDragPlaceNode(viewModel: IBaklavaViewModel): DragCallback {
    return function onDrag(this: IEntryDataWithNodeType, { x, y }) {
        const instance = new this.nodeType.type(); // eslint-disable-line new-cap,max-len
        placeNode(viewModel, instance, x, y);
    };
}

interface CategoryTree {
    [key: string]: CategoryTree;
}

const getSpecialEntries = (
    viewModel: IBaklavaViewModel,
): IVEntry<IEntryDataNode>[] => {
    const specialEntries: IVEntry<IEntryDataNode>[] = [];
    const specialEntriesIcon = { component: Cross, props: { rotate: 45, color: 'white' }, classes: ['cross'] };
    const editorManager = EditorManager.getEditorManagerInstance();

    if ((viewModel as CustomViewModel).settings.editableNodeTypes &&
        (viewModel as CustomViewModel).settings.newNodeType) {
        specialEntries.push({
            id: DEFAULT_CUSTOM_NODE_TYPE,
            data: {
                title: DEFAULT_CUSTOM_NODE_TYPE,
                icon: specialEntriesIcon,
                onDrag: ({ x, y }) => {
                    // TODO: create a dedicatd async function for this

                    menuState.configurationMenu.visible = !menuState.configurationMenu.visible;
                    menuState.configurationMenu.addNode = true;
                    menuState.configurationMenu.placeNode = true;

                    const unwatch = watch(
                        () => menuState.configurationMenu.visible, async (newValue, oldValue) => {
                            const closed = oldValue === true && newValue === false;
                            if (closed) {
                                if (menuState.configurationMenu.placeNode) {
                                    /* eslint-disable max-len */
                                    if (menuState.configurationMenu.addNode) {
                                        menuState.configurationMenu.addNode = false;
                                        if (configurationState.success) {
                                            const newType = configurationState.nodeData.name;
                                            const nodeInformation = viewModel.editor.nodeTypes.get(newType);
                                            // eslint-disable-next-line @typescript-eslint/no-non-null-assertion, new-cap
                                            const instance = new nodeInformation!.type();
                                            placeNode(viewModel, instance, x, y);
                                        }
                                    }
                                    /* eslint-enable max-len */
                                }
                                unwatch();
                            }
                        },
                    );
                },
            },
        });
    }

    if ((viewModel as CustomViewModel).settings.newGraphNode
        && editorManager.isSpecificationLoaded()) {
        specialEntries.push({
            id: DEFAULT_GRAPH_NODE_TYPE,
            data: {
                title: DEFAULT_GRAPH_NODE_TYPE,
                icon: specialEntriesIcon,
                nodeType: viewModel.editor.nodeTypes.get(
                    DEFAULT_GRAPH_NODE_TYPE,
                ) as CustomNodeTypeInformation,
                onDrag: onDragPlaceNode(viewModel),
            },
        });
    }

    return specialEntries;
};

/**
 * Creates a tree of categories based on the passed names of categories.
 *
 * @param Set of categories names of the nodes
 * @returns Tree of parsed categories that represents categories and their subcategories structure
 */
const parseCategories = (categoriesNames: Set<string>) => {
    const categoryTree: CategoryTree = {};
    const toParse: Record<string, Array<string>> = {};

    categoriesNames.forEach((c) => {
        const [newLabel, ...rest] = c.split(/(?<!\\)\//);
        const label = newLabel;
        if (!Object.keys(toParse).includes(label)) {
            toParse[label] = [];
        }
        if (rest.length > 0) {
            toParse[label].push(rest.join('/'));
        }
    });

    Object.entries(toParse).forEach(([label, rest]) => {
        categoryTree[label] = parseCategories(new Set(rest));
    });

    return categoryTree;
};

/**
 * Uses parsed categories structure to create a full tree of nodes based on their categories
 * which can be used to render then in a node palette.
 *
 * @param categoryTree Parsed tree returned by `parseCategories` function. It is used to divide
 * nodes into subcategories.
 * @param nodes list of nodes that has information about their types and icons created
 * in `getNodeTree` function.
 * @param prefix represents category path of a node.
 * @returns Parsed structure where every node has two keys: `nodes`, which represents nodes of
 * a given category and `subcategories` which represent subcategories. Subcategories are
 * also of this type.
 */
const categorizeNodes = (
    categoryTree: CategoryTree,
    nodeTypes: [CustomNodeTypeInformation, IEntryDataNode, ComputedRef<IEntryComputedData>][],
    prefix = '',
): IVEntry<IEntryDataNode>[] => {
    const entries = Object.entries(categoryTree)
        .flatMap(([category, subcategories]) => {
            if ([DEFAULT_CUSTOM_NODE_CATEGORY, DEFAULT_GRAPH_NODE_CATEGORY].includes(category)) {
                return [];
            }

            const currentCategory = prefix === ''
                ? category
                : `${prefix}/${category}`;

            const currentNodeTypes = nodeTypes
                .filter(([nodeType]) => nodeType.category === currentCategory);

            // Category node
            const categoryNodes = currentNodeTypes.filter(([nodeType]) => nodeType.isCategory);
            const [categoryNodeType] = categoryNodes as [typeof nodeTypes[number] | undefined];

            // Non-category entries
            const nonCategoryEntries = currentNodeTypes
                .filter(([nodeType]) => !nodeType.isCategory)
                .map(([, data, computedValue]) => ({
                    id: data.title,
                    data,
                    computed: computedValue,
                } as Omit<IVEntry<IEntryDataNode>, 'computed'> & { computed?: ComputedRef }));

            // Category entries
            const categoryEntries = categorizeNodes(subcategories, nodeTypes, currentCategory);

            const children = nonCategoryEntries.concat(categoryEntries);

            // Flatten empty category
            if (category === '') return children;

            return [{
                id: category,
                data: categoryNodeType?.[1] ?? { title: category },
                computed: categoryNodeType?.[2],
                children,
            }];
        });

    return entries;
};

const getNodeEntries = (
    viewModel: IBaklavaViewModel,
): IVEntry<IEntryDataNode>[] => {
    const editorManager = EditorManager.getEditorManagerInstance();
    const editor = viewModel.editor as unknown as PipelineManagerEditor;
    const isNotAbstractNode = (entry: CustomNodeTypeInformation) => !editorManager
        .specification
        .currentSpecification
        .nodes
        .some((node: any) => node.abstract && node.name === entry.title);
    const nodeTypes = (Array.from(editor.nodeTypes.values()) as CustomNodeTypeInformation[])
        .filter(isNotAbstractNode);
    const categoryNames = new Set(nodeTypes.map(({ category }) => category));
    const categoryTree = parseCategories(categoryNames);

    // Add node data
    const nodeTypesWithData = nodeTypes
        .map((nodeType) => ([
            nodeType,
            {
                title: nodeType.title,
                icon: editor.getNodeIconPath(nodeType.title) as string,
                URLs: editor.getNodeURLs(nodeType.title) as NodeURL[],
                nodeType,
                onDrag: onDragPlaceNode(viewModel),
                onContextMenu(action: string) {
                    switch (action) {
                        case 'duplicate':
                            prepareNodeForDuplication(nodeType.title);
                            menuState.configurationMenu.visible = true;
                            menuState.configurationMenu.addNode = false;
                            menuState.configurationMenu.duplicateNode = true;
                            configurationState.nodeData.name += ' (copy)';
                            break;
                        case 'delete':
                            editorManager.removeNodeType(nodeType.title);
                            break;
                        default:
                            break;
                    }
                },
            },
            computed(() => {
                const itemToDisable = () => !(viewModel.editor as unknown as PipelineManagerEditor)
                    .additionalNodeTypes.has(nodeType.title);

                return {
                    items: [
                        {
                            value: 'delete',
                            label: 'Delete',
                            icon: Bin,
                            disabled: itemToDisable(),
                            onPointerEmit: itemToDisable(),
                            tooltipMsg: itemToDisable() && 'Node type can\'t be deleted',
                        },
                        {
                            value: 'duplicate',
                            label: 'Duplicate node type',
                        },
                    ],
                };
            }),
        ] as [typeof nodeType, IEntryDataWithNodeType, ComputedRef<IEntryComputedData>]));

    const specialEntries = getSpecialEntries(viewModel);
    const regularNodeEntries = categorizeNodes(categoryTree, nodeTypesWithData);
    return specialEntries.concat(regularNodeEntries);
};

const getNodeListEntries = (
    viewModel: IBaklavaViewModel,
): Map<string, IVEntry<IEntryDataNode>[]> => {
    const editorManager = EditorManager.getEditorManagerInstance();
    const editor = viewModel.editor as unknown as PipelineManagerEditor;
    const isNotAbstractNode = (entry: CustomNodeTypeInformation) => !editorManager
        .specification
        .currentSpecification
        .nodes
        .some((node: any) => node.abstract && node.name === entry.title);
    const nodeLists = editorManager.specification.currentSpecification?.metadata?.nodeLists;

    const result = new Map(Object.entries(nodeLists ?? {}).map(([sectionName, queryText]) => {
        const query = jq.compile(queryText);

        const queriedNodesData = editorManager
            .specification
            .currentSpecification
            ?.nodes === undefined ?
            undefined :
            Array.from(
                query(editorManager.specification.currentSpecification?.nodes),
            )[0];

        const nodeTypes = (Array.from(editor.nodeTypes.values()) as CustomNodeTypeInformation[])
            .filter(isNotAbstractNode)
            .filter((nodeType) => {
                if (!Array.isArray(queriedNodesData)) return true;
                return queriedNodesData.some((nodeData) => {
                    if (
                        nodeData == null ||
                        typeof nodeData !== 'object' ||
                        Array.isArray(nodeData) ||
                        !('category' in nodeData) ||
                        !('name' in nodeData)
                    ) return false;
                    const { category, name } = nodeData;
                    return category === nodeType.category && name === nodeType.title;
                });
            });

        const categoryNames = new Set(nodeTypes.map(({ category }) => category));
        const categoryTree = parseCategories(categoryNames);

        // Add node data
        const nodeTypesWithData = nodeTypes
            .map((nodeType) => ([
                nodeType,
                {
                    title: nodeType.title,
                    icon: editor.getNodeIconPath(nodeType.title) as string,
                    URLs: editor.getNodeURLs(nodeType.title) as NodeURL[],
                    nodeType,
                    onDrag: onDragPlaceNode(viewModel),
                    onContextMenu(action: string) {
                        switch (action) {
                            case 'duplicate':
                                prepareNodeForDuplication(nodeType.title);
                                menuState.configurationMenu.visible = true;
                                menuState.configurationMenu.addNode = false;
                                menuState.configurationMenu.duplicateNode = true;
                                configurationState.nodeData.name += ' (copy)';
                                break;
                            case 'delete':
                                editorManager.removeNodeType(nodeType.title);
                                break;
                            default:
                                break;
                        }
                    },
                },
                computed(() => {
                    const itemToDisable = () => !(
                        viewModel.editor as unknown as PipelineManagerEditor
                    ).additionalNodeTypes.has(nodeType.title);

                    return {
                        items: [
                            {
                                value: 'delete',
                                label: 'Delete',
                                icon: Bin,
                                disabled: itemToDisable(),
                                onPointerEmit: itemToDisable(),
                                tooltipMsg: itemToDisable() && 'Node type can\'t be deleted',
                            },
                            {
                                value: 'duplicate',
                                label: 'Duplicate node type',
                            },
                        ],
                    };
                }),
            ] as [typeof nodeType, IEntryDataWithNodeType, ComputedRef<IEntryComputedData>]));

        const regularNodeEntries = categorizeNodes(categoryTree, nodeTypesWithData);

        return [sectionName, regularNodeEntries];
    }));

    return result;
};

const compareNodeEntries = (
    a: IVEntry<IEntryDataNode> | IEntry<IEntryDataNode>,
    b: IVEntry<IEntryDataNode> | IEntry<IEntryDataNode>,
): number =>
    // 1. New node type
    Math.sign(
        Number(b.data.title === DEFAULT_CUSTOM_NODE_TYPE)
        - Number(a.data.title === DEFAULT_CUSTOM_NODE_TYPE))
    // 2. New graph node
    || Math.sign(
        Number(b.data.title === DEFAULT_GRAPH_NODE_TYPE)
        - Number(a.data.title === DEFAULT_GRAPH_NODE_TYPE))
    // 3. Non-category nodes
    || Math.sign(Number('children' in a) - Number('children' in b))
    // 4. Empty category nodes
    || Math.sign(Number('children' in a && a.children.length !== 0) - Number('children' in b && b.children.length !== 0))
    // 5. Sort lexicographically
    || a.data.title.localeCompare(b.data.title);

export function useNodePalette(
    nameFilterRef: Ref<string>,
): Reactive<IEntry<IEntryDataNode>[]> {
    const { viewModel } = useViewModel();
    const entries = computed(() => getNodeEntries(viewModel.value));
    const defaultCollapse = Boolean((viewModel.value as CustomViewModel).collapseSidebar);
    return usePalette(entries, nameFilterRef, compareNodeEntries, defaultCollapse);
}

type SectionMetadata = {
        entries: Reactive<IEntry<IEntryDataNode>[]>,
        title: string,
        target: string,
        show: boolean,
        showAll: boolean
    };

export function createNodeListPalette(
    nameFilterRef: Ref<string>,
    sectionNames: ComputedRef<string[]|undefined>,
): Reactive<Map<string, SectionMetadata>> {
    const { viewModel } = useViewModel();

    const paletteList = reactive(new Map<string, SectionMetadata>());

    const entries = computed(() => getNodeListEntries(viewModel.value));
    const defaultCollapse = Boolean((viewModel.value as CustomViewModel).collapseSidebar);

    watch(sectionNames, () => {
        entries.value.forEach((value, key) => {
            if (!paletteList.has(key)) {
                const entry = usePalette(
                    computed(() => entries.value.get(key) ?? []),
                    nameFilterRef,
                    compareNodeEntries,
                    defaultCollapse,
                    computed(() => paletteList.get(key)?.showAll ?? false),
                );

                paletteList.set(key, {
                    title: key,
                    target: key,
                    show: true,
                    entries: entry,
                    showAll: false,
                },
                );
            }
        });
        const keysToRemove = Array.from(paletteList.keys()).filter((k) => !entries.value.has(k));
        keysToRemove.forEach((key) => {
            paletteList.delete(key);
        });
    });

    const searchUpdate = (searchValue: string, threshold = -50) => {
        if (searchValue.length > 1) {
            paletteList.forEach((res) => {
                res.show = false;
            });

            const searchResults = fuzzysort.go(searchValue, Array.from(paletteList.values()), { threshold, key: 'target' });
            searchResults.forEach((res) => {
                const highlited = fuzzysort.highlight(res, '<span>', '</span>');
                if (highlited !== null) {
                    res.obj.title = highlited;
                    res.obj.show = true;
                }
            });

            // check for search results in nodes list
            paletteList.forEach((res) => {
                if (res.title !== res.target) {
                    res.showAll = true;
                } else {
                    res.showAll = false;
                    const entriesList:any[] = [];

                    const gatherEntries = (entryList:any) => {
                        entryList.forEach((en:any) => {
                            if (en?.children !== undefined) {
                                gatherEntries(en.children);
                            }

                            if (en.titleAnnotated !== undefined) {
                                entriesList.push(en);
                            }
                        });
                    };

                    gatherEntries(res.entries);

                    if (entriesList.length > 0) {
                        res.show = true;
                    }
                }
            });
        } else {
            paletteList.forEach((res) => {
                res.show = true;
                res.title = res.target;
            });
        }
    };

    watch(nameFilterRef, (searchValue) => {
        searchUpdate(searchValue);
    });

    return paletteList;
}
