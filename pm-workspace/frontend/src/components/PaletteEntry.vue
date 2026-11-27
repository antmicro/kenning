<!--
Copyright (c) 2022-2025 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!--
A single entry representing available node type in the sidebar.
-->
<template>
    <div class='__link-menu-wrapper'
        v-if="linkMenu === entry.id && entry.data.URLs"
        v-click-outside="() => linkMenu = null"
        :style="linkMenuWrapperStyle"
    >
        <LinkMenu :URLs="entry.data.URLs" />
    </div>
    <!-- eslint-disable-next-line vue/no-multiple-template-root -->
    <div v-bind="$attrs" ref="entryWrapperRef">
        <!-- calculation of max depth would require additional traversal of the graph which -->
        <!-- slows down when big specifications are processed -->
        <div
            v-if="entry.show"
            class="__entry"
            :class="{ '--toggled': toggled }"
            @click="toggleChildren"
            :style="{
                top: toggled ? `${depth * 4}em` : undefined,
                'z-index': 20 - depth,
            }"
        >
            <div
                class="__entry-content"
                ref="entryRef"
                :class="{
                    draggable: Boolean(entry.data.onDrag),
                    clickable: Boolean(entry.data.onClick),
                    '--active': Boolean(entry.computed?.active),
                }"
                :style="{
                    paddingLeft: !(expandable || parentEntry) ? '1em' : undefined
                }"
                @click="onContentClick"
                @pointerdown.left.stop="onPointerDown"
                @pointerup.left.stop="onPointerUp"
                @pointerdown.right="() => contextMenuToggle!(entry)"
            >
                <div
                    v-if="expandable || parentEntry"
                    class="indent"
                    :style="{
                        width: entryOffset,
                        'border-right': expandable ? '1px solid #393939' : 'none',
                    }"
                    @click.stop="() => expandable && toggleChildren()"
                    @pointerdown.stop="() => expandable"
                    @pointerup.stop="() => expandable"
                >
                    <Arrow
                        v-if="expandable"
                        class="arrow"
                        scale="small"
                        :rotate="(entry?.computed?.showChildren
                            ?? entry.showChildren) ? 'left' : 'right'"
                    />
                </div>
                <img
                    class="__title-icon"
                    v-if="typeof entry.data.icon === 'string'"
                    :src="getIconSrc(entry.data.icon)"
                    draggable="false"
                />
                <component
                    v-else-if="entry.data.icon !== undefined"
                    v-bind="entry.data.icon.props"
                    class="__title-icon"
                    :class="entry.data.icon.classes"
                    :is="entry.data.icon.component"
                />
                <div
                    v-html="titleHTML"
                    :class="{
                        '__title-label': true,
                        '--context-menu': contextMenuEntry === entry
                    }"
                />
                <div
                    v-if="entry.data.URLs?.length"
                    :class="{
                        __vertical_ellipsis: true,
                    }"
                    role="button"
                    @pointerdown.stop
                    @click.stop="linkMenuToggle"
                >
                    <VerticalEllipsis class="smaller_svg" />
                </div>
            </div>
        </div>
        <div v-if="toggled && isInternal(entry)">
            <PaletteEntry
                v-for="entry in entry.children"
                class="__child"
                :depth="depth + 1"
                :key="entry.id"
                :entry="entry"
                :parent-entry="entryRef || undefined"
                :clickable_arrow="clickable_arrow"
            />
        </div>
    </div>
</template>

<script lang="ts" setup generic="T extends IEntryData">
import {
    computed,
    inject,
    Reactive,
    ref,
    Ref,
    useTemplateRef,
} from 'vue';
import { useViewModel } from '@baklavajs/renderer-vue';
import DOMPurify from 'dompurify';

import { usePointer, watchOnce } from '@vueuse/core';
import LinkMenu from '../custom/LinkMenu.vue';
import VerticalEllipsis from '../icons/VerticalEllipsis.vue';
import {
    type CustomViewModel,
    type IEntry,
    type IEntryData, // eslint-disable-line no-unused-vars
    isInternal,
} from '../core/palette/types';
import Arrow from '../icons/Arrow.vue';

// eslint-disable-next-line no-undef
type EntryData = T
type Entry = IEntry<EntryData>

const props =
// eslint-disable-next-line no-undef
    withDefaults(defineProps<{
        depth?: number,
        entry: Reactive<Entry>,
        parentEntry?: HTMLElement,
        clickable_arrow?: boolean
    }>(), {
        depth: 0,
        clickable_arrow: false,
    });

// Children
const expandable = computed(() => isInternal(props.entry) && props.entry.children.length);
const toggled = computed(() => isInternal(props.entry) &&
    (props.entry?.computed?.showChildren ?? props.entry.showChildren));
const toggleChildren = () => {
    if (props.clickable_arrow) {
        props.entry.data.onClick?.();
    }

    if (!isInternal(props.entry)) return;
    // eslint-disable-next-line vue/no-mutating-props
    props.entry.showChildren = !props.entry.showChildren;
};

// Offset
const entryRef = useTemplateRef('entryRef');
const minPadding = 10;
const entryOffset = computed(() => {
    const parentTitleLabel = props.parentEntry
        ?.getElementsByClassName('indent')[0] as HTMLElement | undefined;

    const width = parentTitleLabel?.offsetWidth;

    return `${(width ?? -minPadding) + minPadding}px`;
});

// Title
const titleHTML = computed(() => DOMPurify.sanitize(
    props.entry.titleAnnotated ?? props.entry.data.title,
));

// Icon

const getIconSrc = (name: string) => {
    const { viewModel } = useViewModel();

    const ret:string = (viewModel.value as CustomViewModel).cache[`./${name}`] ?? name;

    return ret;
};

// Link menu
const paletteScroll = inject<Ref<number>>('palettescroll', 0);
const linkMenu = inject<Ref<string | null>>('linkmenu', null);
const linkMenuWrapperStyle = computed(() => ({ translate: `0px -${paletteScroll!.value.toString()}px` }));
const linkMenuToggle = () => {
    linkMenu!.value = (linkMenu!.value === props.entry.id) ? null : props.entry.id;
};

const pointer = usePointer();
const dragStart = inject<(_: Reactive<Entry>) => void>('dragstart', () => {});
const isDragging = ref(false);
const onPointerDown = () => {
    if (props.entry.data.onClick) {
        // if 'onClick' is defined, we understand whether
        // they click or drag
        isDragging.value = false;
        watchOnce([pointer.x, pointer.y], () => {
            isDragging.value = true;
            if (dragStart) {
                dragStart!(props.entry);
            }
        });
    } else {
        isDragging.value = true;
        if (dragStart) {
            dragStart!(props.entry);
        }
    }
};
const onPointerUp = () => {
    if (isDragging.value) return;
    props.entry.data.onClick?.();
};
const onContentClick = (e: Event) => {
    if (props.entry.data.onClick || props.entry.data.onDrag) {
        e.stopPropagation();
    }
};

// Context menu
const contextMenuEntry = inject<Ref<Reactive<Entry>> | null>('contextMenuEntry', null);
const contextMenuToggle = inject<(_: Reactive<Entry>) => void>('contextMenuToggle', () => {});
</script>

<style lang="scss">
.__entry {
    display: flex;
    align-items: center;

    position: sticky;

    gap: 1em;
    width: auto;
    height: 4em;

    overflow: hidden;

    &.--toggled {
        z-index: 1;
    }

    font-size: $fs-small;

    border: 0;
    border-bottom: 1px solid #393939;

    background-color: $gray-700;
    color: $white;

    cursor: pointer;

    & > .__entry-content {
        display: flex;
        align-items: center;
        flex-grow: 1;

        gap: 1em;
        width: 100%;
        height: 100%;

        font-size: $fs-small;
        color: $white;

        &:hover {
            color: $green
        }

        width: auto;

        border: 0;
        background-color: $gray-700;

        padding-right: 1em;

        & > .indent {
            min-width: 15px;
            height: 100%;
            border: none;
            background-color: transparent;
            display: flex;
            align-items: center;
            & > .arrow {
                display: flex;
                margin-left: auto;
            }
        }

        &.draggable {
            background: var(--baklava-node-color-background);
            cursor: grab;
        }

        &.clickable {
            cursor: pointer;
        }

        &.--active {
            background: var(--baklava-node-title-color-background);
        }

        & > .__title-icon {
            &.cross {
                height: 0.75em;
                width: 1.2em;
            }
            flex-grow: 0;
            padding: 0;
            height: 2em;
            width: 2em;
            pointer-events: none;
        }

        & > .__title-label {
            flex-grow: 1;
            width: max-content;

            &.--context-menu, & > span {
                color: $green;
            }
        }

        & > .__vertical_ellipsis {
            cursor: pointer !important;
        }
    }
}

.__link-menu-wrapper {
    width: 18em;
    position: absolute;
    left: 100%;
    background-color: #181818;
    border-bottom: 1px solid #737373;
    z-index: 9999;

    .__url {
        display: inline;
        text-decoration: none;
        color: inherit;
        width: auto;

        img {
            width: 2em;
            height: 2em;
            display: block;
        }
    }

}
</style>
