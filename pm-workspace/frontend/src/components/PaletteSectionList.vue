<!--
Copyright (c) 2026 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->
<!--
    A list of PaletteSection entries.
-->
<template>
    <div>
        <PaletteEntry
            v-for="entry in props.entries"
            :key="entry.id"
            :entry="entry"
        />
        <transition name="fade">
        <div v-if="props.draggedEntry" class="baklava-dragged-node __dragged" :style="{
                width: 'unset',
                top: `${props.pointer.y}px`,
                left: `${props.pointer.x}px`,
            }">
                <PaletteEntry :entry="props.draggedEntry" />
            </div>
        </transition>
        <CustomContextMenu
            v-model="showContextMenu"
            :x="props.contextMenuPosition.x"
            :y="props.contextMenuPosition.y"
            :items="props.contextMenuEntry?.computed?.items ?? []"
            :ignore-close="[props.entriesRef]"
            @click="(...args) => props.contextMenuEntry?.data?.onContextMenu?.(...args)"
        />
    </div>
</template>

<script lang="ts" setup generic="T extends IEntryData">

import {
    Ref,
    Reactive,
    ShallowRef,
    ComputedRef,
    defineModel,
} from 'vue';

// eslint-disable-next-line no-unused-vars
import { type IEntry, type IEntryData } from '../core/palette/types';

import PaletteEntry from './PaletteEntry.vue';
import CustomContextMenu from '../custom/ContextMenu.vue';

// eslint-disable-next-line no-undef
type IPaletteEntryDataT = T;
type Entry = IEntry<IPaletteEntryDataT>;

const props = defineProps<{
    entries: Reactive<Entry[]>,
    draggedEntry: ShallowRef<Reactive<Entry> | null | undefined>,
    pointer: ComputedRef<{x: number, y: number}>,
    contextMenuPosition: Ref<{x: number, y: number}>,
    contextMenuEntry: ShallowRef<Reactive<IEntry> | null>,
    entriesRef: Readonly<ShallowRef<HTMLInputElement | null>>,
}>();

const showContextMenu = defineModel('showContextMenu');

</script>

<style lang="scss" scoped>
.palette-section-header {
    padding: 10px;
    text-align: center;
    border-bottom: 1px solid $gray-500;
}

.entries {
    overflow-y: auto;
    overflow-x: hidden;
}

.__dragged {
    z-index: 1000;

    .__link-menu-wrapper, .__child {
        display: none
    }

    .__entry {
        background-color: unset;
        border-width: 0;

        * {
            display: none;
        }

        & > .__entry-content {
            display: flex;
            border-radius: var(--baklava-node-border-radius);

            & > .__title-label {
                display: revert;
            }
        }
    }
}
</style>
