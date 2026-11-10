<!--
Copyright (c) 2026 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!--
Search bar part of navigation top app bar. Lets the user search for and highlight
specific nodes.
-->
<script>
import { ref } from 'vue';
import { useViewModel } from '@baklavajs/renderer-vue';
import Magnifier from '../../icons/Magnifier.vue';

export default {
    model: {
        prop: 'openPanel',
        event: 'update:openPanel',
    },
    emits: ['hoverStart', 'hoverStop', 'onClicked', 'update:openPanel'],
    props: {
        mobileClasses: {
            required: true,
            type: Object,
        },
        hover: {
            required: true,
            type: Boolean,
        },
        nodesearchInputStyles: {
            required: true,
            type: Object,
        },
        openPanel: {
            type: Boolean,
        },
    },
    watch: {
        searchEditorNodesQuery(newValue) {
            const { viewModel } = useViewModel();
            if (newValue === '') {
                viewModel.value.editor.searchQuery = undefined;
                return;
            }
            viewModel.value.editor.searchQuery = newValue.toLowerCase();
        },
    },
    components: {
        Magnifier,
    },
    methods: {
        getRef() {
            return this.$refs.container;
        },
        getInput() {
            return this.$refs.searchbarInput;
        },
    },
    data() {
        const searchEditorNodesQuery = ref('');
        return {
            searchEditorNodesQuery,
        };
    },
};
</script>
<template>
<div ref="container" class="search-section" >
    <div
        :class="['hoverbox', mobileClasses]"
        role="button"
        @click="() => this.$emit('onClicked')"
        @pointerover="() => this.$emit('hoverStart')"
        @pointerleave="() => this.$emit('hoverStop')"
        v-click-outside="() => {
            this.$emit('update:openPanel', (searchEditorNodesQuery != ''));
        }"
    >
        <Magnifier
            :hover="hover"
            class="small_svg"
        />
        <div :class="['tooltip', mobileClasses]">
            <span v-if="!openPanel">Show node search bar</span>
            <span v-else>Hide node search bar</span>
        </div>
    </div>
    <div
        v-show="openPanel"
        :style="nodesearchInputStyles"
        :class="['search-editor-nodes', mobileClasses]"
    >
        <input
            ref="searchbarInput"
            v-model="searchEditorNodesQuery"
            placeholder="Search for nodes"
        />
    </div>
</div>
</template>
<style lang="scss" scoped>
.search-section {
    display: flex;
}
.search-editor-nodes {
    max-width: calc(3.75em * 4);

    & > input {
        width: 90%;
        height: 100%;
        padding: 0 0.5em;

        color: $white;
        border: none;
        background-color: $gray-600;

        &:focus {
            outline: 1px solid $green;
            z-index: 12;
        }

        &::placeholder {
            opacity: 0.5;
        }
    }

    // on smaller screens display search bellow NavBar
    &.compressed-mobile {
        position: absolute;
        top: calc($navbar-height + 1px);
        max-width: 40vw;

        border: 1px solid $gray-500;
        box-sizing: border-box;
    }
}
@import './simple_toggle_style.scss'
</style>
