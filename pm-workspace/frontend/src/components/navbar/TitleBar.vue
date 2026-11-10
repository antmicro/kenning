<!--
Copyright (c) 2026 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!--
Title part of navigation top app bar. Lets the user rename editor title.
-->

<script>
import InputInterface from '../../interfaces/InputInterface.js';

export default {
    model: {
        prop: 'graphName',
        event: 'update:graphName',
    },
    props: {
        mobileClasses: {
            required: true,
            type: Object,
        },
        openPanel: {
            required: true,
            type: Boolean,
        },
        editorManager: {
            required: true,
            type: Object,
        },
        graphName: {
            type: String,
        },
    },
    computed: {
        editorTitle() {
            if (this.graphName === undefined) {
                return this.appName;
            }
            const normalizedGraphName = this.graphName.trim();
            return normalizedGraphName === '' ? this.appName : normalizedGraphName;
        },
        graphId() {
            return this.editorManager.baklavaView.displayedGraph.id;
        },
    },
    methods: {
        getRef() {
            return this.$refs.container;
        },
        setEditTitle() {
            if (this.readonly) return;
            this.editTitle = true;
            this.$nextTick(() => this.$refs.editorTitleInput._.refs.el.focus());
        },
    },
    data() {
        const appName = process.env.VUE_APP_EDITOR_TITLE ?? 'Pipeline Manager';
        const editorTitleInterface = new InputInterface(
            null,
            '',
        );
        editorTitleInterface.setDefaultComponent();
        return {
            appName,
            editorTitleInterface,
            editTitle: false,
        };
    },
};
</script>
<template>
<div ref="container">
    <component
        v-if="editTitle && openPanel"
        ref="editorTitleInput"
        :is="editorTitleInterface.component"
        :intf="editorTitleInterface"
        :class="['editorTitleInput', mobileClasses]"

        :modelValue="graphName"
        @update:modelValue="(newVal) => this.$emit('update:graphName', newVal)"

        v-click-outside="() => { editTitle = false }"
        @keyup.enter="() => { editTitle = false }"
    />
    <span
        v-if="!editTitle && openPanel"
        :class="['editorTitle', mobileClasses]"
        @dblclick="setEditTitle">
            {{ editorTitle }}
    </span>
    <span
        v-if="editorManager.baklavaView.settings.showIds &&
              openPanel"
        :class="['editorTitle', 'graphId', mobileClasses]">
            Graph ID: {{ graphId }}
    </span>
</div>
</template>
<style lang="scss" scoped>
.graphId {
    -webkit-user-select: text;
    -ms-user-select: text;
    user-select: text;
}
.editorTitle {
    width: auto;
    text-wrap: wrap;
    flex-grow: 1;

    cursor: text;
    text-align: center;
    padding: 0 $spacing-s;

    &.compressed-mobile {
        display: none;
    }
}

.editorTitleInput {
    font-size: $fs-small;
    padding: 0 $spacing-s;
    flex-grow: 1;

    &.compressed-mobile {
        display: none;
    }
}
@import './simple_toggle_style.scss'
</style>
