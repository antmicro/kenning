<!--
Copyright (c) 2022-2023 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!--
The entrypoint of the application.
-->

<template>
    <div>
        <LoadingScreen v-if="loading" />
        <div id="container">
            <NavBar />
            <Editor
                class="inner-editor"
                :view-model="editorManager.baklavaView"
                @setLoad="handleLoad"
                :loading="loading"
            />
            <TerminalPanel v-show="!hideHud || terminalStore.show" />
        </div>
    </div>
</template>

<script>
import { ref, computed } from 'vue';
import NavBar from './NavBar.vue';
import EditorManager from '../core/EditorManager.js';
import TerminalPanel from './TerminalPanel.vue';
import LoadingScreen from './LoadingScreen.vue';
import Editor from '../custom/Editor.vue';
import '@baklavajs/themes/dist/classic.css';
import { terminalStore } from '../core/stores.js';

export default {
    components: {
        NavBar,
        Editor,
        TerminalPanel,
        LoadingScreen,
    },
    setup() {
        const editorManager = EditorManager.getEditorManagerInstance();
        const loading = ref(false);

        const cache = {};
        // Importing all assets to a cache so that they can be accessed dynamically during runtime
        function importAll(r) {
            r.keys().forEach((key) => (cache[key] = r(key))); // eslint-disable-line no-return-assign,max-len
        }
        try {
            importAll(require.context('../../assets', true, /\.(svg|png|jpg|jpeg|gif|webp|avif)$/));
        } catch (e) {
            // assets directory not found
        } finally {
            editorManager.baklavaView.cache = cache;
        }

        const hideHud = computed(() => editorManager.baklavaView.editor.hideHud);

        const handleLoad = (value) => {
            loading.value = value;
        };

        return {
            editorManager,
            hideHud,
            handleLoad,
            loading,
            terminalStore,
        };
    },
};
</script>

<style lang="scss">
</style>
