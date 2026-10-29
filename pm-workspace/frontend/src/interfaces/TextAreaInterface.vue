<!--
Copyright (c) 2022-2025 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<template>
    <div>
        <div class="__label">
            {{ intf.name }}:
        </div>
        <textarea
            ref="el"
            v-if="editMode"
            v-model="v"
            type="text"
            class="baklava-input"
            :placeholder="intf.name"
            :title="intf.name"
            tabindex="-1"
            :style="styles"
            v-click-outside="leaveEditMode"
            @input="handleInput"
        />
        <div
            v-else
            class="__markdown-content baklava-input"
            style="overflow-y: auto"
            @dblclick="enterEditMode"
        >
            <span v-html="renderedMarkdown" />
        </div>
    </div>
</template>

<script>
import {
    computed, defineComponent, ref, reactive, nextTick,
} from 'vue';
import showdown from 'showdown';
import { useResizeObserver } from '@vueuse/core';
import { useViewModel } from '@baklavajs/renderer-vue';

export default defineComponent({
    props: {
        intf: {
            required: true,
        },
        modelValue: {
            type: String,
            required: true,
        },
    },
    emits: ['update:modelValue'],
    setup(props, { emit }) {
        const el = ref(null);
        const editMode = ref(false);
        const previousScroll = ref(0);
        const resizeHeight = 200;
        const { viewModel } = useViewModel();

        const v = computed({
            get: () => props.modelValue,
            set: (val) => {
                emit('update:modelValue', val);
            },
        });

        const converter = new showdown.Converter({
            smartIndentationFix: true,
            simpleLineBreaks: true,
        });

        const renderedMarkdown = computed(() => {
            let html = converter.makeHtml(props.modelValue);
            const aTagRe = /<a href="[a-zA-Z0-9-$_.+!*'()/&?=:%]+">/gm;
            html.match(aTagRe)?.forEach((match) => {
                const hrefParts = match.split('"');
                // Forces the link to open in a new tab instead of closing the pipeline manager
                const newEnd = ` tabindex="-1" target="_blank"${hrefParts[2]}`;
                const newHref = [hrefParts[0], hrefParts[1], newEnd].join('"');
                html = html.replace(match, newHref);
            });
            return html;
        });

        const styles = reactive({
            resize: 'none',
            overflow: 'hidden',
            minHeight: 'none',
            height: 'auto',
        });

        const leaveEditMode = () => {
            if (props.modelValue.trim()) {
                editMode.value = false;
                previousScroll.value = el.value.scrollTop;
            }
        };

        const getContentHeight = () => {
            // Use styles directly to calculate `scrollHeight` correctly
            const previousMinHeight = el.value.style['min-height'];
            const previousHeight = el.value.style.height;

            el.value.style['min-height'] = '0px';
            el.value.style.height = 'auto';
            const contentHeight = el.value.scrollHeight;

            el.value.style['min-height'] = previousMinHeight;
            el.value.style.height = previousHeight;
            styles.height = previousHeight;
            return contentHeight;
        };

        const handleInput = () => {
            const contentHeight = getContentHeight();
            const allowResize = contentHeight > resizeHeight;
            if (allowResize) {
                styles.minHeight = `${resizeHeight}px`;
                styles.resize = 'vertical';
                styles.overflow = 'auto';
            } else {
                styles.minHeight = '0px';
                styles.height = `${contentHeight}px`;
                styles.resize = 'none';
                styles.overflow = 'hidden';
            }
        };

        const enterEditMode = async () => {
            if (props.intf.readonly || viewModel.value.editor.readonly) return;

            editMode.value = true;
            await nextTick();

            // Restore scroll
            el.value.scrollTop = previousScroll.value;

            handleInput();
        };

        // Don't render if the initial value is empty
        nextTick().then(() => { if (!props.modelValue.trim()) enterEditMode(); });

        useResizeObserver(el, (_) => {
            styles.height = el.value.style.height;
        });

        return {
            el,
            v,
            editMode,
            renderedMarkdown,
            enterEditMode,
            leaveEditMode,
            handleInput,
            styles,
            viewModel,
        };
    },
});
</script>
