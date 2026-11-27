<!--
Copyright (c) 2022-2024 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<template>
    <div>
        <div class="__label">
            {{ intf.name }}:
        </div>
        <input
            v-model.lazy="v"
            type="text"
            class="baklava-input"
            :placeholder="intf.name"
            :title="intf.name"
            tabindex="-1"
            :disabled="intf.readonly"
        />
    </div>
</template>

<script>
import { defineComponent, computed } from 'vue';

export default defineComponent({
    props: {
        intf: {
            required: true,
        },
        modelValue: {
            required: true,
        },
    },
    emits: ['update:modelValue'],
    setup(props, { emit }) {
        const dtype = props.intf.dtype || 'string';

        const v = computed({
            /**
             * Function used to parse the value of the option so that it can be displayed.
             * Example ["1", "2"] is parsed into "1 2".
             * If there is no value then an empty string is returned.
             */
            get: () => {
                if (props.modelValue !== undefined) {
                    return props.modelValue.join(' ');
                }
                return '';
            },
            /**
             * Gets called when the user changes value of the input.
             * It parses the value by trimming whitespaces, splitting the value.
             * into elements and converts them into a desired dtype.
             *
             * Then this value is passed to the parent component so when the editor.
             * is saved this parsed value is used for this list input.
             *
             * Example: '  3 2 5' with dtype string is parsed into ['3', '2', '5'].
             *
             * @param v value of the new input.
             *
             */
            set: (val) => {
                const split = val.trim().split(/\s+/);
                const parsed = split.map((e) => {
                    switch (dtype) {
                        case 'string':
                            return e.toString();
                        case 'integer':
                            return parseInt(e, 10);
                        case 'number':
                            return parseFloat(e);
                        case 'boolean':
                            return Boolean(e);
                        default:
                            return e;
                    }
                });
                emit('update:modelValue', parsed);
            },
        });

        return { v };
    },
});
</script>
