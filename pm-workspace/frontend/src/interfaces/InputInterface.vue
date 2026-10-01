<!--
Copyright (c) 2022-2024 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<template>
    <div>
        <div v-if="intf.name" class="__label">
            {{ intf.name }}:
        </div>
        <input
            ref="el"
            v-model="v"
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
import { computed, defineComponent, ref } from 'vue';

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

        const v = computed({
            get: () => props.modelValue,
            set: (val) => {
                emit('update:modelValue', val);
            },
        });
        return { v, el };
    },
});
</script>

<style lang="scss" scoped>
.baklava-input {
    -webkit-box-sizing: border-box;
    -moz-box-sizing: border-box;
    box-sizing: border-box;
}
</style>
