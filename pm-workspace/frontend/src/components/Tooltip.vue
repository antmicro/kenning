<!--
Copyright (c) 2022-2023 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<template>
    <div class="__tooltip" :style="styles">
        {{ text }}
    </div>
</template>

<script>
import { defineComponent, computed } from 'vue';

export default defineComponent({
    props: {
        top: {
            type: Number,
            default: 0,
        },
        left: {
            type: Number,
            default: 0,
        },
        text: {
            type: String,
            required: true,
        },
    },
    setup(props) {
        const styles = computed(() => {
            const s = {};
            s.top = `${props.top}px`;
            s.left = `${props.left}px`;
            return s;
        });

        return { styles };
    },
});
</script>

<style lang="scss">
.__tooltip {
    z-index: 4;
    font-size: $fs-small;
    color: $white;
    position: absolute;
    text-align: center;
    flex-direction: column;
    width: max-content;

    margin: 1em 0;
    padding: 1em;
    background-color: $gray-600;
    border: 1px solid $gray-500;
    border-radius: 15px;

    transform: translateX(-50%);
}
</style>
