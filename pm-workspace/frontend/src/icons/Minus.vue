<!--
Copyright (c) 2025 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<template>
    <svg
        width="20"
        height="20"
        viewBox="0 0 20 20"
        fill="none"
        shape-rendering="crispEdges"
        xmlns="http://www.w3.org/2000/svg"
    >
        <path d="M4,10 H16 z" :class="hoverStatus" />
    </svg>
</template>

<script>
import { computed } from 'vue';

export default {
    props: {
        hover: {
            type: Boolean,
            required: false,
        },
    },
    setup(props) {
        const hoverStatus = computed(() => ({
            hovered: props.hover,
            normal: !props.hover,
        }));

        return { hoverStatus };
    },
};
</script>

<style lang="scss" scoped>
path {
    stroke-width: 2px;
}

.normal {
    stroke: #ffffff;
}

.hovered {
    stroke: $green;
}
</style>
