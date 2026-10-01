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
        <path d="M4.5 11.5V15.5H8.5V16.5H3.5V11.5H4.5Z" :class="hoverStatus" />
        <path d="M16.5 11.5V16.5H11.5V15.5H15.5V11.5H16.5Z" :class="hoverStatus" />
        <path d="M8.5 3.5V4.5H4.5V8.5H3.5V3.5H8.5Z" :class="hoverStatus" />
        <path d="M16.5 3.5V8.5H15.5V4.5H11.5V3.5H16.5Z" :class="hoverStatus" />
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
.normal {
    stroke: #ffffff;
}

.hovered {
    stroke: $green;
}
</style>
