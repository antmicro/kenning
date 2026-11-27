<!--
Copyright (c) 2022-2023 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<template>
    <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
        <circle
            :class="hoverStatus"
            id="magnifier-ellipse"
            cx="8.94758"
            cy="8.93587"
            r="5.85598"
            stroke="#E5E5E5"
            stroke-width="2.28477"
        />
        <rect
            :class="hoverStatus"
            id="magnifier-rectangle"
            x="13.0214"
            y="13.7031"
            width="0.969325"
            height="5.11707"
            rx="0.484663"
            transform="rotate(-45 13.0214 13.7031)"
            stroke="#E5E5E5"
            stroke-width="0.969325"
        />
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
    fill: $gray-600;
}

.hovered {
    stroke: $green;
    fill: $gray-600;
}
</style>
