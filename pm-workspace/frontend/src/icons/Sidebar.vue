<!--
Copyright (c) 2022-2023 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!-- eslint-disable max-len -->
<template>
    <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
        <g clip-path="url(#clip0_1450_188)">
            <path
                d="M9 19V13H11V15H19V17H11V19H9ZM1 17V15H7V17H1ZM5 13V11H1V9H5V7H7V13H5ZM9 11V9H19V11H9ZM13 7V1H15V3H19V5H15V7H13ZM1 5V3H11V5H1Z"
                fill="#E5E5E5"
                :class="hoverStatus"
            />
        </g>
        <defs>
            <clipPath id="clip0_1450_188">
                <rect width="20" height="20" fill="white" />
            </clipPath>
        </defs>
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
.normal{
    fill: #FFFFFF;
}

.hovered {
    fill: $green;
}
</style>
