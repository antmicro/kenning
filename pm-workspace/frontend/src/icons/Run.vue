<!--
Copyright (c) 2022-2023 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!-- eslint-disable max-len -->
<template>
    <svg width="22" height="22" viewBox="0 0 22 22" fill="none" xmlns="http://www.w3.org/2000/svg" class="run">
        <path
            d="M4 18.1192V3.29872C4 2.27117 5.10801 1.65193 5.94778 2.21014L17.1651 9.66657C18.2783 10.4065 18.2783 12.0796 17.1651 12.8196L6.82085 19.6957C5.60466 20.5041 4 19.6073 4 18.1192Z"
            fill="#6F6F6F"
            :class="hoverStatus"
            class="highlighted"
            stroke-width="0"
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
/* Color */
.normal {
    fill: $white;
}

.hovered {
    fill: $green;
}
</style>
