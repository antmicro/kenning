<!--
Copyright (c) 2025 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!-- eslint-disable max-len -->
<template>
    <svg width="26" height="26" viewBox="0 0 26 26" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path
            d="M19.3209 0L26 5.81598V23H0V0H19.3209ZM2.88889 20.125H5.77778V12.2188H20.2222V20.125H23.1111V7.12012L18.2347 2.875H2.88889V20.125ZM8.66667 20.125H17.3333V15.0938H8.66667V20.125ZM11.5556 5.75C12.3533 5.75 13 6.39359 13 7.1875C13 7.98141 12.3533 8.625 11.5556 8.625H7.22222C6.42448 8.625 5.77778 7.98141 5.77778 7.1875C5.77778 6.39359 6.42448 5.75 7.22222 5.75H11.5556Z"
            :class="hoverStatus"
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
    fill: #ffffff;
}
.hovered {
    fill: $green;
}
</style>
