<!--
Copyright (c) 2022-2023 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!-- eslint-disable max-len -->
<template>
    <svg width="21" height="23" viewBox="0 0 21 23" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path
            d="M0.791504 5.42584L10.0009 0L19.2787 5.43356L10.0694 10.8594L0.791504 5.42584Z"
             :class="hoverStatus"
        />
        <path
            d="M10.8307 12.1946L20.0325 6.77649L20.002 17.5664L10.8003 22.9923L10.8307 12.1946Z"
             :class="hoverStatus"
        />
        <path
            d="M0 17.5665L0.0304442 6.76111L9.3083 12.1947L9.27786 23L0 17.5665Z"
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
.normal{
    fill: #FFFFFF;
}

.hovered {
    fill: $green;
}
</style>
