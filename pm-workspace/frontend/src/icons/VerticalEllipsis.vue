<!--
Copyright (c) 2022-2023 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!-- eslint-disable max-len -->
<template>
    <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
        <circle cx="10" cy="2"  r="2" :class="hoverStatus"/>
        <circle cx="10" cy="10" r="2" :class="hoverStatus"/>
        <circle cx="10" cy="18" r="2" :class="hoverStatus"/>
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
.green {
    fill: $green;
}
</style>
