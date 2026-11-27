<!--
Copyright (c) 2024 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!-- eslint-disable max-len -->
<template>
<svg width="27" height="27" viewBox="0 0 27 27" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path
        d="M21.6004 6.47844C21.6004 5.88197 21.1169 5.39844 20.5204 5.39844L6.48039 5.39844C5.88392 5.39844 5.40039 5.88197 5.40039 6.47844V20.5184C5.40039 21.1149 5.88392 21.5984 6.48039 21.5984H20.5204C21.1169 21.5984 21.6004 21.1149 21.6004 20.5184V6.47844Z"
        fill="white"
        :class="[hoverStatus]"
    />
</svg>
</template>

<script lang="ts">
import { computed, defineComponent } from 'vue';

export default defineComponent({
    props: {
        hover: {
            type: Boolean,
            required: false,
        },
    },
    setup(props) {
        const hoverStatus = computed(() => ({
            hovered: props.hover,
            white: !props.hover,
        }));

        return { hoverStatus };
    },
});
</script>

<style lang="scss" scoped>
/* Color */
.white {
    fill: $white;
}

.hovered {
    fill: $green;
}
</style>
