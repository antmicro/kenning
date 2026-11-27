<!--
Copyright (c) 2024 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!-- eslint-disable max-len -->
<template>
    <svg
        class="expand"
        width="26"
        height="26"
        viewBox="3 3 20 20"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
    >
        <path
            class="highlighted"
            :class="[color, hoverStatus]"
            d="M4 5H10V7H6V19H20V17H22V21H4V5ZM12 6C12 5.44772 12.4477 5 13 5H22V14C22 14.5523 21.5523 15 21 15C20.4477 15 20 14.5523 20 14V8.20377L12.5156 14.7526C12.1 15.1163 11.4682 15.0741 11.1046 14.6585C10.7409 14.2429 10.783 13.6111 11.1986 13.2474L18.3386 7H13C12.4477 7 12 6.55228 12 6Z"
            fill="#6F6F6F"
            stroke-width="0"
        />
    </svg>
</template>

<script lang="ts">
import { computed, defineComponent } from 'vue';

export default defineComponent({
    props: {
        color: {
            type: String,
            required: false,
        },
        hover: {
            type: Boolean,
            default: false,
        },
    },
    setup(props) {
        const hoverStatus = computed(() => ({
            hovered: props.hover,
            normal: !props.hover,
        }));

        return { hoverStatus };
    },
});
</script>

<style lang="scss" scoped>
.hovered {
    fill: $green;
}
</style>
