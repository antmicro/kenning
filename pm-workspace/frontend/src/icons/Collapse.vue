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
            d="M4 5H9V7H6V19H20V18H22V21H4V5Z"
            fill="#6F6F6F"
            stroke-width="0"
        />
        <path
            class="highlighted"
            :class="[color, hoverStatus]"
            d="M20.8532 14C20.8532 14.5523 20.4055 15 19.8532 15H10.8532V6C10.8532 5.44771 11.3009 5 11.8532 5C12.4055 5 12.8532 5.44771 12.8532 6V11.7962L20.3376 5.24742C20.7532 4.88374 21.385 4.92586 21.7487 5.3415C22.1123 5.75713 22.0702 6.38889 21.6546 6.75258L14.5147 13H19.8532C20.4055 13 20.8532 13.4477 20.8532 14Z"
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
