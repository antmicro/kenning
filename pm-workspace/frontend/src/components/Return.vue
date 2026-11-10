<!--
Copyright (c) 2025 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!--
This component is used to return from subgraph (specifically for preview mode).
It emits 'click' event when the button is clicked.
-->

<template>
    <div
        role="button"
        class="return-control export-hidden"
        @pointerover="isHovered = true"
        @pointerleave="isHovered = false"
        @click="$emit('click')"
    >
        <Arrow rotate="down" :hover="isHovered" color="white" class="small_svg" />
    </div>
</template>

<script>
import Arrow from '../icons/Arrow.vue';

export default {
    components: {
        Arrow,
    },

    emits: ['click'],

    data() {
        return {
            isHovered: false,
        };
    },
};
</script>

<style lang="scss" scoped>
.return-control {
    width: 44px;
    height: 44px;
    display: flex;
    justify-content: center;
    align-items: center;
    cursor: pointer;
    background-color: $gray-600;
    border: 1px solid $gray-500;
    border-radius: 100%;
    position: fixed;
    left: 30px;
    top: 30px;
    z-index: 100;
}
</style>
