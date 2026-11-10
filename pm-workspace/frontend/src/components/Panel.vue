<!--
Copyright (c) 2022-2023 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<template>
    <div class="blur-panel">
        <slot />
    </div>
</template>

<script>
import { computed, defineComponent } from 'vue';

export default defineComponent({
    props: {
        blur: {
            type: Boolean,
            default: true,
        },
    },
    setup(props) {
        const backgroundColor = computed(() => (props.blur ? '#{$gray-600}80' : null));
        const backdropFilter = computed(() => (props.blur ? 'blur(10px)' : null));
        const position = computed(() => (props.blur ? 'absolute' : null));
        return { backgroundColor, backdropFilter, position };
    },
});
</script>

<style lang="scss">
.blur-panel {
    background-color: v-bind('backgroundColor');
    backdrop-filter: v-bind('backdropFilter');
    position: v-bind('position');
    padding: $spacing-l;
    color: white;
    left: 0%;
    z-index: 10;
    top: 0%;
    width: 100%;
    height: 100%;
    min-height: fit-content;

    & > .panel {
        display: grid;
        grid-row-gap: $spacing-l;
        user-select: none;
        right: 25%;
        top: 25%;
        width: 50%;
        height: 50%;

        & > div {
            & > .option-label {
                padding-bottom: $spacing-s;
                color: $white;
                font-size: $fs-medium;
            }
        }
    }
}
</style>
