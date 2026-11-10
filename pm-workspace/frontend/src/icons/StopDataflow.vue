<!--
Copyright (c) 2022-2024 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!-- eslint-disable max-len -->
<template>
    <svg width="22" height="22" viewBox="0 0 22 22" fill="none" xmlns="http://www.w3.org/2000/svg" class="stopdataflow">
  <path
     fill-rule="evenodd"
     clip-rule="evenodd"
     d="m 11,20.553914 c 5.276436,0 9.553914,-4.277478 9.553914,-9.553914 0,-5.2764865 -4.277478,-9.553914 -9.553914,-9.553914 -5.2764865,0 -9.553914,4.2774275 -9.553914,9.553914 0,5.276436 4.2774275,9.553914 9.553914,9.553914 z M 15.458494,9.089217 H 6.5415069 v 3.821566 h 8.9169871 z"
     fill="#6F6F6F"
     :class="[hoverStatus]"
     class="highlighted"
     stroke-width="0"
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
