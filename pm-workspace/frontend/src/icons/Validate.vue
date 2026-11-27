<!--
Copyright (c) 2022-2023 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!-- eslint-disable max-len -->
<template>
    <svg width="22" height="22" viewBox="0 0 22 22" fill="none" xmlns="http://www.w3.org/2000/svg" class="validate">
        <path
            fill-rule="evenodd"
            clip-rule="evenodd"
            d="M21 9.65333L18.6092 8.80333C18.4442 8.23083 18.2158 7.685 17.9333 7.17417L19.0233 4.88167L17.1192 2.9775L14.8267 4.06667C14.3158 3.78417 13.7692 3.55583 13.1975 3.39083L12.3467 1H9.65333L8.8025 3.39083C8.23083 3.55583 7.68417 3.78333 7.17333 4.06667L4.88083 2.97667L2.97667 4.88083L4.06667 7.17417C3.78417 7.685 3.555 8.23083 3.39083 8.80333L1 9.65333V12.3467L3.39083 13.1967C3.555 13.7692 3.78417 14.315 4.06667 14.8258L2.97667 17.1183L4.88083 19.0233L7.17333 17.9333C7.68417 18.2158 8.23083 18.4442 8.8025 18.6092L9.65333 21H12.3467L13.1975 18.6092C13.77 18.4442 14.3158 18.2158 14.8267 17.9333L17.1192 19.0233L19.0233 17.1183L17.9333 14.8258C18.2158 14.315 18.4442 13.7692 18.6092 13.1967L21 12.3467V9.65333ZM9.96774 15.9859L16.7856 9.16803L15.2143 7.59668L9.96774 12.8432L7.34118 10.2166L5.76983 11.788L9.96774 15.9859Z"
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
