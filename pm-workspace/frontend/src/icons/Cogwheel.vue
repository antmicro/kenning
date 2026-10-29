<!--
Copyright (c) 2022-2023 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!-- eslint-disable max-len -->
<template>
    <svg width="26" height="26" viewBox="0 0 26 26" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path
            d="M25.7593 14.6141V11.1452L22.68 10.0504C22.4674 9.31305 22.1734 8.61003 21.8095 7.9521L23.2134 4.99944L20.7609 2.54695L17.8082 3.94975C17.1503 3.5859 16.4462 3.29182 15.7099 3.0793L14.6141 0H11.1452L10.0493 3.0793C9.31305 3.29182 8.60896 3.58483 7.95102 3.94975L4.99837 2.54587L2.54587 4.99837L3.94975 7.9521C3.5859 8.61003 3.29075 9.31305 3.0793 10.0504L0 11.1452V14.6141L3.0793 15.7089C3.29075 16.4462 3.5859 17.1492 3.94975 17.8072L2.54587 20.7598L4.99837 23.2134L7.95102 21.8095C8.60896 22.1734 9.31305 22.4674 10.0493 22.68L11.1452 25.7593H14.6141L15.7099 22.68C16.4473 22.4674 17.1503 22.1734 17.8082 21.8095L20.7609 23.2134L23.2134 20.7598L21.8095 17.8072C22.1734 17.1492 22.4674 16.4462 22.68 15.7089L25.7593 14.6141ZM12.8796 17.1728C10.5087 17.1728 8.58642 15.2506 8.58642 12.8796C8.58642 10.5087 10.5087 8.58642 12.8796 8.58642C15.2506 8.58642 17.1728 10.5087 17.1728 12.8796C17.1728 15.2506 15.2506 17.1728 12.8796 17.1728Z"
            fill="#6F6F6F"
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
