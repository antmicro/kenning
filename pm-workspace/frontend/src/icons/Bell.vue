<!--
Copyright (c) 2022-2023 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!-- eslint-disable max-len -->
<template>
    <svg width="22" height="22" viewBox="0 0 22 22" fill="none" xmlns="http://www.w3.org/2000/svg" >
        <path
            class="indicator"
            :class="[color]"
            d="M21 18.5C21 20.433 19.433 22 17.5 22C15.567 22 14 20.433 14 18.5C14 16.567 15.567 15 17.5 15C19.433 15 21 16.567 21 18.5Z"
            fill="#6F6F6F"
        />
        <path
            fill-rule="evenodd"
            clip-rule="evenodd"
            d="M12.1665 4.2875C11.6191 3.97583 11.2808 3.39583 11.2816 2.77083V2.76833C11.2824 1.79167 10.4843 1 9.5 1C8.5157 1 7.71755 1.79167 7.71755 2.76833V2.77083C7.7184 3.39667 7.38095 3.97583 6.8327 4.2875C4.89686 5.39036 4.44857 7.74165 4.01527 10.0144C3.56064 12.3989 3.1225 14.697 1 15.3758V16.8333H12.2571C12.796 15.1365 14.1359 13.7964 15.8326 13.2573C15.4233 12.3146 15.2059 11.1749 14.9847 10.0148C14.5513 7.74193 14.1028 5.39043 12.1665 4.2875ZM9.5 1.83333C9.9692 1.83333 10.35 2.2075 10.35 2.66667C10.35 3.12667 9.9692 3.5 9.5 3.5C9.0308 3.5 8.65 3.12667 8.65 2.66667C8.65 2.2075 9.0308 1.83333 9.5 1.83333Z"
            fill="#6F6F6F"
            :class="hoverStatus"
        />
        <path
            d="M6.95 18.5H12C12 18.6346 12.0048 18.7681 12.0143 18.9003C11.8076 20.0573 10.7303 21 9.52465 21C8.1825 21 6.95 19.8317 6.95 18.5Z"
            fill="#6F6F6F"
            :class="hoverStatus"
        />
    </svg>
</template>

<script>
import { computed } from 'vue';

export default {
    props: {
        color: {
            type: String,
            required: false,
        },
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
.normal {
    fill: #FFFFFF;
}

.gray {
    fill: $gray-200;
}

.hovered {
    fill: $green;
}
.green {
    fill: $green;
}

.animate {
    animation: blink 2 cubic-bezier(0.68, -0.55, 0.27, 1.55) 1.5s;
}

@keyframes blink {
    0% {
        fill: $green;
    }
    40% {
        fill: $gold;
    }
    60% {
        fill: $gold;
    }
    100% {
        fill: $green;
    }
}
</style>
