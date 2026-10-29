<!--
Copyright (c) 2022-2024 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!-- eslint-disable max-len -->
<template>
    <svg width="22" height="22" viewBox="0 0 22 22" fill="none" xmlns="http://www.w3.org/2000/svg" class="backend" :class="classes">
        <path
            fill-rule="evenodd"
            clip-rule="evenodd"
            d="M12 18.5C12 19.4002 12.2163 20.2499 12.5997 21H0V14.875H13.3636C12.5146 15.8429 12 17.1114 12 18.5ZM1.81818 19.25L2.60273 16.625H3.76182L2.97727 19.25H1.81818ZM3.69636 19.25L4.48091 16.625H5.64L4.85545 19.25H3.69636ZM5.57545 19.25L6.36 16.625H7.51909L6.73364 19.25H5.57545ZM7.45364 19.25L8.23818 16.625H9.39636L8.61182 19.25H7.45364ZM10.4909 19.25H9.33273L10.1173 16.625H11.2764L10.4909 19.25Z"
            fill="#6F6F6F"
            :class="hoverStatus"
            class="highlighted"
        />
        <path
            d="M8.18182 6.5625L5.85182 4.375L5.15091 5.03212L6.78182 6.5625L5.15091 8.09287L5.85182 8.75L8.18182 6.5625Z"
            fill="#6F6F6F"
            :class="hoverStatus"
            class="highlighted"
        />
        <path
            d="M14.5455 7.875H10V8.75H14.5455V7.875Z"
            fill="#6F6F6F"
            :class="hoverStatus"
        />
        <path
            fill-rule="evenodd"
            clip-rule="evenodd"
            d="M17.5 13C17.902 13 18.2939 13.0431 18.6714 13.125H20V0H0V13.125H16.3286C16.7061 13.0431 17.098 13 17.5 13ZM18.1818 1.75V11.375H1.81818V1.75H18.1818Z"
            fill="#6F6F6F"
            class="highlighted"
            :class="hoverStatus"
        />
        <path
            :class="[color]"
            d="M21 18.5C21 20.433 19.433 22 17.5 22C15.567 22 14 20.433 14 18.5C14 16.567 15.567 15 17.5 15C19.433 15 21 16.567 21 18.5Z"
            fill="#6F6F6F"
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
            default: 'normal',
        },
        active: {
            type: Boolean,
            required: false,
            default: false,
        },
        hover: {
            type: Boolean,
            required: false,
        },
    },
    setup(props) {
        const classes = computed(() => ({
            __active: props.active,
            __inactive: !props.active,
        }));

        const hoverStatus = computed(() => ({
            hovered: props.hover,
            normal: !props.hover,
        }));

        return { classes, hoverStatus };
    },
};
</script>

<style lang="scss" scoped>
/* Color */
.__active {
    > .highlighted {
        fill: $green;
    }

    &:hover > .highlighted {
        fill: #80f2c6;
    }
}

.__inactive {
    > .highlighted, > .normal {
        fill: #FFFFFF;
    }

    &:hover > .highlighted {
        fill: $green;
    }

    > .hovered {
        fill: $green;
    }
}

.connected {
    fill: $green;
}

.disconnected {
    fill: $red;
}

.normal {
    fill: $white;
}
</style>
