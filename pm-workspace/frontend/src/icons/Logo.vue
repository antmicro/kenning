<!--
Copyright (c) 2022-2023 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!-- eslint-disable max-len -->
<template>
    <svg width="22" height="22" viewBox="0 0 22 22" fill="none" xmlns="http://www.w3.org/2000/svg" class="active">
        <path
            d="M15.5181 13.5741C14.0965 13.5741 12.9441 12.4217 12.9441 11.0001C12.9441 9.57847 14.0965 8.42603 15.5181 8.42603C16.9397 8.42603 18.0922 9.57847 18.0922 11.0001C18.0922 12.4217 16.9397 13.5741 15.5181 13.5741Z"
            :class="computeHighlight"
        />
        <path
            d="M7.14384 6.1481C5.72223 6.1481 4.56979 4.99566 4.56979 3.57405C4.56979 2.15244 5.72223 1 7.14384 1C8.56545 1 9.7179 2.15244 9.7179 3.57405C9.7179 4.99566 8.56545 6.1481 7.14384 6.1481Z"
            :class="computeHighlight"
        />
        <path
            d="M7.14384 21.0002C5.72223 21.0002 4.56979 19.8477 4.56979 18.4261C4.56979 17.0045 5.72223 15.8521 7.14384 15.8521C8.56545 15.8521 9.7179 17.0045 9.7179 18.4261C9.7179 19.8477 8.56545 21.0002 7.14384 21.0002Z"
            :class="computeHighlight"
        />
        <line
            y1="-0.499486"
            x2="11.201"
            y2="-0.499486"
            transform="matrix(-0.747629 -0.664117 -0.664117 0.747629 15.5085 11.9131)"
            :class="computeLineHighlight"
        />
        <line
            x1="15.8403"
            y1="10.5492"
            x2="7.46608"
            y2="17.988"
            :class="computeLineHighlight"
        />
    </svg>
</template>

<script>
import { computed } from 'vue';

export default {
    props: {
        active: {
            type: Boolean,
            default: false,
        },
        hover: {
            type: Boolean,
            default: false,
        },
    },
    setup(props) {
        const classes = computed(() => ({
            __active: props.active,
            __inactive: !props.active,
        }));

        const computeHighlight = computed(() => ({
            highlighted: !props.hover,
            hovered: props.hover,
        }));
        const computeLineHighlight = computed(() => ({
            highlighted_line: !props.hover,
            hovered_line: props.hover,
        }));

        return { classes, computeHighlight, computeLineHighlight };
    },
};
</script>

<style lang="scss" scoped>
.active {
    > .highlighted {
        fill: $white;
    }
    > .highlighted_line {
        stroke: $white;
        stroke-width: 0.998972;
    }

    > .hovered {
        fill: $green;
    }
    > .hovered_line {
        stroke: $green;
        stroke-width: 0.998972;
    }
}

</style>
