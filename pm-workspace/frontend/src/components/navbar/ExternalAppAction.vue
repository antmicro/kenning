<!--
Copyright (c) 2026 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!--
Defines the component responsible for displaying external application actions.
-->

<script>
import { defineComponent } from 'vue';
import CassetteStop from '../../icons/CassetteStop.vue';

export default defineComponent({
    components: {
        CassetteStop,
    },
    props: {
        mobileClasses: {
            required: true,
            type: Object,
        },
        navbarItems: {
            required: true,
            type: Array,
        },
        activeNavbarItems: {
            required: true,
            type: Array,
        },
        isInProgress: {
            required: true,
            type: Function,
        },
        isStoppable: {
            required: true,
            type: Function,
        },
        requestDataflowAction: {
            required: true,
            type: Function,
        },
        updateHoverInfo: {
            required: true,
            type: Function,
        },
        resetHoverInfo: {
            required: true,
            type: Function,
        },
        getNavbarActionTooltip: {
            required: true,
            type: Function,
        },
        isHovered: {
            required: true,
            type: Function,
        },
    },
});
</script>
<!-- eslint-disable vue/no-multiple-template-root -->
<template>
    <div
        v-for="actionItem in navbarItems"
        :key="actionItem.name"
        :id="`navbar-button-${actionItem.procedureName}`"
        :class="[
            (
                (activeNavbarItems.includes(actionItem.procedureName)
                || isInProgress(actionItem.procedureName))
                ? 'hoverbox' : 'box'
            ),
            mobileClasses, {
            'button-in-progress': isInProgress(actionItem.procedureName),
        }]"
        role="button"
        @click="() => requestDataflowAction(actionItem)"
        @pointerover="() => updateHoverInfo(actionItem.procedureName, true)"
        @pointerleave="() => resetHoverInfo(actionItem.procedureName)"
    >
        <CassetteStop
            v-if="
                isStoppable(actionItem.procedureName)
                && isInProgress(actionItem.procedureName)
            "
            class="small_svg"
            :hover="isHovered(actionItem.procedureName)"
        />
        <component
            v-else
            class="small_svg"
            :is="actionItem.icon"
            :hover="isHovered(actionItem.procedureName)"
            :imgURI="actionItem.iconName"
        />
        <div class="progress-bar" />
        <div :class="['tooltip', mobileClasses]">
            <span>
                {{ getNavbarActionTooltip(actionItem) }}
            </span>
        </div>
    </div>
</template>
<style lang="scss" scoped>
.progress-bar {
    position: absolute;
    height: calc(60px * 0.2);
    left: 0;
    bottom: 0;
    border-radius: 3px;
    z-index: 5;
    background-color: $green;

    &.animate {
        animation: pulse ease-in-out 2s infinite;

        @keyframes pulse {
            0% {
                left: 0;
                width: 0;
            }
            50% {
                width: 100%;
                left: 0;
            }
            100% {
                left: 100%;
                width: 0%;
            }
        }
    }
}
@import './simple_toggle_style.scss'
</style>
