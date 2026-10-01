<!--
Copyright (c) 2026 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!--
Defines the component responsible for displaying status of external application.
-->

<script>
import Backend from '../../icons/Backend.vue';

export default {
    emits: ['hoverStart', 'hoverStop', 'onClicked', 'onClickOutside'],
    props: {
        mobileClasses: {
            required: true,
            type: Object,
        },
        externalApp: {
            required: true,
            type: Object,
        },
        hover: {
            required: true,
            type: Boolean,
        },
        openPanel: {
            required: true,
            type: Boolean,
        },
    },
    components: {
        Backend,
    },
    methods: {
        getRef() {
            return this.$refs.backend;
        },
    },
};
</script>
<!-- eslint-disable vue/no-multiple-template-root -->
<template>
    <div
        ref="backend"
        :class="['hoverbox', mobileClasses]"
        v-if="externalApp.available && externalApp.backend"
        @click="() => this.$emit('onClicked')"
        @pointerover="() => this.$emit('hoverStart')"
        @pointerleave="() => this.$emit('hoverStop')"
    >
        <Backend
            v-if="externalApp.connected"
            color="connected"
            class="small_svg"
            :active="openPanel"
            :hover="hover"
        />
        <Backend
            v-else color="disconnected"
            class="small_svg"
            :active="openPanel"
            :hover="hover"
        />
        <div :class="['tooltip', mobileClasses]">
            <span>External App status</span>
        </div>
        <div
            v-click-outside="(ev) => this.$emit('onClickOutside', ev)"
            class="external-app-status"
        >
            <div>
                <span>Client status:</span>
                <!-- eslint-disable-next-line max-len -->
                <span v-if="externalApp.connected"
                    class="connected"
                    >Connected</span
                >
                <span v-else class="disconnected">Disconnected</span>
            </div>
        </div>
    </div>
</template>
<style lang="scss" scoped>
.external-app-status {
    user-select: none;
    position: absolute;
    flex-direction: column;
    top: 100%;
    left: 0;
    display: flex;
    background-color: #181818;
    border: 2px solid #737373;
    width: 220px;
    /* Hide external app panel and position it
      to right border of backend icon
    */
    transform: translate(-89%, -180px);
    padding: $spacing-l;
    font-size: $fs-small;
    justify-content: space-between;
    border: none;
    transition: transform 1s;

    & > div {
        display: flex;
        justify-content: space-between;

        & > .disconnected {
            color: $red;
        }

        & > .connected {
            color: $green;
        }
    }
}

@import './simple_toggle_style.scss'
</style>
