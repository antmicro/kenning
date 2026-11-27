<!--
Copyright (c) 2022-2023 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!--
Badge that is displayed next to the notification.
Can be either an exclamation mark or an x mark
-->

<template>
    <div :class="type">
        <ExclamationMark v-if="type === 'info'" rotate="flip" />
        <ExclamationMark v-if="type === 'warning'" />
        <Cross v-if="type === 'error'" color="white" />
    </div>
</template>

<script>
import ExclamationMark from '../icons/ExclamationMark.vue';
import Cross from '../icons/Cross.vue';

export default {
    components: {
        ExclamationMark,
        Cross,
    },
    props: {
        type: {
            type: String,
            required: true,
        },
    },
};
</script>

<style lang="scss" scoped>
div {
    width: 35px;
    height: 35px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
}

.info {
    background-color: $green;
}

.warning {
    background-color: $gold;
}

.error {
    background-color: $red;
}
</style>
