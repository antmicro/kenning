<!--
Copyright (c) 2022-2023 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!-- eslint-disable max-len -->
<template>
    <svg width="7" height="7" viewBox="0 0 22 22" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path
            class="indicator"
            :class="[color]"
            d="M 21.961621,11 C 21.961621,17.056769 17.053947,21.966731 11,21.966731 4.9460531,21.966731 0.03837869,17.056769 0.03837869,11 0.03837869,4.9432309 4.9460531,0.03326902 11,0.03326902 17.053947,0.03326902 21.961621,4.9432309 21.961621,11 Z"
            fill="#6f6f6f"
        />
    </svg>
</template>

<script lang="ts">
export default {
    props: {
        color: {
            type: String,
            required: false,
        },
    },
};
</script>

<style lang="scss">
.normal {
    fill: #FFFFFF;
}

.gray {
    fill: $gray-200;
}

.green {
    fill: $green;
}
</style>
