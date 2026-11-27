<!--
Copyright (c) 2026 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->
<!--
    A header of Palette Section.
-->
<template>
    <div
        v-html="titleHTML"
        class="palette-section-header"/>
</template>

<script lang="ts" setup>
import DOMPurify from 'dompurify';
import { computed } from 'vue';

const props = defineProps<{sectionName: string}>();

const titleHTML = computed(() => DOMPurify.sanitize(
    props.sectionName,
));
</script>

<style lang="scss" scoped>
.palette-section-header {
    padding: 10px;
    text-align: center;
    border-bottom: 1px solid $gray-500;
    > :deep(span)
    {
        color: $green;
    }
}
</style>
