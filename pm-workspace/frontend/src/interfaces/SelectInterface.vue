<!--
Copyright (c) 2022-2025 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<template>
    <div>
        <div class="__label">
            {{ intf.name }}:
        </div>
        <div
            ref="el"
            class="baklava-select"
            :class="{ '--open': open }"
            :title="intf.name" @click="open = !open"
        >
            <div class="__selected">
                <div class="__text">
                    {{ selectedText }}
                </div>
                <div class="__icon">
                    <i-arrow />
                </div>
            </div>
            <transition name="slide-fade">
            <div v-show="open" class="__dropdown" :class="{ 'readonly': intf.readonly}">
                    <div class="item --header">
                        {{ intf.name }}
                    </div>
                    <div
                        v-for="(item, i) in intf.items"
                        :key="i"
                        :class="['item', { '--active': item === selectedItem }]"
                        @click="setSelected(item)"
                    >
                        {{ typeof item === "string" ? item : item.text }}
                    </div>
                </div>
            </transition>
        </div>
    </div>
</template>

<script>
import { defineComponent } from 'vue';
import { SelectInterfaceComponent } from '@baklavajs/renderer-vue';

export default defineComponent({
    extends: SelectInterfaceComponent,
    setup(props) {
        const interfaceComponent = SelectInterfaceComponent.setup(props);
        const setSelected = (item) => {
            if (!props.intf.readonly) {
                // eslint-disable-next-line vue/no-mutating-props, no-param-reassign
                props.intf.value = typeof item === 'string' ? item : item.value;
            }
        };

        return { ...interfaceComponent, setSelected };
    },
});
</script>

<style>
.baklava-select > .__dropdown.readonly > .item:not(.--header):not(.--active):hover {
  background: none;
}
</style>
