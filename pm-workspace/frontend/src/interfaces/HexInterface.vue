<!--
Copyright (c) 2022-2024 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<template>
    <div>
        <div class="__label">
            {{ intf.name }}:
        </div>
        <input
            ref="el"
            v-model="v"
            type="text"
            class="baklava-input hex-input"
            :class="{ '--invalid': invalid }"
            :placeholder="intf.name"
            :title="intf.name"
            @keypress="handleRestrictedKeys"
            @blur="handleBlur"
            tabindex="-1"
            :disabled="intf.readonly"
        />
    </div>
</template>

<script>
import { computed, defineComponent, ref } from 'vue';

const restrictedKeys = /[^a-f0-9]/i;
const swapPrefix = /^(?:(.*)0)?(?:(.*)x)?/;

export default defineComponent({
    props: {
        intf: {
            required: true,
        },
        modelValue: {
            type: String,
            required: true,
        },
    },
    emits: ['update:modelValue'],
    setup(props, { emit }) {
        const el = ref(null);

        const handleRestrictedKeys = (event) => {
            if (event.key.length === 1 && restrictedKeys.test(event.key)) {
                event.preventDefault();
            }
        };

        const v = computed({
            get: () => props.modelValue,
            set: (val) => {
                let validatedVal;
                let cursorPosition;
                if (!val.startsWith('0x') && swapPrefix.test(val)) {
                    const match = val.match(swapPrefix);
                    validatedVal = val.replace(swapPrefix, '0x$1$2');
                    const additionalLength = ((match[1] ?? '').length) + ((match[2] ?? '').length);
                    cursorPosition = 2;
                    if (additionalLength > 0) cursorPosition += additionalLength;
                    else if (props.modelValue.length > validatedVal.length
                            || props.modelValue === '') {
                        cursorPosition += 1;
                    }
                } else {
                    validatedVal = val;
                }
                if (validatedVal === '0x') {
                    validatedVal = '';
                }
                emit('update:modelValue', validatedVal);
                el.value.value = props.intf.value;
                if (cursorPosition !== undefined) {
                    el.value.setSelectionRange(cursorPosition, cursorPosition);
                }
            },
        });

        // eslint-disable-next-line no-self-compare
        const toBN = (x) => ((x === undefined) ? NaN : BigInt(x));

        const invalid = computed(() => {
            let value;
            try {
                value = toBN(props.intf.value);
            } catch (SyntaxError) {
                return true;
            }
            return value < toBN(props.intf.min) || value > toBN(props.intf.max);
        });

        const handleBlur = () => {
            let value;
            try {
                value = toBN(props.intf.value);
            } catch (SyntaxError) {
                emit('update:modelValue', props.modelValue.toLowerCase());
                return;
            }
            if (value > toBN(props.intf.max)) {
                emit('update:modelValue', `0x${toBN(props.intf.max).toString(16)}`);
            } else if (value < toBN(props.intf.min)) {
                emit('update:modelValue', `0x${toBN(props.intf.min).toString(16)}`);
            } else {
                emit('update:modelValue', props.modelValue.toLowerCase());
            }
        };

        return {
            v, el, handleRestrictedKeys, handleBlur, invalid,
        };
    },
});
</script>

<style lang="scss">
.hex-input {
  text-align: right;
}
</style>
