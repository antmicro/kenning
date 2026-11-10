<!--
Copyright (c) 2022-2026 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!--
Interface that is used to display inputs, outputs and properties of a node.

The custom implementations introduces wrapper functions that prevent the user
from creating and deleting connections or altering nodes' values if the editor is read-only.
-->

<template>
    <transition :name="transition">
        <div
            v-show="modelValue"
            ref="el"
            class="baklava-context-menu"
            :class="classes"
            :style="styles"
        >
            <template v-if="itemsWithHoverProperty.length === 0">
                <div
                    class="readonly item"
                    style="cursor: default"
                >
                    <div class="text"><i>No resources</i></div>
                </div>
            </template>
            <template v-for="(item, index) in itemsWithHoverProperty">
                <div
                    v-if="item.url === undefined"
                    :key="`i-${index}`"
                    class="item"
                    :class="{
                        submenu: !!item.submenu,
                        '--disabled': !!item.disabled,
                        hoverbox: !!item.tooltipMsg
                    }"
                    @click.stop="(!!item.disabled) ? null : onClick(item)"
                    @pointerover="onPointerOver(item)"
                    @pointerleave="onPointerLeave(item)"
                >
                    <div class="icon">
                        <component v-if="item.icon !== undefined" :is="item.icon"></component>
                    </div>
                    <div class="text">{{ item.label }}</div>
                    <div :class="'tooltip'">
                        <span> {{ item.tooltipMsg }}</span>
                    </div>
                </div>
                <a
                    v-else
                    :key="item.name"
                    :href="item.url"
                    class="item"
                    target="_blank"
                    draggable="false"
                    @click.stop="onClick(item)"
                >
                    <div class="icon">
                        <img
                            v-if="getIconPath(item.icon) !== undefined && !iconDisabled(item.icon)"
                            :src="getIconPath(item.icon)"
                            :alt="item.icon"
                            draggable="false"
                            @error="disableIcon(item.icon)"
                        />
                    </div>
                    <div class="text">{{ item.name }}</div>
                </a>
                <hr v-if="item.endSection" :key="`hr-${index}`" />
            </template>
        </div>
    </transition>
</template>

<script>
import {
    defineComponent, onUnmounted, ref, watch, nextTick,
} from 'vue';
import { Components, useViewModel } from '@baklavajs/renderer-vue';

export default defineComponent({
    extends: Components.ContextMenu,
    props: {
        urls: {
            type: Array,
            default: () => [],
        },
        ignoreClose: {
            type: Array,
            default: () => [],
        },
        transition: {
            type: String,
            default: 'slide-fade',
        },
    },
    emits: ['update:modelValue', 'click', 'onpointerover', 'onpointerleave', 'close'],
    setup(props, context) {
        const {
            el,
            styles,
            classes,
            itemsWithHoverProperty,
        } = // eslint-disable-line object-curly-newline
            Components.ContextMenu.setup(props, context);

        const { viewModel } = useViewModel();
        const getIconPath = (name) => viewModel.value.cache[`./${name}`] ?? name;
        const disabledIcons = ref(new Set());

        const disableIcon = (icon) => {
            disabledIcons.value.add(icon);
        };
        const iconDisabled = (icon) => disabledIcons.value.has(icon);

        const opened = ref(false);

        const closeContextMenu = (ev) => {
            if (ev === undefined) {
                context.emit('update:modelValue', false);
                opened.value = false;
                return;
            }
            if (!props.modelValue) {
                return;
            }
            if (!opened.value) {
                opened.value = true;
                return;
            }
            if (ev.type === 'wheel') {
                context.emit('update:modelValue', false);
                opened.value = false;
                context.emit('close');
            } else {
                let current = document.elementsFromPoint(ev.clientX, ev.clientY)[0];
                const elements = [];
                while (current) {
                    elements.push(current);
                    current = current.parentNode;
                }

                const hasIgnoredElements = new Set(elements)
                    .intersection(new Set([el.value, ...props.ignoreClose])).size;

                if (!hasIgnoredElements) {
                    context.emit('update:modelValue', false);
                    opened.value = false;
                    context.emit('close');
                }
            }
        };

        const onKeyDown = (ev) => {
            if (ev.key === 'Escape') {
                context.emit('update:modelValue', false);
                opened.value = false;
                context.emit('close');
            }
        };
        /* eslint-disable vue/no-mutating-props, no-param-reassign */
        const moveInsideTheWindow = () => {
            // adjust context menu position based on editor viewport and
            // context menu size
            const width = el.value.offsetWidth;
            const height = el.value.offsetHeight;

            const vw = Math.max(document.documentElement.clientWidth
                || 0, window.innerWidth || 0);
            const vh = Math.max(document.documentElement.clientHeight
                || 0, window.innerHeight || 0);

            let offsetX = 0;
            let offsetY = 0;

            const contextYSpan = props.y + height;

            offsetY = Math.max(0, contextYSpan - vh);
            props.y -= offsetY;

            const contextXSpan = props.x + width;

            offsetX = Math.max(0, contextXSpan - vw);
            props.x -= offsetX;
        };

        watch(() => props.modelValue, (isOpen) => {
            if (isOpen) {
                window.addEventListener('keydown', onKeyDown);
                window.addEventListener('pointerdown', closeContextMenu);
                window.addEventListener('wheel', closeContextMenu);

                nextTick(() => {
                    moveInsideTheWindow();
                });
            } else {
                window.removeEventListener('keydown', onKeyDown);
                window.removeEventListener('pointerdown', closeContextMenu);
                window.removeEventListener('wheel', closeContextMenu);
            }
        });

        onUnmounted(() => {
            window.removeEventListener('keydown', onKeyDown);
            window.removeEventListener('pointerdown', closeContextMenu);
            window.removeEventListener('wheel', closeContextMenu);
        });

        const onClick = (item) => {
            context.emit('click', item.value);
            context.emit('update:modelValue', false);
            opened.value = false;
            context.emit('close');
        };

        const onPointerOver = (item) => {
            if (item.onPointerEmit) {
                context.emit('pointerover', item);
            }
        };

        const onPointerLeave = (item) => {
            if (item.onPointerEmit) {
                context.emit('pointerleave', item);
            }
        };

        return {
            el,
            styles,
            classes,
            itemsWithHoverProperty,
            onClick,
            getIconPath,
            onPointerOver,
            onPointerLeave,
            disableIcon,
            iconDisabled,
            closeContextMenu,
        };
    },
});
</script>

<style lang="scss" scoped>

.dropdown-wrapper {
    user-select: none;
    position: absolute;
    flex-direction: column;
    top: 100%;
    left: 0;
    display: none;
    background-color: #181818;
    border: 2px solid #737373;

    & > div:hover {
        background-color: #2A2A2A;
    }
}

.tooltip {
    @extend .dropdown-wrapper;
    border-radius: 15px;
    background-color: $gray-600;
    border: 1px solid $gray-200;
    padding: $spacing-s;
    top: 0;
    left: 100%;
    transform: translate(10px);
    pointer-events: none;
    white-space: nowrap;
    color: #fff;
    display: none;
}

.hoverbox:hover {
    & > .tooltip {
        display: flex;
        z-index: 11;
    }
}

</style>
