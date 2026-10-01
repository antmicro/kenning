<!--
Copyright (c) 2022-2024 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!--
Interface that is used to display inputs, outputs and properties of a node.

The custom implementations introduces wrapper functions that prevent the user
from creating and deleting connections or altering nodes' values if the editor is read-only.
-->

<template>
    <div
        :id="intf.id"
        ref="el"
        class="baklava-node-interface"
        :class="newClasses(intf)"
    >
        <div
            class="__port-bus"
            :class="{
                '__big-bus': isBigBus(intf),
                greyedout_arrow: highlighted,
            }"
            v-if="intf.port && intf.bus?.type"
            no-drag="true"
            :style="{height: intf.bus?.size.toString() + 'px'}"
            @mouseenter="startHoverWrapper"
            @mouseleave="endHoverWrapper"
            @pointerdown.left="(e) => { onMouseDown(); e.stopPropagation() }"
        >
            <div
                v-if="isExposed && (hovered || editExternalName)"
                :class="{
                    '__port_name_left': intf.side === 'left',
                    '__port_name_top': intf.side == 'top',
                    '__port_name_right': intf.side === 'right',
                    '__port_name_bottom': intf.side == 'bottom'
                }"
            >
                <input
                    v-if="editExternalName"
                    v-model="inputExternalName"
                    ref="externalNameInput"
                    type="text"
                    spellcheck="false"
                    autocomplete="off"
                    class="__port_input"
                    :class="{ '__error': externalNameInputIncorrect }"
                    placeholder="External name"
                    @focusout="externalNameFocusOutCallback"
                    @keydown.enter.exact.stop="(e) => { e.target.blur(); }"
                    @input="externalNameInputCallback"
                    @pointerdown.left.stop="(e) => e.stopPropagation()"
                    @keydown.ctrl.stop="(e) => e.stopPropagation()"
                />
                <span
                    v-else
                    @pointerdown.left.stop="enableExternalNameEdit"
                    @keydown.stop
                >
                    {{ intf.externalName }}
                </span>
            </div>
            <div
                class="__port-bus-stub"
                v-if="tempBusIntfOffset && intf.bus?.size"
                :class="{
                    picked: picked,
                    '__square': isExposed,
                    '--input': intf.side === 'left',
                    '--output': intf.side === 'right',
                }"
            >
                <div
                    class="__port"
                    no-drag="true"
                    :style="{
                        position: 'absolute',
                        top: tempBusIntfOffset + 'px',
                        pointerEvents: 'none',
                    }"
                >
                </div>
            </div>
            <div
                class="__port-bus-stub"
                v-for="stub in intf.bus?.stubs"
                :key="stub.id"
                :id="stub.id"
                :class="newClasses(stub)"
            >
                <div
                    class="__port"
                    no-drag="true"
                    @pointerdown.left.stop="(e) => onStubMouseDown(e, stub.id, stub.offset )"
                    :style="{
                        position: 'absolute',
                        top: stub.offset + 'px',
                    }"
                >
                </div>
            </div>
            <span
                class="__port-bus-name"
            >
                {{ intf.name }}
            </span>
        </div>
        <div
            class="__port"
            v-if="intf.port && !intf.bus?.size"
            @mouseenter="startHoverWrapper"
            @mouseleave="endHoverWrapper"
            @pointerdown.left="onMouseDown"
            no-drag="true"
            :class="{
                greyedout_arrow: highlighted,
                picked: picked,
                '__square': isExposed,
                '__port_centered': positioned
                }"
        >
            <div
                v-if="isExposed && (hovered || editExternalName)"
                :class="{
                    '__port_name_left': intf.side === 'left',
                    '__port_name_top': intf.side == 'top',
                    '__port_name_right': intf.side === 'right',
                    '__port_name_bottom': intf.side == 'bottom'
                }"
            >
                <input
                    v-if="editExternalName"
                    v-model="inputExternalName"
                    ref="externalNameInput"
                    type="text"
                    spellcheck="false"
                    autocomplete="off"
                    class="__port_input"
                    :class="{ '__error': externalNameInputIncorrect }"
                    placeholder="External name"
                    @focusout="externalNameFocusOutCallback"
                    @keydown.enter.exact.stop="(e) => { e.target.blur(); }"
                    @input="externalNameInputCallback"
                    @pointerdown.left.stop="(e) => e.stopPropagation()"
                    @keydown.ctrl.stop="(e) => e.stopPropagation()"
                />
                <span
                    v-else
                    @pointerdown.left.stop="enableExternalNameEdit"
                    @keydown.stop
                >
                    {{ intf.externalName }}
                </span>
            </div>
            <Arrow
                v-if="displayArrow"
                :noninteractable="true"
                color="black"
                scale="big"
                :rotate="arrowRotation"
            />
        </div>

        <div
            no-drag="true"
            class="__port-name"
            @mouseenter="startPropertyHover"
            @mouseleave="endPropertyHover"
            :class="spanClasses"
        >
            <!-- @keydown.stop is added so that events are not bubbled up to the editor -->
            <component
                :is="intf.component"
                v-if="showComponent"
                v-model="intf.value"
                :node="node"
                :intf="intf"
                @keydown.stop
                :tabindex="tabindexValue"
                @click="setValue"
                @input="setValue"
            />
            <span v-else-if="!intf.bus?.size">
                {{ intf.name }}
            </span>
            <div
                v-if="isExposed && (propertyHovered || editExternalName) && !intf.port && !sidebar"
                class="__property_name"
            >
                <input
                    v-if="editExternalName"
                    v-model="inputExternalName"
                    ref="externalNameInput"
                    type="text"
                    spellcheck="false"
                    autocomplete="off"
                    class="__property_input"
                    :class="{ '__error': externalNameInputIncorrect }"
                    placeholder="External name"
                    @focusout="externalNameFocusOutCallback"
                    @keydown.enter.exact.stop="(e) => { e.target.blur(); }"
                    @input="externalNameInputCallback"
                    @pointerdown.left.stop="(e) => e.stopPropagation()"
                    @keydown.ctrl.stop="(e) => e.stopPropagation()"
                />
                <span
                    v-else
                    @pointerdown.left.stop="enableExternalNameEdit"
                    @keydown.stop
                >
                    {{ intf.externalName }}
                </span>
            </div>
        </div>
    </div>
</template>

<script>
import {
    defineComponent, ref, computed, nextTick,
} from 'vue';
import { useElementSize } from '@vueuse/core';
import {
    Components, useViewModel, useGraph, TextInterface,
} from '@baklavajs/renderer-vue';
import Arrow from '../icons/Arrow.vue';
import doubleClick from '../core/doubleClick';
import { ir } from '../core/interfaceRegistry.ts';
import { useTemporaryConnection } from './temporaryConnection';

export default defineComponent({
    extends: Components.NodeInterface,
    props: {
        highlighted: Boolean,
        picked: Boolean,
        positioned: { default: false, required: false },
        switchSides: {},
        toggleGroup: { default: () => {}, required: false },
        updateDynamicInterfaces: { default: () => {}, required: false },
        sidebar: { default: false, required: false },
        tabindexValue: { default: -1, required: false },
    },
    components: {
        Arrow,
    },
    setup(props) {
        const {
            el, isConnected, showComponent, openSidebar,
        } =
            Components.NodeInterface.setup(props);

        const { width } = useElementSize(el);

        const { viewModel } = useViewModel();
        const { graph } = useGraph();
        const { hoveredOver, temporaryConnection } = props.sidebar ?
            { hoveredOver: null, temporaryConnection: null } : useTemporaryConnection();

        const isPositionedInterface = props.intf?.x !== undefined && props.intf?.y !== undefined;

        props.intf.events.beforeSetValue.unsubscribe(props.intf);
        props.intf.events.beforeSetValue.subscribe(props.intf, (value, prevent) => {
            if (viewModel.value.editor.readonly) {
                prevent();
            }

            if (
                (props.intf.min !== undefined && value < props.intf.min) ||
                (props.intf.max !== undefined && value > props.intf.max)
            ) {
                prevent();
            }
        });
        const isBigBus = (intf) => intf.bus?.type === 'twoSided';

        if (props.intf.group) {
            props.toggleGroup(props.intf);
            props.intf.events.setValue.unsubscribe(props.intf);
            props.intf.events.setValue.subscribe(props.intf, () => props.toggleGroup(props.intf));
        }

        if (props.intf.dynamicCounter) {
            props.updateDynamicInterfaces(props.intf);
            props.intf.events.setValue.unsubscribe(props.intf);
            props.intf.events.setValue.subscribe(props.intf, () =>
                props.updateDynamicInterfaces(props.intf),
            );
        }
        const tempBusIntfOffset = ref(undefined);
        const onBusMove = (ev) => {
            if (temporaryConnection.value && props.intf.bus?.type) {
                temporaryConnection.value.updated = ev.offsetY;
                tempBusIntfOffset.value = ev.offsetY;

                const bus = props.intf;
                if (isBigBus(props.intf)) {
                    if (ev.offsetX > width.value / 2) {
                        bus.side = 'right';
                    } else {
                        bus.side = 'left';
                    }
                }
            } else {
                tempBusIntfOffset.value = undefined;
            }
        };
        const mousePosStubDragStart = ref(undefined);
        const movedStubId = ref(undefined);
        const movedStubOffset = ref(undefined);
        const onStubMove = (ev) => {
            if (!viewModel.value.editor.readonly) {
                const stub = props.intf.bus?.stubs.find((s) => s.id === movedStubId.value);
                const curY = ev.clientY;
                // no recalculations on each pan since this is used in callaback
                const dv = (curY - mousePosStubDragStart.value) / graph.value.scaling;
                stub.offset = movedStubOffset.value + dv;
                stub.offset = Math.min(Math.max(stub.offset, 0), props.intf.bus?.size ?? 0);
            }
        };
        const onStubMouseUp = () => {
            if (!viewModel.value.editor.readonly) {
                window.removeEventListener('pointermove', onStubMove);
                window.removeEventListener('pointerup', onStubMouseUp);
                mousePosStubDragStart.value = undefined;
                movedStubId.value = undefined;
                movedStubOffset.value = undefined;
            }
        };
        const stubSideSwitch = doubleClick(700, (id) => {
            if (isBigBus(props.intf)) {
                const stub = props.intf.bus?.stubs.find((s) => s.id === id);
                if (stub.side === 'left') {
                    stub.side = 'right';
                } else {
                    stub.side = 'left';
                }
            }
        });

        const onStubMouseDown = (ev, id, off) => {
            if (viewModel.value.editor.readonly) {
                return;
            }
            window.addEventListener('pointermove', onStubMove);
            window.addEventListener('pointerup', onStubMouseUp);
            mousePosStubDragStart.value = ev.clientY;
            movedStubId.value = id;
            movedStubOffset.value = off;
            stubSideSwitch(id);
        };

        const hovered = ref(false);
        const startHoverWrapper = (ev) => {
            window.addEventListener('pointermove', onBusMove);
            onBusMove(ev);
            hovered.value = true;
            if (!viewModel.value.editor.readonly) {
                hoveredOver(props.intf);
            }
        };

        const endHoverWrapper = () => {
            window.removeEventListener('pointermove', onBusMove);
            tempBusIntfOffset.value = undefined;
            hovered.value = false;
            if (!viewModel.value.editor.readonly) {
                hoveredOver(undefined);
            }
        };

        const propertyHovered = ref(false);
        const startPropertyHover = () => {
            propertyHovered.value = true;
        };

        const endPropertyHover = () => {
            propertyHovered.value = false;
        };

        /* eslint-disable vue/no-mutating-props,no-param-reassign */
        const onMouseDown = doubleClick(700, () => {
            if (!viewModel.value.editor.readonly && !isBigBus(props.intf)) {
                props.switchSides(props.intf);
                hoveredOver(undefined);
            }
        });

        const displayArrow = props.intf.port && props.intf.direction !== 'inout';
        const arrowRotation = computed(() => {
            if (props.intf.direction === 'input') {
                if (props.intf.side === 'left') {
                    return 'right';
                }
                if (props.intf.side === 'right') {
                    return 'down';
                }
                if (props.intf.side === 'top') {
                    return 'left';
                }
                if (props.intf.side === 'bottom') {
                    return 'up';
                }
            }
            if (props.intf.direction === 'output') {
                if (props.intf.side === 'left') {
                    return 'down';
                }
                if (props.intf.side === 'right') {
                    return 'right';
                }
                if (props.intf.side === 'top') {
                    return 'up';
                }
                if (props.intf.side === 'bottom') {
                    return 'left';
                }
            }
            return 'down';
        });

        const newClasses = (intf) => ({
            '--input': intf.side === 'left' && !isBigBus(intf) && !props.positioned,
            '--output': intf.side === 'right' && !isBigBus(intf) && !props.positioned,
            '--connected': intf.connectionCount > 0,
            'baklava-node-interface-positioned': props.positioned,
            __node_interface_positioned: props.positioned,
        });
        const spanClasses = computed(() => ({
            '--top': props.intf.side === 'top' && props.positioned,
            '--bottom': props.intf.side === 'bottom' && props.positioned,
            '--left': props.intf.side === 'left' && props.positioned,
            '--right': props.intf.side === 'right' && props.positioned,
            '--input': props.intf.direction === 'input' && props.positioned,
            '--output': props.intf.direction === 'output' && props.positioned,
            '--inout': props.intf.direction === 'inout' && props.positioned,
        }));

        const isExposed = computed(() =>
            props.intf.externalName !== undefined,
        );

        // External name editing
        const externalNameComponent = new TextInterface('External name', props.intf.externalName).setPort(false);
        externalNameComponent.componentName = 'TextInterface';

        const editExternalName = ref(false);
        const externalNameInputIncorrect = ref(false);
        const externalNameInput = ref(null);
        const externalNames = [];

        const inputExternalName = ref(props.intf.externalName);

        /**
         * Update the list of external names of the interfaces in the current graph.
         */
        const updateExternalNames = () => {
            externalNames.splice(0, externalNames.length);
            externalNames.push(...graph.value.obtainExposedNames());
            externalNames.splice(externalNames.indexOf(props.intf.externalName), 1);
        };

        const enableExternalNameEdit = (e) => {
            // Get the list of external names of the interfaces in the subgraph
            updateExternalNames();

            editExternalName.value = true;
            e.preventDefault();

            // Wait for the next tick to focus the input, so that it is rendered first
            nextTick().then(() => {
                externalNameInput.value.focus();
                externalNameInput.value.select();
                externalNameInputIncorrect.value = graph.value.isIncorrectExternalName(
                    props.intf.externalName,
                    externalNames,
                );
            });
        };

        const externalNameInputCallback = (e) => {
            externalNameInputIncorrect.value = graph.value.isIncorrectExternalName(
                e.target.value,
                externalNames,
            );
        };

        const externalNameFocusOutCallback = () => {
            editExternalName.value = false;
            externalNameInputIncorrect.value = false;
            if (inputExternalName.value === props.intf.externalName) return;

            const newExternalName = graph.value.resolveNewExposedName(inputExternalName.value);
            inputExternalName.value = newExternalName;

            viewModel.value.editor.exposeInterface(
                graph.value.id,
                props.intf,
                newExternalName,
            );
        };

        const setValue = () => {
            if (!props.intf.value) return;
            if (!(isExposed.value || ir.isRegistered(props.intf.id))) return;

            const { editor } = viewModel.value;
            const allNodes = Array.from(editor.graphs).map((g) => g.nodes).flat();
            const allProperties = allNodes.map((n) => Object.entries(n.inputs)).flat()
                .filter(([key, _]) => key.startsWith('property_'));
            const exposedProperties = allProperties.map(([_, value]) => value)
                .filter((prop) => prop.id === props.intf.id);

            // eslint-disable-next-line no-return-assign
            exposedProperties.forEach((prop) => prop.value = props.intf.value);
        };

        return {
            arrowRotation,
            displayArrow,
            editExternalName,
            el,
            enableExternalNameEdit,
            endHoverWrapper,
            externalNameInput,
            externalNameInputIncorrect,
            externalNameInputCallback,
            externalNameFocusOutCallback,
            hovered,
            isConnected,
            inputExternalName,
            newClasses,
            spanClasses,
            onMouseDown,
            openSidebar,
            showComponent,
            isExposed,
            isPositionedInterface,
            startHoverWrapper,
            propertyHovered,
            startPropertyHover,
            endPropertyHover,
            setValue,
            tempBusIntfOffset,
            onStubMouseUp,
            onStubMouseDown,
            isBigBus,
        };
    },
});
</script>
