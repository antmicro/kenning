<!--
Copyright (c) 2025 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!--
Popup menu for configuring node interfaces.
-->

<!-- eslint-disable vue/no-mutating-props -->
<template>
    <div class="create-menu">
        <div v-for="option in configurationOptions" :key="option.id">
            <component
                v-if="option.componentName === 'InputInterface'"
                :is="option.component"
                :intf="option"
                v-model="newInterface[option.configurationVModel as keyof CurrentInterface]"
                class="__name-option"
            />
            <component
                v-else
                :is="option.component"
                :intf="option"
                class="__name-option"
            />
        </div>
        <component :is="addInterfaceMenu.component" :intf="addInterfaceMenu" />
    </div>
</template>

<script lang="ts">
import {
    defineComponent, computed, ref, Ref,
} from 'vue';
import {
    ButtonInterface,
    SelectInterface,
} from '@baklavajs/renderer-vue'; // eslint-disable-line object-curly-newline

import { addInterface } from '../../../core/nodeCreation/Configuration.ts';
import { menuState, configurationState } from '../../../core/nodeCreation/ConfigurationState.ts';

import InputInterface from '../../../interfaces/InputInterface.js';
import IntegerInterface from '../../../interfaces/IntegerInterface.js';
import newInputInterface from './utils.ts';
import EditorManager from '../../../core/EditorManager.js';
import NotificationHandler from '../../../core/notifications.js';
import { parseInterfaces } from '../../../core/interfaceParser.js';

interface CurrentInterface {
    name: string,
    type: string,
    side: Ref<string>,
    direction: Ref<string>,
    maxConnectionsCount: Ref<number>,
}

interface InterfaceInterface extends InputInterface {
    componentName: string,
    configurationVModel?: keyof CurrentInterface
}

export default defineComponent({
    setup() {
        const newInterface: CurrentInterface = {
            name: 'New interface',
            type: '',
            side: ref('left'),
            direction: ref('inout'),
            maxConnectionsCount: ref(0),
        };

        const close = () => {
            menuState.interfaceMenu = false;
        };

        const interfaceName = newInputInterface<InterfaceInterface>('Interface name', newInterface.name, 'name');
        const interfaceType = newInputInterface<InterfaceInterface>('Interface type', newInterface.type, 'type');

        const interfaceSide = computed(() => {
            const option: any = new SelectInterface(
                'Interface side',
                newInterface.side.value,
                ['left', 'right', 'top', 'bottom'],
            ).setPort(false);

            option.events.setValue.subscribe(this, (v: string) => {
                newInterface.side.value = v;
            });

            option.componentName = 'SelectInterface';
            return option as InterfaceInterface;
        });

        const interfaceDirection = computed(() => {
            const option: any = new SelectInterface(
                'Interface direction',
                newInterface.direction.value,
                ['input', 'output', 'inout'],
            ).setPort(false);

            option.events.setValue.subscribe(this, (v: string) => {
                newInterface.direction.value = v;
            });

            option.componentName = 'SelectInterface';
            return option as InterfaceInterface;
        });

        const interfaceConnectionCount = computed(() => {
            const option: any = new IntegerInterface(
                'Max connection count',
                newInterface.maxConnectionsCount.value,
                -1,
                100,
            ).setPort(false);

            option.events.setValue.subscribe(this, (v: number) => {
                newInterface.maxConnectionsCount.value = v;
            });

            option.componentName = 'IntegerInterface';
            return option as InterfaceInterface;
        });

        const validateInterface = () => {
            const editorManager = EditorManager.getEditorManagerInstance();
            if (newInterface.type === '') {
                const intf = {
                    name: newInterface.name,
                    direction: newInterface.direction.value,
                    side: newInterface.side.value,
                    maxConnectionsCount: newInterface.maxConnectionsCount.value,
                };
                const results = editorManager.validateNodeInterface(intf);
                if (results.length > 0) {
                    return results;
                }
                const interfaces = [...configurationState.interfaces, intf];

                const parsedInterfaces = parseInterfaces(interfaces, [], []);
                if (Array.isArray(parsedInterfaces) && parsedInterfaces.length > 0) {
                    return parsedInterfaces;
                }

                return [];
            }
            const typesList = newInterface.type.split(',');

            const intf = {
                name: newInterface.name,
                type: typesList.length === 1 ? newInterface.type : typesList,
                direction: newInterface.direction.value,
                maxConnectionsCount: newInterface.maxConnectionsCount.value,
            };
            const results = editorManager.validateNodeInterface(intf);
            if (results.length > 0) {
                return results;
            }
            const interfaces = [...configurationState.interfaces, intf];

            const parsedInterfaces = parseInterfaces(interfaces, [], []);
            if (Array.isArray(parsedInterfaces) && parsedInterfaces.length > 0) {
                return parsedInterfaces;
            }

            return [];
        };

        const addNewInterface = () => {
            if (newInterface.type === '') {
                const intf = {
                    name: newInterface.name,
                    direction: newInterface.direction.value,
                    side: newInterface.side.value,
                    maxConnectionsCount: newInterface.maxConnectionsCount.value,
                };
                return addInterface(intf);
            }
            const typesList = newInterface.type.split(',');

            const intf = {
                name: newInterface.name,
                type: typesList.length === 1 ? newInterface.type : typesList,
                direction: newInterface.direction.value,
                maxConnectionsCount: newInterface.maxConnectionsCount.value,
            };
            return addInterface(intf);
        };

        let cleanAddInterface = () => {};

        const onKeyDown = (event: KeyboardEvent) => {
            if (event.key === 'Escape') {
                cleanAddInterface();
            }
        };

        const waitForMousePosition = (event: MouseEvent) => {
            // right or middle button click cancel it
            if (event.button !== 0) {
                cleanAddInterface();
                return;
            }
            const editorManager = EditorManager.getEditorManagerInstance();

            const x = event.clientX;
            const y = event.clientY;

            const infX = x - (configurationState?.nodeRect?.x ?? 0);
            const infY = y - (configurationState?.nodeRect?.y ?? 0);

            const nodeWidth = configurationState?.nodeRect?.width ?? 1;
            const nodeHeight = configurationState?.nodeRect?.height ?? 1;

            const nodeName = configurationState.nodeData.name;

            // get node style
            const nodeTypeStyle = (editorManager.editor.nodeTypes.get(nodeName) as any)?.style;
            const nodeStyle = editorManager.editor.getNodeStyle(nodeTypeStyle);
            const result = addNewInterface();
            // Check whether interface has been added
            if (result.length === 0) {
                if (nodeStyle?.positions === undefined) {
                    nodeStyle.positions = {};
                }
                // Add a style for new interface
                nodeStyle.positions[newInterface.name] = {
                    x: Math.max(Math.min((infX / nodeWidth) * 100.0, 100.0), 0),
                    y: Math.max(Math.min((infY / nodeHeight) * 100.0, 100.0), 0),
                };
                editorManager.updateNodeStyle(nodeTypeStyle, nodeStyle);
            }
            cleanAddInterface();
        };

        cleanAddInterface = () => {
            menuState.addingPositionedInterface = false;
            window.removeEventListener('keydown', onKeyDown);
            window.removeEventListener('mousedown', waitForMousePosition);
        };

        const addInterfaceMenu = computed(() => {
            const button: any = new ButtonInterface('Add interface', () => {
                // Check for custom shape
                if (configurationState.nodeData.isShaped) {
                    close();
                    const result = validateInterface();
                    if (result.length > 0) {
                        NotificationHandler.terminalLog('error', 'Invalid interface', result);
                        return;
                    }
                    NotificationHandler.showToast('info', 'Click on the node to add interface.', null, true);
                    menuState.addingPositionedInterface = true;
                    window.addEventListener('mousedown', waitForMousePosition);
                    window.addEventListener('keydown', onKeyDown);
                    return;
                }
                addNewInterface();
                close();
            });
            button.componentName = 'ButtonInterface';
            return button;
        });

        const configurationOptions = computed(
            () => {
                const options = [
                    interfaceName.value,
                    interfaceType.value,
                    interfaceDirection.value,
                    interfaceConnectionCount.value,
                ];

                if (configurationState.nodeData.isShaped) {
                    options.push(interfaceSide.value);
                }

                return options;
            },
        );

        return {
            configurationOptions,
            addInterfaceMenu,
            newInterface,
        };
    },
});
</script>
