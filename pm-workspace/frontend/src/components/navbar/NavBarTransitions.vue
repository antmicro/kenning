<!--
Copyright (c) 2026 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!--
Defines all kinds of transitions for navbar and related components.
-->

<script setup>
import { defineModel, defineComponent } from 'vue';
import {
    NodeConfigurationMenu, PropertyConfigurationMenu, InterfaceConfigurationMenu,
    ListMenu, LayerConfigurationMenu, SaveMenu, ParentMenu, ExportMenu, GroupConfigurationMenu,
} from '../menu';
import { removeInterfaces, removeProperties } from '../../core/nodeCreation/Configuration.ts';
import { menuState, configurationState } from '../../core/nodeCreation/ConfigurationState.ts';
import { exportGraph } from '../saveConfiguration.ts';
import Panel from '../Panel.vue';
import GraphSaveMenu from '../menu/GraphSaveMenu.vue';

const saveMenuShow = defineModel('saveMenuShow');
const exportMenuShow = defineModel('exportMenuShow');
const saveConfiguration = defineModel('saveConfiguration');
const baklavaView = defineModel('baklavaView');
</script>

<script>
export default defineComponent({
    components: {
        GroupConfigurationMenu,
        NodeConfigurationMenu,
        PropertyConfigurationMenu,
        InterfaceConfigurationMenu,
        ListMenu,
        LayerConfigurationMenu,
        SaveMenu,
        ParentMenu,
        ExportMenu,
        Panel,
        GraphSaveMenu,
    },
    data() {
        return {
            menuState,
            configurationState,
            removeInterfaces,
            removeProperties,
        };
    },
});
</script>
<!-- eslint-disable vue/no-multiple-template-root -->
<template>
    <Transition name="fade" @mousedown.self="saveMenuShow = false">
        <Panel v-show="saveMenuShow">
            <ParentMenu
                v-show="saveMenuShow"
                v-model="saveMenuShow"
                :title="'Save configuration'"
            >
                <SaveMenu
                    :saveConfiguration="saveConfiguration"
                    v-model="saveMenuShow"
                />
                <GraphSaveMenu
                    :graphs="baklavaView.editor.graphs"
                    v-show="saveConfiguration.graph || saveConfiguration?.graphName !== undefined"
                />
            </ParentMenu>
        </Panel>
    </Transition>
    <Transition name="fade" @mousedown.self="exportMenuShow = false">
        <Panel v-show="exportMenuShow">
            <ParentMenu
                v-show="exportMenuShow"
                v-model="exportMenuShow"
                :title="'Export graph'"
            >
                <ExportMenu
                    :exportGraph = "exportGraph"
                    v-model="exportMenuShow"
                />
            </ParentMenu>
        </Panel>
    </Transition>
    <Transition name="fade" @mousedown.self="menuState.configurationMenu.visible = false">
        <Panel v-show="menuState.configurationMenu.visible">
            <ParentMenu
                v-if="menuState.configurationMenu.visible"
                v-model="menuState.configurationMenu.visible"
                :title="'Node configuration'"
            >
                <NodeConfigurationMenu/>
            </ParentMenu>
        </Panel>
    </Transition>
    <Transition name="fade" @mousedown.self="menuState.groupMenu = false">
        <Panel v-show="menuState.groupMenu">
            <ParentMenu
                v-if="menuState.groupMenu"
                v-model="menuState.groupMenu"
                :title="'Create group'"
            >
                <GroupConfigurationMenu/>
            </ParentMenu>
        </Panel>
    </Transition>
    <Transition name="fade" @mousedown.self="menuState.propertyMenu = false">
        <Panel v-show="menuState.propertyMenu">
            <ParentMenu
                v-if="menuState.propertyMenu"
                v-model="menuState.propertyMenu"
                :title="'Add property'"
            >
                <PropertyConfigurationMenu/>
            </ParentMenu>
        </Panel>
    </Transition>
    <Transition name="fade" @mousedown.self="menuState.interfaceMenu = false">
        <Panel v-show="menuState.interfaceMenu">
            <ParentMenu
                v-if="menuState.interfaceMenu"
                v-model="menuState.interfaceMenu"
                :title="'Add interface'"
            >
                <InterfaceConfigurationMenu/>
            </ParentMenu>
        </Panel>
    </Transition>
    <Transition name="fade" @mousedown.self="menuState.propertyListMenu = false">
        <Panel v-show="menuState.propertyListMenu">
            <ParentMenu
                v-if="menuState.propertyListMenu"
                v-model="menuState.propertyListMenu"
                :title="'Remove properties'"
            >
                <ListMenu
                    :componentName="'propertyListMenu'"
                    :items=configurationState.properties
                    :disabledItems="configurationState.properties.filter((p) =>
                        p.inherited || p.override)"
                    :disabledReason="'inherited'"
                    :confirmAction="removeProperties"
                    :confirmText="'Remove properties'"
                />
            </ParentMenu>
        </Panel>
    </Transition>
    <Transition name="fade" @mousedown.self="menuState.interfaceListMenu = false">
        <Panel v-show="menuState.interfaceListMenu">
            <ParentMenu
                v-if="menuState.interfaceListMenu"
                v-model="menuState.interfaceListMenu"
                :title="'Remove interfaces'"
            >
                <ListMenu
                    :componentName="'interfaceListMenu'"
                    :items=configurationState.interfaces
                    :disabledItems="configurationState.interfaces.filter((i) =>
                        i.inherited || i.override)"
                    :disabledReason="'inherited'"
                    :confirmAction="removeInterfaces"
                    :confirmText="'Remove interfaces'"
                />
            </ParentMenu>
        </Panel>
    </Transition>
    <Transition name="fade" @mousedown.self="menuState.layerMenu = false">
        <Panel v-show="menuState.layerMenu">
            <ParentMenu
                v-if="menuState.layerMenu"
                v-model="menuState.layerMenu"
                :title="'Set node layer'"
            >
                <LayerConfigurationMenu/>
            </ParentMenu>
        </Panel>
    </Transition>
</template>
<style lang="scss" scoped>
</style>
