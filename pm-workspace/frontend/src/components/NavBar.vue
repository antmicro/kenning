<!--
Copyright (c) 2022-2026 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!--
Navigation bar of the application.
Displays user interface and main details about the Pipeline Manager status.
-->

<script>
import {
    markRaw, provide, computed,
} from 'vue';
import FilesContextMenu from './navbar/FilesContextMenu.vue';
import ExternalAppStatus from './navbar/ExternalAppStatus.vue';
import ExternalAppAction from './navbar/ExternalAppAction.vue';
import SubgraphNavigation from './navbar/SubgraphNavigation.vue';
import NavBarTransitions from './navbar/NavBarTransitions.vue';
import NotificationButton from './navbar/NotificationButton.vue';
import FullscreenButton from './navbar/FullscreenButton.vue';
import SettingsButton from './navbar/SettingsButton.vue';
import GraphDetailsButton from './navbar/GraphDetailsButton.vue';
import NodeBrowserButton from './navbar/NodeBrowserButton.vue';
import SearchBar from './navbar/SearchBar.vue';
import TitleBar from './navbar/TitleBar.vue';
import EditorManager from '../core/EditorManager';
import NotificationHandler from '../core/notifications';
import { notificationStore } from '../core/stores';
import runInfo from '../core/communication/runInformation';
import getExternalApplicationManager from '../core/communication/ExternalApplicationManager';
import Notifications from './Notifications.vue';
import Settings from './Settings.vue';
import CustomSidebar from '../custom/CustomSidebar.vue';
import GraphDetails from './GraphDetails.vue';
import { saveSpecificationConfiguration, saveGraphConfiguration, exportGraph } from './saveConfiguration.ts';

import icons from '../icons';

import InputInterface from '../interfaces/InputInterface.js';

/* eslint-disable no-param-reassign */
export default {
    components: {
        Notifications,
        Settings,
        CustomSidebar,
        GraphDetails,
        FilesContextMenu,
        NavBarTransitions,
        ExternalAppAction,
        ExternalAppStatus,
        SubgraphNavigation,
        NotificationButton,
        FullscreenButton,
        SettingsButton,
        GraphDetailsButton,
        NodeBrowserButton,
        SearchBar,
        TitleBar,
    },
    computed: {
        dataflowGraphName() {
            return this.editorManager.editor.graphName;
        },
        preview() {
            const urlParams = new URLSearchParams(window.location.search);
            if (urlParams.has('preview')) {
                return urlParams.get('preview') === 'true';
            }
            return false;
        },
        hideHud() {
            return this.editorManager.editor.hideHud;
        },
        readonly() {
            return this.editorManager.editor.readonly;
        },
        leftButtonsQuantity() {
            return this.$refs.leftButtons.children.length;
        },
        rightButtonsQuantity() {
            return this.$refs.rightButtons.children.length;
        },
        isNavBarCompressed() {
            return (
                this.isMounted &&
                this.windowWidth <
                (this.leftButtonsQuantity + this.rightButtonsQuantity) * this.buttonWidth
                + this.searchbarWidthMultiplcity * this.buttonWidth // searchbar width
                + 0.5 * this.buttonWidth // offset
            );
        },
        mobileClasses() {
            return { 'compressed-mobile': this.isNavBarCompressed };
        },
        nodesearchInputStyles() {
            return {
                width: `${this.searchbarWidthMultiplcity * this.buttonWidth}px`,
            };
        },
        leftContainerStyles() {
            if (this.isNavBarCompressed) {
                return { 'flex-grow': this.leftButtonsQuantity };
            }
            return {};
        },
        rightContainerStyles() {
            if (this.isNavBarCompressed) {
                return {
                    'flex-grow': this.rightButtonsQuantity,
                    'justify-content': 'right',
                };
            }
            return { 'justify-content': 'right' };
        },
        activeNavbarItems() {
            return this.activeNavbarItemsNames;
        },
    },
    watch: {
        dataflowGraphName(newValue) {
            this.graphName = newValue;

            // Resetting the save configurations
            saveSpecificationConfiguration.reset();
            saveGraphConfiguration.reset();
        },
        graphName(newValue) {
            this.editorManager.updateSubgraphName(newValue);

            // Resetting the save configurations
            saveSpecificationConfiguration.reset();
            saveGraphConfiguration.reset();
        },
        navbarItems(newValue) {
            newValue.forEach((item) => {
                // If there is no such icon then assets are checked and used as a fallback
                if (icons[item.iconName] === undefined) {
                    item.icon = markRaw(icons.Placeholder);
                    item.iconName = this.editorManager.baklavaView.cache[`./${item.iconName}`];
                } else {
                    item.icon = markRaw(icons[item.iconName]);
                }
            });
            this.activeNavbarItemsNames = newValue.map((item) => item.procedureName);
        },
    },
    data() {
        const editorManager = EditorManager.getEditorManagerInstance();
        const graphName = editorManager.baklavaView.editor.graphName ?? '';

        const externalApplicationManager = getExternalApplicationManager();
        const navbarItems = computed(() => [
            ...editorManager.baklavaView.navbarItems,
            ...externalApplicationManager.appCapabilities.value,
        ]);

        const editorTitleInterface = new InputInterface(
            'Graph name',
            '',
        );
        editorTitleInterface.setDefaultComponent();

        // Setup custom hook, which is executed when procedure starts or stops running
        runInfo.setHook(this.updateActiveNavbarItems);

        // Mock hoveredOver to suppress warning when creating Side Panel
        // hoveredOver over is needed only for temporary connections, which are not used here
        provide('hoveredOver', () => {});

        return {
            graphName,
            editorManager,
            editorTitleInterface,
            /* create instance of external manager to control
            connection, dataflow and specification
            */
            externalApplicationManager,
            externalApp: {
                available: false,
                connected: false,
                backend: false,
            },
            navbarItems,
            activeNavbarItemsNames: [],
            saveConfiguration: saveGraphConfiguration,
            saveGraphConfiguration,
            saveSpecificationConfiguration,
            exportGraph,
            saveMenuShow: false,
            exportMenuShow: false,
            editTitle: false,
            notificationStore,
            showSearch: false,
            crossIcon: markRaw(icons.Cross),
            navbarGuard: false,
            isMounted: false,
            windowWidth: 0,
            buttonWidth: 0,
            searchbarWidthMultiplcity: 4,
            hoverInfo: {
                isHovered: false,
                hoveredPanel: undefined,
            },
            // Toggleable panels and their configuration
            panels: {
                notifications: {
                    isOpen: false,
                    class: '.notifications',
                    iconRef: 'notifications',
                    showTransform: '-495px, 0px',
                    hideTransform: '0px, 0px',
                },
                palette: {
                    isOpen: !this.$isMobile,
                    class: '.baklava-node-palette',
                    iconRef: 'palette',
                    showTransform: '0px, 0px',
                    hideTransform: '-450px, 0px',
                },
                settings: {
                    isOpen: false,
                    class: '.settings-panel',
                    iconRef: 'settings',
                    showTransform: '-495px, 0px',
                    hideTransform: '0px, 0px',
                },
                externalAppStatus: {
                    isOpen: false,
                    class: '.external-app-status',
                    iconRef: 'backend',
                    showTransform: '-89%, 0px',
                    hideTransform: '-89%, -180px',
                },
                nodesearch: {
                    isOpen: false,
                },
                fullscreen: {
                    isOpen: false,
                },
                graphDetails: {
                    isOpen: false,
                    class: '.details-panel',
                    iconRef: 'graphDetails',
                    showTransform: '-495px, 0px',
                    hideTransform: '0px, 0px',
                },
            },
        };
    },
    methods: {
        togglePanel(panel, disable = false) {
            const panelSelector = document.querySelector(panel.class);
            const iconRef = this.$refs[panel.iconRef]?.getRef();

            if (disable) {
                panel.isOpen = false;
            } else {
                panel.isOpen = !panel.isOpen;
            }
            const isPanelOpen = panel.isOpen;

            if (panelSelector) {
                if (!iconRef) return;
                panelSelector.style.transition = `transform ${isPanelOpen ? '0.4' : '0.2'}s`;
                panelSelector.style.transform = `translate(${
                    isPanelOpen ? panel.showTransform : panel.hideTransform
                })`;

                iconRef.classList.toggle('open', isPanelOpen);
            }
        },

        clickOutside(event, panel) {
            let icon = this.$refs[panel.iconRef]?.getRef();
            if (!icon) icon = panel.refs;
            if (!icon) return;

            const currentElement = event.target;
            if (currentElement instanceof Node && (icon.contains(currentElement)
                                                   || icon === currentElement)) {
                return;
            }

            this.togglePanel(panel, true);
        },

        async requestDataflowAction(actionItem) {
            if (!this.externalApp.available) return;
            if (
                this.isInProgress(actionItem.procedureName) &&
                this.isStoppable(actionItem.procedureName)
            ) {
                await this.externalApplicationManager.requestDataflowStop(actionItem.procedureName);
                return;
            }
            const activeAction = this.activeNavbarItemsNames.includes(actionItem.procedureName);
            if (activeAction && !this.isInProgress(actionItem.procedureName)) {
                await this.externalApplicationManager.requestDataflowAction(
                    actionItem.procedureName,
                    actionItem.requireResponse,
                );
            } else if (activeAction) {
                NotificationHandler.terminalLog('warning', `${actionItem.name} cannot be stopped`);
            }
        },

        async requestDataflowExport(prompt = true) {
            if (!this.externalApp.available) return;
            const result = await this.externalApplicationManager.requestDataflowExport();

            if (result !== false) {
                // Copy the saveConfiguration object to prevent changing the original object
                this.saveConfiguration = { ...saveGraphConfiguration };
                this.saveConfiguration.saveName = (
                    result.filename ?? saveGraphConfiguration.saveName);
                if (prompt) {
                    this.saveConfiguration.readonly = undefined;
                    this.saveConfiguration.hideHud = undefined;
                    this.saveConfiguration.position = undefined;
                    this.saveConfiguration.saveCallback =
                        () => {
                            this.saveConfiguration.saveCallbackCustomFormat(
                                result.content,
                            );
                            this.saveConfiguration = saveGraphConfiguration;
                        };
                    this.saveMenuShow = true;
                } else {
                    this.saveConfiguration.saveCallbackCustomFormat(
                        result.content,
                    );
                }
            }
        },

        onClickNodeSearch() {
            this.togglePanel(this.panels.nodesearch);
            if (this.panels.nodesearch.isOpen) {
                this.$nextTick(() => this.$refs.searchBar.getInput().focus());
            }
        },

        openNavbar() {
            this.navbarGuard = true;
            this.$refs.navbar.classList.add('isHovered');
        },

        handleMouseLeave(ev) {
            if (!this.$refs.navbar.classList.contains('isHovered')) return;
            // check if event targets UI, if not hide NavBar
            if (
                (this.hideHud || this.$isMobile) &&
                !this.editorManager.baklavaView.displayedGraph.sidebar.visible &&
                !ev.target.closest('.baklava-node-palette')) {
                // Ignore first event to prevent NavBar from hiding when side bar is opened
                if (this.navbarGuard) {
                    this.navbarGuard = false;
                } else {
                    this.togglePanel(this.panels.palette.getRef(), true);
                    this.$refs.navbar.classList.remove('isHovered');
                }
            }
        },

        updateHoverInfo(name, isRunnable = false) {
            if (!isRunnable || this.activeNavbarItemsNames.includes(name) ||
                this.isInProgress(name)) {
                this.hoverInfo.hoveredPanel = name;
                this.hoverInfo.isHovered = true;
            }
        },

        resetHoverInfo(name) {
            if (this.hoverInfo.hoveredPanel === name) {
                this.hoverInfo.hoveredPanel = undefined;
                this.hoverInfo.isHovered = false;
            }
        },
        isHovered(name) {
            return this.hoverInfo.hoveredPanel === name && this.hoverInfo.isHovered;
        },

        isInProgress(procedure) {
            return runInfo.get(procedure).inProgress;
        },

        isStoppable(procedure) {
            return this.externalApplicationManager
                .appCapabilities.stoppable_methods?.includes(procedure) ?? true;
        },

        getNavbarActionTooltip(actionItem) {
            if (
                this.isStoppable(actionItem.procedureName) &&
                this.isInProgress(actionItem.procedureName)
            ) {
                if (actionItem.stopName !== undefined) return actionItem.stopName;
                return `Stop ${actionItem.name}`;
            }
            return actionItem.name;
        },

        updateActiveNavbarItems() {
            const { navbarItems } = this;
            let activeItems = new Set(navbarItems.map((item) => item.procedureName));
            navbarItems.filter((item) => this.isInProgress(item.procedureName)).forEach((item) => {
                // Intersection of current activeItems and items allowToRunInParallelWith
                activeItems = new Set(
                    (item.allowToRunInParallelWith ?? []).filter((name) => activeItems.has(name)),
                ).add(item.procedureName);
            });
            this.activeNavbarItemsNames = Array.from(activeItems);
        },
    },
    async mounted() {
        this.isMounted = true;
        this.buttonWidth = this.$refs.palette.getRef().offsetWidth;
        this.windowWidth = window.innerWidth;

        window.addEventListener('resize', () => {
            this.windowWidth = window.innerWidth;
            if (this.$refs.palette.getRef()) {
                this.buttonWidth = this.$refs.palette.getRef().offsetWidth;
            }
        });

        this.externalApp.available = this.externalApplicationManager.isExternalAppAvailable();
        this.externalApp.backend = this.externalApplicationManager.backend === true;
        this.externalApp.connected = false;
        this.externalApplicationManager.registerConnectionHook(() => {
            this.externalApp.available = this.externalApplicationManager.isExternalAppAvailable();
            this.externalApp.connected = this.externalApplicationManager.isConnected();
            this.externalApp.backend = this.externalApplicationManager.backend === true;
        });
        this.externalApplicationManager.registerDisconnectionHook(() => {
            this.externalApp.available = this.externalApplicationManager.isExternalAppAvailable();
            this.externalApp.connected = false;
            this.externalApp.backend = this.externalApplicationManager.backend === true;
        });
    },
};
</script>

<!-- eslint-disable vue/no-multiple-template-root -->
<!-- eslint-disable vue/no-v-model-argument -->
<template>
    <NavBarTransitions
        :saveConfiguration="saveConfiguration"
        v-model:saveMenuShow="saveMenuShow"
        v-model:exportMenuShow="exportMenuShow"
        v-model:baklavaView="editorManager.baklavaView"
    />

    <div class="wrapper"
        v-click-outside="(ev) => handleMouseLeave(ev)"
    >
        <div ref="navbar" class="wrapper prevent-select"
            v-show="!preview"
            :class="(!hideHud && !$isMobile) ? 'wrapper-hud' : 'wrapper-hidden'"
            @pointerenter="$event.target.classList.add('isHovered')"
        >
            <div class="container">
                <div :style="leftContainerStyles" ref="leftButtons">
                    <div
                        :class="['logo', mobileClasses]"
                        @pointerover="() => updateHoverInfo('logo')"
                        @pointerleave="() => resetHoverInfo('logo')"
                    >
                        <FilesContextMenu ref="contextMenu"
                            :hover="isHovered('logo')"
                            :externalApp="externalApp"
                            :setEditTitle="() => this.$refs.titleBar.setEditTitle()"
                            :mobileClasses="mobileClasses"
                            :hideHud="hideHud"
                            :readonly="readonly"
                            :exportGraphCallback="()=>{
                                exportMenuShow = !exportMenuShow
                                exportGraph = !exportGraph
                            }"
                            :saveGraphCallback="() => {
                                editorManager.editor.resetAllGraphsToSave();
                                saveMenuShow = !saveMenuShow;
                                saveConfiguration = saveGraphConfiguration;
                            }"
                            :saveSpecificationCallback="() => {
                                editorManager.editor.resetAllGraphsToSave();
                                saveMenuShow = !saveMenuShow;
                                saveConfiguration = saveSpecificationConfiguration;
                            }"
                            :requestDataflowExport="requestDataflowExport"
                        />
                    </div>

                    <NodeBrowserButton
                        :mobileClasses="mobileClasses"
                        :hover="isHovered('palette')"
                        :hideHud="hideHud"
                        :readonly="readonly"
                        :openPanel="this.panels.palette.isOpen"
                        @onClicked="()=> togglePanel(panels.palette)"
                        @hover="() => updateHoverInfo(panels.palette.iconRef)"
                        @hoverStop="() => resetHoverInfo(panels.palette.iconRef)"
                        ref="palette"
                    />
                    <template v-if="this.externalApp.available">
                        <ExternalAppAction
                            :mobileClasses="mobileClasses"
                            :navbarItems="navbarItems"
                            :activeNavbarItems="activeNavbarItems"
                            :isInProgress="isInProgress"
                            :isStoppable="isStoppable"
                            :requestDataflowAction="requestDataflowAction"
                            :updateHoverInfo="updateHoverInfo"
                            :resetHoverInfo="resetHoverInfo"
                            :getNavbarActionTooltip="getNavbarActionTooltip"
                            :isHovered="isHovered"
                        />
                    </template>
                    <SubgraphNavigation
                        :navbarItems="navbarItems"
                        :isHovered="isHovered"
                        :mobileClasses="mobileClasses"
                        :resetHoverInfo="resetHoverInfo"
                        :updateHoverInfo="updateHoverInfo"
                        :toggleGraphDetails="(val) => {
                            this.togglePanel(this.panels.graphDetails, val);
                        }"
                    />
                </div>
                <TitleBar
                    v-model:graphName="graphName"
                    :mobileClasses="mobileClasses"
                    :openPanel="!panels.nodesearch.isOpen"
                    :editorManager="editorManager"
                    ref="titleBar"
                />
                <div :style="rightContainerStyles" ref="rightButtons">
                    <SearchBar
                        :mobileClasses="mobileClasses"
                        :hover="isHovered('search')"
                        :nodesearchInputStyles="nodesearchInputStyles"
                        @hoverStart="() => updateHoverInfo('search')"
                        @hoverStop="() => resetHoverInfo('search')"
                        @onClicked="() => onClickNodeSearch()"
                        v-model:openPanel="panels.nodesearch.isOpen"
                        ref="searchBar"
                    />
                    <FullscreenButton ref="fullscreen"
                        :hover="isHovered('Fullscreen')"
                        :mobileClasses="mobileClasses"
                        @hoverStart="() => updateHoverInfo(panels.fullscreen.iconRef)"
                        @hoverStop="() => resetHoverInfo(panels.fullscreen.iconRef)"
                    />
                    <GraphDetailsButton
                        :mobileClasses="mobileClasses"
                        :hover="isHovered('graphDetails')"
                        :editorManager="editorManager"
                        :openPanel="panels.graphDetails.isOpen"
                        @onClicked="()=> togglePanel(panels.graphDetails)"
                        @hover="() => updateHoverInfo(panels.graphDetails.iconRef)"
                        @hoverStop="() => resetHoverInfo(panels.graphDetails.iconRef)"
                        ref="graphDetails"
                    />
                    <SettingsButton
                        :mobileClasses="mobileClasses"
                        :hover="isHovered('settings')"
                        @onClicked="() => togglePanel(panels.settings)"
                        @hover="() => updateHoverInfo(panels.settings.iconRef)"
                        @hoverStop="() => resetHoverInfo(panels.settings.iconRef)"
                        v-model:openPanel="panels.settings.isOpen"
                        ref="settings"
                    />

                    <ExternalAppStatus
                        ref="backend"
                        :externalApp="this.externalApp"
                        :mobileClasses="mobileClasses"
                        :hover="isHovered('externalAppStatus')"
                        :openPanel="panels.externalAppStatus.isOpen"
                        @hoverStart="() => updateHoverInfo('externalAppStatus')"
                        @hoverStop="() => resetHoverInfo('externalAppStatus')"
                        @onClicked="() => togglePanel(panels.externalAppStatus)"
                        @onClickOutside="(ev) => clickOutside(ev, panels.externalAppStatus)"
                     />
                    <NotificationButton
                        ref="notifications"
                        :mobileClasses="mobileClasses"
                        :hideHud="hideHud"
                        :notificationCount="this.notificationStore.notifications.length"
                        :hover="isHovered('notifications')"
                        :openPanel="panels.notifications.isOpen"
                        @onClicked="() => togglePanel(panels.notifications)"
                        @hover="() => updateHoverInfo(panels.notifications.iconRef)"
                        @hoverStop="() => resetHoverInfo(panels.notifications.iconRef)"
                    />
                </div>
            </div>
            <div class="progress-bar" />
        </div>
        <Notifications v-click-outside="(ev) => clickOutside(ev, panels.notifications)" />
        <Settings
            v-click-outside="(ev) => clickOutside(ev, panels.settings)"
            tabindex="-1"
            :viewModel="editorManager.baklavaView"
        />
        <CustomSidebar
            @sidebar-open="openNavbar"
        />
        <GraphDetails
            v-click-outside="(ev) => clickOutside(ev, panels.graphDetails)"
            tabindex="-1"
        />
    </div>
</template>

<style lang="scss" scoped>
.wrapper {
    z-index: 5;
}

.wrapper-hud {
    position: relative;
}

.wrapper-hidden {
    $navbar-padding-bottom: calc($navbar-height * 1.5);
    position: absolute;
    width: 100%;
    top: -$navbar-height;
    padding-bottom: $navbar-padding-bottom;
    transition: 0.2s;

    &.isHovered {
        transform: translateY($navbar-height);
        padding-bottom: 0;
    }

    &:not(.isHovered) > .progress-bar {
        bottom: calc($navbar-padding-bottom + 1px);
    }
}

.container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    height: $navbar-height;
    background-color: $gray-600;
    border: 1px solid $gray-500;
    border-left: 0;
    border-right: 0;

    & > div {
        display: inherit;
        flex-grow: 1;
        & > div {
            &:last-child {
                border-right: 1px solid $gray-500;
            }
        }
    }
}
</style>
