<template>
    <div class="welcome-container">
        <p class="subtitle">
            <span v-if="welcomeMessage">
            {{ welcomeMessage }}
            <br /><br />
            </span>
        </p>

        <div
            class="drop-zone"
            @dragover.prevent="isDragging = true"
            @dragleave="isDragging = false"
            @drop.prevent="isDragging = false"
            :class="{ dragging: isDragging }"
        >
            <svg class="drop-icon" width="40" height="40" viewBox="0 0 24 24" fill="none">
                <path
                    d="M12 15V3m0 0l-4 4m4-4l4 4M2 17l.621 2.485A2 2 0 0 0 4.561 21h14.878a2 2 0 0 0
                        1.94-1.515L22 17"
                    stroke="currentColor"
                    stroke-width="2"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                />
            </svg>

            <h3>
                {{
                    specificationLoaded
                        ? 'Drag & drop graph file or nodes'
                        : 'Drag & drop specification of nodes'
                }}
            </h3>
            <p class="drop-text">
                {{
                    specificationLoaded
                        // eslint-disable-next-line max-len
                        ? 'Either upload an existing graph in JSON format or start a new graph by drag and dropping nodes specified in the left sidebar'
                        // eslint-disable-next-line max-len
                        : 'Load a specification file in JSON format to provide available node types. Optionally upload both specification and graph'
                }}
            </p>

            <div class="or-divider"><span>OR</span></div>

            <button class="browse-button" @click="$refs.fileInput.click()">
                <component :is="'svg'" width="20" height="20" viewBox="0 0 24 24" fill="none">
                    <path
                        d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"
                        stroke="currentColor"
                        stroke-width="2"
                        stroke-linecap="round"
                        stroke-linejoin="round"
                    />
                    <polyline
                        points="14,2 14,8 20,8"
                        stroke="currentColor"
                        stroke-width="2"
                        stroke-linecap="round"
                        stroke-linejoin="round"
                    />
                </component>
                Browse files
            </button>
        </div>

        <nav class="footer-links">
            <a
                v-for="{ text, url, icon, polyline } in links"
                :key="text"
                :href="url"
                target="_blank"
                class="footer-link"
            >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                    <path
                        :d="icon"
                        stroke="currentColor"
                        stroke-width="2"
                        stroke-linecap="round"
                        stroke-linejoin="round"
                    />
                    <polyline
                        v-if="polyline"
                        :points="polyline"
                        stroke="currentColor"
                        stroke-width="2"
                        stroke-linecap="round"
                        stroke-linejoin="round"
                    />
                </svg>
                {{ text }}
            </a>
        </nav>

        <input
            ref="fileInput"
            type="file"
            @change="loadFiles($event.target.files)"
            multiple
            accept=".json"
        />
    </div>
</template>

<script>
import { defineComponent, ref, computed } from 'vue';
import EditorManager from '../../core/EditorManager';

const LINKS = [
    {
        text: 'Documentation',
        url: 'https://antmicro.github.io/kenning-pipeline-manager/introduction.html',
        icon: 'M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z',
        polyline: '14,2 14,8 20,8',
    },
];

export default defineComponent({
    props: {
        loadFiles: { type: Function, required: true },
    },
    setup() {
        const isDragging = ref(false);
        const specificationLoaded = computed(
            () => EditorManager.getEditorManagerInstance().specificationLoaded.value,
        );

        const welcomeMessage = computed(
            () => process.env.VUE_APP_WELCOME_MESSAGE,
        );

        return {
            isDragging,
            specificationLoaded,
            welcomeMessage,
            links: LINKS,
        };
    },
});
</script>

<style lang="scss">
.welcome-container-panel {
    position: fixed !important;
    left: min(300px, 50vw) !important;
    top: 0 !important;
    width: calc(100vw - min(300px, 50vw)) !important;
    height: 100vh !important;
    z-index: 2 !important;
    padding: 0 !important;
    margin: 0 !important;
    right: 0 !important;
    bottom: 0 !important;

    .blur-panel {
        position: static !important;
        left: 0 !important;
        top: 0 !important;
        width: 100% !important;
        height: 100% !important;
        padding: 0 !important;
        margin: 0 !important;
        background-color: transparent !important;
        backdrop-filter: none !important;
    }

    .popup-menu {
        position: static !important;
        transform: none !important;
        left: auto !important;
        top: auto !important;
        width: 100% !important;
        height: 100% !important;
        padding: 0rem !important;
        margin: 0 !important;
        border: none !important;
        border-radius: 0 !important;
        background: #00000045 !important;
        backdrop-filter: blur(10px) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
}
</style>

<style scoped lang="scss">
@import '../../../styles/variables';

.welcome-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    max-width: 500px;
    margin: $spacing-xxl auto;
    background: $gray-600;
    color: $white;
    padding: $spacing-xxl;
    border-radius: $spacing-m;
}

.subtitle {
    font-size: $fs-medium;
    color: $gray-100;
    line-height: 1.5;
    margin: 0 0 $spacing-xxl 0;
    font-weight: 400;
    text-align: center;
}

.drop-zone {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 100%;
    min-height: 200px;
    border: 2px dashed $gray-500;
    border-radius: $spacing-s;
    background: rgba($gray-500, 0.2);
    transition: all 0.3s ease;
    padding: $spacing-xxl $spacing-xl;
    text-align: center;

    &.dragging {
        border-color: $green;
        background: rgba($green, 0.05);
    }

    h3 {
        font-size: $fs-large;
        font-weight: 500;
        margin: 0 0 $spacing-xs 0;
        color: $white;
    }
}

.drop-icon {
    margin-bottom: $spacing-l;
    color: $gray-200;
    padding: $spacing-xs;
}

.drop-text {
    color: $gray-100;
    margin: 0 0 $spacing-xl 0;
    font-size: $fs-small;
    max-width: 280px;
}

.or-divider {
    display: flex;
    align-items: center;
    width: 100%;
    margin: $spacing-l 0;

    &::before,
    &::after {
        content: '';
        flex: 1;
        height: 1px;
        background: $gray-500;
    }

    span {
        padding: 0 $spacing-l;
        color: $gray-200;
        font-size: $fs-small;
        font-weight: 500;
    }
}

.browse-button {
    display: flex;
    align-items: center;
    gap: $spacing-xs;
    background: $green;
    color: $black;
    border-radius: $spacing-s;
    padding: $spacing-m $spacing-xl;
    font-size: $fs-medium;
    font-weight: 500;
    transition: all 0.2s ease;
    margin-bottom: $spacing-l;

    &:hover {
        background: rgba($green, 0.8);
    }
}

.footer-links {
    display: flex;
    gap: $spacing-xl;
    margin-top: $spacing-xl;
    justify-content: center;
}

.footer-link {
    display: flex;
    align-items: center;
    gap: $spacing-xs;
    color: $gray-100;
    text-decoration: none;
    font-size: $fs-small;
    font-weight: 500;
    transition: all 0.2s ease;
    padding: $spacing-xs $spacing-m;
    border-radius: $spacing-xs;

    &:hover {
        color: $green;
        background: rgba($green, 0.05);

        svg {
            opacity: 1;
        }
    }

    svg {
        opacity: 0.7;
    }
}

@media (max-width: 768px) {
    .welcome-container {
        margin: $spacing-l;
        padding: $spacing-xl;
        max-width: none;
    }

    .subtitle {
        font-size: $fs-small;
    }

    .drop-zone {
        min-height: 180px;
    }
}
</style>
