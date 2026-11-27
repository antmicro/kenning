<!--
Copyright (c) 2022-2023 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<template>
    <div v-if="isSvg" v-html="svgElem.outerHTML"></div>
    <img v-else :src="imgURI">
</template>

<script>
import { watch } from 'vue';
import DOMPurify from 'dompurify';
/* eslint-disable no-unused-vars,no-lonely-if,vue/no-setup-props-destructure */

export default {
    props: {
        imgURI: {
            type: String,
            required: true,
        },
        hover: {
            type: Boolean,
            required: false,
        },
    },
    setup(props) {
        const isSvg = props.imgURI.startsWith('data:image');

        if (!isSvg) return { isSvg };

        const element = document.createElement('div');
        // Webpack converts svg images to base64, in needs to be converted to a string
        // so that a class 'hovered' can be added to it.
        const svgString = atob(props.imgURI.slice('data:image/svg+xml;base64,'.length));
        element.innerHTML = DOMPurify.sanitize(svgString, { svg: true });
        const firstChild = element.firstElementChild;

        watch(() => props.hover, (newValue, oldValue) => {
            if (newValue) {
                if (firstChild.classList) {
                    firstChild.classList.add('hovered');
                } else {
                    firstChild.className += ' hovered';
                }
            } else {
                if (firstChild.classList) {
                    firstChild.classList.remove('hovered');
                } else {
                    firstChild.className -= ' hovered';
                }
            }
        });

        return { isSvg, svgElem: firstChild };
    },
};
</script>

<style lang="scss">

.hovered {
    svg {
        display: block;
    }

    path {
        fill: $green !important;
    }
}
</style>
