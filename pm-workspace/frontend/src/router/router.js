/*
 * Copyright (c) 2022-2023 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { createRouter, createWebHistory } from 'vue-router';
import Home from '../components/Home.vue';

const RouterVue = createRouter({
    history: createWebHistory(),
    routes: [
        {
            path: '/:catchAll(.*)',
            name: 'home',
            component: Home,
            meta: {
                title: process.env.VUE_APP_EDITOR_TITLE ?? 'Pipeline Manager',
            },
        },
    ],
});

export default RouterVue;
