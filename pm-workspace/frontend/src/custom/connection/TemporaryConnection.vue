<!--
Copyright (c) 2022-2023 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<template>
    <ConnectionView
        :x1="d.input[0]"
        :y1="d.input[1]"
        :x2="d.output[0]"
        :y2="d.output[1]"
        :state="status"
        :connection="connection"
        is-temporary
    />
</template>

<script>
import { computed } from 'vue';
import { Components } from '@baklavajs/renderer-vue';
import ConnectionView from './ConnectionView.vue';
import getPortCoordinates from './portCoordinates';
import getDomElements from './domResolver';

export default {
    extends: Components.TemporaryConnection,
    components: {
        ConnectionView,
    },
    setup(props) {
        const status = computed(() => (props.connection ? props.connection.status
            : Components.TemporaryConnectionState.NONE));

        const d = computed(() => {
            if (props.connection.updated) ;
            if (!props.connection) {
                return {
                    input: [0, 0],
                    output: [0, 0],
                };
            }

            const start = getPortCoordinates(getDomElements(props.connection.from));
            const end = props.connection.to
                ? getPortCoordinates(getDomElements(props.connection.to))
                : [props.connection.mx || start[0], props.connection.my || start[1]];

            return {
                input: start,
                output: end,
            };
        });

        return { d, status };
    },
};
</script>
