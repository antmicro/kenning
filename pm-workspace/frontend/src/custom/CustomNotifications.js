/*
 * Copyright (c) 2025 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { useViewModel, useGraph } from '@baklavajs/renderer-vue';
import NotificationHandler from '../core/notifications.js';

import notifyEvents from './notifyEvents.js';

export const exposedInterfaceNotification = (intf, graphId, exposed) => {
    const { viewModel } = useViewModel();

    const type = typeof intf.type === 'string' ? 'Property' : 'Interface';

    const msg = exposed ? 'exposed' : 'privatized';

    const graph = [...viewModel.value.editor.graphs].find((g) => g.id === graphId);
    const graphNode = graph?.graphNode;

    if (graphNode) {
        NotificationHandler.terminalLog('info', 'Upper graph updated', `${type} ${intf.name} has been ${msg} from subgraph.`);
    }
};

export const nodeWithExposedInterfaceNotification = (node) => {
    const { viewModel } = useViewModel();

    const { graph } = useGraph();

    const graphSub = [...viewModel.value.editor.graphs].find((g) => g.id === graph.value.id);
    const graphNode = graphSub?.graphNode;

    // node is part of subgrap node
    if (graphNode) {
        const exposedInterfaces = [...Object.values(node.inputs), ...Object.values(node.outputs)]
            .filter((intf) => intf.externalName);

        exposedInterfaces.forEach((intf) => {
            const intfType = typeof intf.type !== 'string' ? 'interface' : 'property';

            NotificationHandler.terminalLog('info', 'Upper graph updated', `Exposed ${intfType} ${intf.externalName} has been removed.`);
        });
    }
};

const initNotificationEvents = () => {
    notifyEvents.exposedInterface.subscribe('notify', (values) => {
        exposedInterfaceNotification(...values);
    });

    notifyEvents.removedNode.subscribe('notify', (node) => {
        nodeWithExposedInterfaceNotification(node);
    });
};

export default initNotificationEvents;
