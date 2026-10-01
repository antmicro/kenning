/*
 * Copyright (c) 2026 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { editorEventBus } from '../../core/nodeCreation/ConfigurationState';
import NotificationHandler from '../../core/notifications.js';

export function unsavedEditorChanges() {
    return new Promise((resolve) => {
        let resolved = false;

        const timeout = setTimeout(() => {
            if (!resolved) {
                resolved = true;
                resolve(false);
            }
        }, 100);

        const event = new CustomEvent('check-validation', {
            detail: {
                resolve: (result) => {
                    if (!resolved) {
                        resolved = true;
                        clearTimeout(timeout);
                        resolve(result);
                    }
                },
            },
        });

        editorEventBus.dispatchEvent(event);
    });
}

export async function checkForUnsavedEditorChangesWithToast() {
    if (await unsavedEditorChanges()) {
        NotificationHandler.showToast('warning', 'You have unsaved changes in the node editor.');
        return true;
    }

    return false;
}
