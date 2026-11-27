/*
 * Copyright (c) 2026 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Holds definition for User Dialog boxes created using API calls.
 */

/* eslint-disable no-param-reassign */
/* eslint-disable no-underscore-dangle */
export default class CustomMenusManager {
    // Holds definition for defined user dialog templates
    userDialogTemplate = new Map();

    // Holds data for current active user dialog
    userDialogData = {};

    // A name of currently active user dialog
    selectedUserDialog = undefined;

    showCustomDialog = false;

    defaultValuesForTypes = new Map(
        [
            ['text', ''],
            ['number', 0.0],
            ['int', 0],
            ['bool', false],
        ],
    );

    /**
    * A setter which is a wrapper for a navbar components.
    * @param {boolean} val Whether to close user dialog box,
    * when it is false dialog box is closed.
    */
    set showCustomDialogWithReset(val) {
        if (!val) {
            this.resetUserDialog();
            this.showCustomDialog = false;
        }
    }

    get showCustomDialogWithReset() {
        return this.showCustomDialog;
    }

    get userDialogTitle() {
        return this.selectedUserDialog !== undefined ?
            this.userDialogTemplate.get(this.selectedUserDialog).title
            : '';
    }

    get dialogTemplate() {
        if (this.selectedUserDialog === undefined) {
            return {};
        }

        return this.userDialogTemplate.get(this.selectedUserDialog).template;
    }

    get confirmText() {
        if (this.selectedUserDialog === undefined) {
            return 'OK';
        }

        return this.userDialogTemplate.get(this.selectedUserDialog).confirmText;
    }

    /**
     * Function that waits asynchronously until dialog box is closed.
     *
     * @returns {Promise} promise that resolve when dialog box is closed.
     */
    async waitForDialogBoxClosed() {
        return new Promise((resolve) => {
            const inv = setInterval(() => {
                if (!this.showCustomDialog) {
                    clearInterval(inv);
                    resolve();
                }
            });
        }, 10);
    }

    /**
     * Registers a new dialog menu template with selected name.
     * @param {string} templateName Name of the dialog box
     * @param {string} title A title of dialog box.
     * @param {string} confirmText A name of confirmation button.
     * @param {object} template Object holding dialog box entries.
     */
    registerDialogTemplate(templateName, title, confirmText, template) {
        this.userDialogTemplate.set(templateName, {
            title,
            confirmText,
            template,
        });
    }

    /**
     * Selects and active dialog box.
     * @param {string} name Name of dialog box to select.
     *
     * @returns {boolean} Returns true when dialog box exits,
     * false otherwise.
     */
    selectDialogBox(name) {
        this.resetUserDialog();
        if (!this.userDialogTemplate.has(name)) {
            return false;
        }

        this.selectedUserDialog = name;

        const templateEntries = Object.entries(this.userDialogTemplate
            .get(this.selectedUserDialog).template);

        templateEntries.forEach(([key, value]) => {
            const { type } = value;
            this.userDialogData[key] = value?.default ?? this.defaultValuesForTypes.get(type);
        });

        this.showCustomDialog = true;
        return true;
    }

    /**
     * Resets state of dialog box.
     */
    resetUserDialog() {
        this.selectedUserDialog = undefined;
        this.userDialogData = {};
    }
}
