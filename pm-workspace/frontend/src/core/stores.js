/*
 * Copyright (c) 2022-2024 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { reactive } from 'vue';

const storageAvailable = (() => {
    try {
        const randomKey = Math.random().toString(36);
        const randomValue = Math.random().toString(36);
        localStorage.setItem(randomKey, randomValue);
        localStorage.removeItem(randomKey);
        return true;
    } catch {
        return false;
    }
})();

const pmStorage = new Map();
const get = (key) => {
    if (storageAvailable) return localStorage.getItem(key);
    return pmStorage.get(key) ?? null;
};

const set = (key, value) => {
    if (storageAvailable) localStorage.setItem(key, value);
    else pmStorage.set(key, value);
};

const remove = (key) => {
    if (storageAvailable) localStorage.removeItem(key);
    else pmStorage.delete(key);
};

/* eslint-disable import/prefer-default-export */
export const notificationStore = reactive({
    notifications: JSON.parse(get('notifications')) || [],
    add(notification) {
        this.notifications.push(notification);

        set('notifications', JSON.stringify(this.notifications));
    },

    remove() {
        remove('notifications');
        this.notifications = [];
    },

    removeOne(index) {
        const newNotifications = this.notifications.filter((_, idx) => index !== idx);

        set('notifications', JSON.stringify(newNotifications));
        this.notifications = newNotifications;
    },
});

export const MAIN_TERMINAL = 'Terminal';

export const terminalStore = reactive({
    // Object
    logs: {
        Terminal: JSON.parse(get(`logs`)) || [],
    },
    readOnly: {
        Terminal: true,
    },
    add(log, instance = MAIN_TERMINAL) {
        this.logs[instance].push(log);

        // Update localStorage only for the main terminal
        if (instance === MAIN_TERMINAL) {
            set(`logs`, JSON.stringify(this.logs[instance]));
        }
    },
    isReadOnly(instance = MAIN_TERMINAL) {
        return this.readOnly[instance];
    },

    /**
     * Adds a parsed notification. If there are messages, then it returns a following message:
     *
     * Title:
     *     message_first_line
     *     message_second_line
     *     ...
     *     message_last_line
     *
     * Otherwise, if messages are empty, then it returns a following message:
     *
     * Title.
     *
     * @param {string} title title of the message
     * @param {Array[string] | string | undefined} messages messages of the message
     */
    addParsed(title, messages, instance = MAIN_TERMINAL) {
        let parsedMessage = title;
        if (messages) {
            if (typeof messages === 'string' || messages instanceof String) {
                messages = [messages]; // eslint-disable-line no-param-reassign
            }
            parsedMessage += ':';

            messages.forEach((message) => {
                parsedMessage += '\n';
                parsedMessage += '    ';
                parsedMessage += message;
            });
        } else {
            parsedMessage += '.';
        }
        this.add(parsedMessage, instance);
    },

    clear(instance = MAIN_TERMINAL) {
        if (instance === MAIN_TERMINAL) {
            remove(`logs`);
        }
        this.logs[instance] = [];
    },

    remove(instance = MAIN_TERMINAL) {
        this.clear(instance);
        delete this.logs[instance];
        delete this.readOnly[instance];
    },

    exists(name = MAIN_TERMINAL) {
        return name in this.logs;
    },

    getInstances() {
        return Object.keys(this.logs);
    },

    /**
     * Creates a new terminal instance.
     * If such terminal already exisists, then false is returned.
     *
     * @param {string} Unique name of the terminal instance to be created.
     * @param {boolean} If true, then the terminal instance will be only for read.
     * @returns returns true if terminal was created, false otherwise.
     */
    createTerminalInstance(name, readOnly = true) {
        if (Object.keys(this.logs).includes(name)) return false;

        this.logs[name] = [];
        this.readOnly[name] = readOnly;
        return true;
    },

    show: false,

    /**
    * @type {import('./communication/utils').TerminalManager|null}
    */
    manager: null,
});
