/*
 * Copyright (c) 2026 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

// A helper class used to keep track of collected chunks of a message
// and to keep track of message buffer expiration time

class MessageTracker {
    time: number;

    data: string[];

    constructor() {
        this.time = 0.0;
        this.data = [];
    }
}

export default MessageTracker;
