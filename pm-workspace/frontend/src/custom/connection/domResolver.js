/*
 * Copyright (c) 2022-2023 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

// Baklava file that is not exported so it has to be copy-pasted here locally into the project

export default function getDomElements(ni) {
    const interfaceDOM = document.getElementById(ni.id);
    const portDOM = interfaceDOM?.getElementsByClassName('__port');

    return {
        node: interfaceDOM?.closest('.baklava-node') ?? null,
        interface: interfaceDOM?.closest('.baklava-node-interface'),
        port: portDOM && portDOM.length > 0 ? portDOM[0] : null,
    };
}
