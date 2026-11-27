/*
 * Copyright (c) 2026 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

// A file with functions utilizes by connections

const getPortCoordinates = (resolved) => {
    if (resolved.node && resolved.interface && resolved.port) {
        return [
            resolved.node.offsetLeft +
                        resolved.interface.offsetLeft +
                        resolved.port.offsetLeft +
                        resolved.port.clientWidth / 2,
            resolved.node.offsetTop +
                        resolved.interface.offsetTop +
                        resolved.port.offsetTop +
                        resolved.port.clientHeight / 2,
        ];
    }
    return [0, 0];
};

export default getPortCoordinates;
