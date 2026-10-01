/*
 * Copyright (c) 2023-2024 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

export interface NavbarItem {
    name: string,
    stopName?: string,
    iconName: string,
    icon?: any, // eslint-disable-line @typescript-eslint/no-explicit-any
    procedureName: string
    allowToRunInParallelWith?: [string]
}
