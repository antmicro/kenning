/*
 * Copyright (c) 2025 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */

/**
 * Module holds a properties available for all modules
 * in the project.
 */

/**
 * A object type that holds global properties
 * available for all modules.
 *
 * @property softLoad - it indicates if soft load mode is enabled
 */
type GlobalProperties = {
    softLoad: boolean
};

const globalProperties:GlobalProperties = {
    softLoad: false,
};

export default globalProperties;
