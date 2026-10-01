/*
 * Copyright (c) 2025 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { BaklavaEvent } from '@baklavajs/events';

const notifyEvents = {};

notifyEvents.exposedInterface = new BaklavaEvent();

notifyEvents.removedNode = new BaklavaEvent();

notifyEvents.subgraphDestroyed = new BaklavaEvent();

notifyEvents.specificationUpdate = new BaklavaEvent();

notifyEvents.specificationRestored = new BaklavaEvent();

export default notifyEvents;
