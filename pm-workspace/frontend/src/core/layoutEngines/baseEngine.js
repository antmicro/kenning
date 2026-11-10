/*
 * Copyright (c) 2022-2023 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* eslint-disable max-classes-per-file */
/**
 * Base class allowing to define autolayout algorithms. Each engine
 * can specify multiple algorithms, names in `availableAlgorithms` array.
 * Currently chosen algorithm is named in `activeAlgorithm`
 */
export default class BaseLayoutEngine {
    availableAlgorithms = [];

    // Base layout is an abstract class
    constructor() {
        if (this.constructor === BaseLayoutEngine) {
            throw new Error("Can't initialize base class instance"); // eslint-disable-line quotes
        }
    }

    chooseAlgorithm(algorithm) {
        if (!this.availableAlgorithms.includes(algorithm)) {
            throw new Error(
                `Could not find ${algorithm} in ${this.constructor.name} engine.
                Available algorithms: ${this.availableAlgorithms}`,
            );
        }
        this.activeAlgorithm = algorithm;
    }

    /* eslint-disable class-methods-use-this */
    /* eslint-disable no-unused-vars */
    async calculate(graph) {
        throw new Error('Method calculate() must be implemented by layout engine');
    }
}

/**
 * Instance of engine with only single algorithm
 */
export class BaseLayoutAlgorithm extends BaseLayoutEngine {
    constructor() {
        super();
        this.availableAlgorithms = [this.constructor.name];
        this.activeAlgorithm = this.constructor.name;
        this.chooseAlgorithm = undefined; // turn of choosing algorithm
        if (this.constructor === BaseLayoutAlgorithm) {
            throw new Error("Can't initialize base class instance"); // eslint-disable-line quotes
        }
    }
}
