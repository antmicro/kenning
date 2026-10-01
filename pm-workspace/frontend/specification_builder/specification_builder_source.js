/**
 * Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Module for constructing and validating the specification for the Pipeline Manager.
 *
 * Can be used by other applications to quickly form the specification
 * for the Pipeline Manager server from the browser level.
 *
 * This is the JavaScript equivalent of the Python version of SpecificationBuilder,
 * which delegates most of its operations to it, so if there are any changes to
 * the Python API, the corespoding methods will need to be adapted.
 */

import { assert } from '@vueuse/core';
import EditorManager from '../src/core/EditorManager.js';

import {
    prepareMethodArgs,
    prepareMethodBody,
    getMethodSpecs,
    prepareAndSetPyodide,
} from './utils/specification_builder_init_utils.js';

/**
Is raised when any inconsistency in the specification generation
is spotted.
*/
class SpecificationBuilderError extends Error {
    constructor(message) {
        super(message);
        this.name = 'SpecificationBuilderError';
    }
}

/**
 * Creates a specification file and checks validity of files.
 *
 * This class allows to:
 *
 * Create and modify specification entries using simple API
 * Merge existing specifications with some consistency checks
 * Check URLs to remote resources in terms of availability
 * * Checks correctness of paths to assets from the given directory
 *
 * This class also performs very strict checking, e.g. it does not
 * allow creating duplicates.
 * In case of any errors spotted it raises a
 * SpecificationBuilderError.
 */
export class SpecificationBuilder {
    /**
     * Asynchronous constructor creates a SpecificationBuilder instance.
     *
     * @param {string} version - Specification version that we are going to use.
     * @param {string} assetsDir - Path to the assets directory.
     * @param {boolean} [checkUrls=false] - Tells if the URLs in the specification should be checked
     */

    constructor(version, { assetsDir = null, checkUrls = false } = {}) {
        // specification builder instance name for pyodide namespace
        this.specBuilderPythonName = 'specBuilder';

        // method name => header, last line, whole definition
        this.generatedMethods = {};

        const generateMethods = () => {
            const methodsSpecs = getMethodSpecs(this);

            methodsSpecs.forEach(({
                name, args, defaults, doesReturn,
            }) => {
                defaults = Array.isArray(defaults) ? defaults : [];

                const {
                    allArguments,
                    mandatoryArgsNames,
                    flags: {
                        toStringifyFlag,
                        replacedExtendsOptionalFlag,
                        replacedDefaultOptionalFlag,
                    },

                } = prepareMethodArgs(args, defaults);

                const methodBody = prepareMethodBody(
                    name,
                    allArguments,
                    mandatoryArgsNames,
                    defaults,
                    toStringifyFlag,
                    replacedDefaultOptionalFlag,
                    replacedExtendsOptionalFlag,
                    doesReturn,
                    this,
                );

                // assigns method to the class
                new Function(`this.${name} = ${methodBody}`).call(this);
            });
        };

        const constructorMain = async () => {
            await prepareAndSetPyodide(this);
            assert(this.pyodide !== undefined && this.namespace !== undefined);

            // initialize Python SpecificationBuilder
            const pythonPode = `mandatory_args = json.loads('${JSON.stringify([version, assetsDir, checkUrls])}')\n`
            + `${this.specBuilderPythonName} = SpecificationBuilder(*mandatory_args)`;
            this.pyodide.runPython(pythonPode, { globals: this.namespace });

            generateMethods();
            return this;
        };

        return constructorMain();
    }

    /**
     * Creates a specification and validates it using schema.
     *
     * @param {boolean} fail_on_warnings - Tells if the specification creation should fail on warnings
     * @param {boolean} sort_spec - True if the entries in the specification should be sorted.
     * @returns {object} - Built specification, if successful
     * @throws {SpecificationBuilderError} - Raised when specification is not valid or when warnings appeared
     */
    async create_and_validate_spec({ fail_on_warning: failNnWarning = true, sort_spec = false } = {}) {
        assert(typeof this._construct_specification === 'function');
        assert(typeof this.pyodide === 'function');

        const spec = this._construct_specification(sort_spec);

        const [res, warnings, errors] = await validate_specification(spec);

        if (Array.isArray(warnings) && warnings.length) {
            console.warn('Validation warnings:');
            warnings.forEach((warning) => console.warn(`* \t${warning}`));
        }

        if (res !== 0) throw new SpecificationBuilderError(errors);

        const line = `${this.specBuilderPythonName}.warnings`;
        const builderWarnings = this.pyodide.runPython(line, { globals: this.namespace });

        if (failNnWarning) {
            if (builderWarnings > 0) {
                throw new SpecificationBuilderError(`Builder reported ${builderWarnings} issues, failing...`);
            }
        }
        return spec;
    }
}

/**
 * Validates specification.
 *
 * @param {object} specification - Specification json-like structure
 * @returns {[number, Array.<string>, Array.<string>]} - Triple of a status code, a warnings array,
 * and an errors array, where status code is: 0 if successful, 1 - invalid specificaion
 */
export async function validate_specification(specification) {
    const instance = EditorManager.getEditorManagerInstance();
    const { errors, warnings } = await instance.updateEditorSpecification(
        JSON.stringify(specification),
    );

    if (Array.isArray(errors) && errors.length) return ([1, warnings, errors]);

    return [0, warnings, errors];
}

/**
 * Validates graph with given associated specification.
 *
 * @param {object} dataflow - Dataflow json-like structures
 * @param {object} specification - Specification json-like structure
 * @returns {[number, Array.<string>, Array.<string>]} - Triple of a status code,
 * warnings array and an errors array, where status code is:
 * 0 if successful, 1 - invalid dataflow
 */
export async function validate_dataflow(dataflow, specification) {
    const instance = EditorManager.getEditorManagerInstance();
    let { errors, warnings } = await instance.updateEditorSpecification(JSON.stringify(specification));

    ({ errors, warnings } = await instance.loadDataflow(JSON.stringify(dataflow)));
    if (Array.isArray(errors) && errors.length) return [1, warnings, errors];
    return [0, warnings, errors];
}
