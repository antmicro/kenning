/*
 * Copyright (c) 2022-2025 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { stringify } from 'ajv';
import jsonMap from 'json-source-map';
import jsonlint from 'jsonlint';

/**
 * Validates JSON according to a given schema.
 *
 * @param {import('ajv/dist/2019').Ajv2019} ajv - JSON validator.
 * @param {Object} schema - Validation schema.
 * @param {Object|string} data - Data to validate.
 * @param {string} reference - Schema entity.
 * @returns {string[]} Validation errors.
 */
export default function validateJSON(ajv, schema, data, reference = '') {
    ajv.removeSchema('root').removeSchema(schema.$id).addSchema(schema, 'root');
    const validate = ajv.getSchema(`root${reference}`);
    if (validate === undefined) {
        return [`Invalid value of "reference" parameter: ${reference}`];
    }

    const isTextFormat = typeof data === 'string';
    let dataJSON;

    try {
        dataJSON = isTextFormat ? jsonlint.parse(data) : data;
    } catch (exception) {
        return [`Not a proper JSON file: ${exception.toString()}`];
    }

    const valid = validate(dataJSON);

    if (valid) {
        return [];
    }

    // Parsing errors messages to a human readable string
    const errors = validate.errors?.map((error) => {
        // It is assumed that the id of the schema is for example `dataflow_schema`
        // Here a prefix is obtained
        const nameOfEntity = schema.$id.replace(/((_params)|(_returns))?(_schema)?(.json)?$/, '');
        const path = `${nameOfEntity}${error.instancePath}`;
        let errorPrefix = '';

        if (isTextFormat) {
            const result = jsonMap.parse(data);
            // 1 is added as the lines are numbered from 0
            const lineStart = result.pointers[error.instancePath].value.line + 1;
            const lineEnd = result.pointers[error.instancePath].valueEnd.line + 1;

            if (lineStart === lineEnd) {
                errorPrefix = `Line ${lineStart} -`;
            } else {
                errorPrefix = `Lines ${lineStart}-${lineEnd} -`;
            }
        }

        switch (error.keyword) {
            case 'enum':
                return `${errorPrefix} ${path} ${error.message} - ${stringify(
                    error.params.allowedValues,
                )}`;
            case 'additionalProperties':
                return `${errorPrefix} ${path} ${error.message} - ${stringify(
                    error.params.additionalProperty,
                )}`;
            case 'const':
                return `${errorPrefix} ${path} ${error.message} - ${stringify(
                    error.params.allowedValue,
                )}`;
            case 'unevaluatedProperties':
                return `${errorPrefix} ${path} ${error.message} - ${stringify(
                    error.params.unevaluatedProperty,
                )}`;
            // Those errors are not informative at all
            case 'not':
            case 'oneOf':
                return '';
            default:
                return `${errorPrefix} ${path} ${error.message}`;
        }
    }) ?? [];

    return errors.filter((err) => err !== '');
}
