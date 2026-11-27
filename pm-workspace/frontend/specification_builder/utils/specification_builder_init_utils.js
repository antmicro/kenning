/**
 * Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Module containing functions used to initialize the SpecificationBuilder,
 * which is method generagion and pyodide preparation.
 */

import { loadPyodide } from 'pyodide';
import { assert } from '@vueuse/core';

import dataStructuresRaw from '../../../dataflow_builder/data_structures.py';
import specificationBuilderRaw from '../../../specification_builder.py';
import retrievePythonMethodsInfoRaw from './retrieve_python_methods_info.py';
import argumentPreparationUtilsRaw from './argument_preparation_utils.py';

/**
 * Prepares string representation of JS equivalent for python args.
 *
 * End result looks like this:
 * "req_arg1, ..., {opt_arg1 = default1, ...} = {}"
 *
 * 'defaultValues; matches 'pythonArguments' by reading both arrays backwards.
 *
 * @param {Array} pythonArguments - Array of python arguments names
 * @param {Array} defaultValues  - Array of default values for python arguments
 * @returns {string} - String representation of args
 */
export function prepareMethodArgs(pythonArguments, defaultValues) {
    // indicates whether python method's output can be stringified
    let toStringifyFlag = false;

    // gets rid of the `self` argument
    const mandatoryArgsNames = pythonArguments.slice(
        1,
        pythonArguments.length - defaultValues.length,
    );

    // indicates whether `extends` or `default` special words appiered in python
    // methods arguments. If so, their names are replaced in optional
    // and mandatory arguments
    let replacedExtendsOptionalFlag = false;
    let replacedDefaultOptionalFlag = false;

    let idx;
    if ((idx = mandatoryArgsNames.indexOf('extends')) !== -1) mandatoryArgsNames[idx] = 'extend';
    if ((idx = mandatoryArgsNames.indexOf('default')) !== -1) mandatoryArgsNames[idx] = 'default_val';

    // result: all_args = "req_arg1, req_arg2, ..."
    let allArguments = mandatoryArgsNames.join(', ');

    // prepares optional arguments for JS method definition
    if (defaultValues.length !== 0) {
        const optionalArgsNames = pythonArguments.slice(
            pythonArguments.length - defaultValues.length,
        );
        const optionalArgs = {};
        optionalArgsNames.forEach((key, index) => {
            optionalArgs[key] = defaultValues[index];
        });

        toStringifyFlag = 'stringify' in optionalArgs;
        replacedExtendsOptionalFlag = 'extends' in optionalArgs;
        replacedDefaultOptionalFlag = 'default' in optionalArgs;

        if (replacedExtendsOptionalFlag) {
            optionalArgs.extend = optionalArgs.extends;
            delete optionalArgs.extends;
        }

        if (replacedDefaultOptionalFlag) {
            optionalArgs.default_val = optionalArgs.default;
            delete optionalArgs.default;
        }

        allArguments += (allArguments) ? ', {' : '{';
        Object.entries(optionalArgs).forEach(([key, value]) => {
            allArguments += `${String(key)} = ${JSON.stringify(value)}, `;
        });

        // result: all_args = "req_arg1, ..., {opt_arg1 = default1, ...} = {}"
        allArguments = `${allArguments.slice(0, -2)}} = {}`;
    }

    return {
        allArguments,
        mandatoryArgsNames,
        flags: {
            toStringifyFlag,
            replacedExtendsOptionalFlag,
            replacedDefaultOptionalFlag,
        },

    };
}

/**
 * Prepares string representation of generating method body.
 *
 * @param {string} methodName  - Name of a generating method
 * @param {string} allArguments - String representation of preapred method args
 * @param {Array} mandatoryArgsNames - Manadatory args names
 * @param {Array} defaultValues - Default values of arguments
 * @param {boolean} toStringifyFlag - Tells whether to delegate Python call with stringify flag
 * @param {boolean} replacedExtendsOptionalFlag - Tells if 'extends' arg name is replaced
 * @param {boolean} replacedDefaultOptionalFlag - Tells if 'default' arg name is replaced
 * @param {boolean} returnsFlag - Tells whether a delegated Python call returns any result
 * @param {object} context - Context in which methods are generating (e.g. specBuilder instance)
 * @returns {string} - String representation of method body
 */
export function prepareMethodBody(
    methodName,
    allArguments,
    mandatoryArgsNames,
    defaultValues,
    toStringifyFlag,
    replacedExtendsOptionalFlag,
    replacedDefaultOptionalFlag,
    returnsFlag,
    context,
) {
    assert(context.specBuilderPythonName !== undefined && typeof context.generateMethods !== 'object');

    // collects method body
    // here are addjustment of return statement and arguments names
    const methodHeaderLine = `function ${methodName} (${allArguments})`;
    const argsLine = `
    let args = [...arguments]
    let max_args_len = ${mandatoryArgsNames.length + ((defaultValues.length) ? 1 : 0)}`;
    const optionalArgsManipulations = `
    if (${defaultValues.length} !== 0) {
        let last_arg = args[max_args_len - 1]
        if (last_arg !== undefined && typeof last_arg === 'object') {
            if (${toStringifyFlag})
                last_arg['stringify'] = true
            if (${replacedExtendsOptionalFlag}) {
                last_arg.extends = last_arg.extend
                delete last_arg.extend
            }
            if (${replacedDefaultOptionalFlag}) {
                last_arg.default = last_arg.default_val
                delete last_arg.default_val
            }
        }
        else if (${toStringifyFlag})
            args.push({'stringify': true})
    }`;
    const argsStrLine = '\nlet args_str = JSON.stringify(args)\n';
    let pythonCodeLine = 'python_code = \`mandatory_args, optional_args = ';
    pythonCodeLine += `prepare_args('\$\{args_str\}', \$\{max_args_len${((defaultValues.length) ? '' : '+ 1')}\})`;
    pythonCodeLine += `\\n${context.specBuilderPythonName}.${methodName}(*mandatory_args, **optional_args)\``;
    const prefix = `${(returnsFlag) ? 'return ' : ''}${(toStringifyFlag) ? 'JSON.parse(' : ''}`;
    const suffix = `${(toStringifyFlag) ? ')' : ''};`;
    const executeReturnLine = `${prefix}this.pyodide.runPython(python_code, { globals: this.namespace })${suffix}`;

    // composes method body
    const method = `
    ${methodHeaderLine} {
        ${argsLine}
        ${optionalArgsManipulations}
        ${argsStrLine}
        ${pythonCodeLine}
        ${executeReturnLine}
    }`;

    context.generatedMethods[methodName] = {
        header: methodHeaderLine,
        lastLine: executeReturnLine,
        wholeDefinition: method,
    };

    return method;
}

/**
 * Retrieves Python method specification for later generation of JS equivalents.
 *
 * @param {object} context - Context in which methods are generating (e.g. specBuilder instance)
 * @returns {object}
 */
export function getMethodSpecs(context) {
    assert(context.pyodide !== undefined && context.namespace !== undefined, 'Invalid context');

    return JSON.parse(context.pyodide.runPython(
        retrievePythonMethodsInfoRaw,
        { globals: context.namespace },
    ));
}

/**
 * Prepares pyodide with its namespace and adds to the given context.
 *
 * Adding dependencies and loading python files.
 *
 * @param {object} context - Context in which methods are generating (e.g. specBuilder instance)
 */
export async function prepareAndSetPyodide(context) {
    const addDependencies = async () => {
        await context.pyodide.loadPackage('micropip');
        const micropip = context.pyodide.pyimport('micropip');
        await micropip.install('requests');
        await micropip.install('urllib3');
    };

    const prepareNamespace = async () => {
        const namespace = context.pyodide.globals.get('dict')();
        context.pyodide.runPython(
            [
                dataStructuresRaw,
                specificationBuilderRaw,
                argumentPreparationUtilsRaw,
            ].join('\n'),
            { globals: namespace });
        return namespace;
    };

    context.pyodide = await loadPyodide({ indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.23.4/full/' });
    await addDependencies();
    context.namespace = await prepareNamespace();
}
