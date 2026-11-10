/*
 * Copyright (c) 2022-2024 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

// eslint-disable-next-line max-classes-per-file
import {
    JSONRPCServerAndClient,
    JSONRPCServer,
    JSONRPCClient,
    JSONRPCRequest,
    JSONRPCResponse,
    JSONRPCServerMiddleware,
    JSONRPCServerMiddlewareNext,
    createJSONRPCErrorResponse,
    JSONRPCErrorResponse,
    JSONRPCID,
} from 'json-rpc-2.0';
import Ajv2019 from 'ajv/dist/2019.js';
import NotificationHandler from '../notifications';
import { PMMessageType } from '../utils';
import commonTypesSchema from '../../../../resources/api_specification/common_types.json' with { type: 'json' };
import specificationSchema from '../../../../resources/api_specification/specification.json' with { type: 'json' };

// eslint-disable-next-line import/no-cycle
import * as remoteProcedures from './remoteProcedures';
import { ClientParams, Endpoints, SpecType } from './utils';
import validateJSON from '../validate-json';

class CustomJSONRPCServerAndClient extends JSONRPCServerAndClient<void, ClientParams> {
    customMethodRegex: RegExp | null = null;

    customMethodReplace: string | null = null;

    customDownloadMethodRegex: RegExp | null = null;

    customDownloadMethodReplace: string | null = null;
}

export class RPCError extends Error {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    constructor(message?: string, public data?: any) {
        super(message);
    }
}

const customMethodRegex = /^custom_.*$/;
const customMethodReplace = 'dataflow_run';
const customDownloadMethodRegex = /^custom_download.*$/;
const customDownloadMethodReplace = 'dataflow_export';

const ajv = new Ajv2019({
    schemas: [commonTypesSchema],
    allowUnionTypes: true,
    strict: true,
});

/**
 * Loads endpoints schemas and assigns $id according to a corresponding key.
 *
 * @param group - Name of the endpoints group;
 * @returns Loaded endpoints.
 */
function loadEndpoints(
    group: 'frontend_endpoints' | 'backend_endpoints' | 'external_endpoints',
): Endpoints {
    const endpoints = specificationSchema[group];
    Object.entries(endpoints).forEach(([key, value]) => {
        const typedValue = value as { params?: { $id?: string}, returns?: { $id?: string }};
        if (typedValue.params) typedValue.params.$id = `${key}_params`;
        if (typedValue.returns) typedValue.returns.$id = `${key}_returns`;
    });
    return endpoints;
}

export const frontendEndpoints = loadEndpoints('frontend_endpoints');
export const backendEndpoints = loadEndpoints('backend_endpoints');
export const externalEndpoints = loadEndpoints('external_endpoints');

// This should become part of the testing suite at some point
let invalidDefinition;
try {
    [frontendEndpoints, backendEndpoints, externalEndpoints].forEach((endpoints) => {
        Object.entries(endpoints).forEach(([definitionName, definition]) => {
            invalidDefinition = definitionName;
            ajv.compile(definition.params);
            ajv.compile(definition.returns ?? {});
        });
    });
} catch (exception) {
    throw new Error(`Procedures specification schema '${invalidDefinition}' is incorrect: ${exception}`);
}

const validateRequestResponse = (
    schema: object,
    data: any, // eslint-disable-line @typescript-eslint/no-explicit-any
    id?: JSONRPCID,
): JSONRPCErrorResponse | undefined => {
    const errors = validateJSON(ajv, schema, data);

    if (!errors.length) return undefined;

    const message = `Request method does not match specification`;
    if (id === undefined) throw new RPCError(message, errors);
    return createJSONRPCErrorResponse(id, PMMessageType.ERROR, message, errors);
};

/**
 * Middleware that validates received requests.
 */
const validateServerRequestResponse = async (
    next: JSONRPCServerMiddlewareNext<void>,
    request: JSONRPCRequest,
    serverParams: void,
): Promise<JSONRPCResponse | null> => {
    // request validation
    if (!(request.method in frontendEndpoints)) {
        if (request.id !== undefined) return createJSONRPCErrorResponse(request.id, 1, 'Requested method does not exist');
        throw new Error('Requested method does not exist');
    }
    const schema = frontendEndpoints[request.method];
    const requestError = validateRequestResponse(schema.params, request.params ?? {}, request.id);
    if (requestError) return requestError;
    const response = await next(request, serverParams);
    if (request.id === undefined) return null;

    // response validation
    if (response?.result !== undefined && schema.returns !== null) {
        const responseError = validateRequestResponse(schema.returns, response.result, response.id);
        if (responseError) return responseError;
    }
    return response;
};

let jsonRPCID = 1;
// eslint-disable-next-line no-plusplus
const createID = () => jsonRPCID++;
const requestSchema = new Map<number | string, SpecType>();

let jsonRPCServer: CustomJSONRPCServerAndClient;

const validateClientResponse = (response: JSONRPCResponse) => {
    if (response.result && response.id && requestSchema.get(response.id)?.returns) {
        const responseError = validateRequestResponse(
            // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
            requestSchema.get(response.id)!.returns!,
            response.result,
            response.id,
        );
        if (responseError) return responseError;
    }
    return response;
};

class CustomJSONRPCClient<C> extends JSONRPCClient<C> {
    receive(responses: JSONRPCResponse | JSONRPCResponse[]): void {
        // eslint-disable-next-line no-param-reassign
        responses = Array.isArray(responses) ? responses : [responses];
        super.receive(responses.map(validateClientResponse));
    }
}

/**
 * Function that creates JSON-RPC client-server and defines how messages are send and received.
 */
function createServer() {
    // Create JSON-RPC server
    jsonRPCServer = new CustomJSONRPCServerAndClient(
        new JSONRPCServer(),
        new CustomJSONRPCClient(async (request: JSONRPCRequest, { externalApp }) => {
            if (!externalApp) {
                throw new Error('Missing backend.');
            }
            const isCustomMethod = customMethodRegex.test(request.method);
            const isCustomDownloadMethod = customDownloadMethodRegex.test(request.method);
            let { method } = request;
            if (isCustomDownloadMethod) {
                method = customDownloadMethodReplace;
            } else if (isCustomMethod) {
                method = customMethodReplace;
            }
            // request validation
            if (!(method in externalEndpoints) && !(method in backendEndpoints)) {
                throw new Error('Requested method not known');
            }

            const endpoints = (method in externalEndpoints) ?
                externalEndpoints : backendEndpoints;
            const schema = endpoints[method];
            // TODO: Improve validation for custom methods
            if (!isCustomMethod) {
                const requestError =
                    validateRequestResponse(schema.params, request.params ?? {}, request.id);
                if (requestError) {
                    throw new RPCError(requestError.error.message, requestError.error.data);
                }
            }
            if (request.id) {
                requestSchema.set(request.id, schema);
            }

            // sending request
            const endpoint = (endpoints === backendEndpoints) ? 'backend-api' : 'external-api';
            try {
                externalApp.request(JSON.parse(JSON.stringify(request)), endpoint);
            } catch (exception) {
                return Promise.reject(exception);
            }
            return Promise.resolve();
        }, createID),
    );
    // Add middlewares
    jsonRPCServer.server.applyMiddleware(
        validateServerRequestResponse as JSONRPCServerMiddleware<void>);
    // Register JSON-RPC methods
    Object.entries(remoteProcedures).forEach(([name, func]) => {
        if (typeof (func) === 'function' && name in frontendEndpoints) jsonRPCServer.addMethod(name, func);
        else if (typeof (func) === 'function') {
            NotificationHandler.showToast('warning', `Function ${name} was not registered as RPC method`);
        }
    });

    jsonRPCServer.customMethodRegex = customMethodRegex;
    jsonRPCServer.customMethodReplace = customMethodReplace;
    jsonRPCServer.customDownloadMethodRegex = customDownloadMethodRegex;
    jsonRPCServer.customDownloadMethodReplace = customDownloadMethodReplace;
}

const obj = {
    get jsonRPC() {
        if (!jsonRPCServer) createServer();
        return jsonRPCServer;
    },
};
export default obj.jsonRPC;
