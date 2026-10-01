/*
 * Copyright (c) 2022-2024 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { createJSONRPCErrorResponse, createJSONRPCSuccessResponse, JSONRPCErrorCode } from 'json-rpc-2.0';
import { charset } from 'mime-types';
import { ref } from 'vue';
import {
    backendApiUrl, PMMessageType, JSONRPCCustomErrorCode, loadingScreen,
} from '../utils';

// eslint-disable-next-line import/no-cycle
import jsonRPC from './rpcCommunication';
import runInfo from './runInformation';
import NotificationHandler from '../notifications';
import EditorManager, { loadJsonFromRemoteLocation } from '../EditorManager';
import ExternalBackendApp from './externalApp/backend.ts';
import ConnectionManager from './connectionManager.ts';
import ExternalFrontendApp from './externalApp/frontend.ts';

// Default external application capabilities
const defaultAppCapabilities = [];

/**
 * Creates notifications based on response received from external application.
 */
export function handleExternalAppResponse(response) {
    // Status is HTTPCodes.OK so a message from the application is received.
    if (response.type === PMMessageType.OK) {
        NotificationHandler.terminalLog('info', response.content);
    } else if (response.type === PMMessageType.ERROR) {
        NotificationHandler.terminalLog('error', `Error occurred: ${response.content}`, response.content);
    } else if (response.type === PMMessageType.WARNING) {
        NotificationHandler.terminalLog('warning', `Warning: ${response.content}`, response.content);
    }
}

/**
 * Creates notifications based on specification.
 * Returns `true` if error appeared.
 */
function handleSpecificationResult({ errors, warnings, info }, errorTitle, warningTitle, infTitle) {
    if (Array.isArray(warnings) && warnings.length) {
        NotificationHandler.terminalLog('warning', warningTitle, warnings);
    }
    if (Array.isArray(errors) && errors.length) {
        NotificationHandler.terminalLog('error', errorTitle, errors);
        return true;
    }
    if (Array.isArray(info) && info.length) {
        NotificationHandler.terminalLog('info', infTitle, info);
    }
    return false;
}

/**
 * Creates notification about mismatched specification versions
 */
function showVersionError(currentVersion, specification) {
    let usedVersion;
    try {
        usedVersion = JSON.parse(specification).version;
    } catch { return; }
    if (currentVersion === usedVersion) return;
    NotificationHandler.terminalLog(
        'error',
        'Mismatched specification version',
        `Specification version (${usedVersion}) differs from the current version (${currentVersion}). It may result in unexpected behaviour.` +
        'Please refer to https://antmicro.github.io/kenning-pipeline-manager/changelogs.html to see if that is the case.',
    );
}

/**
 * Formats received error and logs it.
 *
 * @param {string} type - Notification type.
 * @param {Error|import('./rpcCommunication').RPCError} error - Captured error.
 * @param {string} [message] - Custom error description.
 */
function RPCTerminalLog(type, error, message) {
    const args = message !== undefined
        ? [message, [error.message, ...(error.data ?? [])]]
        : [error.message, error.data];

    NotificationHandler.terminalLog(type, ...args);
}

class ExternalApplicationManager {
    editorManager = EditorManager.getEditorManagerInstance();

    appCapabilities = ref([]);

    externalApp = null;

    connectionHook = null;

    disconnectionHook = null;

    backend = false;

    constructor() {
        /**
         * Assigning the ExternalApplicationManager instance to the EditorManager properties.
         * This assignment bypasses the cycle problem if the ExternalApplicationManager was normally
         * imported in the EditorManager.
         */
        this.editorManager.externalApplicationManager = this;

        // eslint-disable-next-line no-param-reassign
        const resetProgress = () => runInfo.forEach((itemInfo) => { itemInfo.inProgress = false; });
        this.connectionManager = new ConnectionManager(
            (externalApp) => this.initializeConnection(externalApp),
            (_) => {
                resetProgress();
                if (this.disconnectionHook !== null) this.disconnectionHook();
            },
        );
        this.connectionManager.poll();
    }

    isExternalAppAvailable() {
        return this.externalApp !== null;
    }

    isConnected() {
        return Array.from(this.connectionManager.connected.values()).some(Boolean);
    }

    usesBackend() {
        return this.backend;
    }

    /**
     * Wrap the RPC request with the current application.
     *
     * @param {string} method - API endpoint.
     * @param {any} params - Parameters for the given API endpoint.
     * @returns RPC response.
     */
    async request(method, params) {
        const { externalApp } = this;
        return jsonRPC.request(method, params, { externalApp });
    }

    /**
     * Event handler that asks the backend to send a dataflow specification.
     * If the backend did not manage to send it the user is alerted with a feedback message.
     * Otherwise the specification is passed to the editor that renders a new environment.
     */
    async requestSpecification() {
        let message = 'Unknown error';
        try {
            const data = await this.request('specification_get');

            if (data.type === PMMessageType.OK) {
                const specification = data.content;

                await this.updateSpecification(specification);

                NotificationHandler.terminalLog('info', 'Specification loaded successfully');
            } else if (data.type === PMMessageType.WARNING) {
                message = data.content;
                NotificationHandler.terminalLog('warning', message);
            } else if (data.type === PMMessageType.ERROR) {
                message = data.content;
                NotificationHandler.terminalLog('error', message);
            }
        } catch (error) {
            RPCTerminalLog('error', error);
        }
    }

    async conditionalLoadingScreen(load, callback) {
        const { setLoad } = this.editorManager.baklavaView.editor.events;
        return loadingScreen(callback, setLoad, { show: load });
    }

    async validateSpecification(specification) {
        if (typeof specification === 'string' || specification instanceof String) {
            const [success, specificationOrError] = await loadJsonFromRemoteLocation(specification);
            if (!success) {
                NotificationHandler.terminalLog('error', 'Specification is invalid', specificationOrError);
                return Promise.resolve();
            }

            // eslint-disable-next-line no-param-reassign
            specification = specificationOrError;
        }

        if (handleSpecificationResult(
            EditorManager.validateSpecification(specification),
            'Specification is invalid',
            'Warnings when validating specification',
            'Validated specification',
        )) {
            showVersionError(
                this.editorManager.specificationVersion,
                specification,
            );
            return Promise.resolve();
        }

        return specification;
    }

    async preprocessSpecification(specification, { urloverrides, tryMinify } = {}) {
        // eslint-disable-next-line no-param-reassign
        specification = await this.validateSpecification(specification);
        if (!specification) return Promise.resolve();

        if (typeof tryMinify === 'string' || tryMinify instanceof String) {
            const [success, dataflowOrError] = await loadJsonFromRemoteLocation(tryMinify);
            if (!success) {
                NotificationHandler.terminalLog('error', 'Dataflow is invalid', dataflowOrError);
                return Promise.resolve();
            }

            // eslint-disable-next-line no-param-reassign
            tryMinify = dataflowOrError;
        }

        const result = await this.editorManager.preprocessSpecification(specification, {
            urloverrides, tryMinify,
        });
        const error = handleSpecificationResult(
            result,
            'Errors when preprocessing specification',
            'Warnings when preprocessing specification',
            'Specification preprocessed',
        );
        if (error) return Promise.resolve();
        return result.specification;
    }

    async updateSpecification(specification, { urloverrides, tryMinify } = {}) {
        // eslint-disable-next-line no-param-reassign
        specification = await this.validateSpecification(specification);
        if (!specification) return Promise.resolve();

        if (typeof tryMinify === 'string' || tryMinify instanceof String) {
            const [success, dataflowOrError] = await loadJsonFromRemoteLocation(tryMinify);
            if (!success) {
                NotificationHandler.terminalLog('error', 'Dataflow is invalid', dataflowOrError);
                return Promise.resolve();
            }

            // eslint-disable-next-line no-param-reassign
            tryMinify = dataflowOrError;
        }

        const error = handleSpecificationResult(
            await this.editorManager.updateEditorSpecification(
                specification, false, true, urloverrides, tryMinify,
            ),
            'Errors when loading specification',
            'Warnings when loading specification',
            'Loaded specification',
        );
        if (error) {
            showVersionError(
                this.editorManager.specificationVersion,
                specification,
            );
        }
        return error;
    }

    async updateDataflow(dataflow) {
        if (typeof dataflow === 'string' || dataflow instanceof String) {
            const [success, dataflowOrError] = await loadJsonFromRemoteLocation(dataflow);
            if (!success) {
                NotificationHandler.terminalLog('error', 'Dataflow is invalid', dataflowOrError);
                return;
            }

            // eslint-disable-next-line no-param-reassign
            dataflow = dataflowOrError;
        }
        const { errors, warnings, info } = await this.editorManager.loadDataflow(dataflow);
        if (Array.isArray(errors) && errors.length) {
            NotificationHandler.terminalLog('error', 'Dataflow is invalid', errors);
            if (Array.isArray(info) && info.length) {
                NotificationHandler.terminalLog(
                    'error',
                    'Mismatched specification version',
                    `${info} Please refer to https://antmicro.github.io/kenning-pipeline-manager/changelogs.html to see if that is the case.`,
                );
            }
        } else if (Array.isArray(warnings) && warnings.length) {
            NotificationHandler.terminalLog('warning', 'Dataflow loaded with warning', warnings);
        } else if (Array.isArray(info) && info.length) {
            NotificationHandler.terminalLog('info', 'Dataflow loaded', info);
        }
    }

    /**
    * Provides capabilities of the third-party app connected via server
    * to Pipeline Manager's frontend.
    */
    async requestAppCapabilities() {
        try {
            const appCapabilities = await this.request('app_capabilities_get');
            this.appCapabilities.value = appCapabilities;
        } catch (error) {
            this.appCapabilities.value = defaultAppCapabilities;
            RPCTerminalLog('warning', error, 'Application capabilities cannot be retrieved, using defaults');
        }
    }

    /**
    * Handles dataflow export.
    */
    async requestDataflowExport() {
        const dataflow = this.editorManager.saveDataflow();
        if (!dataflow) return false;

        let data;
        try {
            data = await this.request('dataflow_export', { dataflow });
        } catch (error) {
            // The connection was closed.
            RPCTerminalLog('error', error);
            return false;
        }

        // Status is HTTPCodes.OK so a message from the application is received.
        if (data.type === PMMessageType.OK) {
            return data;
        }

        if (data.type === PMMessageType.ERROR) {
            NotificationHandler.terminalLog('error', `Error occurred: ${data.content}`, data.content);
        } else if (data.type === PMMessageType.WARNING) {
            NotificationHandler.terminalLog('warning', `Warning: ${data.content}`, data.content);
        }
        return false;
    }

    /**
     * Event handler that loads a current dataflow from the editor and sends a request
     * to the backend based on the action argument.
     * The user is alerted with a feedback message.
     *
     * @param procedureName Name of the requested procedure.
     * @param requireResponse Whether response from external application should be awaited.
     */
    async requestDataflowAction(procedureName, requireResponse) {
        const dataflow = this.editorManager.saveDataflow();
        const runProcedureInfo = runInfo.get(procedureName);
        if (!dataflow) return;
        const validatedProcedureName = (jsonRPC.customMethodRegex.test(procedureName)) ?
            jsonRPC.customMethodReplace : procedureName;

        if (validatedProcedureName === 'dataflow_run') {
            if (runProcedureInfo.inProgress) {
                NotificationHandler.showToast('error', 'Previous run has not finished, cannot process this request');
                return;
            }
            NotificationHandler.showToast('info', 'Running dataflow');
        }
        if (requireResponse) {
            runProcedureInfo.inProgress = true;

            let data;
            try {
                if (validatedProcedureName.startsWith('dataflow_')) {
                    data = await this.request(procedureName, { dataflow });
                } else {
                    data = await this.request(validatedProcedureName);
                }
            } catch (error) {
                // The connection was closed
                RPCTerminalLog('error', error, 'Cannot create a request');
                runProcedureInfo.inProgress = false;
                return;
            }

            handleExternalAppResponse(data);
            runProcedureInfo.inProgress = false;
        } else if (validatedProcedureName.startsWith('dataflow_')) {
            this.request(procedureName, { dataflow });
        } else {
            this.request(validatedProcedureName);
        }
    }

    /**
     * Event handler that check if remote procedure is running and send stop request.
     * The user is alerted with a feedback message.
     *
     * @param procedureName Name of the requested procedure.
     */
    // eslint-disable-next-line class-methods-use-this
    async requestDataflowStop(procedureName) {
        if (!runInfo.get(procedureName).inProgress) {
            NotificationHandler.showToast('error', 'Nothing to stop, no ongoing jobs running');
            return;
        }

        try {
            const response = await this.request('dataflow_stop', { method: procedureName });
            handleExternalAppResponse(response);
        } catch (error) {
            RPCTerminalLog('error', error);
        }
    }

    /**
     * Event handler that loads a file and asks the backend to delegate this operation
     * to the external application to parse it into the Pipeline Manager format
     * so that it can be loaded into the editor.
     * It the validation is successful it is loaded as the current dataflow.
     * Otherwise the user is alerted with a feedback message.
     *
     * @param dataflow Dataflow to be impported
     */
    async importDataflow() {
        const file = document.getElementById('request-dataflow-button').files[0];
        if (!file) return;

        const reader = new FileReader();
        const encoding = charset(file.type);
        const readerPromise = new Promise((resolve) => {
            reader.onloadend = () => {
                resolve(
                    (encoding) ? reader.result : reader.result.replace(/data:.*;base64,/, ''),
                );
            };
        });
        // Read file as text if possible, otherwise return base64 string
        if (encoding) {
            reader.readAsText(file, encoding);
        } else {
            reader.readAsDataURL(file);
        }
        const dataflow = await readerPromise;
        if (!dataflow) {
            NotificationHandler.showToast('error', 'File cannot be loaded');
            return;
        }

        try {
            const data = await this.request('dataflow_import', { external_application_dataflow: dataflow, mime: file.type, base64: !encoding });
            if (data.type === PMMessageType.OK) {
                const { errors, warnings } = await this.editorManager.loadDataflow(data.content);
                if (Array.isArray(errors) && errors.length) {
                    NotificationHandler.terminalLog('error', 'Dataflow is invalid', errors);
                } else if (Array.isArray(warnings) && warnings.length) {
                    NotificationHandler.terminalLog('warning', 'Dataflow imported with warning', warnings);
                } else {
                    NotificationHandler.showToast('info', 'Imported dataflow');
                }
            } else if (data.type === PMMessageType.ERROR) {
                const message = data.content;
                NotificationHandler.terminalLog('error', `Error occurred: ${data.content}`, message);
            } else if (data.type === PMMessageType.WARNING) {
                NotificationHandler.terminalLog('warning', `Warning: ${data.content}`, 'Imported dataflow');
            }
        } catch (error) {
            RPCTerminalLog('error', error);
        }
    }

    /**
     * Send information to external application about changed values (like nodes, connections,
     * positions, properties).
     *
     * @param method Name of the JSON-RPC method.
     * @param changedProperties Params of the send request, should contain changed values.
     */
    async notifyAboutChange(method, changedProperties) {
        if (!this.isConnected() || !this.editorManager.notifyWhenChanged) return;
        try {
            await this.request(method, changedProperties);
        } catch (error) {
            RPCTerminalLog('warning', error, `Notifying about change failed (${method})`);
        }
    }

    /**
     * Send information to external application about input received by writable terminal.
     *
     * @param terminal Name Name of the terminal
     * @param message Input provided for the terminal
     */
    async requestTerminalRead(terminalName, message) {
        if (!this.isConnected()) return;
        try {
            await this.request('terminal_read', { name: terminalName, message });
        } catch (error) {
            RPCTerminalLog('warning', error, 'Sending terminal input failed');
        }
    }

    /**
     * Function used to initialize connection with the external application and request
     * specification. Should be called after DOM is created.
     *
     * It checks whether a connection is established.
     * If it is then it just requests a specification.
     * If it is not then it opens a TCP port, wait for the application to connect and then
     * requests specification.
     *
     * @param {import('./externalApp/base.ts').ExternalApp} externalApp
     */
    async initializeConnection(_externalApp) {
        if (this.backend) {
            NotificationHandler.terminalLog(
                'info',
                `External application connected successfully`,
                undefined,
            );
        }

        await Promise.all([
            this.requestSpecification(),
            this.requestAppCapabilities(),
        ]);

        try {
            await this.request('frontend_on_connect');
        } catch (error) {
            if (error.code !== JSONRPCErrorCode.MethodNotFound &&
                error.code !== JSONRPCCustomErrorCode.EXTERNAL_APPLICATION_NOT_CONNECTED) {
                RPCTerminalLog('error', error);
            }
        }

        if (this.connectionHook !== null) this.connectionHook();
    }

    registerConnectionHook(connectionHook) {
        this.connectionHook = connectionHook;
    }

    registerDisconnectionHook(disconnectionHook) {
        this.disconnectionHook = disconnectionHook;
    }

    /**
     * Registers external application and calls connection hook, if exists.
     *
     * @param {import('./externalApp/base.ts').ExternalApp} externalApp - External application.
     */
    registerApplication(externalApp) {
        this.externalApp = externalApp;
        this.connectionManager.add(externalApp);

        if (this.connectionHook !== null) this.connectionHook();
    }

    /**
     * Registers external backend application.
     *
     * @param {string} url - Backend URL.
     */
    registerBackendApplication(url) {
        this.registerApplication(new ExternalBackendApp(url, jsonRPC));
        this.backend = true;
    }

    /**
     * Registers external frontend.
     *
     * @param {Window} sourceWindow - Wrapping window.
     * @param {import('json-rpc-2.0').JSONRPCRequest} request - Wrapping window.
     * @returns {import('json-rpc-2.0').JSONRPCResponse} Response with success or error message.
     */
    registerFrontendApplication(sourceWindow, request) {
        const logAndRespond = (msgType, msg) => {
            const [logType, response] = {
                [PMMessageType.ERROR]: ['error', createJSONRPCErrorResponse(request.id, msgType, msg)],
                [PMMessageType.WARNING]: ['warning', createJSONRPCSuccessResponse(request.id, msg)],
                [PMMessageType.OK]: ['info', createJSONRPCSuccessResponse(request.id, msg)],
            }[msgType];
            if (msgType !== PMMessageType.OK) {
                NotificationHandler.terminalLog(logType, msg);
            }
            return response;
        };

        if (sourceWindow === window) return logAndRespond(PMMessageType.ERROR, 'External frontend cannot be a Pipeline Manager itself');
        if (this.externalApp !== null) {
            this.connectionManager.remove(this.externalApp);
            logAndRespond(PMMessageType.WARNING, 'Replacing current external application.');
        }

        this.registerApplication(new ExternalFrontendApp(sourceWindow));
        this.backend = false;
        return logAndRespond(PMMessageType.OK, 'Registered external frontend successfully');
    }
}

let externalApplicationManager;

/**
 * Function managing External Application Manager singleton.
 *
 * @returns {ExternalApplicationManager} Instance of External Application Manager.
 */
export default function getExternalApplicationManager() {
    if (!externalApplicationManager) {
        externalApplicationManager = new ExternalApplicationManager();
        if (backendApiUrl) externalApplicationManager.registerBackendApplication(backendApiUrl);
    }
    return externalApplicationManager;
}
