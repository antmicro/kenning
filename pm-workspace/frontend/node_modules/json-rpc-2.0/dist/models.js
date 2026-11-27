"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        if (typeof b !== "function" && b !== null)
            throw new TypeError("Class extends value " + String(b) + " is not a constructor or null");
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.createJSONRPCNotification = exports.createJSONRPCRequest = exports.createJSONRPCSuccessResponse = exports.createJSONRPCErrorResponse = exports.JSONRPCErrorCode = exports.JSONRPCErrorException = exports.isJSONRPCResponses = exports.isJSONRPCResponse = exports.isJSONRPCRequests = exports.isJSONRPCRequest = exports.isJSONRPCID = exports.JSONRPC = void 0;
exports.JSONRPC = "2.0";
var isJSONRPCID = function (id) {
    return typeof id === "string" || typeof id === "number" || id === null;
};
exports.isJSONRPCID = isJSONRPCID;
var isJSONRPCRequest = function (payload) {
    return (payload.jsonrpc === exports.JSONRPC &&
        payload.method !== undefined &&
        payload.result === undefined &&
        payload.error === undefined);
};
exports.isJSONRPCRequest = isJSONRPCRequest;
var isJSONRPCRequests = function (payload) {
    return Array.isArray(payload) && payload.every(exports.isJSONRPCRequest);
};
exports.isJSONRPCRequests = isJSONRPCRequests;
var isJSONRPCResponse = function (payload) {
    return (payload.jsonrpc === exports.JSONRPC &&
        payload.id !== undefined &&
        (payload.result !== undefined || payload.error !== undefined));
};
exports.isJSONRPCResponse = isJSONRPCResponse;
var isJSONRPCResponses = function (payload) {
    return Array.isArray(payload) && payload.every(exports.isJSONRPCResponse);
};
exports.isJSONRPCResponses = isJSONRPCResponses;
var createJSONRPCError = function (code, message, data) {
    var error = { code: code, message: message };
    if (data != null) {
        error.data = data;
    }
    return error;
};
var JSONRPCErrorException = /** @class */ (function (_super) {
    __extends(JSONRPCErrorException, _super);
    function JSONRPCErrorException(message, code, data) {
        var _this = _super.call(this, message) || this;
        // Manually set the prototype to fix TypeScript issue:
        // https://github.com/Microsoft/TypeScript-wiki/blob/main/Breaking-Changes.md#extending-built-ins-like-error-array-and-map-may-no-longer-work
        Object.setPrototypeOf(_this, JSONRPCErrorException.prototype);
        _this.code = code;
        _this.data = data;
        return _this;
    }
    JSONRPCErrorException.prototype.toObject = function () {
        return createJSONRPCError(this.code, this.message, this.data);
    };
    return JSONRPCErrorException;
}(Error));
exports.JSONRPCErrorException = JSONRPCErrorException;
var JSONRPCErrorCode;
(function (JSONRPCErrorCode) {
    JSONRPCErrorCode[JSONRPCErrorCode["ParseError"] = -32700] = "ParseError";
    JSONRPCErrorCode[JSONRPCErrorCode["InvalidRequest"] = -32600] = "InvalidRequest";
    JSONRPCErrorCode[JSONRPCErrorCode["MethodNotFound"] = -32601] = "MethodNotFound";
    JSONRPCErrorCode[JSONRPCErrorCode["InvalidParams"] = -32602] = "InvalidParams";
    JSONRPCErrorCode[JSONRPCErrorCode["InternalError"] = -32603] = "InternalError";
})(JSONRPCErrorCode = exports.JSONRPCErrorCode || (exports.JSONRPCErrorCode = {}));
var createJSONRPCErrorResponse = function (id, code, message, data) {
    return {
        jsonrpc: exports.JSONRPC,
        id: id,
        error: createJSONRPCError(code, message, data),
    };
};
exports.createJSONRPCErrorResponse = createJSONRPCErrorResponse;
var createJSONRPCSuccessResponse = function (id, result) {
    return {
        jsonrpc: exports.JSONRPC,
        id: id,
        result: result !== null && result !== void 0 ? result : null,
    };
};
exports.createJSONRPCSuccessResponse = createJSONRPCSuccessResponse;
var createJSONRPCRequest = function (id, method, params) {
    return {
        jsonrpc: exports.JSONRPC,
        id: id,
        method: method,
        params: params,
    };
};
exports.createJSONRPCRequest = createJSONRPCRequest;
var createJSONRPCNotification = function (method, params) {
    return {
        jsonrpc: exports.JSONRPC,
        method: method,
        params: params,
    };
};
exports.createJSONRPCNotification = createJSONRPCNotification;
