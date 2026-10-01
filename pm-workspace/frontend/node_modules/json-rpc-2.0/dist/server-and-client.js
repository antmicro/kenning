"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (g && (g = 0, op[0] && (_ = 0)), _) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.JSONRPCServerAndClient = void 0;
var models_1 = require("./models");
var JSONRPCServerAndClient = /** @class */ (function () {
    function JSONRPCServerAndClient(server, client, options) {
        if (options === void 0) { options = {}; }
        var _a;
        this.server = server;
        this.client = client;
        this.errorListener = (_a = options.errorListener) !== null && _a !== void 0 ? _a : console.warn;
    }
    JSONRPCServerAndClient.prototype.applyServerMiddleware = function () {
        var _a;
        var middlewares = [];
        for (var _i = 0; _i < arguments.length; _i++) {
            middlewares[_i] = arguments[_i];
        }
        (_a = this.server).applyMiddleware.apply(_a, middlewares);
    };
    JSONRPCServerAndClient.prototype.hasMethod = function (name) {
        return this.server.hasMethod(name);
    };
    JSONRPCServerAndClient.prototype.addMethod = function (name, method) {
        this.server.addMethod(name, method);
    };
    JSONRPCServerAndClient.prototype.addMethodAdvanced = function (name, method) {
        this.server.addMethodAdvanced(name, method);
    };
    JSONRPCServerAndClient.prototype.timeout = function (delay) {
        return this.client.timeout(delay);
    };
    JSONRPCServerAndClient.prototype.request = function (method, params, clientParams) {
        return this.client.request(method, params, clientParams);
    };
    JSONRPCServerAndClient.prototype.requestAdvanced = function (jsonRPCRequest, clientParams) {
        return this.client.requestAdvanced(jsonRPCRequest, clientParams);
    };
    JSONRPCServerAndClient.prototype.notify = function (method, params, clientParams) {
        this.client.notify(method, params, clientParams);
    };
    JSONRPCServerAndClient.prototype.rejectAllPendingRequests = function (message) {
        this.client.rejectAllPendingRequests(message);
    };
    JSONRPCServerAndClient.prototype.receiveAndSend = function (payload, serverParams, clientParams) {
        return __awaiter(this, void 0, void 0, function () {
            var response, message;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        if (!((0, models_1.isJSONRPCResponse)(payload) || (0, models_1.isJSONRPCResponses)(payload))) return [3 /*break*/, 1];
                        this.client.receive(payload);
                        return [3 /*break*/, 4];
                    case 1:
                        if (!((0, models_1.isJSONRPCRequest)(payload) || (0, models_1.isJSONRPCRequests)(payload))) return [3 /*break*/, 3];
                        return [4 /*yield*/, this.server.receive(payload, serverParams)];
                    case 2:
                        response = _a.sent();
                        if (response) {
                            return [2 /*return*/, this.client.send(response, clientParams)];
                        }
                        return [3 /*break*/, 4];
                    case 3:
                        message = "Received an invalid JSON-RPC message";
                        this.errorListener(message, payload);
                        return [2 /*return*/, Promise.reject(new Error(message))];
                    case 4: return [2 /*return*/];
                }
            });
        });
    };
    return JSONRPCServerAndClient;
}());
exports.JSONRPCServerAndClient = JSONRPCServerAndClient;
