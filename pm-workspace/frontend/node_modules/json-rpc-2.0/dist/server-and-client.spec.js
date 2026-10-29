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
var mocha_1 = require("mocha");
var chai_1 = require("chai");
var sinon = require("sinon");
var _1 = require(".");
var client_1 = require("./client");
(0, mocha_1.describe)("JSONRPCServerAndClient", function () {
    var serverAndClient1;
    var serverAndClient2;
    (0, mocha_1.beforeEach)(function () {
        serverAndClient1 = new _1.JSONRPCServerAndClient(new _1.JSONRPCServer(), new client_1.JSONRPCClient(function (payload) {
            return serverAndClient2.receiveAndSend(payload, undefined);
        }));
        serverAndClient2 = new _1.JSONRPCServerAndClient(new _1.JSONRPCServer(), new client_1.JSONRPCClient(function (payload, params) {
            return serverAndClient1.receiveAndSend(payload, params);
        }));
        serverAndClient1.addMethod("echo1", function (_a) {
            var message = _a.message;
            return message;
        });
        serverAndClient1.addMethodAdvanced("echo1-2", function (jsonRPCRequest, params) { return __awaiter(void 0, void 0, void 0, function () {
            return __generator(this, function (_a) {
                return [2 /*return*/, ({
                        jsonrpc: _1.JSONRPC,
                        id: jsonRPCRequest.id,
                        result: "".concat(params === null || params === void 0 ? void 0 : params.userID, " said ").concat(jsonRPCRequest.params.message),
                    })];
            });
        }); });
        serverAndClient2.addMethod("echo2", function (_a) {
            var message = _a.message;
            return message;
        });
    });
    afterEach(function () {
        sinon.restore();
    });
    (0, mocha_1.describe)("requesting from server 1", function () {
        var result;
        (0, mocha_1.beforeEach)(function () { return __awaiter(void 0, void 0, void 0, function () {
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, serverAndClient1.request("echo2", { message: "foo" })];
                    case 1:
                        result = _a.sent();
                        return [2 /*return*/];
                }
            });
        }); });
        (0, mocha_1.it)("should request to server 2", function () {
            (0, chai_1.expect)(result).to.equal("foo");
        });
    });
    (0, mocha_1.describe)("requesting from server 2", function () {
        var result;
        (0, mocha_1.beforeEach)(function () { return __awaiter(void 0, void 0, void 0, function () {
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, serverAndClient2.request("echo1", { message: "bar" })];
                    case 1:
                        result = _a.sent();
                        return [2 /*return*/];
                }
            });
        }); });
        (0, mocha_1.it)("should request to server 1", function () {
            (0, chai_1.expect)(result).to.equal("bar");
        });
    });
    (0, mocha_1.describe)("requesting from server 1 using advanced method", function () {
        var response;
        (0, mocha_1.beforeEach)(function () { return __awaiter(void 0, void 0, void 0, function () {
            var request;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        request = {
                            jsonrpc: _1.JSONRPC,
                            id: 0,
                            method: "echo1-2",
                            params: { message: "test" },
                        };
                        return [4 /*yield*/, serverAndClient2.requestAdvanced(request, {
                                userID: "baz",
                            })];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/];
                }
            });
        }); });
        (0, mocha_1.it)("should request to server 1", function () {
            (0, chai_1.expect)(response.result).to.equal("baz said test");
        });
    });
    (0, mocha_1.describe)("requesting in batch", function () {
        var responses;
        (0, mocha_1.beforeEach)(function () { return __awaiter(void 0, void 0, void 0, function () {
            var requests;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        requests = [
                            { jsonrpc: _1.JSONRPC, id: 0, method: "echo2", params: { message: "1" } },
                            { jsonrpc: _1.JSONRPC, id: 1, method: "echo2", params: { message: "2" } },
                        ];
                        return [4 /*yield*/, serverAndClient1.requestAdvanced(requests)];
                    case 1:
                        responses = _a.sent();
                        return [2 /*return*/];
                }
            });
        }); });
        (0, mocha_1.it)("should return responses", function () {
            (0, chai_1.expect)(responses).to.deep.equal([
                { jsonrpc: _1.JSONRPC, id: 0, result: "1" },
                { jsonrpc: _1.JSONRPC, id: 1, result: "2" },
            ]);
        });
    });
    (0, mocha_1.describe)("receiving invalid JSON-RPC message", function () {
        var promise;
        (0, mocha_1.beforeEach)(function () {
            promise = serverAndClient1.receiveAndSend({});
        });
        (0, mocha_1.it)("should fail", function () {
            return promise.then(function () { return Promise.reject(new Error("Expected to fail")); }, function () { return undefined; });
        });
    });
    (0, mocha_1.describe)("having a pending request", function () {
        var promise;
        var resolve;
        (0, mocha_1.beforeEach)(function () {
            serverAndClient2.addMethod("hang", function () {
                return new Promise(function (givenResolve) { return (resolve = givenResolve); });
            });
            promise = serverAndClient1.request("hang", undefined);
        });
        (0, mocha_1.describe)("rejecting all pending requests", function () {
            var message;
            (0, mocha_1.beforeEach)(function () {
                message = "Connection is closed.";
                serverAndClient1.rejectAllPendingRequests(message);
                resolve();
            });
            (0, mocha_1.it)("should reject the pending request", function () {
                return promise.then(function () { return Promise.reject(new Error("Expected to fail")); }, function (error) { return (0, chai_1.expect)(error.message).to.equal(message); });
            });
        });
    });
    (0, mocha_1.describe)("requesting with timeout", function () {
        var fakeTimers;
        var delay;
        var resolve;
        var promise;
        (0, mocha_1.beforeEach)(function () {
            fakeTimers = sinon.useFakeTimers();
            delay = 100;
            serverAndClient2.addMethod("timeout", function () { return new Promise(function (givenResolve) { return (resolve = givenResolve); }); });
            promise = serverAndClient1.timeout(delay).request("timeout");
        });
        (0, mocha_1.describe)("timing out", function () {
            (0, mocha_1.beforeEach)(function () {
                fakeTimers.tick(delay);
                resolve();
            });
            (0, mocha_1.it)("should reject", function () {
                return promise.then(function () { return Promise.reject(new Error("Expected to fail")); }, function () { return undefined; });
            });
        });
        (0, mocha_1.describe)("not timing out", function () {
            (0, mocha_1.beforeEach)(function () {
                resolve();
            });
            (0, mocha_1.it)("should succeed", function () {
                return promise;
            });
        });
    });
});
