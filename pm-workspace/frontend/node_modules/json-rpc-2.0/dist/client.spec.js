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
(0, mocha_1.describe)("JSONRPCClient", function () {
    var client;
    var id;
    var lastRequest;
    var lastClientParams;
    var resolve;
    var reject;
    (0, mocha_1.beforeEach)(function () {
        id = 0;
        lastRequest = undefined;
        resolve = undefined;
        reject = undefined;
        var send = function (request, clientParams) {
            lastRequest = request;
            lastClientParams = clientParams;
            return new Promise(function (givenResolve, givenReject) {
                resolve = givenResolve;
                reject = givenReject;
            });
        };
        client = new _1.JSONRPCClient(send, function () { return ++id; });
    });
    afterEach(function () {
        sinon.restore();
    });
    (0, mocha_1.describe)("requesting", function () {
        var result;
        var error;
        var promise;
        (0, mocha_1.beforeEach)(function () {
            result = undefined;
            error = undefined;
            promise = client.request("foo", ["bar"], { token: "" }).then(function (givenResult) { return (result = givenResult); }, function (givenError) { return (error = givenError); });
        });
        (0, mocha_1.it)("should send the request", function () {
            (0, chai_1.expect)(lastRequest).to.deep.equal({
                jsonrpc: _1.JSONRPC,
                id: id,
                method: "foo",
                params: ["bar"],
            });
        });
        (0, mocha_1.describe)("succeeded on client side", function () {
            (0, mocha_1.beforeEach)(function () {
                resolve();
            });
            (0, mocha_1.describe)("and succeeded on server side too", function () {
                var response;
                (0, mocha_1.beforeEach)(function () {
                    response = {
                        jsonrpc: _1.JSONRPC,
                        id: id,
                        result: "foo",
                    };
                    client.receive(response);
                    return promise;
                });
                (0, mocha_1.it)("should resolve the result", function () {
                    (0, chai_1.expect)(result).to.equal(response.result);
                });
            });
            (0, mocha_1.describe)("and succeeded on server side with falsy but defined result", function () {
                (0, mocha_1.beforeEach)(function () {
                    client.receive({
                        jsonrpc: _1.JSONRPC,
                        id: id,
                        result: 0,
                    });
                    return promise;
                });
                (0, mocha_1.it)("should resolve the result", function () {
                    (0, chai_1.expect)(result).to.equal(0);
                });
            });
            (0, mocha_1.describe)("but failed on server side", function () {
                var response;
                (0, mocha_1.beforeEach)(function () {
                    response = {
                        jsonrpc: _1.JSONRPC,
                        id: id,
                        error: {
                            code: 0,
                            message: "This is a test. Do not panic.",
                            data: { optional: "data" },
                        },
                    };
                    client.receive(response);
                    return promise;
                });
                (0, mocha_1.it)("should reject with the error message, code and data", function () {
                    (0, chai_1.expect)(error.message).to.equal(response.error.message);
                    (0, chai_1.expect)(error.code).to.equal(response.error.code);
                    (0, chai_1.expect)(error.data).to.equal(response.error.data);
                });
                (0, mocha_1.it)("should reject with a JSONRPCErrorException", function () {
                    (0, chai_1.expect)(error instanceof Error).to.be.true;
                    (0, chai_1.expect)(error instanceof _1.JSONRPCErrorException).to.be.true;
                    (0, chai_1.expect)(error.toObject()).to.deep.equal(response.error);
                });
            });
            (0, mocha_1.describe)("but server responded invalid response", function () {
                (0, mocha_1.describe)("like having both result and error", function () {
                    var response;
                    (0, mocha_1.beforeEach)(function () {
                        response = {
                            jsonrpc: _1.JSONRPC,
                            id: id,
                            result: "foo",
                            error: {
                                code: 0,
                                message: "bar",
                            },
                        };
                        client.receive(response);
                        return promise;
                    });
                    (0, mocha_1.it)("should reject", function () {
                        (0, chai_1.expect)(error).to.not.be.undefined;
                    });
                });
                (0, mocha_1.describe)("like not having both result and error", function () {
                    var response;
                    (0, mocha_1.beforeEach)(function () {
                        response = {
                            jsonrpc: _1.JSONRPC,
                            id: id,
                        };
                        client.receive(response);
                        return promise;
                    });
                    (0, mocha_1.it)("should reject", function () {
                        (0, chai_1.expect)(error).to.not.be.undefined;
                    });
                });
            });
            (0, mocha_1.describe)("but I reject all pending requests", function () {
                var message;
                (0, mocha_1.beforeEach)(function () {
                    message = "Connection is closed.";
                    client.rejectAllPendingRequests(message);
                    return promise;
                });
                (0, mocha_1.it)("should reject the request", function () {
                    (0, chai_1.expect)(error.message).to.equal(message);
                });
                (0, mocha_1.describe)("receiving a response", function () {
                    (0, mocha_1.beforeEach)(function () {
                        client.receive({
                            jsonrpc: _1.JSONRPC,
                            id: id,
                            result: "foo",
                        });
                        return promise;
                    });
                    (0, mocha_1.it)("should not resolve the promise again", function () {
                        (0, chai_1.expect)(result).to.be.undefined;
                    });
                });
            });
        });
        (0, mocha_1.describe)("failed on client side", function () {
            var expected;
            (0, mocha_1.beforeEach)(function () {
                expected = new Error("This is a test. Do not panic.");
                reject(expected);
                return promise;
            });
            (0, mocha_1.it)("should reject the promise", function () {
                (0, chai_1.expect)(error.message).to.equal(expected.message);
            });
        });
        (0, mocha_1.describe)("failed on client side with no error object", function () {
            (0, mocha_1.beforeEach)(function () {
                reject(undefined);
                return promise;
            });
            (0, mocha_1.it)("should reject the promise", function () {
                (0, chai_1.expect)(error.message).to.equal("Failed to send a request");
            });
        });
        (0, mocha_1.describe)("failed on client side with an error object without message", function () {
            (0, mocha_1.beforeEach)(function () {
                reject({});
                return promise;
            });
            (0, mocha_1.it)("should reject the promise", function () {
                (0, chai_1.expect)(error.message).to.equal("Failed to send a request");
            });
        });
    });
    (0, mocha_1.describe)("requesting batch", function () {
        var requests;
        var actualResponses;
        var expectedResponses;
        (0, mocha_1.beforeEach)(function () {
            requests = [
                { jsonrpc: _1.JSONRPC, id: 0, method: "foo" },
                { jsonrpc: _1.JSONRPC, id: 1, method: "foo2" },
            ];
            client
                .requestAdvanced(requests, { token: "" })
                .then(function (responses) { return (actualResponses = responses); });
            resolve();
            expectedResponses = [
                { jsonrpc: _1.JSONRPC, id: 0, result: "foo" },
                { jsonrpc: _1.JSONRPC, id: 1, result: "foo2" },
            ];
            client.receive(expectedResponses);
        });
        (0, mocha_1.it)("should send requests in batch", function () {
            (0, chai_1.expect)(lastRequest).to.deep.equal(requests);
        });
        (0, mocha_1.it)("should return responses", function () {
            (0, chai_1.expect)(actualResponses).to.deep.equal(expectedResponses);
        });
    });
    (0, mocha_1.describe)("requesting with client params", function () {
        var expected;
        (0, mocha_1.beforeEach)(function () {
            expected = { token: "baz" };
            client.request("foo", undefined, expected);
        });
        (0, mocha_1.it)("should pass the client params to send function", function () {
            (0, chai_1.expect)(lastClientParams).to.deep.equal(expected);
        });
    });
    (0, mocha_1.describe)("requesting with timeout", function () {
        var delay;
        var fakeTimers;
        var promise;
        (0, mocha_1.beforeEach)(function () {
            fakeTimers = sinon.useFakeTimers();
            delay = 1000;
            promise = client.timeout(delay).request("foo");
            resolve();
        });
        (0, mocha_1.describe)("timing out", function () {
            (0, mocha_1.beforeEach)(function () {
                fakeTimers.tick(delay);
            });
            (0, mocha_1.it)("should reject", function () {
                return promise.then(function () { return Promise.reject(new Error("Expected to fail")); }, function () { return undefined; });
            });
        });
        (0, mocha_1.describe)("not timing out", function () {
            var result;
            (0, mocha_1.beforeEach)(function () {
                result = "foo";
                client.receive({
                    jsonrpc: _1.JSONRPC,
                    id: lastRequest.id,
                    result: result,
                });
            });
            (0, mocha_1.it)("should respond", function () { return __awaiter(void 0, void 0, void 0, function () {
                var actual;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0: return [4 /*yield*/, promise];
                        case 1:
                            actual = _a.sent();
                            (0, chai_1.expect)(actual).to.equal(result);
                            return [2 /*return*/];
                    }
                });
            }); });
        });
    });
    (0, mocha_1.describe)("requesting advanced with timeout", function () {
        var delay;
        var fakeTimers;
        var promise;
        (0, mocha_1.beforeEach)(function () {
            fakeTimers = sinon.useFakeTimers();
            delay = 1000;
            promise = client.timeout(delay).requestAdvanced({
                jsonrpc: _1.JSONRPC,
                id: ++id,
                method: "foo",
            });
            resolve();
        });
        (0, mocha_1.describe)("timing out", function () {
            (0, mocha_1.beforeEach)(function () {
                fakeTimers.tick(delay);
            });
            (0, mocha_1.it)("should reject", function () {
                return promise.then(function (result) {
                    if (!result.error) {
                        return Promise.reject(new Error("Expected to fail"));
                    }
                });
            });
        });
        (0, mocha_1.describe)("not timing out", function () {
            var result;
            (0, mocha_1.beforeEach)(function () {
                result = {
                    jsonrpc: _1.JSONRPC,
                    id: lastRequest.id,
                    result: result,
                };
                client.receive(result);
            });
            (0, mocha_1.it)("should respond", function () { return __awaiter(void 0, void 0, void 0, function () {
                var actual;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0: return [4 /*yield*/, promise];
                        case 1:
                            actual = _a.sent();
                            (0, chai_1.expect)(actual).to.deep.equal(result);
                            return [2 /*return*/];
                    }
                });
            }); });
        });
    });
    (0, mocha_1.describe)("requesting with custom timeout error response", function () {
        var delay;
        var fakeTimers;
        var errorCode;
        var errorMessage;
        var errorData;
        var promise;
        (0, mocha_1.beforeEach)(function () {
            fakeTimers = sinon.useFakeTimers();
            delay = 1000;
            errorCode = 123;
            errorMessage = "Custom error message";
            errorData = "Custom error data";
            promise = client
                .timeout(delay, function (id) {
                return (0, _1.createJSONRPCErrorResponse)(id, errorCode, errorMessage, errorData);
            })
                .requestAdvanced({
                jsonrpc: _1.JSONRPC,
                id: ++id,
                method: "foo",
            });
            resolve();
        });
        (0, mocha_1.describe)("timing out", function () {
            (0, mocha_1.beforeEach)(function () {
                fakeTimers.tick(delay);
            });
            (0, mocha_1.it)("should reject with the custom error", function () { return __awaiter(void 0, void 0, void 0, function () {
                var actual, expected;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0: return [4 /*yield*/, promise];
                        case 1:
                            actual = _a.sent();
                            expected = {
                                code: errorCode,
                                message: errorMessage,
                                data: errorData,
                            };
                            (0, chai_1.expect)(actual.error).to.deep.equal(expected);
                            return [2 /*return*/];
                    }
                });
            }); });
        });
    });
    (0, mocha_1.describe)("notifying", function () {
        (0, mocha_1.beforeEach)(function () {
            client.notify("foo", ["bar"], { token: "" });
        });
        (0, mocha_1.it)("should send the notification", function () {
            (0, chai_1.expect)(lastRequest).to.deep.equal({
                jsonrpc: _1.JSONRPC,
                method: "foo",
                params: ["bar"],
            });
        });
    });
    (0, mocha_1.describe)("notifying with client params", function () {
        var expected;
        (0, mocha_1.beforeEach)(function () {
            expected = { token: "baz" };
            client.notify("foo", undefined, expected);
        });
        (0, mocha_1.it)("should pass the client params to send function", function () {
            (0, chai_1.expect)(lastClientParams).to.deep.equal(expected);
        });
    });
});
