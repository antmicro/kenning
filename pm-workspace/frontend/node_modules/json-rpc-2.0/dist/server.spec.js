"use strict";
var __assign = (this && this.__assign) || function () {
    __assign = Object.assign || function(t) {
        for (var s, i = 1, n = arguments.length; i < n; i++) {
            s = arguments[i];
            for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p))
                t[p] = s[p];
        }
        return t;
    };
    return __assign.apply(this, arguments);
};
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
var _1 = require(".");
(0, mocha_1.describe)("JSONRPCServer", function () {
    var server;
    var response;
    (0, mocha_1.beforeEach)(function () {
        response = null;
        server = new _1.JSONRPCServer();
    });
    var waitUntil = function (predicate) {
        return Promise.resolve().then(function () {
            if (!predicate()) {
                return waitUntil(predicate);
            }
        });
    };
    (0, mocha_1.describe)("having an echo method", function () {
        (0, mocha_1.beforeEach)(function () {
            var echoMethod = function (_a, serverParams) {
                var text = _a.text;
                if (serverParams) {
                    return "".concat(serverParams.userID, " said ").concat(text);
                }
                else {
                    return text;
                }
            };
            server.addMethod("echo", echoMethod);
        });
        (0, mocha_1.describe)("receiving a request to the method", function () {
            (0, mocha_1.beforeEach)(function () {
                return server
                    .receive({
                    jsonrpc: _1.JSONRPC,
                    id: 0,
                    method: "echo",
                    params: { text: "foo" },
                })
                    .then(function (givenResponse) { return (response = givenResponse); });
            });
            (0, mocha_1.it)("should echo the text", function () {
                (0, chai_1.expect)(response).to.deep.equal({
                    jsonrpc: _1.JSONRPC,
                    id: 0,
                    result: "foo",
                });
            });
        });
        (0, mocha_1.describe)("receiving a request to the method with user ID", function () {
            (0, mocha_1.beforeEach)(function () {
                return server
                    .receiveJSON(JSON.stringify({
                    jsonrpc: _1.JSONRPC,
                    id: 0,
                    method: "echo",
                    params: { text: "foo" },
                }), { userID: "bar" })
                    .then(function (givenResponse) { return (response = givenResponse); });
            });
            (0, mocha_1.it)("should echo the text with the user ID", function () {
                (0, chai_1.expect)(response).to.deep.equal({
                    jsonrpc: _1.JSONRPC,
                    id: 0,
                    result: "bar said foo",
                });
            });
        });
    });
    (0, mocha_1.describe)("responding undefined", function () {
        (0, mocha_1.beforeEach)(function () {
            server.addMethod("ack", function () { return undefined; });
            return server
                .receive({ jsonrpc: _1.JSONRPC, id: 0, method: "ack" })
                .then(function (givenResponse) { return (response = givenResponse); });
        });
        (0, mocha_1.it)("should response with null result", function () {
            (0, chai_1.expect)(response).to.deep.equal({
                jsonrpc: _1.JSONRPC,
                id: 0,
                result: null,
            });
        });
    });
    (0, mocha_1.describe)("throwing", function () {
        (0, mocha_1.beforeEach)(function () {
            server.addMethod("throw", function () {
                throw new Error("Test throwing");
            });
            return server
                .receive({ jsonrpc: _1.JSONRPC, id: 0, method: "throw" })
                .then(function (givenResponse) { return (response = givenResponse); });
        });
        (0, mocha_1.it)("should respond error", function () {
            (0, chai_1.expect)(response).to.deep.equal({
                jsonrpc: _1.JSONRPC,
                id: 0,
                error: {
                    code: 0,
                    message: "Test throwing",
                },
            });
        });
    });
    (0, mocha_1.describe)("throwing JSONRPCErrorException", function () {
        var expected;
        (0, mocha_1.beforeEach)(function () {
            expected = {
                message: "thrown",
                code: 1234,
                data: {
                    foo: "bar",
                },
            };
            server.addMethod("throw", function () {
                throw new _1.JSONRPCErrorException(expected.message, expected.code, expected.data);
            });
            return server
                .receive({ jsonrpc: _1.JSONRPC, id: 0, method: "throw" })
                .then(function (givenResponse) { return (response = givenResponse); });
        });
        (0, mocha_1.it)("should respond error with custom code and data", function () {
            (0, chai_1.expect)(response.error).to.deep.equal(expected);
        });
    });
    (0, mocha_1.describe)("rejecting", function () {
        (0, mocha_1.beforeEach)(function () {
            server.addMethodAdvanced("reject", function () {
                return Promise.reject(new Error("Test rejecting"));
            });
            return server
                .receive({ jsonrpc: _1.JSONRPC, id: 0, method: "reject" })
                .then(function (givenResponse) { return (response = givenResponse); });
        });
        (0, mocha_1.it)("should respond error", function () {
            (0, chai_1.expect)(response).to.deep.equal({
                jsonrpc: _1.JSONRPC,
                id: 0,
                error: {
                    code: 0,
                    message: "Test rejecting",
                },
            });
        });
    });
    (0, mocha_1.describe)("responding to a notification", function () {
        (0, mocha_1.beforeEach)(function () {
            server.addMethod("foo", function () { return "foo"; });
            return server
                .receive({ jsonrpc: _1.JSONRPC, method: "foo" })
                .then(function (givenResponse) { return (response = givenResponse); });
        });
        (0, mocha_1.it)("should not respond", function () {
            (0, chai_1.expect)(response).to.be.null;
        });
    });
    (0, mocha_1.describe)("error on a notification", function () {
        (0, mocha_1.beforeEach)(function () {
            server.addMethod("foo", function () { return Promise.reject(new Error("foo")); });
            return server
                .receive({ jsonrpc: _1.JSONRPC, method: "foo" })
                .then(function (givenResponse) { return (response = givenResponse); });
        });
        (0, mocha_1.it)("should not respond", function () {
            (0, chai_1.expect)(response).to.be.null;
        });
    });
    (0, mocha_1.describe)("responding null to a request", function () {
        (0, mocha_1.beforeEach)(function () {
            server.addMethodAdvanced("foo", function () { return Promise.resolve(null); });
            return server
                .receive({
                jsonrpc: _1.JSONRPC,
                id: 0,
                method: "foo",
            })
                .then(function (givenResponse) { return (response = givenResponse); });
        });
        (0, mocha_1.it)("should respond error", function () {
            (0, chai_1.expect)(response).to.deep.equal({
                jsonrpc: _1.JSONRPC,
                id: 0,
                error: {
                    code: _1.JSONRPCErrorCode.InternalError,
                    message: "Internal error",
                },
            });
        });
    });
    (0, mocha_1.describe)("receiving a request to an unknown method", function () {
        (0, mocha_1.beforeEach)(function () {
            return server
                .receive({
                jsonrpc: _1.JSONRPC,
                id: 0,
                method: "foo",
            })
                .then(function (givenResponse) { return (response = givenResponse); });
        });
        (0, mocha_1.it)("should respond error", function () {
            (0, chai_1.expect)(response).to.deep.equal({
                jsonrpc: _1.JSONRPC,
                id: 0,
                error: {
                    code: _1.JSONRPCErrorCode.MethodNotFound,
                    message: "Method not found",
                },
            });
        });
    });
    [{}, "", "invalid JSON"].forEach(function (invalidJSON) {
        (0, mocha_1.describe)("receiving an invalid JSON (".concat(invalidJSON, ")"), function () {
            var response;
            (0, mocha_1.beforeEach)(function () { return __awaiter(void 0, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0: return [4 /*yield*/, server.receiveJSON(invalidJSON)];
                        case 1:
                            response = (_a.sent());
                            return [2 /*return*/];
                    }
                });
            }); });
            (0, mocha_1.it)("should respond an error", function () {
                (0, chai_1.expect)(response.error.code).to.equal(_1.JSONRPCErrorCode.ParseError);
            });
        });
    });
    [
        {},
        { jsonrpc: _1.JSONRPC },
        { jsonrpc: _1.JSONRPC + "invalid", method: "" },
    ].forEach(function (invalidRequest) {
        (0, mocha_1.describe)("receiving an invalid request (".concat(JSON.stringify(invalidRequest), ")"), function () {
            var response;
            (0, mocha_1.beforeEach)(function () { return __awaiter(void 0, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0: return [4 /*yield*/, server.receive(invalidRequest)];
                        case 1:
                            response = (_a.sent());
                            return [2 /*return*/];
                    }
                });
            }); });
            (0, mocha_1.it)("should respond an error", function () {
                (0, chai_1.expect)(response.error.code).to.equal(_1.JSONRPCErrorCode.InvalidRequest);
            });
        });
    });
    (0, mocha_1.describe)("having a custom mapErrorToJSONRPCErrorResponse method", function () {
        var errorMessagePrefix;
        var errorData;
        (0, mocha_1.beforeEach)(function () {
            errorMessagePrefix = "Error: ";
            errorData = {
                foo: "bar",
            };
            server.mapErrorToJSONRPCErrorResponse = function (id, error) {
                return (0, _1.createJSONRPCErrorResponse)(id, error.code, "".concat(errorMessagePrefix).concat(error.message), errorData);
            };
        });
        (0, mocha_1.describe)("rejecting", function () {
            var errorCode;
            var errorMessage;
            var response;
            (0, mocha_1.beforeEach)(function () { return __awaiter(void 0, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            errorCode = 123;
                            errorMessage = "test message";
                            server.addMethod("throw", function () {
                                var error = new Error(errorMessage);
                                error.code = errorCode;
                                throw error;
                            });
                            return [4 /*yield*/, server.receive({
                                    jsonrpc: _1.JSONRPC,
                                    id: 0,
                                    method: "throw",
                                })];
                        case 1:
                            response = (_a.sent());
                            return [2 /*return*/];
                    }
                });
            }); });
            (0, mocha_1.it)("should respond a custom error code", function () {
                (0, chai_1.expect)(response.error.code).to.equal(errorCode);
            });
            (0, mocha_1.it)("should respond a custom error message", function () {
                (0, chai_1.expect)(response.error.message).to.equal("".concat(errorMessagePrefix).concat(errorMessage));
            });
            (0, mocha_1.it)("should respond a custom error data", function () {
                (0, chai_1.expect)(response.error.data).to.deep.equal(errorData);
            });
        });
    });
    (0, mocha_1.describe)("having an async method", function () {
        var methodName;
        var receivedRequest;
        var receivedServerParams;
        var returnedResponse;
        var returnFromMethod;
        var throwFromMethod;
        (0, mocha_1.beforeEach)(function () {
            methodName = "foo";
            server.addMethodAdvanced(methodName, function (request, serverParams) {
                receivedRequest = request;
                receivedServerParams = serverParams;
                return new Promise(function (resolve, reject) {
                    returnedResponse = {
                        id: request.id,
                        jsonrpc: _1.JSONRPC,
                        result: {
                            foo: "bar",
                        },
                    };
                    returnFromMethod = function () {
                        resolve(returnedResponse);
                    };
                    throwFromMethod = function (error) {
                        reject(error);
                    };
                });
            });
        });
        (0, mocha_1.describe)("using middleware", function () {
            var middlewareCalled;
            var nextReturned;
            (0, mocha_1.beforeEach)(function () {
                middlewareCalled = false;
                nextReturned = false;
                server.applyMiddleware(function (next, request, serverParams) {
                    middlewareCalled = true;
                    return next(request, serverParams).then(function (result) {
                        nextReturned = true;
                        return result;
                    });
                });
            });
            (0, mocha_1.describe)("requesting", function () {
                var givenRequest;
                var givenServerParams;
                var actualResponse;
                (0, mocha_1.beforeEach)(function () {
                    givenRequest = {
                        jsonrpc: _1.JSONRPC,
                        id: 0,
                        method: methodName,
                        params: { foo: "bar" },
                    };
                    givenServerParams = { userID: "baz" };
                    server
                        .receive(givenRequest, givenServerParams)
                        .then(function (response) { return (actualResponse = response); });
                    return consumeAllEvents();
                });
                (0, mocha_1.it)("should call the middleware", function () {
                    (0, chai_1.expect)(middlewareCalled).to.be.true;
                });
                (0, mocha_1.it)("should receive a request", function () {
                    (0, chai_1.expect)(receivedRequest).to.deep.equal(givenRequest);
                });
                (0, mocha_1.it)("should received server params", function () {
                    (0, chai_1.expect)(receivedServerParams).to.deep.equal(givenServerParams);
                });
                (0, mocha_1.it)("should not return from the next middleware yet", function () {
                    (0, chai_1.expect)(nextReturned).to.be.false;
                });
                (0, mocha_1.describe)("finishing the request", function () {
                    (0, mocha_1.beforeEach)(function () {
                        returnFromMethod();
                        return consumeAllEvents();
                    });
                    (0, mocha_1.it)("should return from the next middleware", function () {
                        (0, chai_1.expect)(nextReturned).to.be.true;
                    });
                    (0, mocha_1.it)("should return a response", function () {
                        (0, chai_1.expect)(actualResponse).to.deep.equal(returnedResponse);
                    });
                });
            });
            (0, mocha_1.describe)("using another middleware", function () {
                var secondMiddlewareCalled;
                (0, mocha_1.beforeEach)(function () {
                    secondMiddlewareCalled = false;
                    server.applyMiddleware(function (next, request, serverParams) {
                        secondMiddlewareCalled = true;
                        return next(request, serverParams);
                    });
                });
                (0, mocha_1.describe)("requesting", function () {
                    (0, mocha_1.beforeEach)(function () {
                        server.receive({
                            jsonrpc: _1.JSONRPC,
                            id: 0,
                            method: methodName,
                            params: {},
                        });
                    });
                    (0, mocha_1.it)("should call the first middleware", function () {
                        (0, chai_1.expect)(middlewareCalled).to.be.true;
                    });
                    (0, mocha_1.it)("should call the second middleware", function () {
                        (0, chai_1.expect)(secondMiddlewareCalled).to.be.true;
                    });
                });
            });
        });
        (0, mocha_1.describe)("using a middleware that changes request and server params", function () {
            var changedParams;
            var changedServerParams;
            (0, mocha_1.beforeEach)(function () {
                changedParams = {
                    foo: "bar",
                };
                changedServerParams = {
                    userID: "changed user ID",
                };
                server.applyMiddleware(function (next, request) {
                    return next(__assign(__assign({}, request), { params: changedParams }), changedServerParams);
                });
            });
            (0, mocha_1.describe)("requesting", function () {
                var givenRequest;
                (0, mocha_1.beforeEach)(function () {
                    givenRequest = {
                        jsonrpc: _1.JSONRPC,
                        id: 0,
                        method: methodName,
                        params: {
                            foo: "foo",
                        },
                    };
                    server.receive(givenRequest);
                    returnFromMethod();
                    return consumeAllEvents();
                });
                (0, mocha_1.it)("should change the request", function () {
                    var expectedRequest = __assign(__assign({}, givenRequest), { params: changedParams });
                    (0, chai_1.expect)(receivedRequest).to.deep.equal(expectedRequest);
                });
                (0, mocha_1.it)("should change the server params", function () {
                    (0, chai_1.expect)(receivedServerParams).to.deep.equal(changedServerParams);
                });
            });
        });
        (0, mocha_1.describe)("using a middleware that changes response", function () {
            var changedResponse;
            (0, mocha_1.beforeEach)(function () {
                server.applyMiddleware(function (next, request, serverParams) {
                    return next(request, serverParams).then(function (response) {
                        changedResponse = {
                            jsonrpc: _1.JSONRPC,
                            id: response.id,
                            result: {
                                foo: new Date().toString(),
                            },
                        };
                        return changedResponse;
                    });
                });
            });
            (0, mocha_1.describe)("requesting", function () {
                var actualResponse;
                (0, mocha_1.beforeEach)(function () {
                    server
                        .receive({
                        jsonrpc: _1.JSONRPC,
                        id: 0,
                        method: methodName,
                        params: {},
                    })
                        .then(function (response) { return (actualResponse = response); });
                    returnFromMethod();
                    return consumeAllEvents();
                });
                (0, mocha_1.it)("should return the changed response", function () {
                    (0, chai_1.expect)(actualResponse).to.deep.equal(changedResponse);
                });
            });
        });
        (0, mocha_1.describe)("using middleware that catches exception", function () {
            (0, mocha_1.beforeEach)(function () {
                server.applyMiddleware(function (next, request, serverParams) { return __awaiter(void 0, void 0, void 0, function () {
                    var error_1;
                    return __generator(this, function (_a) {
                        switch (_a.label) {
                            case 0:
                                _a.trys.push([0, 2, , 3]);
                                return [4 /*yield*/, next(request, serverParams)];
                            case 1: return [2 /*return*/, _a.sent()];
                            case 2:
                                error_1 = _a.sent();
                                return [2 /*return*/, (0, _1.createJSONRPCErrorResponse)(request.id, error_1.code || _1.JSONRPCErrorCode.InternalError, error_1.message)];
                            case 3: return [2 /*return*/];
                        }
                    });
                }); });
            });
            (0, mocha_1.describe)("throwing", function () {
                var error;
                var actualResponse;
                (0, mocha_1.beforeEach)(function () {
                    server
                        .receive({
                        jsonrpc: _1.JSONRPC,
                        id: 0,
                        method: methodName,
                        params: {},
                    })
                        .then(function (response) { return (actualResponse = response); });
                    error = { code: 123, message: "test" };
                    throwFromMethod(error);
                    return consumeAllEvents();
                });
                (0, mocha_1.it)("should catch the exception on middleware", function () {
                    var expected = (0, _1.createJSONRPCErrorResponse)(0, error.code, error.message);
                    (0, chai_1.expect)(actualResponse).to.deep.equal(expected);
                });
            });
            (0, mocha_1.describe)("throwing from non-advanced method", function () {
                var message;
                var code;
                var actualResponse;
                (0, mocha_1.beforeEach)(function () { return __awaiter(void 0, void 0, void 0, function () {
                    return __generator(this, function (_a) {
                        switch (_a.label) {
                            case 0:
                                message = "thrown from non-advanced method";
                                code = 456;
                                server.addMethod("throw", function () { return __awaiter(void 0, void 0, void 0, function () {
                                    return __generator(this, function (_a) {
                                        throw { message: message, code: code };
                                    });
                                }); });
                                return [4 /*yield*/, server.receive({
                                        jsonrpc: _1.JSONRPC,
                                        id: 0,
                                        method: "throw",
                                    })];
                            case 1:
                                actualResponse = (_a.sent());
                                return [2 /*return*/];
                        }
                    });
                }); });
                (0, mocha_1.it)("should catch the exception on middleware", function () {
                    var expected = (0, _1.createJSONRPCErrorResponse)(0, code, message);
                    (0, chai_1.expect)(actualResponse).to.deep.equal(expected);
                });
            });
        });
        (0, mocha_1.describe)("using multiple middleware", function () {
            var count;
            var first;
            var second;
            var third;
            (0, mocha_1.beforeEach)(function () {
                count = 0;
                server.applyMiddleware(function (next, request, serverParams) {
                    first = ++count;
                    return next(request, serverParams);
                }, function (next, request, serverParams) {
                    second = ++count;
                    return next(request, serverParams);
                }, function (next, request, serverParams) {
                    third = ++count;
                    return next(request, serverParams);
                });
                server.receive({
                    jsonrpc: _1.JSONRPC,
                    id: 0,
                    method: methodName,
                });
                return consumeAllEvents();
            });
            (0, mocha_1.it)("should call middleware in the applied order", function () {
                (0, chai_1.expect)([first, second, third]).to.deep.equal([1, 2, 3]);
            });
        });
    });
    (0, mocha_1.describe)("receiving batch requests", function () {
        var responses;
        (0, mocha_1.beforeEach)(function () {
            server.addMethod("echo", function (_a) {
                var message = _a.message;
                return message;
            });
        });
        (0, mocha_1.describe)("of 3 requests", function () {
            (0, mocha_1.beforeEach)(function () { return __awaiter(void 0, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0: return [4 /*yield*/, server.receive([
                                { jsonrpc: _1.JSONRPC, id: 0, method: "echo", params: { message: "1" } },
                                { jsonrpc: _1.JSONRPC, id: 1, method: "echo", params: { message: "2" } },
                                { jsonrpc: _1.JSONRPC, id: 2, method: "echo", params: { message: "3" } },
                            ])];
                        case 1:
                            responses = _a.sent();
                            return [2 /*return*/];
                    }
                });
            }); });
            (0, mocha_1.it)("should return 3 responses", function () {
                (0, chai_1.expect)(responses.map(function (response) { return response.result; })).to.deep.equal(["1", "2", "3"]);
            });
        });
        (0, mocha_1.describe)("of 1 request", function () {
            (0, mocha_1.beforeEach)(function () { return __awaiter(void 0, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0: return [4 /*yield*/, server.receive([
                                { jsonrpc: _1.JSONRPC, id: 0, method: "echo", params: { message: "1" } },
                            ])];
                        case 1:
                            responses = _a.sent();
                            return [2 /*return*/];
                    }
                });
            }); });
            (0, mocha_1.it)("should return 1 response", function () {
                (0, chai_1.expect)(responses.result).to.equal("1");
            });
        });
        (0, mocha_1.describe)("of notifications", function () {
            (0, mocha_1.beforeEach)(function () { return __awaiter(void 0, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0: return [4 /*yield*/, server.receive([
                                { jsonrpc: _1.JSONRPC, method: "echo", params: { message: "1" } },
                            ])];
                        case 1:
                            responses = _a.sent();
                            return [2 /*return*/];
                    }
                });
            }); });
            (0, mocha_1.it)("should return null", function () {
                (0, chai_1.expect)(responses).to.be.null;
            });
        });
        (0, mocha_1.describe)("of a valid and an invalid request", function () {
            (0, mocha_1.beforeEach)(function () { return __awaiter(void 0, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0: return [4 /*yield*/, server.receive([
                                1,
                                { jsonrpc: _1.JSONRPC, id: 0, method: "echo", params: { message: "1" } },
                            ])];
                        case 1:
                            responses = _a.sent();
                            return [2 /*return*/];
                    }
                });
            }); });
            (0, mocha_1.it)("should return a failure and a success response", function () {
                (0, chai_1.expect)(responses).to.deep.equal([
                    {
                        jsonrpc: _1.JSONRPC,
                        id: null,
                        error: {
                            code: _1.JSONRPCErrorCode.InvalidRequest,
                            message: "Invalid Request",
                        },
                    },
                    { jsonrpc: _1.JSONRPC, id: 0, result: "1" },
                ]);
            });
        });
    });
});
var consumeAllEvents = function () { return new Promise(function (resolve) { return setTimeout(resolve, 0); }); };
