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
require("mocha");
var chai_1 = require("chai");
var client_1 = require("./client");
var server_1 = require("./server");
var server_and_client_1 = require("./server-and-client");
describe("interfaces", function () {
    describe("independent server and client", function () {
        var client;
        var server;
        beforeEach(function () {
            client = new client_1.JSONRPCClient(function (request) { return __awaiter(void 0, void 0, void 0, function () {
                var response;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0: return [4 /*yield*/, server.receive(request)];
                        case 1:
                            response = _a.sent();
                            if (response) {
                                client.receive(response);
                            }
                            return [2 /*return*/];
                    }
                });
            }); });
            server = new server_1.JSONRPCServer();
        });
        describe("calling method with no args no return", function () {
            var noArgsNoReturnCalled;
            beforeEach(function () {
                noArgsNoReturnCalled = false;
                server.addMethod("noArgsNoReturn", function () {
                    noArgsNoReturnCalled = true;
                });
                return client.request("noArgsNoReturn");
            });
            it("should call the method", function () {
                (0, chai_1.expect)(noArgsNoReturnCalled).to.be.true;
            });
        });
        describe("calling method with no args", function () {
            var expected;
            var actual;
            beforeEach(function () { return __awaiter(void 0, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            expected = "return value";
                            server.addMethod("noArgs", function () {
                                return expected;
                            });
                            return [4 /*yield*/, client.request("noArgs")];
                        case 1:
                            actual = _a.sent();
                            return [2 /*return*/];
                    }
                });
            }); });
            it("should call the method", function () {
                (0, chai_1.expect)(actual).to.equal(expected);
            });
        });
        describe("calling method with object args", function () {
            var actual;
            beforeEach(function () { return __awaiter(void 0, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            server.addMethod("objectArgs", function (_a) {
                                var foo = _a.foo, bar = _a.bar;
                                return "".concat(foo, ".").concat(bar);
                            });
                            return [4 /*yield*/, client.request("objectArgs", {
                                    foo: "string value",
                                    bar: 123,
                                })];
                        case 1:
                            actual = _a.sent();
                            return [2 /*return*/];
                    }
                });
            }); });
            it("should call the method", function () {
                (0, chai_1.expect)(actual).to.equal("string value.123");
            });
        });
        describe("calling method with array args", function () {
            var actual;
            beforeEach(function () { return __awaiter(void 0, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            server.addMethod("arrayArgs", function (_a) {
                                var foo = _a[0], bar = _a[1];
                                return "".concat(foo, ".").concat(bar);
                            });
                            return [4 /*yield*/, client.request("arrayArgs", ["string value", 123])];
                        case 1:
                            actual = _a.sent();
                            return [2 /*return*/];
                    }
                });
            }); });
            it("should call the method", function () {
                (0, chai_1.expect)(actual).to.equal("string value.123");
            });
        });
    });
    describe("server and client", function () {
        var serverAndClientA;
        var serverAndClientB;
        beforeEach(function () {
            serverAndClientA = new server_and_client_1.JSONRPCServerAndClient(new server_1.JSONRPCServer(), new client_1.JSONRPCClient(function (request) {
                return serverAndClientB.receiveAndSend(request);
            }));
            serverAndClientB = new server_and_client_1.JSONRPCServerAndClient(new server_1.JSONRPCServer(), new client_1.JSONRPCClient(function (request) {
                return serverAndClientA.receiveAndSend(request);
            }));
            serverAndClientA.addMethod("echo", function (_a) {
                var message = _a.message;
                return message;
            });
            serverAndClientB.addMethod("sum", function (_a) {
                var x = _a.x, y = _a.y;
                return x + y;
            });
        });
        describe("calling method from server A to B", function () {
            var actual;
            beforeEach(function () { return __awaiter(void 0, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0: return [4 /*yield*/, serverAndClientA.request("sum", { x: 1, y: 2 })];
                        case 1:
                            actual = _a.sent();
                            return [2 /*return*/];
                    }
                });
            }); });
            it("should call the method", function () {
                (0, chai_1.expect)(actual).to.equal(3);
            });
        });
        describe("calling method from server B to A", function () {
            var expected;
            var actual;
            beforeEach(function () { return __awaiter(void 0, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            expected = "hello";
                            return [4 /*yield*/, serverAndClientB.request("echo", { message: expected })];
                        case 1:
                            actual = _a.sent();
                            return [2 /*return*/];
                    }
                });
            }); });
            it("should call the method", function () {
                (0, chai_1.expect)(actual).to.equal(expected);
            });
        });
    });
});
