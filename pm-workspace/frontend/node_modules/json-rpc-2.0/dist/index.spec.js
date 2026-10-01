"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
require("mocha");
var chai_1 = require("chai");
var _1 = require(".");
describe("JSONRPCClient and JSONRPCServer", function () {
    var server;
    var client;
    var id;
    beforeEach(function () {
        id = 0;
        server = new _1.JSONRPCServer();
        client = new _1.JSONRPCClient(function (request) {
            return server.receive(request).then(function (response) {
                if (response) {
                    client.receive(response);
                }
            });
        }, function () { return ++id; });
    });
    it("sending a request should resolve the result", function () {
        beforeEach(function () {
            server.addMethod("foo", function () { return "bar"; });
            return client
                .request("foo", undefined)
                .then(function (result) { return (0, chai_1.expect)(result).to.equal("bar"); });
        });
    });
    it("sending a notification should send a notification", function () {
        var received;
        beforeEach(function () {
            server.addMethod("foo", function (_a) {
                var text = _a[0];
                return (received = text);
            });
            client.notify("foo", ["bar"]);
            (0, chai_1.expect)(received).to.equal("bar");
        });
    });
});
