"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var shortestpaths_1 = require("./shortestpaths");
var descent_1 = require("./descent");
var rectangle_1 = require("./rectangle");
var linklengths_1 = require("./linklengths");
var Link3D = (function () {
    function Link3D(source, target) {
        this.source = source;
        this.target = target;
    }
    Link3D.prototype.actualLength = function (x) {
        var _this = this;
        return Math.sqrt(x.reduce(function (c, v) {
            var dx = v[_this.target] - v[_this.source];
            return c + dx * dx;
        }, 0));
    };
    return Link3D;
}());
exports.Link3D = Link3D;
var Node3D = (function () {
    function Node3D(x, y, z) {
        if (x === void 0) { x = 0; }
        if (y === void 0) { y = 0; }
        if (z === void 0) { z = 0; }
        this.x = x;
        this.y = y;
        this.z = z;
    }
    return Node3D;
}());
exports.Node3D = Node3D;
var Layout3D = (function () {
    function Layout3D(nodes, links, idealLinkLength) {
        var _this = this;
        if (idealLinkLength === void 0) { idealLinkLength = 1; }
        this.nodes = nodes;
        this.links = links;
        this.idealLinkLength = idealLinkLength;
        this.constraints = null;
        this.useJaccardLinkLengths = true;
        this.result = new Array(Layout3D.k);
        for (var i = 0; i < Layout3D.k; ++i) {
            this.result[i] = new Array(nodes.length);
        }
        nodes.forEach(function (v, i) {
            for (var _i = 0, _a = Layout3D.dims; _i < _a.length; _i++) {
                var dim = _a[_i];
                if (typeof v[dim] == 'undefined')
                    v[dim] = Math.random();
            }
            _this.result[0][i] = v.x;
            _this.result[1][i] = v.y;
            _this.result[2][i] = v.z;
        });
    }
    ;
    Layout3D.prototype.linkLength = function (l) {
        return l.actualLength(this.result);
    };
    Layout3D.prototype.start = function (iterations) {
        var _this = this;
        if (iterations === void 0) { iterations = 100; }
        var n = this.nodes.length;
        var linkAccessor = new LinkAccessor();
        if (this.useJaccardLinkLengths)
            linklengths_1.jaccardLinkLengths(this.links, linkAccessor, 1.5);
        this.links.forEach(function (e) { return e.length *= _this.idealLinkLength; });
        var distanceMatrix = (new shortestpaths_1.Calculator(n, this.links, function (e) { return e.source; }, function (e) { return e.target; }, function (e) { return e.length; })).DistanceMatrix();
        var D = descent_1.Descent.createSquareMatrix(n, function (i, j) { return distanceMatrix[i][j]; });
        var G = descent_1.Descent.createSquareMatrix(n, function () { return 2; });
        this.links.forEach(function (_a) {
            var source = _a.source, target = _a.target;
            return G[source][target] = G[target][source] = 1;
        });
        this.descent = new descent_1.Descent(this.result, D);
        this.descent.threshold = 1e-3;
        this.descent.G = G;
        if (this.constraints)
            this.descent.project = new rectangle_1.Projection(this.nodes, null, null, this.constraints).projectFunctions();
        for (var i = 0; i < this.nodes.length; i++) {
            var v = this.nodes[i];
            if (v.fixed) {
                this.descent.locks.add(i, [v.x, v.y, v.z]);
            }
        }
        this.descent.run(iterations);
        return this;
    };
    Layout3D.prototype.tick = function () {
        this.descent.locks.clear();
        for (var i = 0; i < this.nodes.length; i++) {
            var v = this.nodes[i];
            if (v.fixed) {
                this.descent.locks.add(i, [v.x, v.y, v.z]);
            }
        }
        return this.descent.rungeKutta();
    };
    Layout3D.dims = ['x', 'y', 'z'];
    Layout3D.k = Layout3D.dims.length;
    return Layout3D;
}());
exports.Layout3D = Layout3D;
var LinkAccessor = (function () {
    function LinkAccessor() {
    }
    LinkAccessor.prototype.getSourceIndex = function (e) { return e.source; };
    LinkAccessor.prototype.getTargetIndex = function (e) { return e.target; };
    LinkAccessor.prototype.getLength = function (e) { return e.length; };
    LinkAccessor.prototype.setLength = function (e, l) { e.length = l; };
    return LinkAccessor;
}());
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibGF5b3V0M2QuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi9XZWJDb2xhL3NyYy9sYXlvdXQzZC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiOztBQUFBLGlEQUEwQztBQUMxQyxxQ0FBaUM7QUFDakMseUNBQTREO0FBRTVELDZDQUFvRTtBQUVwRTtJQUVRLGdCQUFtQixNQUFjLEVBQVMsTUFBYztRQUFyQyxXQUFNLEdBQU4sTUFBTSxDQUFRO1FBQVMsV0FBTSxHQUFOLE1BQU0sQ0FBUTtJQUFJLENBQUM7SUFDN0QsNkJBQVksR0FBWixVQUFhLENBQWE7UUFBMUIsaUJBTUM7UUFMRyxPQUFPLElBQUksQ0FBQyxJQUFJLENBQ1osQ0FBQyxDQUFDLE1BQU0sQ0FBQyxVQUFDLENBQVMsRUFBRSxDQUFXO1lBQzVCLElBQU0sRUFBRSxHQUFHLENBQUMsQ0FBQyxLQUFJLENBQUMsTUFBTSxDQUFDLEdBQUcsQ0FBQyxDQUFDLEtBQUksQ0FBQyxNQUFNLENBQUMsQ0FBQztZQUMzQyxPQUFPLENBQUMsR0FBRyxFQUFFLEdBQUcsRUFBRSxDQUFDO1FBQ3ZCLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ2YsQ0FBQztJQUNMLGFBQUM7QUFBRCxDQUFDLEFBVkwsSUFVSztBQVZRLHdCQUFNO0FBV2Y7SUFTSSxnQkFDVyxDQUFhLEVBQ2IsQ0FBYSxFQUNiLENBQWE7UUFGYixrQkFBQSxFQUFBLEtBQWE7UUFDYixrQkFBQSxFQUFBLEtBQWE7UUFDYixrQkFBQSxFQUFBLEtBQWE7UUFGYixNQUFDLEdBQUQsQ0FBQyxDQUFZO1FBQ2IsTUFBQyxHQUFELENBQUMsQ0FBWTtRQUNiLE1BQUMsR0FBRCxDQUFDLENBQVk7SUFBSSxDQUFDO0lBQ2pDLGFBQUM7QUFBRCxDQUFDLEFBYkQsSUFhQztBQWJZLHdCQUFNO0FBY25CO0lBTUksa0JBQW1CLEtBQWUsRUFBUyxLQUFlLEVBQVMsZUFBMkI7UUFBOUYsaUJBYUM7UUFia0UsZ0NBQUEsRUFBQSxtQkFB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