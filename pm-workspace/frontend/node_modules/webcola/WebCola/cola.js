(function(f){if(typeof exports==="object"&&typeof module!=="undefined"){module.exports=f()}else if(typeof define==="function"&&define.amd){define([],f)}else{var g;if(typeof window!=="undefined"){g=window}else if(typeof global!=="undefined"){g=global}else if(typeof self!=="undefined"){g=self}else{g=this}g.cola = f()}})(function(){var define,module,exports;return (function(){function r(e,n,t){function o(i,f){if(!n[i]){if(!e[i]){var c="function"==typeof require&&require;if(!f&&c)return c(i,!0);if(u)return u(i,!0);var a=new Error("Cannot find module '"+i+"'");throw a.code="MODULE_NOT_FOUND",a}var p=n[i]={exports:{}};e[i][0].call(p.exports,function(r){var n=e[i][1][r];return o(n||r)},p,p.exports,r,e,n,t)}return n[i].exports}for(var u="function"==typeof require&&require,i=0;i<t.length;i++)o(t[i]);return o}return r})()({1:[function(require,module,exports){
"use strict";
function __export(m) {
    for (var p in m) if (!exports.hasOwnProperty(p)) exports[p] = m[p];
}
Object.defineProperty(exports, "__esModule", { value: true });
__export(require("./src/adaptor"));
__export(require("./src/d3adaptor"));
__export(require("./src/descent"));
__export(require("./src/geom"));
__export(require("./src/gridrouter"));
__export(require("./src/handledisconnected"));
__export(require("./src/layout"));
__export(require("./src/layout3d"));
__export(require("./src/linklengths"));
__export(require("./src/powergraph"));
__export(require("./src/pqueue"));
__export(require("./src/rbtree"));
__export(require("./src/rectangle"));
__export(require("./src/shortestpaths"));
__export(require("./src/vpsc"));
__export(require("./src/batch"));
},{"./src/adaptor":2,"./src/batch":3,"./src/d3adaptor":4,"./src/descent":7,"./src/geom":8,"./src/gridrouter":9,"./src/handledisconnected":10,"./src/layout":11,"./src/layout3d":12,"./src/linklengths":13,"./src/powergraph":14,"./src/pqueue":15,"./src/rbtree":16,"./src/rectangle":17,"./src/shortestpaths":18,"./src/vpsc":19}],2:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var layout_1 = require("./layout");
var LayoutAdaptor = (function (_super) {
    __extends(LayoutAdaptor, _super);
    function LayoutAdaptor(options) {
        var _this = _super.call(this) || this;
        var self = _this;
        var o = options;
        if (o.trigger) {
            _this.trigger = o.trigger;
        }
        if (o.kick) {
            _this.kick = o.kick;
        }
        if (o.drag) {
            _this.drag = o.drag;
        }
        if (o.on) {
            _this.on = o.on;
        }
        _this.dragstart = _this.dragStart = layout_1.Layout.dragStart;
        _this.dragend = _this.dragEnd = layout_1.Layout.dragEnd;
        return _this;
    }
    LayoutAdaptor.prototype.trigger = function (e) { };
    ;
    LayoutAdaptor.prototype.kick = function () { };
    ;
    LayoutAdaptor.prototype.drag = function () { };
    ;
    LayoutAdaptor.prototype.on = function (eventType, listener) { return this; };
    ;
    return LayoutAdaptor;
}(layout_1.Layout));
exports.LayoutAdaptor = LayoutAdaptor;
function adaptor(options) {
    return new LayoutAdaptor(options);
}
exports.adaptor = adaptor;
},{"./layout":11}],3:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var layout_1 = require("./layout");
var gridrouter_1 = require("./gridrouter");
function gridify(pgLayout, nudgeGap, margin, groupMargin) {
    pgLayout.cola.start(0, 0, 0, 10, false);
    var gridrouter = route(pgLayout.cola.nodes(), pgLayout.cola.groups(), margin, groupMargin);
    return gridrouter.routeEdges(pgLayout.powerGraph.powerEdges, nudgeGap, function (e) { return e.source.routerNode.id; }, function (e) { return e.target.routerNode.id; });
}
exports.gridify = gridify;
function route(nodes, groups, margin, groupMargin) {
    nodes.forEach(function (d) {
        d.routerNode = {
            name: d.name,
            bounds: d.bounds.inflate(-margin)
        };
    });
    groups.forEach(function (d) {
        d.routerNode = {
            bounds: d.bounds.inflate(-groupMargin),
            children: (typeof d.groups !== 'undefined' ? d.groups.map(function (c) { return nodes.length + c.id; }) : [])
                .concat(typeof d.leaves !== 'undefined' ? d.leaves.map(function (c) { return c.index; }) : [])
        };
    });
    var gridRouterNodes = nodes.concat(groups).map(function (d, i) {
        d.routerNode.id = i;
        return d.routerNode;
    });
    return new gridrouter_1.GridRouter(gridRouterNodes, {
        getChildren: function (v) { return v.children; },
        getBounds: function (v) { return v.bounds; }
    }, margin - groupMargin);
}
function powerGraphGridLayout(graph, size, grouppadding) {
    var powerGraph;
    graph.nodes.forEach(function (v, i) { return v.index = i; });
    new layout_1.Layout()
        .avoidOverlaps(false)
        .nodes(graph.nodes)
        .links(graph.links)
        .powerGraphGroups(function (d) {
        powerGraph = d;
        powerGraph.groups.forEach(function (v) { return v.padding = grouppadding; });
    });
    var n = graph.nodes.length;
    var edges = [];
    var vs = graph.nodes.slice(0);
    vs.forEach(function (v, i) { return v.index = i; });
    powerGraph.groups.forEach(function (g) {
        var sourceInd = g.index = g.id + n;
        vs.push(g);
        if (typeof g.leaves !== 'undefined')
            g.leaves.forEach(function (v) { return edges.push({ source: sourceInd, target: v.index }); });
        if (typeof g.groups !== 'undefined')
            g.groups.forEach(function (gg) { return edges.push({ source: sourceInd, target: gg.id + n }); });
    });
    powerGraph.powerEdges.forEach(function (e) {
        edges.push({ source: e.source.index, target: e.target.index });
    });
    new layout_1.Layout()
        .size(size)
        .nodes(vs)
        .links(edges)
        .avoidOverlaps(false)
        .linkDistance(30)
        .symmetricDiffLinkLengths(5)
        .convergenceThreshold(1e-4)
        .start(100, 0, 0, 0, false);
    return {
        cola: new layout_1.Layout()
            .convergenceThreshold(1e-3)
            .size(size)
            .avoidOverlaps(true)
            .nodes(graph.nodes)
            .links(graph.links)
            .groupCompactness(1e-4)
            .linkDistance(30)
            .symmetricDiffLinkLengths(5)
            .powerGraphGroups(function (d) {
            powerGraph = d;
            powerGraph.groups.forEach(function (v) {
                v.padding = grouppadding;
            });
        }).start(50, 0, 100, 0, false),
        powerGraph: powerGraph
    };
}
exports.powerGraphGridLayout = powerGraphGridLayout;
},{"./gridrouter":9,"./layout":11}],4:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var d3v3 = require("./d3v3adaptor");
var d3v4 = require("./d3v4adaptor");
;
function d3adaptor(d3Context) {
    if (!d3Context || isD3V3(d3Context)) {
        return new d3v3.D3StyleLayoutAdaptor();
    }
    return new d3v4.D3StyleLayoutAdaptor(d3Context);
}
exports.d3adaptor = d3adaptor;
function isD3V3(d3Context) {
    var v3exp = /^3\./;
    return d3Context.version && d3Context.version.match(v3exp) !== null;
}
},{"./d3v3adaptor":5,"./d3v4adaptor":6}],5:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var layout_1 = require("./layout");
var D3StyleLayoutAdaptor = (function (_super) {
    __extends(D3StyleLayoutAdaptor, _super);
    function D3StyleLayoutAdaptor() {
        var _this = _super.call(this) || this;
        _this.event = d3.dispatch(layout_1.EventType[layout_1.EventType.start], layout_1.EventType[layout_1.EventType.tick], layout_1.EventType[layout_1.EventType.end]);
        var d3layout = _this;
        var drag;
        _this.drag = function () {
            if (!drag) {
                var drag = d3.behavior.drag()
                    .origin(layout_1.Layout.dragOrigin)
                    .on("dragstart.d3adaptor", layout_1.Layout.dragStart)
                    .on("drag.d3adaptor", function (d) {
                    layout_1.Layout.drag(d, d3.event);
                    d3layout.resume();
                })
                    .on("dragend.d3adaptor", layout_1.Layout.dragEnd);
            }
            if (!arguments.length)
                return drag;
            this
                .call(drag);
        };
        return _this;
    }
    D3StyleLayoutAdaptor.prototype.trigger = function (e) {
        var d3event = { type: layout_1.EventType[e.type], alpha: e.alpha, stress: e.stress };
        this.event[d3event.type](d3event);
    };
    D3StyleLayoutAdaptor.prototype.kick = function () {
        var _this = this;
        d3.timer(function () { return _super.prototype.tick.call(_this); });
    };
    D3StyleLayoutAdaptor.prototype.on = function (eventType, listener) {
        if (typeof eventType === 'string') {
            this.event.on(eventType, listener);
        }
        else {
            this.event.on(layout_1.EventType[eventType], listener);
        }
        return this;
    };
    return D3StyleLayoutAdaptor;
}(layout_1.Layout));
exports.D3StyleLayoutAdaptor = D3StyleLayoutAdaptor;
function d3adaptor() {
    return new D3StyleLayoutAdaptor();
}
exports.d3adaptor = d3adaptor;
},{"./layout":11}],6:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var layout_1 = require("./layout");
var D3StyleLayoutAdaptor = (function (_super) {
    __extends(D3StyleLayoutAdaptor, _super);
    function D3StyleLayoutAdaptor(d3Context) {
        var _this = _super.call(this) || this;
        _this.d3Context = d3Context;
        _this.event = d3Context.dispatch(layout_1.EventType[layout_1.EventType.start], layout_1.EventType[layout_1.EventType.tick], layout_1.EventType[layout_1.EventType.end]);
        var d3layout = _this;
        var drag;
        _this.drag = function () {
            if (!drag) {
                var drag = d3Context.drag()
                    .subject(layout_1.Layout.dragOrigin)
                    .on("start.d3adaptor", layout_1.Layout.dragStart)
                    .on("drag.d3adaptor", function (d) {
                    layout_1.Layout.drag(d, d3Context.event);
                    d3layout.resume();
                })
                    .on("end.d3adaptor", layout_1.Layout.dragEnd);
            }
            if (!arguments.length)
                return drag;
            arguments[0].call(drag);
        };
        return _this;
    }
    D3StyleLayoutAdaptor.prototype.trigger = function (e) {
        var d3event = { type: layout_1.EventType[e.type], alpha: e.alpha, stress: e.stress };
        this.event.call(d3event.type, d3event);
    };
    D3StyleLayoutAdaptor.prototype.kick = function () {
        var _this = this;
        var t = this.d3Context.timer(function () { return _super.prototype.tick.call(_this) && t.stop(); });
    };
    D3StyleLayoutAdaptor.prototype.on = function (eventType, listener) {
        if (typeof eventType === 'string') {
            this.event.on(eventType, listener);
        }
        else {
            this.event.on(layout_1.EventType[eventType], listener);
        }
        return this;
    };
    return D3StyleLayoutAdaptor;
}(layout_1.Layout));
exports.D3StyleLayoutAdaptor = D3StyleLayoutAdaptor;
},{"./layout":11}],7:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var Locks = (function () {
    function Locks() {
        this.locks = {};
    }
    Locks.prototype.add = function (id, x) {
        this.locks[id] = x;
    };
    Locks.prototype.clear = function () {
        this.locks = {};
    };
    Locks.prototype.isEmpty = function () {
        for (var l in this.locks)
            return false;
        return true;
    };
    Locks.prototype.apply = function (f) {
        for (var l in this.locks) {
            f(Number(l), this.locks[l]);
        }
    };
    return Locks;
}());
exports.Locks = Locks;
var Descent = (function () {
    function Descent(x, D, G) {
        if (G === void 0) { G = null; }
        this.D = D;
        this.G = G;
        this.threshold = 0.0001;
        this.numGridSnapNodes = 0;
        this.snapGridSize = 100;
        this.snapStrength = 1000;
        this.scaleSnapByMaxH = false;
        this.random = new PseudoRandom();
        this.project = null;
        this.x = x;
        this.k = x.length;
        var n = this.n = x[0].length;
        this.H = new Array(this.k);
        this.g = new Array(this.k);
        this.Hd = new Array(this.k);
        this.a = new Array(this.k);
        this.b = new Array(this.k);
        this.c = new Array(this.k);
        this.d = new Array(this.k);
        this.e = new Array(this.k);
        this.ia = new Array(this.k);
        this.ib = new Array(this.k);
        this.xtmp = new Array(this.k);
        this.locks = new Locks();
        this.minD = Number.MAX_VALUE;
        var i = n, j;
        while (i--) {
            j = n;
            while (--j > i) {
                var d = D[i][j];
                if (d > 0 && d < this.minD) {
                    this.minD = d;
                }
            }
        }
        if (this.minD === Number.MAX_VALUE)
            this.minD = 1;
        i = this.k;
        while (i--) {
            this.g[i] = new Array(n);
            this.H[i] = new Array(n);
            j = n;
            while (j--) {
                this.H[i][j] = new Array(n);
            }
            this.Hd[i] = new Array(n);
            this.a[i] = new Array(n);
            this.b[i] = new Array(n);
            this.c[i] = new Array(n);
            this.d[i] = new Array(n);
            this.e[i] = new Array(n);
            this.ia[i] = new Array(n);
            this.ib[i] = new Array(n);
            this.xtmp[i] = new Array(n);
        }
    }
    Descent.createSquareMatrix = function (n, f) {
        var M = new Array(n);
        for (var i = 0; i < n; ++i) {
            M[i] = new Array(n);
            for (var j = 0; j < n; ++j) {
                M[i][j] = f(i, j);
            }
        }
        return M;
    };
    Descent.prototype.offsetDir = function () {
        var _this = this;
        var u = new Array(this.k);
        var l = 0;
        for (var i = 0; i < this.k; ++i) {
            var x = u[i] = this.random.getNextBetween(0.01, 1) - 0.5;
            l += x * x;
        }
        l = Math.sqrt(l);
        return u.map(function (x) { return x *= _this.minD / l; });
    };
    Descent.prototype.computeDerivatives = function (x) {
        var _this = this;
        var n = this.n;
        if (n < 1)
            return;
        var i;
        var d = new Array(this.k);
        var d2 = new Array(this.k);
        var Huu = new Array(this.k);
        var maxH = 0;
        for (var u = 0; u < n; ++u) {
            for (i = 0; i < this.k; ++i)
                Huu[i] = this.g[i][u] = 0;
            for (var v = 0; v < n; ++v) {
                if (u === v)
                    continue;
                var maxDisplaces = n;
                while (maxDisplaces--) {
                    var sd2 = 0;
                    for (i = 0; i < this.k; ++i) {
                        var dx = d[i] = x[i][u] - x[i][v];
                        sd2 += d2[i] = dx * dx;
                    }
                    if (sd2 > 1e-9)
                        break;
                    var rd = this.offsetDir();
                    for (i = 0; i < this.k; ++i)
                        x[i][v] += rd[i];
                }
                var l = Math.sqrt(sd2);
                var D = this.D[u][v];
                var weight = this.G != null ? this.G[u][v] : 1;
                if (weight > 1 && l > D || !isFinite(D)) {
                    for (i = 0; i < this.k; ++i)
                        this.H[i][u][v] = 0;
                    continue;
                }
                if (weight > 1) {
                    weight = 1;
                }
                var D2 = D * D;
                var gs = 2 * weight * (l - D) / (D2 * l);
                var l3 = l * l * l;
                var hs = 2 * -weight / (D2 * l3);
                if (!isFinite(gs))
                    console.log(gs);
                for (i = 0; i < this.k; ++i) {
                    this.g[i][u] += d[i] * gs;
                    Huu[i] -= this.H[i][u][v] = hs * (l3 + D * (d2[i] - sd2) + l * sd2);
                }
            }
            for (i = 0; i < this.k; ++i)
                maxH = Math.max(maxH, this.H[i][u][u] = Huu[i]);
        }
        var r = this.snapGridSize / 2;
        var g = this.snapGridSize;
        var w = this.snapStrength;
        var k = w / (r * r);
        var numNodes = this.numGridSnapNodes;
        for (var u = 0; u < numNodes; ++u) {
            for (i = 0; i < this.k; ++i) {
                var xiu = this.x[i][u];
                var m = xiu / g;
                var f = m % 1;
                var q = m - f;
                var a = Math.abs(f);
                var dx = (a <= 0.5) ? xiu - q * g :
                    (xiu > 0) ? xiu - (q + 1) * g : xiu - (q - 1) * g;
                if (-r < dx && dx <= r) {
                    if (this.scaleSnapByMaxH) {
                        this.g[i][u] += maxH * k * dx;
                        this.H[i][u][u] += maxH * k;
                    }
                    else {
                        this.g[i][u] += k * dx;
                        this.H[i][u][u] += k;
                    }
                }
            }
        }
        if (!this.locks.isEmpty()) {
            this.locks.apply(function (u, p) {
                for (i = 0; i < _this.k; ++i) {
                    _this.H[i][u][u] += maxH;
                    _this.g[i][u] -= maxH * (p[i] - x[i][u]);
                }
            });
        }
    };
    Descent.dotProd = function (a, b) {
        var x = 0, i = a.length;
        while (i--)
            x += a[i] * b[i];
        return x;
    };
    Descent.rightMultiply = function (m, v, r) {
        var i = m.length;
        while (i--)
            r[i] = Descent.dotProd(m[i], v);
    };
    Descent.prototype.computeStepSize = function (d) {
        var numerator = 0, denominator = 0;
        for (var i = 0; i < this.k; ++i) {
            numerator += Descent.dotProd(this.g[i], d[i]);
            Descent.rightMultiply(this.H[i], d[i], this.Hd[i]);
            denominator += Descent.dotProd(d[i], this.Hd[i]);
        }
        if (denominator === 0 || !isFinite(denominator))
            return 0;
        return 1 * numerator / denominator;
    };
    Descent.prototype.reduceStress = function () {
        this.computeDerivatives(this.x);
        var alpha = this.computeStepSize(this.g);
        for (var i = 0; i < this.k; ++i) {
            this.takeDescentStep(this.x[i], this.g[i], alpha);
        }
        return this.computeStress();
    };
    Descent.copy = function (a, b) {
        var m = a.length, n = b[0].length;
        for (var i = 0; i < m; ++i) {
            for (var j = 0; j < n; ++j) {
                b[i][j] = a[i][j];
            }
        }
    };
    Descent.prototype.stepAndProject = function (x0, r, d, stepSize) {
        Descent.copy(x0, r);
        this.takeDescentStep(r[0], d[0], stepSize);
        if (this.project)
            this.project[0](x0[0], x0[1], r[0]);
        this.takeDescentStep(r[1], d[1], stepSize);
        if (this.project)
            this.project[1](r[0], x0[1], r[1]);
        for (var i = 2; i < this.k; i++)
            this.takeDescentStep(r[i], d[i], stepSize);
    };
    Descent.mApply = function (m, n, f) {
        var i = m;
        while (i-- > 0) {
            var j = n;
            while (j-- > 0)
                f(i, j);
        }
    };
    Descent.prototype.matrixApply = function (f) {
        Descent.mApply(this.k, this.n, f);
    };
    Descent.prototype.computeNextPosition = function (x0, r) {
        var _this = this;
        this.computeDerivatives(x0);
        var alpha = this.computeStepSize(this.g);
        this.stepAndProject(x0, r, this.g, alpha);
        if (this.project) {
            this.matrixApply(function (i, j) { return _this.e[i][j] = x0[i][j] - r[i][j]; });
            var beta = this.computeStepSize(this.e);
            beta = Math.max(0.2, Math.min(beta, 1));
            this.stepAndProject(x0, r, this.e, beta);
        }
    };
    Descent.prototype.run = function (iterations) {
        var stress = Number.MAX_VALUE, converged = false;
        while (!converged && iterations-- > 0) {
            var s = this.rungeKutta();
            converged = Math.abs(stress / s - 1) < this.threshold;
            stress = s;
        }
        return stress;
    };
    Descent.prototype.rungeKutta = function () {
        var _this = this;
        this.computeNextPosition(this.x, this.a);
        Descent.mid(this.x, this.a, this.ia);
        this.computeNextPosition(this.ia, this.b);
        Descent.mid(this.x, this.b, this.ib);
        this.computeNextPosition(this.ib, this.c);
        this.computeNextPosition(this.c, this.d);
        var disp = 0;
        this.matrixApply(function (i, j) {
            var x = (_this.a[i][j] + 2.0 * _this.b[i][j] + 2.0 * _this.c[i][j] + _this.d[i][j]) / 6.0, d = _this.x[i][j] - x;
            disp += d * d;
            _this.x[i][j] = x;
        });
        return disp;
    };
    Descent.mid = function (a, b, m) {
        Descent.mApply(a.length, a[0].length, function (i, j) {
            return m[i][j] = a[i][j] + (b[i][j] - a[i][j]) / 2.0;
        });
    };
    Descent.prototype.takeDescentStep = function (x, d, stepSize) {
        for (var i = 0; i < this.n; ++i) {
            x[i] = x[i] - stepSize * d[i];
        }
    };
    Descent.prototype.computeStress = function () {
        var stress = 0;
        for (var u = 0, nMinus1 = this.n - 1; u < nMinus1; ++u) {
            for (var v = u + 1, n = this.n; v < n; ++v) {
                var l = 0;
                for (var i = 0; i < this.k; ++i) {
                    var dx = this.x[i][u] - this.x[i][v];
                    l += dx * dx;
                }
                l = Math.sqrt(l);
                var d = this.D[u][v];
                if (!isFinite(d))
                    continue;
                var rl = d - l;
                var d2 = d * d;
                stress += rl * rl / d2;
            }
        }
        return stress;
    };
    Descent.zeroDistance = 1e-10;
    return Descent;
}());
exports.Descent = Descent;
var PseudoRandom = (function () {
    function PseudoRandom(seed) {
        if (seed === void 0) { seed = 1; }
        this.seed = seed;
        this.a = 214013;
        this.c = 2531011;
        this.m = 2147483648;
        this.range = 32767;
    }
    PseudoRandom.prototype.getNext = function () {
        this.seed = (this.seed * this.a + this.c) % this.m;
        return (this.seed >> 16) / this.range;
    };
    PseudoRandom.prototype.getNextBetween = function (min, max) {
        return min + this.getNext() * (max - min);
    };
    return PseudoRandom;
}());
exports.PseudoRandom = PseudoRandom;
},{}],8:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var rectangle_1 = require("./rectangle");
var Point = (function () {
    function Point() {
    }
    return Point;
}());
exports.Point = Point;
var LineSegment = (function () {
    function LineSegment(x1, y1, x2, y2) {
        this.x1 = x1;
        this.y1 = y1;
        this.x2 = x2;
        this.y2 = y2;
    }
    return LineSegment;
}());
exports.LineSegment = LineSegment;
var PolyPoint = (function (_super) {
    __extends(PolyPoint, _super);
    function PolyPoint() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    return PolyPoint;
}(Point));
exports.PolyPoint = PolyPoint;
function isLeft(P0, P1, P2) {
    return (P1.x - P0.x) * (P2.y - P0.y) - (P2.x - P0.x) * (P1.y - P0.y);
}
exports.isLeft = isLeft;
function above(p, vi, vj) {
    return isLeft(p, vi, vj) > 0;
}
function below(p, vi, vj) {
    return isLeft(p, vi, vj) < 0;
}
function ConvexHull(S) {
    var P = S.slice(0).sort(function (a, b) { return a.x !== b.x ? b.x - a.x : b.y - a.y; });
    var n = S.length, i;
    var minmin = 0;
    var xmin = P[0].x;
    for (i = 1; i < n; ++i) {
        if (P[i].x !== xmin)
            break;
    }
    var minmax = i - 1;
    var H = [];
    H.push(P[minmin]);
    if (minmax === n - 1) {
        if (P[minmax].y !== P[minmin].y)
            H.push(P[minmax]);
    }
    else {
        var maxmin, maxmax = n - 1;
        var xmax = P[n - 1].x;
        for (i = n - 2; i >= 0; i--)
            if (P[i].x !== xmax)
                break;
        maxmin = i + 1;
        i = minmax;
        while (++i <= maxmin) {
            if (isLeft(P[minmin], P[maxmin], P[i]) >= 0 && i < maxmin)
                continue;
            while (H.length > 1) {
                if (isLeft(H[H.length - 2], H[H.length - 1], P[i]) > 0)
                    break;
                else
                    H.length -= 1;
            }
            if (i != minmin)
                H.push(P[i]);
        }
        if (maxmax != maxmin)
            H.push(P[maxmax]);
        var bot = H.length;
        i = maxmin;
        while (--i >= minmax) {
            if (isLeft(P[maxmax], P[minmax], P[i]) >= 0 && i > minmax)
                continue;
            while (H.length > bot) {
                if (isLeft(H[H.length - 2], H[H.length - 1], P[i]) > 0)
                    break;
                else
                    H.length -= 1;
            }
            if (i != minmin)
                H.push(P[i]);
        }
    }
    return H;
}
exports.ConvexHull = ConvexHull;
function clockwiseRadialSweep(p, P, f) {
    P.slice(0).sort(function (a, b) { return Math.atan2(a.y - p.y, a.x - p.x) - Math.atan2(b.y - p.y, b.x - p.x); }).forEach(f);
}
exports.clockwiseRadialSweep = clockwiseRadialSweep;
function nextPolyPoint(p, ps) {
    if (p.polyIndex === ps.length - 1)
        return ps[0];
    return ps[p.polyIndex + 1];
}
function prevPolyPoint(p, ps) {
    if (p.polyIndex === 0)
        return ps[ps.length - 1];
    return ps[p.polyIndex - 1];
}
function tangent_PointPolyC(P, V) {
    var Vclosed = V.slice(0);
    Vclosed.push(V[0]);
    return { rtan: Rtangent_PointPolyC(P, Vclosed), ltan: Ltangent_PointPolyC(P, Vclosed) };
}
function Rtangent_PointPolyC(P, V) {
    var n = V.length - 1;
    var a, b, c;
    var upA, dnC;
    if (below(P, V[1], V[0]) && !above(P, V[n - 1], V[0]))
        return 0;
    for (a = 0, b = n;;) {
        if (b - a === 1)
            if (above(P, V[a], V[b]))
                return a;
            else
                return b;
        c = Math.floor((a + b) / 2);
        dnC = below(P, V[c + 1], V[c]);
        if (dnC && !above(P, V[c - 1], V[c]))
            return c;
        upA = above(P, V[a + 1], V[a]);
        if (upA) {
            if (dnC)
                b = c;
            else {
                if (above(P, V[a], V[c]))
                    b = c;
                else
                    a = c;
            }
        }
        else {
            if (!dnC)
                a = c;
            else {
                if (below(P, V[a], V[c]))
                    b = c;
                else
                    a = c;
            }
        }
    }
}
function Ltangent_PointPolyC(P, V) {
    var n = V.length - 1;
    var a, b, c;
    var dnA, dnC;
    if (above(P, V[n - 1], V[0]) && !below(P, V[1], V[0]))
        return 0;
    for (a = 0, b = n;;) {
        if (b - a === 1)
            if (below(P, V[a], V[b]))
                return a;
            else
                return b;
        c = Math.floor((a + b) / 2);
        dnC = below(P, V[c + 1], V[c]);
        if (above(P, V[c - 1], V[c]) && !dnC)
            return c;
        dnA = below(P, V[a + 1], V[a]);
        if (dnA) {
            if (!dnC)
                b = c;
            else {
                if (below(P, V[a], V[c]))
                    b = c;
                else
                    a = c;
            }
        }
        else {
            if (dnC)
                a = c;
            else {
                if (above(P, V[a], V[c]))
                    b = c;
                else
                    a = c;
            }
        }
    }
}
function tangent_PolyPolyC(V, W, t1, t2, cmp1, cmp2) {
    var ix1, ix2;
    ix1 = t1(W[0], V);
    ix2 = t2(V[ix1], W);
    var done = false;
    while (!done) {
        done = true;
        while (true) {
            if (ix1 === V.length - 1)
                ix1 = 0;
            if (cmp1(W[ix2], V[ix1], V[ix1 + 1]))
                break;
            ++ix1;
        }
        while (true) {
            if (ix2 === 0)
                ix2 = W.length - 1;
            if (cmp2(V[ix1], W[ix2], W[ix2 - 1]))
                break;
            --ix2;
            done = false;
        }
    }
    return { t1: ix1, t2: ix2 };
}
exports.tangent_PolyPolyC = tangent_PolyPolyC;
function LRtangent_PolyPolyC(V, W) {
    var rl = RLtangent_PolyPolyC(W, V);
    return { t1: rl.t2, t2: rl.t1 };
}
exports.LRtangent_PolyPolyC = LRtangent_PolyPolyC;
function RLtangent_PolyPolyC(V, W) {
    return tangent_PolyPolyC(V, W, Rtangent_PointPolyC, Ltangent_PointPolyC, above, below);
}
exports.RLtangent_PolyPolyC = RLtangent_PolyPolyC;
function LLtangent_PolyPolyC(V, W) {
    return tangent_PolyPolyC(V, W, Ltangent_PointPolyC, Ltangent_PointPolyC, below, below);
}
exports.LLtangent_PolyPolyC = LLtangent_PolyPolyC;
function RRtangent_PolyPolyC(V, W) {
    return tangent_PolyPolyC(V, W, Rtangent_PointPolyC, Rtangent_PointPolyC, above, above);
}
exports.RRtangent_PolyPolyC = RRtangent_PolyPolyC;
var BiTangent = (function () {
    function BiTangent(t1, t2) {
        this.t1 = t1;
        this.t2 = t2;
    }
    return BiTangent;
}());
exports.BiTangent = BiTangent;
var BiTangents = (function () {
    function BiTangents() {
    }
    return BiTangents;
}());
exports.BiTangents = BiTangents;
var TVGPoint = (function (_super) {
    __extends(TVGPoint, _super);
    function TVGPoint() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    return TVGPoint;
}(Point));
exports.TVGPoint = TVGPoint;
var VisibilityVertex = (function () {
    function VisibilityVertex(id, polyid, polyvertid, p) {
        this.id = id;
        this.polyid = polyid;
        this.polyvertid = polyvertid;
        this.p = p;
        p.vv = this;
    }
    return VisibilityVertex;
}());
exports.VisibilityVertex = VisibilityVertex;
var VisibilityEdge = (function () {
    function VisibilityEdge(source, target) {
        this.source = source;
        this.target = target;
    }
    VisibilityEdge.prototype.length = function () {
        var dx = this.source.p.x - this.target.p.x;
        var dy = this.source.p.y - this.target.p.y;
        return Math.sqrt(dx * dx + dy * dy);
    };
    return VisibilityEdge;
}());
exports.VisibilityEdge = VisibilityEdge;
var TangentVisibilityGraph = (function () {
    function TangentVisibilityGraph(P, g0) {
        this.P = P;
        this.V = [];
        this.E = [];
        if (!g0) {
            var n = P.length;
            for (var i = 0; i < n; i++) {
                var p = P[i];
                for (var j = 0; j < p.length; ++j) {
                    var pj = p[j], vv = new VisibilityVertex(this.V.length, i, j, pj);
                    this.V.push(vv);
                    if (j > 0)
                        this.E.push(new VisibilityEdge(p[j - 1].vv, vv));
                }
                if (p.length > 1)
                    this.E.push(new VisibilityEdge(p[0].vv, p[p.length - 1].vv));
            }
            for (var i = 0; i < n - 1; i++) {
                var Pi = P[i];
                for (var j = i + 1; j < n; j++) {
                    var Pj = P[j], t = tangents(Pi, Pj);
                    for (var q in t) {
                        var c = t[q], source = Pi[c.t1], target = Pj[c.t2];
                        this.addEdgeIfVisible(source, target, i, j);
                    }
                }
            }
        }
        else {
            this.V = g0.V.slice(0);
            this.E = g0.E.slice(0);
        }
    }
    TangentVisibilityGraph.prototype.addEdgeIfVisible = function (u, v, i1, i2) {
        if (!this.intersectsPolys(new LineSegment(u.x, u.y, v.x, v.y), i1, i2)) {
            this.E.push(new VisibilityEdge(u.vv, v.vv));
        }
    };
    TangentVisibilityGraph.prototype.addPoint = function (p, i1) {
        var n = this.P.length;
        this.V.push(new VisibilityVertex(this.V.length, n, 0, p));
        for (var i = 0; i < n; ++i) {
            if (i === i1)
                continue;
            var poly = this.P[i], t = tangent_PointPolyC(p, poly);
            this.addEdgeIfVisible(p, poly[t.ltan], i1, i);
            this.addEdgeIfVisible(p, poly[t.rtan], i1, i);
        }
        return p.vv;
    };
    TangentVisibilityGraph.prototype.intersectsPolys = function (l, i1, i2) {
        for (var i = 0, n = this.P.length; i < n; ++i) {
            if (i != i1 && i != i2 && intersects(l, this.P[i]).length > 0) {
                return true;
            }
        }
        return false;
    };
    return TangentVisibilityGraph;
}());
exports.TangentVisibilityGraph = TangentVisibilityGraph;
function intersects(l, P) {
    var ints = [];
    for (var i = 1, n = P.length; i < n; ++i) {
        var int = rectangle_1.Rectangle.lineIntersection(l.x1, l.y1, l.x2, l.y2, P[i - 1].x, P[i - 1].y, P[i].x, P[i].y);
        if (int)
            ints.push(int);
    }
    return ints;
}
function tangents(V, W) {
    var m = V.length - 1, n = W.length - 1;
    var bt = new BiTangents();
    for (var i = 0; i < m; ++i) {
        for (var j = 0; j < n; ++j) {
            var v1 = V[i == 0 ? m - 1 : i - 1];
            var v2 = V[i];
            var v3 = V[i + 1];
            var w1 = W[j == 0 ? n - 1 : j - 1];
            var w2 = W[j];
            var w3 = W[j + 1];
            var v1v2w2 = isLeft(v1, v2, w2);
            var v2w1w2 = isLeft(v2, w1, w2);
            var v2w2w3 = isLeft(v2, w2, w3);
            var w1w2v2 = isLeft(w1, w2, v2);
            var w2v1v2 = isLeft(w2, v1, v2);
            var w2v2v3 = isLeft(w2, v2, v3);
            if (v1v2w2 >= 0 && v2w1w2 >= 0 && v2w2w3 < 0
                && w1w2v2 >= 0 && w2v1v2 >= 0 && w2v2v3 < 0) {
                bt.ll = new BiTangent(i, j);
            }
            else if (v1v2w2 <= 0 && v2w1w2 <= 0 && v2w2w3 > 0
                && w1w2v2 <= 0 && w2v1v2 <= 0 && w2v2v3 > 0) {
                bt.rr = new BiTangent(i, j);
            }
            else if (v1v2w2 <= 0 && v2w1w2 > 0 && v2w2w3 <= 0
                && w1w2v2 >= 0 && w2v1v2 < 0 && w2v2v3 >= 0) {
                bt.rl = new BiTangent(i, j);
            }
            else if (v1v2w2 >= 0 && v2w1w2 < 0 && v2w2w3 >= 0
                && w1w2v2 <= 0 && w2v1v2 > 0 && w2v2v3 <= 0) {
                bt.lr = new BiTangent(i, j);
            }
        }
    }
    return bt;
}
exports.tangents = tangents;
function isPointInsidePoly(p, poly) {
    for (var i = 1, n = poly.length; i < n; ++i)
        if (below(poly[i - 1], poly[i], p))
            return false;
    return true;
}
function isAnyPInQ(p, q) {
    return !p.every(function (v) { return !isPointInsidePoly(v, q); });
}
function polysOverlap(p, q) {
    if (isAnyPInQ(p, q))
        return true;
    if (isAnyPInQ(q, p))
        return true;
    for (var i = 1, n = p.length; i < n; ++i) {
        var v = p[i], u = p[i - 1];
        if (intersects(new LineSegment(u.x, u.y, v.x, v.y), q).length > 0)
            return true;
    }
    return false;
}
exports.polysOverlap = polysOverlap;
},{"./rectangle":17}],9:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var rectangle_1 = require("./rectangle");
var vpsc_1 = require("./vpsc");
var shortestpaths_1 = require("./shortestpaths");
var NodeWrapper = (function () {
    function NodeWrapper(id, rect, children) {
        this.id = id;
        this.rect = rect;
        this.children = children;
        this.leaf = typeof children === 'undefined' || children.length === 0;
    }
    return NodeWrapper;
}());
exports.NodeWrapper = NodeWrapper;
var Vert = (function () {
    function Vert(id, x, y, node, line) {
        if (node === void 0) { node = null; }
        if (line === void 0) { line = null; }
        this.id = id;
        this.x = x;
        this.y = y;
        this.node = node;
        this.line = line;
    }
    return Vert;
}());
exports.Vert = Vert;
var LongestCommonSubsequence = (function () {
    function LongestCommonSubsequence(s, t) {
        this.s = s;
        this.t = t;
        var mf = LongestCommonSubsequence.findMatch(s, t);
        var tr = t.slice(0).reverse();
        var mr = LongestCommonSubsequence.findMatch(s, tr);
        if (mf.length >= mr.length) {
            this.length = mf.length;
            this.si = mf.si;
            this.ti = mf.ti;
            this.reversed = false;
        }
        else {
            this.length = mr.length;
            this.si = mr.si;
            this.ti = t.length - mr.ti - mr.length;
            this.reversed = true;
        }
    }
    LongestCommonSubsequence.findMatch = function (s, t) {
        var m = s.length;
        var n = t.length;
        var match = { length: 0, si: -1, ti: -1 };
        var l = new Array(m);
        for (var i = 0; i < m; i++) {
            l[i] = new Array(n);
            for (var j = 0; j < n; j++)
                if (s[i] === t[j]) {
                    var v = l[i][j] = (i === 0 || j === 0) ? 1 : l[i - 1][j - 1] + 1;
                    if (v > match.length) {
                        match.length = v;
                        match.si = i - v + 1;
                        match.ti = j - v + 1;
                    }
                    ;
                }
                else
                    l[i][j] = 0;
        }
        return match;
    };
    LongestCommonSubsequence.prototype.getSequence = function () {
        return this.length >= 0 ? this.s.slice(this.si, this.si + this.length) : [];
    };
    return LongestCommonSubsequence;
}());
exports.LongestCommonSubsequence = LongestCommonSubsequence;
var GridRouter = (function () {
    function GridRouter(originalnodes, accessor, groupPadding) {
        var _this = this;
        if (groupPadding === void 0) { groupPadding = 12; }
        this.originalnodes = originalnodes;
        this.groupPadding = groupPadding;
        this.leaves = null;
        this.nodes = originalnodes.map(function (v, i) { return new NodeWrapper(i, accessor.getBounds(v), accessor.getChildren(v)); });
        this.leaves = this.nodes.filter(function (v) { return v.leaf; });
        this.groups = this.nodes.filter(function (g) { return !g.leaf; });
        this.cols = this.getGridLines('x');
        this.rows = this.getGridLines('y');
        this.groups.forEach(function (v) {
            return v.children.forEach(function (c) { return _this.nodes[c].parent = v; });
        });
        this.root = { children: [] };
        this.nodes.forEach(function (v) {
            if (typeof v.parent === 'undefined') {
                v.parent = _this.root;
                _this.root.children.push(v.id);
            }
            v.ports = [];
        });
        this.backToFront = this.nodes.slice(0);
        this.backToFront.sort(function (x, y) { return _this.getDepth(x) - _this.getDepth(y); });
        var frontToBackGroups = this.backToFront.slice(0).reverse().filter(function (g) { return !g.leaf; });
        frontToBackGroups.forEach(function (v) {
            var r = rectangle_1.Rectangle.empty();
            v.children.forEach(function (c) { return r = r.union(_this.nodes[c].rect); });
            v.rect = r.inflate(_this.groupPadding);
        });
        var colMids = this.midPoints(this.cols.map(function (r) { return r.pos; }));
        var rowMids = this.midPoints(this.rows.map(function (r) { return r.pos; }));
        var rowx = colMids[0], rowX = colMids[colMids.length - 1];
        var coly = rowMids[0], colY = rowMids[rowMids.length - 1];
        var hlines = this.rows.map(function (r) { return ({ x1: rowx, x2: rowX, y1: r.pos, y2: r.pos }); })
            .concat(rowMids.map(function (m) { return ({ x1: rowx, x2: rowX, y1: m, y2: m }); }));
        var vlines = this.cols.map(function (c) { return ({ x1: c.pos, x2: c.pos, y1: coly, y2: colY }); })
            .concat(colMids.map(function (m) { return ({ x1: m, x2: m, y1: coly, y2: colY }); }));
        var lines = hlines.concat(vlines);
        lines.forEach(function (l) { return l.verts = []; });
        this.verts = [];
        this.edges = [];
        hlines.forEach(function (h) {
            return vlines.forEach(function (v) {
                var p = new Vert(_this.verts.length, v.x1, h.y1);
                h.verts.push(p);
                v.verts.push(p);
                _this.verts.push(p);
                var i = _this.backToFront.length;
                while (i-- > 0) {
                    var node = _this.backToFront[i], r = node.rect;
                    var dx = Math.abs(p.x - r.cx()), dy = Math.abs(p.y - r.cy());
                    if (dx < r.width() / 2 && dy < r.height() / 2) {
                        p.node = node;
                        break;
                    }
                }
            });
        });
        lines.forEach(function (l, li) {
            _this.nodes.forEach(function (v, i) {
                v.rect.lineIntersections(l.x1, l.y1, l.x2, l.y2).forEach(function (intersect, j) {
                    var p = new Vert(_this.verts.length, intersect.x, intersect.y, v, l);
                    _this.verts.push(p);
                    l.verts.push(p);
                    v.ports.push(p);
                });
            });
            var isHoriz = Math.abs(l.y1 - l.y2) < 0.1;
            var delta = function (a, b) { return isHoriz ? b.x - a.x : b.y - a.y; };
            l.verts.sort(delta);
            for (var i = 1; i < l.verts.length; i++) {
                var u = l.verts[i - 1], v = l.verts[i];
                if (u.node && u.node === v.node && u.node.leaf)
                    continue;
                _this.edges.push({ source: u.id, target: v.id, length: Math.abs(delta(u, v)) });
            }
        });
    }
    GridRouter.prototype.avg = function (a) { return a.reduce(function (x, y) { return x + y; }) / a.length; };
    GridRouter.prototype.getGridLines = function (axis) {
        var columns = [];
        var ls = this.leaves.slice(0, this.leaves.length);
        while (ls.length > 0) {
            var overlapping = ls.filter(function (v) { return v.rect['overlap' + axis.toUpperCase()](ls[0].rect); });
            var col = {
                nodes: overlapping,
                pos: this.avg(overlapping.map(function (v) { return v.rect['c' + axis](); }))
            };
            columns.push(col);
            col.nodes.forEach(function (v) { return ls.splice(ls.indexOf(v), 1); });
        }
        columns.sort(function (a, b) { return a.pos - b.pos; });
        return columns;
    };
    GridRouter.prototype.getDepth = function (v) {
        var depth = 0;
        while (v.parent !== this.root) {
            depth++;
            v = v.parent;
        }
        return depth;
    };
    GridRouter.prototype.midPoints = function (a) {
        var gap = a[1] - a[0];
        var mids = [a[0] - gap / 2];
        for (var i = 1; i < a.length; i++) {
            mids.push((a[i] + a[i - 1]) / 2);
        }
        mids.push(a[a.length - 1] + gap / 2);
        return mids;
    };
    GridRouter.prototype.findLineage = function (v) {
        var lineage = [v];
        do {
            v = v.parent;
            lineage.push(v);
        } while (v !== this.root);
        return lineage.reverse();
    };
    GridRouter.prototype.findAncestorPathBetween = function (a, b) {
        var aa = this.findLineage(a), ba = this.findLineage(b), i = 0;
        while (aa[i] === ba[i])
            i++;
        return { commonAncestor: aa[i - 1], lineages: aa.slice(i).concat(ba.slice(i)) };
    };
    GridRouter.prototype.siblingObstacles = function (a, b) {
        var _this = this;
        var path = this.findAncestorPathBetween(a, b);
        var lineageLookup = {};
        path.lineages.forEach(function (v) { return lineageLookup[v.id] = {}; });
        var obstacles = path.commonAncestor.children.filter(function (v) { return !(v in lineageLookup); });
        path.lineages
            .filter(function (v) { return v.parent !== path.commonAncestor; })
            .forEach(function (v) { return obstacles = obstacles.concat(v.parent.children.filter(function (c) { return c !== v.id; })); });
        return obstacles.map(function (v) { return _this.nodes[v]; });
    };
    GridRouter.getSegmentSets = function (routes, x, y) {
        var vsegments = [];
        for (var ei = 0; ei < routes.length; ei++) {
            var route = routes[ei];
            for (var si = 0; si < route.length; si++) {
                var s = route[si];
                s.edgeid = ei;
                s.i = si;
                var sdx = s[1][x] - s[0][x];
                if (Math.abs(sdx) < 0.1) {
                    vsegments.push(s);
                }
            }
        }
        vsegments.sort(function (a, b) { return a[0][x] - b[0][x]; });
        var vsegmentsets = [];
        var segmentset = null;
        for (var i = 0; i < vsegments.length; i++) {
            var s = vsegments[i];
            if (!segmentset || Math.abs(s[0][x] - segmentset.pos) > 0.1) {
                segmentset = { pos: s[0][x], segments: [] };
                vsegmentsets.push(segmentset);
            }
            segmentset.segments.push(s);
        }
        return vsegmentsets;
    };
    GridRouter.nudgeSegs = function (x, y, routes, segments, leftOf, gap) {
        var n = segments.length;
        if (n <= 1)
            return;
        var vs = segments.map(function (s) { return new vpsc_1.Variable(s[0][x]); });
        var cs = [];
        for (var i = 0; i < n; i++) {
            for (var j = 0; j < n; j++) {
                if (i === j)
                    continue;
                var s1 = segments[i], s2 = segments[j], e1 = s1.edgeid, e2 = s2.edgeid, lind = -1, rind = -1;
                if (x == 'x') {
                    if (leftOf(e1, e2)) {
                        if (s1[0][y] < s1[1][y]) {
                            lind = j, rind = i;
                        }
                        else {
                            lind = i, rind = j;
                        }
                    }
                }
                else {
                    if (leftOf(e1, e2)) {
                        if (s1[0][y] < s1[1][y]) {
                            lind = i, rind = j;
                        }
                        else {
                            lind = j, rind = i;
                        }
                    }
                }
                if (lind >= 0) {
                    cs.push(new vpsc_1.Constraint(vs[lind], vs[rind], gap));
                }
            }
        }
        var solver = new vpsc_1.Solver(vs, cs);
        solver.solve();
        vs.forEach(function (v, i) {
            var s = segments[i];
            var pos = v.position();
            s[0][x] = s[1][x] = pos;
            var route = routes[s.edgeid];
            if (s.i > 0)
                route[s.i - 1][1][x] = pos;
            if (s.i < route.length - 1)
                route[s.i + 1][0][x] = pos;
        });
    };
    GridRouter.nudgeSegments = function (routes, x, y, leftOf, gap) {
        var vsegmentsets = GridRouter.getSegmentSets(routes, x, y);
        for (var i = 0; i < vsegmentsets.length; i++) {
            var ss = vsegmentsets[i];
            var events = [];
            for (var j = 0; j < ss.segments.length; j++) {
                var s = ss.segments[j];
                events.push({ type: 0, s: s, pos: Math.min(s[0][y], s[1][y]) });
                events.push({ type: 1, s: s, pos: Math.max(s[0][y], s[1][y]) });
            }
            events.sort(function (a, b) { return a.pos - b.pos + a.type - b.type; });
            var open = [];
            var openCount = 0;
            events.forEach(function (e) {
                if (e.type === 0) {
                    open.push(e.s);
                    openCount++;
                }
                else {
                    openCount--;
                }
                if (openCount == 0) {
                    GridRouter.nudgeSegs(x, y, routes, open, leftOf, gap);
                    open = [];
                }
            });
        }
    };
    GridRouter.prototype.routeEdges = function (edges, nudgeGap, source, target) {
        var _this = this;
        var routePaths = edges.map(function (e) { return _this.route(source(e), target(e)); });
        var order = GridRouter.orderEdges(routePaths);
        var routes = routePaths.map(function (e) { return GridRouter.makeSegments(e); });
        GridRouter.nudgeSegments(routes, 'x', 'y', order, nudgeGap);
        GridRouter.nudgeSegments(routes, 'y', 'x', order, nudgeGap);
        GridRouter.unreverseEdges(routes, routePaths);
        return routes;
    };
    GridRouter.unreverseEdges = function (routes, routePaths) {
        routes.forEach(function (segments, i) {
            var path = routePaths[i];
            if (path.reversed) {
                segments.reverse();
                segments.forEach(function (segment) {
                    segment.reverse();
                });
            }
        });
    };
    GridRouter.angleBetween2Lines = function (line1, line2) {
        var angle1 = Math.atan2(line1[0].y - line1[1].y, line1[0].x - line1[1].x);
        var angle2 = Math.atan2(line2[0].y - line2[1].y, line2[0].x - line2[1].x);
        var diff = angle1 - angle2;
        if (diff > Math.PI || diff < -Math.PI) {
            diff = angle2 - angle1;
        }
        return diff;
    };
    GridRouter.isLeft = function (a, b, c) {
        return ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)) <= 0;
    };
    GridRouter.getOrder = function (pairs) {
        var outgoing = {};
        for (var i = 0; i < pairs.length; i++) {
            var p = pairs[i];
            if (typeof outgoing[p.l] === 'undefined')
                outgoing[p.l] = {};
            outgoing[p.l][p.r] = true;
        }
        return function (l, r) { return typeof outgoing[l] !== 'undefined' && outgoing[l][r]; };
    };
    GridRouter.orderEdges = function (edges) {
        var edgeOrder = [];
        for (var i = 0; i < edges.length - 1; i++) {
            for (var j = i + 1; j < edges.length; j++) {
                var e = edges[i], f = edges[j], lcs = new LongestCommonSubsequence(e, f);
                var u, vi, vj;
                if (lcs.length === 0)
                    continue;
                if (lcs.reversed) {
                    f.reverse();
                    f.reversed = true;
                    lcs = new LongestCommonSubsequence(e, f);
                }
                if ((lcs.si <= 0 || lcs.ti <= 0) &&
                    (lcs.si + lcs.length >= e.length || lcs.ti + lcs.length >= f.length)) {
                    edgeOrder.push({ l: i, r: j });
                    continue;
                }
                if (lcs.si + lcs.length >= e.length || lcs.ti + lcs.length >= f.length) {
                    u = e[lcs.si + 1];
                    vj = e[lcs.si - 1];
                    vi = f[lcs.ti - 1];
                }
                else {
                    u = e[lcs.si + lcs.length - 2];
                    vi = e[lcs.si + lcs.length];
                    vj = f[lcs.ti + lcs.length];
                }
                if (GridRouter.isLeft(u, vi, vj)) {
                    edgeOrder.push({ l: j, r: i });
                }
                else {
                    edgeOrder.push({ l: i, r: j });
                }
            }
        }
        return GridRouter.getOrder(edgeOrder);
    };
    GridRouter.makeSegments = function (path) {
        function copyPoint(p) {
            return { x: p.x, y: p.y };
        }
        var isStraight = function (a, b, c) { return Math.abs((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)) < 0.001; };
        var segments = [];
        var a = copyPoint(path[0]);
        for (var i = 1; i < path.length; i++) {
            var b = copyPoint(path[i]), c = i < path.length - 1 ? path[i + 1] : null;
            if (!c || !isStraight(a, b, c)) {
                segments.push([a, b]);
                a = b;
            }
        }
        return segments;
    };
    GridRouter.prototype.route = function (s, t) {
        var _this = this;
        var source = this.nodes[s], target = this.nodes[t];
        this.obstacles = this.siblingObstacles(source, target);
        var obstacleLookup = {};
        this.obstacles.forEach(function (o) { return obstacleLookup[o.id] = o; });
        this.passableEdges = this.edges.filter(function (e) {
            var u = _this.verts[e.source], v = _this.verts[e.target];
            return !(u.node && u.node.id in obstacleLookup
                || v.node && v.node.id in obstacleLookup);
        });
        for (var i = 1; i < source.ports.length; i++) {
            var u = source.ports[0].id;
            var v = source.ports[i].id;
            this.passableEdges.push({
                source: u,
                target: v,
                length: 0
            });
        }
        for (var i = 1; i < target.ports.length; i++) {
            var u = target.ports[0].id;
            var v = target.ports[i].id;
            this.passableEdges.push({
                source: u,
                target: v,
                length: 0
            });
        }
        var getSource = function (e) { return e.source; }, getTarget = function (e) { return e.target; }, getLength = function (e) { return e.length; };
        var shortestPathCalculator = new shortestpaths_1.Calculator(this.verts.length, this.passableEdges, getSource, getTarget, getLength);
        var bendPenalty = function (u, v, w) {
            var a = _this.verts[u], b = _this.verts[v], c = _this.verts[w];
            var dx = Math.abs(c.x - a.x), dy = Math.abs(c.y - a.y);
            if (a.node === source && a.node === b.node || b.node === target && b.node === c.node)
                return 0;
            return dx > 1 && dy > 1 ? 1000 : 0;
        };
        var shortestPath = shortestPathCalculator.PathFromNodeToNodeWithPrevCost(source.ports[0].id, target.ports[0].id, bendPenalty);
        var pathPoints = shortestPath.reverse().map(function (vi) { return _this.verts[vi]; });
        pathPoints.push(this.nodes[target.id].ports[0]);
        return pathPoints.filter(function (v, i) {
            return !(i < pathPoints.length - 1 && pathPoints[i + 1].node === source && v.node === source
                || i > 0 && v.node === target && pathPoints[i - 1].node === target);
        });
    };
    GridRouter.getRoutePath = function (route, cornerradius, arrowwidth, arrowheight) {
        var result = {
            routepath: 'M ' + route[0][0].x + ' ' + route[0][0].y + ' ',
            arrowpath: ''
        };
        if (route.length > 1) {
            for (var i = 0; i < route.length; i++) {
                var li = route[i];
                var x = li[1].x, y = li[1].y;
                var dx = x - li[0].x;
                var dy = y - li[0].y;
                if (i < route.length - 1) {
                    if (Math.abs(dx) > 0) {
                        x -= dx / Math.abs(dx) * cornerradius;
                    }
                    else {
                        y -= dy / Math.abs(dy) * cornerradius;
                    }
                    result.routepath += 'L ' + x + ' ' + y + ' ';
                    var l = route[i + 1];
                    var x0 = l[0].x, y0 = l[0].y;
                    var x1 = l[1].x;
                    var y1 = l[1].y;
                    dx = x1 - x0;
                    dy = y1 - y0;
                    var angle = GridRouter.angleBetween2Lines(li, l) < 0 ? 1 : 0;
                    var x2, y2;
                    if (Math.abs(dx) > 0) {
                        x2 = x0 + dx / Math.abs(dx) * cornerradius;
                        y2 = y0;
                    }
                    else {
                        x2 = x0;
                        y2 = y0 + dy / Math.abs(dy) * cornerradius;
                    }
                    var cx = Math.abs(x2 - x);
                    var cy = Math.abs(y2 - y);
                    result.routepath += 'A ' + cx + ' ' + cy + ' 0 0 ' + angle + ' ' + x2 + ' ' + y2 + ' ';
                }
                else {
                    var arrowtip = [x, y];
                    var arrowcorner1, arrowcorner2;
                    if (Math.abs(dx) > 0) {
                        x -= dx / Math.abs(dx) * arrowheight;
                        arrowcorner1 = [x, y + arrowwidth];
                        arrowcorner2 = [x, y - arrowwidth];
                    }
                    else {
                        y -= dy / Math.abs(dy) * arrowheight;
                        arrowcorner1 = [x + arrowwidth, y];
                        arrowcorner2 = [x - arrowwidth, y];
                    }
                    result.routepath += 'L ' + x + ' ' + y + ' ';
                    if (arrowheight > 0) {
                        result.arrowpath = 'M ' + arrowtip[0] + ' ' + arrowtip[1] + ' L ' + arrowcorner1[0] + ' ' + arrowcorner1[1]
                            + ' L ' + arrowcorner2[0] + ' ' + arrowcorner2[1];
                    }
                }
            }
        }
        else {
            var li = route[0];
            var x = li[1].x, y = li[1].y;
            var dx = x - li[0].x;
            var dy = y - li[0].y;
            var arrowtip = [x, y];
            var arrowcorner1, arrowcorner2;
            if (Math.abs(dx) > 0) {
                x -= dx / Math.abs(dx) * arrowheight;
                arrowcorner1 = [x, y + arrowwidth];
                arrowcorner2 = [x, y - arrowwidth];
            }
            else {
                y -= dy / Math.abs(dy) * arrowheight;
                arrowcorner1 = [x + arrowwidth, y];
                arrowcorner2 = [x - arrowwidth, y];
            }
            result.routepath += 'L ' + x + ' ' + y + ' ';
            if (arrowheight > 0) {
                result.arrowpath = 'M ' + arrowtip[0] + ' ' + arrowtip[1] + ' L ' + arrowcorner1[0] + ' ' + arrowcorner1[1]
                    + ' L ' + arrowcorner2[0] + ' ' + arrowcorner2[1];
            }
        }
        return result;
    };
    return GridRouter;
}());
exports.GridRouter = GridRouter;
},{"./rectangle":17,"./shortestpaths":18,"./vpsc":19}],10:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var packingOptions = {
    PADDING: 10,
    GOLDEN_SECTION: (1 + Math.sqrt(5)) / 2,
    FLOAT_EPSILON: 0.0001,
    MAX_INERATIONS: 100
};
function applyPacking(graphs, w, h, node_size, desired_ratio, centerGraph) {
    if (desired_ratio === void 0) { desired_ratio = 1; }
    if (centerGraph === void 0) { centerGraph = true; }
    var init_x = 0, init_y = 0, svg_width = w, svg_height = h, desired_ratio = typeof desired_ratio !== 'undefined' ? desired_ratio : 1, node_size = typeof node_size !== 'undefined' ? node_size : 0, real_width = 0, real_height = 0, min_width = 0, global_bottom = 0, line = [];
    if (graphs.length == 0)
        return;
    calculate_bb(graphs);
    apply(graphs, desired_ratio);
    if (centerGraph) {
        put_nodes_to_right_positions(graphs);
    }
    function calculate_bb(graphs) {
        graphs.forEach(function (g) {
            calculate_single_bb(g);
        });
        function calculate_single_bb(graph) {
            var min_x = Number.MAX_VALUE, min_y = Number.MAX_VALUE, max_x = 0, max_y = 0;
            graph.array.forEach(function (v) {
                var w = typeof v.width !== 'undefined' ? v.width : node_size;
                var h = typeof v.height !== 'undefined' ? v.height : node_size;
                w /= 2;
                h /= 2;
                max_x = Math.max(v.x + w, max_x);
                min_x = Math.min(v.x - w, min_x);
                max_y = Math.max(v.y + h, max_y);
                min_y = Math.min(v.y - h, min_y);
            });
            graph.width = max_x - min_x;
            graph.height = max_y - min_y;
        }
    }
    function put_nodes_to_right_positions(graphs) {
        graphs.forEach(function (g) {
            var center = { x: 0, y: 0 };
            g.array.forEach(function (node) {
                center.x += node.x;
                center.y += node.y;
            });
            center.x /= g.array.length;
            center.y /= g.array.length;
            var corner = { x: center.x - g.width / 2, y: center.y - g.height / 2 };
            var offset = { x: g.x - corner.x + svg_width / 2 - real_width / 2, y: g.y - corner.y + svg_height / 2 - real_height / 2 };
            g.array.forEach(function (node) {
                node.x += offset.x;
                node.y += offset.y;
            });
        });
    }
    function apply(data, desired_ratio) {
        var curr_best_f = Number.POSITIVE_INFINITY;
        var curr_best = 0;
        data.sort(function (a, b) { return b.height - a.height; });
        min_width = data.reduce(function (a, b) {
            return a.width < b.width ? a.width : b.width;
        });
        var left = x1 = min_width;
        var right = x2 = get_entire_width(data);
        var iterationCounter = 0;
        var f_x1 = Number.MAX_VALUE;
        var f_x2 = Number.MAX_VALUE;
        var flag = -1;
        var dx = Number.MAX_VALUE;
        var df = Number.MAX_VALUE;
        while ((dx > min_width) || df > packingOptions.FLOAT_EPSILON) {
            if (flag != 1) {
                var x1 = right - (right - left) / packingOptions.GOLDEN_SECTION;
                var f_x1 = step(data, x1);
            }
            if (flag != 0) {
                var x2 = left + (right - left) / packingOptions.GOLDEN_SECTION;
                var f_x2 = step(data, x2);
            }
            dx = Math.abs(x1 - x2);
            df = Math.abs(f_x1 - f_x2);
            if (f_x1 < curr_best_f) {
                curr_best_f = f_x1;
                curr_best = x1;
            }
            if (f_x2 < curr_best_f) {
                curr_best_f = f_x2;
                curr_best = x2;
            }
            if (f_x1 > f_x2) {
                left = x1;
                x1 = x2;
                f_x1 = f_x2;
                flag = 1;
            }
            else {
                right = x2;
                x2 = x1;
                f_x2 = f_x1;
                flag = 0;
            }
            if (iterationCounter++ > 100) {
                break;
            }
        }
        step(data, curr_best);
    }
    function step(data, max_width) {
        line = [];
        real_width = 0;
        real_height = 0;
        global_bottom = init_y;
        for (var i = 0; i < data.length; i++) {
            var o = data[i];
            put_rect(o, max_width);
        }
        return Math.abs(get_real_ratio() - desired_ratio);
    }
    function put_rect(rect, max_width) {
        var parent = undefined;
        for (var i = 0; i < line.length; i++) {
            if ((line[i].space_left >= rect.height) && (line[i].x + line[i].width + rect.width + packingOptions.PADDING - max_width) <= packingOptions.FLOAT_EPSILON) {
                parent = line[i];
                break;
            }
        }
        line.push(rect);
        if (parent !== undefined) {
            rect.x = parent.x + parent.width + packingOptions.PADDING;
            rect.y = parent.bottom;
            rect.space_left = rect.height;
            rect.bottom = rect.y;
            parent.space_left -= rect.height + packingOptions.PADDING;
            parent.bottom += rect.height + packingOptions.PADDING;
        }
        else {
            rect.y = global_bottom;
            global_bottom += rect.height + packingOptions.PADDING;
            rect.x = init_x;
            rect.bottom = rect.y;
            rect.space_left = rect.height;
        }
        if (rect.y + rect.height - real_height > -packingOptions.FLOAT_EPSILON)
            real_height = rect.y + rect.height - init_y;
        if (rect.x + rect.width - real_width > -packingOptions.FLOAT_EPSILON)
            real_width = rect.x + rect.width - init_x;
    }
    ;
    function get_entire_width(data) {
        var width = 0;
        data.forEach(function (d) { return width += d.width + packingOptions.PADDING; });
        return width;
    }
    function get_real_ratio() {
        return (real_width / real_height);
    }
}
exports.applyPacking = applyPacking;
function separateGraphs(nodes, links) {
    var marks = {};
    var ways = {};
    var graphs = [];
    var clusters = 0;
    for (var i = 0; i < links.length; i++) {
        var link = links[i];
        var n1 = link.source;
        var n2 = link.target;
        if (ways[n1.index])
            ways[n1.index].push(n2);
        else
            ways[n1.index] = [n2];
        if (ways[n2.index])
            ways[n2.index].push(n1);
        else
            ways[n2.index] = [n1];
    }
    for (var i = 0; i < nodes.length; i++) {
        var node = nodes[i];
        if (marks[node.index])
            continue;
        explore_node(node, true);
    }
    function explore_node(n, is_new) {
        if (marks[n.index] !== undefined)
            return;
        if (is_new) {
            clusters++;
            graphs.push({ array: [] });
        }
        marks[n.index] = clusters;
        graphs[clusters - 1].array.push(n);
        var adjacent = ways[n.index];
        if (!adjacent)
            return;
        for (var j = 0; j < adjacent.length; j++) {
            explore_node(adjacent[j], false);
        }
    }
    return graphs;
}
exports.separateGraphs = separateGraphs;
},{}],11:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var powergraph = require("./powergraph");
var linklengths_1 = require("./linklengths");
var descent_1 = require("./descent");
var rectangle_1 = require("./rectangle");
var shortestpaths_1 = require("./shortestpaths");
var geom_1 = require("./geom");
var handledisconnected_1 = require("./handledisconnected");
var EventType;
(function (EventType) {
    EventType[EventType["start"] = 0] = "start";
    EventType[EventType["tick"] = 1] = "tick";
    EventType[EventType["end"] = 2] = "end";
})(EventType = exports.EventType || (exports.EventType = {}));
;
function isGroup(g) {
    return typeof g.leaves !== 'undefined' || typeof g.groups !== 'undefined';
}
var Layout = (function () {
    function Layout() {
        var _this = this;
        this._canvasSize = [1, 1];
        this._linkDistance = 20;
        this._defaultNodeSize = 10;
        this._linkLengthCalculator = null;
        this._linkType = null;
        this._avoidOverlaps = false;
        this._handleDisconnected = true;
        this._running = false;
        this._nodes = [];
        this._groups = [];
        this._rootGroup = null;
        this._links = [];
        this._constraints = [];
        this._distanceMatrix = null;
        this._descent = null;
        this._directedLinkConstraints = null;
        this._threshold = 0.01;
        this._visibilityGraph = null;
        this._groupCompactness = 1e-6;
        this.event = null;
        this.linkAccessor = {
            getSourceIndex: Layout.getSourceIndex,
            getTargetIndex: Layout.getTargetIndex,
            setLength: Layout.setLinkLength,
            getType: function (l) { return typeof _this._linkType === "function" ? _this._linkType(l) : 0; }
        };
    }
    Layout.prototype.on = function (e, listener) {
        if (!this.event)
            this.event = {};
        if (typeof e === 'string') {
            this.event[EventType[e]] = listener;
        }
        else {
            this.event[e] = listener;
        }
        return this;
    };
    Layout.prototype.trigger = function (e) {
        if (this.event && typeof this.event[e.type] !== 'undefined') {
            this.event[e.type](e);
        }
    };
    Layout.prototype.kick = function () {
        while (!this.tick())
            ;
    };
    Layout.prototype.tick = function () {
        if (this._alpha < this._threshold) {
            this._running = false;
            this.trigger({ type: EventType.end, alpha: this._alpha = 0, stress: this._lastStress });
            return true;
        }
        var n = this._nodes.length, m = this._links.length;
        var o, i;
        this._descent.locks.clear();
        for (i = 0; i < n; ++i) {
            o = this._nodes[i];
            if (o.fixed) {
                if (typeof o.px === 'undefined' || typeof o.py === 'undefined') {
                    o.px = o.x;
                    o.py = o.y;
                }
                var p = [o.px, o.py];
                this._descent.locks.add(i, p);
            }
        }
        var s1 = this._descent.rungeKutta();
        if (s1 === 0) {
            this._alpha = 0;
        }
        else if (typeof this._lastStress !== 'undefined') {
            this._alpha = s1;
        }
        this._lastStress = s1;
        this.updateNodePositions();
        this.trigger({ type: EventType.tick, alpha: this._alpha, stress: this._lastStress });
        return false;
    };
    Layout.prototype.updateNodePositions = function () {
        var x = this._descent.x[0], y = this._descent.x[1];
        var o, i = this._nodes.length;
        while (i--) {
            o = this._nodes[i];
            o.x = x[i];
            o.y = y[i];
        }
    };
    Layout.prototype.nodes = function (v) {
        if (!v) {
            if (this._nodes.length === 0 && this._links.length > 0) {
                var n = 0;
                this._links.forEach(function (l) {
                    n = Math.max(n, l.source, l.target);
                });
                this._nodes = new Array(++n);
                for (var i = 0; i < n; ++i) {
                    this._nodes[i] = {};
                }
            }
            return this._nodes;
        }
        this._nodes = v;
        return this;
    };
    Layout.prototype.groups = function (x) {
        var _this = this;
        if (!x)
            return this._groups;
        this._groups = x;
        this._rootGroup = {};
        this._groups.forEach(function (g) {
            if (typeof g.padding === "undefined")
                g.padding = 1;
            if (typeof g.leaves !== "undefined") {
                g.leaves.forEach(function (v, i) {
                    if (typeof v === 'number')
                        (g.leaves[i] = _this._nodes[v]).parent = g;
                });
            }
            if (typeof g.groups !== "undefined") {
                g.groups.forEach(function (gi, i) {
                    if (typeof gi === 'number')
                        (g.groups[i] = _this._groups[gi]).parent = g;
                });
            }
        });
        this._rootGroup.leaves = this._nodes.filter(function (v) { return typeof v.parent === 'undefined'; });
        this._rootGroup.groups = this._groups.filter(function (g) { return typeof g.parent === 'undefined'; });
        return this;
    };
    Layout.prototype.powerGraphGroups = function (f) {
        var g = powergraph.getGroups(this._nodes, this._links, this.linkAccessor, this._rootGroup);
        this.groups(g.groups);
        f(g);
        return this;
    };
    Layout.prototype.avoidOverlaps = function (v) {
        if (!arguments.length)
            return this._avoidOverlaps;
        this._avoidOverlaps = v;
        return this;
    };
    Layout.prototype.handleDisconnected = function (v) {
        if (!arguments.length)
            return this._handleDisconnected;
        this._handleDisconnected = v;
        return this;
    };
    Layout.prototype.flowLayout = function (axis, minSeparation) {
        if (!arguments.length)
            axis = 'y';
        this._directedLinkConstraints = {
            axis: axis,
            getMinSeparation: typeof minSeparation === 'number' ? function () { return minSeparation; } : minSeparation
        };
        return this;
    };
    Layout.prototype.links = function (x) {
        if (!arguments.length)
            return this._links;
        this._links = x;
        return this;
    };
    Layout.prototype.constraints = function (c) {
        if (!arguments.length)
            return this._constraints;
        this._constraints = c;
        return this;
    };
    Layout.prototype.distanceMatrix = function (d) {
        if (!arguments.length)
            return this._distanceMatrix;
        this._distanceMatrix = d;
        return this;
    };
    Layout.prototype.size = function (x) {
        if (!x)
            return this._canvasSize;
        this._canvasSize = x;
        return this;
    };
    Layout.prototype.defaultNodeSize = function (x) {
        if (!x)
            return this._defaultNodeSize;
        this._defaultNodeSize = x;
        return this;
    };
    Layout.prototype.groupCompactness = function (x) {
        if (!x)
            return this._groupCompactness;
        this._groupCompactness = x;
        return this;
    };
    Layout.prototype.linkDistance = function (x) {
        if (!x) {
            return this._linkDistance;
        }
        this._linkDistance = typeof x === "function" ? x : +x;
        this._linkLengthCalculator = null;
        return this;
    };
    Layout.prototype.linkType = function (f) {
        this._linkType = f;
        return this;
    };
    Layout.prototype.convergenceThreshold = function (x) {
        if (!x)
            return this._threshold;
        this._threshold = typeof x === "function" ? x : +x;
        return this;
    };
    Layout.prototype.alpha = function (x) {
        if (!arguments.length)
            return this._alpha;
        else {
            x = +x;
            if (this._alpha) {
                if (x > 0)
                    this._alpha = x;
                else
                    this._alpha = 0;
            }
            else if (x > 0) {
                if (!this._running) {
                    this._running = true;
                    this.trigger({ type: EventType.start, alpha: this._alpha = x });
                    this.kick();
                }
            }
            return this;
        }
    };
    Layout.prototype.getLinkLength = function (link) {
        return typeof this._linkDistance === "function" ? +(this._linkDistance(link)) : this._linkDistance;
    };
    Layout.setLinkLength = function (link, length) {
        link.length = length;
    };
    Layout.prototype.getLinkType = function (link) {
        return typeof this._linkType === "function" ? this._linkType(link) : 0;
    };
    Layout.prototype.symmetricDiffLinkLengths = function (idealLength, w) {
        var _this = this;
        if (w === void 0) { w = 1; }
        this.linkDistance(function (l) { return idealLength * l.length; });
        this._linkLengthCalculator = function () { return linklengths_1.symmetricDiffLinkLengths(_this._links, _this.linkAccessor, w); };
        return this;
    };
    Layout.prototype.jaccardLinkLengths = function (idealLength, w) {
        var _this = this;
        if (w === void 0) { w = 1; }
        this.linkDistance(function (l) { return idealLength * l.length; });
        this._linkLengthCalculator = function () { return linklengths_1.jaccardLinkLengths(_this._links, _this.linkAccessor, w); };
        return this;
    };
    Layout.prototype.start = function (initialUnconstrainedIterations, initialUserConstraintIterations, initialAllConstraintsIterations, gridSnapIterations, keepRunning, centerGraph) {
        var _this = this;
        if (initialUnconstrainedIterations === void 0) { initialUnconstrainedIterations = 0; }
        if (initialUserConstraintIterations === void 0) { initialUserConstraintIterations = 0; }
        if (initialAllConstraintsIterations === void 0) { initialAllConstraintsIterations = 0; }
        if (gridSnapIterations === void 0) { gridSnapIterations = 0; }
        if (keepRunning === void 0) { keepRunning = true; }
        if (centerGraph === void 0) { centerGraph = true; }
        var i, j, n = this.nodes().length, N = n + 2 * this._groups.length, m = this._links.length, w = this._canvasSize[0], h = this._canvasSize[1];
        var x = new Array(N), y = new Array(N);
        var G = null;
        var ao = this._avoidOverlaps;
        this._nodes.forEach(function (v, i) {
            v.index = i;
            if (typeof v.x === 'undefined') {
                v.x = w / 2, v.y = h / 2;
            }
            x[i] = v.x, y[i] = v.y;
        });
        if (this._linkLengthCalculator)
            this._linkLengthCalculator();
        var distances;
        if (this._distanceMatrix) {
            distances = this._distanceMatrix;
        }
        else {
            distances = (new shortestpaths_1.Calculator(N, this._links, Layout.getSourceIndex, Layout.getTargetIndex, function (l) { return _this.getLinkLength(l); })).DistanceMatrix();
            G = descent_1.Descent.createSquareMatrix(N, function () { return 2; });
            this._links.forEach(function (l) {
                if (typeof l.source == "number")
                    l.source = _this._nodes[l.source];
                if (typeof l.target == "number")
                    l.target = _this._nodes[l.target];
            });
            this._links.forEach(function (e) {
                var u = Layout.getSourceIndex(e), v = Layout.getTargetIndex(e);
                G[u][v] = G[v][u] = e.weight || 1;
            });
        }
        var D = descent_1.Descent.createSquareMatrix(N, function (i, j) {
            return distances[i][j];
        });
        if (this._rootGroup && typeof this._rootGroup.groups !== 'undefined') {
            var i = n;
            var addAttraction = function (i, j, strength, idealDistance) {
                G[i][j] = G[j][i] = strength;
                D[i][j] = D[j][i] = idealDistance;
            };
            this._groups.forEach(function (g) {
                addAttraction(i, i + 1, _this._groupCompactness, 0.1);
                x[i] = 0, y[i++] = 0;
                x[i] = 0, y[i++] = 0;
            });
        }
        else
            this._rootGroup = { leaves: this._nodes, groups: [] };
        var curConstraints = this._constraints || [];
        if (this._directedLinkConstraints) {
            this.linkAccessor.getMinSeparation = this._directedLinkConstraints.getMinSeparation;
            curConstraints = curConstraints.concat(linklengths_1.generateDirectedEdgeConstraints(n, this._links, this._directedLinkConstraints.axis, (this.linkAccessor)));
        }
        this.avoidOverlaps(false);
        this._descent = new descent_1.Descent([x, y], D);
        this._descent.locks.clear();
        for (var i = 0; i < n; ++i) {
            var o = this._nodes[i];
            if (o.fixed) {
                o.px = o.x;
                o.py = o.y;
                var p = [o.x, o.y];
                this._descent.locks.add(i, p);
            }
        }
        this._descent.threshold = this._threshold;
        this.initialLayout(initialUnconstrainedIterations, x, y);
        if (curConstraints.length > 0)
            this._descent.project = new rectangle_1.Projection(this._nodes, this._groups, this._rootGroup, curConstraints).projectFunctions();
        this._descent.run(initialUserConstraintIterations);
        this.separateOverlappingComponents(w, h, centerGraph);
        this.avoidOverlaps(ao);
        if (ao) {
            this._nodes.forEach(function (v, i) { v.x = x[i], v.y = y[i]; });
            this._descent.project = new rectangle_1.Projection(this._nodes, this._groups, this._rootGroup, curConstraints, true).projectFunctions();
            this._nodes.forEach(function (v, i) { x[i] = v.x, y[i] = v.y; });
        }
        this._descent.G = G;
        this._descent.run(initialAllConstraintsIterations);
        if (gridSnapIterations) {
            this._descent.snapStrength = 1000;
            this._descent.snapGridSize = this._nodes[0].width;
            this._descent.numGridSnapNodes = n;
            this._descent.scaleSnapByMaxH = n != N;
            var G0 = descent_1.Descent.createSquareMatrix(N, function (i, j) {
                if (i >= n || j >= n)
                    return G[i][j];
                return 0;
            });
            this._descent.G = G0;
            this._descent.run(gridSnapIterations);
        }
        this.updateNodePositions();
        this.separateOverlappingComponents(w, h, centerGraph);
        return keepRunning ? this.resume() : this;
    };
    Layout.prototype.initialLayout = function (iterations, x, y) {
        if (this._groups.length > 0 && iterations > 0) {
            var n = this._nodes.length;
            var edges = this._links.map(function (e) { return ({ source: e.source.index, target: e.target.index }); });
            var vs = this._nodes.map(function (v) { return ({ index: v.index }); });
            this._groups.forEach(function (g, i) {
                vs.push({ index: g.index = n + i });
            });
            this._groups.forEach(function (g, i) {
                if (typeof g.leaves !== 'undefined')
                    g.leaves.forEach(function (v) { return edges.push({ source: g.index, target: v.index }); });
                if (typeof g.groups !== 'undefined')
                    g.groups.forEach(function (gg) { return edges.push({ source: g.index, target: gg.index }); });
            });
            new Layout()
                .size(this.size())
                .nodes(vs)
                .links(edges)
                .avoidOverlaps(false)
                .linkDistance(this.linkDistance())
                .symmetricDiffLinkLengths(5)
                .convergenceThreshold(1e-4)
                .start(iterations, 0, 0, 0, false);
            this._nodes.forEach(function (v) {
                x[v.index] = vs[v.index].x;
                y[v.index] = vs[v.index].y;
            });
        }
        else {
            this._descent.run(iterations);
        }
    };
    Layout.prototype.separateOverlappingComponents = function (width, height, centerGraph) {
        var _this = this;
        if (centerGraph === void 0) { centerGraph = true; }
        if (!this._distanceMatrix && this._handleDisconnected) {
            var x_1 = this._descent.x[0], y_1 = this._descent.x[1];
            this._nodes.forEach(function (v, i) { v.x = x_1[i], v.y = y_1[i]; });
            var graphs = handledisconnected_1.separateGraphs(this._nodes, this._links);
            handledisconnected_1.applyPacking(graphs, width, height, this._defaultNodeSize, 1, centerGraph);
            this._nodes.forEach(function (v, i) {
                _this._descent.x[0][i] = v.x, _this._descent.x[1][i] = v.y;
                if (v.bounds) {
                    v.bounds.setXCentre(v.x);
                    v.bounds.setYCentre(v.y);
                }
            });
        }
    };
    Layout.prototype.resume = function () {
        return this.alpha(0.1);
    };
    Layout.prototype.stop = function () {
        return this.alpha(0);
    };
    Layout.prototype.prepareEdgeRouting = function (nodeMargin) {
        if (nodeMargin === void 0) { nodeMargin = 0; }
        this._visibilityGraph = new geom_1.TangentVisibilityGraph(this._nodes.map(function (v) {
            return v.bounds.inflate(-nodeMargin).vertices();
        }));
    };
    Layout.prototype.routeEdge = function (edge, ah, draw) {
        if (ah === void 0) { ah = 5; }
        var lineData = [];
        var vg2 = new geom_1.TangentVisibilityGraph(this._visibilityGraph.P, { V: this._visibilityGraph.V, E: this._visibilityGraph.E }), port1 = { x: edge.source.x, y: edge.source.y }, port2 = { x: edge.target.x, y: edge.target.y }, start = vg2.addPoint(port1, edge.source.index), end = vg2.addPoint(port2, edge.target.index);
        vg2.addEdgeIfVisible(port1, port2, edge.source.index, edge.target.index);
        if (typeof draw !== 'undefined') {
            draw(vg2);
        }
        var sourceInd = function (e) { return e.source.id; }, targetInd = function (e) { return e.target.id; }, length = function (e) { return e.length(); }, spCalc = new shortestpaths_1.Calculator(vg2.V.length, vg2.E, sourceInd, targetInd, length), shortestPath = spCalc.PathFromNodeToNode(start.id, end.id);
        if (shortestPath.length === 1 || shortestPath.length === vg2.V.length) {
            var route = rectangle_1.makeEdgeBetween(edge.source.innerBounds, edge.target.innerBounds, ah);
            lineData = [route.sourceIntersection, route.arrowStart];
        }
        else {
            var n = shortestPath.length - 2, p = vg2.V[shortestPath[n]].p, q = vg2.V[shortestPath[0]].p, lineData = [edge.source.innerBounds.rayIntersection(p.x, p.y)];
            for (var i = n; i >= 0; --i)
                lineData.push(vg2.V[shortestPath[i]].p);
            lineData.push(rectangle_1.makeEdgeTo(q, edge.target.innerBounds, ah));
        }
        return lineData;
    };
    Layout.getSourceIndex = function (e) {
        return typeof e.source === 'number' ? e.source : e.source.index;
    };
    Layout.getTargetIndex = function (e) {
        return typeof e.target === 'number' ? e.target : e.target.index;
    };
    Layout.linkId = function (e) {
        return Layout.getSourceIndex(e) + "-" + Layout.getTargetIndex(e);
    };
    Layout.dragStart = function (d) {
        if (isGroup(d)) {
            Layout.storeOffset(d, Layout.dragOrigin(d));
        }
        else {
            Layout.stopNode(d);
            d.fixed |= 2;
        }
    };
    Layout.stopNode = function (v) {
        v.px = v.x;
        v.py = v.y;
    };
    Layout.storeOffset = function (d, origin) {
        if (typeof d.leaves !== 'undefined') {
            d.leaves.forEach(function (v) {
                v.fixed |= 2;
                Layout.stopNode(v);
                v._dragGroupOffsetX = v.x - origin.x;
                v._dragGroupOffsetY = v.y - origin.y;
            });
        }
        if (typeof d.groups !== 'undefined') {
            d.groups.forEach(function (g) { return Layout.storeOffset(g, origin); });
        }
    };
    Layout.dragOrigin = function (d) {
        if (isGroup(d)) {
            return {
                x: d.bounds.cx(),
                y: d.bounds.cy()
            };
        }
        else {
            return d;
        }
    };
    Layout.drag = function (d, position) {
        if (isGroup(d)) {
            if (typeof d.leaves !== 'undefined') {
                d.leaves.forEach(function (v) {
                    d.bounds.setXCentre(position.x);
                    d.bounds.setYCentre(position.y);
                    v.px = v._dragGroupOffsetX + position.x;
                    v.py = v._dragGroupOffsetY + position.y;
                });
            }
            if (typeof d.groups !== 'undefined') {
                d.groups.forEach(function (g) { return Layout.drag(g, position); });
            }
        }
        else {
            d.px = position.x;
            d.py = position.y;
        }
    };
    Layout.dragEnd = function (d) {
        if (isGroup(d)) {
            if (typeof d.leaves !== 'undefined') {
                d.leaves.forEach(function (v) {
                    Layout.dragEnd(v);
                    delete v._dragGroupOffsetX;
                    delete v._dragGroupOffsetY;
                });
            }
            if (typeof d.groups !== 'undefined') {
                d.groups.forEach(Layout.dragEnd);
            }
        }
        else {
            d.fixed &= ~6;
        }
    };
    Layout.mouseOver = function (d) {
        d.fixed |= 4;
        d.px = d.x, d.py = d.y;
    };
    Layout.mouseOut = function (d) {
        d.fixed &= ~4;
    };
    return Layout;
}());
exports.Layout = Layout;
},{"./descent":7,"./geom":8,"./handledisconnected":10,"./linklengths":13,"./powergraph":14,"./rectangle":17,"./shortestpaths":18}],12:[function(require,module,exports){
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
},{"./descent":7,"./linklengths":13,"./rectangle":17,"./shortestpaths":18}],13:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
function unionCount(a, b) {
    var u = {};
    for (var i in a)
        u[i] = {};
    for (var i in b)
        u[i] = {};
    return Object.keys(u).length;
}
function intersectionCount(a, b) {
    var n = 0;
    for (var i in a)
        if (typeof b[i] !== 'undefined')
            ++n;
    return n;
}
function getNeighbours(links, la) {
    var neighbours = {};
    var addNeighbours = function (u, v) {
        if (typeof neighbours[u] === 'undefined')
            neighbours[u] = {};
        neighbours[u][v] = {};
    };
    links.forEach(function (e) {
        var u = la.getSourceIndex(e), v = la.getTargetIndex(e);
        addNeighbours(u, v);
        addNeighbours(v, u);
    });
    return neighbours;
}
function computeLinkLengths(links, w, f, la) {
    var neighbours = getNeighbours(links, la);
    links.forEach(function (l) {
        var a = neighbours[la.getSourceIndex(l)];
        var b = neighbours[la.getTargetIndex(l)];
        la.setLength(l, 1 + w * f(a, b));
    });
}
function symmetricDiffLinkLengths(links, la, w) {
    if (w === void 0) { w = 1; }
    computeLinkLengths(links, w, function (a, b) { return Math.sqrt(unionCount(a, b) - intersectionCount(a, b)); }, la);
}
exports.symmetricDiffLinkLengths = symmetricDiffLinkLengths;
function jaccardLinkLengths(links, la, w) {
    if (w === void 0) { w = 1; }
    computeLinkLengths(links, w, function (a, b) {
        return Math.min(Object.keys(a).length, Object.keys(b).length) < 1.1 ? 0 : intersectionCount(a, b) / unionCount(a, b);
    }, la);
}
exports.jaccardLinkLengths = jaccardLinkLengths;
function generateDirectedEdgeConstraints(n, links, axis, la) {
    var components = stronglyConnectedComponents(n, links, la);
    var nodes = {};
    components.forEach(function (c, i) {
        return c.forEach(function (v) { return nodes[v] = i; });
    });
    var constraints = [];
    links.forEach(function (l) {
        var ui = la.getSourceIndex(l), vi = la.getTargetIndex(l), u = nodes[ui], v = nodes[vi];
        if (u !== v) {
            constraints.push({
                axis: axis,
                left: ui,
                right: vi,
                gap: la.getMinSeparation(l)
            });
        }
    });
    return constraints;
}
exports.generateDirectedEdgeConstraints = generateDirectedEdgeConstraints;
function stronglyConnectedComponents(numVertices, edges, la) {
    var nodes = [];
    var index = 0;
    var stack = [];
    var components = [];
    function strongConnect(v) {
        v.index = v.lowlink = index++;
        stack.push(v);
        v.onStack = true;
        for (var _i = 0, _a = v.out; _i < _a.length; _i++) {
            var w = _a[_i];
            if (typeof w.index === 'undefined') {
                strongConnect(w);
                v.lowlink = Math.min(v.lowlink, w.lowlink);
            }
            else if (w.onStack) {
                v.lowlink = Math.min(v.lowlink, w.index);
            }
        }
        if (v.lowlink === v.index) {
            var component = [];
            while (stack.length) {
                w = stack.pop();
                w.onStack = false;
                component.push(w);
                if (w === v)
                    break;
            }
            components.push(component.map(function (v) { return v.id; }));
        }
    }
    for (var i = 0; i < numVertices; i++) {
        nodes.push({ id: i, out: [] });
    }
    for (var _i = 0, edges_1 = edges; _i < edges_1.length; _i++) {
        var e = edges_1[_i];
        var v_1 = nodes[la.getSourceIndex(e)], w = nodes[la.getTargetIndex(e)];
        v_1.out.push(w);
    }
    for (var _a = 0, nodes_1 = nodes; _a < nodes_1.length; _a++) {
        var v = nodes_1[_a];
        if (typeof v.index === 'undefined')
            strongConnect(v);
    }
    return components;
}
exports.stronglyConnectedComponents = stronglyConnectedComponents;
},{}],14:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var PowerEdge = (function () {
    function PowerEdge(source, target, type) {
        this.source = source;
        this.target = target;
        this.type = type;
    }
    return PowerEdge;
}());
exports.PowerEdge = PowerEdge;
var Configuration = (function () {
    function Configuration(n, edges, linkAccessor, rootGroup) {
        var _this = this;
        this.linkAccessor = linkAccessor;
        this.modules = new Array(n);
        this.roots = [];
        if (rootGroup) {
            this.initModulesFromGroup(rootGroup);
        }
        else {
            this.roots.push(new ModuleSet());
            for (var i = 0; i < n; ++i)
                this.roots[0].add(this.modules[i] = new Module(i));
        }
        this.R = edges.length;
        edges.forEach(function (e) {
            var s = _this.modules[linkAccessor.getSourceIndex(e)], t = _this.modules[linkAccessor.getTargetIndex(e)], type = linkAccessor.getType(e);
            s.outgoing.add(type, t);
            t.incoming.add(type, s);
        });
    }
    Configuration.prototype.initModulesFromGroup = function (group) {
        var moduleSet = new ModuleSet();
        this.roots.push(moduleSet);
        for (var i = 0; i < group.leaves.length; ++i) {
            var node = group.leaves[i];
            var module = new Module(node.id);
            this.modules[node.id] = module;
            moduleSet.add(module);
        }
        if (group.groups) {
            for (var j = 0; j < group.groups.length; ++j) {
                var child = group.groups[j];
                var definition = {};
                for (var prop in child)
                    if (prop !== "leaves" && prop !== "groups" && child.hasOwnProperty(prop))
                        definition[prop] = child[prop];
                moduleSet.add(new Module(-1 - j, new LinkSets(), new LinkSets(), this.initModulesFromGroup(child), definition));
            }
        }
        return moduleSet;
    };
    Configuration.prototype.merge = function (a, b, k) {
        if (k === void 0) { k = 0; }
        var inInt = a.incoming.intersection(b.incoming), outInt = a.outgoing.intersection(b.outgoing);
        var children = new ModuleSet();
        children.add(a);
        children.add(b);
        var m = new Module(this.modules.length, outInt, inInt, children);
        this.modules.push(m);
        var update = function (s, i, o) {
            s.forAll(function (ms, linktype) {
                ms.forAll(function (n) {
                    var nls = n[i];
                    nls.add(linktype, m);
                    nls.remove(linktype, a);
                    nls.remove(linktype, b);
                    a[o].remove(linktype, n);
                    b[o].remove(linktype, n);
                });
            });
        };
        update(outInt, "incoming", "outgoing");
        update(inInt, "outgoing", "incoming");
        this.R -= inInt.count() + outInt.count();
        this.roots[k].remove(a);
        this.roots[k].remove(b);
        this.roots[k].add(m);
        return m;
    };
    Configuration.prototype.rootMerges = function (k) {
        if (k === void 0) { k = 0; }
        var rs = this.roots[k].modules();
        var n = rs.length;
        var merges = new Array(n * (n - 1));
        var ctr = 0;
        for (var i = 0, i_ = n - 1; i < i_; ++i) {
            for (var j = i + 1; j < n; ++j) {
                var a = rs[i], b = rs[j];
                merges[ctr] = { id: ctr, nEdges: this.nEdges(a, b), a: a, b: b };
                ctr++;
            }
        }
        return merges;
    };
    Configuration.prototype.greedyMerge = function () {
        for (var i = 0; i < this.roots.length; ++i) {
            if (this.roots[i].modules().length < 2)
                continue;
            var ms = this.rootMerges(i).sort(function (a, b) { return a.nEdges == b.nEdges ? a.id - b.id : a.nEdges - b.nEdges; });
            var m = ms[0];
            if (m.nEdges >= this.R)
                continue;
            this.merge(m.a, m.b, i);
            return true;
        }
    };
    Configuration.prototype.nEdges = function (a, b) {
        var inInt = a.incoming.intersection(b.incoming), outInt = a.outgoing.intersection(b.outgoing);
        return this.R - inInt.count() - outInt.count();
    };
    Configuration.prototype.getGroupHierarchy = function (retargetedEdges) {
        var _this = this;
        var groups = [];
        var root = {};
        toGroups(this.roots[0], root, groups);
        var es = this.allEdges();
        es.forEach(function (e) {
            var a = _this.modules[e.source];
            var b = _this.modules[e.target];
            retargetedEdges.push(new PowerEdge(typeof a.gid === "undefined" ? e.source : groups[a.gid], typeof b.gid === "undefined" ? e.target : groups[b.gid], e.type));
        });
        return groups;
    };
    Configuration.prototype.allEdges = function () {
        var es = [];
        Configuration.getEdges(this.roots[0], es);
        return es;
    };
    Configuration.getEdges = function (modules, es) {
        modules.forAll(function (m) {
            m.getEdges(es);
            Configuration.getEdges(m.children, es);
        });
    };
    return Configuration;
}());
exports.Configuration = Configuration;
function toGroups(modules, group, groups) {
    modules.forAll(function (m) {
        if (m.isLeaf()) {
            if (!group.leaves)
                group.leaves = [];
            group.leaves.push(m.id);
        }
        else {
            var g = group;
            m.gid = groups.length;
            if (!m.isIsland() || m.isPredefined()) {
                g = { id: m.gid };
                if (m.isPredefined())
                    for (var prop in m.definition)
                        g[prop] = m.definition[prop];
                if (!group.groups)
                    group.groups = [];
                group.groups.push(m.gid);
                groups.push(g);
            }
            toGroups(m.children, g, groups);
        }
    });
}
var Module = (function () {
    function Module(id, outgoing, incoming, children, definition) {
        if (outgoing === void 0) { outgoing = new LinkSets(); }
        if (incoming === void 0) { incoming = new LinkSets(); }
        if (children === void 0) { children = new ModuleSet(); }
        this.id = id;
        this.outgoing = outgoing;
        this.incoming = incoming;
        this.children = children;
        this.definition = definition;
    }
    Module.prototype.getEdges = function (es) {
        var _this = this;
        this.outgoing.forAll(function (ms, edgetype) {
            ms.forAll(function (target) {
                es.push(new PowerEdge(_this.id, target.id, edgetype));
            });
        });
    };
    Module.prototype.isLeaf = function () {
        return this.children.count() === 0;
    };
    Module.prototype.isIsland = function () {
        return this.outgoing.count() === 0 && this.incoming.count() === 0;
    };
    Module.prototype.isPredefined = function () {
        return typeof this.definition !== "undefined";
    };
    return Module;
}());
exports.Module = Module;
function intersection(m, n) {
    var i = {};
    for (var v in m)
        if (v in n)
            i[v] = m[v];
    return i;
}
var ModuleSet = (function () {
    function ModuleSet() {
        this.table = {};
    }
    ModuleSet.prototype.count = function () {
        return Object.keys(this.table).length;
    };
    ModuleSet.prototype.intersection = function (other) {
        var result = new ModuleSet();
        result.table = intersection(this.table, other.table);
        return result;
    };
    ModuleSet.prototype.intersectionCount = function (other) {
        return this.intersection(other).count();
    };
    ModuleSet.prototype.contains = function (id) {
        return id in this.table;
    };
    ModuleSet.prototype.add = function (m) {
        this.table[m.id] = m;
    };
    ModuleSet.prototype.remove = function (m) {
        delete this.table[m.id];
    };
    ModuleSet.prototype.forAll = function (f) {
        for (var mid in this.table) {
            f(this.table[mid]);
        }
    };
    ModuleSet.prototype.modules = function () {
        var vs = [];
        this.forAll(function (m) {
            if (!m.isPredefined())
                vs.push(m);
        });
        return vs;
    };
    return ModuleSet;
}());
exports.ModuleSet = ModuleSet;
var LinkSets = (function () {
    function LinkSets() {
        this.sets = {};
        this.n = 0;
    }
    LinkSets.prototype.count = function () {
        return this.n;
    };
    LinkSets.prototype.contains = function (id) {
        var result = false;
        this.forAllModules(function (m) {
            if (!result && m.id == id) {
                result = true;
            }
        });
        return result;
    };
    LinkSets.prototype.add = function (linktype, m) {
        var s = linktype in this.sets ? this.sets[linktype] : this.sets[linktype] = new ModuleSet();
        s.add(m);
        ++this.n;
    };
    LinkSets.prototype.remove = function (linktype, m) {
        var ms = this.sets[linktype];
        ms.remove(m);
        if (ms.count() === 0) {
            delete this.sets[linktype];
        }
        --this.n;
    };
    LinkSets.prototype.forAll = function (f) {
        for (var linktype in this.sets) {
            f(this.sets[linktype], Number(linktype));
        }
    };
    LinkSets.prototype.forAllModules = function (f) {
        this.forAll(function (ms, lt) { return ms.forAll(f); });
    };
    LinkSets.prototype.intersection = function (other) {
        var result = new LinkSets();
        this.forAll(function (ms, lt) {
            if (lt in other.sets) {
                var i = ms.intersection(other.sets[lt]), n = i.count();
                if (n > 0) {
                    result.sets[lt] = i;
                    result.n += n;
                }
            }
        });
        return result;
    };
    return LinkSets;
}());
exports.LinkSets = LinkSets;
function intersectionCount(m, n) {
    return Object.keys(intersection(m, n)).length;
}
function getGroups(nodes, links, la, rootGroup) {
    var n = nodes.length, c = new Configuration(n, links, la, rootGroup);
    while (c.greedyMerge())
        ;
    var powerEdges = [];
    var g = c.getGroupHierarchy(powerEdges);
    powerEdges.forEach(function (e) {
        var f = function (end) {
            var g = e[end];
            if (typeof g == "number")
                e[end] = nodes[g];
        };
        f("source");
        f("target");
    });
    return { groups: g, powerEdges: powerEdges };
}
exports.getGroups = getGroups;
},{}],15:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var PairingHeap = (function () {
    function PairingHeap(elem) {
        this.elem = elem;
        this.subheaps = [];
    }
    PairingHeap.prototype.toString = function (selector) {
        var str = "", needComma = false;
        for (var i = 0; i < this.subheaps.length; ++i) {
            var subheap = this.subheaps[i];
            if (!subheap.elem) {
                needComma = false;
                continue;
            }
            if (needComma) {
                str = str + ",";
            }
            str = str + subheap.toString(selector);
            needComma = true;
        }
        if (str !== "") {
            str = "(" + str + ")";
        }
        return (this.elem ? selector(this.elem) : "") + str;
    };
    PairingHeap.prototype.forEach = function (f) {
        if (!this.empty()) {
            f(this.elem, this);
            this.subheaps.forEach(function (s) { return s.forEach(f); });
        }
    };
    PairingHeap.prototype.count = function () {
        return this.empty() ? 0 : 1 + this.subheaps.reduce(function (n, h) {
            return n + h.count();
        }, 0);
    };
    PairingHeap.prototype.min = function () {
        return this.elem;
    };
    PairingHeap.prototype.empty = function () {
        return this.elem == null;
    };
    PairingHeap.prototype.contains = function (h) {
        if (this === h)
            return true;
        for (var i = 0; i < this.subheaps.length; i++) {
            if (this.subheaps[i].contains(h))
                return true;
        }
        return false;
    };
    PairingHeap.prototype.isHeap = function (lessThan) {
        var _this = this;
        return this.subheaps.every(function (h) { return lessThan(_this.elem, h.elem) && h.isHeap(lessThan); });
    };
    PairingHeap.prototype.insert = function (obj, lessThan) {
        return this.merge(new PairingHeap(obj), lessThan);
    };
    PairingHeap.prototype.merge = function (heap2, lessThan) {
        if (this.empty())
            return heap2;
        else if (heap2.empty())
            return this;
        else if (lessThan(this.elem, heap2.elem)) {
            this.subheaps.push(heap2);
            return this;
        }
        else {
            heap2.subheaps.push(this);
            return heap2;
        }
    };
    PairingHeap.prototype.removeMin = function (lessThan) {
        if (this.empty())
            return null;
        else
            return this.mergePairs(lessThan);
    };
    PairingHeap.prototype.mergePairs = function (lessThan) {
        if (this.subheaps.length == 0)
            return new PairingHeap(null);
        else if (this.subheaps.length == 1) {
            return this.subheaps[0];
        }
        else {
            var firstPair = this.subheaps.pop().merge(this.subheaps.pop(), lessThan);
            var remaining = this.mergePairs(lessThan);
            return firstPair.merge(remaining, lessThan);
        }
    };
    PairingHeap.prototype.decreaseKey = function (subheap, newValue, setHeapNode, lessThan) {
        var newHeap = subheap.removeMin(lessThan);
        subheap.elem = newHeap.elem;
        subheap.subheaps = newHeap.subheaps;
        if (setHeapNode !== null && newHeap.elem !== null) {
            setHeapNode(subheap.elem, subheap);
        }
        var pairingNode = new PairingHeap(newValue);
        if (setHeapNode !== null) {
            setHeapNode(newValue, pairingNode);
        }
        return this.merge(pairingNode, lessThan);
    };
    return PairingHeap;
}());
exports.PairingHeap = PairingHeap;
var PriorityQueue = (function () {
    function PriorityQueue(lessThan) {
        this.lessThan = lessThan;
    }
    PriorityQueue.prototype.top = function () {
        if (this.empty()) {
            return null;
        }
        return this.root.elem;
    };
    PriorityQueue.prototype.push = function () {
        var args = [];
        for (var _i = 0; _i < arguments.length; _i++) {
            args[_i] = arguments[_i];
        }
        var pairingNode;
        for (var i = 0, arg; arg = args[i]; ++i) {
            pairingNode = new PairingHeap(arg);
            this.root = this.empty() ?
                pairingNode : this.root.merge(pairingNode, this.lessThan);
        }
        return pairingNode;
    };
    PriorityQueue.prototype.empty = function () {
        return !this.root || !this.root.elem;
    };
    PriorityQueue.prototype.isHeap = function () {
        return this.root.isHeap(this.lessThan);
    };
    PriorityQueue.prototype.forEach = function (f) {
        this.root.forEach(f);
    };
    PriorityQueue.prototype.pop = function () {
        if (this.empty()) {
            return null;
        }
        var obj = this.root.min();
        this.root = this.root.removeMin(this.lessThan);
        return obj;
    };
    PriorityQueue.prototype.reduceKey = function (heapNode, newKey, setHeapNode) {
        if (setHeapNode === void 0) { setHeapNode = null; }
        this.root = this.root.decreaseKey(heapNode, newKey, setHeapNode, this.lessThan);
    };
    PriorityQueue.prototype.toString = function (selector) {
        return this.root.toString(selector);
    };
    PriorityQueue.prototype.count = function () {
        return this.root.count();
    };
    return PriorityQueue;
}());
exports.PriorityQueue = PriorityQueue;
},{}],16:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var TreeBase = (function () {
    function TreeBase() {
        this.findIter = function (data) {
            var res = this._root;
            var iter = this.iterator();
            while (res !== null) {
                var c = this._comparator(data, res.data);
                if (c === 0) {
                    iter._cursor = res;
                    return iter;
                }
                else {
                    iter._ancestors.push(res);
                    res = res.get_child(c > 0);
                }
            }
            return null;
        };
    }
    TreeBase.prototype.clear = function () {
        this._root = null;
        this.size = 0;
    };
    ;
    TreeBase.prototype.find = function (data) {
        var res = this._root;
        while (res !== null) {
            var c = this._comparator(data, res.data);
            if (c === 0) {
                return res.data;
            }
            else {
                res = res.get_child(c > 0);
            }
        }
        return null;
    };
    ;
    TreeBase.prototype.lowerBound = function (data) {
        return this._bound(data, this._comparator);
    };
    ;
    TreeBase.prototype.upperBound = function (data) {
        var cmp = this._comparator;
        function reverse_cmp(a, b) {
            return cmp(b, a);
        }
        return this._bound(data, reverse_cmp);
    };
    ;
    TreeBase.prototype.min = function () {
        var res = this._root;
        if (res === null) {
            return null;
        }
        while (res.left !== null) {
            res = res.left;
        }
        return res.data;
    };
    ;
    TreeBase.prototype.max = function () {
        var res = this._root;
        if (res === null) {
            return null;
        }
        while (res.right !== null) {
            res = res.right;
        }
        return res.data;
    };
    ;
    TreeBase.prototype.iterator = function () {
        return new Iterator(this);
    };
    ;
    TreeBase.prototype.each = function (cb) {
        var it = this.iterator(), data;
        while ((data = it.next()) !== null) {
            cb(data);
        }
    };
    ;
    TreeBase.prototype.reach = function (cb) {
        var it = this.iterator(), data;
        while ((data = it.prev()) !== null) {
            cb(data);
        }
    };
    ;
    TreeBase.prototype._bound = function (data, cmp) {
        var cur = this._root;
        var iter = this.iterator();
        while (cur !== null) {
            var c = this._comparator(data, cur.data);
            if (c === 0) {
                iter._cursor = cur;
                return iter;
            }
            iter._ancestors.push(cur);
            cur = cur.get_child(c > 0);
        }
        for (var i = iter._ancestors.length - 1; i >= 0; --i) {
            cur = iter._ancestors[i];
            if (cmp(data, cur.data) > 0) {
                iter._cursor = cur;
                iter._ancestors.length = i;
                return iter;
            }
        }
        iter._ancestors.length = 0;
        return iter;
    };
    ;
    return TreeBase;
}());
exports.TreeBase = TreeBase;
var Iterator = (function () {
    function Iterator(tree) {
        this._tree = tree;
        this._ancestors = [];
        this._cursor = null;
    }
    Iterator.prototype.data = function () {
        return this._cursor !== null ? this._cursor.data : null;
    };
    ;
    Iterator.prototype.next = function () {
        if (this._cursor === null) {
            var root = this._tree._root;
            if (root !== null) {
                this._minNode(root);
            }
        }
        else {
            if (this._cursor.right === null) {
                var save;
                do {
                    save = this._cursor;
                    if (this._ancestors.length) {
                        this._cursor = this._ancestors.pop();
                    }
                    else {
                        this._cursor = null;
                        break;
                    }
                } while (this._cursor.right === save);
            }
            else {
                this._ancestors.push(this._cursor);
                this._minNode(this._cursor.right);
            }
        }
        return this._cursor !== null ? this._cursor.data : null;
    };
    ;
    Iterator.prototype.prev = function () {
        if (this._cursor === null) {
            var root = this._tree._root;
            if (root !== null) {
                this._maxNode(root);
            }
        }
        else {
            if (this._cursor.left === null) {
                var save;
                do {
                    save = this._cursor;
                    if (this._ancestors.length) {
                        this._cursor = this._ancestors.pop();
                    }
                    else {
                        this._cursor = null;
                        break;
                    }
                } while (this._cursor.left === save);
            }
            else {
                this._ancestors.push(this._cursor);
                this._maxNode(this._cursor.left);
            }
        }
        return this._cursor !== null ? this._cursor.data : null;
    };
    ;
    Iterator.prototype._minNode = function (start) {
        while (start.left !== null) {
            this._ancestors.push(start);
            start = start.left;
        }
        this._cursor = start;
    };
    ;
    Iterator.prototype._maxNode = function (start) {
        while (start.right !== null) {
            this._ancestors.push(start);
            start = start.right;
        }
        this._cursor = start;
    };
    ;
    return Iterator;
}());
exports.Iterator = Iterator;
var Node = (function () {
    function Node(data) {
        this.data = data;
        this.left = null;
        this.right = null;
        this.red = true;
    }
    Node.prototype.get_child = function (dir) {
        return dir ? this.right : this.left;
    };
    ;
    Node.prototype.set_child = function (dir, val) {
        if (dir) {
            this.right = val;
        }
        else {
            this.left = val;
        }
    };
    ;
    return Node;
}());
var RBTree = (function (_super) {
    __extends(RBTree, _super);
    function RBTree(comparator) {
        var _this = _super.call(this) || this;
        _this._root = null;
        _this._comparator = comparator;
        _this.size = 0;
        return _this;
    }
    RBTree.prototype.insert = function (data) {
        var ret = false;
        if (this._root === null) {
            this._root = new Node(data);
            ret = true;
            this.size++;
        }
        else {
            var head = new Node(undefined);
            var dir = false;
            var last = false;
            var gp = null;
            var ggp = head;
            var p = null;
            var node = this._root;
            ggp.right = this._root;
            while (true) {
                if (node === null) {
                    node = new Node(data);
                    p.set_child(dir, node);
                    ret = true;
                    this.size++;
                }
                else if (RBTree.is_red(node.left) && RBTree.is_red(node.right)) {
                    node.red = true;
                    node.left.red = false;
                    node.right.red = false;
                }
                if (RBTree.is_red(node) && RBTree.is_red(p)) {
                    var dir2 = ggp.right === gp;
                    if (node === p.get_child(last)) {
                        ggp.set_child(dir2, RBTree.single_rotate(gp, !last));
                    }
                    else {
                        ggp.set_child(dir2, RBTree.double_rotate(gp, !last));
                    }
                }
                var cmp = this._comparator(node.data, data);
                if (cmp === 0) {
                    break;
                }
                last = dir;
                dir = cmp < 0;
                if (gp !== null) {
                    ggp = gp;
                }
                gp = p;
                p = node;
                node = node.get_child(dir);
            }
            this._root = head.right;
        }
        this._root.red = false;
        return ret;
    };
    ;
    RBTree.prototype.remove = function (data) {
        if (this._root === null) {
            return false;
        }
        var head = new Node(undefined);
        var node = head;
        node.right = this._root;
        var p = null;
        var gp = null;
        var found = null;
        var dir = true;
        while (node.get_child(dir) !== null) {
            var last = dir;
            gp = p;
            p = node;
            node = node.get_child(dir);
            var cmp = this._comparator(data, node.data);
            dir = cmp > 0;
            if (cmp === 0) {
                found = node;
            }
            if (!RBTree.is_red(node) && !RBTree.is_red(node.get_child(dir))) {
                if (RBTree.is_red(node.get_child(!dir))) {
                    var sr = RBTree.single_rotate(node, dir);
                    p.set_child(last, sr);
                    p = sr;
                }
                else if (!RBTree.is_red(node.get_child(!dir))) {
                    var sibling = p.get_child(!last);
                    if (sibling !== null) {
                        if (!RBTree.is_red(sibling.get_child(!last)) && !RBTree.is_red(sibling.get_child(last))) {
                            p.red = false;
                            sibling.red = true;
                            node.red = true;
                        }
                        else {
                            var dir2 = gp.right === p;
                            if (RBTree.is_red(sibling.get_child(last))) {
                                gp.set_child(dir2, RBTree.double_rotate(p, last));
                            }
                            else if (RBTree.is_red(sibling.get_child(!last))) {
                                gp.set_child(dir2, RBTree.single_rotate(p, last));
                            }
                            var gpc = gp.get_child(dir2);
                            gpc.red = true;
                            node.red = true;
                            gpc.left.red = false;
                            gpc.right.red = false;
                        }
                    }
                }
            }
        }
        if (found !== null) {
            found.data = node.data;
            p.set_child(p.right === node, node.get_child(node.left === null));
            this.size--;
        }
        this._root = head.right;
        if (this._root !== null) {
            this._root.red = false;
        }
        return found !== null;
    };
    ;
    RBTree.is_red = function (node) {
        return node !== null && node.red;
    };
    RBTree.single_rotate = function (root, dir) {
        var save = root.get_child(!dir);
        root.set_child(!dir, save.get_child(dir));
        save.set_child(dir, root);
        root.red = true;
        save.red = false;
        return save;
    };
    RBTree.double_rotate = function (root, dir) {
        root.set_child(!dir, RBTree.single_rotate(root.get_child(!dir), !dir));
        return RBTree.single_rotate(root, dir);
    };
    return RBTree;
}(TreeBase));
exports.RBTree = RBTree;
},{}],17:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var vpsc_1 = require("./vpsc");
var rbtree_1 = require("./rbtree");
function computeGroupBounds(g) {
    g.bounds = typeof g.leaves !== "undefined" ?
        g.leaves.reduce(function (r, c) { return c.bounds.union(r); }, Rectangle.empty()) :
        Rectangle.empty();
    if (typeof g.groups !== "undefined")
        g.bounds = g.groups.reduce(function (r, c) { return computeGroupBounds(c).union(r); }, g.bounds);
    g.bounds = g.bounds.inflate(g.padding);
    return g.bounds;
}
exports.computeGroupBounds = computeGroupBounds;
var Rectangle = (function () {
    function Rectangle(x, X, y, Y) {
        this.x = x;
        this.X = X;
        this.y = y;
        this.Y = Y;
    }
    Rectangle.empty = function () { return new Rectangle(Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY); };
    Rectangle.prototype.cx = function () { return (this.x + this.X) / 2; };
    Rectangle.prototype.cy = function () { return (this.y + this.Y) / 2; };
    Rectangle.prototype.overlapX = function (r) {
        var ux = this.cx(), vx = r.cx();
        if (ux <= vx && r.x < this.X)
            return this.X - r.x;
        if (vx <= ux && this.x < r.X)
            return r.X - this.x;
        return 0;
    };
    Rectangle.prototype.overlapY = function (r) {
        var uy = this.cy(), vy = r.cy();
        if (uy <= vy && r.y < this.Y)
            return this.Y - r.y;
        if (vy <= uy && this.y < r.Y)
            return r.Y - this.y;
        return 0;
    };
    Rectangle.prototype.setXCentre = function (cx) {
        var dx = cx - this.cx();
        this.x += dx;
        this.X += dx;
    };
    Rectangle.prototype.setYCentre = function (cy) {
        var dy = cy - this.cy();
        this.y += dy;
        this.Y += dy;
    };
    Rectangle.prototype.width = function () {
        return this.X - this.x;
    };
    Rectangle.prototype.height = function () {
        return this.Y - this.y;
    };
    Rectangle.prototype.union = function (r) {
        return new Rectangle(Math.min(this.x, r.x), Math.max(this.X, r.X), Math.min(this.y, r.y), Math.max(this.Y, r.Y));
    };
    Rectangle.prototype.lineIntersections = function (x1, y1, x2, y2) {
        var sides = [[this.x, this.y, this.X, this.y],
            [this.X, this.y, this.X, this.Y],
            [this.X, this.Y, this.x, this.Y],
            [this.x, this.Y, this.x, this.y]];
        var intersections = [];
        for (var i = 0; i < 4; ++i) {
            var r = Rectangle.lineIntersection(x1, y1, x2, y2, sides[i][0], sides[i][1], sides[i][2], sides[i][3]);
            if (r !== null)
                intersections.push({ x: r.x, y: r.y });
        }
        return intersections;
    };
    Rectangle.prototype.rayIntersection = function (x2, y2) {
        var ints = this.lineIntersections(this.cx(), this.cy(), x2, y2);
        return ints.length > 0 ? ints[0] : null;
    };
    Rectangle.prototype.vertices = function () {
        return [
            { x: this.x, y: this.y },
            { x: this.X, y: this.y },
            { x: this.X, y: this.Y },
            { x: this.x, y: this.Y }
        ];
    };
    Rectangle.lineIntersection = function (x1, y1, x2, y2, x3, y3, x4, y4) {
        var dx12 = x2 - x1, dx34 = x4 - x3, dy12 = y2 - y1, dy34 = y4 - y3, denominator = dy34 * dx12 - dx34 * dy12;
        if (denominator == 0)
            return null;
        var dx31 = x1 - x3, dy31 = y1 - y3, numa = dx34 * dy31 - dy34 * dx31, a = numa / denominator, numb = dx12 * dy31 - dy12 * dx31, b = numb / denominator;
        if (a >= 0 && a <= 1 && b >= 0 && b <= 1) {
            return {
                x: x1 + a * dx12,
                y: y1 + a * dy12
            };
        }
        return null;
    };
    Rectangle.prototype.inflate = function (pad) {
        return new Rectangle(this.x - pad, this.X + pad, this.y - pad, this.Y + pad);
    };
    return Rectangle;
}());
exports.Rectangle = Rectangle;
function makeEdgeBetween(source, target, ah) {
    var si = source.rayIntersection(target.cx(), target.cy()) || { x: source.cx(), y: source.cy() }, ti = target.rayIntersection(source.cx(), source.cy()) || { x: target.cx(), y: target.cy() }, dx = ti.x - si.x, dy = ti.y - si.y, l = Math.sqrt(dx * dx + dy * dy), al = l - ah;
    return {
        sourceIntersection: si,
        targetIntersection: ti,
        arrowStart: { x: si.x + al * dx / l, y: si.y + al * dy / l }
    };
}
exports.makeEdgeBetween = makeEdgeBetween;
function makeEdgeTo(s, target, ah) {
    var ti = target.rayIntersection(s.x, s.y);
    if (!ti)
        ti = { x: target.cx(), y: target.cy() };
    var dx = ti.x - s.x, dy = ti.y - s.y, l = Math.sqrt(dx * dx + dy * dy);
    return { x: ti.x - ah * dx / l, y: ti.y - ah * dy / l };
}
exports.makeEdgeTo = makeEdgeTo;
var Node = (function () {
    function Node(v, r, pos) {
        this.v = v;
        this.r = r;
        this.pos = pos;
        this.prev = makeRBTree();
        this.next = makeRBTree();
    }
    return Node;
}());
var Event = (function () {
    function Event(isOpen, v, pos) {
        this.isOpen = isOpen;
        this.v = v;
        this.pos = pos;
    }
    return Event;
}());
function compareEvents(a, b) {
    if (a.pos > b.pos) {
        return 1;
    }
    if (a.pos < b.pos) {
        return -1;
    }
    if (a.isOpen) {
        return -1;
    }
    if (b.isOpen) {
        return 1;
    }
    return 0;
}
function makeRBTree() {
    return new rbtree_1.RBTree(function (a, b) { return a.pos - b.pos; });
}
var xRect = {
    getCentre: function (r) { return r.cx(); },
    getOpen: function (r) { return r.y; },
    getClose: function (r) { return r.Y; },
    getSize: function (r) { return r.width(); },
    makeRect: function (open, close, center, size) { return new Rectangle(center - size / 2, center + size / 2, open, close); },
    findNeighbours: findXNeighbours
};
var yRect = {
    getCentre: function (r) { return r.cy(); },
    getOpen: function (r) { return r.x; },
    getClose: function (r) { return r.X; },
    getSize: function (r) { return r.height(); },
    makeRect: function (open, close, center, size) { return new Rectangle(open, close, center - size / 2, center + size / 2); },
    findNeighbours: findYNeighbours
};
function generateGroupConstraints(root, f, minSep, isContained) {
    if (isContained === void 0) { isContained = false; }
    var padding = root.padding, gn = typeof root.groups !== 'undefined' ? root.groups.length : 0, ln = typeof root.leaves !== 'undefined' ? root.leaves.length : 0, childConstraints = !gn ? []
        : root.groups.reduce(function (ccs, g) { return ccs.concat(generateGroupConstraints(g, f, minSep, true)); }, []), n = (isContained ? 2 : 0) + ln + gn, vs = new Array(n), rs = new Array(n), i = 0, add = function (r, v) { rs[i] = r; vs[i++] = v; };
    if (isContained) {
        var b = root.bounds, c = f.getCentre(b), s = f.getSize(b) / 2, open = f.getOpen(b), close = f.getClose(b), min = c - s + padding / 2, max = c + s - padding / 2;
        root.minVar.desiredPosition = min;
        add(f.makeRect(open, close, min, padding), root.minVar);
        root.maxVar.desiredPosition = max;
        add(f.makeRect(open, close, max, padding), root.maxVar);
    }
    if (ln)
        root.leaves.forEach(function (l) { return add(l.bounds, l.variable); });
    if (gn)
        root.groups.forEach(function (g) {
            var b = g.bounds;
            add(f.makeRect(f.getOpen(b), f.getClose(b), f.getCentre(b), f.getSize(b)), g.minVar);
        });
    var cs = generateConstraints(rs, vs, f, minSep);
    if (gn) {
        vs.forEach(function (v) { v.cOut = [], v.cIn = []; });
        cs.forEach(function (c) { c.left.cOut.push(c), c.right.cIn.push(c); });
        root.groups.forEach(function (g) {
            var gapAdjustment = (g.padding - f.getSize(g.bounds)) / 2;
            g.minVar.cIn.forEach(function (c) { return c.gap += gapAdjustment; });
            g.minVar.cOut.forEach(function (c) { c.left = g.maxVar; c.gap += gapAdjustment; });
        });
    }
    return childConstraints.concat(cs);
}
function generateConstraints(rs, vars, rect, minSep) {
    var i, n = rs.length;
    var N = 2 * n;
    console.assert(vars.length >= n);
    var events = new Array(N);
    for (i = 0; i < n; ++i) {
        var r = rs[i];
        var v = new Node(vars[i], r, rect.getCentre(r));
        events[i] = new Event(true, v, rect.getOpen(r));
        events[i + n] = new Event(false, v, rect.getClose(r));
    }
    events.sort(compareEvents);
    var cs = new Array();
    var scanline = makeRBTree();
    for (i = 0; i < N; ++i) {
        var e = events[i];
        var v = e.v;
        if (e.isOpen) {
            scanline.insert(v);
            rect.findNeighbours(v, scanline);
        }
        else {
            scanline.remove(v);
            var makeConstraint = function (l, r) {
                var sep = (rect.getSize(l.r) + rect.getSize(r.r)) / 2 + minSep;
                cs.push(new vpsc_1.Constraint(l.v, r.v, sep));
            };
            var visitNeighbours = function (forward, reverse, mkcon) {
                var u, it = v[forward].iterator();
                while ((u = it[forward]()) !== null) {
                    mkcon(u, v);
                    u[reverse].remove(v);
                }
            };
            visitNeighbours("prev", "next", function (u, v) { return makeConstraint(u, v); });
            visitNeighbours("next", "prev", function (u, v) { return makeConstraint(v, u); });
        }
    }
    console.assert(scanline.size === 0);
    return cs;
}
function findXNeighbours(v, scanline) {
    var f = function (forward, reverse) {
        var it = scanline.findIter(v);
        var u;
        while ((u = it[forward]()) !== null) {
            var uovervX = u.r.overlapX(v.r);
            if (uovervX <= 0 || uovervX <= u.r.overlapY(v.r)) {
                v[forward].insert(u);
                u[reverse].insert(v);
            }
            if (uovervX <= 0) {
                break;
            }
        }
    };
    f("next", "prev");
    f("prev", "next");
}
function findYNeighbours(v, scanline) {
    var f = function (forward, reverse) {
        var u = scanline.findIter(v)[forward]();
        if (u !== null && u.r.overlapX(v.r) > 0) {
            v[forward].insert(u);
            u[reverse].insert(v);
        }
    };
    f("next", "prev");
    f("prev", "next");
}
function generateXConstraints(rs, vars) {
    return generateConstraints(rs, vars, xRect, 1e-6);
}
exports.generateXConstraints = generateXConstraints;
function generateYConstraints(rs, vars) {
    return generateConstraints(rs, vars, yRect, 1e-6);
}
exports.generateYConstraints = generateYConstraints;
function generateXGroupConstraints(root) {
    return generateGroupConstraints(root, xRect, 1e-6);
}
exports.generateXGroupConstraints = generateXGroupConstraints;
function generateYGroupConstraints(root) {
    return generateGroupConstraints(root, yRect, 1e-6);
}
exports.generateYGroupConstraints = generateYGroupConstraints;
function removeOverlaps(rs) {
    var vs = rs.map(function (r) { return new vpsc_1.Variable(r.cx()); });
    var cs = generateXConstraints(rs, vs);
    var solver = new vpsc_1.Solver(vs, cs);
    solver.solve();
    vs.forEach(function (v, i) { return rs[i].setXCentre(v.position()); });
    vs = rs.map(function (r) { return new vpsc_1.Variable(r.cy()); });
    cs = generateYConstraints(rs, vs);
    solver = new vpsc_1.Solver(vs, cs);
    solver.solve();
    vs.forEach(function (v, i) { return rs[i].setYCentre(v.position()); });
}
exports.removeOverlaps = removeOverlaps;
var IndexedVariable = (function (_super) {
    __extends(IndexedVariable, _super);
    function IndexedVariable(index, w) {
        var _this = _super.call(this, 0, w) || this;
        _this.index = index;
        return _this;
    }
    return IndexedVariable;
}(vpsc_1.Variable));
exports.IndexedVariable = IndexedVariable;
var Projection = (function () {
    function Projection(nodes, groups, rootGroup, constraints, avoidOverlaps) {
        var _this = this;
        if (rootGroup === void 0) { rootGroup = null; }
        if (constraints === void 0) { constraints = null; }
        if (avoidOverlaps === void 0) { avoidOverlaps = false; }
        this.nodes = nodes;
        this.groups = groups;
        this.rootGroup = rootGroup;
        this.avoidOverlaps = avoidOverlaps;
        this.variables = nodes.map(function (v, i) {
            return v.variable = new IndexedVariable(i, 1);
        });
        if (constraints)
            this.createConstraints(constraints);
        if (avoidOverlaps && rootGroup && typeof rootGroup.groups !== 'undefined') {
            nodes.forEach(function (v) {
                if (!v.width || !v.height) {
                    v.bounds = new Rectangle(v.x, v.x, v.y, v.y);
                    return;
                }
                var w2 = v.width / 2, h2 = v.height / 2;
                v.bounds = new Rectangle(v.x - w2, v.x + w2, v.y - h2, v.y + h2);
            });
            computeGroupBounds(rootGroup);
            var i = nodes.length;
            groups.forEach(function (g) {
                _this.variables[i] = g.minVar = new IndexedVariable(i++, typeof g.stiffness !== "undefined" ? g.stiffness : 0.01);
                _this.variables[i] = g.maxVar = new IndexedVariable(i++, typeof g.stiffness !== "undefined" ? g.stiffness : 0.01);
            });
        }
    }
    Projection.prototype.createSeparation = function (c) {
        return new vpsc_1.Constraint(this.nodes[c.left].variable, this.nodes[c.right].variable, c.gap, typeof c.equality !== "undefined" ? c.equality : false);
    };
    Projection.prototype.makeFeasible = function (c) {
        var _this = this;
        if (!this.avoidOverlaps)
            return;
        var axis = 'x', dim = 'width';
        if (c.axis === 'x')
            axis = 'y', dim = 'height';
        var vs = c.offsets.map(function (o) { return _this.nodes[o.node]; }).sort(function (a, b) { return a[axis] - b[axis]; });
        var p = null;
        vs.forEach(function (v) {
            if (p) {
                var nextPos = p[axis] + p[dim];
                if (nextPos > v[axis]) {
                    v[axis] = nextPos;
                }
            }
            p = v;
        });
    };
    Projection.prototype.createAlignment = function (c) {
        var _this = this;
        var u = this.nodes[c.offsets[0].node].variable;
        this.makeFeasible(c);
        var cs = c.axis === 'x' ? this.xConstraints : this.yConstraints;
        c.offsets.slice(1).forEach(function (o) {
            var v = _this.nodes[o.node].variable;
            cs.push(new vpsc_1.Constraint(u, v, o.offset, true));
        });
    };
    Projection.prototype.createConstraints = function (constraints) {
        var _this = this;
        var isSep = function (c) { return typeof c.type === 'undefined' || c.type === 'separation'; };
        this.xConstraints = constraints
            .filter(function (c) { return c.axis === "x" && isSep(c); })
            .map(function (c) { return _this.createSeparation(c); });
        this.yConstraints = constraints
            .filter(function (c) { return c.axis === "y" && isSep(c); })
            .map(function (c) { return _this.createSeparation(c); });
        constraints
            .filter(function (c) { return c.type === 'alignment'; })
            .forEach(function (c) { return _this.createAlignment(c); });
    };
    Projection.prototype.setupVariablesAndBounds = function (x0, y0, desired, getDesired) {
        this.nodes.forEach(function (v, i) {
            if (v.fixed) {
                v.variable.weight = v.fixedWeight ? v.fixedWeight : 1000;
                desired[i] = getDesired(v);
            }
            else {
                v.variable.weight = 1;
            }
            var w = (v.width || 0) / 2, h = (v.height || 0) / 2;
            var ix = x0[i], iy = y0[i];
            v.bounds = new Rectangle(ix - w, ix + w, iy - h, iy + h);
        });
    };
    Projection.prototype.xProject = function (x0, y0, x) {
        if (!this.rootGroup && !(this.avoidOverlaps || this.xConstraints))
            return;
        this.project(x0, y0, x0, x, function (v) { return v.px; }, this.xConstraints, generateXGroupConstraints, function (v) { return v.bounds.setXCentre(x[v.variable.index] = v.variable.position()); }, function (g) {
            var xmin = x[g.minVar.index] = g.minVar.position();
            var xmax = x[g.maxVar.index] = g.maxVar.position();
            var p2 = g.padding / 2;
            g.bounds.x = xmin - p2;
            g.bounds.X = xmax + p2;
        });
    };
    Projection.prototype.yProject = function (x0, y0, y) {
        if (!this.rootGroup && !this.yConstraints)
            return;
        this.project(x0, y0, y0, y, function (v) { return v.py; }, this.yConstraints, generateYGroupConstraints, function (v) { return v.bounds.setYCentre(y[v.variable.index] = v.variable.position()); }, function (g) {
            var ymin = y[g.minVar.index] = g.minVar.position();
            var ymax = y[g.maxVar.index] = g.maxVar.position();
            var p2 = g.padding / 2;
            g.bounds.y = ymin - p2;
            ;
            g.bounds.Y = ymax + p2;
        });
    };
    Projection.prototype.projectFunctions = function () {
        var _this = this;
        return [
            function (x0, y0, x) { return _this.xProject(x0, y0, x); },
            function (x0, y0, y) { return _this.yProject(x0, y0, y); }
        ];
    };
    Projection.prototype.project = function (x0, y0, start, desired, getDesired, cs, generateConstraints, updateNodeBounds, updateGroupBounds) {
        this.setupVariablesAndBounds(x0, y0, desired, getDesired);
        if (this.rootGroup && this.avoidOverlaps) {
            computeGroupBounds(this.rootGroup);
            cs = cs.concat(generateConstraints(this.rootGroup));
        }
        this.solve(this.variables, cs, start, desired);
        this.nodes.forEach(updateNodeBounds);
        if (this.rootGroup && this.avoidOverlaps) {
            this.groups.forEach(updateGroupBounds);
            computeGroupBounds(this.rootGroup);
        }
    };
    Projection.prototype.solve = function (vs, cs, starting, desired) {
        var solver = new vpsc_1.Solver(vs, cs);
        solver.setStartingPositions(starting);
        solver.setDesiredPositions(desired);
        solver.solve();
    };
    return Projection;
}());
exports.Projection = Projection;
},{"./rbtree":16,"./vpsc":19}],18:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var pqueue_1 = require("./pqueue");
var Neighbour = (function () {
    function Neighbour(id, distance) {
        this.id = id;
        this.distance = distance;
    }
    return Neighbour;
}());
var Node = (function () {
    function Node(id) {
        this.id = id;
        this.neighbours = [];
    }
    return Node;
}());
var QueueEntry = (function () {
    function QueueEntry(node, prev, d) {
        this.node = node;
        this.prev = prev;
        this.d = d;
    }
    return QueueEntry;
}());
var Calculator = (function () {
    function Calculator(n, es, getSourceIndex, getTargetIndex, getLength) {
        this.n = n;
        this.es = es;
        this.neighbours = new Array(this.n);
        var i = this.n;
        while (i--)
            this.neighbours[i] = new Node(i);
        i = this.es.length;
        while (i--) {
            var e = this.es[i];
            var u = getSourceIndex(e), v = getTargetIndex(e);
            var d = getLength(e);
            this.neighbours[u].neighbours.push(new Neighbour(v, d));
            this.neighbours[v].neighbours.push(new Neighbour(u, d));
        }
    }
    Calculator.prototype.DistanceMatrix = function () {
        var D = new Array(this.n);
        for (var i = 0; i < this.n; ++i) {
            D[i] = this.dijkstraNeighbours(i);
        }
        return D;
    };
    Calculator.prototype.DistancesFromNode = function (start) {
        return this.dijkstraNeighbours(start);
    };
    Calculator.prototype.PathFromNodeToNode = function (start, end) {
        return this.dijkstraNeighbours(start, end);
    };
    Calculator.prototype.PathFromNodeToNodeWithPrevCost = function (start, end, prevCost) {
        var q = new pqueue_1.PriorityQueue(function (a, b) { return a.d <= b.d; }), u = this.neighbours[start], qu = new QueueEntry(u, null, 0), visitedFrom = {};
        q.push(qu);
        while (!q.empty()) {
            qu = q.pop();
            u = qu.node;
            if (u.id === end) {
                break;
            }
            var i = u.neighbours.length;
            while (i--) {
                var neighbour = u.neighbours[i], v = this.neighbours[neighbour.id];
                if (qu.prev && v.id === qu.prev.node.id)
                    continue;
                var viduid = v.id + ',' + u.id;
                if (viduid in visitedFrom && visitedFrom[viduid] <= qu.d)
                    continue;
                var cc = qu.prev ? prevCost(qu.prev.node.id, u.id, v.id) : 0, t = qu.d + neighbour.distance + cc;
                visitedFrom[viduid] = t;
                q.push(new QueueEntry(v, qu, t));
            }
        }
        var path = [];
        while (qu.prev) {
            qu = qu.prev;
            path.push(qu.node.id);
        }
        return path;
    };
    Calculator.prototype.dijkstraNeighbours = function (start, dest) {
        if (dest === void 0) { dest = -1; }
        var q = new pqueue_1.PriorityQueue(function (a, b) { return a.d <= b.d; }), i = this.neighbours.length, d = new Array(i);
        while (i--) {
            var node = this.neighbours[i];
            node.d = i === start ? 0 : Number.POSITIVE_INFINITY;
            node.q = q.push(node);
        }
        while (!q.empty()) {
            var u = q.pop();
            d[u.id] = u.d;
            if (u.id === dest) {
                var path = [];
                var v = u;
                while (typeof v.prev !== 'undefined') {
                    path.push(v.prev.id);
                    v = v.prev;
                }
                return path;
            }
            i = u.neighbours.length;
            while (i--) {
                var neighbour = u.neighbours[i];
                var v = this.neighbours[neighbour.id];
                var t = u.d + neighbour.distance;
                if (u.d !== Number.MAX_VALUE && v.d > t) {
                    v.d = t;
                    v.prev = u;
                    q.reduceKey(v.q, v, function (e, q) { return e.q = q; });
                }
            }
        }
        return d;
    };
    return Calculator;
}());
exports.Calculator = Calculator;
},{"./pqueue":15}],19:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var PositionStats = (function () {
    function PositionStats(scale) {
        this.scale = scale;
        this.AB = 0;
        this.AD = 0;
        this.A2 = 0;
    }
    PositionStats.prototype.addVariable = function (v) {
        var ai = this.scale / v.scale;
        var bi = v.offset / v.scale;
        var wi = v.weight;
        this.AB += wi * ai * bi;
        this.AD += wi * ai * v.desiredPosition;
        this.A2 += wi * ai * ai;
    };
    PositionStats.prototype.getPosn = function () {
        return (this.AD - this.AB) / this.A2;
    };
    return PositionStats;
}());
exports.PositionStats = PositionStats;
var Constraint = (function () {
    function Constraint(left, right, gap, equality) {
        if (equality === void 0) { equality = false; }
        this.left = left;
        this.right = right;
        this.gap = gap;
        this.equality = equality;
        this.active = false;
        this.unsatisfiable = false;
        this.left = left;
        this.right = right;
        this.gap = gap;
        this.equality = equality;
    }
    Constraint.prototype.slack = function () {
        return this.unsatisfiable ? Number.MAX_VALUE
            : this.right.scale * this.right.position() - this.gap
                - this.left.scale * this.left.position();
    };
    return Constraint;
}());
exports.Constraint = Constraint;
var Variable = (function () {
    function Variable(desiredPosition, weight, scale) {
        if (weight === void 0) { weight = 1; }
        if (scale === void 0) { scale = 1; }
        this.desiredPosition = desiredPosition;
        this.weight = weight;
        this.scale = scale;
        this.offset = 0;
    }
    Variable.prototype.dfdv = function () {
        return 2.0 * this.weight * (this.position() - this.desiredPosition);
    };
    Variable.prototype.position = function () {
        return (this.block.ps.scale * this.block.posn + this.offset) / this.scale;
    };
    Variable.prototype.visitNeighbours = function (prev, f) {
        var ff = function (c, next) { return c.active && prev !== next && f(c, next); };
        this.cOut.forEach(function (c) { return ff(c, c.right); });
        this.cIn.forEach(function (c) { return ff(c, c.left); });
    };
    return Variable;
}());
exports.Variable = Variable;
var Block = (function () {
    function Block(v) {
        this.vars = [];
        v.offset = 0;
        this.ps = new PositionStats(v.scale);
        this.addVariable(v);
    }
    Block.prototype.addVariable = function (v) {
        v.block = this;
        this.vars.push(v);
        this.ps.addVariable(v);
        this.posn = this.ps.getPosn();
    };
    Block.prototype.updateWeightedPosition = function () {
        this.ps.AB = this.ps.AD = this.ps.A2 = 0;
        for (var i = 0, n = this.vars.length; i < n; ++i)
            this.ps.addVariable(this.vars[i]);
        this.posn = this.ps.getPosn();
    };
    Block.prototype.compute_lm = function (v, u, postAction) {
        var _this = this;
        var dfdv = v.dfdv();
        v.visitNeighbours(u, function (c, next) {
            var _dfdv = _this.compute_lm(next, v, postAction);
            if (next === c.right) {
                dfdv += _dfdv * c.left.scale;
                c.lm = _dfdv;
            }
            else {
                dfdv += _dfdv * c.right.scale;
                c.lm = -_dfdv;
            }
            postAction(c);
        });
        return dfdv / v.scale;
    };
    Block.prototype.populateSplitBlock = function (v, prev) {
        var _this = this;
        v.visitNeighbours(prev, function (c, next) {
            next.offset = v.offset + (next === c.right ? c.gap : -c.gap);
            _this.addVariable(next);
            _this.populateSplitBlock(next, v);
        });
    };
    Block.prototype.traverse = function (visit, acc, v, prev) {
        var _this = this;
        if (v === void 0) { v = this.vars[0]; }
        if (prev === void 0) { prev = null; }
        v.visitNeighbours(prev, function (c, next) {
            acc.push(visit(c));
            _this.traverse(visit, acc, next, v);
        });
    };
    Block.prototype.findMinLM = function () {
        var m = null;
        this.compute_lm(this.vars[0], null, function (c) {
            if (!c.equality && (m === null || c.lm < m.lm))
                m = c;
        });
        return m;
    };
    Block.prototype.findMinLMBetween = function (lv, rv) {
        this.compute_lm(lv, null, function () { });
        var m = null;
        this.findPath(lv, null, rv, function (c, next) {
            if (!c.equality && c.right === next && (m === null || c.lm < m.lm))
                m = c;
        });
        return m;
    };
    Block.prototype.findPath = function (v, prev, to, visit) {
        var _this = this;
        var endFound = false;
        v.visitNeighbours(prev, function (c, next) {
            if (!endFound && (next === to || _this.findPath(next, v, to, visit))) {
                endFound = true;
                visit(c, next);
            }
        });
        return endFound;
    };
    Block.prototype.isActiveDirectedPathBetween = function (u, v) {
        if (u === v)
            return true;
        var i = u.cOut.length;
        while (i--) {
            var c = u.cOut[i];
            if (c.active && this.isActiveDirectedPathBetween(c.right, v))
                return true;
        }
        return false;
    };
    Block.split = function (c) {
        c.active = false;
        return [Block.createSplitBlock(c.left), Block.createSplitBlock(c.right)];
    };
    Block.createSplitBlock = function (startVar) {
        var b = new Block(startVar);
        b.populateSplitBlock(startVar, null);
        return b;
    };
    Block.prototype.splitBetween = function (vl, vr) {
        var c = this.findMinLMBetween(vl, vr);
        if (c !== null) {
            var bs = Block.split(c);
            return { constraint: c, lb: bs[0], rb: bs[1] };
        }
        return null;
    };
    Block.prototype.mergeAcross = function (b, c, dist) {
        c.active = true;
        for (var i = 0, n = b.vars.length; i < n; ++i) {
            var v = b.vars[i];
            v.offset += dist;
            this.addVariable(v);
        }
        this.posn = this.ps.getPosn();
    };
    Block.prototype.cost = function () {
        var sum = 0, i = this.vars.length;
        while (i--) {
            var v = this.vars[i], d = v.position() - v.desiredPosition;
            sum += d * d * v.weight;
        }
        return sum;
    };
    return Block;
}());
exports.Block = Block;
var Blocks = (function () {
    function Blocks(vs) {
        this.vs = vs;
        var n = vs.length;
        this.list = new Array(n);
        while (n--) {
            var b = new Block(vs[n]);
            this.list[n] = b;
            b.blockInd = n;
        }
    }
    Blocks.prototype.cost = function () {
        var sum = 0, i = this.list.length;
        while (i--)
            sum += this.list[i].cost();
        return sum;
    };
    Blocks.prototype.insert = function (b) {
        b.blockInd = this.list.length;
        this.list.push(b);
    };
    Blocks.prototype.remove = function (b) {
        var last = this.list.length - 1;
        var swapBlock = this.list[last];
        this.list.length = last;
        if (b !== swapBlock) {
            this.list[b.blockInd] = swapBlock;
            swapBlock.blockInd = b.blockInd;
        }
    };
    Blocks.prototype.merge = function (c) {
        var l = c.left.block, r = c.right.block;
        var dist = c.right.offset - c.left.offset - c.gap;
        if (l.vars.length < r.vars.length) {
            r.mergeAcross(l, c, dist);
            this.remove(l);
        }
        else {
            l.mergeAcross(r, c, -dist);
            this.remove(r);
        }
    };
    Blocks.prototype.forEach = function (f) {
        this.list.forEach(f);
    };
    Blocks.prototype.updateBlockPositions = function () {
        this.list.forEach(function (b) { return b.updateWeightedPosition(); });
    };
    Blocks.prototype.split = function (inactive) {
        var _this = this;
        this.updateBlockPositions();
        this.list.forEach(function (b) {
            var v = b.findMinLM();
            if (v !== null && v.lm < Solver.LAGRANGIAN_TOLERANCE) {
                b = v.left.block;
                Block.split(v).forEach(function (nb) { return _this.insert(nb); });
                _this.remove(b);
                inactive.push(v);
            }
        });
    };
    return Blocks;
}());
exports.Blocks = Blocks;
var Solver = (function () {
    function Solver(vs, cs) {
        this.vs = vs;
        this.cs = cs;
        this.vs = vs;
        vs.forEach(function (v) {
            v.cIn = [], v.cOut = [];
        });
        this.cs = cs;
        cs.forEach(function (c) {
            c.left.cOut.push(c);
            c.right.cIn.push(c);
        });
        this.inactive = cs.map(function (c) { c.active = false; return c; });
        this.bs = null;
    }
    Solver.prototype.cost = function () {
        return this.bs.cost();
    };
    Solver.prototype.setStartingPositions = function (ps) {
        this.inactive = this.cs.map(function (c) { c.active = false; return c; });
        this.bs = new Blocks(this.vs);
        this.bs.forEach(function (b, i) { return b.posn = ps[i]; });
    };
    Solver.prototype.setDesiredPositions = function (ps) {
        this.vs.forEach(function (v, i) { return v.desiredPosition = ps[i]; });
    };
    Solver.prototype.mostViolated = function () {
        var minSlack = Number.MAX_VALUE, v = null, l = this.inactive, n = l.length, deletePoint = n;
        for (var i = 0; i < n; ++i) {
            var c = l[i];
            if (c.unsatisfiable)
                continue;
            var slack = c.slack();
            if (c.equality || slack < minSlack) {
                minSlack = slack;
                v = c;
                deletePoint = i;
                if (c.equality)
                    break;
            }
        }
        if (deletePoint !== n &&
            (minSlack < Solver.ZERO_UPPERBOUND && !v.active || v.equality)) {
            l[deletePoint] = l[n - 1];
            l.length = n - 1;
        }
        return v;
    };
    Solver.prototype.satisfy = function () {
        if (this.bs == null) {
            this.bs = new Blocks(this.vs);
        }
        this.bs.split(this.inactive);
        var v = null;
        while ((v = this.mostViolated()) && (v.equality || v.slack() < Solver.ZERO_UPPERBOUND && !v.active)) {
            var lb = v.left.block, rb = v.right.block;
            if (lb !== rb) {
                this.bs.merge(v);
            }
            else {
                if (lb.isActiveDirectedPathBetween(v.right, v.left)) {
                    v.unsatisfiable = true;
                    continue;
                }
                var split = lb.splitBetween(v.left, v.right);
                if (split !== null) {
                    this.bs.insert(split.lb);
                    this.bs.insert(split.rb);
                    this.bs.remove(lb);
                    this.inactive.push(split.constraint);
                }
                else {
                    v.unsatisfiable = true;
                    continue;
                }
                if (v.slack() >= 0) {
                    this.inactive.push(v);
                }
                else {
                    this.bs.merge(v);
                }
            }
        }
    };
    Solver.prototype.solve = function () {
        this.satisfy();
        var lastcost = Number.MAX_VALUE, cost = this.bs.cost();
        while (Math.abs(lastcost - cost) > 0.0001) {
            this.satisfy();
            lastcost = cost;
            cost = this.bs.cost();
        }
        return cost;
    };
    Solver.LAGRANGIAN_TOLERANCE = -1e-4;
    Solver.ZERO_UPPERBOUND = -1e-10;
    return Solver;
}());
exports.Solver = Solver;
function removeOverlapInOneDimension(spans, lowerBound, upperBound) {
    var vs = spans.map(function (s) { return new Variable(s.desiredCenter); });
    var cs = [];
    var n = spans.length;
    for (var i = 0; i < n - 1; i++) {
        var left = spans[i], right = spans[i + 1];
        cs.push(new Constraint(vs[i], vs[i + 1], (left.size + right.size) / 2));
    }
    var leftMost = vs[0], rightMost = vs[n - 1], leftMostSize = spans[0].size / 2, rightMostSize = spans[n - 1].size / 2;
    var vLower = null, vUpper = null;
    if (lowerBound) {
        vLower = new Variable(lowerBound, leftMost.weight * 1000);
        vs.push(vLower);
        cs.push(new Constraint(vLower, leftMost, leftMostSize));
    }
    if (upperBound) {
        vUpper = new Variable(upperBound, rightMost.weight * 1000);
        vs.push(vUpper);
        cs.push(new Constraint(rightMost, vUpper, rightMostSize));
    }
    var solver = new Solver(vs, cs);
    solver.solve();
    return {
        newCenters: vs.slice(0, spans.length).map(function (v) { return v.position(); }),
        lowerBound: vLower ? vLower.position() : leftMost.position() - leftMostSize,
        upperBound: vUpper ? vUpper.position() : rightMost.position() + rightMostSize
    };
}
exports.removeOverlapInOneDimension = removeOverlapInOneDimension;
},{}]},{},[1])(1)
});

//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIm5vZGVfbW9kdWxlcy9icm93c2VyLXBhY2svX3ByZWx1ZGUuanMiLCJkaXN0L2luZGV4LmpzIiwiZGlzdC9zcmMvYWRhcHRvci5qcyIsImRpc3Qvc3JjL2JhdGNoLmpzIiwiZGlzdC9zcmMvZDNhZGFwdG9yLmpzIiwiZGlzdC9zcmMvZDN2M2FkYXB0b3IuanMiLCJkaXN0L3NyYy9kM3Y0YWRhcHRvci5qcyIsImRpc3Qvc3JjL2Rlc2NlbnQuanMiLCJkaXN0L3NyYy9nZW9tLmpzIiwiZGlzdC9zcmMvZ3JpZHJvdXRlci5qcyIsImRpc3Qvc3JjL2hhbmRsZWRpc2Nvbm5lY3RlZC5qcyIsImRpc3Qvc3JjL2xheW91dC5qcyIsImRpc3Qvc3JjL2xheW91dDNkLmpzIiwiZGlzdC9zcmMvbGlua2xlbmd0aHMuanMiLCJkaXN0L3NyYy9wb3dlcmdyYXBoLmpzIiwiZGlzdC9zcmMvcHF1ZXVlLmpzIiwiZGlzdC9zcmMvcmJ0cmVlLmpzIiwiZGlzdC9zcmMvcmVjdGFuZ2xlLmpzIiwiZGlzdC9zcmMvc2hvcnRlc3RwYXRocy5qcyIsImRpc3Qvc3JjL3Zwc2MuanMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7QUNBQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQ3JCQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUNyREE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQ3hGQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FDaEJBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQ2pFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FDN0RBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQ3hWQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQ3ZhQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQ3JpQkE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQzFNQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FDNWlCQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQ2xIQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUN2SEE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQzVUQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FDaEtBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQ3JZQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUNuZEE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQ3pIQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSIsImZpbGUiOiJnZW5lcmF0ZWQuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlc0NvbnRlbnQiOlsiKGZ1bmN0aW9uKCl7ZnVuY3Rpb24gcihlLG4sdCl7ZnVuY3Rpb24gbyhpLGYpe2lmKCFuW2ldKXtpZighZVtpXSl7dmFyIGM9XCJmdW5jdGlvblwiPT10eXBlb2YgcmVxdWlyZSYmcmVxdWlyZTtpZighZiYmYylyZXR1cm4gYyhpLCEwKTtpZih1KXJldHVybiB1KGksITApO3ZhciBhPW5ldyBFcnJvcihcIkNhbm5vdCBmaW5kIG1vZHVsZSAnXCIraStcIidcIik7dGhyb3cgYS5jb2RlPVwiTU9EVUxFX05PVF9GT1VORFwiLGF9dmFyIHA9bltpXT17ZXhwb3J0czp7fX07ZVtpXVswXS5jYWxsKHAuZXhwb3J0cyxmdW5jdGlvbihyKXt2YXIgbj1lW2ldWzFdW3JdO3JldHVybiBvKG58fHIpfSxwLHAuZXhwb3J0cyxyLGUsbix0KX1yZXR1cm4gbltpXS5leHBvcnRzfWZvcih2YXIgdT1cImZ1bmN0aW9uXCI9PXR5cGVvZiByZXF1aXJlJiZyZXF1aXJlLGk9MDtpPHQubGVuZ3RoO2krKylvKHRbaV0pO3JldHVybiBvfXJldHVybiByfSkoKSIsIlwidXNlIHN0cmljdFwiO1xyXG5mdW5jdGlvbiBfX2V4cG9ydChtKSB7XHJcbiAgICBmb3IgKHZhciBwIGluIG0pIGlmICghZXhwb3J0cy5oYXNPd25Qcm9wZXJ0eShwKSkgZXhwb3J0c1twXSA9IG1bcF07XHJcbn1cclxuT2JqZWN0LmRlZmluZVByb3BlcnR5KGV4cG9ydHMsIFwiX19lc01vZHVsZVwiLCB7IHZhbHVlOiB0cnVlIH0pO1xyXG5fX2V4cG9ydChyZXF1aXJlKFwiLi9zcmMvYWRhcHRvclwiKSk7XHJcbl9fZXhwb3J0KHJlcXVpcmUoXCIuL3NyYy9kM2FkYXB0b3JcIikpO1xyXG5fX2V4cG9ydChyZXF1aXJlKFwiLi9zcmMvZGVzY2VudFwiKSk7XHJcbl9fZXhwb3J0KHJlcXVpcmUoXCIuL3NyYy9nZW9tXCIpKTtcclxuX19leHBvcnQocmVxdWlyZShcIi4vc3JjL2dyaWRyb3V0ZXJcIikpO1xyXG5fX2V4cG9ydChyZXF1aXJlKFwiLi9zcmMvaGFuZGxlZGlzY29ubmVjdGVkXCIpKTtcclxuX19leHBvcnQocmVxdWlyZShcIi4vc3JjL2xheW91dFwiKSk7XHJcbl9fZXhwb3J0KHJlcXVpcmUoXCIuL3NyYy9sYXlvdXQzZFwiKSk7XHJcbl9fZXhwb3J0KHJlcXVpcmUoXCIuL3NyYy9saW5rbGVuZ3Roc1wiKSk7XHJcbl9fZXhwb3J0KHJlcXVpcmUoXCIuL3NyYy9wb3dlcmdyYXBoXCIpKTtcclxuX19leHBvcnQocmVxdWlyZShcIi4vc3JjL3BxdWV1ZVwiKSk7XHJcbl9fZXhwb3J0KHJlcXVpcmUoXCIuL3NyYy9yYnRyZWVcIikpO1xyXG5fX2V4cG9ydChyZXF1aXJlKFwiLi9zcmMvcmVjdGFuZ2xlXCIpKTtcclxuX19leHBvcnQocmVxdWlyZShcIi4vc3JjL3Nob3J0ZXN0cGF0aHNcIikpO1xyXG5fX2V4cG9ydChyZXF1aXJlKFwiLi9zcmMvdnBzY1wiKSk7XHJcbl9fZXhwb3J0KHJlcXVpcmUoXCIuL3NyYy9iYXRjaFwiKSk7XHJcbi8vIyBzb3VyY2VNYXBwaW5nVVJMPWRhdGE6YXBwbGljYXRpb24vanNvbjtiYXNlNjQsZXlKMlpYSnphVzl1SWpvekxDSm1hV3hsSWpvaWFXNWtaWGd1YW5NaUxDSnpiM1Z5WTJWU2IyOTBJam9pSWl3aWMyOTFjbU5sY3lJNld5SXVMaTlYWldKRGIyeGhMMmx1WkdWNExuUnpJbDBzSW01aGJXVnpJanBiWFN3aWJXRndjR2x1WjNNaU9pSTdPenM3TzBGQlFVRXNiVU5CUVRaQ08wRkJRemRDTEhGRFFVRXJRanRCUVVNdlFpeHRRMEZCTmtJN1FVRkROMElzWjBOQlFUQkNPMEZCUXpGQ0xITkRRVUZuUXp0QlFVTm9ReXc0UTBGQmQwTTdRVUZEZUVNc2EwTkJRVFJDTzBGQlF6VkNMRzlEUVVFNFFqdEJRVU01UWl4MVEwRkJhVU03UVVGRGFrTXNjME5CUVdkRE8wRkJRMmhETEd0RFFVRTBRanRCUVVNMVFpeHJRMEZCTkVJN1FVRkROVUlzY1VOQlFTdENPMEZCUXk5Q0xIbERRVUZ0UXp0QlFVTnVReXhuUTBGQk1FSTdRVUZETVVJc2FVTkJRVEpDSW4wPSIsIlwidXNlIHN0cmljdFwiO1xyXG52YXIgX19leHRlbmRzID0gKHRoaXMgJiYgdGhpcy5fX2V4dGVuZHMpIHx8IChmdW5jdGlvbiAoKSB7XHJcbiAgICB2YXIgZXh0ZW5kU3RhdGljcyA9IGZ1bmN0aW9uIChkLCBiKSB7XHJcbiAgICAgICAgZXh0ZW5kU3RhdGljcyA9IE9iamVjdC5zZXRQcm90b3R5cGVPZiB8fFxyXG4gICAgICAgICAgICAoeyBfX3Byb3RvX186IFtdIH0gaW5zdGFuY2VvZiBBcnJheSAmJiBmdW5jdGlvbiAoZCwgYikgeyBkLl9fcHJvdG9fXyA9IGI7IH0pIHx8XHJcbiAgICAgICAgICAgIGZ1bmN0aW9uIChkLCBiKSB7IGZvciAodmFyIHAgaW4gYikgaWYgKGIuaGFzT3duUHJvcGVydHkocCkpIGRbcF0gPSBiW3BdOyB9O1xyXG4gICAgICAgIHJldHVybiBleHRlbmRTdGF0aWNzKGQsIGIpO1xyXG4gICAgfTtcclxuICAgIHJldHVybiBmdW5jdGlvbiAoZCwgYikge1xyXG4gICAgICAgIGV4dGVuZFN0YXRpY3MoZCwgYik7XHJcbiAgICAgICAgZnVuY3Rpb24gX18oKSB7IHRoaXMuY29uc3RydWN0b3IgPSBkOyB9XHJcbiAgICAgICAgZC5wcm90b3R5cGUgPSBiID09PSBudWxsID8gT2JqZWN0LmNyZWF0ZShiKSA6IChfXy5wcm90b3R5cGUgPSBiLnByb3RvdHlwZSwgbmV3IF9fKCkpO1xyXG4gICAgfTtcclxufSkoKTtcclxuT2JqZWN0LmRlZmluZVByb3BlcnR5KGV4cG9ydHMsIFwiX19lc01vZHVsZVwiLCB7IHZhbHVlOiB0cnVlIH0pO1xyXG52YXIgbGF5b3V0XzEgPSByZXF1aXJlKFwiLi9sYXlvdXRcIik7XHJcbnZhciBMYXlvdXRBZGFwdG9yID0gKGZ1bmN0aW9uIChfc3VwZXIpIHtcclxuICAgIF9fZXh0ZW5kcyhMYXlvdXRBZGFwdG9yLCBfc3VwZXIpO1xyXG4gICAgZnVuY3Rpb24gTGF5b3V0QWRhcHRvcihvcHRpb25zKSB7XHJcbiAgICAgICAgdmFyIF90aGlzID0gX3N1cGVyLmNhbGwodGhpcykgfHwgdGhpcztcclxuICAgICAgICB2YXIgc2VsZiA9IF90aGlzO1xyXG4gICAgICAgIHZhciBvID0gb3B0aW9ucztcclxuICAgICAgICBpZiAoby50cmlnZ2VyKSB7XHJcbiAgICAgICAgICAgIF90aGlzLnRyaWdnZXIgPSBvLnRyaWdnZXI7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIGlmIChvLmtpY2spIHtcclxuICAgICAgICAgICAgX3RoaXMua2ljayA9IG8ua2ljaztcclxuICAgICAgICB9XHJcbiAgICAgICAgaWYgKG8uZHJhZykge1xyXG4gICAgICAgICAgICBfdGhpcy5kcmFnID0gby5kcmFnO1xyXG4gICAgICAgIH1cclxuICAgICAgICBpZiAoby5vbikge1xyXG4gICAgICAgICAgICBfdGhpcy5vbiA9IG8ub247XHJcbiAgICAgICAgfVxyXG4gICAgICAgIF90aGlzLmRyYWdzdGFydCA9IF90aGlzLmRyYWdTdGFydCA9IGxheW91dF8xLkxheW91dC5kcmFnU3RhcnQ7XHJcbiAgICAgICAgX3RoaXMuZHJhZ2VuZCA9IF90aGlzLmRyYWdFbmQgPSBsYXlvdXRfMS5MYXlvdXQuZHJhZ0VuZDtcclxuICAgICAgICByZXR1cm4gX3RoaXM7XHJcbiAgICB9XHJcbiAgICBMYXlvdXRBZGFwdG9yLnByb3RvdHlwZS50cmlnZ2VyID0gZnVuY3Rpb24gKGUpIHsgfTtcclxuICAgIDtcclxuICAgIExheW91dEFkYXB0b3IucHJvdG90eXBlLmtpY2sgPSBmdW5jdGlvbiAoKSB7IH07XHJcbiAgICA7XHJcbiAgICBMYXlvdXRBZGFwdG9yLnByb3RvdHlwZS5kcmFnID0gZnVuY3Rpb24gKCkgeyB9O1xyXG4gICAgO1xyXG4gICAgTGF5b3V0QWRhcHRvci5wcm90b3R5cGUub24gPSBmdW5jdGlvbiAoZXZlbnRUeXBlLCBsaXN0ZW5lcikgeyByZXR1cm4gdGhpczsgfTtcclxuICAgIDtcclxuICAgIHJldHVybiBMYXlvdXRBZGFwdG9yO1xyXG59KGxheW91dF8xLkxheW91dCkpO1xyXG5leHBvcnRzLkxheW91dEFkYXB0b3IgPSBMYXlvdXRBZGFwdG9yO1xyXG5mdW5jdGlvbiBhZGFwdG9yKG9wdGlvbnMpIHtcclxuICAgIHJldHVybiBuZXcgTGF5b3V0QWRhcHRvcihvcHRpb25zKTtcclxufVxyXG5leHBvcnRzLmFkYXB0b3IgPSBhZGFwdG9yO1xyXG4vLyMgc291cmNlTWFwcGluZ1VSTD1kYXRhOmFwcGxpY2F0aW9uL2pzb247YmFzZTY0LGV5SjJaWEp6YVc5dUlqb3pMQ0ptYVd4bElqb2lZV1JoY0hSdmNpNXFjeUlzSW5OdmRYSmpaVkp2YjNRaU9pSWlMQ0p6YjNWeVkyVnpJanBiSWk0dUx5NHVMMWRsWWtOdmJHRXZjM0pqTDJGa1lYQjBiM0l1ZEhNaVhTd2libUZ0WlhNaU9sdGRMQ0p0WVhCd2FXNW5jeUk2SWpzN096czdPenM3T3pzN096czdPMEZCUVVFc2JVTkJRV2xFTzBGQlJUZERPMGxCUVcxRExHbERRVUZOTzBsQllYSkRMSFZDUVVGaExFOUJRVTg3VVVGQmNFSXNXVUZEU1N4cFFrRkJUeXhUUVhsQ1ZqdFJRWEpDUnl4SlFVRkpMRWxCUVVrc1IwRkJSeXhMUVVGSkxFTkJRVU03VVVGRGFFSXNTVUZCU1N4RFFVRkRMRWRCUVVjc1QwRkJUeXhEUVVGRE8xRkJSV2hDTEVsQlFVc3NRMEZCUXl4RFFVRkRMRTlCUVU4c1JVRkJSenRaUVVOaUxFdEJRVWtzUTBGQlF5eFBRVUZQTEVkQlFVY3NRMEZCUXl4RFFVRkRMRTlCUVU4c1EwRkJRenRUUVVNMVFqdFJRVVZFTEVsQlFVc3NRMEZCUXl4RFFVRkRMRWxCUVVrc1JVRkJSVHRaUVVOVUxFdEJRVWtzUTBGQlF5eEpRVUZKTEVkQlFVY3NRMEZCUXl4RFFVRkRMRWxCUVVrc1EwRkJRenRUUVVOMFFqdFJRVVZFTEVsQlFVc3NRMEZCUXl4RFFVRkRMRWxCUVVrc1JVRkJSVHRaUVVOVUxFdEJRVWtzUTBGQlF5eEpRVUZKTEVkQlFVY3NRMEZCUXl4RFFVRkRMRWxCUVVrc1EwRkJRenRUUVVOMFFqdFJRVVZFTEVsQlFVc3NRMEZCUXl4RFFVRkRMRVZCUVVVc1JVRkJSVHRaUVVOUUxFdEJRVWtzUTBGQlF5eEZRVUZGTEVkQlFVY3NRMEZCUXl4RFFVRkRMRVZCUVVVc1EwRkJRenRUUVVOc1FqdFJRVVZFTEV0QlFVa3NRMEZCUXl4VFFVRlRMRWRCUVVjc1MwRkJTU3hEUVVGRExGTkJRVk1zUjBGQlJ5eGxRVUZOTEVOQlFVTXNVMEZCVXl4RFFVRkRPMUZCUTI1RUxFdEJRVWtzUTBGQlF5eFBRVUZQTEVkQlFVY3NTMEZCU1N4RFFVRkRMRTlCUVU4c1IwRkJSeXhsUVVGTkxFTkJRVU1zVDBGQlR5eERRVUZET3p0SlFVTnFSQ3hEUVVGRE8wbEJjRU5FTEN0Q1FVRlBMRWRCUVZBc1ZVRkJVU3hEUVVGUkxFbEJRVWNzUTBGQlF6dEpRVUZCTEVOQlFVTTdTVUZEY2tJc05FSkJRVWtzUjBGQlNpeGpRVUZSTEVOQlFVTTdTVUZCUVN4RFFVRkRPMGxCUTFZc05FSkJRVWtzUjBGQlNpeGpRVUZSTEVOQlFVTTdTVUZCUVN4RFFVRkRPMGxCUTFZc01FSkJRVVVzUjBGQlJpeFZRVUZITEZOQlFUWkNMRVZCUVVVc1VVRkJiMElzU1VGQlZ5eFBRVUZQTEVsQlFVa3NRMEZCUXl4RFFVRkRMRU5CUVVNN1NVRkJRU3hEUVVGRE8wbEJhME53Uml4dlFrRkJRenRCUVVGRUxFTkJRVU1zUVVGNFEwUXNRMEZCYlVNc1pVRkJUU3hIUVhkRGVFTTdRVUY0UTFrc2MwTkJRV0U3UVVFMlF6RkNMRk5CUVdkQ0xFOUJRVThzUTBGQlJTeFBRVUZQTzBsQlF6VkNMRTlCUVU4c1NVRkJTU3hoUVVGaExFTkJRVVVzVDBGQlR5eERRVUZGTEVOQlFVTTdRVUZEZUVNc1EwRkJRenRCUVVaRUxEQkNRVVZESW4wPSIsIlwidXNlIHN0cmljdFwiO1xyXG5PYmplY3QuZGVmaW5lUHJvcGVydHkoZXhwb3J0cywgXCJfX2VzTW9kdWxlXCIsIHsgdmFsdWU6IHRydWUgfSk7XHJcbnZhciBsYXlvdXRfMSA9IHJlcXVpcmUoXCIuL2xheW91dFwiKTtcclxudmFyIGdyaWRyb3V0ZXJfMSA9IHJlcXVpcmUoXCIuL2dyaWRyb3V0ZXJcIik7XHJcbmZ1bmN0aW9uIGdyaWRpZnkocGdMYXlvdXQsIG51ZGdlR2FwLCBtYXJnaW4sIGdyb3VwTWFyZ2luKSB7XHJcbiAgICBwZ0xheW91dC5jb2xhLnN0YXJ0KDAsIDAsIDAsIDEwLCBmYWxzZSk7XHJcbiAgICB2YXIgZ3JpZHJvdXRlciA9IHJvdXRlKHBnTGF5b3V0LmNvbGEubm9kZXMoKSwgcGdMYXlvdXQuY29sYS5ncm91cHMoKSwgbWFyZ2luLCBncm91cE1hcmdpbik7XHJcbiAgICByZXR1cm4gZ3JpZHJvdXRlci5yb3V0ZUVkZ2VzKHBnTGF5b3V0LnBvd2VyR3JhcGgucG93ZXJFZGdlcywgbnVkZ2VHYXAsIGZ1bmN0aW9uIChlKSB7IHJldHVybiBlLnNvdXJjZS5yb3V0ZXJOb2RlLmlkOyB9LCBmdW5jdGlvbiAoZSkgeyByZXR1cm4gZS50YXJnZXQucm91dGVyTm9kZS5pZDsgfSk7XHJcbn1cclxuZXhwb3J0cy5ncmlkaWZ5ID0gZ3JpZGlmeTtcclxuZnVuY3Rpb24gcm91dGUobm9kZXMsIGdyb3VwcywgbWFyZ2luLCBncm91cE1hcmdpbikge1xyXG4gICAgbm9kZXMuZm9yRWFjaChmdW5jdGlvbiAoZCkge1xyXG4gICAgICAgIGQucm91dGVyTm9kZSA9IHtcclxuICAgICAgICAgICAgbmFtZTogZC5uYW1lLFxyXG4gICAgICAgICAgICBib3VuZHM6IGQuYm91bmRzLmluZmxhdGUoLW1hcmdpbilcclxuICAgICAgICB9O1xyXG4gICAgfSk7XHJcbiAgICBncm91cHMuZm9yRWFjaChmdW5jdGlvbiAoZCkge1xyXG4gICAgICAgIGQucm91dGVyTm9kZSA9IHtcclxuICAgICAgICAgICAgYm91bmRzOiBkLmJvdW5kcy5pbmZsYXRlKC1ncm91cE1hcmdpbiksXHJcbiAgICAgICAgICAgIGNoaWxkcmVuOiAodHlwZW9mIGQuZ3JvdXBzICE9PSAndW5kZWZpbmVkJyA/IGQuZ3JvdXBzLm1hcChmdW5jdGlvbiAoYykgeyByZXR1cm4gbm9kZXMubGVuZ3RoICsgYy5pZDsgfSkgOiBbXSlcclxuICAgICAgICAgICAgICAgIC5jb25jYXQodHlwZW9mIGQubGVhdmVzICE9PSAndW5kZWZpbmVkJyA/IGQubGVhdmVzLm1hcChmdW5jdGlvbiAoYykgeyByZXR1cm4gYy5pbmRleDsgfSkgOiBbXSlcclxuICAgICAgICB9O1xyXG4gICAgfSk7XHJcbiAgICB2YXIgZ3JpZFJvdXRlck5vZGVzID0gbm9kZXMuY29uY2F0KGdyb3VwcykubWFwKGZ1bmN0aW9uIChkLCBpKSB7XHJcbiAgICAgICAgZC5yb3V0ZXJOb2RlLmlkID0gaTtcclxuICAgICAgICByZXR1cm4gZC5yb3V0ZXJOb2RlO1xyXG4gICAgfSk7XHJcbiAgICByZXR1cm4gbmV3IGdyaWRyb3V0ZXJfMS5HcmlkUm91dGVyKGdyaWRSb3V0ZXJOb2Rlcywge1xyXG4gICAgICAgIGdldENoaWxkcmVuOiBmdW5jdGlvbiAodikgeyByZXR1cm4gdi5jaGlsZHJlbjsgfSxcclxuICAgICAgICBnZXRCb3VuZHM6IGZ1bmN0aW9uICh2KSB7IHJldHVybiB2LmJvdW5kczsgfVxyXG4gICAgfSwgbWFyZ2luIC0gZ3JvdXBNYXJnaW4pO1xyXG59XHJcbmZ1bmN0aW9uIHBvd2VyR3JhcGhHcmlkTGF5b3V0KGdyYXBoLCBzaXplLCBncm91cHBhZGRpbmcpIHtcclxuICAgIHZhciBwb3dlckdyYXBoO1xyXG4gICAgZ3JhcGgubm9kZXMuZm9yRWFjaChmdW5jdGlvbiAodiwgaSkgeyByZXR1cm4gdi5pbmRleCA9IGk7IH0pO1xyXG4gICAgbmV3IGxheW91dF8xLkxheW91dCgpXHJcbiAgICAgICAgLmF2b2lkT3ZlcmxhcHMoZmFsc2UpXHJcbiAgICAgICAgLm5vZGVzKGdyYXBoLm5vZGVzKVxyXG4gICAgICAgIC5saW5rcyhncmFwaC5saW5rcylcclxuICAgICAgICAucG93ZXJHcmFwaEdyb3VwcyhmdW5jdGlvbiAoZCkge1xyXG4gICAgICAgIHBvd2VyR3JhcGggPSBkO1xyXG4gICAgICAgIHBvd2VyR3JhcGguZ3JvdXBzLmZvckVhY2goZnVuY3Rpb24gKHYpIHsgcmV0dXJuIHYucGFkZGluZyA9IGdyb3VwcGFkZGluZzsgfSk7XHJcbiAgICB9KTtcclxuICAgIHZhciBuID0gZ3JhcGgubm9kZXMubGVuZ3RoO1xyXG4gICAgdmFyIGVkZ2VzID0gW107XHJcbiAgICB2YXIgdnMgPSBncmFwaC5ub2Rlcy5zbGljZSgwKTtcclxuICAgIHZzLmZvckVhY2goZnVuY3Rpb24gKHYsIGkpIHsgcmV0dXJuIHYuaW5kZXggPSBpOyB9KTtcclxuICAgIHBvd2VyR3JhcGguZ3JvdXBzLmZvckVhY2goZnVuY3Rpb24gKGcpIHtcclxuICAgICAgICB2YXIgc291cmNlSW5kID0gZy5pbmRleCA9IGcuaWQgKyBuO1xyXG4gICAgICAgIHZzLnB1c2goZyk7XHJcbiAgICAgICAgaWYgKHR5cGVvZiBnLmxlYXZlcyAhPT0gJ3VuZGVmaW5lZCcpXHJcbiAgICAgICAgICAgIGcubGVhdmVzLmZvckVhY2goZnVuY3Rpb24gKHYpIHsgcmV0dXJuIGVkZ2VzLnB1c2goeyBzb3VyY2U6IHNvdXJjZUluZCwgdGFyZ2V0OiB2LmluZGV4IH0pOyB9KTtcclxuICAgICAgICBpZiAodHlwZW9mIGcuZ3JvdXBzICE9PSAndW5kZWZpbmVkJylcclxuICAgICAgICAgICAgZy5ncm91cHMuZm9yRWFjaChmdW5jdGlvbiAoZ2cpIHsgcmV0dXJuIGVkZ2VzLnB1c2goeyBzb3VyY2U6IHNvdXJjZUluZCwgdGFyZ2V0OiBnZy5pZCArIG4gfSk7IH0pO1xyXG4gICAgfSk7XHJcbiAgICBwb3dlckdyYXBoLnBvd2VyRWRnZXMuZm9yRWFjaChmdW5jdGlvbiAoZSkge1xyXG4gICAgICAgIGVkZ2VzLnB1c2goeyBzb3VyY2U6IGUuc291cmNlLmluZGV4LCB0YXJnZXQ6IGUudGFyZ2V0LmluZGV4IH0pO1xyXG4gICAgfSk7XHJcbiAgICBuZXcgbGF5b3V0XzEuTGF5b3V0KClcclxuICAgICAgICAuc2l6ZShzaXplKVxyXG4gICAgICAgIC5ub2Rlcyh2cylcclxuICAgICAgICAubGlua3MoZWRnZXMpXHJcbiAgICAgICAgLmF2b2lkT3ZlcmxhcHMoZmFsc2UpXHJcbiAgICAgICAgLmxpbmtEaXN0YW5jZSgzMClcclxuICAgICAgICAuc3ltbWV0cmljRGlmZkxpbmtMZW5ndGhzKDUpXHJcbiAgICAgICAgLmNvbnZlcmdlbmNlVGhyZXNob2xkKDFlLTQpXHJcbiAgICAgICAgLnN0YXJ0KDEwMCwgMCwgMCwgMCwgZmFsc2UpO1xyXG4gICAgcmV0dXJuIHtcclxuICAgICAgICBjb2xhOiBuZXcgbGF5b3V0XzEuTGF5b3V0KClcclxuICAgICAgICAgICAgLmNvbnZlcmdlbmNlVGhyZXNob2xkKDFlLTMpXHJcbiAgICAgICAgICAgIC5zaXplKHNpemUpXHJcbiAgICAgICAgICAgIC5hdm9pZE92ZXJsYXBzKHRydWUpXHJcbiAgICAgICAgICAgIC5ub2RlcyhncmFwaC5ub2RlcylcclxuICAgICAgICAgICAgLmxpbmtzKGdyYXBoLmxpbmtzKVxyXG4gICAgICAgICAgICAuZ3JvdXBDb21wYWN0bmVzcygxZS00KVxyXG4gICAgICAgICAgICAubGlua0Rpc3RhbmNlKDMwKVxyXG4gICAgICAgICAgICAuc3ltbWV0cmljRGlmZkxpbmtMZW5ndGhzKDUpXHJcbiAgICAgICAgICAgIC5wb3dlckdyYXBoR3JvdXBzKGZ1bmN0aW9uIChkKSB7XHJcbiAgICAgICAgICAgIHBvd2VyR3JhcGggPSBkO1xyXG4gICAgICAgICAgICBwb3dlckdyYXBoLmdyb3Vwcy5mb3JFYWNoKGZ1bmN0aW9uICh2KSB7XHJcbiAgICAgICAgICAgICAgICB2LnBhZGRpbmcgPSBncm91cHBhZGRpbmc7XHJcbiAgICAgICAgICAgIH0pO1xyXG4gICAgICAgIH0pLnN0YXJ0KDUwLCAwLCAxMDAsIDAsIGZhbHNlKSxcclxuICAgICAgICBwb3dlckdyYXBoOiBwb3dlckdyYXBoXHJcbiAgICB9O1xyXG59XHJcbmV4cG9ydHMucG93ZXJHcmFwaEdyaWRMYXlvdXQgPSBwb3dlckdyYXBoR3JpZExheW91dDtcclxuLy8jIHNvdXJjZU1hcHBpbmdVUkw9ZGF0YTphcHBsaWNhdGlvbi9qc29uO2Jhc2U2NCxleUoyWlhKemFXOXVJam96TENKbWFXeGxJam9pWW1GMFkyZ3Vhbk1pTENKemIzVnlZMlZTYjI5MElqb2lJaXdpYzI5MWNtTmxjeUk2V3lJdUxpOHVMaTlYWldKRGIyeGhMM055WXk5aVlYUmphQzUwY3lKZExDSnVZVzFsY3lJNlcxMHNJbTFoY0hCcGJtZHpJam9pT3p0QlFVRkJMRzFEUVVFeVF6dEJRVU16UXl3eVEwRkJkVU03UVVGUmRrTXNVMEZCWjBJc1QwRkJUeXhEUVVGRExGRkJRVkVzUlVGQlJTeFJRVUZuUWl4RlFVRkZMRTFCUVdNc1JVRkJSU3hYUVVGdFFqdEpRVU51Uml4UlFVRlJMRU5CUVVNc1NVRkJTU3hEUVVGRExFdEJRVXNzUTBGQlF5eERRVUZETEVWQlFVVXNRMEZCUXl4RlFVRkZMRU5CUVVNc1JVRkJSU3hGUVVGRkxFVkJRVVVzUzBGQlN5eERRVUZETEVOQlFVTTdTVUZEZUVNc1NVRkJTU3hWUVVGVkxFZEJRVWNzUzBGQlN5eERRVUZETEZGQlFWRXNRMEZCUXl4SlFVRkpMRU5CUVVNc1MwRkJTeXhGUVVGRkxFVkJRVVVzVVVGQlVTeERRVUZETEVsQlFVa3NRMEZCUXl4TlFVRk5MRVZCUVVVc1JVRkJSU3hOUVVGTkxFVkJRVVVzVjBGQlZ5eERRVUZETEVOQlFVTTdTVUZETTBZc1QwRkJUeXhWUVVGVkxFTkJRVU1zVlVGQlZTeERRVUZOTEZGQlFWRXNRMEZCUXl4VlFVRlZMRU5CUVVNc1ZVRkJWU3hGUVVGRkxGRkJRVkVzUlVGQlJTeFZRVUZCTEVOQlFVTXNTVUZCUnl4UFFVRkJMRU5CUVVNc1EwRkJReXhOUVVGTkxFTkJRVU1zVlVGQlZTeERRVUZETEVWQlFVVXNSVUZCZEVJc1EwRkJjMElzUlVGQlJTeFZRVUZCTEVOQlFVTXNTVUZCUnl4UFFVRkJMRU5CUVVNc1EwRkJReXhOUVVGTkxFTkJRVU1zVlVGQlZTeERRVUZETEVWQlFVVXNSVUZCZEVJc1EwRkJjMElzUTBGQlF5eERRVUZETzBGQlEzaEpMRU5CUVVNN1FVRktSQ3d3UWtGSlF6dEJRVVZFTEZOQlFWTXNTMEZCU3l4RFFVRkRMRXRCUVVzc1JVRkJSU3hOUVVGTkxFVkJRVVVzVFVGQll5eEZRVUZGTEZkQlFXMUNPMGxCUXpkRUxFdEJRVXNzUTBGQlF5eFBRVUZQTEVOQlFVTXNWVUZCUVN4RFFVRkRPMUZCUTFnc1EwRkJReXhEUVVGRExGVkJRVlVzUjBGQlVUdFpRVU5vUWl4SlFVRkpMRVZCUVVVc1EwRkJReXhEUVVGRExFbEJRVWs3V1VGRFdpeE5RVUZOTEVWQlFVVXNRMEZCUXl4RFFVRkRMRTFCUVUwc1EwRkJReXhQUVVGUExFTkJRVU1zUTBGQlF5eE5RVUZOTEVOQlFVTTdVMEZEY0VNc1EwRkJRenRKUVVOT0xFTkJRVU1zUTBGQlF5eERRVUZETzBsQlEwZ3NUVUZCVFN4RFFVRkRMRTlCUVU4c1EwRkJReXhWUVVGQkxFTkJRVU03VVVGRFdpeERRVUZETEVOQlFVTXNWVUZCVlN4SFFVRlJPMWxCUTJoQ0xFMUJRVTBzUlVGQlJTeERRVUZETEVOQlFVTXNUVUZCVFN4RFFVRkRMRTlCUVU4c1EwRkJReXhEUVVGRExGZEJRVmNzUTBGQlF6dFpRVU4wUXl4UlFVRlJMRVZCUVVVc1EwRkJReXhQUVVGUExFTkJRVU1zUTBGQlF5eE5RVUZOTEV0QlFVc3NWMEZCVnl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zVFVGQlRTeERRVUZETEVkQlFVY3NRMEZCUXl4VlFVRkJMRU5CUVVNc1NVRkJSeXhQUVVGQkxFdEJRVXNzUTBGQlF5eE5RVUZOTEVkQlFVY3NRMEZCUXl4RFFVRkRMRVZCUVVVc1JVRkJia0lzUTBGQmJVSXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhGUVVGRkxFTkJRVU03YVVKQlEyNUdMRTFCUVUwc1EwRkJReXhQUVVGUExFTkJRVU1zUTBGQlF5eE5RVUZOTEV0QlFVc3NWMEZCVnl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zVFVGQlRTeERRVUZETEVkQlFVY3NRMEZCUXl4VlFVRkJMRU5CUVVNc1NVRkJSeXhQUVVGQkxFTkJRVU1zUTBGQlF5eExRVUZMTEVWQlFWQXNRMEZCVHl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFVkJRVVVzUTBGQlF6dFRRVU5vUml4RFFVRkRPMGxCUTA0c1EwRkJReXhEUVVGRExFTkJRVU03U1VGRFNDeEpRVUZKTEdWQlFXVXNSMEZCUnl4TFFVRkxMRU5CUVVNc1RVRkJUU3hEUVVGRExFMUJRVTBzUTBGQlF5eERRVUZETEVkQlFVY3NRMEZCUXl4VlFVRkRMRU5CUVVNc1JVRkJSU3hEUVVGRE8xRkJRMmhFTEVOQlFVTXNRMEZCUXl4VlFVRlZMRU5CUVVNc1JVRkJSU3hIUVVGSExFTkJRVU1zUTBGQlF6dFJRVU53UWl4UFFVRlBMRU5CUVVNc1EwRkJReXhWUVVGVkxFTkJRVU03U1VGRGVFSXNRMEZCUXl4RFFVRkRMRU5CUVVNN1NVRkRTQ3hQUVVGUExFbEJRVWtzZFVKQlFWVXNRMEZCUXl4bFFVRmxMRVZCUVVVN1VVRkRia01zVjBGQlZ5eEZRVUZGTEZWQlFVTXNRMEZCVFN4SlFVRkxMRTlCUVVFc1EwRkJReXhEUVVGRExGRkJRVkVzUlVGQlZpeERRVUZWTzFGQlEyNURMRk5CUVZNc1JVRkJSU3hWUVVGQkxFTkJRVU1zU1VGQlNTeFBRVUZCTEVOQlFVTXNRMEZCUXl4TlFVRk5MRVZCUVZJc1EwRkJVVHRMUVVNelFpeEZRVUZGTEUxQlFVMHNSMEZCUnl4WFFVRlhMRU5CUVVNc1EwRkJRenRCUVVNM1FpeERRVUZETzBGQlJVUXNVMEZCWjBJc2IwSkJRVzlDTEVOQlEyaERMRXRCUVRaRExFVkJRemRETEVsQlFXTXNSVUZEWkN4WlFVRnZRanRKUVVkd1FpeEpRVUZKTEZWQlFWVXNRMEZCUXp0SlFVTm1MRXRCUVVzc1EwRkJReXhMUVVGTExFTkJRVU1zVDBGQlR5eERRVUZETEZWQlFVTXNRMEZCUXl4RlFVRkRMRU5CUVVNc1NVRkJTeXhQUVVGTkxFTkJRVVVzUTBGQlF5eExRVUZMTEVkQlFVY3NRMEZCUXl4RlFVRnNRaXhEUVVGclFpeERRVUZETEVOQlFVTTdTVUZEYWtRc1NVRkJTU3hsUVVGTkxFVkJRVVU3VTBGRFVDeGhRVUZoTEVOQlFVTXNTMEZCU3l4RFFVRkRPMU5CUTNCQ0xFdEJRVXNzUTBGQlF5eExRVUZMTEVOQlFVTXNTMEZCU3l4RFFVRkRPMU5CUTJ4Q0xFdEJRVXNzUTBGQlF5eExRVUZMTEVOQlFVTXNTMEZCU3l4RFFVRkRPMU5CUTJ4Q0xHZENRVUZuUWl4RFFVRkRMRlZCUVZVc1EwRkJRenRSUVVONlFpeFZRVUZWTEVkQlFVY3NRMEZCUXl4RFFVRkRPMUZCUTJZc1ZVRkJWU3hEUVVGRExFMUJRVTBzUTBGQlF5eFBRVUZQTEVOQlFVTXNWVUZCUVN4RFFVRkRMRWxCUVVjc1QwRkJRU3hEUVVGRExFTkJRVU1zVDBGQlR5eEhRVUZITEZsQlFWa3NSVUZCZUVJc1EwRkJkMElzUTBGQlF5eERRVUZETzBsQlF6VkVMRU5CUVVNc1EwRkJReXhEUVVGRE8wbEJTVkFzU1VGQlNTeERRVUZETEVkQlFVY3NTMEZCU3l4RFFVRkRMRXRCUVVzc1EwRkJReXhOUVVGTkxFTkJRVU03U1VGRE0wSXNTVUZCU1N4TFFVRkxMRWRCUVVjc1JVRkJSU3hEUVVGRE8wbEJRMllzU1VGQlNTeEZRVUZGTEVkQlFVY3NTMEZCU3l4RFFVRkRMRXRCUVVzc1EwRkJReXhMUVVGTExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTTdTVUZET1VJc1JVRkJSU3hEUVVGRExFOUJRVThzUTBGQlF5eFZRVUZETEVOQlFVTXNSVUZCUlN4RFFVRkRMRWxCUVVzc1QwRkJUU3hEUVVGRkxFTkJRVU1zUzBGQlN5eEhRVUZITEVOQlFVTXNSVUZCYkVJc1EwRkJhMElzUTBGQlF5eERRVUZETzBsQlEzcERMRlZCUVZVc1EwRkJReXhOUVVGTkxFTkJRVU1zVDBGQlR5eERRVUZETEZWQlFVRXNRMEZCUXp0UlFVTjJRaXhKUVVGSkxGTkJRVk1zUjBGQlJ5eERRVUZETEVOQlFVTXNTMEZCU3l4SFFVRkhMRU5CUVVNc1EwRkJReXhGUVVGRkxFZEJRVWNzUTBGQlF5eERRVUZETzFGQlEyNURMRVZCUVVVc1EwRkJReXhKUVVGSkxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTTdVVUZEV0N4SlFVRkpMRTlCUVU4c1EwRkJReXhEUVVGRExFMUJRVTBzUzBGQlN5eFhRVUZYTzFsQlF5OUNMRU5CUVVNc1EwRkJReXhOUVVGTkxFTkJRVU1zVDBGQlR5eERRVUZETEZWQlFVRXNRMEZCUXl4SlFVRkpMRTlCUVVFc1MwRkJTeXhEUVVGRExFbEJRVWtzUTBGQlF5eEZRVUZGTEUxQlFVMHNSVUZCUlN4VFFVRlRMRVZCUVVVc1RVRkJUU3hGUVVGRkxFTkJRVU1zUTBGQlF5eExRVUZMTEVWQlFVVXNRMEZCUXl4RlFVRnNSQ3hEUVVGclJDeERRVUZETEVOQlFVTTdVVUZET1VVc1NVRkJTU3hQUVVGUExFTkJRVU1zUTBGQlF5eE5RVUZOTEV0QlFVc3NWMEZCVnp0WlFVTXZRaXhEUVVGRExFTkJRVU1zVFVGQlRTeERRVUZETEU5QlFVOHNRMEZCUXl4VlFVRkJMRVZCUVVVc1NVRkJTU3hQUVVGQkxFdEJRVXNzUTBGQlF5eEpRVUZKTEVOQlFVTXNSVUZCUlN4TlFVRk5MRVZCUVVVc1UwRkJVeXhGUVVGRkxFMUJRVTBzUlVGQlJTeEZRVUZGTEVOQlFVTXNSVUZCUlN4SFFVRkhMRU5CUVVNc1JVRkJSU3hEUVVGRExFVkJRWEJFTEVOQlFXOUVMRU5CUVVNc1EwRkJRenRKUVVOeVJpeERRVUZETEVOQlFVTXNRMEZCUXp0SlFVTklMRlZCUVZVc1EwRkJReXhWUVVGVkxFTkJRVU1zVDBGQlR5eERRVUZETEZWQlFVRXNRMEZCUXp0UlFVTXpRaXhMUVVGTExFTkJRVU1zU1VGQlNTeERRVUZETEVWQlFVVXNUVUZCVFN4RlFVRkZMRU5CUVVNc1EwRkJReXhOUVVGTkxFTkJRVU1zUzBGQlN5eEZRVUZGTEUxQlFVMHNSVUZCUlN4RFFVRkRMRU5CUVVNc1RVRkJUU3hEUVVGRExFdEJRVXNzUlVGQlJTeERRVUZETEVOQlFVTTdTVUZEYmtVc1EwRkJReXhEUVVGRExFTkJRVU03U1VGSFNDeEpRVUZKTEdWQlFVMHNSVUZCUlR0VFFVTlFMRWxCUVVrc1EwRkJReXhKUVVGSkxFTkJRVU03VTBGRFZpeExRVUZMTEVOQlFVTXNSVUZCUlN4RFFVRkRPMU5CUTFRc1MwRkJTeXhEUVVGRExFdEJRVXNzUTBGQlF6dFRRVU5hTEdGQlFXRXNRMEZCUXl4TFFVRkxMRU5CUVVNN1UwRkRjRUlzV1VGQldTeERRVUZETEVWQlFVVXNRMEZCUXp0VFFVTm9RaXgzUWtGQmQwSXNRMEZCUXl4RFFVRkRMRU5CUVVNN1UwRkRNMElzYjBKQlFXOUNMRU5CUVVNc1NVRkJTU3hEUVVGRE8xTkJRekZDTEV0QlFVc3NRMEZCUXl4SFFVRkhMRVZCUVVVc1EwRkJReXhGUVVGRkxFTkJRVU1zUlVGQlJTeERRVUZETEVWQlFVVXNTMEZCU3l4RFFVRkRMRU5CUVVNN1NVRkxhRU1zVDBGQlR6dFJRVU5JTEVsQlFVa3NSVUZEUVN4SlFVRkpMR1ZCUVUwc1JVRkJSVHRoUVVOWUxHOUNRVUZ2UWl4RFFVRkRMRWxCUVVrc1EwRkJRenRoUVVNeFFpeEpRVUZKTEVOQlFVTXNTVUZCU1N4RFFVRkRPMkZCUTFZc1lVRkJZU3hEUVVGRExFbEJRVWtzUTBGQlF6dGhRVU51UWl4TFFVRkxMRU5CUVVNc1MwRkJTeXhEUVVGRExFdEJRVXNzUTBGQlF6dGhRVU5zUWl4TFFVRkxMRU5CUVVNc1MwRkJTeXhEUVVGRExFdEJRVXNzUTBGQlF6dGhRVVZzUWl4blFrRkJaMElzUTBGQlF5eEpRVUZKTEVOQlFVTTdZVUZEZEVJc1dVRkJXU3hEUVVGRExFVkJRVVVzUTBGQlF6dGhRVU5vUWl4M1FrRkJkMElzUTBGQlF5eERRVUZETEVOQlFVTTdZVUZETTBJc1owSkJRV2RDTEVOQlFVTXNWVUZCVlN4RFFVRkRPMWxCUTNwQ0xGVkJRVlVzUjBGQlJ5eERRVUZETEVOQlFVTTdXVUZEWml4VlFVRlZMRU5CUVVNc1RVRkJUU3hEUVVGRExFOUJRVThzUTBGQlF5eFZRVUZWTEVOQlFVTTdaMEpCUTJwRExFTkJRVU1zUTBGQlF5eFBRVUZQTEVkQlFVY3NXVUZCV1N4RFFVRkJPMWxCUXpWQ0xFTkJRVU1zUTBGQlF5eERRVUZETzFGQlExQXNRMEZCUXl4RFFVRkRMRU5CUVVNc1MwRkJTeXhEUVVGRExFVkJRVVVzUlVGQlJTeERRVUZETEVWQlFVVXNSMEZCUnl4RlFVRkZMRU5CUVVNc1JVRkJSU3hMUVVGTExFTkJRVU03VVVGRGJFTXNWVUZCVlN4RlFVRkZMRlZCUVZVN1MwRkRla0lzUTBGQlF6dEJRVU5PTEVOQlFVTTdRVUZ5UlVRc2IwUkJjVVZESW4wPSIsIlwidXNlIHN0cmljdFwiO1xyXG5PYmplY3QuZGVmaW5lUHJvcGVydHkoZXhwb3J0cywgXCJfX2VzTW9kdWxlXCIsIHsgdmFsdWU6IHRydWUgfSk7XHJcbnZhciBkM3YzID0gcmVxdWlyZShcIi4vZDN2M2FkYXB0b3JcIik7XHJcbnZhciBkM3Y0ID0gcmVxdWlyZShcIi4vZDN2NGFkYXB0b3JcIik7XHJcbjtcclxuZnVuY3Rpb24gZDNhZGFwdG9yKGQzQ29udGV4dCkge1xyXG4gICAgaWYgKCFkM0NvbnRleHQgfHwgaXNEM1YzKGQzQ29udGV4dCkpIHtcclxuICAgICAgICByZXR1cm4gbmV3IGQzdjMuRDNTdHlsZUxheW91dEFkYXB0b3IoKTtcclxuICAgIH1cclxuICAgIHJldHVybiBuZXcgZDN2NC5EM1N0eWxlTGF5b3V0QWRhcHRvcihkM0NvbnRleHQpO1xyXG59XHJcbmV4cG9ydHMuZDNhZGFwdG9yID0gZDNhZGFwdG9yO1xyXG5mdW5jdGlvbiBpc0QzVjMoZDNDb250ZXh0KSB7XHJcbiAgICB2YXIgdjNleHAgPSAvXjNcXC4vO1xyXG4gICAgcmV0dXJuIGQzQ29udGV4dC52ZXJzaW9uICYmIGQzQ29udGV4dC52ZXJzaW9uLm1hdGNoKHYzZXhwKSAhPT0gbnVsbDtcclxufVxyXG4vLyMgc291cmNlTWFwcGluZ1VSTD1kYXRhOmFwcGxpY2F0aW9uL2pzb247YmFzZTY0LGV5SjJaWEp6YVc5dUlqb3pMQ0ptYVd4bElqb2laRE5oWkdGd2RHOXlMbXB6SWl3aWMyOTFjbU5sVW05dmRDSTZJaUlzSW5OdmRYSmpaWE1pT2xzaUxpNHZMaTR2VjJWaVEyOXNZUzl6Y21NdlpETmhaR0Z3ZEc5eUxuUnpJbDBzSW01aGJXVnpJanBiWFN3aWJXRndjR2x1WjNNaU9pSTdPMEZCUVVFc2IwTkJRWEZETzBGQlEzSkRMRzlEUVVGeFF6dEJRVWRWTEVOQlFVTTdRVUUwUW1oRUxGTkJRV2RDTEZOQlFWTXNRMEZCUXl4VFFVRjNRenRKUVVNNVJDeEpRVUZKTEVOQlFVTXNVMEZCVXl4SlFVRkpMRTFCUVUwc1EwRkJReXhUUVVGVExFTkJRVU1zUlVGQlJUdFJRVU5xUXl4UFFVRlBMRWxCUVVrc1NVRkJTU3hEUVVGRExHOUNRVUZ2UWl4RlFVRkZMRU5CUVVNN1MwRkRNVU03U1VGRFJDeFBRVUZQTEVsQlFVa3NTVUZCU1N4RFFVRkRMRzlDUVVGdlFpeERRVUZETEZOQlFWTXNRMEZCUXl4RFFVRkRPMEZCUTNCRUxFTkJRVU03UVVGTVJDdzRRa0ZMUXp0QlFVVkVMRk5CUVZNc1RVRkJUU3hEUVVGRExGTkJRWFZETzBsQlEyNUVMRWxCUVUwc1MwRkJTeXhIUVVGSExFMUJRVTBzUTBGQlF6dEpRVU55UWl4UFFVRmhMRk5CUVZVc1EwRkJReXhQUVVGUExFbEJRVlVzVTBGQlZTeERRVUZETEU5QlFVOHNRMEZCUXl4TFFVRkxMRU5CUVVNc1MwRkJTeXhEUVVGRExFdEJRVXNzU1VGQlNTeERRVUZETzBGQlEzUkdMRU5CUVVNaWZRPT0iLCJcInVzZSBzdHJpY3RcIjtcclxudmFyIF9fZXh0ZW5kcyA9ICh0aGlzICYmIHRoaXMuX19leHRlbmRzKSB8fCAoZnVuY3Rpb24gKCkge1xyXG4gICAgdmFyIGV4dGVuZFN0YXRpY3MgPSBmdW5jdGlvbiAoZCwgYikge1xyXG4gICAgICAgIGV4dGVuZFN0YXRpY3MgPSBPYmplY3Quc2V0UHJvdG90eXBlT2YgfHxcclxuICAgICAgICAgICAgKHsgX19wcm90b19fOiBbXSB9IGluc3RhbmNlb2YgQXJyYXkgJiYgZnVuY3Rpb24gKGQsIGIpIHsgZC5fX3Byb3RvX18gPSBiOyB9KSB8fFxyXG4gICAgICAgICAgICBmdW5jdGlvbiAoZCwgYikgeyBmb3IgKHZhciBwIGluIGIpIGlmIChiLmhhc093blByb3BlcnR5KHApKSBkW3BdID0gYltwXTsgfTtcclxuICAgICAgICByZXR1cm4gZXh0ZW5kU3RhdGljcyhkLCBiKTtcclxuICAgIH07XHJcbiAgICByZXR1cm4gZnVuY3Rpb24gKGQsIGIpIHtcclxuICAgICAgICBleHRlbmRTdGF0aWNzKGQsIGIpO1xyXG4gICAgICAgIGZ1bmN0aW9uIF9fKCkgeyB0aGlzLmNvbnN0cnVjdG9yID0gZDsgfVxyXG4gICAgICAgIGQucHJvdG90eXBlID0gYiA9PT0gbnVsbCA/IE9iamVjdC5jcmVhdGUoYikgOiAoX18ucHJvdG90eXBlID0gYi5wcm90b3R5cGUsIG5ldyBfXygpKTtcclxuICAgIH07XHJcbn0pKCk7XHJcbk9iamVjdC5kZWZpbmVQcm9wZXJ0eShleHBvcnRzLCBcIl9fZXNNb2R1bGVcIiwgeyB2YWx1ZTogdHJ1ZSB9KTtcclxudmFyIGxheW91dF8xID0gcmVxdWlyZShcIi4vbGF5b3V0XCIpO1xyXG52YXIgRDNTdHlsZUxheW91dEFkYXB0b3IgPSAoZnVuY3Rpb24gKF9zdXBlcikge1xyXG4gICAgX19leHRlbmRzKEQzU3R5bGVMYXlvdXRBZGFwdG9yLCBfc3VwZXIpO1xyXG4gICAgZnVuY3Rpb24gRDNTdHlsZUxheW91dEFkYXB0b3IoKSB7XHJcbiAgICAgICAgdmFyIF90aGlzID0gX3N1cGVyLmNhbGwodGhpcykgfHwgdGhpcztcclxuICAgICAgICBfdGhpcy5ldmVudCA9IGQzLmRpc3BhdGNoKGxheW91dF8xLkV2ZW50VHlwZVtsYXlvdXRfMS5FdmVudFR5cGUuc3RhcnRdLCBsYXlvdXRfMS5FdmVudFR5cGVbbGF5b3V0XzEuRXZlbnRUeXBlLnRpY2tdLCBsYXlvdXRfMS5FdmVudFR5cGVbbGF5b3V0XzEuRXZlbnRUeXBlLmVuZF0pO1xyXG4gICAgICAgIHZhciBkM2xheW91dCA9IF90aGlzO1xyXG4gICAgICAgIHZhciBkcmFnO1xyXG4gICAgICAgIF90aGlzLmRyYWcgPSBmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgICAgIGlmICghZHJhZykge1xyXG4gICAgICAgICAgICAgICAgdmFyIGRyYWcgPSBkMy5iZWhhdmlvci5kcmFnKClcclxuICAgICAgICAgICAgICAgICAgICAub3JpZ2luKGxheW91dF8xLkxheW91dC5kcmFnT3JpZ2luKVxyXG4gICAgICAgICAgICAgICAgICAgIC5vbihcImRyYWdzdGFydC5kM2FkYXB0b3JcIiwgbGF5b3V0XzEuTGF5b3V0LmRyYWdTdGFydClcclxuICAgICAgICAgICAgICAgICAgICAub24oXCJkcmFnLmQzYWRhcHRvclwiLCBmdW5jdGlvbiAoZCkge1xyXG4gICAgICAgICAgICAgICAgICAgIGxheW91dF8xLkxheW91dC5kcmFnKGQsIGQzLmV2ZW50KTtcclxuICAgICAgICAgICAgICAgICAgICBkM2xheW91dC5yZXN1bWUoKTtcclxuICAgICAgICAgICAgICAgIH0pXHJcbiAgICAgICAgICAgICAgICAgICAgLm9uKFwiZHJhZ2VuZC5kM2FkYXB0b3JcIiwgbGF5b3V0XzEuTGF5b3V0LmRyYWdFbmQpO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIGlmICghYXJndW1lbnRzLmxlbmd0aClcclxuICAgICAgICAgICAgICAgIHJldHVybiBkcmFnO1xyXG4gICAgICAgICAgICB0aGlzXHJcbiAgICAgICAgICAgICAgICAuY2FsbChkcmFnKTtcclxuICAgICAgICB9O1xyXG4gICAgICAgIHJldHVybiBfdGhpcztcclxuICAgIH1cclxuICAgIEQzU3R5bGVMYXlvdXRBZGFwdG9yLnByb3RvdHlwZS50cmlnZ2VyID0gZnVuY3Rpb24gKGUpIHtcclxuICAgICAgICB2YXIgZDNldmVudCA9IHsgdHlwZTogbGF5b3V0XzEuRXZlbnRUeXBlW2UudHlwZV0sIGFscGhhOiBlLmFscGhhLCBzdHJlc3M6IGUuc3RyZXNzIH07XHJcbiAgICAgICAgdGhpcy5ldmVudFtkM2V2ZW50LnR5cGVdKGQzZXZlbnQpO1xyXG4gICAgfTtcclxuICAgIEQzU3R5bGVMYXlvdXRBZGFwdG9yLnByb3RvdHlwZS5raWNrID0gZnVuY3Rpb24gKCkge1xyXG4gICAgICAgIHZhciBfdGhpcyA9IHRoaXM7XHJcbiAgICAgICAgZDMudGltZXIoZnVuY3Rpb24gKCkgeyByZXR1cm4gX3N1cGVyLnByb3RvdHlwZS50aWNrLmNhbGwoX3RoaXMpOyB9KTtcclxuICAgIH07XHJcbiAgICBEM1N0eWxlTGF5b3V0QWRhcHRvci5wcm90b3R5cGUub24gPSBmdW5jdGlvbiAoZXZlbnRUeXBlLCBsaXN0ZW5lcikge1xyXG4gICAgICAgIGlmICh0eXBlb2YgZXZlbnRUeXBlID09PSAnc3RyaW5nJykge1xyXG4gICAgICAgICAgICB0aGlzLmV2ZW50Lm9uKGV2ZW50VHlwZSwgbGlzdGVuZXIpO1xyXG4gICAgICAgIH1cclxuICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgdGhpcy5ldmVudC5vbihsYXlvdXRfMS5FdmVudFR5cGVbZXZlbnRUeXBlXSwgbGlzdGVuZXIpO1xyXG4gICAgICAgIH1cclxuICAgICAgICByZXR1cm4gdGhpcztcclxuICAgIH07XHJcbiAgICByZXR1cm4gRDNTdHlsZUxheW91dEFkYXB0b3I7XHJcbn0obGF5b3V0XzEuTGF5b3V0KSk7XHJcbmV4cG9ydHMuRDNTdHlsZUxheW91dEFkYXB0b3IgPSBEM1N0eWxlTGF5b3V0QWRhcHRvcjtcclxuZnVuY3Rpb24gZDNhZGFwdG9yKCkge1xyXG4gICAgcmV0dXJuIG5ldyBEM1N0eWxlTGF5b3V0QWRhcHRvcigpO1xyXG59XHJcbmV4cG9ydHMuZDNhZGFwdG9yID0gZDNhZGFwdG9yO1xyXG4vLyMgc291cmNlTWFwcGluZ1VSTD1kYXRhOmFwcGxpY2F0aW9uL2pzb247YmFzZTY0LGV5SjJaWEp6YVc5dUlqb3pMQ0ptYVd4bElqb2laRE4yTTJGa1lYQjBiM0l1YW5NaUxDSnpiM1Z5WTJWU2IyOTBJam9pSWl3aWMyOTFjbU5sY3lJNld5SXVMaTh1TGk5WFpXSkRiMnhoTDNOeVl5OWtNM1l6WVdSaGNIUnZjaTUwY3lKZExDSnVZVzFsY3lJNlcxMHNJbTFoY0hCcGJtZHpJam9pT3pzN096czdPenM3T3pzN096czdRVUZOUVN4dFEwRkJhMFE3UVVGSE9VTTdTVUZCTUVNc2QwTkJRVTA3U1VGblFqVkRPMUZCUVVFc1dVRkRTU3hwUWtGQlR5eFRRWFZDVmp0UlFYWkRSQ3hYUVVGTExFZEJRVWNzUlVGQlJTeERRVUZETEZGQlFWRXNRMEZCUXl4clFrRkJVeXhEUVVGRExHdENRVUZUTEVOQlFVTXNTMEZCU3l4RFFVRkRMRVZCUVVVc2EwSkJRVk1zUTBGQlF5eHJRa0ZCVXl4RFFVRkRMRWxCUVVrc1EwRkJReXhGUVVGRkxHdENRVUZUTEVOQlFVTXNhMEpCUVZNc1EwRkJReXhIUVVGSExFTkJRVU1zUTBGQlF5eERRVUZETzFGQmEwSnFSeXhKUVVGSkxGRkJRVkVzUjBGQlJ5eExRVUZKTEVOQlFVTTdVVUZEY0VJc1NVRkJTU3hKUVVGSkxFTkJRVU03VVVGRFZDeExRVUZKTEVOQlFVTXNTVUZCU1N4SFFVRkhPMWxCUTFJc1NVRkJTU3hEUVVGRExFbEJRVWtzUlVGQlJUdG5Ra0ZEVUN4SlFVRkpMRWxCUVVrc1IwRkJSeXhGUVVGRkxFTkJRVU1zVVVGQlVTeERRVUZETEVsQlFVa3NSVUZCUlR0eFFrRkRlRUlzVFVGQlRTeERRVUZETEdWQlFVMHNRMEZCUXl4VlFVRlZMRU5CUVVNN2NVSkJRM3BDTEVWQlFVVXNRMEZCUXl4eFFrRkJjVUlzUlVGQlJTeGxRVUZOTEVOQlFVTXNVMEZCVXl4RFFVRkRPM0ZDUVVNelF5eEZRVUZGTEVOQlFVTXNaMEpCUVdkQ0xFVkJRVVVzVlVGQlFTeERRVUZETzI5Q1FVTnVRaXhsUVVGTkxFTkJRVU1zU1VGQlNTeERRVUZETEVOQlFVTXNSVUZCVHl4RlFVRkZMRU5CUVVNc1MwRkJTeXhEUVVGRExFTkJRVU03YjBKQlF6bENMRkZCUVZFc1EwRkJReXhOUVVGTkxFVkJRVVVzUTBGQlF6dG5Ra0ZEZEVJc1EwRkJReXhEUVVGRE8zRkNRVU5FTEVWQlFVVXNRMEZCUXl4dFFrRkJiVUlzUlVGQlJTeGxRVUZOTEVOQlFVTXNUMEZCVHl4RFFVRkRMRU5CUVVNN1lVRkRhRVE3V1VGRlJDeEpRVUZKTEVOQlFVTXNVMEZCVXl4RFFVRkRMRTFCUVUwN1owSkJRVVVzVDBGQlR5eEpRVUZKTEVOQlFVTTdXVUZIYmtNc1NVRkJTVHRwUWtGRlF5eEpRVUZKTEVOQlFVTXNTVUZCU1N4RFFVRkRMRU5CUVVNN1VVRkRjRUlzUTBGQlF5eERRVUZCT3p0SlFVTk1MRU5CUVVNN1NVRnlRMFFzYzBOQlFVOHNSMEZCVUN4VlFVRlJMRU5CUVZFN1VVRkRXaXhKUVVGSkxFOUJRVThzUjBGQlJ5eEZRVUZGTEVsQlFVa3NSVUZCUlN4clFrRkJVeXhEUVVGRExFTkJRVU1zUTBGQlF5eEpRVUZKTEVOQlFVTXNSVUZCUlN4TFFVRkxMRVZCUVVVc1EwRkJReXhEUVVGRExFdEJRVXNzUlVGQlJTeE5RVUZOTEVWQlFVVXNRMEZCUXl4RFFVRkRMRTFCUVUwc1JVRkJSU3hEUVVGRE8xRkJRelZGTEVsQlFVa3NRMEZCUXl4TFFVRkxMRU5CUVVNc1QwRkJUeXhEUVVGRExFbEJRVWtzUTBGQlF5eERRVUZETEU5QlFVOHNRMEZCUXl4RFFVRkRPMGxCUTNSRExFTkJRVU03U1VGSFJDeHRRMEZCU1N4SFFVRktPMUZCUVVFc2FVSkJSVU03VVVGRVJ5eEZRVUZGTEVOQlFVTXNTMEZCU3l4RFFVRkRMR05CUVUwc1QwRkJRU3hwUWtGQlRTeEpRVUZKTEZsQlFVVXNSVUZCV2l4RFFVRlpMRU5CUVVNc1EwRkJRenRKUVVOcVF5eERRVUZETzBsQlowTkVMR2xEUVVGRkxFZEJRVVlzVlVGQlJ5eFRRVUUyUWl4RlFVRkZMRkZCUVc5Q08xRkJRMnhFTEVsQlFVa3NUMEZCVHl4VFFVRlRMRXRCUVVzc1VVRkJVU3hGUVVGRk8xbEJReTlDTEVsQlFVa3NRMEZCUXl4TFFVRkxMRU5CUVVNc1JVRkJSU3hEUVVGRExGTkJRVk1zUlVGQlJTeFJRVUZSTEVOQlFVTXNRMEZCUXp0VFFVTjBRenRoUVVGTk8xbEJRMGdzU1VGQlNTeERRVUZETEV0QlFVc3NRMEZCUXl4RlFVRkZMRU5CUVVNc2EwSkJRVk1zUTBGQlF5eFRRVUZUTEVOQlFVTXNSVUZCUlN4UlFVRlJMRU5CUVVNc1EwRkJRenRUUVVOcVJEdFJRVU5FTEU5QlFVOHNTVUZCU1N4RFFVRkRPMGxCUTJoQ0xFTkJRVU03U1VGRFRDd3lRa0ZCUXp0QlFVRkVMRU5CUVVNc1FVRnVSRVFzUTBGQk1FTXNaVUZCVFN4SFFXMUVMME03UVVGdVJGa3NiMFJCUVc5Q08wRkJhVVZxUXl4VFFVRm5RaXhUUVVGVE8wbEJRM0pDTEU5QlFVOHNTVUZCU1N4dlFrRkJiMElzUlVGQlJTeERRVUZETzBGQlEzUkRMRU5CUVVNN1FVRkdSQ3c0UWtGRlF5SjkiLCJcInVzZSBzdHJpY3RcIjtcclxudmFyIF9fZXh0ZW5kcyA9ICh0aGlzICYmIHRoaXMuX19leHRlbmRzKSB8fCAoZnVuY3Rpb24gKCkge1xyXG4gICAgdmFyIGV4dGVuZFN0YXRpY3MgPSBmdW5jdGlvbiAoZCwgYikge1xyXG4gICAgICAgIGV4dGVuZFN0YXRpY3MgPSBPYmplY3Quc2V0UHJvdG90eXBlT2YgfHxcclxuICAgICAgICAgICAgKHsgX19wcm90b19fOiBbXSB9IGluc3RhbmNlb2YgQXJyYXkgJiYgZnVuY3Rpb24gKGQsIGIpIHsgZC5fX3Byb3RvX18gPSBiOyB9KSB8fFxyXG4gICAgICAgICAgICBmdW5jdGlvbiAoZCwgYikgeyBmb3IgKHZhciBwIGluIGIpIGlmIChiLmhhc093blByb3BlcnR5KHApKSBkW3BdID0gYltwXTsgfTtcclxuICAgICAgICByZXR1cm4gZXh0ZW5kU3RhdGljcyhkLCBiKTtcclxuICAgIH07XHJcbiAgICByZXR1cm4gZnVuY3Rpb24gKGQsIGIpIHtcclxuICAgICAgICBleHRlbmRTdGF0aWNzKGQsIGIpO1xyXG4gICAgICAgIGZ1bmN0aW9uIF9fKCkgeyB0aGlzLmNvbnN0cnVjdG9yID0gZDsgfVxyXG4gICAgICAgIGQucHJvdG90eXBlID0gYiA9PT0gbnVsbCA/IE9iamVjdC5jcmVhdGUoYikgOiAoX18ucHJvdG90eXBlID0gYi5wcm90b3R5cGUsIG5ldyBfXygpKTtcclxuICAgIH07XHJcbn0pKCk7XHJcbk9iamVjdC5kZWZpbmVQcm9wZXJ0eShleHBvcnRzLCBcIl9fZXNNb2R1bGVcIiwgeyB2YWx1ZTogdHJ1ZSB9KTtcclxudmFyIGxheW91dF8xID0gcmVxdWlyZShcIi4vbGF5b3V0XCIpO1xyXG52YXIgRDNTdHlsZUxheW91dEFkYXB0b3IgPSAoZnVuY3Rpb24gKF9zdXBlcikge1xyXG4gICAgX19leHRlbmRzKEQzU3R5bGVMYXlvdXRBZGFwdG9yLCBfc3VwZXIpO1xyXG4gICAgZnVuY3Rpb24gRDNTdHlsZUxheW91dEFkYXB0b3IoZDNDb250ZXh0KSB7XHJcbiAgICAgICAgdmFyIF90aGlzID0gX3N1cGVyLmNhbGwodGhpcykgfHwgdGhpcztcclxuICAgICAgICBfdGhpcy5kM0NvbnRleHQgPSBkM0NvbnRleHQ7XHJcbiAgICAgICAgX3RoaXMuZXZlbnQgPSBkM0NvbnRleHQuZGlzcGF0Y2gobGF5b3V0XzEuRXZlbnRUeXBlW2xheW91dF8xLkV2ZW50VHlwZS5zdGFydF0sIGxheW91dF8xLkV2ZW50VHlwZVtsYXlvdXRfMS5FdmVudFR5cGUudGlja10sIGxheW91dF8xLkV2ZW50VHlwZVtsYXlvdXRfMS5FdmVudFR5cGUuZW5kXSk7XHJcbiAgICAgICAgdmFyIGQzbGF5b3V0ID0gX3RoaXM7XHJcbiAgICAgICAgdmFyIGRyYWc7XHJcbiAgICAgICAgX3RoaXMuZHJhZyA9IGZ1bmN0aW9uICgpIHtcclxuICAgICAgICAgICAgaWYgKCFkcmFnKSB7XHJcbiAgICAgICAgICAgICAgICB2YXIgZHJhZyA9IGQzQ29udGV4dC5kcmFnKClcclxuICAgICAgICAgICAgICAgICAgICAuc3ViamVjdChsYXlvdXRfMS5MYXlvdXQuZHJhZ09yaWdpbilcclxuICAgICAgICAgICAgICAgICAgICAub24oXCJzdGFydC5kM2FkYXB0b3JcIiwgbGF5b3V0XzEuTGF5b3V0LmRyYWdTdGFydClcclxuICAgICAgICAgICAgICAgICAgICAub24oXCJkcmFnLmQzYWRhcHRvclwiLCBmdW5jdGlvbiAoZCkge1xyXG4gICAgICAgICAgICAgICAgICAgIGxheW91dF8xLkxheW91dC5kcmFnKGQsIGQzQ29udGV4dC5ldmVudCk7XHJcbiAgICAgICAgICAgICAgICAgICAgZDNsYXlvdXQucmVzdW1lKCk7XHJcbiAgICAgICAgICAgICAgICB9KVxyXG4gICAgICAgICAgICAgICAgICAgIC5vbihcImVuZC5kM2FkYXB0b3JcIiwgbGF5b3V0XzEuTGF5b3V0LmRyYWdFbmQpO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIGlmICghYXJndW1lbnRzLmxlbmd0aClcclxuICAgICAgICAgICAgICAgIHJldHVybiBkcmFnO1xyXG4gICAgICAgICAgICBhcmd1bWVudHNbMF0uY2FsbChkcmFnKTtcclxuICAgICAgICB9O1xyXG4gICAgICAgIHJldHVybiBfdGhpcztcclxuICAgIH1cclxuICAgIEQzU3R5bGVMYXlvdXRBZGFwdG9yLnByb3RvdHlwZS50cmlnZ2VyID0gZnVuY3Rpb24gKGUpIHtcclxuICAgICAgICB2YXIgZDNldmVudCA9IHsgdHlwZTogbGF5b3V0XzEuRXZlbnRUeXBlW2UudHlwZV0sIGFscGhhOiBlLmFscGhhLCBzdHJlc3M6IGUuc3RyZXNzIH07XHJcbiAgICAgICAgdGhpcy5ldmVudC5jYWxsKGQzZXZlbnQudHlwZSwgZDNldmVudCk7XHJcbiAgICB9O1xyXG4gICAgRDNTdHlsZUxheW91dEFkYXB0b3IucHJvdG90eXBlLmtpY2sgPSBmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgdmFyIF90aGlzID0gdGhpcztcclxuICAgICAgICB2YXIgdCA9IHRoaXMuZDNDb250ZXh0LnRpbWVyKGZ1bmN0aW9uICgpIHsgcmV0dXJuIF9zdXBlci5wcm90b3R5cGUudGljay5jYWxsKF90aGlzKSAmJiB0LnN0b3AoKTsgfSk7XHJcbiAgICB9O1xyXG4gICAgRDNTdHlsZUxheW91dEFkYXB0b3IucHJvdG90eXBlLm9uID0gZnVuY3Rpb24gKGV2ZW50VHlwZSwgbGlzdGVuZXIpIHtcclxuICAgICAgICBpZiAodHlwZW9mIGV2ZW50VHlwZSA9PT0gJ3N0cmluZycpIHtcclxuICAgICAgICAgICAgdGhpcy5ldmVudC5vbihldmVudFR5cGUsIGxpc3RlbmVyKTtcclxuICAgICAgICB9XHJcbiAgICAgICAgZWxzZSB7XHJcbiAgICAgICAgICAgIHRoaXMuZXZlbnQub24obGF5b3V0XzEuRXZlbnRUeXBlW2V2ZW50VHlwZV0sIGxpc3RlbmVyKTtcclxuICAgICAgICB9XHJcbiAgICAgICAgcmV0dXJuIHRoaXM7XHJcbiAgICB9O1xyXG4gICAgcmV0dXJuIEQzU3R5bGVMYXlvdXRBZGFwdG9yO1xyXG59KGxheW91dF8xLkxheW91dCkpO1xyXG5leHBvcnRzLkQzU3R5bGVMYXlvdXRBZGFwdG9yID0gRDNTdHlsZUxheW91dEFkYXB0b3I7XHJcbi8vIyBzb3VyY2VNYXBwaW5nVVJMPWRhdGE6YXBwbGljYXRpb24vanNvbjtiYXNlNjQsZXlKMlpYSnphVzl1SWpvekxDSm1hV3hsSWpvaVpETjJOR0ZrWVhCMGIzSXVhbk1pTENKemIzVnlZMlZTYjI5MElqb2lJaXdpYzI5MWNtTmxjeUk2V3lJdUxpOHVMaTlYWldKRGIyeGhMM055WXk5a00zWTBZV1JoY0hSdmNpNTBjeUpkTENKdVlXMWxjeUk2VzEwc0ltMWhjSEJwYm1keklqb2lPenM3T3pzN096czdPenM3T3pzN1FVRkhRU3h0UTBGQmFVUTdRVUZWYWtRN1NVRkJNRU1zZDBOQlFVMDdTVUZwUWpWRExEaENRVUZ2UWl4VFFVRnZRanRSUVVGNFF5eFpRVU5KTEdsQ1FVRlBMRk5CZVVKV08xRkJNVUp0UWl4bFFVRlRMRWRCUVZRc1UwRkJVeXhEUVVGWE8xRkJSWEJETEV0QlFVa3NRMEZCUXl4TFFVRkxMRWRCUVVjc1UwRkJVeXhEUVVGRExGRkJRVkVzUTBGQlF5eHJRa0ZCVXl4RFFVRkRMR3RDUVVGVExFTkJRVU1zUzBGQlN5eERRVUZETEVWQlFVVXNhMEpCUVZNc1EwRkJReXhyUWtGQlV5eERRVUZETEVsQlFVa3NRMEZCUXl4RlFVRkZMR3RDUVVGVExFTkJRVU1zYTBKQlFWTXNRMEZCUXl4SFFVRkhMRU5CUVVNc1EwRkJReXhEUVVGRE8xRkJSMnBJTEVsQlFVa3NVVUZCVVN4SFFVRkhMRXRCUVVrc1EwRkJRenRSUVVOd1FpeEpRVUZKTEVsQlFVa3NRMEZCUXp0UlFVTlVMRXRCUVVrc1EwRkJReXhKUVVGSkxFZEJRVWM3V1VGRFVpeEpRVUZKTEVOQlFVTXNTVUZCU1N4RlFVRkZPMmRDUVVOUUxFbEJRVWtzU1VGQlNTeEhRVUZITEZOQlFWTXNRMEZCUXl4SlFVRkpMRVZCUVVVN2NVSkJRM1JDTEU5QlFVOHNRMEZCUXl4bFFVRk5MRU5CUVVNc1ZVRkJWU3hEUVVGRE8zRkNRVU14UWl4RlFVRkZMRU5CUVVNc2FVSkJRV2xDTEVWQlFVVXNaVUZCVFN4RFFVRkRMRk5CUVZNc1EwRkJRenR4UWtGRGRrTXNSVUZCUlN4RFFVRkRMR2RDUVVGblFpeEZRVUZGTEZWQlFVRXNRMEZCUXp0dlFrRkRia0lzWlVGQlRTeERRVUZETEVsQlFVa3NRMEZCVFN4RFFVRkRMRVZCUVVVc1UwRkJVeXhEUVVGRExFdEJRVXNzUTBGQlF5eERRVUZETzI5Q1FVTnlReXhSUVVGUkxFTkJRVU1zVFVGQlRTeEZRVUZGTEVOQlFVTTdaMEpCUTNSQ0xFTkJRVU1zUTBGQlF6dHhRa0ZEUkN4RlFVRkZMRU5CUVVNc1pVRkJaU3hGUVVGRkxHVkJRVTBzUTBGQlF5eFBRVUZQTEVOQlFVTXNRMEZCUXp0aFFVTTFRenRaUVVWRUxFbEJRVWtzUTBGQlF5eFRRVUZUTEVOQlFVTXNUVUZCVFR0blFrRkJSU3hQUVVGUExFbEJRVWtzUTBGQlF6dFpRVXR1UXl4VFFVRlRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zU1VGQlNTeERRVUZETEVsQlFVa3NRMEZCUXl4RFFVRkRPMUZCUXpWQ0xFTkJRVU1zUTBGQlFUczdTVUZEVEN4RFFVRkRPMGxCZWtORUxITkRRVUZQTEVkQlFWQXNWVUZCVVN4RFFVRlJPMUZCUTFvc1NVRkJTU3hQUVVGUExFZEJRVWNzUlVGQlJTeEpRVUZKTEVWQlFVVXNhMEpCUVZNc1EwRkJReXhEUVVGRExFTkJRVU1zU1VGQlNTeERRVUZETEVWQlFVVXNTMEZCU3l4RlFVRkZMRU5CUVVNc1EwRkJReXhMUVVGTExFVkJRVVVzVFVGQlRTeEZRVUZGTEVOQlFVTXNRMEZCUXl4TlFVRk5MRVZCUVVVc1EwRkJRenRSUVVjMVJTeEpRVUZKTEVOQlFVTXNTMEZCU3l4RFFVRkRMRWxCUVVrc1EwRkJReXhQUVVGUExFTkJRVU1zU1VGQlNTeEZRVUZQTEU5QlFVOHNRMEZCUXl4RFFVRkRPMGxCUTJoRUxFTkJRVU03U1VGSFJDeHRRMEZCU1N4SFFVRktPMUZCUVVFc2FVSkJSVU03VVVGRVJ5eEpRVUZKTEVOQlFVTXNSMEZCUnl4SlFVRkpMRU5CUVVNc1UwRkJVeXhEUVVGRExFdEJRVXNzUTBGQlF5eGpRVUZOTEU5QlFVRXNhVUpCUVUwc1NVRkJTU3haUVVGRkxFbEJRVWtzUTBGQlF5eERRVUZETEVsQlFVa3NSVUZCUlN4RlFVRjRRaXhEUVVGM1FpeERRVUZETEVOQlFVTTdTVUZEYWtVc1EwRkJRenRKUVd0RFJDeHBRMEZCUlN4SFFVRkdMRlZCUVVjc1UwRkJOa0lzUlVGQlJTeFJRVUZ2UWp0UlFVTnNSQ3hKUVVGSkxFOUJRVThzVTBGQlV5eExRVUZMTEZGQlFWRXNSVUZCUlR0WlFVTXZRaXhKUVVGSkxFTkJRVU1zUzBGQlN5eERRVUZETEVWQlFVVXNRMEZCUXl4VFFVRlRMRVZCUVVVc1VVRkJVU3hEUVVGRExFTkJRVU03VTBGRGRFTTdZVUZCVFR0WlFVTklMRWxCUVVrc1EwRkJReXhMUVVGTExFTkJRVU1zUlVGQlJTeERRVUZETEd0Q1FVRlRMRU5CUVVNc1UwRkJVeXhEUVVGRExFVkJRVVVzVVVGQlVTeERRVUZETEVOQlFVTTdVMEZEYWtRN1VVRkRSQ3hQUVVGUExFbEJRVWtzUTBGQlF6dEpRVU5vUWl4RFFVRkRPMGxCUTB3c01rSkJRVU03UVVGQlJDeERRVUZETEVGQmRFUkVMRU5CUVRCRExHVkJRVTBzUjBGelJDOURPMEZCZEVSWkxHOUVRVUZ2UWlKOSIsIlwidXNlIHN0cmljdFwiO1xyXG5PYmplY3QuZGVmaW5lUHJvcGVydHkoZXhwb3J0cywgXCJfX2VzTW9kdWxlXCIsIHsgdmFsdWU6IHRydWUgfSk7XHJcbnZhciBMb2NrcyA9IChmdW5jdGlvbiAoKSB7XHJcbiAgICBmdW5jdGlvbiBMb2NrcygpIHtcclxuICAgICAgICB0aGlzLmxvY2tzID0ge307XHJcbiAgICB9XHJcbiAgICBMb2Nrcy5wcm90b3R5cGUuYWRkID0gZnVuY3Rpb24gKGlkLCB4KSB7XHJcbiAgICAgICAgdGhpcy5sb2Nrc1tpZF0gPSB4O1xyXG4gICAgfTtcclxuICAgIExvY2tzLnByb3RvdHlwZS5jbGVhciA9IGZ1bmN0aW9uICgpIHtcclxuICAgICAgICB0aGlzLmxvY2tzID0ge307XHJcbiAgICB9O1xyXG4gICAgTG9ja3MucHJvdG90eXBlLmlzRW1wdHkgPSBmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgZm9yICh2YXIgbCBpbiB0aGlzLmxvY2tzKVxyXG4gICAgICAgICAgICByZXR1cm4gZmFsc2U7XHJcbiAgICAgICAgcmV0dXJuIHRydWU7XHJcbiAgICB9O1xyXG4gICAgTG9ja3MucHJvdG90eXBlLmFwcGx5ID0gZnVuY3Rpb24gKGYpIHtcclxuICAgICAgICBmb3IgKHZhciBsIGluIHRoaXMubG9ja3MpIHtcclxuICAgICAgICAgICAgZihOdW1iZXIobCksIHRoaXMubG9ja3NbbF0pO1xyXG4gICAgICAgIH1cclxuICAgIH07XHJcbiAgICByZXR1cm4gTG9ja3M7XHJcbn0oKSk7XHJcbmV4cG9ydHMuTG9ja3MgPSBMb2NrcztcclxudmFyIERlc2NlbnQgPSAoZnVuY3Rpb24gKCkge1xyXG4gICAgZnVuY3Rpb24gRGVzY2VudCh4LCBELCBHKSB7XHJcbiAgICAgICAgaWYgKEcgPT09IHZvaWQgMCkgeyBHID0gbnVsbDsgfVxyXG4gICAgICAgIHRoaXMuRCA9IEQ7XHJcbiAgICAgICAgdGhpcy5HID0gRztcclxuICAgICAgICB0aGlzLnRocmVzaG9sZCA9IDAuMDAwMTtcclxuICAgICAgICB0aGlzLm51bUdyaWRTbmFwTm9kZXMgPSAwO1xyXG4gICAgICAgIHRoaXMuc25hcEdyaWRTaXplID0gMTAwO1xyXG4gICAgICAgIHRoaXMuc25hcFN0cmVuZ3RoID0gMTAwMDtcclxuICAgICAgICB0aGlzLnNjYWxlU25hcEJ5TWF4SCA9IGZhbHNlO1xyXG4gICAgICAgIHRoaXMucmFuZG9tID0gbmV3IFBzZXVkb1JhbmRvbSgpO1xyXG4gICAgICAgIHRoaXMucHJvamVjdCA9IG51bGw7XHJcbiAgICAgICAgdGhpcy54ID0geDtcclxuICAgICAgICB0aGlzLmsgPSB4Lmxlbmd0aDtcclxuICAgICAgICB2YXIgbiA9IHRoaXMubiA9IHhbMF0ubGVuZ3RoO1xyXG4gICAgICAgIHRoaXMuSCA9IG5ldyBBcnJheSh0aGlzLmspO1xyXG4gICAgICAgIHRoaXMuZyA9IG5ldyBBcnJheSh0aGlzLmspO1xyXG4gICAgICAgIHRoaXMuSGQgPSBuZXcgQXJyYXkodGhpcy5rKTtcclxuICAgICAgICB0aGlzLmEgPSBuZXcgQXJyYXkodGhpcy5rKTtcclxuICAgICAgICB0aGlzLmIgPSBuZXcgQXJyYXkodGhpcy5rKTtcclxuICAgICAgICB0aGlzLmMgPSBuZXcgQXJyYXkodGhpcy5rKTtcclxuICAgICAgICB0aGlzLmQgPSBuZXcgQXJyYXkodGhpcy5rKTtcclxuICAgICAgICB0aGlzLmUgPSBuZXcgQXJyYXkodGhpcy5rKTtcclxuICAgICAgICB0aGlzLmlhID0gbmV3IEFycmF5KHRoaXMuayk7XHJcbiAgICAgICAgdGhpcy5pYiA9IG5ldyBBcnJheSh0aGlzLmspO1xyXG4gICAgICAgIHRoaXMueHRtcCA9IG5ldyBBcnJheSh0aGlzLmspO1xyXG4gICAgICAgIHRoaXMubG9ja3MgPSBuZXcgTG9ja3MoKTtcclxuICAgICAgICB0aGlzLm1pbkQgPSBOdW1iZXIuTUFYX1ZBTFVFO1xyXG4gICAgICAgIHZhciBpID0gbiwgajtcclxuICAgICAgICB3aGlsZSAoaS0tKSB7XHJcbiAgICAgICAgICAgIGogPSBuO1xyXG4gICAgICAgICAgICB3aGlsZSAoLS1qID4gaSkge1xyXG4gICAgICAgICAgICAgICAgdmFyIGQgPSBEW2ldW2pdO1xyXG4gICAgICAgICAgICAgICAgaWYgKGQgPiAwICYmIGQgPCB0aGlzLm1pbkQpIHtcclxuICAgICAgICAgICAgICAgICAgICB0aGlzLm1pbkQgPSBkO1xyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG4gICAgICAgIGlmICh0aGlzLm1pbkQgPT09IE51bWJlci5NQVhfVkFMVUUpXHJcbiAgICAgICAgICAgIHRoaXMubWluRCA9IDE7XHJcbiAgICAgICAgaSA9IHRoaXMuaztcclxuICAgICAgICB3aGlsZSAoaS0tKSB7XHJcbiAgICAgICAgICAgIHRoaXMuZ1tpXSA9IG5ldyBBcnJheShuKTtcclxuICAgICAgICAgICAgdGhpcy5IW2ldID0gbmV3IEFycmF5KG4pO1xyXG4gICAgICAgICAgICBqID0gbjtcclxuICAgICAgICAgICAgd2hpbGUgKGotLSkge1xyXG4gICAgICAgICAgICAgICAgdGhpcy5IW2ldW2pdID0gbmV3IEFycmF5KG4pO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIHRoaXMuSGRbaV0gPSBuZXcgQXJyYXkobik7XHJcbiAgICAgICAgICAgIHRoaXMuYVtpXSA9IG5ldyBBcnJheShuKTtcclxuICAgICAgICAgICAgdGhpcy5iW2ldID0gbmV3IEFycmF5KG4pO1xyXG4gICAgICAgICAgICB0aGlzLmNbaV0gPSBuZXcgQXJyYXkobik7XHJcbiAgICAgICAgICAgIHRoaXMuZFtpXSA9IG5ldyBBcnJheShuKTtcclxuICAgICAgICAgICAgdGhpcy5lW2ldID0gbmV3IEFycmF5KG4pO1xyXG4gICAgICAgICAgICB0aGlzLmlhW2ldID0gbmV3IEFycmF5KG4pO1xyXG4gICAgICAgICAgICB0aGlzLmliW2ldID0gbmV3IEFycmF5KG4pO1xyXG4gICAgICAgICAgICB0aGlzLnh0bXBbaV0gPSBuZXcgQXJyYXkobik7XHJcbiAgICAgICAgfVxyXG4gICAgfVxyXG4gICAgRGVzY2VudC5jcmVhdGVTcXVhcmVNYXRyaXggPSBmdW5jdGlvbiAobiwgZikge1xyXG4gICAgICAgIHZhciBNID0gbmV3IEFycmF5KG4pO1xyXG4gICAgICAgIGZvciAodmFyIGkgPSAwOyBpIDwgbjsgKytpKSB7XHJcbiAgICAgICAgICAgIE1baV0gPSBuZXcgQXJyYXkobik7XHJcbiAgICAgICAgICAgIGZvciAodmFyIGogPSAwOyBqIDwgbjsgKytqKSB7XHJcbiAgICAgICAgICAgICAgICBNW2ldW2pdID0gZihpLCBqKTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH1cclxuICAgICAgICByZXR1cm4gTTtcclxuICAgIH07XHJcbiAgICBEZXNjZW50LnByb3RvdHlwZS5vZmZzZXREaXIgPSBmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgdmFyIF90aGlzID0gdGhpcztcclxuICAgICAgICB2YXIgdSA9IG5ldyBBcnJheSh0aGlzLmspO1xyXG4gICAgICAgIHZhciBsID0gMDtcclxuICAgICAgICBmb3IgKHZhciBpID0gMDsgaSA8IHRoaXMuazsgKytpKSB7XHJcbiAgICAgICAgICAgIHZhciB4ID0gdVtpXSA9IHRoaXMucmFuZG9tLmdldE5leHRCZXR3ZWVuKDAuMDEsIDEpIC0gMC41O1xyXG4gICAgICAgICAgICBsICs9IHggKiB4O1xyXG4gICAgICAgIH1cclxuICAgICAgICBsID0gTWF0aC5zcXJ0KGwpO1xyXG4gICAgICAgIHJldHVybiB1Lm1hcChmdW5jdGlvbiAoeCkgeyByZXR1cm4geCAqPSBfdGhpcy5taW5EIC8gbDsgfSk7XHJcbiAgICB9O1xyXG4gICAgRGVzY2VudC5wcm90b3R5cGUuY29tcHV0ZURlcml2YXRpdmVzID0gZnVuY3Rpb24gKHgpIHtcclxuICAgICAgICB2YXIgX3RoaXMgPSB0aGlzO1xyXG4gICAgICAgIHZhciBuID0gdGhpcy5uO1xyXG4gICAgICAgIGlmIChuIDwgMSlcclxuICAgICAgICAgICAgcmV0dXJuO1xyXG4gICAgICAgIHZhciBpO1xyXG4gICAgICAgIHZhciBkID0gbmV3IEFycmF5KHRoaXMuayk7XHJcbiAgICAgICAgdmFyIGQyID0gbmV3IEFycmF5KHRoaXMuayk7XHJcbiAgICAgICAgdmFyIEh1dSA9IG5ldyBBcnJheSh0aGlzLmspO1xyXG4gICAgICAgIHZhciBtYXhIID0gMDtcclxuICAgICAgICBmb3IgKHZhciB1ID0gMDsgdSA8IG47ICsrdSkge1xyXG4gICAgICAgICAgICBmb3IgKGkgPSAwOyBpIDwgdGhpcy5rOyArK2kpXHJcbiAgICAgICAgICAgICAgICBIdXVbaV0gPSB0aGlzLmdbaV1bdV0gPSAwO1xyXG4gICAgICAgICAgICBmb3IgKHZhciB2ID0gMDsgdiA8IG47ICsrdikge1xyXG4gICAgICAgICAgICAgICAgaWYgKHUgPT09IHYpXHJcbiAgICAgICAgICAgICAgICAgICAgY29udGludWU7XHJcbiAgICAgICAgICAgICAgICB2YXIgbWF4RGlzcGxhY2VzID0gbjtcclxuICAgICAgICAgICAgICAgIHdoaWxlIChtYXhEaXNwbGFjZXMtLSkge1xyXG4gICAgICAgICAgICAgICAgICAgIHZhciBzZDIgPSAwO1xyXG4gICAgICAgICAgICAgICAgICAgIGZvciAoaSA9IDA7IGkgPCB0aGlzLms7ICsraSkge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICB2YXIgZHggPSBkW2ldID0geFtpXVt1XSAtIHhbaV1bdl07XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIHNkMiArPSBkMltpXSA9IGR4ICogZHg7XHJcbiAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgIGlmIChzZDIgPiAxZS05KVxyXG4gICAgICAgICAgICAgICAgICAgICAgICBicmVhaztcclxuICAgICAgICAgICAgICAgICAgICB2YXIgcmQgPSB0aGlzLm9mZnNldERpcigpO1xyXG4gICAgICAgICAgICAgICAgICAgIGZvciAoaSA9IDA7IGkgPCB0aGlzLms7ICsraSlcclxuICAgICAgICAgICAgICAgICAgICAgICAgeFtpXVt2XSArPSByZFtpXTtcclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIHZhciBsID0gTWF0aC5zcXJ0KHNkMik7XHJcbiAgICAgICAgICAgICAgICB2YXIgRCA9IHRoaXMuRFt1XVt2XTtcclxuICAgICAgICAgICAgICAgIHZhciB3ZWlnaHQgPSB0aGlzLkcgIT0gbnVsbCA/IHRoaXMuR1t1XVt2XSA6IDE7XHJcbiAgICAgICAgICAgICAgICBpZiAod2VpZ2h0ID4gMSAmJiBsID4gRCB8fCAhaXNGaW5pdGUoRCkpIHtcclxuICAgICAgICAgICAgICAgICAgICBmb3IgKGkgPSAwOyBpIDwgdGhpcy5rOyArK2kpXHJcbiAgICAgICAgICAgICAgICAgICAgICAgIHRoaXMuSFtpXVt1XVt2XSA9IDA7XHJcbiAgICAgICAgICAgICAgICAgICAgY29udGludWU7XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICBpZiAod2VpZ2h0ID4gMSkge1xyXG4gICAgICAgICAgICAgICAgICAgIHdlaWdodCA9IDE7XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICB2YXIgRDIgPSBEICogRDtcclxuICAgICAgICAgICAgICAgIHZhciBncyA9IDIgKiB3ZWlnaHQgKiAobCAtIEQpIC8gKEQyICogbCk7XHJcbiAgICAgICAgICAgICAgICB2YXIgbDMgPSBsICogbCAqIGw7XHJcbiAgICAgICAgICAgICAgICB2YXIgaHMgPSAyICogLXdlaWdodCAvIChEMiAqIGwzKTtcclxuICAgICAgICAgICAgICAgIGlmICghaXNGaW5pdGUoZ3MpKVxyXG4gICAgICAgICAgICAgICAgICAgIGNvbnNvbGUubG9nKGdzKTtcclxuICAgICAgICAgICAgICAgIGZvciAoaSA9IDA7IGkgPCB0aGlzLms7ICsraSkge1xyXG4gICAgICAgICAgICAgICAgICAgIHRoaXMuZ1tpXVt1XSArPSBkW2ldICogZ3M7XHJcbiAgICAgICAgICAgICAgICAgICAgSHV1W2ldIC09IHRoaXMuSFtpXVt1XVt2XSA9IGhzICogKGwzICsgRCAqIChkMltpXSAtIHNkMikgKyBsICogc2QyKTtcclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICBmb3IgKGkgPSAwOyBpIDwgdGhpcy5rOyArK2kpXHJcbiAgICAgICAgICAgICAgICBtYXhIID0gTWF0aC5tYXgobWF4SCwgdGhpcy5IW2ldW3VdW3VdID0gSHV1W2ldKTtcclxuICAgICAgICB9XHJcbiAgICAgICAgdmFyIHIgPSB0aGlzLnNuYXBHcmlkU2l6ZSAvIDI7XHJcbiAgICAgICAgdmFyIGcgPSB0aGlzLnNuYXBHcmlkU2l6ZTtcclxuICAgICAgICB2YXIgdyA9IHRoaXMuc25hcFN0cmVuZ3RoO1xyXG4gICAgICAgIHZhciBrID0gdyAvIChyICogcik7XHJcbiAgICAgICAgdmFyIG51bU5vZGVzID0gdGhpcy5udW1HcmlkU25hcE5vZGVzO1xyXG4gICAgICAgIGZvciAodmFyIHUgPSAwOyB1IDwgbnVtTm9kZXM7ICsrdSkge1xyXG4gICAgICAgICAgICBmb3IgKGkgPSAwOyBpIDwgdGhpcy5rOyArK2kpIHtcclxuICAgICAgICAgICAgICAgIHZhciB4aXUgPSB0aGlzLnhbaV1bdV07XHJcbiAgICAgICAgICAgICAgICB2YXIgbSA9IHhpdSAvIGc7XHJcbiAgICAgICAgICAgICAgICB2YXIgZiA9IG0gJSAxO1xyXG4gICAgICAgICAgICAgICAgdmFyIHEgPSBtIC0gZjtcclxuICAgICAgICAgICAgICAgIHZhciBhID0gTWF0aC5hYnMoZik7XHJcbiAgICAgICAgICAgICAgICB2YXIgZHggPSAoYSA8PSAwLjUpID8geGl1IC0gcSAqIGcgOlxyXG4gICAgICAgICAgICAgICAgICAgICh4aXUgPiAwKSA/IHhpdSAtIChxICsgMSkgKiBnIDogeGl1IC0gKHEgLSAxKSAqIGc7XHJcbiAgICAgICAgICAgICAgICBpZiAoLXIgPCBkeCAmJiBkeCA8PSByKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKHRoaXMuc2NhbGVTbmFwQnlNYXhIKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIHRoaXMuZ1tpXVt1XSArPSBtYXhIICogayAqIGR4O1xyXG4gICAgICAgICAgICAgICAgICAgICAgICB0aGlzLkhbaV1bdV1bdV0gKz0gbWF4SCAqIGs7XHJcbiAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgIGVsc2Uge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICB0aGlzLmdbaV1bdV0gKz0gayAqIGR4O1xyXG4gICAgICAgICAgICAgICAgICAgICAgICB0aGlzLkhbaV1bdV1bdV0gKz0gaztcclxuICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICAgICAgaWYgKCF0aGlzLmxvY2tzLmlzRW1wdHkoKSkge1xyXG4gICAgICAgICAgICB0aGlzLmxvY2tzLmFwcGx5KGZ1bmN0aW9uICh1LCBwKSB7XHJcbiAgICAgICAgICAgICAgICBmb3IgKGkgPSAwOyBpIDwgX3RoaXMuazsgKytpKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgX3RoaXMuSFtpXVt1XVt1XSArPSBtYXhIO1xyXG4gICAgICAgICAgICAgICAgICAgIF90aGlzLmdbaV1bdV0gLT0gbWF4SCAqIChwW2ldIC0geFtpXVt1XSk7XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIH0pO1xyXG4gICAgICAgIH1cclxuICAgIH07XHJcbiAgICBEZXNjZW50LmRvdFByb2QgPSBmdW5jdGlvbiAoYSwgYikge1xyXG4gICAgICAgIHZhciB4ID0gMCwgaSA9IGEubGVuZ3RoO1xyXG4gICAgICAgIHdoaWxlIChpLS0pXHJcbiAgICAgICAgICAgIHggKz0gYVtpXSAqIGJbaV07XHJcbiAgICAgICAgcmV0dXJuIHg7XHJcbiAgICB9O1xyXG4gICAgRGVzY2VudC5yaWdodE11bHRpcGx5ID0gZnVuY3Rpb24gKG0sIHYsIHIpIHtcclxuICAgICAgICB2YXIgaSA9IG0ubGVuZ3RoO1xyXG4gICAgICAgIHdoaWxlIChpLS0pXHJcbiAgICAgICAgICAgIHJbaV0gPSBEZXNjZW50LmRvdFByb2QobVtpXSwgdik7XHJcbiAgICB9O1xyXG4gICAgRGVzY2VudC5wcm90b3R5cGUuY29tcHV0ZVN0ZXBTaXplID0gZnVuY3Rpb24gKGQpIHtcclxuICAgICAgICB2YXIgbnVtZXJhdG9yID0gMCwgZGVub21pbmF0b3IgPSAwO1xyXG4gICAgICAgIGZvciAodmFyIGkgPSAwOyBpIDwgdGhpcy5rOyArK2kpIHtcclxuICAgICAgICAgICAgbnVtZXJhdG9yICs9IERlc2NlbnQuZG90UHJvZCh0aGlzLmdbaV0sIGRbaV0pO1xyXG4gICAgICAgICAgICBEZXNjZW50LnJpZ2h0TXVsdGlwbHkodGhpcy5IW2ldLCBkW2ldLCB0aGlzLkhkW2ldKTtcclxuICAgICAgICAgICAgZGVub21pbmF0b3IgKz0gRGVzY2VudC5kb3RQcm9kKGRbaV0sIHRoaXMuSGRbaV0pO1xyXG4gICAgICAgIH1cclxuICAgICAgICBpZiAoZGVub21pbmF0b3IgPT09IDAgfHwgIWlzRmluaXRlKGRlbm9taW5hdG9yKSlcclxuICAgICAgICAgICAgcmV0dXJuIDA7XHJcbiAgICAgICAgcmV0dXJuIDEgKiBudW1lcmF0b3IgLyBkZW5vbWluYXRvcjtcclxuICAgIH07XHJcbiAgICBEZXNjZW50LnByb3RvdHlwZS5yZWR1Y2VTdHJlc3MgPSBmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgdGhpcy5jb21wdXRlRGVyaXZhdGl2ZXModGhpcy54KTtcclxuICAgICAgICB2YXIgYWxwaGEgPSB0aGlzLmNvbXB1dGVTdGVwU2l6ZSh0aGlzLmcpO1xyXG4gICAgICAgIGZvciAodmFyIGkgPSAwOyBpIDwgdGhpcy5rOyArK2kpIHtcclxuICAgICAgICAgICAgdGhpcy50YWtlRGVzY2VudFN0ZXAodGhpcy54W2ldLCB0aGlzLmdbaV0sIGFscGhhKTtcclxuICAgICAgICB9XHJcbiAgICAgICAgcmV0dXJuIHRoaXMuY29tcHV0ZVN0cmVzcygpO1xyXG4gICAgfTtcclxuICAgIERlc2NlbnQuY29weSA9IGZ1bmN0aW9uIChhLCBiKSB7XHJcbiAgICAgICAgdmFyIG0gPSBhLmxlbmd0aCwgbiA9IGJbMF0ubGVuZ3RoO1xyXG4gICAgICAgIGZvciAodmFyIGkgPSAwOyBpIDwgbTsgKytpKSB7XHJcbiAgICAgICAgICAgIGZvciAodmFyIGogPSAwOyBqIDwgbjsgKytqKSB7XHJcbiAgICAgICAgICAgICAgICBiW2ldW2pdID0gYVtpXVtqXTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH1cclxuICAgIH07XHJcbiAgICBEZXNjZW50LnByb3RvdHlwZS5zdGVwQW5kUHJvamVjdCA9IGZ1bmN0aW9uICh4MCwgciwgZCwgc3RlcFNpemUpIHtcclxuICAgICAgICBEZXNjZW50LmNvcHkoeDAsIHIpO1xyXG4gICAgICAgIHRoaXMudGFrZURlc2NlbnRTdGVwKHJbMF0sIGRbMF0sIHN0ZXBTaXplKTtcclxuICAgICAgICBpZiAodGhpcy5wcm9qZWN0KVxyXG4gICAgICAgICAgICB0aGlzLnByb2plY3RbMF0oeDBbMF0sIHgwWzFdLCByWzBdKTtcclxuICAgICAgICB0aGlzLnRha2VEZXNjZW50U3RlcChyWzFdLCBkWzFdLCBzdGVwU2l6ZSk7XHJcbiAgICAgICAgaWYgKHRoaXMucHJvamVjdClcclxuICAgICAgICAgICAgdGhpcy5wcm9qZWN0WzFdKHJbMF0sIHgwWzFdLCByWzFdKTtcclxuICAgICAgICBmb3IgKHZhciBpID0gMjsgaSA8IHRoaXMuazsgaSsrKVxyXG4gICAgICAgICAgICB0aGlzLnRha2VEZXNjZW50U3RlcChyW2ldLCBkW2ldLCBzdGVwU2l6ZSk7XHJcbiAgICB9O1xyXG4gICAgRGVzY2VudC5tQXBwbHkgPSBmdW5jdGlvbiAobSwgbiwgZikge1xyXG4gICAgICAgIHZhciBpID0gbTtcclxuICAgICAgICB3aGlsZSAoaS0tID4gMCkge1xyXG4gICAgICAgICAgICB2YXIgaiA9IG47XHJcbiAgICAgICAgICAgIHdoaWxlIChqLS0gPiAwKVxyXG4gICAgICAgICAgICAgICAgZihpLCBqKTtcclxuICAgICAgICB9XHJcbiAgICB9O1xyXG4gICAgRGVzY2VudC5wcm90b3R5cGUubWF0cml4QXBwbHkgPSBmdW5jdGlvbiAoZikge1xyXG4gICAgICAgIERlc2NlbnQubUFwcGx5KHRoaXMuaywgdGhpcy5uLCBmKTtcclxuICAgIH07XHJcbiAgICBEZXNjZW50LnByb3RvdHlwZS5jb21wdXRlTmV4dFBvc2l0aW9uID0gZnVuY3Rpb24gKHgwLCByKSB7XHJcbiAgICAgICAgdmFyIF90aGlzID0gdGhpcztcclxuICAgICAgICB0aGlzLmNvbXB1dGVEZXJpdmF0aXZlcyh4MCk7XHJcbiAgICAgICAgdmFyIGFscGhhID0gdGhpcy5jb21wdXRlU3RlcFNpemUodGhpcy5nKTtcclxuICAgICAgICB0aGlzLnN0ZXBBbmRQcm9qZWN0KHgwLCByLCB0aGlzLmcsIGFscGhhKTtcclxuICAgICAgICBpZiAodGhpcy5wcm9qZWN0KSB7XHJcbiAgICAgICAgICAgIHRoaXMubWF0cml4QXBwbHkoZnVuY3Rpb24gKGksIGopIHsgcmV0dXJuIF90aGlzLmVbaV1bal0gPSB4MFtpXVtqXSAtIHJbaV1bal07IH0pO1xyXG4gICAgICAgICAgICB2YXIgYmV0YSA9IHRoaXMuY29tcHV0ZVN0ZXBTaXplKHRoaXMuZSk7XHJcbiAgICAgICAgICAgIGJldGEgPSBNYXRoLm1heCgwLjIsIE1hdGgubWluKGJldGEsIDEpKTtcclxuICAgICAgICAgICAgdGhpcy5zdGVwQW5kUHJvamVjdCh4MCwgciwgdGhpcy5lLCBiZXRhKTtcclxuICAgICAgICB9XHJcbiAgICB9O1xyXG4gICAgRGVzY2VudC5wcm90b3R5cGUucnVuID0gZnVuY3Rpb24gKGl0ZXJhdGlvbnMpIHtcclxuICAgICAgICB2YXIgc3RyZXNzID0gTnVtYmVyLk1BWF9WQUxVRSwgY29udmVyZ2VkID0gZmFsc2U7XHJcbiAgICAgICAgd2hpbGUgKCFjb252ZXJnZWQgJiYgaXRlcmF0aW9ucy0tID4gMCkge1xyXG4gICAgICAgICAgICB2YXIgcyA9IHRoaXMucnVuZ2VLdXR0YSgpO1xyXG4gICAgICAgICAgICBjb252ZXJnZWQgPSBNYXRoLmFicyhzdHJlc3MgLyBzIC0gMSkgPCB0aGlzLnRocmVzaG9sZDtcclxuICAgICAgICAgICAgc3RyZXNzID0gcztcclxuICAgICAgICB9XHJcbiAgICAgICAgcmV0dXJuIHN0cmVzcztcclxuICAgIH07XHJcbiAgICBEZXNjZW50LnByb3RvdHlwZS5ydW5nZUt1dHRhID0gZnVuY3Rpb24gKCkge1xyXG4gICAgICAgIHZhciBfdGhpcyA9IHRoaXM7XHJcbiAgICAgICAgdGhpcy5jb21wdXRlTmV4dFBvc2l0aW9uKHRoaXMueCwgdGhpcy5hKTtcclxuICAgICAgICBEZXNjZW50Lm1pZCh0aGlzLngsIHRoaXMuYSwgdGhpcy5pYSk7XHJcbiAgICAgICAgdGhpcy5jb21wdXRlTmV4dFBvc2l0aW9uKHRoaXMuaWEsIHRoaXMuYik7XHJcbiAgICAgICAgRGVzY2VudC5taWQodGhpcy54LCB0aGlzLmIsIHRoaXMuaWIpO1xyXG4gICAgICAgIHRoaXMuY29tcHV0ZU5leHRQb3NpdGlvbih0aGlzLmliLCB0aGlzLmMpO1xyXG4gICAgICAgIHRoaXMuY29tcHV0ZU5leHRQb3NpdGlvbih0aGlzLmMsIHRoaXMuZCk7XHJcbiAgICAgICAgdmFyIGRpc3AgPSAwO1xyXG4gICAgICAgIHRoaXMubWF0cml4QXBwbHkoZnVuY3Rpb24gKGksIGopIHtcclxuICAgICAgICAgICAgdmFyIHggPSAoX3RoaXMuYVtpXVtqXSArIDIuMCAqIF90aGlzLmJbaV1bal0gKyAyLjAgKiBfdGhpcy5jW2ldW2pdICsgX3RoaXMuZFtpXVtqXSkgLyA2LjAsIGQgPSBfdGhpcy54W2ldW2pdIC0geDtcclxuICAgICAgICAgICAgZGlzcCArPSBkICogZDtcclxuICAgICAgICAgICAgX3RoaXMueFtpXVtqXSA9IHg7XHJcbiAgICAgICAgfSk7XHJcbiAgICAgICAgcmV0dXJuIGRpc3A7XHJcbiAgICB9O1xyXG4gICAgRGVzY2VudC5taWQgPSBmdW5jdGlvbiAoYSwgYiwgbSkge1xyXG4gICAgICAgIERlc2NlbnQubUFwcGx5KGEubGVuZ3RoLCBhWzBdLmxlbmd0aCwgZnVuY3Rpb24gKGksIGopIHtcclxuICAgICAgICAgICAgcmV0dXJuIG1baV1bal0gPSBhW2ldW2pdICsgKGJbaV1bal0gLSBhW2ldW2pdKSAvIDIuMDtcclxuICAgICAgICB9KTtcclxuICAgIH07XHJcbiAgICBEZXNjZW50LnByb3RvdHlwZS50YWtlRGVzY2VudFN0ZXAgPSBmdW5jdGlvbiAoeCwgZCwgc3RlcFNpemUpIHtcclxuICAgICAgICBmb3IgKHZhciBpID0gMDsgaSA8IHRoaXMubjsgKytpKSB7XHJcbiAgICAgICAgICAgIHhbaV0gPSB4W2ldIC0gc3RlcFNpemUgKiBkW2ldO1xyXG4gICAgICAgIH1cclxuICAgIH07XHJcbiAgICBEZXNjZW50LnByb3RvdHlwZS5jb21wdXRlU3RyZXNzID0gZnVuY3Rpb24gKCkge1xyXG4gICAgICAgIHZhciBzdHJlc3MgPSAwO1xyXG4gICAgICAgIGZvciAodmFyIHUgPSAwLCBuTWludXMxID0gdGhpcy5uIC0gMTsgdSA8IG5NaW51czE7ICsrdSkge1xyXG4gICAgICAgICAgICBmb3IgKHZhciB2ID0gdSArIDEsIG4gPSB0aGlzLm47IHYgPCBuOyArK3YpIHtcclxuICAgICAgICAgICAgICAgIHZhciBsID0gMDtcclxuICAgICAgICAgICAgICAgIGZvciAodmFyIGkgPSAwOyBpIDwgdGhpcy5rOyArK2kpIHtcclxuICAgICAgICAgICAgICAgICAgICB2YXIgZHggPSB0aGlzLnhbaV1bdV0gLSB0aGlzLnhbaV1bdl07XHJcbiAgICAgICAgICAgICAgICAgICAgbCArPSBkeCAqIGR4O1xyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgbCA9IE1hdGguc3FydChsKTtcclxuICAgICAgICAgICAgICAgIHZhciBkID0gdGhpcy5EW3VdW3ZdO1xyXG4gICAgICAgICAgICAgICAgaWYgKCFpc0Zpbml0ZShkKSlcclxuICAgICAgICAgICAgICAgICAgICBjb250aW51ZTtcclxuICAgICAgICAgICAgICAgIHZhciBybCA9IGQgLSBsO1xyXG4gICAgICAgICAgICAgICAgdmFyIGQyID0gZCAqIGQ7XHJcbiAgICAgICAgICAgICAgICBzdHJlc3MgKz0gcmwgKiBybCAvIGQyO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHJldHVybiBzdHJlc3M7XHJcbiAgICB9O1xyXG4gICAgRGVzY2VudC56ZXJvRGlzdGFuY2UgPSAxZS0xMDtcclxuICAgIHJldHVybiBEZXNjZW50O1xyXG59KCkpO1xyXG5leHBvcnRzLkRlc2NlbnQgPSBEZXNjZW50O1xyXG52YXIgUHNldWRvUmFuZG9tID0gKGZ1bmN0aW9uICgpIHtcclxuICAgIGZ1bmN0aW9uIFBzZXVkb1JhbmRvbShzZWVkKSB7XHJcbiAgICAgICAgaWYgKHNlZWQgPT09IHZvaWQgMCkgeyBzZWVkID0gMTsgfVxyXG4gICAgICAgIHRoaXMuc2VlZCA9IHNlZWQ7XHJcbiAgICAgICAgdGhpcy5hID0gMjE0MDEzO1xyXG4gICAgICAgIHRoaXMuYyA9IDI1MzEwMTE7XHJcbiAgICAgICAgdGhpcy5tID0gMjE0NzQ4MzY0ODtcclxuICAgICAgICB0aGlzLnJhbmdlID0gMzI3Njc7XHJcbiAgICB9XHJcbiAgICBQc2V1ZG9SYW5kb20ucHJvdG90eXBlLmdldE5leHQgPSBmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgdGhpcy5zZWVkID0gKHRoaXMuc2VlZCAqIHRoaXMuYSArIHRoaXMuYykgJSB0aGlzLm07XHJcbiAgICAgICAgcmV0dXJuICh0aGlzLnNlZWQgPj4gMTYpIC8gdGhpcy5yYW5nZTtcclxuICAgIH07XHJcbiAgICBQc2V1ZG9SYW5kb20ucHJvdG90eXBlLmdldE5leHRCZXR3ZWVuID0gZnVuY3Rpb24gKG1pbiwgbWF4KSB7XHJcbiAgICAgICAgcmV0dXJuIG1pbiArIHRoaXMuZ2V0TmV4dCgpICogKG1heCAtIG1pbik7XHJcbiAgICB9O1xyXG4gICAgcmV0dXJuIFBzZXVkb1JhbmRvbTtcclxufSgpKTtcclxuZXhwb3J0cy5Qc2V1ZG9SYW5kb20gPSBQc2V1ZG9SYW5kb207XHJcbi8vIyBzb3VyY2VNYXBwaW5nVVJMPWRhdGE6YXBwbGljYXRpb24vanNvbjtiYXNlNjQsZXlKMlpYSnphVzl1SWpvekxDSm1hV3hsSWpvaVpHVnpZMlZ1ZEM1cWN5SXNJbk52ZFhKalpWSnZiM1FpT2lJaUxDSnpiM1Z5WTJWeklqcGJJaTR1THk0dUwxZGxZa052YkdFdmMzSmpMMlJsYzJObGJuUXVkSE1pWFN3aWJtRnRaWE1pT2x0ZExDSnRZWEJ3YVc1bmN5STZJanM3UVVGSlNUdEpRVUZCTzFGQlEwa3NWVUZCU3l4SFFVRTJRaXhGUVVGRkxFTkJRVU03U1VGdlEzcERMRU5CUVVNN1NVRTNRa2NzYlVKQlFVY3NSMEZCU0N4VlFVRkpMRVZCUVZVc1JVRkJSU3hEUVVGWE8xRkJTWFpDTEVsQlFVa3NRMEZCUXl4TFFVRkxMRU5CUVVNc1JVRkJSU3hEUVVGRExFZEJRVWNzUTBGQlF5eERRVUZETzBsQlEzWkNMRU5CUVVNN1NVRkpSQ3h4UWtGQlN5eEhRVUZNTzFGQlEwa3NTVUZCU1N4RFFVRkRMRXRCUVVzc1IwRkJSeXhGUVVGRkxFTkJRVU03U1VGRGNFSXNRMEZCUXp0SlFVdEVMSFZDUVVGUExFZEJRVkE3VVVGRFNTeExRVUZMTEVsQlFVa3NRMEZCUXl4SlFVRkpMRWxCUVVrc1EwRkJReXhMUVVGTE8xbEJRVVVzVDBGQlR5eExRVUZMTEVOQlFVTTdVVUZEZGtNc1QwRkJUeXhKUVVGSkxFTkJRVU03U1VGRGFFSXNRMEZCUXp0SlFVdEVMSEZDUVVGTExFZEJRVXdzVlVGQlRTeERRVUZ2UXp0UlFVTjBReXhMUVVGTExFbEJRVWtzUTBGQlF5eEpRVUZKTEVsQlFVa3NRMEZCUXl4TFFVRkxMRVZCUVVVN1dVRkRkRUlzUTBGQlF5eERRVUZETEUxQlFVMHNRMEZCUXl4RFFVRkRMRU5CUVVNc1JVRkJSU3hKUVVGSkxFTkJRVU1zUzBGQlN5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNN1UwRkRMMEk3U1VGRFRDeERRVUZETzBsQlEwd3NXVUZCUXp0QlFVRkVMRU5CUVVNc1FVRnlRMFFzU1VGeFEwTTdRVUZ5UTFrc2MwSkJRVXM3UVVGcFJHeENPMGxCTmtSSkxHbENRVUZaTEVOQlFXRXNSVUZCVXl4RFFVRmhMRVZCUVZNc1EwRkJiVUk3VVVGQmJrSXNhMEpCUVVFc1JVRkJRU3hSUVVGdFFqdFJRVUY2UXl4TlFVRkRMRWRCUVVRc1EwRkJReXhEUVVGWk8xRkJRVk1zVFVGQlF5eEhRVUZFTEVOQlFVTXNRMEZCYTBJN1VVRTFSSEJGTEdOQlFWTXNSMEZCVnl4TlFVRk5MRU5CUVVNN1VVRXlRek5DTEhGQ1FVRm5RaXhIUVVGWExFTkJRVU1zUTBGQlF6dFJRVU0zUWl4cFFrRkJXU3hIUVVGWExFZEJRVWNzUTBGQlF6dFJRVU16UWl4cFFrRkJXU3hIUVVGWExFbEJRVWtzUTBGQlF6dFJRVU0xUWl4dlFrRkJaU3hIUVVGWkxFdEJRVXNzUTBGQlF6dFJRVVZvUXl4WFFVRk5MRWRCUVVjc1NVRkJTU3haUVVGWkxFVkJRVVVzUTBGQlF6dFJRVVUzUWl4WlFVRlBMRWRCUVRCRUxFbEJRVWtzUTBGQlF6dFJRVmQ2UlN4SlFVRkpMRU5CUVVNc1EwRkJReXhIUVVGSExFTkJRVU1zUTBGQlF6dFJRVU5ZTEVsQlFVa3NRMEZCUXl4RFFVRkRMRWRCUVVjc1EwRkJReXhEUVVGRExFMUJRVTBzUTBGQlF6dFJRVU5zUWl4SlFVRkpMRU5CUVVNc1IwRkJSeXhKUVVGSkxFTkJRVU1zUTBGQlF5eEhRVUZITEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhOUVVGTkxFTkJRVU03VVVGRE4wSXNTVUZCU1N4RFFVRkRMRU5CUVVNc1IwRkJSeXhKUVVGSkxFdEJRVXNzUTBGQlF5eEpRVUZKTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNN1VVRkRNMElzU1VGQlNTeERRVUZETEVOQlFVTXNSMEZCUnl4SlFVRkpMRXRCUVVzc1EwRkJReXhKUVVGSkxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTTdVVUZETTBJc1NVRkJTU3hEUVVGRExFVkJRVVVzUjBGQlJ5eEpRVUZKTEV0QlFVc3NRMEZCUXl4SlFVRkpMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU03VVVGRE5VSXNTVUZCU1N4RFFVRkRMRU5CUVVNc1IwRkJSeXhKUVVGSkxFdEJRVXNzUTBGQlF5eEpRVUZKTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNN1VVRkRNMElzU1VGQlNTeERRVUZETEVOQlFVTXNSMEZCUnl4SlFVRkpMRXRCUVVzc1EwRkJReXhKUVVGSkxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTTdVVUZETTBJc1NVRkJTU3hEUVVGRExFTkJRVU1zUjBGQlJ5eEpRVUZKTEV0QlFVc3NRMEZCUXl4SlFVRkpMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU03VVVGRE0wSXNTVUZCU1N4RFFVRkRMRU5CUVVNc1IwRkJSeXhKUVVGSkxFdEJRVXNzUTBGQlF5eEpRVUZKTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNN1VVRkRNMElzU1VGQlNTeERRVUZETEVOQlFVTXNSMEZCUnl4SlFVRkpMRXRCUVVzc1EwRkJReXhKUVVGSkxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTTdVVUZETTBJc1NVRkJTU3hEUVVGRExFVkJRVVVzUjBGQlJ5eEpRVUZKTEV0QlFVc3NRMEZCUXl4SlFVRkpMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU03VVVGRE5VSXNTVUZCU1N4RFFVRkRMRVZCUVVVc1IwRkJSeXhKUVVGSkxFdEJRVXNzUTBGQlF5eEpRVUZKTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNN1VVRkROVUlzU1VGQlNTeERRVUZETEVsQlFVa3NSMEZCUnl4SlFVRkpMRXRCUVVzc1EwRkJReXhKUVVGSkxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTTdVVUZET1VJc1NVRkJTU3hEUVVGRExFdEJRVXNzUjBGQlJ5eEpRVUZKTEV0QlFVc3NSVUZCUlN4RFFVRkRPMUZCUTNwQ0xFbEJRVWtzUTBGQlF5eEpRVUZKTEVkQlFVY3NUVUZCVFN4RFFVRkRMRk5CUVZNc1EwRkJRenRSUVVNM1FpeEpRVUZKTEVOQlFVTXNSMEZCUnl4RFFVRkRMRVZCUVVVc1EwRkJReXhEUVVGRE8xRkJRMklzVDBGQlR5eERRVUZETEVWQlFVVXNSVUZCUlR0WlFVTlNMRU5CUVVNc1IwRkJSeXhEUVVGRExFTkJRVU03V1VGRFRpeFBRVUZQTEVWQlFVVXNRMEZCUXl4SFFVRkhMRU5CUVVNc1JVRkJSVHRuUWtGRFdpeEpRVUZKTEVOQlFVTXNSMEZCUnl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTTdaMEpCUTJoQ0xFbEJRVWtzUTBGQlF5eEhRVUZITEVOQlFVTXNTVUZCU1N4RFFVRkRMRWRCUVVjc1NVRkJTU3hEUVVGRExFbEJRVWtzUlVGQlJUdHZRa0ZEZUVJc1NVRkJTU3hEUVVGRExFbEJRVWtzUjBGQlJ5eERRVUZETEVOQlFVTTdhVUpCUTJwQ08yRkJRMG83VTBGRFNqdFJRVU5FTEVsQlFVa3NTVUZCU1N4RFFVRkRMRWxCUVVrc1MwRkJTeXhOUVVGTkxFTkJRVU1zVTBGQlV6dFpRVUZGTEVsQlFVa3NRMEZCUXl4SlFVRkpMRWRCUVVjc1EwRkJReXhEUVVGRE8xRkJRMnhFTEVOQlFVTXNSMEZCUnl4SlFVRkpMRU5CUVVNc1EwRkJReXhEUVVGRE8xRkJRMWdzVDBGQlR5eERRVUZETEVWQlFVVXNSVUZCUlR0WlFVTlNMRWxCUVVrc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVkQlFVY3NTVUZCU1N4TFFVRkxMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU03V1VGRGVrSXNTVUZCU1N4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUjBGQlJ5eEpRVUZKTEV0QlFVc3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJRenRaUVVONlFpeERRVUZETEVkQlFVY3NRMEZCUXl4RFFVRkRPMWxCUTA0c1QwRkJUeXhEUVVGRExFVkJRVVVzUlVGQlJUdG5Ra0ZEVWl4SlFVRkpMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4SFFVRkhMRWxCUVVrc1MwRkJTeXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETzJGQlF5OUNPMWxCUTBRc1NVRkJTU3hEUVVGRExFVkJRVVVzUTBGQlF5eERRVUZETEVOQlFVTXNSMEZCUnl4SlFVRkpMRXRCUVVzc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dFpRVU14UWl4SlFVRkpMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eEhRVUZITEVsQlFVa3NTMEZCU3l4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRE8xbEJRM3BDTEVsQlFVa3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFZEJRVWNzU1VGQlNTeExRVUZMTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNN1dVRkRla0lzU1VGQlNTeERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1IwRkJSeXhKUVVGSkxFdEJRVXNzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXp0WlFVTjZRaXhKUVVGSkxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4SFFVRkhMRWxCUVVrc1MwRkJTeXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETzFsQlEzcENMRWxCUVVrc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVkQlFVY3NTVUZCU1N4TFFVRkxMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU03V1VGRGVrSXNTVUZCU1N4RFFVRkRMRVZCUVVVc1EwRkJReXhEUVVGRExFTkJRVU1zUjBGQlJ5eEpRVUZKTEV0QlFVc3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJRenRaUVVNeFFpeEpRVUZKTEVOQlFVTXNSVUZCUlN4RFFVRkRMRU5CUVVNc1EwRkJReXhIUVVGSExFbEJRVWtzUzBGQlN5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRPMWxCUXpGQ0xFbEJRVWtzUTBGQlF5eEpRVUZKTEVOQlFVTXNRMEZCUXl4RFFVRkRMRWRCUVVjc1NVRkJTU3hMUVVGTExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTTdVMEZETDBJN1NVRkRUQ3hEUVVGRE8wbEJSV0VzTUVKQlFXdENMRWRCUVdoRExGVkJRV2xETEVOQlFWTXNSVUZCUlN4RFFVRnRRenRSUVVNelJTeEpRVUZKTEVOQlFVTXNSMEZCUnl4SlFVRkpMRXRCUVVzc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dFJRVU55UWl4TFFVRkxMRWxCUVVrc1EwRkJReXhIUVVGSExFTkJRVU1zUlVGQlJTeERRVUZETEVkQlFVY3NRMEZCUXl4RlFVRkZMRVZCUVVVc1EwRkJReXhGUVVGRk8xbEJRM2hDTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1IwRkJSeXhKUVVGSkxFdEJRVXNzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXp0WlFVTndRaXhMUVVGTExFbEJRVWtzUTBGQlF5eEhRVUZITEVOQlFVTXNSVUZCUlN4RFFVRkRMRWRCUVVjc1EwRkJReXhGUVVGRkxFVkJRVVVzUTBGQlF5eEZRVUZGTzJkQ1FVTjRRaXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRWRCUVVjc1EwRkJReXhEUVVGRExFTkJRVU1zUlVGQlJTeERRVUZETEVOQlFVTXNRMEZCUXp0aFFVTnlRanRUUVVOS08xRkJRMFFzVDBGQlR5eERRVUZETEVOQlFVTTdTVUZEWWl4RFFVRkRPMGxCUlU4c01rSkJRVk1zUjBGQmFrSTdVVUZCUVN4cFFrRlRRenRSUVZKSExFbEJRVWtzUTBGQlF5eEhRVUZITEVsQlFVa3NTMEZCU3l4RFFVRkRMRWxCUVVrc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dFJRVU14UWl4SlFVRkpMRU5CUVVNc1IwRkJSeXhEUVVGRExFTkJRVU03VVVGRFZpeExRVUZMTEVsQlFVa3NRMEZCUXl4SFFVRkhMRU5CUVVNc1JVRkJSU3hEUVVGRExFZEJRVWNzU1VGQlNTeERRVUZETEVOQlFVTXNSVUZCUlN4RlFVRkZMRU5CUVVNc1JVRkJSVHRaUVVNM1FpeEpRVUZKTEVOQlFVTXNSMEZCUnl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFZEJRVWNzU1VGQlNTeERRVUZETEUxQlFVMHNRMEZCUXl4alFVRmpMRU5CUVVNc1NVRkJTU3hGUVVGRkxFTkJRVU1zUTBGQlF5eEhRVUZITEVkQlFVY3NRMEZCUXp0WlFVTjZSQ3hEUVVGRExFbEJRVWtzUTBGQlF5eEhRVUZITEVOQlFVTXNRMEZCUXp0VFFVTmtPMUZCUTBRc1EwRkJReXhIUVVGSExFbEJRVWtzUTBGQlF5eEpRVUZKTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNN1VVRkRha0lzVDBGQlR5eERRVUZETEVOQlFVTXNSMEZCUnl4RFFVRkRMRlZCUVVFc1EwRkJReXhKUVVGSExFOUJRVUVzUTBGQlF5eEpRVUZKTEV0QlFVa3NRMEZCUXl4SlFVRkpMRWRCUVVjc1EwRkJReXhGUVVGc1FpeERRVUZyUWl4RFFVRkRMRU5CUVVNN1NVRkRla01zUTBGQlF6dEpRVWROTEc5RFFVRnJRaXhIUVVGNlFpeFZRVUV3UWl4RFFVRmhPMUZCUVhaRExHbENRV2RIUXp0UlFTOUdSeXhKUVVGSkxFTkJRVU1zUjBGQlZ5eEpRVUZKTEVOQlFVTXNRMEZCUXl4RFFVRkRPMUZCUTNaQ0xFbEJRVWtzUTBGQlF5eEhRVUZITEVOQlFVTTdXVUZCUlN4UFFVRlBPMUZCUTJ4Q0xFbEJRVWtzUTBGQlV5eERRVUZETzFGQlRXUXNTVUZCU1N4RFFVRkRMRWRCUVdFc1NVRkJTU3hMUVVGTExFTkJRVU1zU1VGQlNTeERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRPMUZCUTNCRExFbEJRVWtzUlVGQlJTeEhRVUZoTEVsQlFVa3NTMEZCU3l4RFFVRkRMRWxCUVVrc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dFJRVU55UXl4SlFVRkpMRWRCUVVjc1IwRkJZU3hKUVVGSkxFdEJRVXNzUTBGQlF5eEpRVUZKTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNN1VVRkRkRU1zU1VGQlNTeEpRVUZKTEVkQlFWY3NRMEZCUXl4RFFVRkRPMUZCUTNKQ0xFdEJRVXNzU1VGQlNTeERRVUZETEVkQlFWY3NRMEZCUXl4RlFVRkZMRU5CUVVNc1IwRkJSeXhEUVVGRExFVkJRVVVzUlVGQlJTeERRVUZETEVWQlFVVTdXVUZEYUVNc1MwRkJTeXhEUVVGRExFZEJRVWNzUTBGQlF5eEZRVUZGTEVOQlFVTXNSMEZCUnl4SlFVRkpMRU5CUVVNc1EwRkJReXhGUVVGRkxFVkJRVVVzUTBGQlF6dG5Ra0ZCUlN4SFFVRkhMRU5CUVVNc1EwRkJReXhEUVVGRExFZEJRVWNzU1VGQlNTeERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUjBGQlJ5eERRVUZETEVOQlFVTTdXVUZEZGtRc1MwRkJTeXhKUVVGSkxFTkJRVU1zUjBGQlJ5eERRVUZETEVWQlFVVXNRMEZCUXl4SFFVRkhMRU5CUVVNc1JVRkJSU3hGUVVGRkxFTkJRVU1zUlVGQlJUdG5Ra0ZEZUVJc1NVRkJTU3hEUVVGRExFdEJRVXNzUTBGQlF6dHZRa0ZCUlN4VFFVRlRPMmRDUVVkMFFpeEpRVUZKTEZsQlFWa3NSMEZCUnl4RFFVRkRMRU5CUVVNN1owSkJRM0pDTEU5QlFVOHNXVUZCV1N4RlFVRkZMRVZCUVVVN2IwSkJRMjVDTEVsQlFVa3NSMEZCUnl4SFFVRkhMRU5CUVVNc1EwRkJRenR2UWtGRFdpeExRVUZMTEVOQlFVTXNSMEZCUnl4RFFVRkRMRVZCUVVVc1EwRkJReXhIUVVGSExFbEJRVWtzUTBGQlF5eERRVUZETEVWQlFVVXNSVUZCUlN4RFFVRkRMRVZCUVVVN2QwSkJRM3BDTEVsQlFVa3NSVUZCUlN4SFFVRkhMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUjBGQlJ5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFZEJRVWNzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRE8zZENRVU5zUXl4SFFVRkhMRWxCUVVrc1JVRkJSU3hEUVVGRExFTkJRVU1zUTBGQlF5eEhRVUZITEVWQlFVVXNSMEZCUnl4RlFVRkZMRU5CUVVNN2NVSkJRekZDTzI5Q1FVTkVMRWxCUVVrc1IwRkJSeXhIUVVGSExFbEJRVWs3ZDBKQlFVVXNUVUZCVFR0dlFrRkRkRUlzU1VGQlNTeEZRVUZGTEVkQlFVY3NTVUZCU1N4RFFVRkRMRk5CUVZNc1JVRkJSU3hEUVVGRE8yOUNRVU14UWl4TFFVRkxMRU5CUVVNc1IwRkJSeXhEUVVGRExFVkJRVVVzUTBGQlF5eEhRVUZITEVsQlFVa3NRMEZCUXl4RFFVRkRMRVZCUVVVc1JVRkJSU3hEUVVGRE8zZENRVUZGTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zU1VGQlNTeEZRVUZGTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNN2FVSkJRMnBFTzJkQ1FVTkVMRWxCUVVrc1EwRkJReXhIUVVGWExFbEJRVWtzUTBGQlF5eEpRVUZKTEVOQlFVTXNSMEZCUnl4RFFVRkRMRU5CUVVNN1owSkJReTlDTEVsQlFVa3NRMEZCUXl4SFFVRlhMRWxCUVVrc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNN1owSkJRemRDTEVsQlFVa3NUVUZCVFN4SFFVRkhMRWxCUVVrc1EwRkJReXhEUVVGRExFbEJRVWtzU1VGQlNTeERRVUZETEVOQlFVTXNRMEZCUXl4SlFVRkpMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU03WjBKQlF5OURMRWxCUVVrc1RVRkJUU3hIUVVGSExFTkJRVU1zU1VGQlNTeERRVUZETEVkQlFVY3NRMEZCUXl4SlFVRkpMRU5CUVVNc1VVRkJVU3hEUVVGRExFTkJRVU1zUTBGQlF5eEZRVUZGTzI5Q1FVTnlReXhMUVVGTExFTkJRVU1zUjBGQlJ5eERRVUZETEVWQlFVVXNRMEZCUXl4SFFVRkhMRWxCUVVrc1EwRkJReXhEUVVGRExFVkJRVVVzUlVGQlJTeERRVUZETzNkQ1FVRkZMRWxCUVVrc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFZEJRVWNzUTBGQlF5eERRVUZETzI5Q1FVTnFSQ3hUUVVGVE8ybENRVU5hTzJkQ1FVTkVMRWxCUVVrc1RVRkJUU3hIUVVGSExFTkJRVU1zUlVGQlJUdHZRa0ZEV2l4TlFVRk5MRWRCUVVjc1EwRkJReXhEUVVGRE8ybENRVU5rTzJkQ1FVTkVMRWxCUVVrc1JVRkJSU3hIUVVGWExFTkJRVU1zUjBGQlJ5eERRVUZETEVOQlFVTTdaMEpCUTNaQ0xFbEJRVWtzUlVGQlJTeEhRVUZYTEVOQlFVTXNSMEZCUnl4TlFVRk5MRWRCUVVjc1EwRkJReXhEUVVGRExFZEJRVWNzUTBGQlF5eERRVUZETEVkQlFVY3NRMEZCUXl4RlFVRkZMRWRCUVVjc1EwRkJReXhEUVVGRExFTkJRVU03WjBKQlEycEVMRWxCUVVrc1JVRkJSU3hIUVVGSExFTkJRVU1zUjBGQlJ5eERRVUZETEVkQlFVY3NRMEZCUXl4RFFVRkRPMmRDUVVOdVFpeEpRVUZKTEVWQlFVVXNSMEZCVnl4RFFVRkRMRWRCUVVjc1EwRkJReXhOUVVGTkxFZEJRVWNzUTBGQlF5eEZRVUZGTEVkQlFVY3NSVUZCUlN4RFFVRkRMRU5CUVVNN1owSkJRM3BETEVsQlFVa3NRMEZCUXl4UlFVRlJMRU5CUVVNc1JVRkJSU3hEUVVGRE8yOUNRVU5pTEU5QlFVOHNRMEZCUXl4SFFVRkhMRU5CUVVNc1JVRkJSU3hEUVVGRExFTkJRVU03WjBKQlEzQkNMRXRCUVVzc1EwRkJReXhIUVVGSExFTkJRVU1zUlVGQlJTeERRVUZETEVkQlFVY3NTVUZCU1N4RFFVRkRMRU5CUVVNc1JVRkJSU3hGUVVGRkxFTkJRVU1zUlVGQlJUdHZRa0ZEZWtJc1NVRkJTU3hEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1NVRkJTU3hEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVkQlFVY3NSVUZCUlN4RFFVRkRPMjlDUVVNeFFpeEhRVUZITEVOQlFVTXNRMEZCUXl4RFFVRkRMRWxCUVVrc1NVRkJTU3hEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUjBGQlJ5eEZRVUZGTEVkQlFVY3NRMEZCUXl4RlFVRkZMRWRCUVVjc1EwRkJReXhIUVVGSExFTkJRVU1zUlVGQlJTeERRVUZETEVOQlFVTXNRMEZCUXl4SFFVRkhMRWRCUVVjc1EwRkJReXhIUVVGSExFTkJRVU1zUjBGQlJ5eEhRVUZITEVOQlFVTXNRMEZCUXp0cFFrRkRka1U3WVVGRFNqdFpRVU5FTEV0QlFVc3NRMEZCUXl4SFFVRkhMRU5CUVVNc1JVRkJSU3hEUVVGRExFZEJRVWNzU1VGQlNTeERRVUZETEVOQlFVTXNSVUZCUlN4RlFVRkZMRU5CUVVNN1owSkJRVVVzU1VGQlNTeEhRVUZITEVsQlFVa3NRMEZCUXl4SFFVRkhMRU5CUVVNc1NVRkJTU3hGUVVGRkxFbEJRVWtzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVkQlFVY3NSMEZCUnl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU03VTBGRGFFWTdVVUZGUkN4SlFVRkpMRU5CUVVNc1IwRkJSeXhKUVVGSkxFTkJRVU1zV1VGQldTeEhRVUZETEVOQlFVTXNRMEZCUXp0UlFVTTFRaXhKUVVGSkxFTkJRVU1zUjBGQlJ5eEpRVUZKTEVOQlFVTXNXVUZCV1N4RFFVRkRPMUZCUXpGQ0xFbEJRVWtzUTBGQlF5eEhRVUZITEVsQlFVa3NRMEZCUXl4WlFVRlpMRU5CUVVNN1VVRkRNVUlzU1VGQlNTeERRVUZETEVkQlFVY3NRMEZCUXl4SFFVRkhMRU5CUVVNc1EwRkJReXhIUVVGSExFTkJRVU1zUTBGQlF5eERRVUZETzFGQlEzQkNMRWxCUVVrc1VVRkJVU3hIUVVGSExFbEJRVWtzUTBGQlF5eG5Ra0ZCWjBJc1EwRkJRenRSUVVWeVF5eExRVUZMTEVsQlFVa3NRMEZCUXl4SFFVRlhMRU5CUVVNc1JVRkJSU3hEUVVGRExFZEJRVWNzVVVGQlVTeEZRVUZGTEVWQlFVVXNRMEZCUXl4RlFVRkZPMWxCUTNaRExFdEJRVXNzUTBGQlF5eEhRVUZITEVOQlFVTXNSVUZCUlN4RFFVRkRMRWRCUVVjc1NVRkJTU3hEUVVGRExFTkJRVU1zUlVGQlJTeEZRVUZGTEVOQlFVTXNSVUZCUlR0blFrRkRla0lzU1VGQlNTeEhRVUZITEVkQlFVY3NTVUZCU1N4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXp0blFrRkRka0lzU1VGQlNTeERRVUZETEVkQlFVY3NSMEZCUnl4SFFVRkhMRU5CUVVNc1EwRkJRenRuUWtGRGFFSXNTVUZCU1N4RFFVRkRMRWRCUVVjc1EwRkJReXhIUVVGSExFTkJRVU1zUTBGQlF6dG5Ra0ZEWkN4SlFVRkpMRU5CUVVNc1IwRkJSeXhEUVVGRExFZEJRVWNzUTBGQlF5eERRVUZETzJkQ1FVTmtMRWxCUVVrc1EwRkJReXhIUVVGSExFbEJRVWtzUTBGQlF5eEhRVUZITEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNN1owSkJRM0JDTEVsQlFVa3NSVUZCUlN4SFFVRkhMRU5CUVVNc1EwRkJReXhKUVVGSkxFZEJRVWNzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4SFFVRkhMRWRCUVVjc1EwRkJReXhIUVVGSExFTkJRVU1zUTBGQlF5eERRVUZETzI5Q1FVTXZRaXhEUVVGRExFZEJRVWNzUjBGQlJ5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1IwRkJSeXhIUVVGSExFTkJRVU1zUTBGQlF5eEhRVUZITEVOQlFVTXNRMEZCUXl4SFFVRkhMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUjBGQlJ5eEhRVUZITEVOQlFVTXNRMEZCUXl4SFFVRkhMRU5CUVVNc1EwRkJReXhIUVVGSExFTkJRVU1zUTBGQlF6dG5Ra0ZEZEVRc1NVRkJTU3hEUVVGRExFTkJRVU1zUjBGQlJ5eEZRVUZGTEVsQlFVa3NSVUZCUlN4SlFVRkpMRU5CUVVNc1JVRkJSVHR2UWtGRGNFSXNTVUZCU1N4SlFVRkpMRU5CUVVNc1pVRkJaU3hGUVVGRk8zZENRVU4wUWl4SlFVRkpMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4SlFVRkpMRWxCUVVrc1IwRkJSeXhEUVVGRExFZEJRVWNzUlVGQlJTeERRVUZETzNkQ1FVTTVRaXhKUVVGSkxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eEpRVUZKTEVsQlFVa3NSMEZCUnl4RFFVRkRMRU5CUVVNN2NVSkJReTlDTzNsQ1FVRk5PM2RDUVVOSUxFbEJRVWtzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFbEJRVWtzUTBGQlF5eEhRVUZITEVWQlFVVXNRMEZCUXp0M1FrRkRka0lzU1VGQlNTeERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNTVUZCU1N4RFFVRkRMRU5CUVVNN2NVSkJRM2hDTzJsQ1FVTktPMkZCUTBvN1UwRkRTanRSUVVORUxFbEJRVWtzUTBGQlF5eEpRVUZKTEVOQlFVTXNTMEZCU3l4RFFVRkRMRTlCUVU4c1JVRkJSU3hGUVVGRk8xbEJRM1pDTEVsQlFVa3NRMEZCUXl4TFFVRkxMRU5CUVVNc1MwRkJTeXhEUVVGRExGVkJRVU1zUTBGQlF5eEZRVUZGTEVOQlFVTTdaMEpCUTJ4Q0xFdEJRVXNzUTBGQlF5eEhRVUZITEVOQlFVTXNSVUZCUlN4RFFVRkRMRWRCUVVjc1MwRkJTU3hEUVVGRExFTkJRVU1zUlVGQlJTeEZRVUZGTEVOQlFVTXNSVUZCUlR0dlFrRkRla0lzUzBGQlNTeERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNTVUZCU1N4SlFVRkpMRU5CUVVNN2IwSkJRM2hDTEV0QlFVa3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVsQlFVa3NTVUZCU1N4SFFVRkhMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eEhRVUZITEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETzJsQ1FVTXpRenRaUVVOTUxFTkJRVU1zUTBGQlF5eERRVUZETzFOQlEwNDdTVUZUVEN4RFFVRkRPMGxCUldNc1pVRkJUeXhIUVVGMFFpeFZRVUYxUWl4RFFVRlhMRVZCUVVVc1EwRkJWenRSUVVNelF5eEpRVUZKTEVOQlFVTXNSMEZCUnl4RFFVRkRMRVZCUVVVc1EwRkJReXhIUVVGSExFTkJRVU1zUTBGQlF5eE5RVUZOTEVOQlFVTTdVVUZEZUVJc1QwRkJUeXhEUVVGRExFVkJRVVU3V1VGQlJTeERRVUZETEVsQlFVa3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhIUVVGSExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXp0UlFVTTNRaXhQUVVGUExFTkJRVU1zUTBGQlF6dEpRVU5pTEVOQlFVTTdTVUZIWXl4eFFrRkJZU3hIUVVFMVFpeFZRVUUyUWl4RFFVRmhMRVZCUVVVc1EwRkJWeXhGUVVGRkxFTkJRVmM3VVVGRGFFVXNTVUZCU1N4RFFVRkRMRWRCUVVjc1EwRkJReXhEUVVGRExFMUJRVTBzUTBGQlF6dFJRVU5xUWl4UFFVRlBMRU5CUVVNc1JVRkJSVHRaUVVGRkxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNSMEZCUnl4UFFVRlBMRU5CUVVNc1QwRkJUeXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNSVUZCUlN4RFFVRkRMRU5CUVVNc1EwRkJRenRKUVVOb1JDeERRVUZETzBsQlMwMHNhVU5CUVdVc1IwRkJkRUlzVlVGQmRVSXNRMEZCWVR0UlFVTm9ReXhKUVVGSkxGTkJRVk1zUjBGQlJ5eERRVUZETEVWQlFVVXNWMEZCVnl4SFFVRkhMRU5CUVVNc1EwRkJRenRSUVVOdVF5eExRVUZMTEVsQlFVa3NRMEZCUXl4SFFVRkhMRU5CUVVNc1JVRkJSU3hEUVVGRExFZEJRVWNzU1VGQlNTeERRVUZETEVOQlFVTXNSVUZCUlN4RlFVRkZMRU5CUVVNc1JVRkJSVHRaUVVNM1FpeFRRVUZUTEVsQlFVa3NUMEZCVHl4RFFVRkRMRTlCUVU4c1EwRkJReXhKUVVGSkxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RlFVRkZMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETzFsQlF6bERMRTlCUVU4c1EwRkJReXhoUVVGaExFTkJRVU1zU1VGQlNTeERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1JVRkJSU3hEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVWQlFVVXNTVUZCU1N4RFFVRkRMRVZCUVVVc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETzFsQlEyNUVMRmRCUVZjc1NVRkJTU3hQUVVGUExFTkJRVU1zVDBGQlR5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1JVRkJSU3hKUVVGSkxFTkJRVU1zUlVGQlJTeERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNN1UwRkRjRVE3VVVGRFJDeEpRVUZKTEZkQlFWY3NTMEZCU3l4RFFVRkRMRWxCUVVrc1EwRkJReXhSUVVGUkxFTkJRVU1zVjBGQlZ5eERRVUZETzFsQlFVVXNUMEZCVHl4RFFVRkRMRU5CUVVNN1VVRkRNVVFzVDBGQlR5eERRVUZETEVkQlFVY3NVMEZCVXl4SFFVRkhMRmRCUVZjc1EwRkJRenRKUVVOMlF5eERRVUZETzBsQlJVMHNPRUpCUVZrc1IwRkJia0k3VVVGRFNTeEpRVUZKTEVOQlFVTXNhMEpCUVd0Q0xFTkJRVU1zU1VGQlNTeERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRPMUZCUTJoRExFbEJRVWtzUzBGQlN5eEhRVUZITEVsQlFVa3NRMEZCUXl4bFFVRmxMRU5CUVVNc1NVRkJTU3hEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETzFGQlEzcERMRXRCUVVzc1NVRkJTU3hEUVVGRExFZEJRVWNzUTBGQlF5eEZRVUZGTEVOQlFVTXNSMEZCUnl4SlFVRkpMRU5CUVVNc1EwRkJReXhGUVVGRkxFVkJRVVVzUTBGQlF5eEZRVUZGTzFsQlF6ZENMRWxCUVVrc1EwRkJReXhsUVVGbExFTkJRVU1zU1VGQlNTeERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1JVRkJSU3hKUVVGSkxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RlFVRkZMRXRCUVVzc1EwRkJReXhEUVVGRE8xTkJRM0pFTzFGQlEwUXNUMEZCVHl4SlFVRkpMRU5CUVVNc1lVRkJZU3hGUVVGRkxFTkJRVU03U1VGRGFFTXNRMEZCUXp0SlFVVmpMRmxCUVVrc1IwRkJia0lzVlVGQmIwSXNRMEZCWVN4RlFVRkZMRU5CUVdFN1VVRkROVU1zU1VGQlNTeERRVUZETEVkQlFVY3NRMEZCUXl4RFFVRkRMRTFCUVUwc1JVRkJSU3hEUVVGRExFZEJRVWNzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRTFCUVUwc1EwRkJRenRSUVVOc1F5eExRVUZMTEVsQlFVa3NRMEZCUXl4SFFVRkhMRU5CUVVNc1JVRkJSU3hEUVVGRExFZEJRVWNzUTBGQlF5eEZRVUZGTEVWQlFVVXNRMEZCUXl4RlFVRkZPMWxCUTNoQ0xFdEJRVXNzU1VGQlNTeERRVUZETEVkQlFVY3NRMEZCUXl4RlFVRkZMRU5CUVVNc1IwRkJSeXhEUVVGRExFVkJRVVVzUlVGQlJTeERRVUZETEVWQlFVVTdaMEpCUTNoQ0xFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1IwRkJSeXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNN1lVRkRja0k3VTBGRFNqdEpRVU5NTEVOQlFVTTdTVUZSVHl4blEwRkJZeXhIUVVGMFFpeFZRVUYxUWl4RlFVRmpMRVZCUVVVc1EwRkJZU3hGUVVGRkxFTkJRV0VzUlVGQlJTeFJRVUZuUWp0UlFVTnFSaXhQUVVGUExFTkJRVU1zU1VGQlNTeERRVUZETEVWQlFVVXNSVUZCUlN4RFFVRkRMRU5CUVVNc1EwRkJRenRSUVVOd1FpeEpRVUZKTEVOQlFVTXNaVUZCWlN4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUlVGQlJTeERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRVZCUVVVc1VVRkJVU3hEUVVGRExFTkJRVU03VVVGRE0wTXNTVUZCU1N4SlFVRkpMRU5CUVVNc1QwRkJUenRaUVVGRkxFbEJRVWtzUTBGQlF5eFBRVUZQTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1JVRkJSU3hEUVVGRExFTkJRVU1zUTBGQlF5eEZRVUZGTEVWQlFVVXNRMEZCUXl4RFFVRkRMRU5CUVVNc1JVRkJSU3hEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXp0UlFVTjBSQ3hKUVVGSkxFTkJRVU1zWlVGQlpTeERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1JVRkJSU3hEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVWQlFVVXNVVUZCVVN4RFFVRkRMRU5CUVVNN1VVRkRNME1zU1VGQlNTeEpRVUZKTEVOQlFVTXNUMEZCVHp0WlFVRkZMRWxCUVVrc1EwRkJReXhQUVVGUExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhGUVVGRkxFVkJRVVVzUTBGQlF5eERRVUZETEVOQlFVTXNSVUZCUlN4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dFJRVWR5UkN4TFFVRkxMRWxCUVVrc1EwRkJReXhIUVVGSExFTkJRVU1zUlVGQlJTeERRVUZETEVkQlFVY3NTVUZCU1N4RFFVRkRMRU5CUVVNc1JVRkJSU3hEUVVGRExFVkJRVVU3V1VGRE0wSXNTVUZCU1N4RFFVRkRMR1ZCUVdVc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVWQlFVVXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhGUVVGRkxGRkJRVkVzUTBGQlF5eERRVUZETzBsQlZXNUVMRU5CUVVNN1NVRkZZeXhqUVVGTkxFZEJRWEpDTEZWQlFYTkNMRU5CUVZNc1JVRkJSU3hEUVVGVExFVkJRVVVzUTBGQlowTTdVVUZEZUVVc1NVRkJTU3hEUVVGRExFZEJRVWNzUTBGQlF5eERRVUZETzFGQlFVTXNUMEZCVHl4RFFVRkRMRVZCUVVVc1IwRkJSeXhEUVVGRExFVkJRVVU3V1VGRGRrSXNTVUZCU1N4RFFVRkRMRWRCUVVjc1EwRkJReXhEUVVGRE8xbEJRVU1zVDBGQlR5eERRVUZETEVWQlFVVXNSMEZCUnl4RFFVRkRPMmRDUVVGRkxFTkJRVU1zUTBGQlF5eERRVUZETEVWQlFVVXNRMEZCUXl4RFFVRkRMRU5CUVVNN1UwRkRkRU03U1VGRFRDeERRVUZETzBsQlEwOHNOa0pCUVZjc1IwRkJia0lzVlVGQmIwSXNRMEZCWjBNN1VVRkRhRVFzVDBGQlR5eERRVUZETEUxQlFVMHNRMEZCUXl4SlFVRkpMRU5CUVVNc1EwRkJReXhGUVVGRkxFbEJRVWtzUTBGQlF5eERRVUZETEVWQlFVVXNRMEZCUXl4RFFVRkRMRU5CUVVNN1NVRkRkRU1zUTBGQlF6dEpRVVZQTEhGRFFVRnRRaXhIUVVFelFpeFZRVUUwUWl4RlFVRmpMRVZCUVVVc1EwRkJZVHRSUVVGNlJDeHBRa0ZsUXp0UlFXUkhMRWxCUVVrc1EwRkJReXhyUWtGQmEwSXNRMEZCUXl4RlFVRkZMRU5CUVVNc1EwRkJRenRSUVVNMVFpeEpRVUZKTEV0QlFVc3NSMEZCUnl4SlFVRkpMRU5CUVVNc1pVRkJaU3hEUVVGRExFbEJRVWtzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXp0UlFVTjZReXhKUVVGSkxFTkJRVU1zWTBGQll5eERRVUZETEVWQlFVVXNSVUZCUlN4RFFVRkRMRVZCUVVVc1NVRkJTU3hEUVVGRExFTkJRVU1zUlVGQlJTeExRVUZMTEVOQlFVTXNRMEZCUXp0UlFVMHhReXhKUVVGSkxFbEJRVWtzUTBGQlF5eFBRVUZQTEVWQlFVVTdXVUZEWkN4SlFVRkpMRU5CUVVNc1YwRkJWeXhEUVVGRExGVkJRVU1zUTBGQlF5eEZRVUZGTEVOQlFVTXNTVUZCU3l4UFFVRkJMRXRCUVVrc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRWRCUVVjc1JVRkJSU3hEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4SFFVRkhMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNSVUZCYWtNc1EwRkJhVU1zUTBGQlF5eERRVUZETzFsQlF6bEVMRWxCUVVrc1NVRkJTU3hIUVVGSExFbEJRVWtzUTBGQlF5eGxRVUZsTEVOQlFVTXNTVUZCU1N4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRE8xbEJRM2hETEVsQlFVa3NSMEZCUnl4SlFVRkpMRU5CUVVNc1IwRkJSeXhEUVVGRExFZEJRVWNzUlVGQlJTeEpRVUZKTEVOQlFVTXNSMEZCUnl4RFFVRkRMRWxCUVVrc1JVRkJSU3hEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETzFsQlEzaERMRWxCUVVrc1EwRkJReXhqUVVGakxFTkJRVU1zUlVGQlJTeEZRVUZGTEVOQlFVTXNSVUZCUlN4SlFVRkpMRU5CUVVNc1EwRkJReXhGUVVGRkxFbEJRVWtzUTBGQlF5eERRVUZETzFOQlF6VkRPMGxCUTB3c1EwRkJRenRKUVVWTkxIRkNRVUZITEVkQlFWWXNWVUZCVnl4VlFVRnJRanRSUVVONlFpeEpRVUZKTEUxQlFVMHNSMEZCUnl4TlFVRk5MRU5CUVVNc1UwRkJVeXhGUVVGRkxGTkJRVk1zUjBGQlJ5eExRVUZMTEVOQlFVTTdVVUZEYWtRc1QwRkJUeXhEUVVGRExGTkJRVk1zU1VGQlNTeFZRVUZWTEVWQlFVVXNSMEZCUnl4RFFVRkRMRVZCUVVVN1dVRkRia01zU1VGQlNTeERRVUZETEVkQlFVY3NTVUZCU1N4RFFVRkRMRlZCUVZVc1JVRkJSU3hEUVVGRE8xbEJRekZDTEZOQlFWTXNSMEZCUnl4SlFVRkpMRU5CUVVNc1IwRkJSeXhEUVVGRExFMUJRVTBzUjBGQlJ5eERRVUZETEVkQlFVY3NRMEZCUXl4RFFVRkRMRWRCUVVjc1NVRkJTU3hEUVVGRExGTkJRVk1zUTBGQlF6dFpRVU4wUkN4TlFVRk5MRWRCUVVjc1EwRkJReXhEUVVGRE8xTkJRMlE3VVVGRFJDeFBRVUZQTEUxQlFVMHNRMEZCUXp0SlFVTnNRaXhEUVVGRE8wbEJSVTBzTkVKQlFWVXNSMEZCYWtJN1VVRkJRU3hwUWtGbFF6dFJRV1JITEVsQlFVa3NRMEZCUXl4dFFrRkJiVUlzUTBGQlF5eEpRVUZKTEVOQlFVTXNRMEZCUXl4RlFVRkZMRWxCUVVrc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dFJRVU42UXl4UFFVRlBMRU5CUVVNc1IwRkJSeXhEUVVGRExFbEJRVWtzUTBGQlF5eERRVUZETEVWQlFVVXNTVUZCU1N4RFFVRkRMRU5CUVVNc1JVRkJSU3hKUVVGSkxFTkJRVU1zUlVGQlJTeERRVUZETEVOQlFVTTdVVUZEY2tNc1NVRkJTU3hEUVVGRExHMUNRVUZ0UWl4RFFVRkRMRWxCUVVrc1EwRkJReXhGUVVGRkxFVkJRVVVzU1VGQlNTeERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRPMUZCUXpGRExFOUJRVThzUTBGQlF5eEhRVUZITEVOQlFVTXNTVUZCU1N4RFFVRkRMRU5CUVVNc1JVRkJSU3hKUVVGSkxFTkJRVU1zUTBGQlF5eEZRVUZGTEVsQlFVa3NRMEZCUXl4RlFVRkZMRU5CUVVNc1EwRkJRenRSUVVOeVF5eEpRVUZKTEVOQlFVTXNiVUpCUVcxQ0xFTkJRVU1zU1VGQlNTeERRVUZETEVWQlFVVXNSVUZCUlN4SlFVRkpMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU03VVVGRE1VTXNTVUZCU1N4RFFVRkRMRzFDUVVGdFFpeERRVUZETEVsQlFVa3NRMEZCUXl4RFFVRkRMRVZCUVVVc1NVRkJTU3hEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETzFGQlEzcERMRWxCUVVrc1NVRkJTU3hIUVVGSExFTkJRVU1zUTBGQlF6dFJRVU5pTEVsQlFVa3NRMEZCUXl4WFFVRlhMRU5CUVVNc1ZVRkJReXhEUVVGRExFVkJRVVVzUTBGQlF6dFpRVU5zUWl4SlFVRkpMRU5CUVVNc1IwRkJSeXhEUVVGRExFdEJRVWtzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFZEJRVWNzUjBGQlJ5eEhRVUZITEV0QlFVa3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVkQlFVY3NSMEZCUnl4SFFVRkhMRXRCUVVrc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRWRCUVVjc1MwRkJTU3hEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhIUVVGSExFZEJRVWNzUlVGRGFrWXNRMEZCUXl4SFFVRkhMRXRCUVVrc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRWRCUVVjc1EwRkJReXhEUVVGRE8xbEJRM3BDTEVsQlFVa3NTVUZCU1N4RFFVRkRMRWRCUVVjc1EwRkJReXhEUVVGRE8xbEJRMlFzUzBGQlNTeERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUjBGQlJ5eERRVUZETEVOQlFVTTdVVUZEY2tJc1EwRkJReXhEUVVGRExFTkJRVU03VVVGRFNDeFBRVUZQTEVsQlFVa3NRMEZCUXp0SlFVTm9RaXhEUVVGRE8wbEJSV01zVjBGQlJ5eEhRVUZzUWl4VlFVRnRRaXhEUVVGaExFVkJRVVVzUTBGQllTeEZRVUZGTEVOQlFXRTdVVUZETVVRc1QwRkJUeXhEUVVGRExFMUJRVTBzUTBGQlF5eERRVUZETEVOQlFVTXNUVUZCVFN4RlFVRkZMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eE5RVUZOTEVWQlFVVXNWVUZCUXl4RFFVRkRMRVZCUVVVc1EwRkJRenRaUVVOMlF5eFBRVUZCTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUjBGQlJ5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFZEJRVWNzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFZEJRVWNzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFZEJRVWNzUjBGQlJ6dFJRVUUzUXl4RFFVRTJReXhEUVVGRExFTkJRVU03U1VGRGRrUXNRMEZCUXp0SlFVVk5MR2xEUVVGbExFZEJRWFJDTEZWQlFYVkNMRU5CUVZjc1JVRkJSU3hEUVVGWExFVkJRVVVzVVVGQlowSTdVVUZETjBRc1MwRkJTeXhKUVVGSkxFTkJRVU1zUjBGQlJ5eERRVUZETEVWQlFVVXNRMEZCUXl4SFFVRkhMRWxCUVVrc1EwRkJReXhEUVVGRExFVkJRVVVzUlVGQlJTeERRVUZETEVWQlFVVTdXVUZETjBJc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eEhRVUZITEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1IwRkJSeXhSUVVGUkxFZEJRVWNzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRPMU5CUTJwRE8wbEJRMHdzUTBGQlF6dEpRVVZOTEN0Q1FVRmhMRWRCUVhCQ08xRkJRMGtzU1VGQlNTeE5RVUZOTEVkQlFVY3NRMEZCUXl4RFFVRkRPMUZCUTJZc1MwRkJTeXhKUVVGSkxFTkJRVU1zUjBGQlJ5eERRVUZETEVWQlFVVXNUMEZCVHl4SFFVRkhMRWxCUVVrc1EwRkJReXhEUVVGRExFZEJRVWNzUTBGQlF5eEZRVUZGTEVOQlFVTXNSMEZCUnl4UFFVRlBMRVZCUVVVc1JVRkJSU3hEUVVGRExFVkJRVVU3V1VGRGNFUXNTMEZCU3l4SlFVRkpMRU5CUVVNc1IwRkJSeXhEUVVGRExFZEJRVWNzUTBGQlF5eEZRVUZGTEVOQlFVTXNSMEZCUnl4SlFVRkpMRU5CUVVNc1EwRkJReXhGUVVGRkxFTkJRVU1zUjBGQlJ5eERRVUZETEVWQlFVVXNSVUZCUlN4RFFVRkRMRVZCUVVVN1owSkJRM2hETEVsQlFVa3NRMEZCUXl4SFFVRkhMRU5CUVVNc1EwRkJRenRuUWtGRFZpeExRVUZMTEVsQlFVa3NRMEZCUXl4SFFVRkhMRU5CUVVNc1JVRkJSU3hEUVVGRExFZEJRVWNzU1VGQlNTeERRVUZETEVOQlFVTXNSVUZCUlN4RlFVRkZMRU5CUVVNc1JVRkJSVHR2UWtGRE4wSXNTVUZCU1N4RlFVRkZMRWRCUVVjc1NVRkJTU3hEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1IwRkJSeXhKUVVGSkxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRE8yOUNRVU55UXl4RFFVRkRMRWxCUVVrc1JVRkJSU3hIUVVGSExFVkJRVVVzUTBGQlF6dHBRa0ZEYUVJN1owSkJRMFFzUTBGQlF5eEhRVUZITEVsQlFVa3NRMEZCUXl4SlFVRkpMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU03WjBKQlEycENMRWxCUVVrc1EwRkJReXhIUVVGSExFbEJRVWtzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU03WjBKQlEzSkNMRWxCUVVrc1EwRkJReXhSUVVGUkxFTkJRVU1zUTBGQlF5eERRVUZETzI5Q1FVRkZMRk5CUVZNN1owSkJRek5DTEVsQlFVa3NSVUZCUlN4SFFVRkhMRU5CUVVNc1IwRkJSeXhEUVVGRExFTkJRVU03WjBKQlEyWXNTVUZCU1N4RlFVRkZMRWRCUVVjc1EwRkJReXhIUVVGSExFTkJRVU1zUTBGQlF6dG5Ra0ZEWml4TlFVRk5MRWxCUVVrc1JVRkJSU3hIUVVGSExFVkJRVVVzUjBGQlJ5eEZRVUZGTEVOQlFVTTdZVUZETVVJN1UwRkRTanRSUVVORUxFOUJRVThzVFVGQlRTeERRVUZETzBsQlEyeENMRU5CUVVNN1NVRnlWMk1zYjBKQlFWa3NSMEZCVnl4TFFVRkxMRU5CUVVNN1NVRnpWMmhFTEdOQlFVTTdRMEZCUVN4QlFXaFpSQ3hKUVdkWlF6dEJRV2haV1N3d1FrRkJUenRCUVcxWmNFSTdTVUZOU1N4elFrRkJiVUlzU1VGQlowSTdVVUZCYUVJc2NVSkJRVUVzUlVGQlFTeFJRVUZuUWp0UlFVRm9RaXhUUVVGSkxFZEJRVW9zU1VGQlNTeERRVUZaTzFGQlRETkNMRTFCUVVNc1IwRkJWeXhOUVVGTkxFTkJRVU03VVVGRGJrSXNUVUZCUXl4SFFVRlhMRTlCUVU4c1EwRkJRenRSUVVOd1FpeE5RVUZETEVkQlFWY3NWVUZCVlN4RFFVRkRPMUZCUTNaQ0xGVkJRVXNzUjBGQlZ5eExRVUZMTEVOQlFVTTdTVUZGVXl4RFFVRkRPMGxCUjNoRExEaENRVUZQTEVkQlFWQTdVVUZEU1N4SlFVRkpMRU5CUVVNc1NVRkJTU3hIUVVGSExFTkJRVU1zU1VGQlNTeERRVUZETEVsQlFVa3NSMEZCUnl4SlFVRkpMRU5CUVVNc1EwRkJReXhIUVVGSExFbEJRVWtzUTBGQlF5eERRVUZETEVOQlFVTXNSMEZCUnl4SlFVRkpMRU5CUVVNc1EwRkJReXhEUVVGRE8xRkJRMjVFTEU5QlFVOHNRMEZCUXl4SlFVRkpMRU5CUVVNc1NVRkJTU3hKUVVGSkxFVkJRVVVzUTBGQlF5eEhRVUZITEVsQlFVa3NRMEZCUXl4TFFVRkxMRU5CUVVNN1NVRkRNVU1zUTBGQlF6dEpRVWRFTEhGRFFVRmpMRWRCUVdRc1ZVRkJaU3hIUVVGWExFVkJRVVVzUjBGQlZ6dFJRVU51UXl4UFFVRlBMRWRCUVVjc1IwRkJSeXhKUVVGSkxFTkJRVU1zVDBGQlR5eEZRVUZGTEVkQlFVY3NRMEZCUXl4SFFVRkhMRWRCUVVjc1IwRkJSeXhEUVVGRExFTkJRVU03U1VGRE9VTXNRMEZCUXp0SlFVTk1MRzFDUVVGRE8wRkJRVVFzUTBGQlF5eEJRV3hDUkN4SlFXdENRenRCUVd4Q1dTeHZRMEZCV1NKOSIsIlwidXNlIHN0cmljdFwiO1xyXG52YXIgX19leHRlbmRzID0gKHRoaXMgJiYgdGhpcy5fX2V4dGVuZHMpIHx8IChmdW5jdGlvbiAoKSB7XHJcbiAgICB2YXIgZXh0ZW5kU3RhdGljcyA9IGZ1bmN0aW9uIChkLCBiKSB7XHJcbiAgICAgICAgZXh0ZW5kU3RhdGljcyA9IE9iamVjdC5zZXRQcm90b3R5cGVPZiB8fFxyXG4gICAgICAgICAgICAoeyBfX3Byb3RvX186IFtdIH0gaW5zdGFuY2VvZiBBcnJheSAmJiBmdW5jdGlvbiAoZCwgYikgeyBkLl9fcHJvdG9fXyA9IGI7IH0pIHx8XHJcbiAgICAgICAgICAgIGZ1bmN0aW9uIChkLCBiKSB7IGZvciAodmFyIHAgaW4gYikgaWYgKGIuaGFzT3duUHJvcGVydHkocCkpIGRbcF0gPSBiW3BdOyB9O1xyXG4gICAgICAgIHJldHVybiBleHRlbmRTdGF0aWNzKGQsIGIpO1xyXG4gICAgfTtcclxuICAgIHJldHVybiBmdW5jdGlvbiAoZCwgYikge1xyXG4gICAgICAgIGV4dGVuZFN0YXRpY3MoZCwgYik7XHJcbiAgICAgICAgZnVuY3Rpb24gX18oKSB7IHRoaXMuY29uc3RydWN0b3IgPSBkOyB9XHJcbiAgICAgICAgZC5wcm90b3R5cGUgPSBiID09PSBudWxsID8gT2JqZWN0LmNyZWF0ZShiKSA6IChfXy5wcm90b3R5cGUgPSBiLnByb3RvdHlwZSwgbmV3IF9fKCkpO1xyXG4gICAgfTtcclxufSkoKTtcclxuT2JqZWN0LmRlZmluZVByb3BlcnR5KGV4cG9ydHMsIFwiX19lc01vZHVsZVwiLCB7IHZhbHVlOiB0cnVlIH0pO1xyXG52YXIgcmVjdGFuZ2xlXzEgPSByZXF1aXJlKFwiLi9yZWN0YW5nbGVcIik7XHJcbnZhciBQb2ludCA9IChmdW5jdGlvbiAoKSB7XHJcbiAgICBmdW5jdGlvbiBQb2ludCgpIHtcclxuICAgIH1cclxuICAgIHJldHVybiBQb2ludDtcclxufSgpKTtcclxuZXhwb3J0cy5Qb2ludCA9IFBvaW50O1xyXG52YXIgTGluZVNlZ21lbnQgPSAoZnVuY3Rpb24gKCkge1xyXG4gICAgZnVuY3Rpb24gTGluZVNlZ21lbnQoeDEsIHkxLCB4MiwgeTIpIHtcclxuICAgICAgICB0aGlzLngxID0geDE7XHJcbiAgICAgICAgdGhpcy55MSA9IHkxO1xyXG4gICAgICAgIHRoaXMueDIgPSB4MjtcclxuICAgICAgICB0aGlzLnkyID0geTI7XHJcbiAgICB9XHJcbiAgICByZXR1cm4gTGluZVNlZ21lbnQ7XHJcbn0oKSk7XHJcbmV4cG9ydHMuTGluZVNlZ21lbnQgPSBMaW5lU2VnbWVudDtcclxudmFyIFBvbHlQb2ludCA9IChmdW5jdGlvbiAoX3N1cGVyKSB7XHJcbiAgICBfX2V4dGVuZHMoUG9seVBvaW50LCBfc3VwZXIpO1xyXG4gICAgZnVuY3Rpb24gUG9seVBvaW50KCkge1xyXG4gICAgICAgIHJldHVybiBfc3VwZXIgIT09IG51bGwgJiYgX3N1cGVyLmFwcGx5KHRoaXMsIGFyZ3VtZW50cykgfHwgdGhpcztcclxuICAgIH1cclxuICAgIHJldHVybiBQb2x5UG9pbnQ7XHJcbn0oUG9pbnQpKTtcclxuZXhwb3J0cy5Qb2x5UG9pbnQgPSBQb2x5UG9pbnQ7XHJcbmZ1bmN0aW9uIGlzTGVmdChQMCwgUDEsIFAyKSB7XHJcbiAgICByZXR1cm4gKFAxLnggLSBQMC54KSAqIChQMi55IC0gUDAueSkgLSAoUDIueCAtIFAwLngpICogKFAxLnkgLSBQMC55KTtcclxufVxyXG5leHBvcnRzLmlzTGVmdCA9IGlzTGVmdDtcclxuZnVuY3Rpb24gYWJvdmUocCwgdmksIHZqKSB7XHJcbiAgICByZXR1cm4gaXNMZWZ0KHAsIHZpLCB2aikgPiAwO1xyXG59XHJcbmZ1bmN0aW9uIGJlbG93KHAsIHZpLCB2aikge1xyXG4gICAgcmV0dXJuIGlzTGVmdChwLCB2aSwgdmopIDwgMDtcclxufVxyXG5mdW5jdGlvbiBDb252ZXhIdWxsKFMpIHtcclxuICAgIHZhciBQID0gUy5zbGljZSgwKS5zb3J0KGZ1bmN0aW9uIChhLCBiKSB7IHJldHVybiBhLnggIT09IGIueCA/IGIueCAtIGEueCA6IGIueSAtIGEueTsgfSk7XHJcbiAgICB2YXIgbiA9IFMubGVuZ3RoLCBpO1xyXG4gICAgdmFyIG1pbm1pbiA9IDA7XHJcbiAgICB2YXIgeG1pbiA9IFBbMF0ueDtcclxuICAgIGZvciAoaSA9IDE7IGkgPCBuOyArK2kpIHtcclxuICAgICAgICBpZiAoUFtpXS54ICE9PSB4bWluKVxyXG4gICAgICAgICAgICBicmVhaztcclxuICAgIH1cclxuICAgIHZhciBtaW5tYXggPSBpIC0gMTtcclxuICAgIHZhciBIID0gW107XHJcbiAgICBILnB1c2goUFttaW5taW5dKTtcclxuICAgIGlmIChtaW5tYXggPT09IG4gLSAxKSB7XHJcbiAgICAgICAgaWYgKFBbbWlubWF4XS55ICE9PSBQW21pbm1pbl0ueSlcclxuICAgICAgICAgICAgSC5wdXNoKFBbbWlubWF4XSk7XHJcbiAgICB9XHJcbiAgICBlbHNlIHtcclxuICAgICAgICB2YXIgbWF4bWluLCBtYXhtYXggPSBuIC0gMTtcclxuICAgICAgICB2YXIgeG1heCA9IFBbbiAtIDFdLng7XHJcbiAgICAgICAgZm9yIChpID0gbiAtIDI7IGkgPj0gMDsgaS0tKVxyXG4gICAgICAgICAgICBpZiAoUFtpXS54ICE9PSB4bWF4KVxyXG4gICAgICAgICAgICAgICAgYnJlYWs7XHJcbiAgICAgICAgbWF4bWluID0gaSArIDE7XHJcbiAgICAgICAgaSA9IG1pbm1heDtcclxuICAgICAgICB3aGlsZSAoKytpIDw9IG1heG1pbikge1xyXG4gICAgICAgICAgICBpZiAoaXNMZWZ0KFBbbWlubWluXSwgUFttYXhtaW5dLCBQW2ldKSA+PSAwICYmIGkgPCBtYXhtaW4pXHJcbiAgICAgICAgICAgICAgICBjb250aW51ZTtcclxuICAgICAgICAgICAgd2hpbGUgKEgubGVuZ3RoID4gMSkge1xyXG4gICAgICAgICAgICAgICAgaWYgKGlzTGVmdChIW0gubGVuZ3RoIC0gMl0sIEhbSC5sZW5ndGggLSAxXSwgUFtpXSkgPiAwKVxyXG4gICAgICAgICAgICAgICAgICAgIGJyZWFrO1xyXG4gICAgICAgICAgICAgICAgZWxzZVxyXG4gICAgICAgICAgICAgICAgICAgIEgubGVuZ3RoIC09IDE7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgaWYgKGkgIT0gbWlubWluKVxyXG4gICAgICAgICAgICAgICAgSC5wdXNoKFBbaV0pO1xyXG4gICAgICAgIH1cclxuICAgICAgICBpZiAobWF4bWF4ICE9IG1heG1pbilcclxuICAgICAgICAgICAgSC5wdXNoKFBbbWF4bWF4XSk7XHJcbiAgICAgICAgdmFyIGJvdCA9IEgubGVuZ3RoO1xyXG4gICAgICAgIGkgPSBtYXhtaW47XHJcbiAgICAgICAgd2hpbGUgKC0taSA+PSBtaW5tYXgpIHtcclxuICAgICAgICAgICAgaWYgKGlzTGVmdChQW21heG1heF0sIFBbbWlubWF4XSwgUFtpXSkgPj0gMCAmJiBpID4gbWlubWF4KVxyXG4gICAgICAgICAgICAgICAgY29udGludWU7XHJcbiAgICAgICAgICAgIHdoaWxlIChILmxlbmd0aCA+IGJvdCkge1xyXG4gICAgICAgICAgICAgICAgaWYgKGlzTGVmdChIW0gubGVuZ3RoIC0gMl0sIEhbSC5sZW5ndGggLSAxXSwgUFtpXSkgPiAwKVxyXG4gICAgICAgICAgICAgICAgICAgIGJyZWFrO1xyXG4gICAgICAgICAgICAgICAgZWxzZVxyXG4gICAgICAgICAgICAgICAgICAgIEgubGVuZ3RoIC09IDE7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgaWYgKGkgIT0gbWlubWluKVxyXG4gICAgICAgICAgICAgICAgSC5wdXNoKFBbaV0pO1xyXG4gICAgICAgIH1cclxuICAgIH1cclxuICAgIHJldHVybiBIO1xyXG59XHJcbmV4cG9ydHMuQ29udmV4SHVsbCA9IENvbnZleEh1bGw7XHJcbmZ1bmN0aW9uIGNsb2Nrd2lzZVJhZGlhbFN3ZWVwKHAsIFAsIGYpIHtcclxuICAgIFAuc2xpY2UoMCkuc29ydChmdW5jdGlvbiAoYSwgYikgeyByZXR1cm4gTWF0aC5hdGFuMihhLnkgLSBwLnksIGEueCAtIHAueCkgLSBNYXRoLmF0YW4yKGIueSAtIHAueSwgYi54IC0gcC54KTsgfSkuZm9yRWFjaChmKTtcclxufVxyXG5leHBvcnRzLmNsb2Nrd2lzZVJhZGlhbFN3ZWVwID0gY2xvY2t3aXNlUmFkaWFsU3dlZXA7XHJcbmZ1bmN0aW9uIG5leHRQb2x5UG9pbnQocCwgcHMpIHtcclxuICAgIGlmIChwLnBvbHlJbmRleCA9PT0gcHMubGVuZ3RoIC0gMSlcclxuICAgICAgICByZXR1cm4gcHNbMF07XHJcbiAgICByZXR1cm4gcHNbcC5wb2x5SW5kZXggKyAxXTtcclxufVxyXG5mdW5jdGlvbiBwcmV2UG9seVBvaW50KHAsIHBzKSB7XHJcbiAgICBpZiAocC5wb2x5SW5kZXggPT09IDApXHJcbiAgICAgICAgcmV0dXJuIHBzW3BzLmxlbmd0aCAtIDFdO1xyXG4gICAgcmV0dXJuIHBzW3AucG9seUluZGV4IC0gMV07XHJcbn1cclxuZnVuY3Rpb24gdGFuZ2VudF9Qb2ludFBvbHlDKFAsIFYpIHtcclxuICAgIHZhciBWY2xvc2VkID0gVi5zbGljZSgwKTtcclxuICAgIFZjbG9zZWQucHVzaChWWzBdKTtcclxuICAgIHJldHVybiB7IHJ0YW46IFJ0YW5nZW50X1BvaW50UG9seUMoUCwgVmNsb3NlZCksIGx0YW46IEx0YW5nZW50X1BvaW50UG9seUMoUCwgVmNsb3NlZCkgfTtcclxufVxyXG5mdW5jdGlvbiBSdGFuZ2VudF9Qb2ludFBvbHlDKFAsIFYpIHtcclxuICAgIHZhciBuID0gVi5sZW5ndGggLSAxO1xyXG4gICAgdmFyIGEsIGIsIGM7XHJcbiAgICB2YXIgdXBBLCBkbkM7XHJcbiAgICBpZiAoYmVsb3coUCwgVlsxXSwgVlswXSkgJiYgIWFib3ZlKFAsIFZbbiAtIDFdLCBWWzBdKSlcclxuICAgICAgICByZXR1cm4gMDtcclxuICAgIGZvciAoYSA9IDAsIGIgPSBuOzspIHtcclxuICAgICAgICBpZiAoYiAtIGEgPT09IDEpXHJcbiAgICAgICAgICAgIGlmIChhYm92ZShQLCBWW2FdLCBWW2JdKSlcclxuICAgICAgICAgICAgICAgIHJldHVybiBhO1xyXG4gICAgICAgICAgICBlbHNlXHJcbiAgICAgICAgICAgICAgICByZXR1cm4gYjtcclxuICAgICAgICBjID0gTWF0aC5mbG9vcigoYSArIGIpIC8gMik7XHJcbiAgICAgICAgZG5DID0gYmVsb3coUCwgVltjICsgMV0sIFZbY10pO1xyXG4gICAgICAgIGlmIChkbkMgJiYgIWFib3ZlKFAsIFZbYyAtIDFdLCBWW2NdKSlcclxuICAgICAgICAgICAgcmV0dXJuIGM7XHJcbiAgICAgICAgdXBBID0gYWJvdmUoUCwgVlthICsgMV0sIFZbYV0pO1xyXG4gICAgICAgIGlmICh1cEEpIHtcclxuICAgICAgICAgICAgaWYgKGRuQylcclxuICAgICAgICAgICAgICAgIGIgPSBjO1xyXG4gICAgICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgICAgIGlmIChhYm92ZShQLCBWW2FdLCBWW2NdKSlcclxuICAgICAgICAgICAgICAgICAgICBiID0gYztcclxuICAgICAgICAgICAgICAgIGVsc2VcclxuICAgICAgICAgICAgICAgICAgICBhID0gYztcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH1cclxuICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgaWYgKCFkbkMpXHJcbiAgICAgICAgICAgICAgICBhID0gYztcclxuICAgICAgICAgICAgZWxzZSB7XHJcbiAgICAgICAgICAgICAgICBpZiAoYmVsb3coUCwgVlthXSwgVltjXSkpXHJcbiAgICAgICAgICAgICAgICAgICAgYiA9IGM7XHJcbiAgICAgICAgICAgICAgICBlbHNlXHJcbiAgICAgICAgICAgICAgICAgICAgYSA9IGM7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICB9XHJcbn1cclxuZnVuY3Rpb24gTHRhbmdlbnRfUG9pbnRQb2x5QyhQLCBWKSB7XHJcbiAgICB2YXIgbiA9IFYubGVuZ3RoIC0gMTtcclxuICAgIHZhciBhLCBiLCBjO1xyXG4gICAgdmFyIGRuQSwgZG5DO1xyXG4gICAgaWYgKGFib3ZlKFAsIFZbbiAtIDFdLCBWWzBdKSAmJiAhYmVsb3coUCwgVlsxXSwgVlswXSkpXHJcbiAgICAgICAgcmV0dXJuIDA7XHJcbiAgICBmb3IgKGEgPSAwLCBiID0gbjs7KSB7XHJcbiAgICAgICAgaWYgKGIgLSBhID09PSAxKVxyXG4gICAgICAgICAgICBpZiAoYmVsb3coUCwgVlthXSwgVltiXSkpXHJcbiAgICAgICAgICAgICAgICByZXR1cm4gYTtcclxuICAgICAgICAgICAgZWxzZVxyXG4gICAgICAgICAgICAgICAgcmV0dXJuIGI7XHJcbiAgICAgICAgYyA9IE1hdGguZmxvb3IoKGEgKyBiKSAvIDIpO1xyXG4gICAgICAgIGRuQyA9IGJlbG93KFAsIFZbYyArIDFdLCBWW2NdKTtcclxuICAgICAgICBpZiAoYWJvdmUoUCwgVltjIC0gMV0sIFZbY10pICYmICFkbkMpXHJcbiAgICAgICAgICAgIHJldHVybiBjO1xyXG4gICAgICAgIGRuQSA9IGJlbG93KFAsIFZbYSArIDFdLCBWW2FdKTtcclxuICAgICAgICBpZiAoZG5BKSB7XHJcbiAgICAgICAgICAgIGlmICghZG5DKVxyXG4gICAgICAgICAgICAgICAgYiA9IGM7XHJcbiAgICAgICAgICAgIGVsc2Uge1xyXG4gICAgICAgICAgICAgICAgaWYgKGJlbG93KFAsIFZbYV0sIFZbY10pKVxyXG4gICAgICAgICAgICAgICAgICAgIGIgPSBjO1xyXG4gICAgICAgICAgICAgICAgZWxzZVxyXG4gICAgICAgICAgICAgICAgICAgIGEgPSBjO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG4gICAgICAgIGVsc2Uge1xyXG4gICAgICAgICAgICBpZiAoZG5DKVxyXG4gICAgICAgICAgICAgICAgYSA9IGM7XHJcbiAgICAgICAgICAgIGVsc2Uge1xyXG4gICAgICAgICAgICAgICAgaWYgKGFib3ZlKFAsIFZbYV0sIFZbY10pKVxyXG4gICAgICAgICAgICAgICAgICAgIGIgPSBjO1xyXG4gICAgICAgICAgICAgICAgZWxzZVxyXG4gICAgICAgICAgICAgICAgICAgIGEgPSBjO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG4gICAgfVxyXG59XHJcbmZ1bmN0aW9uIHRhbmdlbnRfUG9seVBvbHlDKFYsIFcsIHQxLCB0MiwgY21wMSwgY21wMikge1xyXG4gICAgdmFyIGl4MSwgaXgyO1xyXG4gICAgaXgxID0gdDEoV1swXSwgVik7XHJcbiAgICBpeDIgPSB0MihWW2l4MV0sIFcpO1xyXG4gICAgdmFyIGRvbmUgPSBmYWxzZTtcclxuICAgIHdoaWxlICghZG9uZSkge1xyXG4gICAgICAgIGRvbmUgPSB0cnVlO1xyXG4gICAgICAgIHdoaWxlICh0cnVlKSB7XHJcbiAgICAgICAgICAgIGlmIChpeDEgPT09IFYubGVuZ3RoIC0gMSlcclxuICAgICAgICAgICAgICAgIGl4MSA9IDA7XHJcbiAgICAgICAgICAgIGlmIChjbXAxKFdbaXgyXSwgVltpeDFdLCBWW2l4MSArIDFdKSlcclxuICAgICAgICAgICAgICAgIGJyZWFrO1xyXG4gICAgICAgICAgICArK2l4MTtcclxuICAgICAgICB9XHJcbiAgICAgICAgd2hpbGUgKHRydWUpIHtcclxuICAgICAgICAgICAgaWYgKGl4MiA9PT0gMClcclxuICAgICAgICAgICAgICAgIGl4MiA9IFcubGVuZ3RoIC0gMTtcclxuICAgICAgICAgICAgaWYgKGNtcDIoVltpeDFdLCBXW2l4Ml0sIFdbaXgyIC0gMV0pKVxyXG4gICAgICAgICAgICAgICAgYnJlYWs7XHJcbiAgICAgICAgICAgIC0taXgyO1xyXG4gICAgICAgICAgICBkb25lID0gZmFsc2U7XHJcbiAgICAgICAgfVxyXG4gICAgfVxyXG4gICAgcmV0dXJuIHsgdDE6IGl4MSwgdDI6IGl4MiB9O1xyXG59XHJcbmV4cG9ydHMudGFuZ2VudF9Qb2x5UG9seUMgPSB0YW5nZW50X1BvbHlQb2x5QztcclxuZnVuY3Rpb24gTFJ0YW5nZW50X1BvbHlQb2x5QyhWLCBXKSB7XHJcbiAgICB2YXIgcmwgPSBSTHRhbmdlbnRfUG9seVBvbHlDKFcsIFYpO1xyXG4gICAgcmV0dXJuIHsgdDE6IHJsLnQyLCB0MjogcmwudDEgfTtcclxufVxyXG5leHBvcnRzLkxSdGFuZ2VudF9Qb2x5UG9seUMgPSBMUnRhbmdlbnRfUG9seVBvbHlDO1xyXG5mdW5jdGlvbiBSTHRhbmdlbnRfUG9seVBvbHlDKFYsIFcpIHtcclxuICAgIHJldHVybiB0YW5nZW50X1BvbHlQb2x5QyhWLCBXLCBSdGFuZ2VudF9Qb2ludFBvbHlDLCBMdGFuZ2VudF9Qb2ludFBvbHlDLCBhYm92ZSwgYmVsb3cpO1xyXG59XHJcbmV4cG9ydHMuUkx0YW5nZW50X1BvbHlQb2x5QyA9IFJMdGFuZ2VudF9Qb2x5UG9seUM7XHJcbmZ1bmN0aW9uIExMdGFuZ2VudF9Qb2x5UG9seUMoViwgVykge1xyXG4gICAgcmV0dXJuIHRhbmdlbnRfUG9seVBvbHlDKFYsIFcsIEx0YW5nZW50X1BvaW50UG9seUMsIEx0YW5nZW50X1BvaW50UG9seUMsIGJlbG93LCBiZWxvdyk7XHJcbn1cclxuZXhwb3J0cy5MTHRhbmdlbnRfUG9seVBvbHlDID0gTEx0YW5nZW50X1BvbHlQb2x5QztcclxuZnVuY3Rpb24gUlJ0YW5nZW50X1BvbHlQb2x5QyhWLCBXKSB7XHJcbiAgICByZXR1cm4gdGFuZ2VudF9Qb2x5UG9seUMoViwgVywgUnRhbmdlbnRfUG9pbnRQb2x5QywgUnRhbmdlbnRfUG9pbnRQb2x5QywgYWJvdmUsIGFib3ZlKTtcclxufVxyXG5leHBvcnRzLlJSdGFuZ2VudF9Qb2x5UG9seUMgPSBSUnRhbmdlbnRfUG9seVBvbHlDO1xyXG52YXIgQmlUYW5nZW50ID0gKGZ1bmN0aW9uICgpIHtcclxuICAgIGZ1bmN0aW9uIEJpVGFuZ2VudCh0MSwgdDIpIHtcclxuICAgICAgICB0aGlzLnQxID0gdDE7XHJcbiAgICAgICAgdGhpcy50MiA9IHQyO1xyXG4gICAgfVxyXG4gICAgcmV0dXJuIEJpVGFuZ2VudDtcclxufSgpKTtcclxuZXhwb3J0cy5CaVRhbmdlbnQgPSBCaVRhbmdlbnQ7XHJcbnZhciBCaVRhbmdlbnRzID0gKGZ1bmN0aW9uICgpIHtcclxuICAgIGZ1bmN0aW9uIEJpVGFuZ2VudHMoKSB7XHJcbiAgICB9XHJcbiAgICByZXR1cm4gQmlUYW5nZW50cztcclxufSgpKTtcclxuZXhwb3J0cy5CaVRhbmdlbnRzID0gQmlUYW5nZW50cztcclxudmFyIFRWR1BvaW50ID0gKGZ1bmN0aW9uIChfc3VwZXIpIHtcclxuICAgIF9fZXh0ZW5kcyhUVkdQb2ludCwgX3N1cGVyKTtcclxuICAgIGZ1bmN0aW9uIFRWR1BvaW50KCkge1xyXG4gICAgICAgIHJldHVybiBfc3VwZXIgIT09IG51bGwgJiYgX3N1cGVyLmFwcGx5KHRoaXMsIGFyZ3VtZW50cykgfHwgdGhpcztcclxuICAgIH1cclxuICAgIHJldHVybiBUVkdQb2ludDtcclxufShQb2ludCkpO1xyXG5leHBvcnRzLlRWR1BvaW50ID0gVFZHUG9pbnQ7XHJcbnZhciBWaXNpYmlsaXR5VmVydGV4ID0gKGZ1bmN0aW9uICgpIHtcclxuICAgIGZ1bmN0aW9uIFZpc2liaWxpdHlWZXJ0ZXgoaWQsIHBvbHlpZCwgcG9seXZlcnRpZCwgcCkge1xyXG4gICAgICAgIHRoaXMuaWQgPSBpZDtcclxuICAgICAgICB0aGlzLnBvbHlpZCA9IHBvbHlpZDtcclxuICAgICAgICB0aGlzLnBvbHl2ZXJ0aWQgPSBwb2x5dmVydGlkO1xyXG4gICAgICAgIHRoaXMucCA9IHA7XHJcbiAgICAgICAgcC52diA9IHRoaXM7XHJcbiAgICB9XHJcbiAgICByZXR1cm4gVmlzaWJpbGl0eVZlcnRleDtcclxufSgpKTtcclxuZXhwb3J0cy5WaXNpYmlsaXR5VmVydGV4ID0gVmlzaWJpbGl0eVZlcnRleDtcclxudmFyIFZpc2liaWxpdHlFZGdlID0gKGZ1bmN0aW9uICgpIHtcclxuICAgIGZ1bmN0aW9uIFZpc2liaWxpdHlFZGdlKHNvdXJjZSwgdGFyZ2V0KSB7XHJcbiAgICAgICAgdGhpcy5zb3VyY2UgPSBzb3VyY2U7XHJcbiAgICAgICAgdGhpcy50YXJnZXQgPSB0YXJnZXQ7XHJcbiAgICB9XHJcbiAgICBWaXNpYmlsaXR5RWRnZS5wcm90b3R5cGUubGVuZ3RoID0gZnVuY3Rpb24gKCkge1xyXG4gICAgICAgIHZhciBkeCA9IHRoaXMuc291cmNlLnAueCAtIHRoaXMudGFyZ2V0LnAueDtcclxuICAgICAgICB2YXIgZHkgPSB0aGlzLnNvdXJjZS5wLnkgLSB0aGlzLnRhcmdldC5wLnk7XHJcbiAgICAgICAgcmV0dXJuIE1hdGguc3FydChkeCAqIGR4ICsgZHkgKiBkeSk7XHJcbiAgICB9O1xyXG4gICAgcmV0dXJuIFZpc2liaWxpdHlFZGdlO1xyXG59KCkpO1xyXG5leHBvcnRzLlZpc2liaWxpdHlFZGdlID0gVmlzaWJpbGl0eUVkZ2U7XHJcbnZhciBUYW5nZW50VmlzaWJpbGl0eUdyYXBoID0gKGZ1bmN0aW9uICgpIHtcclxuICAgIGZ1bmN0aW9uIFRhbmdlbnRWaXNpYmlsaXR5R3JhcGgoUCwgZzApIHtcclxuICAgICAgICB0aGlzLlAgPSBQO1xyXG4gICAgICAgIHRoaXMuViA9IFtdO1xyXG4gICAgICAgIHRoaXMuRSA9IFtdO1xyXG4gICAgICAgIGlmICghZzApIHtcclxuICAgICAgICAgICAgdmFyIG4gPSBQLmxlbmd0aDtcclxuICAgICAgICAgICAgZm9yICh2YXIgaSA9IDA7IGkgPCBuOyBpKyspIHtcclxuICAgICAgICAgICAgICAgIHZhciBwID0gUFtpXTtcclxuICAgICAgICAgICAgICAgIGZvciAodmFyIGogPSAwOyBqIDwgcC5sZW5ndGg7ICsraikge1xyXG4gICAgICAgICAgICAgICAgICAgIHZhciBwaiA9IHBbal0sIHZ2ID0gbmV3IFZpc2liaWxpdHlWZXJ0ZXgodGhpcy5WLmxlbmd0aCwgaSwgaiwgcGopO1xyXG4gICAgICAgICAgICAgICAgICAgIHRoaXMuVi5wdXNoKHZ2KTtcclxuICAgICAgICAgICAgICAgICAgICBpZiAoaiA+IDApXHJcbiAgICAgICAgICAgICAgICAgICAgICAgIHRoaXMuRS5wdXNoKG5ldyBWaXNpYmlsaXR5RWRnZShwW2ogLSAxXS52diwgdnYpKTtcclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIGlmIChwLmxlbmd0aCA+IDEpXHJcbiAgICAgICAgICAgICAgICAgICAgdGhpcy5FLnB1c2gobmV3IFZpc2liaWxpdHlFZGdlKHBbMF0udnYsIHBbcC5sZW5ndGggLSAxXS52dikpO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIGZvciAodmFyIGkgPSAwOyBpIDwgbiAtIDE7IGkrKykge1xyXG4gICAgICAgICAgICAgICAgdmFyIFBpID0gUFtpXTtcclxuICAgICAgICAgICAgICAgIGZvciAodmFyIGogPSBpICsgMTsgaiA8IG47IGorKykge1xyXG4gICAgICAgICAgICAgICAgICAgIHZhciBQaiA9IFBbal0sIHQgPSB0YW5nZW50cyhQaSwgUGopO1xyXG4gICAgICAgICAgICAgICAgICAgIGZvciAodmFyIHEgaW4gdCkge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICB2YXIgYyA9IHRbcV0sIHNvdXJjZSA9IFBpW2MudDFdLCB0YXJnZXQgPSBQaltjLnQyXTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgdGhpcy5hZGRFZGdlSWZWaXNpYmxlKHNvdXJjZSwgdGFyZ2V0LCBpLCBqKTtcclxuICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICAgICAgZWxzZSB7XHJcbiAgICAgICAgICAgIHRoaXMuViA9IGcwLlYuc2xpY2UoMCk7XHJcbiAgICAgICAgICAgIHRoaXMuRSA9IGcwLkUuc2xpY2UoMCk7XHJcbiAgICAgICAgfVxyXG4gICAgfVxyXG4gICAgVGFuZ2VudFZpc2liaWxpdHlHcmFwaC5wcm90b3R5cGUuYWRkRWRnZUlmVmlzaWJsZSA9IGZ1bmN0aW9uICh1LCB2LCBpMSwgaTIpIHtcclxuICAgICAgICBpZiAoIXRoaXMuaW50ZXJzZWN0c1BvbHlzKG5ldyBMaW5lU2VnbWVudCh1LngsIHUueSwgdi54LCB2LnkpLCBpMSwgaTIpKSB7XHJcbiAgICAgICAgICAgIHRoaXMuRS5wdXNoKG5ldyBWaXNpYmlsaXR5RWRnZSh1LnZ2LCB2LnZ2KSk7XHJcbiAgICAgICAgfVxyXG4gICAgfTtcclxuICAgIFRhbmdlbnRWaXNpYmlsaXR5R3JhcGgucHJvdG90eXBlLmFkZFBvaW50ID0gZnVuY3Rpb24gKHAsIGkxKSB7XHJcbiAgICAgICAgdmFyIG4gPSB0aGlzLlAubGVuZ3RoO1xyXG4gICAgICAgIHRoaXMuVi5wdXNoKG5ldyBWaXNpYmlsaXR5VmVydGV4KHRoaXMuVi5sZW5ndGgsIG4sIDAsIHApKTtcclxuICAgICAgICBmb3IgKHZhciBpID0gMDsgaSA8IG47ICsraSkge1xyXG4gICAgICAgICAgICBpZiAoaSA9PT0gaTEpXHJcbiAgICAgICAgICAgICAgICBjb250aW51ZTtcclxuICAgICAgICAgICAgdmFyIHBvbHkgPSB0aGlzLlBbaV0sIHQgPSB0YW5nZW50X1BvaW50UG9seUMocCwgcG9seSk7XHJcbiAgICAgICAgICAgIHRoaXMuYWRkRWRnZUlmVmlzaWJsZShwLCBwb2x5W3QubHRhbl0sIGkxLCBpKTtcclxuICAgICAgICAgICAgdGhpcy5hZGRFZGdlSWZWaXNpYmxlKHAsIHBvbHlbdC5ydGFuXSwgaTEsIGkpO1xyXG4gICAgICAgIH1cclxuICAgICAgICByZXR1cm4gcC52djtcclxuICAgIH07XHJcbiAgICBUYW5nZW50VmlzaWJpbGl0eUdyYXBoLnByb3RvdHlwZS5pbnRlcnNlY3RzUG9seXMgPSBmdW5jdGlvbiAobCwgaTEsIGkyKSB7XHJcbiAgICAgICAgZm9yICh2YXIgaSA9IDAsIG4gPSB0aGlzLlAubGVuZ3RoOyBpIDwgbjsgKytpKSB7XHJcbiAgICAgICAgICAgIGlmIChpICE9IGkxICYmIGkgIT0gaTIgJiYgaW50ZXJzZWN0cyhsLCB0aGlzLlBbaV0pLmxlbmd0aCA+IDApIHtcclxuICAgICAgICAgICAgICAgIHJldHVybiB0cnVlO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHJldHVybiBmYWxzZTtcclxuICAgIH07XHJcbiAgICByZXR1cm4gVGFuZ2VudFZpc2liaWxpdHlHcmFwaDtcclxufSgpKTtcclxuZXhwb3J0cy5UYW5nZW50VmlzaWJpbGl0eUdyYXBoID0gVGFuZ2VudFZpc2liaWxpdHlHcmFwaDtcclxuZnVuY3Rpb24gaW50ZXJzZWN0cyhsLCBQKSB7XHJcbiAgICB2YXIgaW50cyA9IFtdO1xyXG4gICAgZm9yICh2YXIgaSA9IDEsIG4gPSBQLmxlbmd0aDsgaSA8IG47ICsraSkge1xyXG4gICAgICAgIHZhciBpbnQgPSByZWN0YW5nbGVfMS5SZWN0YW5nbGUubGluZUludGVyc2VjdGlvbihsLngxLCBsLnkxLCBsLngyLCBsLnkyLCBQW2kgLSAxXS54LCBQW2kgLSAxXS55LCBQW2ldLngsIFBbaV0ueSk7XHJcbiAgICAgICAgaWYgKGludClcclxuICAgICAgICAgICAgaW50cy5wdXNoKGludCk7XHJcbiAgICB9XHJcbiAgICByZXR1cm4gaW50cztcclxufVxyXG5mdW5jdGlvbiB0YW5nZW50cyhWLCBXKSB7XHJcbiAgICB2YXIgbSA9IFYubGVuZ3RoIC0gMSwgbiA9IFcubGVuZ3RoIC0gMTtcclxuICAgIHZhciBidCA9IG5ldyBCaVRhbmdlbnRzKCk7XHJcbiAgICBmb3IgKHZhciBpID0gMDsgaSA8IG07ICsraSkge1xyXG4gICAgICAgIGZvciAodmFyIGogPSAwOyBqIDwgbjsgKytqKSB7XHJcbiAgICAgICAgICAgIHZhciB2MSA9IFZbaSA9PSAwID8gbSAtIDEgOiBpIC0gMV07XHJcbiAgICAgICAgICAgIHZhciB2MiA9IFZbaV07XHJcbiAgICAgICAgICAgIHZhciB2MyA9IFZbaSArIDFdO1xyXG4gICAgICAgICAgICB2YXIgdzEgPSBXW2ogPT0gMCA/IG4gLSAxIDogaiAtIDFdO1xyXG4gICAgICAgICAgICB2YXIgdzIgPSBXW2pdO1xyXG4gICAgICAgICAgICB2YXIgdzMgPSBXW2ogKyAxXTtcclxuICAgICAgICAgICAgdmFyIHYxdjJ3MiA9IGlzTGVmdCh2MSwgdjIsIHcyKTtcclxuICAgICAgICAgICAgdmFyIHYydzF3MiA9IGlzTGVmdCh2MiwgdzEsIHcyKTtcclxuICAgICAgICAgICAgdmFyIHYydzJ3MyA9IGlzTGVmdCh2MiwgdzIsIHczKTtcclxuICAgICAgICAgICAgdmFyIHcxdzJ2MiA9IGlzTGVmdCh3MSwgdzIsIHYyKTtcclxuICAgICAgICAgICAgdmFyIHcydjF2MiA9IGlzTGVmdCh3MiwgdjEsIHYyKTtcclxuICAgICAgICAgICAgdmFyIHcydjJ2MyA9IGlzTGVmdCh3MiwgdjIsIHYzKTtcclxuICAgICAgICAgICAgaWYgKHYxdjJ3MiA+PSAwICYmIHYydzF3MiA+PSAwICYmIHYydzJ3MyA8IDBcclxuICAgICAgICAgICAgICAgICYmIHcxdzJ2MiA+PSAwICYmIHcydjF2MiA+PSAwICYmIHcydjJ2MyA8IDApIHtcclxuICAgICAgICAgICAgICAgIGJ0LmxsID0gbmV3IEJpVGFuZ2VudChpLCBqKTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICBlbHNlIGlmICh2MXYydzIgPD0gMCAmJiB2MncxdzIgPD0gMCAmJiB2MncydzMgPiAwXHJcbiAgICAgICAgICAgICAgICAmJiB3MXcydjIgPD0gMCAmJiB3MnYxdjIgPD0gMCAmJiB3MnYydjMgPiAwKSB7XHJcbiAgICAgICAgICAgICAgICBidC5yciA9IG5ldyBCaVRhbmdlbnQoaSwgaik7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgZWxzZSBpZiAodjF2MncyIDw9IDAgJiYgdjJ3MXcyID4gMCAmJiB2MncydzMgPD0gMFxyXG4gICAgICAgICAgICAgICAgJiYgdzF3MnYyID49IDAgJiYgdzJ2MXYyIDwgMCAmJiB3MnYydjMgPj0gMCkge1xyXG4gICAgICAgICAgICAgICAgYnQucmwgPSBuZXcgQmlUYW5nZW50KGksIGopO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIGVsc2UgaWYgKHYxdjJ3MiA+PSAwICYmIHYydzF3MiA8IDAgJiYgdjJ3MnczID49IDBcclxuICAgICAgICAgICAgICAgICYmIHcxdzJ2MiA8PSAwICYmIHcydjF2MiA+IDAgJiYgdzJ2MnYzIDw9IDApIHtcclxuICAgICAgICAgICAgICAgIGJ0LmxyID0gbmV3IEJpVGFuZ2VudChpLCBqKTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH1cclxuICAgIH1cclxuICAgIHJldHVybiBidDtcclxufVxyXG5leHBvcnRzLnRhbmdlbnRzID0gdGFuZ2VudHM7XHJcbmZ1bmN0aW9uIGlzUG9pbnRJbnNpZGVQb2x5KHAsIHBvbHkpIHtcclxuICAgIGZvciAodmFyIGkgPSAxLCBuID0gcG9seS5sZW5ndGg7IGkgPCBuOyArK2kpXHJcbiAgICAgICAgaWYgKGJlbG93KHBvbHlbaSAtIDFdLCBwb2x5W2ldLCBwKSlcclxuICAgICAgICAgICAgcmV0dXJuIGZhbHNlO1xyXG4gICAgcmV0dXJuIHRydWU7XHJcbn1cclxuZnVuY3Rpb24gaXNBbnlQSW5RKHAsIHEpIHtcclxuICAgIHJldHVybiAhcC5ldmVyeShmdW5jdGlvbiAodikgeyByZXR1cm4gIWlzUG9pbnRJbnNpZGVQb2x5KHYsIHEpOyB9KTtcclxufVxyXG5mdW5jdGlvbiBwb2x5c092ZXJsYXAocCwgcSkge1xyXG4gICAgaWYgKGlzQW55UEluUShwLCBxKSlcclxuICAgICAgICByZXR1cm4gdHJ1ZTtcclxuICAgIGlmIChpc0FueVBJblEocSwgcCkpXHJcbiAgICAgICAgcmV0dXJuIHRydWU7XHJcbiAgICBmb3IgKHZhciBpID0gMSwgbiA9IHAubGVuZ3RoOyBpIDwgbjsgKytpKSB7XHJcbiAgICAgICAgdmFyIHYgPSBwW2ldLCB1ID0gcFtpIC0gMV07XHJcbiAgICAgICAgaWYgKGludGVyc2VjdHMobmV3IExpbmVTZWdtZW50KHUueCwgdS55LCB2LngsIHYueSksIHEpLmxlbmd0aCA+IDApXHJcbiAgICAgICAgICAgIHJldHVybiB0cnVlO1xyXG4gICAgfVxyXG4gICAgcmV0dXJuIGZhbHNlO1xyXG59XHJcbmV4cG9ydHMucG9seXNPdmVybGFwID0gcG9seXNPdmVybGFwO1xyXG4vLyMgc291cmNlTWFwcGluZ1VSTD1kYXRhOmFwcGxpY2F0aW9uL2pzb247YmFzZTY0LGV5SjJaWEp6YVc5dUlqb3pMQ0ptYVd4bElqb2laMlZ2YlM1cWN5SXNJbk52ZFhKalpWSnZiM1FpT2lJaUxDSnpiM1Z5WTJWeklqcGJJaTR1THk0dUwxZGxZa052YkdFdmMzSmpMMmRsYjIwdWRITWlYU3dpYm1GdFpYTWlPbHRkTENKdFlYQndhVzVuY3lJNklqczdPenM3T3pzN096czdPenM3TzBGQlFVRXNlVU5CUVhGRE8wRkJRMnBETzBsQlFVRTdTVUZIUVN4RFFVRkRPMGxCUVVRc1dVRkJRenRCUVVGRUxFTkJRVU1zUVVGSVJDeEpRVWRETzBGQlNGa3NjMEpCUVVzN1FVRkxiRUk3U1VGRFNTeHhRa0ZCYlVJc1JVRkJWU3hGUVVGVExFVkJRVlVzUlVGQlV5eEZRVUZWTEVWQlFWTXNSVUZCVlR0UlFVRnVSU3hQUVVGRkxFZEJRVVlzUlVGQlJTeERRVUZSTzFGQlFWTXNUMEZCUlN4SFFVRkdMRVZCUVVVc1EwRkJVVHRSUVVGVExFOUJRVVVzUjBGQlJpeEZRVUZGTEVOQlFWRTdVVUZCVXl4UFFVRkZMRWRCUVVZc1JVRkJSU3hEUVVGUk8wbEJRVWtzUTBGQlF6dEpRVU12Uml4clFrRkJRenRCUVVGRUxFTkJRVU1zUVVGR1JDeEpRVVZETzBGQlJsa3NhME5CUVZjN1FVRkplRUk3U1VGQkswSXNOa0pCUVVzN1NVRkJjRU03TzBsQlJVRXNRMEZCUXp0SlFVRkVMR2RDUVVGRE8wRkJRVVFzUTBGQlF5eEJRVVpFTEVOQlFTdENMRXRCUVVzc1IwRkZia003UVVGR1dTdzRRa0ZCVXp0QlFWVjBRaXhUUVVGblFpeE5RVUZOTEVOQlFVTXNSVUZCVXl4RlFVRkZMRVZCUVZNc1JVRkJSU3hGUVVGVE8wbEJRMnhFTEU5QlFVOHNRMEZCUXl4RlFVRkZMRU5CUVVNc1EwRkJReXhIUVVGSExFVkJRVVVzUTBGQlF5eERRVUZETEVOQlFVTXNSMEZCUnl4RFFVRkRMRVZCUVVVc1EwRkJReXhEUVVGRExFZEJRVWNzUlVGQlJTeERRVUZETEVOQlFVTXNRMEZCUXl4SFFVRkhMRU5CUVVNc1JVRkJSU3hEUVVGRExFTkJRVU1zUjBGQlJ5eEZRVUZGTEVOQlFVTXNRMEZCUXl4RFFVRkRMRWRCUVVjc1EwRkJReXhGUVVGRkxFTkJRVU1zUTBGQlF5eEhRVUZITEVWQlFVVXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJRenRCUVVONlJTeERRVUZETzBGQlJrUXNkMEpCUlVNN1FVRkZSQ3hUUVVGVExFdEJRVXNzUTBGQlF5eERRVUZSTEVWQlFVVXNSVUZCVXl4RlFVRkZMRVZCUVZNN1NVRkRla01zVDBGQlR5eE5RVUZOTEVOQlFVTXNRMEZCUXl4RlFVRkZMRVZCUVVVc1JVRkJSU3hGUVVGRkxFTkJRVU1zUjBGQlJ5eERRVUZETEVOQlFVTTdRVUZEYWtNc1EwRkJRenRCUVVWRUxGTkJRVk1zUzBGQlN5eERRVUZETEVOQlFWRXNSVUZCUlN4RlFVRlRMRVZCUVVVc1JVRkJVenRKUVVONlF5eFBRVUZQTEUxQlFVMHNRMEZCUXl4RFFVRkRMRVZCUVVVc1JVRkJSU3hGUVVGRkxFVkJRVVVzUTBGQlF5eEhRVUZITEVOQlFVTXNRMEZCUXp0QlFVTnFReXhEUVVGRE8wRkJVMFFzVTBGQlowSXNWVUZCVlN4RFFVRkRMRU5CUVZVN1NVRkRha01zU1VGQlNTeERRVUZETEVkQlFVY3NRMEZCUXl4RFFVRkRMRXRCUVVzc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eEpRVUZKTEVOQlFVTXNWVUZCUXl4RFFVRkRMRVZCUVVVc1EwRkJReXhKUVVGTExFOUJRVUVzUTBGQlF5eERRVUZETEVOQlFVTXNTMEZCU3l4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4SFFVRkhMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRWRCUVVjc1EwRkJReXhEUVVGRExFTkJRVU1zUlVGQmJrTXNRMEZCYlVNc1EwRkJReXhEUVVGRE8wbEJRM1pGTEVsQlFVa3NRMEZCUXl4SFFVRkhMRU5CUVVNc1EwRkJReXhOUVVGTkxFVkJRVVVzUTBGQlF5eERRVUZETzBsQlEzQkNMRWxCUVVrc1RVRkJUU3hIUVVGSExFTkJRVU1zUTBGQlF6dEpRVU5tTEVsQlFVa3NTVUZCU1N4SFFVRkhMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTTdTVUZEYkVJc1MwRkJTeXhEUVVGRExFZEJRVWNzUTBGQlF5eEZRVUZGTEVOQlFVTXNSMEZCUnl4RFFVRkRMRVZCUVVVc1JVRkJSU3hEUVVGRExFVkJRVVU3VVVGRGNFSXNTVUZCU1N4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eExRVUZMTEVsQlFVazdXVUZCUlN4TlFVRk5PMHRCUXpsQ08wbEJRMFFzU1VGQlNTeE5RVUZOTEVkQlFVY3NRMEZCUXl4SFFVRkhMRU5CUVVNc1EwRkJRenRKUVVOdVFpeEpRVUZKTEVOQlFVTXNSMEZCV1N4RlFVRkZMRU5CUVVNN1NVRkRjRUlzUTBGQlF5eERRVUZETEVsQlFVa3NRMEZCUXl4RFFVRkRMRU5CUVVNc1RVRkJUU3hEUVVGRExFTkJRVU1zUTBGQlF6dEpRVU5zUWl4SlFVRkpMRTFCUVUwc1MwRkJTeXhEUVVGRExFZEJRVWNzUTBGQlF5eEZRVUZGTzFGQlEyeENMRWxCUVVrc1EwRkJReXhEUVVGRExFMUJRVTBzUTBGQlF5eERRVUZETEVOQlFVTXNTMEZCU3l4RFFVRkRMRU5CUVVNc1RVRkJUU3hEUVVGRExFTkJRVU1zUTBGQlF6dFpRVU16UWl4RFFVRkRMRU5CUVVNc1NVRkJTU3hEUVVGRExFTkJRVU1zUTBGQlF5eE5RVUZOTEVOQlFVTXNRMEZCUXl4RFFVRkRPMHRCUTNwQ08xTkJRVTA3VVVGRlNDeEpRVUZKTEUxQlFVMHNSVUZCUlN4TlFVRk5MRWRCUVVjc1EwRkJReXhIUVVGSExFTkJRVU1zUTBGQlF6dFJRVU16UWl4SlFVRkpMRWxCUVVrc1IwRkJSeXhEUVVGRExFTkJRVU1zUTBGQlF5eEhRVUZITEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJRenRSUVVOMFFpeExRVUZMTEVOQlFVTXNSMEZCUnl4RFFVRkRMRWRCUVVjc1EwRkJReXhGUVVGRkxFTkJRVU1zU1VGQlNTeERRVUZETEVWQlFVVXNRMEZCUXl4RlFVRkZPMWxCUTNaQ0xFbEJRVWtzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1MwRkJTeXhKUVVGSk8yZENRVUZGTEUxQlFVMDdVVUZETDBJc1RVRkJUU3hIUVVGSExFTkJRVU1zUjBGQlJ5eERRVUZETEVOQlFVTTdVVUZIWml4RFFVRkRMRWRCUVVjc1RVRkJUU3hEUVVGRE8xRkJRMWdzVDBGQlR5eEZRVUZGTEVOQlFVTXNTVUZCU1N4TlFVRk5MRVZCUVVVN1dVRkZiRUlzU1VGQlNTeE5RVUZOTEVOQlFVTXNRMEZCUXl4RFFVRkRMRTFCUVUwc1EwRkJReXhGUVVGRkxFTkJRVU1zUTBGQlF5eE5RVUZOTEVOQlFVTXNSVUZCUlN4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zU1VGQlNTeERRVUZETEVsQlFVa3NRMEZCUXl4SFFVRkhMRTFCUVUwN1owSkJRM0pFTEZOQlFWTTdXVUZGWWl4UFFVRlBMRU5CUVVNc1EwRkJReXhOUVVGTkxFZEJRVWNzUTBGQlF5eEZRVU51UWp0blFrRkZTU3hKUVVGSkxFMUJRVTBzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRTFCUVUwc1IwRkJSeXhEUVVGRExFTkJRVU1zUlVGQlJTeERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRTFCUVUwc1IwRkJSeXhEUVVGRExFTkJRVU1zUlVGQlJTeERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1IwRkJSeXhEUVVGRE8yOUNRVU5zUkN4TlFVRk5PenR2UWtGRlRpeERRVUZETEVOQlFVTXNUVUZCVFN4SlFVRkpMRU5CUVVNc1EwRkJRenRoUVVOeVFqdFpRVU5FTEVsQlFVa3NRMEZCUXl4SlFVRkpMRTFCUVUwN1owSkJRVVVzUTBGQlF5eERRVUZETEVsQlFVa3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dFRRVU5xUXp0UlFVZEVMRWxCUVVrc1RVRkJUU3hKUVVGSkxFMUJRVTA3V1VGRGFFSXNRMEZCUXl4RFFVRkRMRWxCUVVrc1EwRkJReXhEUVVGRExFTkJRVU1zVFVGQlRTeERRVUZETEVOQlFVTXNRMEZCUXp0UlFVTjBRaXhKUVVGSkxFZEJRVWNzUjBGQlJ5eERRVUZETEVOQlFVTXNUVUZCVFN4RFFVRkRPMUZCUTI1Q0xFTkJRVU1zUjBGQlJ5eE5RVUZOTEVOQlFVTTdVVUZEV0N4UFFVRlBMRVZCUVVVc1EwRkJReXhKUVVGSkxFMUJRVTBzUlVGQlJUdFpRVVZzUWl4SlFVRkpMRTFCUVUwc1EwRkJReXhEUVVGRExFTkJRVU1zVFVGQlRTeERRVUZETEVWQlFVVXNRMEZCUXl4RFFVRkRMRTFCUVUwc1EwRkJReXhGUVVGRkxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4SlFVRkpMRU5CUVVNc1NVRkJTU3hEUVVGRExFZEJRVWNzVFVGQlRUdG5Ra0ZEY2tRc1UwRkJVenRaUVVWaUxFOUJRVThzUTBGQlF5eERRVUZETEUxQlFVMHNSMEZCUnl4SFFVRkhMRVZCUTNKQ08yZENRVVZKTEVsQlFVa3NUVUZCVFN4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zVFVGQlRTeEhRVUZITEVOQlFVTXNRMEZCUXl4RlFVRkZMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zVFVGQlRTeEhRVUZITEVOQlFVTXNRMEZCUXl4RlFVRkZMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eEhRVUZITEVOQlFVTTdiMEpCUTJ4RUxFMUJRVTA3TzI5Q1FVVk9MRU5CUVVNc1EwRkJReXhOUVVGTkxFbEJRVWtzUTBGQlF5eERRVUZETzJGQlEzSkNPMWxCUTBRc1NVRkJTU3hEUVVGRExFbEJRVWtzVFVGQlRUdG5Ra0ZCUlN4RFFVRkRMRU5CUVVNc1NVRkJTU3hEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRPMU5CUTJwRE8wdEJRMG83U1VGRFJDeFBRVUZQTEVOQlFVTXNRMEZCUXp0QlFVTmlMRU5CUVVNN1FVRTVSRVFzWjBOQk9FUkRPMEZCUjBRc1UwRkJaMElzYjBKQlFXOUNMRU5CUVVNc1EwRkJVU3hGUVVGRkxFTkJRVlVzUlVGQlJTeERRVUZ4UWp0SlFVTTFSU3hEUVVGRExFTkJRVU1zUzBGQlN5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRWxCUVVrc1EwRkRXQ3hWUVVGRExFTkJRVU1zUlVGQlJTeERRVUZETEVsQlFVc3NUMEZCUVN4SlFVRkpMRU5CUVVNc1MwRkJTeXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVkQlFVY3NRMEZCUXl4RFFVRkRMRU5CUVVNc1JVRkJSU3hEUVVGRExFTkJRVU1zUTBGQlF5eEhRVUZITEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1IwRkJSeXhKUVVGSkxFTkJRVU1zUzBGQlN5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRWRCUVVjc1EwRkJReXhEUVVGRExFTkJRVU1zUlVGQlJTeERRVUZETEVOQlFVTXNRMEZCUXl4SFFVRkhMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUlVGQmJrVXNRMEZCYlVVc1EwRkROVVVzUTBGQlF5eFBRVUZQTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNN1FVRkRja0lzUTBGQlF6dEJRVXBFTEc5RVFVbERPMEZCUlVRc1UwRkJVeXhoUVVGaExFTkJRVU1zUTBGQldTeEZRVUZGTEVWQlFXVTdTVUZEYUVRc1NVRkJTU3hEUVVGRExFTkJRVU1zVTBGQlV5eExRVUZMTEVWQlFVVXNRMEZCUXl4TlFVRk5MRWRCUVVjc1EwRkJRenRSUVVGRkxFOUJRVThzUlVGQlJTeERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRPMGxCUTJoRUxFOUJRVThzUlVGQlJTeERRVUZETEVOQlFVTXNRMEZCUXl4VFFVRlRMRWRCUVVjc1EwRkJReXhEUVVGRExFTkJRVU03UVVGREwwSXNRMEZCUXp0QlFVVkVMRk5CUVZNc1lVRkJZU3hEUVVGRExFTkJRVmtzUlVGQlJTeEZRVUZsTzBsQlEyaEVMRWxCUVVrc1EwRkJReXhEUVVGRExGTkJRVk1zUzBGQlN5eERRVUZETzFGQlFVVXNUMEZCVHl4RlFVRkZMRU5CUVVNc1JVRkJSU3hEUVVGRExFMUJRVTBzUjBGQlJ5eERRVUZETEVOQlFVTXNRMEZCUXp0SlFVTm9SQ3hQUVVGUExFVkJRVVVzUTBGQlF5eERRVUZETEVOQlFVTXNVMEZCVXl4SFFVRkhMRU5CUVVNc1EwRkJReXhEUVVGRE8wRkJReTlDTEVOQlFVTTdRVUZSUkN4VFFVRlRMR3RDUVVGclFpeERRVUZETEVOQlFWRXNSVUZCUlN4RFFVRlZPMGxCUnpWRExFbEJRVWtzVDBGQlR5eEhRVUZITEVOQlFVTXNRMEZCUXl4TFFVRkxMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU03U1VGRGVrSXNUMEZCVHl4RFFVRkRMRWxCUVVrc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXp0SlFVVnVRaXhQUVVGUExFVkJRVVVzU1VGQlNTeEZRVUZGTEcxQ1FVRnRRaXhEUVVGRExFTkJRVU1zUlVGQlJTeFBRVUZQTEVOQlFVTXNSVUZCUlN4SlFVRkpMRVZCUVVVc2JVSkJRVzFDTEVOQlFVTXNRMEZCUXl4RlFVRkZMRTlCUVU4c1EwRkJReXhGUVVGRkxFTkJRVU03UVVGRE5VWXNRMEZCUXp0QlFWTkVMRk5CUVZNc2JVSkJRVzFDTEVOQlFVTXNRMEZCVVN4RlFVRkZMRU5CUVZVN1NVRkROME1zU1VGQlNTeERRVUZETEVkQlFVY3NRMEZCUXl4RFFVRkRMRTFCUVUwc1IwRkJSeXhEUVVGRExFTkJRVU03U1VGSGNrSXNTVUZCU1N4RFFVRlRMRVZCUVVVc1EwRkJVeXhGUVVGRkxFTkJRVk1zUTBGQlF6dEpRVU53UXl4SlFVRkpMRWRCUVZrc1JVRkJSU3hIUVVGWkxFTkJRVU03U1VGSkwwSXNTVUZCU1N4TFFVRkxMRU5CUVVNc1EwRkJReXhGUVVGRkxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNSVUZCUlN4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zU1VGQlNTeERRVUZETEV0QlFVc3NRMEZCUXl4RFFVRkRMRVZCUVVVc1EwRkJReXhEUVVGRExFTkJRVU1zUjBGQlJ5eERRVUZETEVOQlFVTXNSVUZCUlN4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU03VVVGRGFrUXNUMEZCVHl4RFFVRkRMRU5CUVVNN1NVRkZZaXhMUVVGTExFTkJRVU1zUjBGQlJ5eERRVUZETEVWQlFVVXNRMEZCUXl4SFFVRkhMRU5CUVVNc1NVRkJTenRSUVVOc1FpeEpRVUZKTEVOQlFVTXNSMEZCUnl4RFFVRkRMRXRCUVVzc1EwRkJRenRaUVVOWUxFbEJRVWtzUzBGQlN5eERRVUZETEVOQlFVTXNSVUZCUlN4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFVkJRVVVzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRPMmRDUVVOd1FpeFBRVUZQTEVOQlFVTXNRMEZCUXpzN1owSkJSVlFzVDBGQlR5eERRVUZETEVOQlFVTTdVVUZGYWtJc1EwRkJReXhIUVVGSExFbEJRVWtzUTBGQlF5eExRVUZMTEVOQlFVTXNRMEZCUXl4RFFVRkRMRWRCUVVjc1EwRkJReXhEUVVGRExFZEJRVWNzUTBGQlF5eERRVUZETEVOQlFVTTdVVUZETlVJc1IwRkJSeXhIUVVGSExFdEJRVXNzUTBGQlF5eERRVUZETEVWQlFVVXNRMEZCUXl4RFFVRkRMRU5CUVVNc1IwRkJSeXhEUVVGRExFTkJRVU1zUlVGQlJTeERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJRenRSUVVNdlFpeEpRVUZKTEVkQlFVY3NTVUZCU1N4RFFVRkRMRXRCUVVzc1EwRkJReXhEUVVGRExFVkJRVVVzUTBGQlF5eERRVUZETEVOQlFVTXNSMEZCUnl4RFFVRkRMRU5CUVVNc1JVRkJSU3hEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTTdXVUZEYUVNc1QwRkJUeXhEUVVGRExFTkJRVU03VVVGSllpeEhRVUZITEVkQlFVY3NTMEZCU3l4RFFVRkRMRU5CUVVNc1JVRkJSU3hEUVVGRExFTkJRVU1zUTBGQlF5eEhRVUZITEVOQlFVTXNRMEZCUXl4RlFVRkZMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETzFGQlF5OUNMRWxCUVVrc1IwRkJSeXhGUVVGRk8xbEJRMHdzU1VGQlNTeEhRVUZITzJkQ1FVTklMRU5CUVVNc1IwRkJSeXhEUVVGRExFTkJRVU03YVVKQlEwdzdaMEpCUTBRc1NVRkJTU3hMUVVGTExFTkJRVU1zUTBGQlF5eEZRVUZGTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1JVRkJSU3hEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTTdiMEpCUTNCQ0xFTkJRVU1zUjBGQlJ5eERRVUZETEVOQlFVTTdPMjlDUVVWT0xFTkJRVU1zUjBGQlJ5eERRVUZETEVOQlFVTTdZVUZEWWp0VFFVTktPMkZCUTBrN1dVRkRSQ3hKUVVGSkxFTkJRVU1zUjBGQlJ6dG5Ra0ZEU2l4RFFVRkRMRWRCUVVjc1EwRkJReXhEUVVGRE8ybENRVU5NTzJkQ1FVTkVMRWxCUVVrc1MwRkJTeXhEUVVGRExFTkJRVU1zUlVGQlJTeERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRVZCUVVVc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETzI5Q1FVTndRaXhEUVVGRExFZEJRVWNzUTBGQlF5eERRVUZET3p0dlFrRkZUaXhEUVVGRExFZEJRVWNzUTBGQlF5eERRVUZETzJGQlEySTdVMEZEU2p0TFFVTktPMEZCUTB3c1EwRkJRenRCUVZGRUxGTkJRVk1zYlVKQlFXMUNMRU5CUVVNc1EwRkJVU3hGUVVGRkxFTkJRVlU3U1VGRE4wTXNTVUZCU1N4RFFVRkRMRWRCUVVjc1EwRkJReXhEUVVGRExFMUJRVTBzUjBGQlJ5eERRVUZETEVOQlFVTTdTVUZGY2tJc1NVRkJTU3hEUVVGVExFVkJRVVVzUTBGQlV5eEZRVUZGTEVOQlFWTXNRMEZCUXp0SlFVTndReXhKUVVGSkxFZEJRVmtzUlVGQlJTeEhRVUZaTEVOQlFVTTdTVUZKTDBJc1NVRkJTU3hMUVVGTExFTkJRVU1zUTBGQlF5eEZRVUZGTEVOQlFVTXNRMEZCUXl4RFFVRkRMRWRCUVVjc1EwRkJReXhEUVVGRExFVkJRVVVzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRWxCUVVrc1EwRkJReXhMUVVGTExFTkJRVU1zUTBGQlF5eEZRVUZGTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1JVRkJSU3hEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTTdVVUZEYWtRc1QwRkJUeXhEUVVGRExFTkJRVU03U1VGRllpeExRVUZMTEVOQlFVTXNSMEZCUnl4RFFVRkRMRVZCUVVVc1EwRkJReXhIUVVGSExFTkJRVU1zU1VGQlN6dFJRVU5zUWl4SlFVRkpMRU5CUVVNc1IwRkJSeXhEUVVGRExFdEJRVXNzUTBGQlF6dFpRVU5ZTEVsQlFVa3NTMEZCU3l4RFFVRkRMRU5CUVVNc1JVRkJSU3hEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVWQlFVVXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRE8yZENRVU53UWl4UFFVRlBMRU5CUVVNc1EwRkJRenM3WjBKQlJWUXNUMEZCVHl4RFFVRkRMRU5CUVVNN1VVRkZha0lzUTBGQlF5eEhRVUZITEVsQlFVa3NRMEZCUXl4TFFVRkxMRU5CUVVNc1EwRkJReXhEUVVGRExFZEJRVWNzUTBGQlF5eERRVUZETEVkQlFVY3NRMEZCUXl4RFFVRkRMRU5CUVVNN1VVRkROVUlzUjBGQlJ5eEhRVUZITEV0QlFVc3NRMEZCUXl4RFFVRkRMRVZCUVVVc1EwRkJReXhEUVVGRExFTkJRVU1zUjBGQlJ5eERRVUZETEVOQlFVTXNSVUZCUlN4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dFJRVU12UWl4SlFVRkpMRXRCUVVzc1EwRkJReXhEUVVGRExFVkJRVVVzUTBGQlF5eERRVUZETEVOQlFVTXNSMEZCUnl4RFFVRkRMRU5CUVVNc1JVRkJSU3hEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNTVUZCU1N4RFFVRkRMRWRCUVVjN1dVRkRhRU1zVDBGQlR5eERRVUZETEVOQlFVTTdVVUZKWWl4SFFVRkhMRWRCUVVjc1MwRkJTeXhEUVVGRExFTkJRVU1zUlVGQlJTeERRVUZETEVOQlFVTXNRMEZCUXl4SFFVRkhMRU5CUVVNc1EwRkJReXhGUVVGRkxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRPMUZCUXk5Q0xFbEJRVWtzUjBGQlJ5eEZRVUZGTzFsQlEwd3NTVUZCU1N4RFFVRkRMRWRCUVVjN1owSkJRMG9zUTBGQlF5eEhRVUZITEVOQlFVTXNRMEZCUXp0cFFrRkRURHRuUWtGRFJDeEpRVUZKTEV0QlFVc3NRMEZCUXl4RFFVRkRMRVZCUVVVc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eEZRVUZGTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJRenR2UWtGRGNFSXNRMEZCUXl4SFFVRkhMRU5CUVVNc1EwRkJRenM3YjBKQlJVNHNRMEZCUXl4SFFVRkhMRU5CUVVNc1EwRkJRenRoUVVOaU8xTkJRMG83WVVGRFNUdFpRVU5FTEVsQlFVa3NSMEZCUnp0blFrRkRTQ3hEUVVGRExFZEJRVWNzUTBGQlF5eERRVUZETzJsQ1FVTk1PMmRDUVVORUxFbEJRVWtzUzBGQlN5eERRVUZETEVOQlFVTXNSVUZCUlN4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFVkJRVVVzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRPMjlDUVVOd1FpeERRVUZETEVkQlFVY3NRMEZCUXl4RFFVRkRPenR2UWtGRlRpeERRVUZETEVkQlFVY3NRMEZCUXl4RFFVRkRPMkZCUTJJN1UwRkRTanRMUVVOS08wRkJRMHdzUTBGQlF6dEJRVk5FTEZOQlFXZENMR2xDUVVGcFFpeERRVUZETEVOQlFWVXNSVUZCUlN4RFFVRlZMRVZCUVVVc1JVRkJiME1zUlVGQlJTeEZRVUZ2UXl4RlFVRkZMRWxCUVN0RExFVkJRVVVzU1VGQkswTTdTVUZEYkU4c1NVRkJTU3hIUVVGWExFVkJRVVVzUjBGQlZ5eERRVUZETzBsQlJ6ZENMRWRCUVVjc1IwRkJSeXhGUVVGRkxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RlFVRkZMRU5CUVVNc1EwRkJReXhEUVVGRE8wbEJRMnhDTEVkQlFVY3NSMEZCUnl4RlFVRkZMRU5CUVVNc1EwRkJReXhEUVVGRExFZEJRVWNzUTBGQlF5eEZRVUZGTEVOQlFVTXNRMEZCUXl4RFFVRkRPMGxCUjNCQ0xFbEJRVWtzU1VGQlNTeEhRVUZITEV0QlFVc3NRMEZCUXp0SlFVTnFRaXhQUVVGUExFTkJRVU1zU1VGQlNTeEZRVUZGTzFGQlExWXNTVUZCU1N4SFFVRkhMRWxCUVVrc1EwRkJRenRSUVVOYUxFOUJRVThzU1VGQlNTeEZRVUZGTzFsQlExUXNTVUZCU1N4SFFVRkhMRXRCUVVzc1EwRkJReXhEUVVGRExFMUJRVTBzUjBGQlJ5eERRVUZETzJkQ1FVRkZMRWRCUVVjc1IwRkJSeXhEUVVGRExFTkJRVU03V1VGRGJFTXNTVUZCU1N4SlFVRkpMRU5CUVVNc1EwRkJReXhEUVVGRExFZEJRVWNzUTBGQlF5eEZRVUZGTEVOQlFVTXNRMEZCUXl4SFFVRkhMRU5CUVVNc1JVRkJSU3hEUVVGRExFTkJRVU1zUjBGQlJ5eEhRVUZITEVOQlFVTXNRMEZCUXl4RFFVRkRPMmRDUVVGRkxFMUJRVTA3V1VGRE5VTXNSVUZCUlN4SFFVRkhMRU5CUVVNN1UwRkRWRHRSUVVORUxFOUJRVThzU1VGQlNTeEZRVUZGTzFsQlExUXNTVUZCU1N4SFFVRkhMRXRCUVVzc1EwRkJRenRuUWtGQlJTeEhRVUZITEVkQlFVY3NRMEZCUXl4RFFVRkRMRTFCUVUwc1IwRkJSeXhEUVVGRExFTkJRVU03V1VGRGJFTXNTVUZCU1N4SlFVRkpMRU5CUVVNc1EwRkJReXhEUVVGRExFZEJRVWNzUTBGQlF5eEZRVUZGTEVOQlFVTXNRMEZCUXl4SFFVRkhMRU5CUVVNc1JVRkJSU3hEUVVGRExFTkJRVU1zUjBGQlJ5eEhRVUZITEVOQlFVTXNRMEZCUXl4RFFVRkRPMmRDUVVGRkxFMUJRVTA3V1VGRE5VTXNSVUZCUlN4SFFVRkhMRU5CUVVNN1dVRkRUaXhKUVVGSkxFZEJRVWNzUzBGQlN5eERRVUZETzFOQlEyaENPMHRCUTBvN1NVRkRSQ3hQUVVGUExFVkJRVVVzUlVGQlJTeEZRVUZGTEVkQlFVY3NSVUZCUlN4RlFVRkZMRVZCUVVVc1IwRkJSeXhGUVVGRkxFTkJRVU03UVVGRGFFTXNRMEZCUXp0QlFYaENSQ3c0UTBGM1FrTTdRVUZGUkN4VFFVRm5RaXh0UWtGQmJVSXNRMEZCUXl4RFFVRlZMRVZCUVVVc1EwRkJWVHRKUVVOMFJDeEpRVUZKTEVWQlFVVXNSMEZCUnl4dFFrRkJiVUlzUTBGQlF5eERRVUZETEVWQlFVVXNRMEZCUXl4RFFVRkRMRU5CUVVNN1NVRkRia01zVDBGQlR5eEZRVUZGTEVWQlFVVXNSVUZCUlN4RlFVRkZMRU5CUVVNc1JVRkJSU3hGUVVGRkxFVkJRVVVzUlVGQlJTeEZRVUZGTEVOQlFVTXNSVUZCUlN4RlFVRkZMRU5CUVVNN1FVRkRjRU1zUTBGQlF6dEJRVWhFTEd0RVFVZERPMEZCUlVRc1UwRkJaMElzYlVKQlFXMUNMRU5CUVVNc1EwRkJWU3hGUVVGRkxFTkJRVlU3U1VGRGRFUXNUMEZCVHl4cFFrRkJhVUlzUTBGQlF5eERRVUZETEVWQlFVVXNRMEZCUXl4RlFVRkZMRzFDUVVGdFFpeEZRVUZGTEcxQ1FVRnRRaXhGUVVGRkxFdEJRVXNzUlVGQlJTeExRVUZMTEVOQlFVTXNRMEZCUXp0QlFVTXpSaXhEUVVGRE8wRkJSa1FzYTBSQlJVTTdRVUZGUkN4VFFVRm5RaXh0UWtGQmJVSXNRMEZCUXl4RFFVRlZMRVZCUVVVc1EwRkJWVHRKUVVOMFJDeFBRVUZQTEdsQ1FVRnBRaXhEUVVGRExFTkJRVU1zUlVGQlJTeERRVUZETEVWQlFVVXNiVUpCUVcxQ0xFVkJRVVVzYlVKQlFXMUNMRVZCUVVVc1MwRkJTeXhGUVVGRkxFdEJRVXNzUTBGQlF5eERRVUZETzBGQlF6TkdMRU5CUVVNN1FVRkdSQ3hyUkVGRlF6dEJRVVZFTEZOQlFXZENMRzFDUVVGdFFpeERRVUZETEVOQlFWVXNSVUZCUlN4RFFVRlZPMGxCUTNSRUxFOUJRVThzYVVKQlFXbENMRU5CUVVNc1EwRkJReXhGUVVGRkxFTkJRVU1zUlVGQlJTeHRRa0ZCYlVJc1JVRkJSU3h0UWtGQmJVSXNSVUZCUlN4TFFVRkxMRVZCUVVVc1MwRkJTeXhEUVVGRExFTkJRVU03UVVGRE0wWXNRMEZCUXp0QlFVWkVMR3RFUVVWRE8wRkJSVVE3U1VGRFNTeHRRa0ZCYlVJc1JVRkJWU3hGUVVGVExFVkJRVlU3VVVGQk4wSXNUMEZCUlN4SFFVRkdMRVZCUVVVc1EwRkJVVHRSUVVGVExFOUJRVVVzUjBGQlJpeEZRVUZGTEVOQlFWRTdTVUZCU1N4RFFVRkRPMGxCUTNwRUxHZENRVUZETzBGQlFVUXNRMEZCUXl4QlFVWkVMRWxCUlVNN1FVRkdXU3c0UWtGQlV6dEJRVWwwUWp0SlFVRkJPMGxCUzBFc1EwRkJRenRKUVVGRUxHbENRVUZETzBGQlFVUXNRMEZCUXl4QlFVeEVMRWxCUzBNN1FVRk1XU3huUTBGQlZUdEJRVTkyUWp0SlFVRTRRaXcwUWtGQlN6dEpRVUZ1UXpzN1NVRkZRU3hEUVVGRE8wbEJRVVFzWlVGQlF6dEJRVUZFTEVOQlFVTXNRVUZHUkN4RFFVRTRRaXhMUVVGTExFZEJSV3hETzBGQlJsa3NORUpCUVZFN1FVRkpja0k3U1VGRFNTd3dRa0ZEVnl4RlFVRlZMRVZCUTFZc1RVRkJZeXhGUVVOa0xGVkJRV3RDTEVWQlEyeENMRU5CUVZjN1VVRklXQ3hQUVVGRkxFZEJRVVlzUlVGQlJTeERRVUZSTzFGQlExWXNWMEZCVFN4SFFVRk9MRTFCUVUwc1EwRkJVVHRSUVVOa0xHVkJRVlVzUjBGQlZpeFZRVUZWTEVOQlFWRTdVVUZEYkVJc1RVRkJReXhIUVVGRUxFTkJRVU1zUTBGQlZUdFJRVVZzUWl4RFFVRkRMRU5CUVVNc1JVRkJSU3hIUVVGSExFbEJRVWtzUTBGQlF6dEpRVU5vUWl4RFFVRkRPMGxCUTB3c2RVSkJRVU03UVVGQlJDeERRVUZETEVGQlZFUXNTVUZUUXp0QlFWUlpMRFJEUVVGblFqdEJRVmMzUWp0SlFVTkpMSGRDUVVOWExFMUJRWGRDTEVWQlEzaENMRTFCUVhkQ08xRkJSSGhDTEZkQlFVMHNSMEZCVGl4TlFVRk5MRU5CUVd0Q08xRkJRM2hDTEZkQlFVMHNSMEZCVGl4TlFVRk5MRU5CUVd0Q08wbEJRVWtzUTBGQlF6dEpRVU40UXl3clFrRkJUU3hIUVVGT08xRkJRMGtzU1VGQlNTeEZRVUZGTEVkQlFVY3NTVUZCU1N4RFFVRkRMRTFCUVUwc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eEhRVUZITEVsQlFVa3NRMEZCUXl4TlFVRk5MRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dFJRVU16UXl4SlFVRkpMRVZCUVVVc1IwRkJSeXhKUVVGSkxFTkJRVU1zVFVGQlRTeERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRWRCUVVjc1NVRkJTU3hEUVVGRExFMUJRVTBzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRPMUZCUXpORExFOUJRVThzU1VGQlNTeERRVUZETEVsQlFVa3NRMEZCUXl4RlFVRkZMRWRCUVVjc1JVRkJSU3hIUVVGSExFVkJRVVVzUjBGQlJ5eEZRVUZGTEVOQlFVTXNRMEZCUXp0SlFVTjRReXhEUVVGRE8wbEJRMHdzY1VKQlFVTTdRVUZCUkN4RFFVRkRMRUZCVkVRc1NVRlRRenRCUVZSWkxIZERRVUZqTzBGQlZ6TkNPMGxCUjBrc1owTkJRVzFDTEVOQlFXVXNSVUZCUlN4RlFVRnRSRHRSUVVGd1JTeE5RVUZETEVkQlFVUXNRMEZCUXl4RFFVRmpPMUZCUm14RExFMUJRVU1zUjBGQmRVSXNSVUZCUlN4RFFVRkRPMUZCUXpOQ0xFMUJRVU1zUjBGQmNVSXNSVUZCUlN4RFFVRkRPMUZCUlhKQ0xFbEJRVWtzUTBGQlF5eEZRVUZGTEVWQlFVVTdXVUZEVEN4SlFVRkpMRU5CUVVNc1IwRkJSeXhEUVVGRExFTkJRVU1zVFVGQlRTeERRVUZETzFsQlJXcENMRXRCUVVzc1NVRkJTU3hEUVVGRExFZEJRVWNzUTBGQlF5eEZRVUZGTEVOQlFVTXNSMEZCUnl4RFFVRkRMRVZCUVVVc1EwRkJReXhGUVVGRkxFVkJRVVU3WjBKQlEzaENMRWxCUVVrc1EwRkJReXhIUVVGSExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXp0blFrRkZZaXhMUVVGTExFbEJRVWtzUTBGQlF5eEhRVUZITEVOQlFVTXNSVUZCUlN4RFFVRkRMRWRCUVVjc1EwRkJReXhEUVVGRExFMUJRVTBzUlVGQlJTeEZRVUZGTEVOQlFVTXNSVUZCUlR0dlFrRkRMMElzU1VGQlNTeEZRVUZGTEVkQlFVY3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhGUVVOVUxFVkJRVVVzUjBGQlJ5eEpRVUZKTEdkQ1FVRm5RaXhEUVVGRExFbEJRVWtzUTBGQlF5eERRVUZETEVOQlFVTXNUVUZCVFN4RlFVRkZMRU5CUVVNc1JVRkJSU3hEUVVGRExFVkJRVVVzUlVGQlJTeERRVUZETEVOQlFVTTdiMEpCUTNaRUxFbEJRVWtzUTBGQlF5eERRVUZETEVOQlFVTXNTVUZCU1N4RFFVRkRMRVZCUVVVc1EwRkJReXhEUVVGRE8yOUNRVWxvUWl4SlFVRkpMRU5CUVVNc1IwRkJSeXhEUVVGRE8zZENRVUZGTEVsQlFVa3NRMEZCUXl4RFFVRkRMRU5CUVVNc1NVRkJTU3hEUVVGRExFbEJRVWtzWTBGQll5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRWRCUVVjc1EwRkJReXhEUVVGRExFTkJRVU1zUlVGQlJTeEZRVUZGTEVWQlFVVXNRMEZCUXl4RFFVRkRMRU5CUVVNN2FVSkJReTlFTzJkQ1FVVkVMRWxCUVVrc1EwRkJReXhEUVVGRExFMUJRVTBzUjBGQlJ5eERRVUZETzI5Q1FVRkZMRWxCUVVrc1EwRkJReXhEUVVGRExFTkJRVU1zU1VGQlNTeERRVUZETEVsQlFVa3NZMEZCWXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eEZRVUZGTEVWQlFVVXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhOUVVGTkxFZEJRVWNzUTBGQlF5eERRVUZETEVOQlFVTXNSVUZCUlN4RFFVRkRMRU5CUVVNc1EwRkJRenRoUVVOc1JqdFpRVU5FTEV0QlFVc3NTVUZCU1N4RFFVRkRMRWRCUVVjc1EwRkJReXhGUVVGRkxFTkJRVU1zUjBGQlJ5eERRVUZETEVkQlFVY3NRMEZCUXl4RlFVRkZMRU5CUVVNc1JVRkJSU3hGUVVGRk8yZENRVU0xUWl4SlFVRkpMRVZCUVVVc1IwRkJSeXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTTdaMEpCUTJRc1MwRkJTeXhKUVVGSkxFTkJRVU1zUjBGQlJ5eERRVUZETEVkQlFVY3NRMEZCUXl4RlFVRkZMRU5CUVVNc1IwRkJSeXhEUVVGRExFVkJRVVVzUTBGQlF5eEZRVUZGTEVWQlFVVTdiMEpCUXpWQ0xFbEJRVWtzUlVGQlJTeEhRVUZITEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1JVRkRWQ3hEUVVGRExFZEJRVWNzVVVGQlVTeERRVUZETEVWQlFVVXNSVUZCUlN4RlFVRkZMRU5CUVVNc1EwRkJRenR2UWtGRGVrSXNTMEZCU3l4SlFVRkpMRU5CUVVNc1NVRkJTU3hEUVVGRExFVkJRVVU3ZDBKQlEySXNTVUZCU1N4RFFVRkRMRWRCUVVjc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eEZRVU5TTEUxQlFVMHNSMEZCUnl4RlFVRkZMRU5CUVVNc1EwRkJReXhEUVVGRExFVkJRVVVzUTBGQlF5eEZRVUZGTEUxQlFVMHNSMEZCUnl4RlFVRkZMRU5CUVVNc1EwRkJReXhEUVVGRExFVkJRVVVzUTBGQlF5eERRVUZETzNkQ1FVTjZReXhKUVVGSkxFTkJRVU1zWjBKQlFXZENMRU5CUVVNc1RVRkJUU3hGUVVGRkxFMUJRVTBzUlVGQlJTeERRVUZETEVWQlFVVXNRMEZCUXl4RFFVRkRMRU5CUVVNN2NVSkJReTlETzJsQ1FVTktPMkZCUTBvN1UwRkRTanRoUVVGTk8xbEJRMGdzU1VGQlNTeERRVUZETEVOQlFVTXNSMEZCUnl4RlFVRkZMRU5CUVVNc1EwRkJReXhEUVVGRExFdEJRVXNzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXp0WlFVTjJRaXhKUVVGSkxFTkJRVU1zUTBGQlF5eEhRVUZITEVWQlFVVXNRMEZCUXl4RFFVRkRMRU5CUVVNc1MwRkJTeXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETzFOQlF6RkNPMGxCUTB3c1EwRkJRenRKUVVORUxHbEVRVUZuUWl4SFFVRm9RaXhWUVVGcFFpeERRVUZYTEVWQlFVVXNRMEZCVnl4RlFVRkZMRVZCUVZVc1JVRkJSU3hGUVVGVk8xRkJRemRFTEVsQlFVa3NRMEZCUXl4SlFVRkpMRU5CUVVNc1pVRkJaU3hEUVVGRExFbEJRVWtzVjBGQlZ5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRVZCUVVVc1EwRkJReXhEUVVGRExFTkJRVU1zUlVGQlJTeERRVUZETEVOQlFVTXNRMEZCUXl4RlFVRkZMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUlVGQlJTeEZRVUZGTEVWQlFVVXNSVUZCUlN4RFFVRkRMRVZCUVVVN1dVRkRjRVVzU1VGQlNTeERRVUZETEVOQlFVTXNRMEZCUXl4SlFVRkpMRU5CUVVNc1NVRkJTU3hqUVVGakxFTkJRVU1zUTBGQlF5eERRVUZETEVWQlFVVXNSVUZCUlN4RFFVRkRMRU5CUVVNc1JVRkJSU3hEUVVGRExFTkJRVU1zUTBGQlF6dFRRVU12UXp0SlFVTk1MRU5CUVVNN1NVRkRSQ3g1UTBGQlVTeEhRVUZTTEZWQlFWTXNRMEZCVnl4RlFVRkZMRVZCUVZVN1VVRkROVUlzU1VGQlNTeERRVUZETEVkQlFVY3NTVUZCU1N4RFFVRkRMRU5CUVVNc1EwRkJReXhOUVVGTkxFTkJRVU03VVVGRGRFSXNTVUZCU1N4RFFVRkRMRU5CUVVNc1EwRkJReXhKUVVGSkxFTkJRVU1zU1VGQlNTeG5Ra0ZCWjBJc1EwRkJReXhKUVVGSkxFTkJRVU1zUTBGQlF5eERRVUZETEUxQlFVMHNSVUZCUlN4RFFVRkRMRVZCUVVVc1EwRkJReXhGUVVGRkxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTTdVVUZETVVRc1MwRkJTeXhKUVVGSkxFTkJRVU1zUjBGQlJ5eERRVUZETEVWQlFVVXNRMEZCUXl4SFFVRkhMRU5CUVVNc1JVRkJSU3hGUVVGRkxFTkJRVU1zUlVGQlJUdFpRVU40UWl4SlFVRkpMRU5CUVVNc1MwRkJTeXhGUVVGRk8yZENRVUZGTEZOQlFWTTdXVUZEZGtJc1NVRkJTU3hKUVVGSkxFZEJRVWNzU1VGQlNTeERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1JVRkRhRUlzUTBGQlF5eEhRVUZITEd0Q1FVRnJRaXhEUVVGRExFTkJRVU1zUlVGQlJTeEpRVUZKTEVOQlFVTXNRMEZCUXp0WlFVTndReXhKUVVGSkxFTkJRVU1zWjBKQlFXZENMRU5CUVVNc1EwRkJReXhGUVVGRkxFbEJRVWtzUTBGQlF5eERRVUZETEVOQlFVTXNTVUZCU1N4RFFVRkRMRVZCUVVVc1JVRkJSU3hGUVVGRkxFTkJRVU1zUTBGQlF5eERRVUZETzFsQlF6bERMRWxCUVVrc1EwRkJReXhuUWtGQlowSXNRMEZCUXl4RFFVRkRMRVZCUVVVc1NVRkJTU3hEUVVGRExFTkJRVU1zUTBGQlF5eEpRVUZKTEVOQlFVTXNSVUZCUlN4RlFVRkZMRVZCUVVVc1EwRkJReXhEUVVGRExFTkJRVU03VTBGRGFrUTdVVUZEUkN4UFFVRlBMRU5CUVVNc1EwRkJReXhGUVVGRkxFTkJRVU03U1VGRGFFSXNRMEZCUXp0SlFVTlBMR2RFUVVGbExFZEJRWFpDTEZWQlFYZENMRU5CUVdNc1JVRkJSU3hGUVVGVkxFVkJRVVVzUlVGQlZUdFJRVU14UkN4TFFVRkxMRWxCUVVrc1EwRkJReXhIUVVGSExFTkJRVU1zUlVGQlJTeERRVUZETEVkQlFVY3NTVUZCU1N4RFFVRkRMRU5CUVVNc1EwRkJReXhOUVVGTkxFVkJRVVVzUTBGQlF5eEhRVUZITEVOQlFVTXNSVUZCUlN4RlFVRkZMRU5CUVVNc1JVRkJSVHRaUVVNelF5eEpRVUZKTEVOQlFVTXNTVUZCU1N4RlFVRkZMRWxCUVVrc1EwRkJReXhKUVVGSkxFVkJRVVVzU1VGQlNTeFZRVUZWTEVOQlFVTXNRMEZCUXl4RlFVRkZMRWxCUVVrc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4TlFVRk5MRWRCUVVjc1EwRkJReXhGUVVGRk8yZENRVU16UkN4UFFVRlBMRWxCUVVrc1EwRkJRenRoUVVObU8xTkJRMG83VVVGRFJDeFBRVUZQTEV0QlFVc3NRMEZCUXp0SlFVTnFRaXhEUVVGRE8wbEJRMHdzTmtKQlFVTTdRVUZCUkN4RFFVRkRMRUZCYUVWRUxFbEJaMFZETzBGQmFFVlpMSGRFUVVGelFqdEJRV3RGYmtNc1UwRkJVeXhWUVVGVkxFTkJRVU1zUTBGQll5eEZRVUZGTEVOQlFWVTdTVUZETVVNc1NVRkJTU3hKUVVGSkxFZEJRVWNzUlVGQlJTeERRVUZETzBsQlEyUXNTMEZCU3l4SlFVRkpMRU5CUVVNc1IwRkJSeXhEUVVGRExFVkJRVVVzUTBGQlF5eEhRVUZITEVOQlFVTXNRMEZCUXl4TlFVRk5MRVZCUVVVc1EwRkJReXhIUVVGSExFTkJRVU1zUlVGQlJTeEZRVUZGTEVOQlFVTXNSVUZCUlR0UlFVTjBReXhKUVVGSkxFZEJRVWNzUjBGQlJ5eHhRa0ZCVXl4RFFVRkRMR2RDUVVGblFpeERRVU5vUXl4RFFVRkRMRU5CUVVNc1JVRkJSU3hGUVVGRkxFTkJRVU1zUTBGQlF5eEZRVUZGTEVWQlExWXNRMEZCUXl4RFFVRkRMRVZCUVVVc1JVRkJSU3hEUVVGRExFTkJRVU1zUlVGQlJTeEZRVU5XTEVOQlFVTXNRMEZCUXl4RFFVRkRMRWRCUVVjc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eEZRVUZGTEVOQlFVTXNRMEZCUXl4RFFVRkRMRWRCUVVjc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eEZRVU4wUWl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eEZRVUZGTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRMklzUTBGQlF6dFJRVU5PTEVsQlFVa3NSMEZCUnp0WlFVRkZMRWxCUVVrc1EwRkJReXhKUVVGSkxFTkJRVU1zUjBGQlJ5eERRVUZETEVOQlFVTTdTMEZETTBJN1NVRkRSQ3hQUVVGUExFbEJRVWtzUTBGQlF6dEJRVU5vUWl4RFFVRkRPMEZCUlVRc1UwRkJaMElzVVVGQlVTeERRVUZETEVOQlFWVXNSVUZCUlN4RFFVRlZPMGxCUlRORExFbEJRVWtzUTBGQlF5eEhRVUZITEVOQlFVTXNRMEZCUXl4TlFVRk5MRWRCUVVjc1EwRkJReXhGUVVGRkxFTkJRVU1zUjBGQlJ5eERRVUZETEVOQlFVTXNUVUZCVFN4SFFVRkhMRU5CUVVNc1EwRkJRenRKUVVOMlF5eEpRVUZKTEVWQlFVVXNSMEZCUnl4SlFVRkpMRlZCUVZVc1JVRkJSU3hEUVVGRE8wbEJRekZDTEV0QlFVc3NTVUZCU1N4RFFVRkRMRWRCUVVjc1EwRkJReXhGUVVGRkxFTkJRVU1zUjBGQlJ5eERRVUZETEVWQlFVVXNSVUZCUlN4RFFVRkRMRVZCUVVVN1VVRkRlRUlzUzBGQlN5eEpRVUZKTEVOQlFVTXNSMEZCUnl4RFFVRkRMRVZCUVVVc1EwRkJReXhIUVVGSExFTkJRVU1zUlVGQlJTeEZRVUZGTEVOQlFVTXNSVUZCUlR0WlFVTjRRaXhKUVVGSkxFVkJRVVVzUjBGQlJ5eERRVUZETEVOQlFVTXNRMEZCUXl4SlFVRkpMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eEhRVUZITEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhIUVVGSExFTkJRVU1zUTBGQlF5eERRVUZETzFsQlEyNURMRWxCUVVrc1JVRkJSU3hIUVVGSExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXp0WlFVTmtMRWxCUVVrc1JVRkJSU3hIUVVGSExFTkJRVU1zUTBGQlF5eERRVUZETEVkQlFVY3NRMEZCUXl4RFFVRkRMRU5CUVVNN1dVRkRiRUlzU1VGQlNTeEZRVUZGTEVkQlFVY3NRMEZCUXl4RFFVRkRMRU5CUVVNc1NVRkJTU3hEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNSMEZCUnl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUjBGQlJ5eERRVUZETEVOQlFVTXNRMEZCUXp0WlFVTnVReXhKUVVGSkxFVkJRVVVzUjBGQlJ5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNN1dVRkRaQ3hKUVVGSkxFVkJRVVVzUjBGQlJ5eERRVUZETEVOQlFVTXNRMEZCUXl4SFFVRkhMRU5CUVVNc1EwRkJReXhEUVVGRE8xbEJRMnhDTEVsQlFVa3NUVUZCVFN4SFFVRkhMRTFCUVUwc1EwRkJReXhGUVVGRkxFVkJRVVVzUlVGQlJTeEZRVUZGTEVWQlFVVXNRMEZCUXl4RFFVRkRPMWxCUTJoRExFbEJRVWtzVFVGQlRTeEhRVUZITEUxQlFVMHNRMEZCUXl4RlFVRkZMRVZCUVVVc1JVRkJSU3hGUVVGRkxFVkJRVVVzUTBGQlF5eERRVUZETzFsQlEyaERMRWxCUVVrc1RVRkJUU3hIUVVGSExFMUJRVTBzUTBGQlF5eEZRVUZGTEVWQlFVVXNSVUZCUlN4RlFVRkZMRVZCUVVVc1EwRkJReXhEUVVGRE8xbEJRMmhETEVsQlFVa3NUVUZCVFN4SFFVRkhMRTFCUVUwc1EwRkJReXhGUVVGRkxFVkJRVVVzUlVGQlJTeEZRVUZGTEVWQlFVVXNRMEZCUXl4RFFVRkRPMWxCUTJoRExFbEJRVWtzVFVGQlRTeEhRVUZITEUxQlFVMHNRMEZCUXl4RlFVRkZMRVZCUVVVc1JVRkJSU3hGUVVGRkxFVkJRVVVzUTBGQlF5eERRVUZETzFsQlEyaERMRWxCUVVrc1RVRkJUU3hIUVVGSExFMUJRVTBzUTBGQlF5eEZRVUZGTEVWQlFVVXNSVUZCUlN4RlFVRkZMRVZCUVVVc1EwRkJReXhEUVVGRE8xbEJRMmhETEVsQlFVa3NUVUZCVFN4SlFVRkpMRU5CUVVNc1NVRkJTU3hOUVVGTkxFbEJRVWtzUTBGQlF5eEpRVUZKTEUxQlFVMHNSMEZCUnl4RFFVRkRPMjFDUVVOeVF5eE5RVUZOTEVsQlFVa3NRMEZCUXl4SlFVRkpMRTFCUVUwc1NVRkJTU3hEUVVGRExFbEJRVWtzVFVGQlRTeEhRVUZITEVOQlFVTXNSVUZCUlR0blFrRkRla01zUlVGQlJTeERRVUZETEVWQlFVVXNSMEZCUnl4SlFVRkpMRk5CUVZNc1EwRkJReXhEUVVGRExFVkJRVVVzUTBGQlF5eERRVUZETEVOQlFVTTdZVUZEYmtNN2FVSkJRVTBzU1VGQlNTeE5RVUZOTEVsQlFVa3NRMEZCUXl4SlFVRkpMRTFCUVUwc1NVRkJTU3hEUVVGRExFbEJRVWtzVFVGQlRTeEhRVUZITEVOQlFVTTdiVUpCUXpWRExFMUJRVTBzU1VGQlNTeERRVUZETEVsQlFVa3NUVUZCVFN4SlFVRkpMRU5CUVVNc1NVRkJTU3hOUVVGTkxFZEJRVWNzUTBGQlF5eEZRVUZGTzJkQ1FVTjZReXhGUVVGRkxFTkJRVU1zUlVGQlJTeEhRVUZITEVsQlFVa3NVMEZCVXl4RFFVRkRMRU5CUVVNc1JVRkJSU3hEUVVGRExFTkJRVU1zUTBGQlF6dGhRVU51UXp0cFFrRkJUU3hKUVVGSkxFMUJRVTBzU1VGQlNTeERRVUZETEVsQlFVa3NUVUZCVFN4SFFVRkhMRU5CUVVNc1NVRkJTU3hOUVVGTkxFbEJRVWtzUTBGQlF6dHRRa0ZETlVNc1RVRkJUU3hKUVVGSkxFTkJRVU1zU1VGQlNTeE5RVUZOTEVkQlFVY3NRMEZCUXl4SlFVRkpMRTFCUVUwc1NVRkJTU3hEUVVGRExFVkJRVVU3WjBKQlEzcERMRVZCUVVVc1EwRkJReXhGUVVGRkxFZEJRVWNzU1VGQlNTeFRRVUZUTEVOQlFVTXNRMEZCUXl4RlFVRkZMRU5CUVVNc1EwRkJReXhEUVVGRE8yRkJRMjVETzJsQ1FVRk5MRWxCUVVrc1RVRkJUU3hKUVVGSkxFTkJRVU1zU1VGQlNTeE5RVUZOTEVkQlFVY3NRMEZCUXl4SlFVRkpMRTFCUVUwc1NVRkJTU3hEUVVGRE8yMUNRVU0xUXl4TlFVRk5MRWxCUVVrc1EwRkJReXhKUVVGSkxFMUJRVTBzUjBGQlJ5eERRVUZETEVsQlFVa3NUVUZCVFN4SlFVRkpMRU5CUVVNc1JVRkJSVHRuUWtGRGVrTXNSVUZCUlN4RFFVRkRMRVZCUVVVc1IwRkJSeXhKUVVGSkxGTkJRVk1zUTBGQlF5eERRVUZETEVWQlFVVXNRMEZCUXl4RFFVRkRMRU5CUVVNN1lVRkRia003VTBGRFNqdExRVU5LTzBsQlEwUXNUMEZCVHl4RlFVRkZMRU5CUVVNN1FVRkRaQ3hEUVVGRE8wRkJiRU5FTERSQ1FXdERRenRCUVVWRUxGTkJRVk1zYVVKQlFXbENMRU5CUVVNc1EwRkJVU3hGUVVGRkxFbEJRV0U3U1VGRE9VTXNTMEZCU3l4SlFVRkpMRU5CUVVNc1IwRkJSeXhEUVVGRExFVkJRVVVzUTBGQlF5eEhRVUZITEVsQlFVa3NRMEZCUXl4TlFVRk5MRVZCUVVVc1EwRkJReXhIUVVGSExFTkJRVU1zUlVGQlJTeEZRVUZGTEVOQlFVTTdVVUZEZGtNc1NVRkJTU3hMUVVGTExFTkJRVU1zU1VGQlNTeERRVUZETEVOQlFVTXNSMEZCUnl4RFFVRkRMRU5CUVVNc1JVRkJSU3hKUVVGSkxFTkJRVU1zUTBGQlF5eERRVUZETEVWQlFVVXNRMEZCUXl4RFFVRkRPMWxCUVVVc1QwRkJUeXhMUVVGTExFTkJRVU03U1VGRGNrUXNUMEZCVHl4SlFVRkpMRU5CUVVNN1FVRkRhRUlzUTBGQlF6dEJRVVZFTEZOQlFWTXNVMEZCVXl4RFFVRkRMRU5CUVZVc1JVRkJSU3hEUVVGVk8wbEJRM0pETEU5QlFVOHNRMEZCUXl4RFFVRkRMRU5CUVVNc1MwRkJTeXhEUVVGRExGVkJRVUVzUTBGQlF5eEpRVUZKTEU5QlFVRXNRMEZCUXl4cFFrRkJhVUlzUTBGQlF5eERRVUZETEVWQlFVVXNRMEZCUXl4RFFVRkRMRVZCUVhoQ0xFTkJRWGRDTEVOQlFVTXNRMEZCUXp0QlFVTnVSQ3hEUVVGRE8wRkJSVVFzVTBGQlowSXNXVUZCV1N4RFFVRkRMRU5CUVZVc1JVRkJSU3hEUVVGVk8wbEJReTlETEVsQlFVa3NVMEZCVXl4RFFVRkRMRU5CUVVNc1JVRkJSU3hEUVVGRExFTkJRVU03VVVGQlJTeFBRVUZQTEVsQlFVa3NRMEZCUXp0SlFVTnFReXhKUVVGSkxGTkJRVk1zUTBGQlF5eERRVUZETEVWQlFVVXNRMEZCUXl4RFFVRkRPMUZCUVVVc1QwRkJUeXhKUVVGSkxFTkJRVU03U1VGRGFrTXNTMEZCU3l4SlFVRkpMRU5CUVVNc1IwRkJSeXhEUVVGRExFVkJRVVVzUTBGQlF5eEhRVUZITEVOQlFVTXNRMEZCUXl4TlFVRk5MRVZCUVVVc1EwRkJReXhIUVVGSExFTkJRVU1zUlVGQlJTeEZRVUZGTEVOQlFVTXNSVUZCUlR0UlFVTjBReXhKUVVGSkxFTkJRVU1zUjBGQlJ5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRVZCUVVVc1EwRkJReXhIUVVGSExFTkJRVU1zUTBGQlF5eERRVUZETEVkQlFVY3NRMEZCUXl4RFFVRkRMRU5CUVVNN1VVRkRNMElzU1VGQlNTeFZRVUZWTEVOQlFVTXNTVUZCU1N4WFFVRlhMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUlVGQlJTeERRVUZETEVOQlFVTXNRMEZCUXl4RlFVRkZMRU5CUVVNc1EwRkJReXhEUVVGRExFVkJRVVVzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RlFVRkZMRU5CUVVNc1EwRkJReXhEUVVGRExFMUJRVTBzUjBGQlJ5eERRVUZETzFsQlFVVXNUMEZCVHl4SlFVRkpMRU5CUVVNN1MwRkRiRVk3U1VGRFJDeFBRVUZQTEV0QlFVc3NRMEZCUXp0QlFVTnFRaXhEUVVGRE8wRkJVa1FzYjBOQlVVTWlmUT09IiwiXCJ1c2Ugc3RyaWN0XCI7XHJcbk9iamVjdC5kZWZpbmVQcm9wZXJ0eShleHBvcnRzLCBcIl9fZXNNb2R1bGVcIiwgeyB2YWx1ZTogdHJ1ZSB9KTtcclxudmFyIHJlY3RhbmdsZV8xID0gcmVxdWlyZShcIi4vcmVjdGFuZ2xlXCIpO1xyXG52YXIgdnBzY18xID0gcmVxdWlyZShcIi4vdnBzY1wiKTtcclxudmFyIHNob3J0ZXN0cGF0aHNfMSA9IHJlcXVpcmUoXCIuL3Nob3J0ZXN0cGF0aHNcIik7XHJcbnZhciBOb2RlV3JhcHBlciA9IChmdW5jdGlvbiAoKSB7XHJcbiAgICBmdW5jdGlvbiBOb2RlV3JhcHBlcihpZCwgcmVjdCwgY2hpbGRyZW4pIHtcclxuICAgICAgICB0aGlzLmlkID0gaWQ7XHJcbiAgICAgICAgdGhpcy5yZWN0ID0gcmVjdDtcclxuICAgICAgICB0aGlzLmNoaWxkcmVuID0gY2hpbGRyZW47XHJcbiAgICAgICAgdGhpcy5sZWFmID0gdHlwZW9mIGNoaWxkcmVuID09PSAndW5kZWZpbmVkJyB8fCBjaGlsZHJlbi5sZW5ndGggPT09IDA7XHJcbiAgICB9XHJcbiAgICByZXR1cm4gTm9kZVdyYXBwZXI7XHJcbn0oKSk7XHJcbmV4cG9ydHMuTm9kZVdyYXBwZXIgPSBOb2RlV3JhcHBlcjtcclxudmFyIFZlcnQgPSAoZnVuY3Rpb24gKCkge1xyXG4gICAgZnVuY3Rpb24gVmVydChpZCwgeCwgeSwgbm9kZSwgbGluZSkge1xyXG4gICAgICAgIGlmIChub2RlID09PSB2b2lkIDApIHsgbm9kZSA9IG51bGw7IH1cclxuICAgICAgICBpZiAobGluZSA9PT0gdm9pZCAwKSB7IGxpbmUgPSBudWxsOyB9XHJcbiAgICAgICAgdGhpcy5pZCA9IGlkO1xyXG4gICAgICAgIHRoaXMueCA9IHg7XHJcbiAgICAgICAgdGhpcy55ID0geTtcclxuICAgICAgICB0aGlzLm5vZGUgPSBub2RlO1xyXG4gICAgICAgIHRoaXMubGluZSA9IGxpbmU7XHJcbiAgICB9XHJcbiAgICByZXR1cm4gVmVydDtcclxufSgpKTtcclxuZXhwb3J0cy5WZXJ0ID0gVmVydDtcclxudmFyIExvbmdlc3RDb21tb25TdWJzZXF1ZW5jZSA9IChmdW5jdGlvbiAoKSB7XHJcbiAgICBmdW5jdGlvbiBMb25nZXN0Q29tbW9uU3Vic2VxdWVuY2UocywgdCkge1xyXG4gICAgICAgIHRoaXMucyA9IHM7XHJcbiAgICAgICAgdGhpcy50ID0gdDtcclxuICAgICAgICB2YXIgbWYgPSBMb25nZXN0Q29tbW9uU3Vic2VxdWVuY2UuZmluZE1hdGNoKHMsIHQpO1xyXG4gICAgICAgIHZhciB0ciA9IHQuc2xpY2UoMCkucmV2ZXJzZSgpO1xyXG4gICAgICAgIHZhciBtciA9IExvbmdlc3RDb21tb25TdWJzZXF1ZW5jZS5maW5kTWF0Y2gocywgdHIpO1xyXG4gICAgICAgIGlmIChtZi5sZW5ndGggPj0gbXIubGVuZ3RoKSB7XHJcbiAgICAgICAgICAgIHRoaXMubGVuZ3RoID0gbWYubGVuZ3RoO1xyXG4gICAgICAgICAgICB0aGlzLnNpID0gbWYuc2k7XHJcbiAgICAgICAgICAgIHRoaXMudGkgPSBtZi50aTtcclxuICAgICAgICAgICAgdGhpcy5yZXZlcnNlZCA9IGZhbHNlO1xyXG4gICAgICAgIH1cclxuICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgdGhpcy5sZW5ndGggPSBtci5sZW5ndGg7XHJcbiAgICAgICAgICAgIHRoaXMuc2kgPSBtci5zaTtcclxuICAgICAgICAgICAgdGhpcy50aSA9IHQubGVuZ3RoIC0gbXIudGkgLSBtci5sZW5ndGg7XHJcbiAgICAgICAgICAgIHRoaXMucmV2ZXJzZWQgPSB0cnVlO1xyXG4gICAgICAgIH1cclxuICAgIH1cclxuICAgIExvbmdlc3RDb21tb25TdWJzZXF1ZW5jZS5maW5kTWF0Y2ggPSBmdW5jdGlvbiAocywgdCkge1xyXG4gICAgICAgIHZhciBtID0gcy5sZW5ndGg7XHJcbiAgICAgICAgdmFyIG4gPSB0Lmxlbmd0aDtcclxuICAgICAgICB2YXIgbWF0Y2ggPSB7IGxlbmd0aDogMCwgc2k6IC0xLCB0aTogLTEgfTtcclxuICAgICAgICB2YXIgbCA9IG5ldyBBcnJheShtKTtcclxuICAgICAgICBmb3IgKHZhciBpID0gMDsgaSA8IG07IGkrKykge1xyXG4gICAgICAgICAgICBsW2ldID0gbmV3IEFycmF5KG4pO1xyXG4gICAgICAgICAgICBmb3IgKHZhciBqID0gMDsgaiA8IG47IGorKylcclxuICAgICAgICAgICAgICAgIGlmIChzW2ldID09PSB0W2pdKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgdmFyIHYgPSBsW2ldW2pdID0gKGkgPT09IDAgfHwgaiA9PT0gMCkgPyAxIDogbFtpIC0gMV1baiAtIDFdICsgMTtcclxuICAgICAgICAgICAgICAgICAgICBpZiAodiA+IG1hdGNoLmxlbmd0aCkge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICBtYXRjaC5sZW5ndGggPSB2O1xyXG4gICAgICAgICAgICAgICAgICAgICAgICBtYXRjaC5zaSA9IGkgLSB2ICsgMTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgbWF0Y2gudGkgPSBqIC0gdiArIDE7XHJcbiAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgIDtcclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIGVsc2VcclxuICAgICAgICAgICAgICAgICAgICBsW2ldW2pdID0gMDtcclxuICAgICAgICB9XHJcbiAgICAgICAgcmV0dXJuIG1hdGNoO1xyXG4gICAgfTtcclxuICAgIExvbmdlc3RDb21tb25TdWJzZXF1ZW5jZS5wcm90b3R5cGUuZ2V0U2VxdWVuY2UgPSBmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgcmV0dXJuIHRoaXMubGVuZ3RoID49IDAgPyB0aGlzLnMuc2xpY2UodGhpcy5zaSwgdGhpcy5zaSArIHRoaXMubGVuZ3RoKSA6IFtdO1xyXG4gICAgfTtcclxuICAgIHJldHVybiBMb25nZXN0Q29tbW9uU3Vic2VxdWVuY2U7XHJcbn0oKSk7XHJcbmV4cG9ydHMuTG9uZ2VzdENvbW1vblN1YnNlcXVlbmNlID0gTG9uZ2VzdENvbW1vblN1YnNlcXVlbmNlO1xyXG52YXIgR3JpZFJvdXRlciA9IChmdW5jdGlvbiAoKSB7XHJcbiAgICBmdW5jdGlvbiBHcmlkUm91dGVyKG9yaWdpbmFsbm9kZXMsIGFjY2Vzc29yLCBncm91cFBhZGRpbmcpIHtcclxuICAgICAgICB2YXIgX3RoaXMgPSB0aGlzO1xyXG4gICAgICAgIGlmIChncm91cFBhZGRpbmcgPT09IHZvaWQgMCkgeyBncm91cFBhZGRpbmcgPSAxMjsgfVxyXG4gICAgICAgIHRoaXMub3JpZ2luYWxub2RlcyA9IG9yaWdpbmFsbm9kZXM7XHJcbiAgICAgICAgdGhpcy5ncm91cFBhZGRpbmcgPSBncm91cFBhZGRpbmc7XHJcbiAgICAgICAgdGhpcy5sZWF2ZXMgPSBudWxsO1xyXG4gICAgICAgIHRoaXMubm9kZXMgPSBvcmlnaW5hbG5vZGVzLm1hcChmdW5jdGlvbiAodiwgaSkgeyByZXR1cm4gbmV3IE5vZGVXcmFwcGVyKGksIGFjY2Vzc29yLmdldEJvdW5kcyh2KSwgYWNjZXNzb3IuZ2V0Q2hpbGRyZW4odikpOyB9KTtcclxuICAgICAgICB0aGlzLmxlYXZlcyA9IHRoaXMubm9kZXMuZmlsdGVyKGZ1bmN0aW9uICh2KSB7IHJldHVybiB2LmxlYWY7IH0pO1xyXG4gICAgICAgIHRoaXMuZ3JvdXBzID0gdGhpcy5ub2Rlcy5maWx0ZXIoZnVuY3Rpb24gKGcpIHsgcmV0dXJuICFnLmxlYWY7IH0pO1xyXG4gICAgICAgIHRoaXMuY29scyA9IHRoaXMuZ2V0R3JpZExpbmVzKCd4Jyk7XHJcbiAgICAgICAgdGhpcy5yb3dzID0gdGhpcy5nZXRHcmlkTGluZXMoJ3knKTtcclxuICAgICAgICB0aGlzLmdyb3Vwcy5mb3JFYWNoKGZ1bmN0aW9uICh2KSB7XHJcbiAgICAgICAgICAgIHJldHVybiB2LmNoaWxkcmVuLmZvckVhY2goZnVuY3Rpb24gKGMpIHsgcmV0dXJuIF90aGlzLm5vZGVzW2NdLnBhcmVudCA9IHY7IH0pO1xyXG4gICAgICAgIH0pO1xyXG4gICAgICAgIHRoaXMucm9vdCA9IHsgY2hpbGRyZW46IFtdIH07XHJcbiAgICAgICAgdGhpcy5ub2Rlcy5mb3JFYWNoKGZ1bmN0aW9uICh2KSB7XHJcbiAgICAgICAgICAgIGlmICh0eXBlb2Ygdi5wYXJlbnQgPT09ICd1bmRlZmluZWQnKSB7XHJcbiAgICAgICAgICAgICAgICB2LnBhcmVudCA9IF90aGlzLnJvb3Q7XHJcbiAgICAgICAgICAgICAgICBfdGhpcy5yb290LmNoaWxkcmVuLnB1c2godi5pZCk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgdi5wb3J0cyA9IFtdO1xyXG4gICAgICAgIH0pO1xyXG4gICAgICAgIHRoaXMuYmFja1RvRnJvbnQgPSB0aGlzLm5vZGVzLnNsaWNlKDApO1xyXG4gICAgICAgIHRoaXMuYmFja1RvRnJvbnQuc29ydChmdW5jdGlvbiAoeCwgeSkgeyByZXR1cm4gX3RoaXMuZ2V0RGVwdGgoeCkgLSBfdGhpcy5nZXREZXB0aCh5KTsgfSk7XHJcbiAgICAgICAgdmFyIGZyb250VG9CYWNrR3JvdXBzID0gdGhpcy5iYWNrVG9Gcm9udC5zbGljZSgwKS5yZXZlcnNlKCkuZmlsdGVyKGZ1bmN0aW9uIChnKSB7IHJldHVybiAhZy5sZWFmOyB9KTtcclxuICAgICAgICBmcm9udFRvQmFja0dyb3Vwcy5mb3JFYWNoKGZ1bmN0aW9uICh2KSB7XHJcbiAgICAgICAgICAgIHZhciByID0gcmVjdGFuZ2xlXzEuUmVjdGFuZ2xlLmVtcHR5KCk7XHJcbiAgICAgICAgICAgIHYuY2hpbGRyZW4uZm9yRWFjaChmdW5jdGlvbiAoYykgeyByZXR1cm4gciA9IHIudW5pb24oX3RoaXMubm9kZXNbY10ucmVjdCk7IH0pO1xyXG4gICAgICAgICAgICB2LnJlY3QgPSByLmluZmxhdGUoX3RoaXMuZ3JvdXBQYWRkaW5nKTtcclxuICAgICAgICB9KTtcclxuICAgICAgICB2YXIgY29sTWlkcyA9IHRoaXMubWlkUG9pbnRzKHRoaXMuY29scy5tYXAoZnVuY3Rpb24gKHIpIHsgcmV0dXJuIHIucG9zOyB9KSk7XHJcbiAgICAgICAgdmFyIHJvd01pZHMgPSB0aGlzLm1pZFBvaW50cyh0aGlzLnJvd3MubWFwKGZ1bmN0aW9uIChyKSB7IHJldHVybiByLnBvczsgfSkpO1xyXG4gICAgICAgIHZhciByb3d4ID0gY29sTWlkc1swXSwgcm93WCA9IGNvbE1pZHNbY29sTWlkcy5sZW5ndGggLSAxXTtcclxuICAgICAgICB2YXIgY29seSA9IHJvd01pZHNbMF0sIGNvbFkgPSByb3dNaWRzW3Jvd01pZHMubGVuZ3RoIC0gMV07XHJcbiAgICAgICAgdmFyIGhsaW5lcyA9IHRoaXMucm93cy5tYXAoZnVuY3Rpb24gKHIpIHsgcmV0dXJuICh7IHgxOiByb3d4LCB4Mjogcm93WCwgeTE6IHIucG9zLCB5Mjogci5wb3MgfSk7IH0pXHJcbiAgICAgICAgICAgIC5jb25jYXQocm93TWlkcy5tYXAoZnVuY3Rpb24gKG0pIHsgcmV0dXJuICh7IHgxOiByb3d4LCB4Mjogcm93WCwgeTE6IG0sIHkyOiBtIH0pOyB9KSk7XHJcbiAgICAgICAgdmFyIHZsaW5lcyA9IHRoaXMuY29scy5tYXAoZnVuY3Rpb24gKGMpIHsgcmV0dXJuICh7IHgxOiBjLnBvcywgeDI6IGMucG9zLCB5MTogY29seSwgeTI6IGNvbFkgfSk7IH0pXHJcbiAgICAgICAgICAgIC5jb25jYXQoY29sTWlkcy5tYXAoZnVuY3Rpb24gKG0pIHsgcmV0dXJuICh7IHgxOiBtLCB4MjogbSwgeTE6IGNvbHksIHkyOiBjb2xZIH0pOyB9KSk7XHJcbiAgICAgICAgdmFyIGxpbmVzID0gaGxpbmVzLmNvbmNhdCh2bGluZXMpO1xyXG4gICAgICAgIGxpbmVzLmZvckVhY2goZnVuY3Rpb24gKGwpIHsgcmV0dXJuIGwudmVydHMgPSBbXTsgfSk7XHJcbiAgICAgICAgdGhpcy52ZXJ0cyA9IFtdO1xyXG4gICAgICAgIHRoaXMuZWRnZXMgPSBbXTtcclxuICAgICAgICBobGluZXMuZm9yRWFjaChmdW5jdGlvbiAoaCkge1xyXG4gICAgICAgICAgICByZXR1cm4gdmxpbmVzLmZvckVhY2goZnVuY3Rpb24gKHYpIHtcclxuICAgICAgICAgICAgICAgIHZhciBwID0gbmV3IFZlcnQoX3RoaXMudmVydHMubGVuZ3RoLCB2LngxLCBoLnkxKTtcclxuICAgICAgICAgICAgICAgIGgudmVydHMucHVzaChwKTtcclxuICAgICAgICAgICAgICAgIHYudmVydHMucHVzaChwKTtcclxuICAgICAgICAgICAgICAgIF90aGlzLnZlcnRzLnB1c2gocCk7XHJcbiAgICAgICAgICAgICAgICB2YXIgaSA9IF90aGlzLmJhY2tUb0Zyb250Lmxlbmd0aDtcclxuICAgICAgICAgICAgICAgIHdoaWxlIChpLS0gPiAwKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgdmFyIG5vZGUgPSBfdGhpcy5iYWNrVG9Gcm9udFtpXSwgciA9IG5vZGUucmVjdDtcclxuICAgICAgICAgICAgICAgICAgICB2YXIgZHggPSBNYXRoLmFicyhwLnggLSByLmN4KCkpLCBkeSA9IE1hdGguYWJzKHAueSAtIHIuY3koKSk7XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKGR4IDwgci53aWR0aCgpIC8gMiAmJiBkeSA8IHIuaGVpZ2h0KCkgLyAyKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIHAubm9kZSA9IG5vZGU7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGJyZWFrO1xyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgfSk7XHJcbiAgICAgICAgfSk7XHJcbiAgICAgICAgbGluZXMuZm9yRWFjaChmdW5jdGlvbiAobCwgbGkpIHtcclxuICAgICAgICAgICAgX3RoaXMubm9kZXMuZm9yRWFjaChmdW5jdGlvbiAodiwgaSkge1xyXG4gICAgICAgICAgICAgICAgdi5yZWN0LmxpbmVJbnRlcnNlY3Rpb25zKGwueDEsIGwueTEsIGwueDIsIGwueTIpLmZvckVhY2goZnVuY3Rpb24gKGludGVyc2VjdCwgaikge1xyXG4gICAgICAgICAgICAgICAgICAgIHZhciBwID0gbmV3IFZlcnQoX3RoaXMudmVydHMubGVuZ3RoLCBpbnRlcnNlY3QueCwgaW50ZXJzZWN0LnksIHYsIGwpO1xyXG4gICAgICAgICAgICAgICAgICAgIF90aGlzLnZlcnRzLnB1c2gocCk7XHJcbiAgICAgICAgICAgICAgICAgICAgbC52ZXJ0cy5wdXNoKHApO1xyXG4gICAgICAgICAgICAgICAgICAgIHYucG9ydHMucHVzaChwKTtcclxuICAgICAgICAgICAgICAgIH0pO1xyXG4gICAgICAgICAgICB9KTtcclxuICAgICAgICAgICAgdmFyIGlzSG9yaXogPSBNYXRoLmFicyhsLnkxIC0gbC55MikgPCAwLjE7XHJcbiAgICAgICAgICAgIHZhciBkZWx0YSA9IGZ1bmN0aW9uIChhLCBiKSB7IHJldHVybiBpc0hvcml6ID8gYi54IC0gYS54IDogYi55IC0gYS55OyB9O1xyXG4gICAgICAgICAgICBsLnZlcnRzLnNvcnQoZGVsdGEpO1xyXG4gICAgICAgICAgICBmb3IgKHZhciBpID0gMTsgaSA8IGwudmVydHMubGVuZ3RoOyBpKyspIHtcclxuICAgICAgICAgICAgICAgIHZhciB1ID0gbC52ZXJ0c1tpIC0gMV0sIHYgPSBsLnZlcnRzW2ldO1xyXG4gICAgICAgICAgICAgICAgaWYgKHUubm9kZSAmJiB1Lm5vZGUgPT09IHYubm9kZSAmJiB1Lm5vZGUubGVhZilcclxuICAgICAgICAgICAgICAgICAgICBjb250aW51ZTtcclxuICAgICAgICAgICAgICAgIF90aGlzLmVkZ2VzLnB1c2goeyBzb3VyY2U6IHUuaWQsIHRhcmdldDogdi5pZCwgbGVuZ3RoOiBNYXRoLmFicyhkZWx0YSh1LCB2KSkgfSk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9KTtcclxuICAgIH1cclxuICAgIEdyaWRSb3V0ZXIucHJvdG90eXBlLmF2ZyA9IGZ1bmN0aW9uIChhKSB7IHJldHVybiBhLnJlZHVjZShmdW5jdGlvbiAoeCwgeSkgeyByZXR1cm4geCArIHk7IH0pIC8gYS5sZW5ndGg7IH07XHJcbiAgICBHcmlkUm91dGVyLnByb3RvdHlwZS5nZXRHcmlkTGluZXMgPSBmdW5jdGlvbiAoYXhpcykge1xyXG4gICAgICAgIHZhciBjb2x1bW5zID0gW107XHJcbiAgICAgICAgdmFyIGxzID0gdGhpcy5sZWF2ZXMuc2xpY2UoMCwgdGhpcy5sZWF2ZXMubGVuZ3RoKTtcclxuICAgICAgICB3aGlsZSAobHMubGVuZ3RoID4gMCkge1xyXG4gICAgICAgICAgICB2YXIgb3ZlcmxhcHBpbmcgPSBscy5maWx0ZXIoZnVuY3Rpb24gKHYpIHsgcmV0dXJuIHYucmVjdFsnb3ZlcmxhcCcgKyBheGlzLnRvVXBwZXJDYXNlKCldKGxzWzBdLnJlY3QpOyB9KTtcclxuICAgICAgICAgICAgdmFyIGNvbCA9IHtcclxuICAgICAgICAgICAgICAgIG5vZGVzOiBvdmVybGFwcGluZyxcclxuICAgICAgICAgICAgICAgIHBvczogdGhpcy5hdmcob3ZlcmxhcHBpbmcubWFwKGZ1bmN0aW9uICh2KSB7IHJldHVybiB2LnJlY3RbJ2MnICsgYXhpc10oKTsgfSkpXHJcbiAgICAgICAgICAgIH07XHJcbiAgICAgICAgICAgIGNvbHVtbnMucHVzaChjb2wpO1xyXG4gICAgICAgICAgICBjb2wubm9kZXMuZm9yRWFjaChmdW5jdGlvbiAodikgeyByZXR1cm4gbHMuc3BsaWNlKGxzLmluZGV4T2YodiksIDEpOyB9KTtcclxuICAgICAgICB9XHJcbiAgICAgICAgY29sdW1ucy5zb3J0KGZ1bmN0aW9uIChhLCBiKSB7IHJldHVybiBhLnBvcyAtIGIucG9zOyB9KTtcclxuICAgICAgICByZXR1cm4gY29sdW1ucztcclxuICAgIH07XHJcbiAgICBHcmlkUm91dGVyLnByb3RvdHlwZS5nZXREZXB0aCA9IGZ1bmN0aW9uICh2KSB7XHJcbiAgICAgICAgdmFyIGRlcHRoID0gMDtcclxuICAgICAgICB3aGlsZSAodi5wYXJlbnQgIT09IHRoaXMucm9vdCkge1xyXG4gICAgICAgICAgICBkZXB0aCsrO1xyXG4gICAgICAgICAgICB2ID0gdi5wYXJlbnQ7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHJldHVybiBkZXB0aDtcclxuICAgIH07XHJcbiAgICBHcmlkUm91dGVyLnByb3RvdHlwZS5taWRQb2ludHMgPSBmdW5jdGlvbiAoYSkge1xyXG4gICAgICAgIHZhciBnYXAgPSBhWzFdIC0gYVswXTtcclxuICAgICAgICB2YXIgbWlkcyA9IFthWzBdIC0gZ2FwIC8gMl07XHJcbiAgICAgICAgZm9yICh2YXIgaSA9IDE7IGkgPCBhLmxlbmd0aDsgaSsrKSB7XHJcbiAgICAgICAgICAgIG1pZHMucHVzaCgoYVtpXSArIGFbaSAtIDFdKSAvIDIpO1xyXG4gICAgICAgIH1cclxuICAgICAgICBtaWRzLnB1c2goYVthLmxlbmd0aCAtIDFdICsgZ2FwIC8gMik7XHJcbiAgICAgICAgcmV0dXJuIG1pZHM7XHJcbiAgICB9O1xyXG4gICAgR3JpZFJvdXRlci5wcm90b3R5cGUuZmluZExpbmVhZ2UgPSBmdW5jdGlvbiAodikge1xyXG4gICAgICAgIHZhciBsaW5lYWdlID0gW3ZdO1xyXG4gICAgICAgIGRvIHtcclxuICAgICAgICAgICAgdiA9IHYucGFyZW50O1xyXG4gICAgICAgICAgICBsaW5lYWdlLnB1c2godik7XHJcbiAgICAgICAgfSB3aGlsZSAodiAhPT0gdGhpcy5yb290KTtcclxuICAgICAgICByZXR1cm4gbGluZWFnZS5yZXZlcnNlKCk7XHJcbiAgICB9O1xyXG4gICAgR3JpZFJvdXRlci5wcm90b3R5cGUuZmluZEFuY2VzdG9yUGF0aEJldHdlZW4gPSBmdW5jdGlvbiAoYSwgYikge1xyXG4gICAgICAgIHZhciBhYSA9IHRoaXMuZmluZExpbmVhZ2UoYSksIGJhID0gdGhpcy5maW5kTGluZWFnZShiKSwgaSA9IDA7XHJcbiAgICAgICAgd2hpbGUgKGFhW2ldID09PSBiYVtpXSlcclxuICAgICAgICAgICAgaSsrO1xyXG4gICAgICAgIHJldHVybiB7IGNvbW1vbkFuY2VzdG9yOiBhYVtpIC0gMV0sIGxpbmVhZ2VzOiBhYS5zbGljZShpKS5jb25jYXQoYmEuc2xpY2UoaSkpIH07XHJcbiAgICB9O1xyXG4gICAgR3JpZFJvdXRlci5wcm90b3R5cGUuc2libGluZ09ic3RhY2xlcyA9IGZ1bmN0aW9uIChhLCBiKSB7XHJcbiAgICAgICAgdmFyIF90aGlzID0gdGhpcztcclxuICAgICAgICB2YXIgcGF0aCA9IHRoaXMuZmluZEFuY2VzdG9yUGF0aEJldHdlZW4oYSwgYik7XHJcbiAgICAgICAgdmFyIGxpbmVhZ2VMb29rdXAgPSB7fTtcclxuICAgICAgICBwYXRoLmxpbmVhZ2VzLmZvckVhY2goZnVuY3Rpb24gKHYpIHsgcmV0dXJuIGxpbmVhZ2VMb29rdXBbdi5pZF0gPSB7fTsgfSk7XHJcbiAgICAgICAgdmFyIG9ic3RhY2xlcyA9IHBhdGguY29tbW9uQW5jZXN0b3IuY2hpbGRyZW4uZmlsdGVyKGZ1bmN0aW9uICh2KSB7IHJldHVybiAhKHYgaW4gbGluZWFnZUxvb2t1cCk7IH0pO1xyXG4gICAgICAgIHBhdGgubGluZWFnZXNcclxuICAgICAgICAgICAgLmZpbHRlcihmdW5jdGlvbiAodikgeyByZXR1cm4gdi5wYXJlbnQgIT09IHBhdGguY29tbW9uQW5jZXN0b3I7IH0pXHJcbiAgICAgICAgICAgIC5mb3JFYWNoKGZ1bmN0aW9uICh2KSB7IHJldHVybiBvYnN0YWNsZXMgPSBvYnN0YWNsZXMuY29uY2F0KHYucGFyZW50LmNoaWxkcmVuLmZpbHRlcihmdW5jdGlvbiAoYykgeyByZXR1cm4gYyAhPT0gdi5pZDsgfSkpOyB9KTtcclxuICAgICAgICByZXR1cm4gb2JzdGFjbGVzLm1hcChmdW5jdGlvbiAodikgeyByZXR1cm4gX3RoaXMubm9kZXNbdl07IH0pO1xyXG4gICAgfTtcclxuICAgIEdyaWRSb3V0ZXIuZ2V0U2VnbWVudFNldHMgPSBmdW5jdGlvbiAocm91dGVzLCB4LCB5KSB7XHJcbiAgICAgICAgdmFyIHZzZWdtZW50cyA9IFtdO1xyXG4gICAgICAgIGZvciAodmFyIGVpID0gMDsgZWkgPCByb3V0ZXMubGVuZ3RoOyBlaSsrKSB7XHJcbiAgICAgICAgICAgIHZhciByb3V0ZSA9IHJvdXRlc1tlaV07XHJcbiAgICAgICAgICAgIGZvciAodmFyIHNpID0gMDsgc2kgPCByb3V0ZS5sZW5ndGg7IHNpKyspIHtcclxuICAgICAgICAgICAgICAgIHZhciBzID0gcm91dGVbc2ldO1xyXG4gICAgICAgICAgICAgICAgcy5lZGdlaWQgPSBlaTtcclxuICAgICAgICAgICAgICAgIHMuaSA9IHNpO1xyXG4gICAgICAgICAgICAgICAgdmFyIHNkeCA9IHNbMV1beF0gLSBzWzBdW3hdO1xyXG4gICAgICAgICAgICAgICAgaWYgKE1hdGguYWJzKHNkeCkgPCAwLjEpIHtcclxuICAgICAgICAgICAgICAgICAgICB2c2VnbWVudHMucHVzaChzKTtcclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH1cclxuICAgICAgICB2c2VnbWVudHMuc29ydChmdW5jdGlvbiAoYSwgYikgeyByZXR1cm4gYVswXVt4XSAtIGJbMF1beF07IH0pO1xyXG4gICAgICAgIHZhciB2c2VnbWVudHNldHMgPSBbXTtcclxuICAgICAgICB2YXIgc2VnbWVudHNldCA9IG51bGw7XHJcbiAgICAgICAgZm9yICh2YXIgaSA9IDA7IGkgPCB2c2VnbWVudHMubGVuZ3RoOyBpKyspIHtcclxuICAgICAgICAgICAgdmFyIHMgPSB2c2VnbWVudHNbaV07XHJcbiAgICAgICAgICAgIGlmICghc2VnbWVudHNldCB8fCBNYXRoLmFicyhzWzBdW3hdIC0gc2VnbWVudHNldC5wb3MpID4gMC4xKSB7XHJcbiAgICAgICAgICAgICAgICBzZWdtZW50c2V0ID0geyBwb3M6IHNbMF1beF0sIHNlZ21lbnRzOiBbXSB9O1xyXG4gICAgICAgICAgICAgICAgdnNlZ21lbnRzZXRzLnB1c2goc2VnbWVudHNldCk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgc2VnbWVudHNldC5zZWdtZW50cy5wdXNoKHMpO1xyXG4gICAgICAgIH1cclxuICAgICAgICByZXR1cm4gdnNlZ21lbnRzZXRzO1xyXG4gICAgfTtcclxuICAgIEdyaWRSb3V0ZXIubnVkZ2VTZWdzID0gZnVuY3Rpb24gKHgsIHksIHJvdXRlcywgc2VnbWVudHMsIGxlZnRPZiwgZ2FwKSB7XHJcbiAgICAgICAgdmFyIG4gPSBzZWdtZW50cy5sZW5ndGg7XHJcbiAgICAgICAgaWYgKG4gPD0gMSlcclxuICAgICAgICAgICAgcmV0dXJuO1xyXG4gICAgICAgIHZhciB2cyA9IHNlZ21lbnRzLm1hcChmdW5jdGlvbiAocykgeyByZXR1cm4gbmV3IHZwc2NfMS5WYXJpYWJsZShzWzBdW3hdKTsgfSk7XHJcbiAgICAgICAgdmFyIGNzID0gW107XHJcbiAgICAgICAgZm9yICh2YXIgaSA9IDA7IGkgPCBuOyBpKyspIHtcclxuICAgICAgICAgICAgZm9yICh2YXIgaiA9IDA7IGogPCBuOyBqKyspIHtcclxuICAgICAgICAgICAgICAgIGlmIChpID09PSBqKVxyXG4gICAgICAgICAgICAgICAgICAgIGNvbnRpbnVlO1xyXG4gICAgICAgICAgICAgICAgdmFyIHMxID0gc2VnbWVudHNbaV0sIHMyID0gc2VnbWVudHNbal0sIGUxID0gczEuZWRnZWlkLCBlMiA9IHMyLmVkZ2VpZCwgbGluZCA9IC0xLCByaW5kID0gLTE7XHJcbiAgICAgICAgICAgICAgICBpZiAoeCA9PSAneCcpIHtcclxuICAgICAgICAgICAgICAgICAgICBpZiAobGVmdE9mKGUxLCBlMikpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgaWYgKHMxWzBdW3ldIDwgczFbMV1beV0pIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGxpbmQgPSBqLCByaW5kID0gaTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGxpbmQgPSBpLCByaW5kID0gajtcclxuICAgICAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIGVsc2Uge1xyXG4gICAgICAgICAgICAgICAgICAgIGlmIChsZWZ0T2YoZTEsIGUyKSkge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICBpZiAoczFbMF1beV0gPCBzMVsxXVt5XSkge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgbGluZCA9IGksIHJpbmQgPSBqO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGVsc2Uge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgbGluZCA9IGosIHJpbmQgPSBpO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgaWYgKGxpbmQgPj0gMCkge1xyXG4gICAgICAgICAgICAgICAgICAgIGNzLnB1c2gobmV3IHZwc2NfMS5Db25zdHJhaW50KHZzW2xpbmRdLCB2c1tyaW5kXSwgZ2FwKSk7XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICAgICAgdmFyIHNvbHZlciA9IG5ldyB2cHNjXzEuU29sdmVyKHZzLCBjcyk7XHJcbiAgICAgICAgc29sdmVyLnNvbHZlKCk7XHJcbiAgICAgICAgdnMuZm9yRWFjaChmdW5jdGlvbiAodiwgaSkge1xyXG4gICAgICAgICAgICB2YXIgcyA9IHNlZ21lbnRzW2ldO1xyXG4gICAgICAgICAgICB2YXIgcG9zID0gdi5wb3NpdGlvbigpO1xyXG4gICAgICAgICAgICBzWzBdW3hdID0gc1sxXVt4XSA9IHBvcztcclxuICAgICAgICAgICAgdmFyIHJvdXRlID0gcm91dGVzW3MuZWRnZWlkXTtcclxuICAgICAgICAgICAgaWYgKHMuaSA+IDApXHJcbiAgICAgICAgICAgICAgICByb3V0ZVtzLmkgLSAxXVsxXVt4XSA9IHBvcztcclxuICAgICAgICAgICAgaWYgKHMuaSA8IHJvdXRlLmxlbmd0aCAtIDEpXHJcbiAgICAgICAgICAgICAgICByb3V0ZVtzLmkgKyAxXVswXVt4XSA9IHBvcztcclxuICAgICAgICB9KTtcclxuICAgIH07XHJcbiAgICBHcmlkUm91dGVyLm51ZGdlU2VnbWVudHMgPSBmdW5jdGlvbiAocm91dGVzLCB4LCB5LCBsZWZ0T2YsIGdhcCkge1xyXG4gICAgICAgIHZhciB2c2VnbWVudHNldHMgPSBHcmlkUm91dGVyLmdldFNlZ21lbnRTZXRzKHJvdXRlcywgeCwgeSk7XHJcbiAgICAgICAgZm9yICh2YXIgaSA9IDA7IGkgPCB2c2VnbWVudHNldHMubGVuZ3RoOyBpKyspIHtcclxuICAgICAgICAgICAgdmFyIHNzID0gdnNlZ21lbnRzZXRzW2ldO1xyXG4gICAgICAgICAgICB2YXIgZXZlbnRzID0gW107XHJcbiAgICAgICAgICAgIGZvciAodmFyIGogPSAwOyBqIDwgc3Muc2VnbWVudHMubGVuZ3RoOyBqKyspIHtcclxuICAgICAgICAgICAgICAgIHZhciBzID0gc3Muc2VnbWVudHNbal07XHJcbiAgICAgICAgICAgICAgICBldmVudHMucHVzaCh7IHR5cGU6IDAsIHM6IHMsIHBvczogTWF0aC5taW4oc1swXVt5XSwgc1sxXVt5XSkgfSk7XHJcbiAgICAgICAgICAgICAgICBldmVudHMucHVzaCh7IHR5cGU6IDEsIHM6IHMsIHBvczogTWF0aC5tYXgoc1swXVt5XSwgc1sxXVt5XSkgfSk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgZXZlbnRzLnNvcnQoZnVuY3Rpb24gKGEsIGIpIHsgcmV0dXJuIGEucG9zIC0gYi5wb3MgKyBhLnR5cGUgLSBiLnR5cGU7IH0pO1xyXG4gICAgICAgICAgICB2YXIgb3BlbiA9IFtdO1xyXG4gICAgICAgICAgICB2YXIgb3BlbkNvdW50ID0gMDtcclxuICAgICAgICAgICAgZXZlbnRzLmZvckVhY2goZnVuY3Rpb24gKGUpIHtcclxuICAgICAgICAgICAgICAgIGlmIChlLnR5cGUgPT09IDApIHtcclxuICAgICAgICAgICAgICAgICAgICBvcGVuLnB1c2goZS5zKTtcclxuICAgICAgICAgICAgICAgICAgICBvcGVuQ291bnQrKztcclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIGVsc2Uge1xyXG4gICAgICAgICAgICAgICAgICAgIG9wZW5Db3VudC0tO1xyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgaWYgKG9wZW5Db3VudCA9PSAwKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgR3JpZFJvdXRlci5udWRnZVNlZ3MoeCwgeSwgcm91dGVzLCBvcGVuLCBsZWZ0T2YsIGdhcCk7XHJcbiAgICAgICAgICAgICAgICAgICAgb3BlbiA9IFtdO1xyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICB9KTtcclxuICAgICAgICB9XHJcbiAgICB9O1xyXG4gICAgR3JpZFJvdXRlci5wcm90b3R5cGUucm91dGVFZGdlcyA9IGZ1bmN0aW9uIChlZGdlcywgbnVkZ2VHYXAsIHNvdXJjZSwgdGFyZ2V0KSB7XHJcbiAgICAgICAgdmFyIF90aGlzID0gdGhpcztcclxuICAgICAgICB2YXIgcm91dGVQYXRocyA9IGVkZ2VzLm1hcChmdW5jdGlvbiAoZSkgeyByZXR1cm4gX3RoaXMucm91dGUoc291cmNlKGUpLCB0YXJnZXQoZSkpOyB9KTtcclxuICAgICAgICB2YXIgb3JkZXIgPSBHcmlkUm91dGVyLm9yZGVyRWRnZXMocm91dGVQYXRocyk7XHJcbiAgICAgICAgdmFyIHJvdXRlcyA9IHJvdXRlUGF0aHMubWFwKGZ1bmN0aW9uIChlKSB7IHJldHVybiBHcmlkUm91dGVyLm1ha2VTZWdtZW50cyhlKTsgfSk7XHJcbiAgICAgICAgR3JpZFJvdXRlci5udWRnZVNlZ21lbnRzKHJvdXRlcywgJ3gnLCAneScsIG9yZGVyLCBudWRnZUdhcCk7XHJcbiAgICAgICAgR3JpZFJvdXRlci5udWRnZVNlZ21lbnRzKHJvdXRlcywgJ3knLCAneCcsIG9yZGVyLCBudWRnZUdhcCk7XHJcbiAgICAgICAgR3JpZFJvdXRlci51bnJldmVyc2VFZGdlcyhyb3V0ZXMsIHJvdXRlUGF0aHMpO1xyXG4gICAgICAgIHJldHVybiByb3V0ZXM7XHJcbiAgICB9O1xyXG4gICAgR3JpZFJvdXRlci51bnJldmVyc2VFZGdlcyA9IGZ1bmN0aW9uIChyb3V0ZXMsIHJvdXRlUGF0aHMpIHtcclxuICAgICAgICByb3V0ZXMuZm9yRWFjaChmdW5jdGlvbiAoc2VnbWVudHMsIGkpIHtcclxuICAgICAgICAgICAgdmFyIHBhdGggPSByb3V0ZVBhdGhzW2ldO1xyXG4gICAgICAgICAgICBpZiAocGF0aC5yZXZlcnNlZCkge1xyXG4gICAgICAgICAgICAgICAgc2VnbWVudHMucmV2ZXJzZSgpO1xyXG4gICAgICAgICAgICAgICAgc2VnbWVudHMuZm9yRWFjaChmdW5jdGlvbiAoc2VnbWVudCkge1xyXG4gICAgICAgICAgICAgICAgICAgIHNlZ21lbnQucmV2ZXJzZSgpO1xyXG4gICAgICAgICAgICAgICAgfSk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9KTtcclxuICAgIH07XHJcbiAgICBHcmlkUm91dGVyLmFuZ2xlQmV0d2VlbjJMaW5lcyA9IGZ1bmN0aW9uIChsaW5lMSwgbGluZTIpIHtcclxuICAgICAgICB2YXIgYW5nbGUxID0gTWF0aC5hdGFuMihsaW5lMVswXS55IC0gbGluZTFbMV0ueSwgbGluZTFbMF0ueCAtIGxpbmUxWzFdLngpO1xyXG4gICAgICAgIHZhciBhbmdsZTIgPSBNYXRoLmF0YW4yKGxpbmUyWzBdLnkgLSBsaW5lMlsxXS55LCBsaW5lMlswXS54IC0gbGluZTJbMV0ueCk7XHJcbiAgICAgICAgdmFyIGRpZmYgPSBhbmdsZTEgLSBhbmdsZTI7XHJcbiAgICAgICAgaWYgKGRpZmYgPiBNYXRoLlBJIHx8IGRpZmYgPCAtTWF0aC5QSSkge1xyXG4gICAgICAgICAgICBkaWZmID0gYW5nbGUyIC0gYW5nbGUxO1xyXG4gICAgICAgIH1cclxuICAgICAgICByZXR1cm4gZGlmZjtcclxuICAgIH07XHJcbiAgICBHcmlkUm91dGVyLmlzTGVmdCA9IGZ1bmN0aW9uIChhLCBiLCBjKSB7XHJcbiAgICAgICAgcmV0dXJuICgoYi54IC0gYS54KSAqIChjLnkgLSBhLnkpIC0gKGIueSAtIGEueSkgKiAoYy54IC0gYS54KSkgPD0gMDtcclxuICAgIH07XHJcbiAgICBHcmlkUm91dGVyLmdldE9yZGVyID0gZnVuY3Rpb24gKHBhaXJzKSB7XHJcbiAgICAgICAgdmFyIG91dGdvaW5nID0ge307XHJcbiAgICAgICAgZm9yICh2YXIgaSA9IDA7IGkgPCBwYWlycy5sZW5ndGg7IGkrKykge1xyXG4gICAgICAgICAgICB2YXIgcCA9IHBhaXJzW2ldO1xyXG4gICAgICAgICAgICBpZiAodHlwZW9mIG91dGdvaW5nW3AubF0gPT09ICd1bmRlZmluZWQnKVxyXG4gICAgICAgICAgICAgICAgb3V0Z29pbmdbcC5sXSA9IHt9O1xyXG4gICAgICAgICAgICBvdXRnb2luZ1twLmxdW3Aucl0gPSB0cnVlO1xyXG4gICAgICAgIH1cclxuICAgICAgICByZXR1cm4gZnVuY3Rpb24gKGwsIHIpIHsgcmV0dXJuIHR5cGVvZiBvdXRnb2luZ1tsXSAhPT0gJ3VuZGVmaW5lZCcgJiYgb3V0Z29pbmdbbF1bcl07IH07XHJcbiAgICB9O1xyXG4gICAgR3JpZFJvdXRlci5vcmRlckVkZ2VzID0gZnVuY3Rpb24gKGVkZ2VzKSB7XHJcbiAgICAgICAgdmFyIGVkZ2VPcmRlciA9IFtdO1xyXG4gICAgICAgIGZvciAodmFyIGkgPSAwOyBpIDwgZWRnZXMubGVuZ3RoIC0gMTsgaSsrKSB7XHJcbiAgICAgICAgICAgIGZvciAodmFyIGogPSBpICsgMTsgaiA8IGVkZ2VzLmxlbmd0aDsgaisrKSB7XHJcbiAgICAgICAgICAgICAgICB2YXIgZSA9IGVkZ2VzW2ldLCBmID0gZWRnZXNbal0sIGxjcyA9IG5ldyBMb25nZXN0Q29tbW9uU3Vic2VxdWVuY2UoZSwgZik7XHJcbiAgICAgICAgICAgICAgICB2YXIgdSwgdmksIHZqO1xyXG4gICAgICAgICAgICAgICAgaWYgKGxjcy5sZW5ndGggPT09IDApXHJcbiAgICAgICAgICAgICAgICAgICAgY29udGludWU7XHJcbiAgICAgICAgICAgICAgICBpZiAobGNzLnJldmVyc2VkKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgZi5yZXZlcnNlKCk7XHJcbiAgICAgICAgICAgICAgICAgICAgZi5yZXZlcnNlZCA9IHRydWU7XHJcbiAgICAgICAgICAgICAgICAgICAgbGNzID0gbmV3IExvbmdlc3RDb21tb25TdWJzZXF1ZW5jZShlLCBmKTtcclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIGlmICgobGNzLnNpIDw9IDAgfHwgbGNzLnRpIDw9IDApICYmXHJcbiAgICAgICAgICAgICAgICAgICAgKGxjcy5zaSArIGxjcy5sZW5ndGggPj0gZS5sZW5ndGggfHwgbGNzLnRpICsgbGNzLmxlbmd0aCA+PSBmLmxlbmd0aCkpIHtcclxuICAgICAgICAgICAgICAgICAgICBlZGdlT3JkZXIucHVzaCh7IGw6IGksIHI6IGogfSk7XHJcbiAgICAgICAgICAgICAgICAgICAgY29udGludWU7XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICBpZiAobGNzLnNpICsgbGNzLmxlbmd0aCA+PSBlLmxlbmd0aCB8fCBsY3MudGkgKyBsY3MubGVuZ3RoID49IGYubGVuZ3RoKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgdSA9IGVbbGNzLnNpICsgMV07XHJcbiAgICAgICAgICAgICAgICAgICAgdmogPSBlW2xjcy5zaSAtIDFdO1xyXG4gICAgICAgICAgICAgICAgICAgIHZpID0gZltsY3MudGkgLSAxXTtcclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIGVsc2Uge1xyXG4gICAgICAgICAgICAgICAgICAgIHUgPSBlW2xjcy5zaSArIGxjcy5sZW5ndGggLSAyXTtcclxuICAgICAgICAgICAgICAgICAgICB2aSA9IGVbbGNzLnNpICsgbGNzLmxlbmd0aF07XHJcbiAgICAgICAgICAgICAgICAgICAgdmogPSBmW2xjcy50aSArIGxjcy5sZW5ndGhdO1xyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgaWYgKEdyaWRSb3V0ZXIuaXNMZWZ0KHUsIHZpLCB2aikpIHtcclxuICAgICAgICAgICAgICAgICAgICBlZGdlT3JkZXIucHVzaCh7IGw6IGosIHI6IGkgfSk7XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgICAgICAgICBlZGdlT3JkZXIucHVzaCh7IGw6IGksIHI6IGogfSk7XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICAgICAgcmV0dXJuIEdyaWRSb3V0ZXIuZ2V0T3JkZXIoZWRnZU9yZGVyKTtcclxuICAgIH07XHJcbiAgICBHcmlkUm91dGVyLm1ha2VTZWdtZW50cyA9IGZ1bmN0aW9uIChwYXRoKSB7XHJcbiAgICAgICAgZnVuY3Rpb24gY29weVBvaW50KHApIHtcclxuICAgICAgICAgICAgcmV0dXJuIHsgeDogcC54LCB5OiBwLnkgfTtcclxuICAgICAgICB9XHJcbiAgICAgICAgdmFyIGlzU3RyYWlnaHQgPSBmdW5jdGlvbiAoYSwgYiwgYykgeyByZXR1cm4gTWF0aC5hYnMoKGIueCAtIGEueCkgKiAoYy55IC0gYS55KSAtIChiLnkgLSBhLnkpICogKGMueCAtIGEueCkpIDwgMC4wMDE7IH07XHJcbiAgICAgICAgdmFyIHNlZ21lbnRzID0gW107XHJcbiAgICAgICAgdmFyIGEgPSBjb3B5UG9pbnQocGF0aFswXSk7XHJcbiAgICAgICAgZm9yICh2YXIgaSA9IDE7IGkgPCBwYXRoLmxlbmd0aDsgaSsrKSB7XHJcbiAgICAgICAgICAgIHZhciBiID0gY29weVBvaW50KHBhdGhbaV0pLCBjID0gaSA8IHBhdGgubGVuZ3RoIC0gMSA/IHBhdGhbaSArIDFdIDogbnVsbDtcclxuICAgICAgICAgICAgaWYgKCFjIHx8ICFpc1N0cmFpZ2h0KGEsIGIsIGMpKSB7XHJcbiAgICAgICAgICAgICAgICBzZWdtZW50cy5wdXNoKFthLCBiXSk7XHJcbiAgICAgICAgICAgICAgICBhID0gYjtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH1cclxuICAgICAgICByZXR1cm4gc2VnbWVudHM7XHJcbiAgICB9O1xyXG4gICAgR3JpZFJvdXRlci5wcm90b3R5cGUucm91dGUgPSBmdW5jdGlvbiAocywgdCkge1xyXG4gICAgICAgIHZhciBfdGhpcyA9IHRoaXM7XHJcbiAgICAgICAgdmFyIHNvdXJjZSA9IHRoaXMubm9kZXNbc10sIHRhcmdldCA9IHRoaXMubm9kZXNbdF07XHJcbiAgICAgICAgdGhpcy5vYnN0YWNsZXMgPSB0aGlzLnNpYmxpbmdPYnN0YWNsZXMoc291cmNlLCB0YXJnZXQpO1xyXG4gICAgICAgIHZhciBvYnN0YWNsZUxvb2t1cCA9IHt9O1xyXG4gICAgICAgIHRoaXMub2JzdGFjbGVzLmZvckVhY2goZnVuY3Rpb24gKG8pIHsgcmV0dXJuIG9ic3RhY2xlTG9va3VwW28uaWRdID0gbzsgfSk7XHJcbiAgICAgICAgdGhpcy5wYXNzYWJsZUVkZ2VzID0gdGhpcy5lZGdlcy5maWx0ZXIoZnVuY3Rpb24gKGUpIHtcclxuICAgICAgICAgICAgdmFyIHUgPSBfdGhpcy52ZXJ0c1tlLnNvdXJjZV0sIHYgPSBfdGhpcy52ZXJ0c1tlLnRhcmdldF07XHJcbiAgICAgICAgICAgIHJldHVybiAhKHUubm9kZSAmJiB1Lm5vZGUuaWQgaW4gb2JzdGFjbGVMb29rdXBcclxuICAgICAgICAgICAgICAgIHx8IHYubm9kZSAmJiB2Lm5vZGUuaWQgaW4gb2JzdGFjbGVMb29rdXApO1xyXG4gICAgICAgIH0pO1xyXG4gICAgICAgIGZvciAodmFyIGkgPSAxOyBpIDwgc291cmNlLnBvcnRzLmxlbmd0aDsgaSsrKSB7XHJcbiAgICAgICAgICAgIHZhciB1ID0gc291cmNlLnBvcnRzWzBdLmlkO1xyXG4gICAgICAgICAgICB2YXIgdiA9IHNvdXJjZS5wb3J0c1tpXS5pZDtcclxuICAgICAgICAgICAgdGhpcy5wYXNzYWJsZUVkZ2VzLnB1c2goe1xyXG4gICAgICAgICAgICAgICAgc291cmNlOiB1LFxyXG4gICAgICAgICAgICAgICAgdGFyZ2V0OiB2LFxyXG4gICAgICAgICAgICAgICAgbGVuZ3RoOiAwXHJcbiAgICAgICAgICAgIH0pO1xyXG4gICAgICAgIH1cclxuICAgICAgICBmb3IgKHZhciBpID0gMTsgaSA8IHRhcmdldC5wb3J0cy5sZW5ndGg7IGkrKykge1xyXG4gICAgICAgICAgICB2YXIgdSA9IHRhcmdldC5wb3J0c1swXS5pZDtcclxuICAgICAgICAgICAgdmFyIHYgPSB0YXJnZXQucG9ydHNbaV0uaWQ7XHJcbiAgICAgICAgICAgIHRoaXMucGFzc2FibGVFZGdlcy5wdXNoKHtcclxuICAgICAgICAgICAgICAgIHNvdXJjZTogdSxcclxuICAgICAgICAgICAgICAgIHRhcmdldDogdixcclxuICAgICAgICAgICAgICAgIGxlbmd0aDogMFxyXG4gICAgICAgICAgICB9KTtcclxuICAgICAgICB9XHJcbiAgICAgICAgdmFyIGdldFNvdXJjZSA9IGZ1bmN0aW9uIChlKSB7IHJldHVybiBlLnNvdXJjZTsgfSwgZ2V0VGFyZ2V0ID0gZnVuY3Rpb24gKGUpIHsgcmV0dXJuIGUudGFyZ2V0OyB9LCBnZXRMZW5ndGggPSBmdW5jdGlvbiAoZSkgeyByZXR1cm4gZS5sZW5ndGg7IH07XHJcbiAgICAgICAgdmFyIHNob3J0ZXN0UGF0aENhbGN1bGF0b3IgPSBuZXcgc2hvcnRlc3RwYXRoc18xLkNhbGN1bGF0b3IodGhpcy52ZXJ0cy5sZW5ndGgsIHRoaXMucGFzc2FibGVFZGdlcywgZ2V0U291cmNlLCBnZXRUYXJnZXQsIGdldExlbmd0aCk7XHJcbiAgICAgICAgdmFyIGJlbmRQZW5hbHR5ID0gZnVuY3Rpb24gKHUsIHYsIHcpIHtcclxuICAgICAgICAgICAgdmFyIGEgPSBfdGhpcy52ZXJ0c1t1XSwgYiA9IF90aGlzLnZlcnRzW3ZdLCBjID0gX3RoaXMudmVydHNbd107XHJcbiAgICAgICAgICAgIHZhciBkeCA9IE1hdGguYWJzKGMueCAtIGEueCksIGR5ID0gTWF0aC5hYnMoYy55IC0gYS55KTtcclxuICAgICAgICAgICAgaWYgKGEubm9kZSA9PT0gc291cmNlICYmIGEubm9kZSA9PT0gYi5ub2RlIHx8IGIubm9kZSA9PT0gdGFyZ2V0ICYmIGIubm9kZSA9PT0gYy5ub2RlKVxyXG4gICAgICAgICAgICAgICAgcmV0dXJuIDA7XHJcbiAgICAgICAgICAgIHJldHVybiBkeCA+IDEgJiYgZHkgPiAxID8gMTAwMCA6IDA7XHJcbiAgICAgICAgfTtcclxuICAgICAgICB2YXIgc2hvcnRlc3RQYXRoID0gc2hvcnRlc3RQYXRoQ2FsY3VsYXRvci5QYXRoRnJvbU5vZGVUb05vZGVXaXRoUHJldkNvc3Qoc291cmNlLnBvcnRzWzBdLmlkLCB0YXJnZXQucG9ydHNbMF0uaWQsIGJlbmRQZW5hbHR5KTtcclxuICAgICAgICB2YXIgcGF0aFBvaW50cyA9IHNob3J0ZXN0UGF0aC5yZXZlcnNlKCkubWFwKGZ1bmN0aW9uICh2aSkgeyByZXR1cm4gX3RoaXMudmVydHNbdmldOyB9KTtcclxuICAgICAgICBwYXRoUG9pbnRzLnB1c2godGhpcy5ub2Rlc1t0YXJnZXQuaWRdLnBvcnRzWzBdKTtcclxuICAgICAgICByZXR1cm4gcGF0aFBvaW50cy5maWx0ZXIoZnVuY3Rpb24gKHYsIGkpIHtcclxuICAgICAgICAgICAgcmV0dXJuICEoaSA8IHBhdGhQb2ludHMubGVuZ3RoIC0gMSAmJiBwYXRoUG9pbnRzW2kgKyAxXS5ub2RlID09PSBzb3VyY2UgJiYgdi5ub2RlID09PSBzb3VyY2VcclxuICAgICAgICAgICAgICAgIHx8IGkgPiAwICYmIHYubm9kZSA9PT0gdGFyZ2V0ICYmIHBhdGhQb2ludHNbaSAtIDFdLm5vZGUgPT09IHRhcmdldCk7XHJcbiAgICAgICAgfSk7XHJcbiAgICB9O1xyXG4gICAgR3JpZFJvdXRlci5nZXRSb3V0ZVBhdGggPSBmdW5jdGlvbiAocm91dGUsIGNvcm5lcnJhZGl1cywgYXJyb3d3aWR0aCwgYXJyb3doZWlnaHQpIHtcclxuICAgICAgICB2YXIgcmVzdWx0ID0ge1xyXG4gICAgICAgICAgICByb3V0ZXBhdGg6ICdNICcgKyByb3V0ZVswXVswXS54ICsgJyAnICsgcm91dGVbMF1bMF0ueSArICcgJyxcclxuICAgICAgICAgICAgYXJyb3dwYXRoOiAnJ1xyXG4gICAgICAgIH07XHJcbiAgICAgICAgaWYgKHJvdXRlLmxlbmd0aCA+IDEpIHtcclxuICAgICAgICAgICAgZm9yICh2YXIgaSA9IDA7IGkgPCByb3V0ZS5sZW5ndGg7IGkrKykge1xyXG4gICAgICAgICAgICAgICAgdmFyIGxpID0gcm91dGVbaV07XHJcbiAgICAgICAgICAgICAgICB2YXIgeCA9IGxpWzFdLngsIHkgPSBsaVsxXS55O1xyXG4gICAgICAgICAgICAgICAgdmFyIGR4ID0geCAtIGxpWzBdLng7XHJcbiAgICAgICAgICAgICAgICB2YXIgZHkgPSB5IC0gbGlbMF0ueTtcclxuICAgICAgICAgICAgICAgIGlmIChpIDwgcm91dGUubGVuZ3RoIC0gMSkge1xyXG4gICAgICAgICAgICAgICAgICAgIGlmIChNYXRoLmFicyhkeCkgPiAwKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIHggLT0gZHggLyBNYXRoLmFicyhkeCkgKiBjb3JuZXJyYWRpdXM7XHJcbiAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgIGVsc2Uge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICB5IC09IGR5IC8gTWF0aC5hYnMoZHkpICogY29ybmVycmFkaXVzO1xyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICByZXN1bHQucm91dGVwYXRoICs9ICdMICcgKyB4ICsgJyAnICsgeSArICcgJztcclxuICAgICAgICAgICAgICAgICAgICB2YXIgbCA9IHJvdXRlW2kgKyAxXTtcclxuICAgICAgICAgICAgICAgICAgICB2YXIgeDAgPSBsWzBdLngsIHkwID0gbFswXS55O1xyXG4gICAgICAgICAgICAgICAgICAgIHZhciB4MSA9IGxbMV0ueDtcclxuICAgICAgICAgICAgICAgICAgICB2YXIgeTEgPSBsWzFdLnk7XHJcbiAgICAgICAgICAgICAgICAgICAgZHggPSB4MSAtIHgwO1xyXG4gICAgICAgICAgICAgICAgICAgIGR5ID0geTEgLSB5MDtcclxuICAgICAgICAgICAgICAgICAgICB2YXIgYW5nbGUgPSBHcmlkUm91dGVyLmFuZ2xlQmV0d2VlbjJMaW5lcyhsaSwgbCkgPCAwID8gMSA6IDA7XHJcbiAgICAgICAgICAgICAgICAgICAgdmFyIHgyLCB5MjtcclxuICAgICAgICAgICAgICAgICAgICBpZiAoTWF0aC5hYnMoZHgpID4gMCkge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICB4MiA9IHgwICsgZHggLyBNYXRoLmFicyhkeCkgKiBjb3JuZXJyYWRpdXM7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIHkyID0geTA7XHJcbiAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgIGVsc2Uge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICB4MiA9IHgwO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICB5MiA9IHkwICsgZHkgLyBNYXRoLmFicyhkeSkgKiBjb3JuZXJyYWRpdXM7XHJcbiAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgIHZhciBjeCA9IE1hdGguYWJzKHgyIC0geCk7XHJcbiAgICAgICAgICAgICAgICAgICAgdmFyIGN5ID0gTWF0aC5hYnMoeTIgLSB5KTtcclxuICAgICAgICAgICAgICAgICAgICByZXN1bHQucm91dGVwYXRoICs9ICdBICcgKyBjeCArICcgJyArIGN5ICsgJyAwIDAgJyArIGFuZ2xlICsgJyAnICsgeDIgKyAnICcgKyB5MiArICcgJztcclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIGVsc2Uge1xyXG4gICAgICAgICAgICAgICAgICAgIHZhciBhcnJvd3RpcCA9IFt4LCB5XTtcclxuICAgICAgICAgICAgICAgICAgICB2YXIgYXJyb3djb3JuZXIxLCBhcnJvd2Nvcm5lcjI7XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKE1hdGguYWJzKGR4KSA+IDApIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgeCAtPSBkeCAvIE1hdGguYWJzKGR4KSAqIGFycm93aGVpZ2h0O1xyXG4gICAgICAgICAgICAgICAgICAgICAgICBhcnJvd2Nvcm5lcjEgPSBbeCwgeSArIGFycm93d2lkdGhdO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICBhcnJvd2Nvcm5lcjIgPSBbeCwgeSAtIGFycm93d2lkdGhdO1xyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgeSAtPSBkeSAvIE1hdGguYWJzKGR5KSAqIGFycm93aGVpZ2h0O1xyXG4gICAgICAgICAgICAgICAgICAgICAgICBhcnJvd2Nvcm5lcjEgPSBbeCArIGFycm93d2lkdGgsIHldO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICBhcnJvd2Nvcm5lcjIgPSBbeCAtIGFycm93d2lkdGgsIHldO1xyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICByZXN1bHQucm91dGVwYXRoICs9ICdMICcgKyB4ICsgJyAnICsgeSArICcgJztcclxuICAgICAgICAgICAgICAgICAgICBpZiAoYXJyb3doZWlnaHQgPiAwKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIHJlc3VsdC5hcnJvd3BhdGggPSAnTSAnICsgYXJyb3d0aXBbMF0gKyAnICcgKyBhcnJvd3RpcFsxXSArICcgTCAnICsgYXJyb3djb3JuZXIxWzBdICsgJyAnICsgYXJyb3djb3JuZXIxWzFdXHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICArICcgTCAnICsgYXJyb3djb3JuZXIyWzBdICsgJyAnICsgYXJyb3djb3JuZXIyWzFdO1xyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH1cclxuICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgdmFyIGxpID0gcm91dGVbMF07XHJcbiAgICAgICAgICAgIHZhciB4ID0gbGlbMV0ueCwgeSA9IGxpWzFdLnk7XHJcbiAgICAgICAgICAgIHZhciBkeCA9IHggLSBsaVswXS54O1xyXG4gICAgICAgICAgICB2YXIgZHkgPSB5IC0gbGlbMF0ueTtcclxuICAgICAgICAgICAgdmFyIGFycm93dGlwID0gW3gsIHldO1xyXG4gICAgICAgICAgICB2YXIgYXJyb3djb3JuZXIxLCBhcnJvd2Nvcm5lcjI7XHJcbiAgICAgICAgICAgIGlmIChNYXRoLmFicyhkeCkgPiAwKSB7XHJcbiAgICAgICAgICAgICAgICB4IC09IGR4IC8gTWF0aC5hYnMoZHgpICogYXJyb3doZWlnaHQ7XHJcbiAgICAgICAgICAgICAgICBhcnJvd2Nvcm5lcjEgPSBbeCwgeSArIGFycm93d2lkdGhdO1xyXG4gICAgICAgICAgICAgICAgYXJyb3djb3JuZXIyID0gW3gsIHkgLSBhcnJvd3dpZHRoXTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgICAgIHkgLT0gZHkgLyBNYXRoLmFicyhkeSkgKiBhcnJvd2hlaWdodDtcclxuICAgICAgICAgICAgICAgIGFycm93Y29ybmVyMSA9IFt4ICsgYXJyb3d3aWR0aCwgeV07XHJcbiAgICAgICAgICAgICAgICBhcnJvd2Nvcm5lcjIgPSBbeCAtIGFycm93d2lkdGgsIHldO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIHJlc3VsdC5yb3V0ZXBhdGggKz0gJ0wgJyArIHggKyAnICcgKyB5ICsgJyAnO1xyXG4gICAgICAgICAgICBpZiAoYXJyb3doZWlnaHQgPiAwKSB7XHJcbiAgICAgICAgICAgICAgICByZXN1bHQuYXJyb3dwYXRoID0gJ00gJyArIGFycm93dGlwWzBdICsgJyAnICsgYXJyb3d0aXBbMV0gKyAnIEwgJyArIGFycm93Y29ybmVyMVswXSArICcgJyArIGFycm93Y29ybmVyMVsxXVxyXG4gICAgICAgICAgICAgICAgICAgICsgJyBMICcgKyBhcnJvd2Nvcm5lcjJbMF0gKyAnICcgKyBhcnJvd2Nvcm5lcjJbMV07XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICAgICAgcmV0dXJuIHJlc3VsdDtcclxuICAgIH07XHJcbiAgICByZXR1cm4gR3JpZFJvdXRlcjtcclxufSgpKTtcclxuZXhwb3J0cy5HcmlkUm91dGVyID0gR3JpZFJvdXRlcjtcclxuLy8jIHNvdXJjZU1hcHBpbmdVUkw9ZGF0YTphcHBsaWNhdGlvbi9qc29uO2Jhc2U2NCxleUoyWlhKemFXOXVJam96TENKbWFXeGxJam9pWjNKcFpISnZkWFJsY2k1cWN5SXNJbk52ZFhKalpWSnZiM1FpT2lJaUxDSnpiM1Z5WTJWeklqcGJJaTR1THk0dUwxZGxZa052YkdFdmMzSmpMMmR5YVdSeWIzVjBaWEl1ZEhNaVhTd2libUZ0WlhNaU9sdGRMQ0p0WVhCd2FXNW5jeUk2SWpzN1FVRkRRU3g1UTBGQmNVTTdRVUZEY2tNc0swSkJRVzFFTzBGQlEyNUVMR2xFUVVFd1F6dEJRVXQwUXp0SlFVbEpMSEZDUVVGdFFpeEZRVUZWTEVWQlFWTXNTVUZCWlN4RlFVRlRMRkZCUVd0Q08xRkJRVGRFTEU5QlFVVXNSMEZCUml4RlFVRkZMRU5CUVZFN1VVRkJVeXhUUVVGSkxFZEJRVW9zU1VGQlNTeERRVUZYTzFGQlFWTXNZVUZCVVN4SFFVRlNMRkZCUVZFc1EwRkJWVHRSUVVNMVJTeEpRVUZKTEVOQlFVTXNTVUZCU1N4SFFVRkhMRTlCUVU4c1VVRkJVU3hMUVVGTExGZEJRVmNzU1VGQlNTeFJRVUZSTEVOQlFVTXNUVUZCVFN4TFFVRkxMRU5CUVVNc1EwRkJRenRKUVVONlJTeERRVUZETzBsQlEwd3NhMEpCUVVNN1FVRkJSQ3hEUVVGRExFRkJVRVFzU1VGUFF6dEJRVkJaTEd0RFFVRlhPMEZCVVhoQ08wbEJRMGtzWTBGQmJVSXNSVUZCVlN4RlFVRlRMRU5CUVZFc1JVRkJVeXhEUVVGVExFVkJRVk1zU1VGQmQwSXNSVUZCVXl4SlFVRlhPMUZCUVRWRExIRkNRVUZCTEVWQlFVRXNWMEZCZDBJN1VVRkJVeXh4UWtGQlFTeEZRVUZCTEZkQlFWYzdVVUZCYkVjc1QwRkJSU3hIUVVGR0xFVkJRVVVzUTBGQlVUdFJRVUZUTEUxQlFVTXNSMEZCUkN4RFFVRkRMRU5CUVU4N1VVRkJVeXhOUVVGRExFZEJRVVFzUTBGQlF5eERRVUZSTzFGQlFWTXNVMEZCU1N4SFFVRktMRWxCUVVrc1EwRkJiMEk3VVVGQlV5eFRRVUZKTEVkQlFVb3NTVUZCU1N4RFFVRlBPMGxCUVVjc1EwRkJRenRKUVVNM1NDeFhRVUZETzBGQlFVUXNRMEZCUXl4QlFVWkVMRWxCUlVNN1FVRkdXU3h2UWtGQlNUdEJRVWxxUWp0SlFVdEpMR3REUVVGdFFpeERRVUZOTEVWQlFWTXNRMEZCVFR0UlFVRnlRaXhOUVVGRExFZEJRVVFzUTBGQlF5eERRVUZMTzFGQlFWTXNUVUZCUXl4SFFVRkVMRU5CUVVNc1EwRkJTenRSUVVOd1F5eEpRVUZKTEVWQlFVVXNSMEZCUnl4M1FrRkJkMElzUTBGQlF5eFRRVUZUTEVOQlFVTXNRMEZCUXl4RlFVRkZMRU5CUVVNc1EwRkJReXhEUVVGRE8xRkJRMnhFTEVsQlFVa3NSVUZCUlN4SFFVRkhMRU5CUVVNc1EwRkJReXhMUVVGTExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNUMEZCVHl4RlFVRkZMRU5CUVVNN1VVRkRPVUlzU1VGQlNTeEZRVUZGTEVkQlFVY3NkMEpCUVhkQ0xFTkJRVU1zVTBGQlV5eERRVUZETEVOQlFVTXNSVUZCUlN4RlFVRkZMRU5CUVVNc1EwRkJRenRSUVVOdVJDeEpRVUZKTEVWQlFVVXNRMEZCUXl4TlFVRk5MRWxCUVVrc1JVRkJSU3hEUVVGRExFMUJRVTBzUlVGQlJUdFpRVU40UWl4SlFVRkpMRU5CUVVNc1RVRkJUU3hIUVVGSExFVkJRVVVzUTBGQlF5eE5RVUZOTEVOQlFVTTdXVUZEZUVJc1NVRkJTU3hEUVVGRExFVkJRVVVzUjBGQlJ5eEZRVUZGTEVOQlFVTXNSVUZCUlN4RFFVRkRPMWxCUTJoQ0xFbEJRVWtzUTBGQlF5eEZRVUZGTEVkQlFVY3NSVUZCUlN4RFFVRkRMRVZCUVVVc1EwRkJRenRaUVVOb1FpeEpRVUZKTEVOQlFVTXNVVUZCVVN4SFFVRkhMRXRCUVVzc1EwRkJRenRUUVVONlFqdGhRVUZOTzFsQlEwZ3NTVUZCU1N4RFFVRkRMRTFCUVUwc1IwRkJSeXhGUVVGRkxFTkJRVU1zVFVGQlRTeERRVUZETzFsQlEzaENMRWxCUVVrc1EwRkJReXhGUVVGRkxFZEJRVWNzUlVGQlJTeERRVUZETEVWQlFVVXNRMEZCUXp0WlFVTm9RaXhKUVVGSkxFTkJRVU1zUlVGQlJTeEhRVUZITEVOQlFVTXNRMEZCUXl4TlFVRk5MRWRCUVVjc1JVRkJSU3hEUVVGRExFVkJRVVVzUjBGQlJ5eEZRVUZGTEVOQlFVTXNUVUZCVFN4RFFVRkRPMWxCUTNaRExFbEJRVWtzUTBGQlF5eFJRVUZSTEVkQlFVY3NTVUZCU1N4RFFVRkRPMU5CUTNoQ08wbEJRMHdzUTBGQlF6dEpRVU5qTEd0RFFVRlRMRWRCUVhoQ0xGVkJRVFJDTEVOQlFVMHNSVUZCUlN4RFFVRk5PMUZCUTNSRExFbEJRVWtzUTBGQlF5eEhRVUZITEVOQlFVTXNRMEZCUXl4TlFVRk5MRU5CUVVNN1VVRkRha0lzU1VGQlNTeERRVUZETEVkQlFVY3NRMEZCUXl4RFFVRkRMRTFCUVUwc1EwRkJRenRSUVVOcVFpeEpRVUZKTEV0QlFVc3NSMEZCUnl4RlFVRkZMRTFCUVUwc1JVRkJSU3hEUVVGRExFVkJRVVVzUlVGQlJTeEZRVUZGTEVOQlFVTXNRMEZCUXl4RlFVRkZMRVZCUVVVc1JVRkJSU3hEUVVGRExFTkJRVU1zUlVGQlJTeERRVUZETzFGQlF6RkRMRWxCUVVrc1EwRkJReXhIUVVGSExFbEJRVWtzUzBGQlN5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRPMUZCUTNKQ0xFdEJRVXNzU1VGQlNTeERRVUZETEVkQlFVY3NRMEZCUXl4RlFVRkZMRU5CUVVNc1IwRkJSeXhEUVVGRExFVkJRVVVzUTBGQlF5eEZRVUZGTEVWQlFVVTdXVUZEZUVJc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eEhRVUZITEVsQlFVa3NTMEZCU3l4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRE8xbEJRM0JDTEV0QlFVc3NTVUZCU1N4RFFVRkRMRWRCUVVjc1EwRkJReXhGUVVGRkxFTkJRVU1zUjBGQlJ5eERRVUZETEVWQlFVVXNRMEZCUXl4RlFVRkZPMmRDUVVOMFFpeEpRVUZKTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1MwRkJTeXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVWQlFVVTdiMEpCUTJZc1NVRkJTU3hEUVVGRExFZEJRVWNzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhIUVVGSExFTkJRVU1zUTBGQlF5eExRVUZMTEVOQlFVTXNTVUZCU1N4RFFVRkRMRXRCUVVzc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUjBGQlJ5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRWRCUVVjc1EwRkJReXhEUVVGRExFZEJRVWNzUTBGQlF5eERRVUZETzI5Q1FVTnFSU3hKUVVGSkxFTkJRVU1zUjBGQlJ5eExRVUZMTEVOQlFVTXNUVUZCVFN4RlFVRkZPM2RDUVVOc1FpeExRVUZMTEVOQlFVTXNUVUZCVFN4SFFVRkhMRU5CUVVNc1EwRkJRenQzUWtGRGFrSXNTMEZCU3l4RFFVRkRMRVZCUVVVc1IwRkJSeXhEUVVGRExFZEJRVWNzUTBGQlF5eEhRVUZITEVOQlFVTXNRMEZCUXp0M1FrRkRja0lzUzBGQlN5eERRVUZETEVWQlFVVXNSMEZCUnl4RFFVRkRMRWRCUVVjc1EwRkJReXhIUVVGSExFTkJRVU1zUTBGQlF6dHhRa0ZEZUVJN2IwSkJRVUVzUTBGQlF6dHBRa0ZEVERzN2IwSkJRVTBzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhIUVVGSExFTkJRVU1zUTBGQlF6dFRRVU14UWp0UlFVTkVMRTlCUVU4c1MwRkJTeXhEUVVGRE8wbEJRMnBDTEVOQlFVTTdTVUZEUkN3NFEwRkJWeXhIUVVGWU8xRkJRMGtzVDBGQlR5eEpRVUZKTEVOQlFVTXNUVUZCVFN4SlFVRkpMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zU1VGQlNTeERRVUZETEVOQlFVTXNRMEZCUXl4TFFVRkxMRU5CUVVNc1NVRkJTU3hEUVVGRExFVkJRVVVzUlVGQlJTeEpRVUZKTEVOQlFVTXNSVUZCUlN4SFFVRkhMRWxCUVVrc1EwRkJReXhOUVVGTkxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNSVUZCUlN4RFFVRkRPMGxCUTJoR0xFTkJRVU03U1VGRFRDd3JRa0ZCUXp0QlFVRkVMRU5CUVVNc1FVRXpRMFFzU1VFeVEwTTdRVUV6UTFrc05FUkJRWGRDTzBGQmFVUnlRenRKUVhORVNTeHZRa0ZCYlVJc1lVRkJjVUlzUlVGQlJTeFJRVUUwUWl4RlFVRlRMRmxCUVhsQ08xRkJRWGhITEdsQ1FXdElRenRSUVd4SU9FVXNOa0pCUVVFc1JVRkJRU3hwUWtGQmVVSTdVVUZCY2tZc2EwSkJRV0VzUjBGQllpeGhRVUZoTEVOQlFWRTdVVUZCZFVNc2FVSkJRVmtzUjBGQldpeFpRVUZaTEVOQlFXRTdVVUZ5UkhoSExGZEJRVTBzUjBGQmEwSXNTVUZCU1N4RFFVRkRPMUZCYzBSNlFpeEpRVUZKTEVOQlFVTXNTMEZCU3l4SFFVRkhMR0ZCUVdFc1EwRkJReXhIUVVGSExFTkJRVU1zVlVGQlF5eERRVUZETEVWQlFVVXNRMEZCUXl4SlFVRkxMRTlCUVVFc1NVRkJTU3hYUVVGWExFTkJRVU1zUTBGQlF5eEZRVUZGTEZGQlFWRXNRMEZCUXl4VFFVRlRMRU5CUVVNc1EwRkJReXhEUVVGRExFVkJRVVVzVVVGQlVTeERRVUZETEZkQlFWY3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhGUVVGc1JTeERRVUZyUlN4RFFVRkRMRU5CUVVNN1VVRkROMGNzU1VGQlNTeERRVUZETEUxQlFVMHNSMEZCUnl4SlFVRkpMRU5CUVVNc1MwRkJTeXhEUVVGRExFMUJRVTBzUTBGQlF5eFZRVUZCTEVOQlFVTXNTVUZCUnl4UFFVRkJMRU5CUVVNc1EwRkJReXhKUVVGSkxFVkJRVTRzUTBGQlRTeERRVUZETEVOQlFVTTdVVUZETlVNc1NVRkJTU3hEUVVGRExFMUJRVTBzUjBGQlJ5eEpRVUZKTEVOQlFVTXNTMEZCU3l4RFFVRkRMRTFCUVUwc1EwRkJReXhWUVVGQkxFTkJRVU1zU1VGQlJ5eFBRVUZCTEVOQlFVTXNRMEZCUXl4RFFVRkRMRWxCUVVrc1JVRkJVQ3hEUVVGUExFTkJRVU1zUTBGQlF6dFJRVU0zUXl4SlFVRkpMRU5CUVVNc1NVRkJTU3hIUVVGSExFbEJRVWtzUTBGQlF5eFpRVUZaTEVOQlFVTXNSMEZCUnl4RFFVRkRMRU5CUVVNN1VVRkRia01zU1VGQlNTeERRVUZETEVsQlFVa3NSMEZCUnl4SlFVRkpMRU5CUVVNc1dVRkJXU3hEUVVGRExFZEJRVWNzUTBGQlF5eERRVUZETzFGQlIyNURMRWxCUVVrc1EwRkJReXhOUVVGTkxFTkJRVU1zVDBGQlR5eERRVUZETEZWQlFVRXNRMEZCUXp0WlFVTnFRaXhQUVVGQkxFTkJRVU1zUTBGQlF5eFJRVUZSTEVOQlFVTXNUMEZCVHl4RFFVRkRMRlZCUVVFc1EwRkJReXhKUVVGSExFOUJRVUVzUzBGQlNTeERRVUZETEV0QlFVc3NRMEZCVXl4RFFVRkRMRU5CUVVNc1EwRkJReXhOUVVGTkxFZEJRVWNzUTBGQlF5eEZRVUZvUXl4RFFVRm5ReXhEUVVGRE8xRkJRWGhFTEVOQlFYZEVMRU5CUVVNc1EwRkJRenRSUVVjNVJDeEpRVUZKTEVOQlFVTXNTVUZCU1N4SFFVRkhMRVZCUVVVc1VVRkJVU3hGUVVGRkxFVkJRVVVzUlVGQlJTeERRVUZETzFGQlF6ZENMRWxCUVVrc1EwRkJReXhMUVVGTExFTkJRVU1zVDBGQlR5eERRVUZETEZWQlFVRXNRMEZCUXp0WlFVTm9RaXhKUVVGSkxFOUJRVThzUTBGQlF5eERRVUZETEUxQlFVMHNTMEZCU3l4WFFVRlhMRVZCUVVVN1owSkJRMnBETEVOQlFVTXNRMEZCUXl4TlFVRk5MRWRCUVVjc1MwRkJTU3hEUVVGRExFbEJRVWtzUTBGQlF6dG5Ra0ZEY2tJc1MwRkJTU3hEUVVGRExFbEJRVWtzUTBGQlF5eFJRVUZSTEVOQlFVTXNTVUZCU1N4RFFVRkRMRU5CUVVNc1EwRkJReXhGUVVGRkxFTkJRVU1zUTBGQlF6dGhRVU5xUXp0WlFVOUVMRU5CUVVNc1EwRkJReXhMUVVGTExFZEJRVWNzUlVGQlJTeERRVUZCTzFGQlEyaENMRU5CUVVNc1EwRkJReXhEUVVGRE8xRkJSMGdzU1VGQlNTeERRVUZETEZkQlFWY3NSMEZCUnl4SlFVRkpMRU5CUVVNc1MwRkJTeXhEUVVGRExFdEJRVXNzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXp0UlFVTjJReXhKUVVGSkxFTkJRVU1zVjBGQlZ5eERRVUZETEVsQlFVa3NRMEZCUXl4VlFVRkRMRU5CUVVNc1JVRkJSU3hEUVVGRExFbEJRVXNzVDBGQlFTeExRVUZKTEVOQlFVTXNVVUZCVVN4RFFVRkRMRU5CUVVNc1EwRkJReXhIUVVGSExFdEJRVWtzUTBGQlF5eFJRVUZSTEVOQlFVTXNRMEZCUXl4RFFVRkRMRVZCUVc1RExFTkJRVzFETEVOQlFVTXNRMEZCUXp0UlFVdHlSU3hKUVVGSkxHbENRVUZwUWl4SFFVRkhMRWxCUVVrc1EwRkJReXhYUVVGWExFTkJRVU1zUzBGQlN5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRTlCUVU4c1JVRkJSU3hEUVVGRExFMUJRVTBzUTBGQlF5eFZRVUZCTEVOQlFVTXNTVUZCUnl4UFFVRkJMRU5CUVVNc1EwRkJReXhEUVVGRExFbEJRVWtzUlVGQlVDeERRVUZQTEVOQlFVTXNRMEZCUXp0UlFVTm9SaXhwUWtGQmFVSXNRMEZCUXl4UFFVRlBMRU5CUVVNc1ZVRkJRU3hEUVVGRE8xbEJRM1pDTEVsQlFVa3NRMEZCUXl4SFFVRkhMSEZDUVVGVExFTkJRVU1zUzBGQlN5eEZRVUZGTEVOQlFVTTdXVUZETVVJc1EwRkJReXhEUVVGRExGRkJRVkVzUTBGQlF5eFBRVUZQTEVOQlFVTXNWVUZCUVN4RFFVRkRMRWxCUVVjc1QwRkJRU3hEUVVGRExFZEJRVWNzUTBGQlF5eERRVUZETEV0QlFVc3NRMEZCUXl4TFFVRkpMRU5CUVVNc1MwRkJTeXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVsQlFVa3NRMEZCUXl4RlFVRXZRaXhEUVVFclFpeERRVUZETEVOQlFVTTdXVUZEZUVRc1EwRkJReXhEUVVGRExFbEJRVWtzUjBGQlJ5eERRVUZETEVOQlFVTXNUMEZCVHl4RFFVRkRMRXRCUVVrc1EwRkJReXhaUVVGWkxFTkJRVU1zUTBGQlF6dFJRVU14UXl4RFFVRkRMRU5CUVVNc1EwRkJRenRSUVVWSUxFbEJRVWtzVDBGQlR5eEhRVUZITEVsQlFVa3NRMEZCUXl4VFFVRlRMRU5CUVVNc1NVRkJTU3hEUVVGRExFbEJRVWtzUTBGQlF5eEhRVUZITEVOQlFVTXNWVUZCUVN4RFFVRkRMRWxCUVVjc1QwRkJRU3hEUVVGRExFTkJRVU1zUjBGQlJ5eEZRVUZNTEVOQlFVc3NRMEZCUXl4RFFVRkRMRU5CUVVNN1VVRkRka1FzU1VGQlNTeFBRVUZQTEVkQlFVY3NTVUZCU1N4RFFVRkRMRk5CUVZNc1EwRkJReXhKUVVGSkxFTkJRVU1zU1VGQlNTeERRVUZETEVkQlFVY3NRMEZCUXl4VlFVRkJMRU5CUVVNc1NVRkJSeXhQUVVGQkxFTkJRVU1zUTBGQlF5eEhRVUZITEVWQlFVd3NRMEZCU3l4RFFVRkRMRU5CUVVNc1EwRkJRenRSUVVkMlJDeEpRVUZKTEVsQlFVa3NSMEZCUnl4UFFVRlBMRU5CUVVNc1EwRkJReXhEUVVGRExFVkJRVVVzU1VGQlNTeEhRVUZITEU5QlFVOHNRMEZCUXl4UFFVRlBMRU5CUVVNc1RVRkJUU3hIUVVGSExFTkJRVU1zUTBGQlF5eERRVUZETzFGQlF6RkVMRWxCUVVrc1NVRkJTU3hIUVVGSExFOUJRVThzUTBGQlF5eERRVUZETEVOQlFVTXNSVUZCUlN4SlFVRkpMRWRCUVVjc1QwRkJUeXhEUVVGRExFOUJRVThzUTBGQlF5eE5RVUZOTEVkQlFVY3NRMEZCUXl4RFFVRkRMRU5CUVVNN1VVRkhNVVFzU1VGQlNTeE5RVUZOTEVkQlFVY3NTVUZCU1N4RFFVRkRMRWxCUVVrc1EwRkJReXhIUVVGSExFTkJRVU1zVlVGQlFTeERRVUZETEVsQlFVY3NUMEZCUVN4RFFVRkxMRVZCUVVVc1JVRkJSU3hGUVVGRkxFbEJRVWtzUlVGQlJTeEZRVUZGTEVWQlFVVXNTVUZCU1N4RlFVRkZMRVZCUVVVc1JVRkJSU3hEUVVGRExFTkJRVU1zUjBGQlJ5eEZRVUZGTEVWQlFVVXNSVUZCUlN4RFFVRkRMRU5CUVVNc1IwRkJSeXhGUVVGRkxFTkJRVUVzUlVGQmFrUXNRMEZCYVVRc1EwRkJRenRoUVVNMVJTeE5RVUZOTEVOQlFVTXNUMEZCVHl4RFFVRkRMRWRCUVVjc1EwRkJReXhWUVVGQkxFTkJRVU1zU1VGQlJ5eFBRVUZCTEVOQlFVc3NSVUZCUlN4RlFVRkZMRVZCUVVVc1NVRkJTU3hGUVVGRkxFVkJRVVVzUlVGQlJTeEpRVUZKTEVWQlFVVXNSVUZCUlN4RlFVRkZMRU5CUVVNc1JVRkJSU3hGUVVGRkxFVkJRVVVzUTBGQlF5eEZRVUZGTEVOQlFVRXNSVUZCZWtNc1EwRkJlVU1zUTBGQlF5eERRVUZETEVOQlFVTTdVVUZIZUVVc1NVRkJTU3hOUVVGTkxFZEJRVWNzU1VGQlNTeERRVUZETEVsQlFVa3NRMEZCUXl4SFFVRkhMRU5CUVVNc1ZVRkJRU3hEUVVGRExFbEJRVWNzVDBGQlFTeERRVUZMTEVWQlFVVXNSVUZCUlN4RlFVRkZMRU5CUVVNc1EwRkJReXhIUVVGSExFVkJRVVVzUlVGQlJTeEZRVUZGTEVOQlFVTXNRMEZCUXl4SFFVRkhMRVZCUVVVc1JVRkJSU3hGUVVGRkxFbEJRVWtzUlVGQlJTeEZRVUZGTEVWQlFVVXNTVUZCU1N4RlFVRkZMRU5CUVVFc1JVRkJha1FzUTBGQmFVUXNRMEZCUXp0aFFVTTFSU3hOUVVGTkxFTkJRVU1zVDBGQlR5eERRVUZETEVkQlFVY3NRMEZCUXl4VlFVRkJMRU5CUVVNc1NVRkJSeXhQUVVGQkxFTkJRVXNzUlVGQlJTeEZRVUZGTEVWQlFVVXNRMEZCUXl4RlFVRkZMRVZCUVVVc1JVRkJSU3hEUVVGRExFVkJRVVVzUlVGQlJTeEZRVUZGTEVsQlFVa3NSVUZCUlN4RlFVRkZMRVZCUVVVc1NVRkJTU3hGUVVGRkxFTkJRVUVzUlVGQmVrTXNRMEZCZVVNc1EwRkJReXhEUVVGRExFTkJRVU03VVVGSGVFVXNTVUZCU1N4TFFVRkxMRWRCUVVjc1RVRkJUU3hEUVVGRExFMUJRVTBzUTBGQlF5eE5RVUZOTEVOQlFVTXNRMEZCUXp0UlFVZHNReXhMUVVGTExFTkJRVU1zVDBGQlR5eERRVUZETEZWQlFVRXNRMEZCUXl4SlFVRkhMRTlCUVVFc1EwRkJReXhEUVVGRExFdEJRVXNzUjBGQlJ5eEZRVUZGTEVWQlFWb3NRMEZCV1N4RFFVRkRMRU5CUVVNN1VVRkhhRU1zU1VGQlNTeERRVUZETEV0QlFVc3NSMEZCUnl4RlFVRkZMRU5CUVVNN1VVRkRhRUlzU1VGQlNTeERRVUZETEV0QlFVc3NSMEZCUnl4RlFVRkZMRU5CUVVNN1VVRkhhRUlzVFVGQlRTeERRVUZETEU5QlFVOHNRMEZCUXl4VlFVRkJMRU5CUVVNN1dVRkRXaXhQUVVGQkxFMUJRVTBzUTBGQlF5eFBRVUZQTEVOQlFVTXNWVUZCUVN4RFFVRkRPMmRDUVVOYUxFbEJRVWtzUTBGQlF5eEhRVUZITEVsQlFVa3NTVUZCU1N4RFFVRkRMRXRCUVVrc1EwRkJReXhMUVVGTExFTkJRVU1zVFVGQlRTeEZRVUZGTEVOQlFVTXNRMEZCUXl4RlFVRkZMRVZCUVVVc1EwRkJReXhEUVVGRExFVkJRVVVzUTBGQlF5eERRVUZETzJkQ1FVTm9SQ3hEUVVGRExFTkJRVU1zUzBGQlN5eERRVUZETEVsQlFVa3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJRenRuUWtGRGFFSXNRMEZCUXl4RFFVRkRMRXRCUVVzc1EwRkJReXhKUVVGSkxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTTdaMEpCUTJoQ0xFdEJRVWtzUTBGQlF5eExRVUZMTEVOQlFVTXNTVUZCU1N4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRE8yZENRVWR1UWl4SlFVRkpMRU5CUVVNc1IwRkJSeXhMUVVGSkxFTkJRVU1zVjBGQlZ5eERRVUZETEUxQlFVMHNRMEZCUXp0blFrRkRhRU1zVDBGQlR5eERRVUZETEVWQlFVVXNSMEZCUnl4RFFVRkRMRVZCUVVVN2IwSkJRMW9zU1VGQlNTeEpRVUZKTEVkQlFVY3NTMEZCU1N4RFFVRkRMRmRCUVZjc1EwRkJReXhEUVVGRExFTkJRVU1zUlVGRE1VSXNRMEZCUXl4SFFVRkhMRWxCUVVrc1EwRkJReXhKUVVGSkxFTkJRVU03YjBKQlEyeENMRWxCUVVrc1JVRkJSU3hIUVVGSExFbEJRVWtzUTBGQlF5eEhRVUZITEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1IwRkJSeXhEUVVGRExFTkJRVU1zUlVGQlJTeEZRVUZGTEVOQlFVTXNSVUZETTBJc1JVRkJSU3hIUVVGSExFbEJRVWtzUTBGQlF5eEhRVUZITEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1IwRkJSeXhEUVVGRExFTkJRVU1zUlVGQlJTeEZRVUZGTEVOQlFVTXNRMEZCUXp0dlFrRkRhRU1zU1VGQlNTeEZRVUZGTEVkQlFVY3NRMEZCUXl4RFFVRkRMRXRCUVVzc1JVRkJSU3hIUVVGSExFTkJRVU1zU1VGQlNTeEZRVUZGTEVkQlFVY3NRMEZCUXl4RFFVRkRMRTFCUVUwc1JVRkJSU3hIUVVGSExFTkJRVU1zUlVGQlJUdDNRa0ZEY2tNc1EwRkJSU3hEUVVGRExFbEJRVWtzUjBGQlJ5eEpRVUZKTEVOQlFVTTdkMEpCUTNKQ0xFMUJRVTA3Y1VKQlExUTdhVUpCUTBvN1dVRkRUQ3hEUVVGRExFTkJRVU03VVVGc1FrWXNRMEZyUWtVc1EwRkRSQ3hEUVVGRE8xRkJSVTRzUzBGQlN5eERRVUZETEU5QlFVOHNRMEZCUXl4VlFVRkRMRU5CUVVNc1JVRkJSU3hGUVVGRk8xbEJSV2hDTEV0QlFVa3NRMEZCUXl4TFFVRkxMRU5CUVVNc1QwRkJUeXhEUVVGRExGVkJRVU1zUTBGQlF5eEZRVUZGTEVOQlFVTTdaMEpCUTNCQ0xFTkJRVU1zUTBGQlF5eEpRVUZKTEVOQlFVTXNhVUpCUVdsQ0xFTkJRVU1zUTBGQlF5eERRVUZETEVWQlFVVXNSVUZCUlN4RFFVRkRMRU5CUVVNc1JVRkJSU3hGUVVGRkxFTkJRVU1zUTBGQlF5eEZRVUZGTEVWQlFVVXNRMEZCUXl4RFFVRkRMRVZCUVVVc1EwRkJReXhEUVVGRExFOUJRVThzUTBGQlF5eFZRVUZETEZOQlFWTXNSVUZCUlN4RFFVRkRPMjlDUVVWc1JTeEpRVUZKTEVOQlFVTXNSMEZCUnl4SlFVRkpMRWxCUVVrc1EwRkJReXhMUVVGSkxFTkJRVU1zUzBGQlN5eERRVUZETEUxQlFVMHNSVUZCUlN4VFFVRlRMRU5CUVVNc1EwRkJReXhGUVVGRkxGTkJRVk1zUTBGQlF5eERRVUZETEVWQlFVVXNRMEZCUXl4RlFVRkZMRU5CUVVNc1EwRkJReXhEUVVGRE8yOUNRVU53UlN4TFFVRkpMRU5CUVVNc1MwRkJTeXhEUVVGRExFbEJRVWtzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXp0dlFrRkRia0lzUTBGQlF5eERRVUZETEV0QlFVc3NRMEZCUXl4SlFVRkpMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU03YjBKQlEyaENMRU5CUVVNc1EwRkJReXhMUVVGTExFTkJRVU1zU1VGQlNTeERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRPMmRDUVVOd1FpeERRVUZETEVOQlFVTXNRMEZCUXp0WlFVTlFMRU5CUVVNc1EwRkJReXhEUVVGRE8xbEJSMGdzU1VGQlNTeFBRVUZQTEVkQlFVY3NTVUZCU1N4RFFVRkRMRWRCUVVjc1EwRkJReXhEUVVGRExFTkJRVU1zUlVGQlJTeEhRVUZITEVOQlFVTXNRMEZCUXl4RlFVRkZMRU5CUVVNc1IwRkJSeXhIUVVGSExFTkJRVU03V1VGRE1VTXNTVUZCU1N4TFFVRkxMRWRCUVVjc1ZVRkJReXhEUVVGRExFVkJRVVVzUTBGQlF5eEpRVUZMTEU5QlFVRXNUMEZCVHl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eEhRVUZITEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVkQlFVY3NRMEZCUXl4RFFVRkRMRU5CUVVNc1JVRkJMMElzUTBGQkswSXNRMEZCUXp0WlFVTjBSQ3hEUVVGRExFTkJRVU1zUzBGQlN5eERRVUZETEVsQlFVa3NRMEZCUXl4TFFVRkxMRU5CUVVNc1EwRkJRenRaUVVOd1FpeExRVUZMTEVsQlFVa3NRMEZCUXl4SFFVRkhMRU5CUVVNc1JVRkJSU3hEUVVGRExFZEJRVWNzUTBGQlF5eERRVUZETEV0QlFVc3NRMEZCUXl4TlFVRk5MRVZCUVVVc1EwRkJReXhGUVVGRkxFVkJRVVU3WjBKQlEzSkRMRWxCUVVrc1EwRkJReXhIUVVGSExFTkJRVU1zUTBGQlF5eExRVUZMTEVOQlFVTXNRMEZCUXl4SFFVRkhMRU5CUVVNc1EwRkJReXhGUVVGRkxFTkJRVU1zUjBGQlJ5eERRVUZETEVOQlFVTXNTMEZCU3l4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRE8yZENRVU4yUXl4SlFVRkpMRU5CUVVNc1EwRkJReXhKUVVGSkxFbEJRVWtzUTBGQlF5eERRVUZETEVsQlFVa3NTMEZCU3l4RFFVRkRMRU5CUVVNc1NVRkJTU3hKUVVGSkxFTkJRVU1zUTBGQlF5eEpRVUZKTEVOQlFVTXNTVUZCU1R0dlFrRkJSU3hUUVVGVE8yZENRVU42UkN4TFFVRkpMRU5CUVVNc1MwRkJTeXhEUVVGRExFbEJRVWtzUTBGQlF5eEZRVUZGTEUxQlFVMHNSVUZCUlN4RFFVRkRMRU5CUVVNc1JVRkJSU3hGUVVGRkxFMUJRVTBzUlVGQlJTeERRVUZETEVOQlFVTXNSVUZCUlN4RlFVRkZMRTFCUVUwc1JVRkJSU3hKUVVGSkxFTkJRVU1zUjBGQlJ5eERRVUZETEV0QlFVc3NRMEZCUXl4RFFVRkRMRVZCUVVVc1EwRkJReXhEUVVGRExFTkJRVU1zUlVGQlJTeERRVUZETEVOQlFVTTdZVUZEYkVZN1VVRkRUQ3hEUVVGRExFTkJRVU1zUTBGQlF6dEpRVWxRTEVOQlFVTTdTVUUxU2s4c2QwSkJRVWNzUjBGQldDeFZRVUZaTEVOQlFVTXNTVUZCU1N4UFFVRlBMRU5CUVVNc1EwRkJReXhOUVVGTkxFTkJRVU1zVlVGQlF5eERRVUZETEVWQlFVVXNRMEZCUXl4SlFVRkxMRTlCUVVFc1EwRkJReXhIUVVGSExFTkJRVU1zUlVGQlRDeERRVUZMTEVOQlFVTXNSMEZCUnl4RFFVRkRMRU5CUVVNc1RVRkJUU3hEUVVGQkxFTkJRVU1zUTBGQlF6dEpRVWwwUkN4cFEwRkJXU3hIUVVGd1FpeFZRVUZ4UWl4SlFVRkpPMUZCUTNKQ0xFbEJRVWtzVDBGQlR5eEhRVUZITEVWQlFVVXNRMEZCUXp0UlFVTnFRaXhKUVVGSkxFVkJRVVVzUjBGQlJ5eEpRVUZKTEVOQlFVTXNUVUZCVFN4RFFVRkRMRXRCUVVzc1EwRkJReXhEUVVGRExFVkJRVVVzU1VGQlNTeERRVUZETEUxQlFVMHNRMEZCUXl4TlFVRk5MRU5CUVVNc1EwRkJRenRSUVVOc1JDeFBRVUZQTEVWQlFVVXNRMEZCUXl4TlFVRk5MRWRCUVVjc1EwRkJReXhGUVVGRk8xbEJSV3hDTEVsQlFVa3NWMEZCVnl4SFFVRkhMRVZCUVVVc1EwRkJReXhOUVVGTkxFTkJRVU1zVlVGQlFTeERRVUZETEVsQlFVY3NUMEZCUVN4RFFVRkRMRU5CUVVNc1NVRkJTU3hEUVVGRExGTkJRVk1zUjBGQlJ5eEpRVUZKTEVOQlFVTXNWMEZCVnl4RlFVRkZMRU5CUVVNc1EwRkJReXhGUVVGRkxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNTVUZCU1N4RFFVRkRMRVZCUVd4RUxFTkJRV3RFTEVOQlFVTXNRMEZCUXp0WlFVTndSaXhKUVVGSkxFZEJRVWNzUjBGQlJ6dG5Ra0ZEVGl4TFFVRkxMRVZCUVVVc1YwRkJWenRuUWtGRGJFSXNSMEZCUnl4RlFVRkZMRWxCUVVrc1EwRkJReXhIUVVGSExFTkJRVU1zVjBGQlZ5eERRVUZETEVkQlFVY3NRMEZCUXl4VlFVRkJMRU5CUVVNc1NVRkJSeXhQUVVGQkxFTkJRVU1zUTBGQlF5eEpRVUZKTEVOQlFVTXNSMEZCUnl4SFFVRkhMRWxCUVVrc1EwRkJReXhGUVVGRkxFVkJRWEJDTEVOQlFXOUNMRU5CUVVNc1EwRkJRenRoUVVNelJDeERRVUZETzFsQlEwWXNUMEZCVHl4RFFVRkRMRWxCUVVrc1EwRkJReXhIUVVGSExFTkJRVU1zUTBGQlF6dFpRVU5zUWl4SFFVRkhMRU5CUVVNc1MwRkJTeXhEUVVGRExFOUJRVThzUTBGQlF5eFZRVUZCTEVOQlFVTXNTVUZCUnl4UFFVRkJMRVZCUVVVc1EwRkJReXhOUVVGTkxFTkJRVU1zUlVGQlJTeERRVUZETEU5QlFVOHNRMEZCUXl4RFFVRkRMRU5CUVVNc1JVRkJSU3hEUVVGRExFTkJRVU1zUlVGQk0wSXNRMEZCTWtJc1EwRkJReXhEUVVGRE8xTkJRM1JFTzFGQlEwUXNUMEZCVHl4RFFVRkRMRWxCUVVrc1EwRkJReXhWUVVGRExFTkJRVU1zUlVGQlJTeERRVUZETEVsQlFVc3NUMEZCUVN4RFFVRkRMRU5CUVVNc1IwRkJSeXhIUVVGSExFTkJRVU1zUTBGQlF5eEhRVUZITEVWQlFXSXNRMEZCWVN4RFFVRkRMRU5CUVVFN1VVRkRja01zVDBGQlR5eFBRVUZQTEVOQlFVTTdTVUZEYmtJc1EwRkJRenRKUVVkUExEWkNRVUZSTEVkQlFXaENMRlZCUVdsQ0xFTkJRVU03VVVGRFpDeEpRVUZKTEV0QlFVc3NSMEZCUnl4RFFVRkRMRU5CUVVNN1VVRkRaQ3hQUVVGUExFTkJRVU1zUTBGQlF5eE5RVUZOTEV0QlFVc3NTVUZCU1N4RFFVRkRMRWxCUVVrc1JVRkJSVHRaUVVNelFpeExRVUZMTEVWQlFVVXNRMEZCUXp0WlFVTlNMRU5CUVVNc1IwRkJSeXhEUVVGRExFTkJRVU1zVFVGQlRTeERRVUZETzFOQlEyaENPMUZCUTBRc1QwRkJUeXhMUVVGTExFTkJRVU03U1VGRGFrSXNRMEZCUXp0SlFVZFBMRGhDUVVGVExFZEJRV3BDTEZWQlFXdENMRU5CUVVNN1VVRkRaaXhKUVVGSkxFZEJRVWNzUjBGQlJ5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRWRCUVVjc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETzFGQlEzUkNMRWxCUVVrc1NVRkJTU3hIUVVGSExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4SFFVRkhMRWRCUVVjc1IwRkJSeXhEUVVGRExFTkJRVU1zUTBGQlF6dFJRVU0xUWl4TFFVRkxMRWxCUVVrc1EwRkJReXhIUVVGSExFTkJRVU1zUlVGQlJTeERRVUZETEVkQlFVY3NRMEZCUXl4RFFVRkRMRTFCUVUwc1JVRkJSU3hEUVVGRExFVkJRVVVzUlVGQlJUdFpRVU12UWl4SlFVRkpMRU5CUVVNc1NVRkJTU3hEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4SFFVRkhMRU5CUVVNc1EwRkJReXhEUVVGRExFZEJRVWNzUTBGQlF5eERRVUZETEVOQlFVTXNSMEZCUnl4RFFVRkRMRU5CUVVNc1EwRkJRenRUUVVOd1F6dFJRVU5FTEVsQlFVa3NRMEZCUXl4SlFVRkpMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eE5RVUZOTEVkQlFVY3NRMEZCUXl4RFFVRkRMRWRCUVVjc1IwRkJSeXhIUVVGSExFTkJRVU1zUTBGQlF5eERRVUZETzFGQlEzSkRMRTlCUVU4c1NVRkJTU3hEUVVGRE8wbEJRMmhDTEVOQlFVTTdTVUYxU0U4c1owTkJRVmNzUjBGQmJrSXNWVUZCYjBJc1EwRkJRenRSUVVOcVFpeEpRVUZKTEU5QlFVOHNSMEZCUnl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRE8xRkJRMnhDTEVkQlFVYzdXVUZEUXl4RFFVRkRMRWRCUVVjc1EwRkJReXhEUVVGRExFMUJRVTBzUTBGQlF6dFpRVU5pTEU5QlFVOHNRMEZCUXl4SlFVRkpMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU03VTBGRGJrSXNVVUZCVVN4RFFVRkRMRXRCUVVzc1NVRkJTU3hEUVVGRExFbEJRVWtzUlVGQlJUdFJRVU14UWl4UFFVRlBMRTlCUVU4c1EwRkJReXhQUVVGUExFVkJRVVVzUTBGQlF6dEpRVU0zUWl4RFFVRkRPMGxCUjA4c05FTkJRWFZDTEVkQlFTOUNMRlZCUVdkRExFTkJRVU1zUlVGQlJTeERRVUZETzFGQlEyaERMRWxCUVVrc1JVRkJSU3hIUVVGSExFbEJRVWtzUTBGQlF5eFhRVUZYTEVOQlFVTXNRMEZCUXl4RFFVRkRMRVZCUVVVc1JVRkJSU3hIUVVGSExFbEJRVWtzUTBGQlF5eFhRVUZYTEVOQlFVTXNRMEZCUXl4RFFVRkRMRVZCUVVVc1EwRkJReXhIUVVGSExFTkJRVU1zUTBGQlF6dFJRVU01UkN4UFFVRlBMRVZCUVVVc1EwRkJReXhEUVVGRExFTkJRVU1zUzBGQlN5eEZRVUZGTEVOQlFVTXNRMEZCUXl4RFFVRkRPMWxCUVVVc1EwRkJReXhGUVVGRkxFTkJRVU03VVVGRk5VSXNUMEZCVHl4RlFVRkZMR05CUVdNc1JVRkJSU3hGUVVGRkxFTkJRVU1zUTBGQlF5eEhRVUZITEVOQlFVTXNRMEZCUXl4RlFVRkZMRkZCUVZFc1JVRkJSU3hGUVVGRkxFTkJRVU1zUzBGQlN5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRTFCUVUwc1EwRkJReXhGUVVGRkxFTkJRVU1zUzBGQlN5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRVZCUVVVc1EwRkJRenRKUVVOd1JpeERRVUZETzBsQlNVUXNjVU5CUVdkQ0xFZEJRV2hDTEZWQlFXbENMRU5CUVVNc1JVRkJSU3hEUVVGRE8xRkJRWEpDTEdsQ1FWZERPMUZCVmtjc1NVRkJTU3hKUVVGSkxFZEJRVWNzU1VGQlNTeERRVUZETEhWQ1FVRjFRaXhEUVVGRExFTkJRVU1zUlVGQlJTeERRVUZETEVOQlFVTXNRMEZCUXp0UlFVTTVReXhKUVVGSkxHRkJRV0VzUjBGQlJ5eEZRVUZGTEVOQlFVTTdVVUZEZGtJc1NVRkJTU3hEUVVGRExGRkJRVkVzUTBGQlF5eFBRVUZQTEVOQlFVTXNWVUZCUVN4RFFVRkRMRWxCUVVjc1QwRkJRU3hoUVVGaExFTkJRVU1zUTBGQlF5eERRVUZETEVWQlFVVXNRMEZCUXl4SFFVRkhMRVZCUVVVc1JVRkJlRUlzUTBGQmQwSXNRMEZCUXl4RFFVRkRPMUZCUTNCRUxFbEJRVWtzVTBGQlV5eEhRVUZITEVsQlFVa3NRMEZCUXl4alFVRmpMRU5CUVVNc1VVRkJVU3hEUVVGRExFMUJRVTBzUTBGQlF5eFZRVUZCTEVOQlFVTXNTVUZCUnl4UFFVRkJMRU5CUVVNc1EwRkJReXhEUVVGRExFbEJRVWtzWVVGQllTeERRVUZETEVWQlFYSkNMRU5CUVhGQ0xFTkJRVU1zUTBGQlF6dFJRVVV2UlN4SlFVRkpMRU5CUVVNc1VVRkJVVHRoUVVOU0xFMUJRVTBzUTBGQlF5eFZRVUZCTEVOQlFVTXNTVUZCUnl4UFFVRkJMRU5CUVVNc1EwRkJReXhOUVVGTkxFdEJRVXNzU1VGQlNTeERRVUZETEdOQlFXTXNSVUZCYUVNc1EwRkJaME1zUTBGQlF6dGhRVU0xUXl4UFFVRlBMRU5CUVVNc1ZVRkJRU3hEUVVGRExFbEJRVWNzVDBGQlFTeFRRVUZUTEVkQlFVY3NVMEZCVXl4RFFVRkRMRTFCUVUwc1EwRkJReXhEUVVGRExFTkJRVU1zVFVGQlRTeERRVUZETEZGQlFWRXNRMEZCUXl4TlFVRk5MRU5CUVVNc1ZVRkJRU3hEUVVGRExFbEJRVWNzVDBGQlFTeERRVUZETEV0QlFVc3NRMEZCUXl4RFFVRkRMRVZCUVVVc1JVRkJWaXhEUVVGVkxFTkJRVU1zUTBGQlF5eEZRVUYwUlN4RFFVRnpSU3hEUVVGRExFTkJRVU03VVVGRmVrWXNUMEZCVHl4VFFVRlRMRU5CUVVNc1IwRkJSeXhEUVVGRExGVkJRVUVzUTBGQlF5eEpRVUZITEU5QlFVRXNTMEZCU1N4RFFVRkRMRXRCUVVzc1EwRkJReXhEUVVGRExFTkJRVU1zUlVGQllpeERRVUZoTEVOQlFVTXNRMEZCUXp0SlFVTTFReXhEUVVGRE8wbEJTVTBzZVVKQlFXTXNSMEZCY2tJc1ZVRkJjMElzVFVGQlRTeEZRVUZGTEVOQlFVTXNSVUZCUlN4RFFVRkRPMUZCUlRsQ0xFbEJRVWtzVTBGQlV5eEhRVUZITEVWQlFVVXNRMEZCUXp0UlFVTnVRaXhMUVVGTExFbEJRVWtzUlVGQlJTeEhRVUZITEVOQlFVTXNSVUZCUlN4RlFVRkZMRWRCUVVjc1RVRkJUU3hEUVVGRExFMUJRVTBzUlVGQlJTeEZRVUZGTEVWQlFVVXNSVUZCUlR0WlFVTjJReXhKUVVGSkxFdEJRVXNzUjBGQlJ5eE5RVUZOTEVOQlFVTXNSVUZCUlN4RFFVRkRMRU5CUVVNN1dVRkRka0lzUzBGQlN5eEpRVUZKTEVWQlFVVXNSMEZCUnl4RFFVRkRMRVZCUVVVc1JVRkJSU3hIUVVGSExFdEJRVXNzUTBGQlF5eE5RVUZOTEVWQlFVVXNSVUZCUlN4RlFVRkZMRVZCUVVVN1owSkJRM1JETEVsQlFVa3NRMEZCUXl4SFFVRlJMRXRCUVVzc1EwRkJReXhGUVVGRkxFTkJRVU1zUTBGQlF6dG5Ra0ZEZGtJc1EwRkJReXhEUVVGRExFMUJRVTBzUjBGQlJ5eEZRVUZGTEVOQlFVTTdaMEpCUTJRc1EwRkJReXhEUVVGRExFTkJRVU1zUjBGQlJ5eEZRVUZGTEVOQlFVTTdaMEpCUTFRc1NVRkJTU3hIUVVGSExFZEJRVWNzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhIUVVGSExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJRenRuUWtGRE5VSXNTVUZCU1N4SlFVRkpMRU5CUVVNc1IwRkJSeXhEUVVGRExFZEJRVWNzUTBGQlF5eEhRVUZITEVkQlFVY3NSVUZCUlR0dlFrRkRja0lzVTBGQlV5eERRVUZETEVsQlFVa3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJRenRwUWtGRGNrSTdZVUZEU2p0VFFVTktPMUZCUTBRc1UwRkJVeXhEUVVGRExFbEJRVWtzUTBGQlF5eFZRVUZETEVOQlFVTXNSVUZCUlN4RFFVRkRMRWxCUVVzc1QwRkJRU3hEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRWRCUVVjc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RlFVRnFRaXhEUVVGcFFpeERRVUZETEVOQlFVTTdVVUZITlVNc1NVRkJTU3haUVVGWkxFZEJRVWNzUlVGQlJTeERRVUZETzFGQlEzUkNMRWxCUVVrc1ZVRkJWU3hIUVVGSExFbEJRVWtzUTBGQlF6dFJRVU4wUWl4TFFVRkxMRWxCUVVrc1EwRkJReXhIUVVGSExFTkJRVU1zUlVGQlJTeERRVUZETEVkQlFVY3NVMEZCVXl4RFFVRkRMRTFCUVUwc1JVRkJSU3hEUVVGRExFVkJRVVVzUlVGQlJUdFpRVU4yUXl4SlFVRkpMRU5CUVVNc1IwRkJSeXhUUVVGVExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTTdXVUZEY2tJc1NVRkJTU3hEUVVGRExGVkJRVlVzU1VGQlNTeEpRVUZKTEVOQlFVTXNSMEZCUnl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNSMEZCUnl4VlFVRlZMRU5CUVVNc1IwRkJSeXhEUVVGRExFZEJRVWNzUjBGQlJ5eEZRVUZGTzJkQ1FVTjZSQ3hWUVVGVkxFZEJRVWNzUlVGQlJTeEhRVUZITEVWQlFVVXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eEZRVUZGTEZGQlFWRXNSVUZCUlN4RlFVRkZMRVZCUVVVc1EwRkJRenRuUWtGRE5VTXNXVUZCV1N4RFFVRkRMRWxCUVVrc1EwRkJReXhWUVVGVkxFTkJRVU1zUTBGQlF6dGhRVU5xUXp0WlFVTkVMRlZCUVZVc1EwRkJReXhSUVVGUkxFTkJRVU1zU1VGQlNTeERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRPMU5CUXk5Q08xRkJRMFFzVDBGQlR5eFpRVUZaTEVOQlFVTTdTVUZEZUVJc1EwRkJRenRKUVZOTkxHOUNRVUZUTEVkQlFXaENMRlZCUVdsQ0xFTkJRVk1zUlVGQlJTeERRVUZUTEVWQlFVVXNUVUZCVFN4RlFVRkZMRkZCUVZFc1JVRkJSU3hOUVVGTkxFVkJRVVVzUjBGQlZ6dFJRVU40UlN4SlFVRkpMRU5CUVVNc1IwRkJSeXhSUVVGUkxFTkJRVU1zVFVGQlRTeERRVUZETzFGQlEzaENMRWxCUVVrc1EwRkJReXhKUVVGSkxFTkJRVU03V1VGQlJTeFBRVUZQTzFGQlEyNUNMRWxCUVVrc1JVRkJSU3hIUVVGSExGRkJRVkVzUTBGQlF5eEhRVUZITEVOQlFVTXNWVUZCUVN4RFFVRkRMRWxCUVVrc1QwRkJRU3hKUVVGSkxHVkJRVkVzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUlVGQmNrSXNRMEZCY1VJc1EwRkJReXhEUVVGRE8xRkJRMnhFTEVsQlFVa3NSVUZCUlN4SFFVRkhMRVZCUVVVc1EwRkJRenRSUVVOYUxFdEJRVXNzU1VGQlNTeERRVUZETEVkQlFVY3NRMEZCUXl4RlFVRkZMRU5CUVVNc1IwRkJSeXhEUVVGRExFVkJRVVVzUTBGQlF5eEZRVUZGTEVWQlFVVTdXVUZEZUVJc1MwRkJTeXhKUVVGSkxFTkJRVU1zUjBGQlJ5eERRVUZETEVWQlFVVXNRMEZCUXl4SFFVRkhMRU5CUVVNc1JVRkJSU3hEUVVGRExFVkJRVVVzUlVGQlJUdG5Ra0ZEZUVJc1NVRkJTU3hEUVVGRExFdEJRVXNzUTBGQlF6dHZRa0ZCUlN4VFFVRlRPMmRDUVVOMFFpeEpRVUZKTEVWQlFVVXNSMEZCUnl4UlFVRlJMRU5CUVVNc1EwRkJReXhEUVVGRExFVkJRMmhDTEVWQlFVVXNSMEZCUnl4UlFVRlJMRU5CUVVNc1EwRkJReXhEUVVGRExFVkJRMmhDTEVWQlFVVXNSMEZCUnl4RlFVRkZMRU5CUVVNc1RVRkJUU3hGUVVOa0xFVkJRVVVzUjBGQlJ5eEZRVUZGTEVOQlFVTXNUVUZCVFN4RlFVTmtMRWxCUVVrc1IwRkJSeXhEUVVGRExFTkJRVU1zUlVGRFZDeEpRVUZKTEVkQlFVY3NRMEZCUXl4RFFVRkRMRU5CUVVNN1owSkJUV1FzU1VGQlNTeERRVUZETEVsQlFVa3NSMEZCUnl4RlFVRkZPMjlDUVVOV0xFbEJRVWtzVFVGQlRTeERRVUZETEVWQlFVVXNSVUZCUlN4RlFVRkZMRU5CUVVNc1JVRkJSVHQzUWtGRmFFSXNTVUZCU1N4RlFVRkZMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVkQlFVY3NSVUZCUlN4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eEZRVUZGT3pSQ1FVTnlRaXhKUVVGSkxFZEJRVWNzUTBGQlF5eEZRVUZGTEVsQlFVa3NSMEZCUnl4RFFVRkRMRU5CUVVNN2VVSkJRM1JDT3paQ1FVRk5PelJDUVVOSUxFbEJRVWtzUjBGQlJ5eERRVUZETEVWQlFVVXNTVUZCU1N4SFFVRkhMRU5CUVVNc1EwRkJRenQ1UWtGRGRFSTdjVUpCUTBvN2FVSkJRMG83Y1VKQlFVMDdiMEpCUTBnc1NVRkJTU3hOUVVGTkxFTkJRVU1zUlVGQlJTeEZRVUZGTEVWQlFVVXNRMEZCUXl4RlFVRkZPM2RDUVVOb1FpeEpRVUZKTEVWQlFVVXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUjBGQlJ5eEZRVUZGTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFVkJRVVU3TkVKQlEzSkNMRWxCUVVrc1IwRkJSeXhEUVVGRExFVkJRVVVzU1VGQlNTeEhRVUZITEVOQlFVTXNRMEZCUXp0NVFrRkRkRUk3TmtKQlFVMDdORUpCUTBnc1NVRkJTU3hIUVVGSExFTkJRVU1zUlVGQlJTeEpRVUZKTEVkQlFVY3NRMEZCUXl4RFFVRkRPM2xDUVVOMFFqdHhRa0ZEU2p0cFFrRkRTanRuUWtGRFJDeEpRVUZKTEVsQlFVa3NTVUZCU1N4RFFVRkRMRVZCUVVVN2IwSkJSVmdzUlVGQlJTeERRVUZETEVsQlFVa3NRMEZCUXl4SlFVRkpMR2xDUVVGVkxFTkJRVU1zUlVGQlJTeERRVUZETEVsQlFVa3NRMEZCUXl4RlFVRkZMRVZCUVVVc1EwRkJReXhKUVVGSkxFTkJRVU1zUlVGQlJTeEhRVUZITEVOQlFVTXNRMEZCUXl4RFFVRkRPMmxDUVVOd1JEdGhRVU5LTzFOQlEwbzdVVUZEUkN4SlFVRkpMRTFCUVUwc1IwRkJSeXhKUVVGSkxHRkJRVTBzUTBGQlF5eEZRVUZGTEVWQlFVVXNSVUZCUlN4RFFVRkRMRU5CUVVNN1VVRkRhRU1zVFVGQlRTeERRVUZETEV0QlFVc3NSVUZCUlN4RFFVRkRPMUZCUTJZc1JVRkJSU3hEUVVGRExFOUJRVThzUTBGQlF5eFZRVUZETEVOQlFVTXNSVUZCUlN4RFFVRkRPMWxCUTFvc1NVRkJTU3hEUVVGRExFZEJRVWNzVVVGQlVTeERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRPMWxCUTNCQ0xFbEJRVWtzUjBGQlJ5eEhRVUZITEVOQlFVTXNRMEZCUXl4UlFVRlJMRVZCUVVVc1EwRkJRenRaUVVOMlFpeERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFZEJRVWNzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhIUVVGSExFZEJRVWNzUTBGQlF6dFpRVU40UWl4SlFVRkpMRXRCUVVzc1IwRkJSeXhOUVVGTkxFTkJRVU1zUTBGQlF5eERRVUZETEUxQlFVMHNRMEZCUXl4RFFVRkRPMWxCUXpkQ0xFbEJRVWtzUTBGQlF5eERRVUZETEVOQlFVTXNSMEZCUnl4RFFVRkRPMmRDUVVGRkxFdEJRVXNzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4SFFVRkhMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4SFFVRkhMRWRCUVVjc1EwRkJRenRaUVVONFF5eEpRVUZKTEVOQlFVTXNRMEZCUXl4RFFVRkRMRWRCUVVjc1MwRkJTeXhEUVVGRExFMUJRVTBzUjBGQlJ5eERRVUZETzJkQ1FVRkZMRXRCUVVzc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eEhRVUZITEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eEhRVUZITEVkQlFVY3NRMEZCUXp0UlFVTXpSQ3hEUVVGRExFTkJRVU1zUTBGQlF6dEpRVU5RTEVOQlFVTTdTVUZGVFN4M1FrRkJZU3hIUVVGd1FpeFZRVUZ4UWl4TlFVRk5MRVZCUVVVc1EwRkJVeXhGUVVGRkxFTkJRVk1zUlVGQlJTeE5RVUV5UXl4RlFVRkZMRWRCUVZjN1VVRkRka2NzU1VGQlNTeFpRVUZaTEVkQlFVY3NWVUZCVlN4RFFVRkRMR05CUVdNc1EwRkJReXhOUVVGTkxFVkJRVVVzUTBGQlF5eEZRVUZGTEVOQlFVTXNRMEZCUXl4RFFVRkRPMUZCUlRORUxFdEJRVXNzU1VGQlNTeERRVUZETEVkQlFVY3NRMEZCUXl4RlFVRkZMRU5CUVVNc1IwRkJSeXhaUVVGWkxFTkJRVU1zVFVGQlRTeEZRVUZGTEVOQlFVTXNSVUZCUlN4RlFVRkZPMWxCUXpGRExFbEJRVWtzUlVGQlJTeEhRVUZITEZsQlFWa3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJRenRaUVVONlFpeEpRVUZKTEUxQlFVMHNSMEZCUnl4RlFVRkZMRU5CUVVNN1dVRkRhRUlzUzBGQlN5eEpRVUZKTEVOQlFVTXNSMEZCUnl4RFFVRkRMRVZCUVVVc1EwRkJReXhIUVVGSExFVkJRVVVzUTBGQlF5eFJRVUZSTEVOQlFVTXNUVUZCVFN4RlFVRkZMRU5CUVVNc1JVRkJSU3hGUVVGRk8yZENRVU42UXl4SlFVRkpMRU5CUVVNc1IwRkJSeXhGUVVGRkxFTkJRVU1zVVVGQlVTeERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRPMmRDUVVOMlFpeE5RVUZOTEVOQlFVTXNTVUZCU1N4RFFVRkRMRVZCUVVVc1NVRkJTU3hGUVVGRkxFTkJRVU1zUlVGQlJTeERRVUZETEVWQlFVVXNRMEZCUXl4RlFVRkZMRWRCUVVjc1JVRkJSU3hKUVVGSkxFTkJRVU1zUjBGQlJ5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUlVGQlJTeERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUlVGQlJTeERRVUZETEVOQlFVTTdaMEpCUTJoRkxFMUJRVTBzUTBGQlF5eEpRVUZKTEVOQlFVTXNSVUZCUlN4SlFVRkpMRVZCUVVVc1EwRkJReXhGUVVGRkxFTkJRVU1zUlVGQlJTeERRVUZETEVWQlFVVXNSMEZCUnl4RlFVRkZMRWxCUVVrc1EwRkJReXhIUVVGSExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhGUVVGRkxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhGUVVGRkxFTkJRVU1zUTBGQlF6dGhRVU51UlR0WlFVTkVMRTFCUVUwc1EwRkJReXhKUVVGSkxFTkJRVU1zVlVGQlF5eERRVUZETEVWQlFVVXNRMEZCUXl4SlFVRkxMRTlCUVVFc1EwRkJReXhEUVVGRExFZEJRVWNzUjBGQlJ5eERRVUZETEVOQlFVTXNSMEZCUnl4SFFVRkhMRU5CUVVNc1EwRkJReXhKUVVGSkxFZEJRVWNzUTBGQlF5eERRVUZETEVsQlFVa3NSVUZCTDBJc1EwRkJLMElzUTBGQlF5eERRVUZETzFsQlEzWkVMRWxCUVVrc1NVRkJTU3hIUVVGSExFVkJRVVVzUTBGQlF6dFpRVU5rTEVsQlFVa3NVMEZCVXl4SFFVRkhMRU5CUVVNc1EwRkJRenRaUVVOc1FpeE5RVUZOTEVOQlFVTXNUMEZCVHl4RFFVRkRMRlZCUVVFc1EwRkJRenRuUWtGRFdpeEpRVUZKTEVOQlFVTXNRMEZCUXl4SlFVRkpMRXRCUVVzc1EwRkJReXhGUVVGRk8yOUNRVU5rTEVsQlFVa3NRMEZCUXl4SlFVRkpMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETzI5Q1FVTm1MRk5CUVZNc1JVRkJSU3hEUVVGRE8ybENRVU5tTzNGQ1FVRk5PMjlDUVVOSUxGTkJRVk1zUlVGQlJTeERRVUZETzJsQ1FVTm1PMmRDUVVORUxFbEJRVWtzVTBGQlV5eEpRVUZKTEVOQlFVTXNSVUZCUlR0dlFrRkRhRUlzVlVGQlZTeERRVUZETEZOQlFWTXNRMEZCUXl4RFFVRkRMRVZCUVVVc1EwRkJReXhGUVVGRkxFMUJRVTBzUlVGQlJTeEpRVUZKTEVWQlFVVXNUVUZCVFN4RlFVRkZMRWRCUVVjc1EwRkJReXhEUVVGRE8yOUNRVU4wUkN4SlFVRkpMRWRCUVVjc1JVRkJSU3hEUVVGRE8ybENRVU5pTzFsQlEwd3NRMEZCUXl4RFFVRkRMRU5CUVVNN1UwRkRUanRKUVVOTUxFTkJRVU03U1VGVFJDd3JRa0ZCVlN4SFFVRldMRlZCUVdsQ0xFdEJRV0VzUlVGQlJTeFJRVUZuUWl4RlFVRkZMRTFCUVRKQ0xFVkJRVVVzVFVGQk1rSTdVVUZCTVVjc2FVSkJVVU03VVVGUVJ5eEpRVUZKTEZWQlFWVXNSMEZCUnl4TFFVRkxMRU5CUVVNc1IwRkJSeXhEUVVGRExGVkJRVUVzUTBGQlF5eEpRVUZITEU5QlFVRXNTMEZCU1N4RFFVRkRMRXRCUVVzc1EwRkJReXhOUVVGTkxFTkJRVU1zUTBGQlF5eERRVUZETEVWQlFVVXNUVUZCVFN4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFVkJRV2hETEVOQlFXZERMRU5CUVVNc1EwRkJRenRSUVVOcVJTeEpRVUZKTEV0QlFVc3NSMEZCUnl4VlFVRlZMRU5CUVVNc1ZVRkJWU3hEUVVGRExGVkJRVlVzUTBGQlF5eERRVUZETzFGQlF6bERMRWxCUVVrc1RVRkJUU3hIUVVGSExGVkJRVlVzUTBGQlF5eEhRVUZITEVOQlFVTXNWVUZCVlN4RFFVRkRMRWxCUVVrc1QwRkJUeXhWUVVGVkxFTkJRVU1zV1VGQldTeERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU03VVVGRGFrWXNWVUZCVlN4RFFVRkRMR0ZCUVdFc1EwRkJReXhOUVVGTkxFVkJRVVVzUjBGQlJ5eEZRVUZGTEVkQlFVY3NSVUZCUlN4TFFVRkxMRVZCUVVVc1VVRkJVU3hEUVVGRExFTkJRVU03VVVGRE5VUXNWVUZCVlN4RFFVRkRMR0ZCUVdFc1EwRkJReXhOUVVGTkxFVkJRVVVzUjBGQlJ5eEZRVUZGTEVkQlFVY3NSVUZCUlN4TFFVRkxMRVZCUVVVc1VVRkJVU3hEUVVGRExFTkJRVU03VVVGRE5VUXNWVUZCVlN4RFFVRkRMR05CUVdNc1EwRkJReXhOUVVGTkxFVkJRVVVzVlVGQlZTeERRVUZETEVOQlFVTTdVVUZET1VNc1QwRkJUeXhOUVVGTkxFTkJRVU03U1VGRGJFSXNRMEZCUXp0SlFVbE5MSGxDUVVGakxFZEJRWEpDTEZWQlFYTkNMRTFCUVUwc1JVRkJSU3hWUVVGVk8xRkJRM0JETEUxQlFVMHNRMEZCUXl4UFFVRlBMRU5CUVVNc1ZVRkJReXhSUVVGUkxFVkJRVVVzUTBGQlF6dFpRVU4yUWl4SlFVRkpMRWxCUVVrc1IwRkJSeXhWUVVGVkxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTTdXVUZEZWtJc1NVRkJWU3hKUVVGTExFTkJRVU1zVVVGQlVTeEZRVUZGTzJkQ1FVTjBRaXhSUVVGUkxFTkJRVU1zVDBGQlR5eEZRVUZGTEVOQlFVTTdaMEpCUTI1Q0xGRkJRVkVzUTBGQlF5eFBRVUZQTEVOQlFVTXNWVUZCVlN4UFFVRlBPMjlDUVVNNVFpeFBRVUZQTEVOQlFVTXNUMEZCVHl4RlFVRkZMRU5CUVVNN1owSkJRM1JDTEVOQlFVTXNRMEZCUXl4RFFVRkRPMkZCUTA0N1VVRkRUQ3hEUVVGRExFTkJRVU1zUTBGQlF6dEpRVU5RTEVOQlFVTTdTVUZGVFN3MlFrRkJhMElzUjBGQmVrSXNWVUZCTUVJc1MwRkJZeXhGUVVGRkxFdEJRV003VVVGRGNFUXNTVUZCU1N4TlFVRk5MRWRCUVVjc1NVRkJTU3hEUVVGRExFdEJRVXNzUTBGQlF5eExRVUZMTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhIUVVGSExFdEJRVXNzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRVZCUXpORExFdEJRVXNzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRWRCUVVjc1MwRkJTeXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRPMUZCUXpkQ0xFbEJRVWtzVFVGQlRTeEhRVUZITEVsQlFVa3NRMEZCUXl4TFFVRkxMRU5CUVVNc1MwRkJTeXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNSMEZCUnl4TFFVRkxMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eEZRVU16UXl4TFFVRkxMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eEhRVUZITEV0QlFVc3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dFJRVU0zUWl4SlFVRkpMRWxCUVVrc1IwRkJSeXhOUVVGTkxFZEJRVWNzVFVGQlRTeERRVUZETzFGQlF6TkNMRWxCUVVrc1NVRkJTU3hIUVVGSExFbEJRVWtzUTBGQlF5eEZRVUZGTEVsQlFVa3NTVUZCU1N4SFFVRkhMRU5CUVVNc1NVRkJTU3hEUVVGRExFVkJRVVVzUlVGQlJUdFpRVU51UXl4SlFVRkpMRWRCUVVjc1RVRkJUU3hIUVVGSExFMUJRVTBzUTBGQlF6dFRRVU14UWp0UlFVTkVMRTlCUVU4c1NVRkJTU3hEUVVGRE8wbEJRMmhDTEVOQlFVTTdTVUZIWXl4cFFrRkJUU3hIUVVGeVFpeFZRVUZ6UWl4RFFVRkRMRVZCUVVVc1EwRkJReXhGUVVGRkxFTkJRVU03VVVGRGVrSXNUMEZCVHl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUjBGQlJ5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRWRCUVVjc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eEhRVUZITEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1IwRkJSeXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVkQlFVY3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhIUVVGSExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNSMEZCUnl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zU1VGQlNTeERRVUZETEVOQlFVTTdTVUZEZUVVc1EwRkJRenRKUVVsakxHMUNRVUZSTEVkQlFYWkNMRlZCUVhkQ0xFdEJRV2xETzFGQlEzSkVMRWxCUVVrc1VVRkJVU3hIUVVGSExFVkJRVVVzUTBGQlF6dFJRVU5zUWl4TFFVRkxMRWxCUVVrc1EwRkJReXhIUVVGSExFTkJRVU1zUlVGQlJTeERRVUZETEVkQlFVY3NTMEZCU3l4RFFVRkRMRTFCUVUwc1JVRkJSU3hEUVVGRExFVkJRVVVzUlVGQlJUdFpRVU51UXl4SlFVRkpMRU5CUVVNc1IwRkJSeXhMUVVGTExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTTdXVUZEYWtJc1NVRkJTU3hQUVVGUExGRkJRVkVzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRXRCUVVzc1YwRkJWenRuUWtGQlJTeFJRVUZSTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhIUVVGSExFVkJRVVVzUTBGQlF6dFpRVU0zUkN4UlFVRlJMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1IwRkJSeXhKUVVGSkxFTkJRVU03VTBGRE4wSTdVVUZEUkN4UFFVRlBMRlZCUVVNc1EwRkJReXhGUVVGRkxFTkJRVU1zU1VGQlN5eFBRVUZCTEU5QlFVOHNVVUZCVVN4RFFVRkRMRU5CUVVNc1EwRkJReXhMUVVGTExGZEJRVmNzU1VGQlNTeFJRVUZSTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFVkJRWEJFTEVOQlFXOUVMRU5CUVVNN1NVRkRNVVVzUTBGQlF6dEpRVWxOTEhGQ1FVRlZMRWRCUVdwQ0xGVkJRV3RDTEV0QlFVczdVVUZEYmtJc1NVRkJTU3hUUVVGVExFZEJRVWNzUlVGQlJTeERRVUZETzFGQlEyNUNMRXRCUVVzc1NVRkJTU3hEUVVGRExFZEJRVWNzUTBGQlF5eEZRVUZGTEVOQlFVTXNSMEZCUnl4TFFVRkxMRU5CUVVNc1RVRkJUU3hIUVVGSExFTkJRVU1zUlVGQlJTeERRVUZETEVWQlFVVXNSVUZCUlR0WlFVTjJReXhMUVVGTExFbEJRVWtzUTBGQlF5eEhRVUZITEVOQlFVTXNSMEZCUnl4RFFVRkRMRVZCUVVVc1EwRkJReXhIUVVGSExFdEJRVXNzUTBGQlF5eE5RVUZOTEVWQlFVVXNRMEZCUXl4RlFVRkZMRVZCUVVVN1owSkJRM1pETEVsQlFVa3NRMEZCUXl4SFFVRkhMRXRCUVVzc1EwRkJReXhEUVVGRExFTkJRVU1zUlVGRFdpeERRVUZETEVkQlFVY3NTMEZCU3l4RFFVRkRMRU5CUVVNc1EwRkJReXhGUVVOYUxFZEJRVWNzUjBGQlJ5eEpRVUZKTEhkQ1FVRjNRaXhEUVVGRExFTkJRVU1zUlVGQlJTeERRVUZETEVOQlFVTXNRMEZCUXp0blFrRkROME1zU1VGQlNTeERRVUZETEVWQlFVVXNSVUZCUlN4RlFVRkZMRVZCUVVVc1EwRkJRenRuUWtGRFpDeEpRVUZKTEVkQlFVY3NRMEZCUXl4TlFVRk5MRXRCUVVzc1EwRkJRenR2UWtGRGFFSXNVMEZCVXp0blFrRkRZaXhKUVVGSkxFZEJRVWNzUTBGQlF5eFJRVUZSTEVWQlFVVTdiMEpCUjJRc1EwRkJReXhEUVVGRExFOUJRVThzUlVGQlJTeERRVUZETzI5Q1FVTmFMRU5CUVVNc1EwRkJReXhSUVVGUkxFZEJRVWNzU1VGQlNTeERRVUZETzI5Q1FVTnNRaXhIUVVGSExFZEJRVWNzU1VGQlNTeDNRa0ZCZDBJc1EwRkJReXhEUVVGRExFVkJRVVVzUTBGQlF5eERRVUZETEVOQlFVTTdhVUpCUXpWRE8yZENRVU5FTEVsQlFVa3NRMEZCUXl4SFFVRkhMRU5CUVVNc1JVRkJSU3hKUVVGSkxFTkJRVU1zU1VGQlNTeEhRVUZITEVOQlFVTXNSVUZCUlN4SlFVRkpMRU5CUVVNc1EwRkJRenR2UWtGRE5VSXNRMEZCUXl4SFFVRkhMRU5CUVVNc1JVRkJSU3hIUVVGSExFZEJRVWNzUTBGQlF5eE5RVUZOTEVsQlFVa3NRMEZCUXl4RFFVRkRMRTFCUVUwc1NVRkJTU3hIUVVGSExFTkJRVU1zUlVGQlJTeEhRVUZITEVkQlFVY3NRMEZCUXl4TlFVRk5MRWxCUVVrc1EwRkJReXhEUVVGRExFMUJRVTBzUTBGQlF5eEZRVUZGTzI5Q1FVVjBSU3hUUVVGVExFTkJRVU1zU1VGQlNTeERRVUZETEVWQlFVVXNRMEZCUXl4RlFVRkZMRU5CUVVNc1JVRkJSU3hEUVVGRExFVkJRVVVzUTBGQlF5eEZRVUZGTEVOQlFVTXNRMEZCUXp0dlFrRkRMMElzVTBGQlV6dHBRa0ZEV2p0blFrRkRSQ3hKUVVGSkxFZEJRVWNzUTBGQlF5eEZRVUZGTEVkQlFVY3NSMEZCUnl4RFFVRkRMRTFCUVUwc1NVRkJTU3hEUVVGRExFTkJRVU1zVFVGQlRTeEpRVUZKTEVkQlFVY3NRMEZCUXl4RlFVRkZMRWRCUVVjc1IwRkJSeXhEUVVGRExFMUJRVTBzU1VGQlNTeERRVUZETEVOQlFVTXNUVUZCVFN4RlFVRkZPMjlDUVUxd1JTeERRVUZETEVkQlFVY3NRMEZCUXl4RFFVRkRMRWRCUVVjc1EwRkJReXhGUVVGRkxFZEJRVWNzUTBGQlF5eERRVUZETEVOQlFVTTdiMEpCUTJ4Q0xFVkJRVVVzUjBGQlJ5eERRVUZETEVOQlFVTXNSMEZCUnl4RFFVRkRMRVZCUVVVc1IwRkJSeXhEUVVGRExFTkJRVU1zUTBGQlF6dHZRa0ZEYmtJc1JVRkJSU3hIUVVGSExFTkJRVU1zUTBGQlF5eEhRVUZITEVOQlFVTXNSVUZCUlN4SFFVRkhMRU5CUVVNc1EwRkJReXhEUVVGRE8ybENRVU4wUWp0eFFrRkJUVHR2UWtGRFNDeERRVUZETEVkQlFVY3NRMEZCUXl4RFFVRkRMRWRCUVVjc1EwRkJReXhGUVVGRkxFZEJRVWNzUjBGQlJ5eERRVUZETEUxQlFVMHNSMEZCUnl4RFFVRkRMRU5CUVVNc1EwRkJRenR2UWtGREwwSXNSVUZCUlN4SFFVRkhMRU5CUVVNc1EwRkJReXhIUVVGSExFTkJRVU1zUlVGQlJTeEhRVUZITEVkQlFVY3NRMEZCUXl4TlFVRk5MRU5CUVVNc1EwRkJRenR2UWtGRE5VSXNSVUZCUlN4SFFVRkhMRU5CUVVNc1EwRkJReXhIUVVGSExFTkJRVU1zUlVGQlJTeEhRVUZITEVkQlFVY3NRMEZCUXl4TlFVRk5MRU5CUVVNc1EwRkJRenRwUWtGREwwSTdaMEpCUTBRc1NVRkJTU3hWUVVGVkxFTkJRVU1zVFVGQlRTeERRVUZETEVOQlFVTXNSVUZCUlN4RlFVRkZMRVZCUVVVc1JVRkJSU3hEUVVGRExFVkJRVVU3YjBKQlF6bENMRk5CUVZNc1EwRkJReXhKUVVGSkxFTkJRVU1zUlVGQlJTeERRVUZETEVWQlFVVXNRMEZCUXl4RlFVRkZMRU5CUVVNc1JVRkJSU3hEUVVGRExFVkJRVVVzUTBGQlF5eERRVUZETzJsQ1FVTnNRenR4UWtGQlRUdHZRa0ZEU0N4VFFVRlRMRU5CUVVNc1NVRkJTU3hEUVVGRExFVkJRVVVzUTBGQlF5eEZRVUZGTEVOQlFVTXNSVUZCUlN4RFFVRkRMRVZCUVVVc1EwRkJReXhGUVVGRkxFTkJRVU1zUTBGQlF6dHBRa0ZEYkVNN1lVRkRTanRUUVVOS08xRkJSVVFzVDBGQlR5eFZRVUZWTEVOQlFVTXNVVUZCVVN4RFFVRkRMRk5CUVZNc1EwRkJReXhEUVVGRE8wbEJRekZETEVOQlFVTTdTVUZMVFN4MVFrRkJXU3hIUVVGdVFpeFZRVUZ2UWl4SlFVRmhPMUZCUXpkQ0xGTkJRVk1zVTBGQlV5eERRVUZETEVOQlFWRTdXVUZEZGtJc1QwRkJZeXhGUVVGRkxFTkJRVU1zUlVGQlJTeERRVUZETEVOQlFVTXNRMEZCUXl4RlFVRkZMRU5CUVVNc1JVRkJSU3hEUVVGRExFTkJRVU1zUTBGQlF5eEZRVUZGTEVOQlFVTTdVVUZEY2tNc1EwRkJRenRSUVVORUxFbEJRVWtzVlVGQlZTeEhRVUZITEZWQlFVTXNRMEZCUXl4RlFVRkZMRU5CUVVNc1JVRkJSU3hEUVVGRExFbEJRVXNzVDBGQlFTeEpRVUZKTEVOQlFVTXNSMEZCUnl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUjBGQlJ5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRWRCUVVjc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eEhRVUZITEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1IwRkJSeXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVkQlFVY3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhIUVVGSExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNSMEZCUnl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUjBGQlJ5eExRVUZMTEVWQlFYWkZMRU5CUVhWRkxFTkJRVU03VVVGRGRFY3NTVUZCU1N4UlFVRlJMRWRCUVVjc1JVRkJSU3hEUVVGRE8xRkJRMnhDTEVsQlFVa3NRMEZCUXl4SFFVRkhMRk5CUVZNc1EwRkJReXhKUVVGSkxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXp0UlFVTXpRaXhMUVVGTExFbEJRVWtzUTBGQlF5eEhRVUZITEVOQlFVTXNSVUZCUlN4RFFVRkRMRWRCUVVjc1NVRkJTU3hEUVVGRExFMUJRVTBzUlVGQlJTeERRVUZETEVWQlFVVXNSVUZCUlR0WlFVTnNReXhKUVVGSkxFTkJRVU1zUjBGQlJ5eFRRVUZUTEVOQlFVTXNTVUZCU1N4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFVkJRVVVzUTBGQlF5eEhRVUZITEVOQlFVTXNSMEZCUnl4SlFVRkpMRU5CUVVNc1RVRkJUU3hIUVVGSExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNTVUZCU1N4RFFVRkRMRU5CUVVNc1IwRkJSeXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNTVUZCU1N4RFFVRkRPMWxCUTNwRkxFbEJRVWtzUTBGQlF5eERRVUZETEVsQlFVa3NRMEZCUXl4VlFVRlZMRU5CUVVNc1EwRkJReXhGUVVGRkxFTkJRVU1zUlVGQlJTeERRVUZETEVOQlFVTXNSVUZCUlR0blFrRkROVUlzVVVGQlVTeERRVUZETEVsQlFVa3NRMEZCUXl4RFFVRkRMRU5CUVVNc1JVRkJSU3hEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETzJkQ1FVTjBRaXhEUVVGRExFZEJRVWNzUTBGQlF5eERRVUZETzJGQlExUTdVMEZEU2p0UlFVTkVMRTlCUVU4c1VVRkJVU3hEUVVGRE8wbEJRM0JDTEVOQlFVTTdTVUZKUkN3d1FrRkJTeXhIUVVGTUxGVkJRVTBzUTBGQlV5eEZRVUZGTEVOQlFWTTdVVUZCTVVJc2FVSkJORVJETzFGQk0wUkhMRWxCUVVrc1RVRkJUU3hIUVVGSExFbEJRVWtzUTBGQlF5eExRVUZMTEVOQlFWTXNRMEZCUXl4RFFVRkRMRVZCUVVVc1RVRkJUU3hIUVVGSExFbEJRVWtzUTBGQlF5eExRVUZMTEVOQlFWTXNRMEZCUXl4RFFVRkRMRU5CUVVNN1VVRkRia1VzU1VGQlNTeERRVUZETEZOQlFWTXNSMEZCUnl4SlFVRkpMRU5CUVVNc1owSkJRV2RDTEVOQlFVTXNUVUZCVFN4RlFVRkZMRTFCUVUwc1EwRkJReXhEUVVGRE8xRkJSWFpFTEVsQlFVa3NZMEZCWXl4SFFVRkhMRVZCUVVVc1EwRkJRenRSUVVONFFpeEpRVUZKTEVOQlFVTXNVMEZCVXl4RFFVRkRMRTlCUVU4c1EwRkJReXhWUVVGQkxFTkJRVU1zU1VGQlNTeFBRVUZCTEdOQlFXTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1JVRkJSU3hEUVVGRExFZEJRVWNzUTBGQlF5eEZRVUY0UWl4RFFVRjNRaXhEUVVGRExFTkJRVU03VVVGRGRFUXNTVUZCU1N4RFFVRkRMR0ZCUVdFc1IwRkJSeXhKUVVGSkxFTkJRVU1zUzBGQlN5eERRVUZETEUxQlFVMHNRMEZCUXl4VlFVRkJMRU5CUVVNN1dVRkRjRU1zU1VGQlNTeERRVUZETEVkQlFVY3NTMEZCU1N4RFFVRkRMRXRCUVVzc1EwRkJReXhEUVVGRExFTkJRVU1zVFVGQlRTeERRVUZETEVWQlEzaENMRU5CUVVNc1IwRkJSeXhMUVVGSkxFTkJRVU1zUzBGQlN5eERRVUZETEVOQlFVTXNRMEZCUXl4TlFVRk5MRU5CUVVNc1EwRkJRenRaUVVNM1FpeFBRVUZQTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1NVRkJTU3hKUVVGSkxFTkJRVU1zUTBGQlF5eEpRVUZKTEVOQlFVTXNSVUZCUlN4SlFVRkpMR05CUVdNN2JVSkJRM1pETEVOQlFVTXNRMEZCUXl4SlFVRkpMRWxCUVVrc1EwRkJReXhEUVVGRExFbEJRVWtzUTBGQlF5eEZRVUZGTEVsQlFVa3NZMEZCWXl4RFFVRkRMRU5CUVVNN1VVRkRiRVFzUTBGQlF5eERRVUZETEVOQlFVTTdVVUZIU0N4TFFVRkxMRWxCUVVrc1EwRkJReXhIUVVGSExFTkJRVU1zUlVGQlJTeERRVUZETEVkQlFVY3NUVUZCVFN4RFFVRkRMRXRCUVVzc1EwRkJReXhOUVVGTkxFVkJRVVVzUTBGQlF5eEZRVUZGTEVWQlFVVTdXVUZETVVNc1NVRkJTU3hEUVVGRExFZEJRVWNzVFVGQlRTeERRVUZETEV0QlFVc3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhGUVVGRkxFTkJRVU03V1VGRE0wSXNTVUZCU1N4RFFVRkRMRWRCUVVjc1RVRkJUU3hEUVVGRExFdEJRVXNzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RlFVRkZMRU5CUVVNN1dVRkRNMElzU1VGQlNTeERRVUZETEdGQlFXRXNRMEZCUXl4SlFVRkpMRU5CUVVNN1owSkJRM0JDTEUxQlFVMHNSVUZCUlN4RFFVRkRPMmRDUVVOVUxFMUJRVTBzUlVGQlJTeERRVUZETzJkQ1FVTlVMRTFCUVUwc1JVRkJSU3hEUVVGRE8yRkJRMW9zUTBGQlF5eERRVUZETzFOQlEwNDdVVUZEUkN4TFFVRkxMRWxCUVVrc1EwRkJReXhIUVVGSExFTkJRVU1zUlVGQlJTeERRVUZETEVkQlFVY3NUVUZCVFN4RFFVRkRMRXRCUVVzc1EwRkJReXhOUVVGTkxFVkJRVVVzUTBGQlF5eEZRVUZGTEVWQlFVVTdXVUZETVVNc1NVRkJTU3hEUVVGRExFZEJRVWNzVFVGQlRTeERRVUZETEV0QlFVc3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhGUVVGRkxFTkJRVU03V1VGRE0wSXNTVUZCU1N4RFFVRkRMRWRCUVVjc1RVRkJUU3hEUVVGRExFdEJRVXNzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RlFVRkZMRU5CUVVNN1dVRkRNMElzU1VGQlNTeERRVUZETEdGQlFXRXNRMEZCUXl4SlFVRkpMRU5CUVVNN1owSkJRM0JDTEUxQlFVMHNSVUZCUlN4RFFVRkRPMmRDUVVOVUxFMUJRVTBzUlVGQlJTeERRVUZETzJkQ1FVTlVMRTFCUVUwc1JVRkJSU3hEUVVGRE8yRkJRMW9zUTBGQlF5eERRVUZETzFOQlEwNDdVVUZGUkN4SlFVRkpMRk5CUVZNc1IwRkJSeXhWUVVGQkxFTkJRVU1zU1VGQlJ5eFBRVUZCTEVOQlFVTXNRMEZCUXl4TlFVRk5MRVZCUVZJc1EwRkJVU3hGUVVONFFpeFRRVUZUTEVkQlFVY3NWVUZCUVN4RFFVRkRMRWxCUVVjc1QwRkJRU3hEUVVGRExFTkJRVU1zVFVGQlRTeEZRVUZTTEVOQlFWRXNSVUZEZUVJc1UwRkJVeXhIUVVGSExGVkJRVUVzUTBGQlF5eEpRVUZITEU5QlFVRXNRMEZCUXl4RFFVRkRMRTFCUVUwc1JVRkJVaXhEUVVGUkxFTkJRVU03VVVGRk4wSXNTVUZCU1N4elFrRkJjMElzUjBGQlJ5eEpRVUZKTERCQ1FVRlZMRU5CUVVNc1NVRkJTU3hEUVVGRExFdEJRVXNzUTBGQlF5eE5RVUZOTEVWQlFVVXNTVUZCU1N4RFFVRkRMR0ZCUVdFc1JVRkJSU3hUUVVGVExFVkJRVVVzVTBGQlV5eEZRVUZGTEZOQlFWTXNRMEZCUXl4RFFVRkRPMUZCUTNCSUxFbEJRVWtzVjBGQlZ5eEhRVUZITEZWQlFVTXNRMEZCUXl4RlFVRkZMRU5CUVVNc1JVRkJSU3hEUVVGRE8xbEJRM1JDTEVsQlFVa3NRMEZCUXl4SFFVRkhMRXRCUVVrc1EwRkJReXhMUVVGTExFTkJRVU1zUTBGQlF5eERRVUZETEVWQlFVVXNRMEZCUXl4SFFVRkhMRXRCUVVrc1EwRkJReXhMUVVGTExFTkJRVU1zUTBGQlF5eERRVUZETEVWQlFVVXNRMEZCUXl4SFFVRkhMRXRCUVVrc1EwRkJReXhMUVVGTExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTTdXVUZETlVRc1NVRkJTU3hGUVVGRkxFZEJRVWNzU1VGQlNTeERRVUZETEVkQlFVY3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhIUVVGSExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNSVUZCUlN4RlFVRkZMRWRCUVVjc1NVRkJTU3hEUVVGRExFZEJRVWNzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4SFFVRkhMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dFpRVVYyUkN4SlFVRkpMRU5CUVVNc1EwRkJReXhKUVVGSkxFdEJRVXNzVFVGQlRTeEpRVUZKTEVOQlFVTXNRMEZCUXl4SlFVRkpMRXRCUVVzc1EwRkJReXhEUVVGRExFbEJRVWtzU1VGQlNTeERRVUZETEVOQlFVTXNTVUZCU1N4TFFVRkxMRTFCUVUwc1NVRkJTU3hEUVVGRExFTkJRVU1zU1VGQlNTeExRVUZMTEVOQlFVTXNRMEZCUXl4SlFVRkpPMmRDUVVOb1JpeFBRVUZQTEVOQlFVTXNRMEZCUXp0WlFVTmlMRTlCUVU4c1JVRkJSU3hIUVVGSExFTkJRVU1zU1VGQlNTeEZRVUZGTEVkQlFVY3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhKUVVGSkxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXp0UlFVTjJReXhEUVVGRExFTkJRVU03VVVGSFJpeEpRVUZKTEZsQlFWa3NSMEZCUnl4elFrRkJjMElzUTBGQlF5dzRRa0ZCT0VJc1EwRkRjRVVzVFVGQlRTeERRVUZETEV0QlFVc3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhGUVVGRkxFVkJRVVVzVFVGQlRTeERRVUZETEV0QlFVc3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhGUVVGRkxFVkJRM1JETEZkQlFWY3NRMEZCUXl4RFFVRkRPMUZCUjJwQ0xFbEJRVWtzVlVGQlZTeEhRVUZITEZsQlFWa3NRMEZCUXl4UFFVRlBMRVZCUVVVc1EwRkJReXhIUVVGSExFTkJRVU1zVlVGQlFTeEZRVUZGTEVsQlFVa3NUMEZCUVN4TFFVRkpMRU5CUVVNc1MwRkJTeXhEUVVGRExFVkJRVVVzUTBGQlF5eEZRVUZrTEVOQlFXTXNRMEZCUXl4RFFVRkRPMUZCUTJ4RkxGVkJRVlVzUTBGQlF5eEpRVUZKTEVOQlFVTXNTVUZCU1N4RFFVRkRMRXRCUVVzc1EwRkJReXhOUVVGTkxFTkJRVU1zUlVGQlJTeERRVUZETEVOQlFVTXNTMEZCU3l4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU03VVVGSGFFUXNUMEZCVHl4VlFVRlZMRU5CUVVNc1RVRkJUU3hEUVVGRExGVkJRVU1zUTBGQlF5eEZRVUZGTEVOQlFVTTdXVUZETVVJc1QwRkJRU3hEUVVGRExFTkJRVU1zUTBGQlF5eEhRVUZITEZWQlFWVXNRMEZCUXl4TlFVRk5MRWRCUVVjc1EwRkJReXhKUVVGSkxGVkJRVlVzUTBGQlF5eERRVUZETEVkQlFVY3NRMEZCUXl4RFFVRkRMRU5CUVVNc1NVRkJTU3hMUVVGTExFMUJRVTBzU1VGQlNTeERRVUZETEVOQlFVTXNTVUZCU1N4TFFVRkxMRTFCUVUwN2JVSkJRemxGTEVOQlFVTXNSMEZCUnl4RFFVRkRMRWxCUVVrc1EwRkJReXhEUVVGRExFbEJRVWtzUzBGQlN5eE5RVUZOTEVsQlFVa3NWVUZCVlN4RFFVRkRMRU5CUVVNc1IwRkJSeXhEUVVGRExFTkJRVU1zUTBGQlF5eEpRVUZKTEV0QlFVc3NUVUZCVFN4RFFVRkRPMUZCUkhaRkxFTkJRM1ZGTEVOQlFVTXNRMEZCUXp0SlFVTnFSaXhEUVVGRE8wbEJSVTBzZFVKQlFWa3NSMEZCYmtJc1ZVRkJiMElzUzBGQlowSXNSVUZCUlN4WlFVRnZRaXhGUVVGRkxGVkJRV3RDTEVWQlFVVXNWMEZCYlVJN1VVRkRMMFlzU1VGQlNTeE5RVUZOTEVkQlFVYzdXVUZEVkN4VFFVRlRMRVZCUVVVc1NVRkJTU3hIUVVGSExFdEJRVXNzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFZEJRVWNzUjBGQlJ5eEhRVUZITEV0QlFVc3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVkQlFVY3NSMEZCUnp0WlFVTXpSQ3hUUVVGVExFVkJRVVVzUlVGQlJUdFRRVU5vUWl4RFFVRkRPMUZCUTBZc1NVRkJTU3hMUVVGTExFTkJRVU1zVFVGQlRTeEhRVUZITEVOQlFVTXNSVUZCUlR0WlFVTnNRaXhMUVVGTExFbEJRVWtzUTBGQlF5eEhRVUZITEVOQlFVTXNSVUZCUlN4RFFVRkRMRWRCUVVjc1MwRkJTeXhEUVVGRExFMUJRVTBzUlVGQlJTeERRVUZETEVWQlFVVXNSVUZCUlR0blFrRkRia01zU1VGQlNTeEZRVUZGTEVkQlFVY3NTMEZCU3l4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRE8yZENRVU5zUWl4SlFVRkpMRU5CUVVNc1IwRkJSeXhGUVVGRkxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RlFVRkZMRU5CUVVNc1IwRkJSeXhGUVVGRkxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRPMmRDUVVNM1FpeEpRVUZKTEVWQlFVVXNSMEZCUnl4RFFVRkRMRWRCUVVjc1JVRkJSU3hEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXp0blFrRkRja0lzU1VGQlNTeEZRVUZGTEVkQlFVY3NRMEZCUXl4SFFVRkhMRVZCUVVVc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTTdaMEpCUTNKQ0xFbEJRVWtzUTBGQlF5eEhRVUZITEV0QlFVc3NRMEZCUXl4TlFVRk5MRWRCUVVjc1EwRkJReXhGUVVGRk8yOUNRVU4wUWl4SlFVRkpMRWxCUVVrc1EwRkJReXhIUVVGSExFTkJRVU1zUlVGQlJTeERRVUZETEVkQlFVY3NRMEZCUXl4RlFVRkZPM2RDUVVOc1FpeERRVUZETEVsQlFVa3NSVUZCUlN4SFFVRkhMRWxCUVVrc1EwRkJReXhIUVVGSExFTkJRVU1zUlVGQlJTeERRVUZETEVkQlFVY3NXVUZCV1N4RFFVRkRPM0ZDUVVONlF6dDVRa0ZCVFR0M1FrRkRTQ3hEUVVGRExFbEJRVWtzUlVGQlJTeEhRVUZITEVsQlFVa3NRMEZCUXl4SFFVRkhMRU5CUVVNc1JVRkJSU3hEUVVGRExFZEJRVWNzV1VGQldTeERRVUZETzNGQ1FVTjZRenR2UWtGRFJDeE5RVUZOTEVOQlFVTXNVMEZCVXl4SlFVRkpMRWxCUVVrc1IwRkJSeXhEUVVGRExFZEJRVWNzUjBGQlJ5eEhRVUZITEVOQlFVTXNSMEZCUnl4SFFVRkhMRU5CUVVNN2IwSkJRemRETEVsQlFVa3NRMEZCUXl4SFFVRkhMRXRCUVVzc1EwRkJReXhEUVVGRExFZEJRVWNzUTBGQlF5eERRVUZETEVOQlFVTTdiMEpCUTNKQ0xFbEJRVWtzUlVGQlJTeEhRVUZITEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFVkJRVVVzUlVGQlJTeEhRVUZITEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU03YjBKQlF6ZENMRWxCUVVrc1JVRkJSU3hIUVVGSExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNN2IwSkJRMmhDTEVsQlFVa3NSVUZCUlN4SFFVRkhMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTTdiMEpCUTJoQ0xFVkJRVVVzUjBGQlJ5eEZRVUZGTEVkQlFVY3NSVUZCUlN4RFFVRkRPMjlDUVVOaUxFVkJRVVVzUjBGQlJ5eEZRVUZGTEVkQlFVY3NSVUZCUlN4RFFVRkRPMjlDUVVOaUxFbEJRVWtzUzBGQlN5eEhRVUZITEZWQlFWVXNRMEZCUXl4clFrRkJhMElzUTBGQlF5eEZRVUZGTEVWQlFVVXNRMEZCUXl4RFFVRkRMRWRCUVVjc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJRenR2UWtGRk4wUXNTVUZCU1N4RlFVRkZMRVZCUVVVc1JVRkJSU3hEUVVGRE8yOUNRVU5ZTEVsQlFVa3NTVUZCU1N4RFFVRkRMRWRCUVVjc1EwRkJReXhGUVVGRkxFTkJRVU1zUjBGQlJ5eERRVUZETEVWQlFVVTdkMEpCUTJ4Q0xFVkJRVVVzUjBGQlJ5eEZRVUZGTEVkQlFVY3NSVUZCUlN4SFFVRkhMRWxCUVVrc1EwRkJReXhIUVVGSExFTkJRVU1zUlVGQlJTeERRVUZETEVkQlFVY3NXVUZCV1N4RFFVRkRPM2RDUVVNelF5eEZRVUZGTEVkQlFVY3NSVUZCUlN4RFFVRkRPM0ZDUVVOWU8zbENRVUZOTzNkQ1FVTklMRVZCUVVVc1IwRkJSeXhGUVVGRkxFTkJRVU03ZDBKQlExSXNSVUZCUlN4SFFVRkhMRVZCUVVVc1IwRkJSeXhGUVVGRkxFZEJRVWNzU1VGQlNTeERRVUZETEVkQlFVY3NRMEZCUXl4RlFVRkZMRU5CUVVNc1IwRkJSeXhaUVVGWkxFTkJRVU03Y1VKQlF6bERPMjlDUVVORUxFbEJRVWtzUlVGQlJTeEhRVUZITEVsQlFVa3NRMEZCUXl4SFFVRkhMRU5CUVVNc1JVRkJSU3hIUVVGSExFTkJRVU1zUTBGQlF5eERRVUZETzI5Q1FVTXhRaXhKUVVGSkxFVkJRVVVzUjBGQlJ5eEpRVUZKTEVOQlFVTXNSMEZCUnl4RFFVRkRMRVZCUVVVc1IwRkJSeXhEUVVGRExFTkJRVU1zUTBGQlF6dHZRa0ZETVVJc1RVRkJUU3hEUVVGRExGTkJRVk1zU1VGQlNTeEpRVUZKTEVkQlFVY3NSVUZCUlN4SFFVRkhMRWRCUVVjc1IwRkJSeXhGUVVGRkxFZEJRVWNzVDBGQlR5eEhRVUZITEV0QlFVc3NSMEZCUnl4SFFVRkhMRWRCUVVjc1JVRkJSU3hIUVVGSExFZEJRVWNzUjBGQlJ5eEZRVUZGTEVkQlFVY3NSMEZCUnl4RFFVRkRPMmxDUVVNeFJqdHhRa0ZCVFR0dlFrRkRTQ3hKUVVGSkxGRkJRVkVzUjBGQlJ5eERRVUZETEVOQlFVTXNSVUZCUlN4RFFVRkRMRU5CUVVNc1EwRkJRenR2UWtGRGRFSXNTVUZCU1N4WlFVRlpMRVZCUVVVc1dVRkJXU3hEUVVGRE8yOUNRVU12UWl4SlFVRkpMRWxCUVVrc1EwRkJReXhIUVVGSExFTkJRVU1zUlVGQlJTeERRVUZETEVkQlFVY3NRMEZCUXl4RlFVRkZPM2RDUVVOc1FpeERRVUZETEVsQlFVa3NSVUZCUlN4SFFVRkhMRWxCUVVrc1EwRkJReXhIUVVGSExFTkJRVU1zUlVGQlJTeERRVUZETEVkQlFVY3NWMEZCVnl4RFFVRkRPM2RDUVVOeVF5eFpRVUZaTEVkQlFVY3NRMEZCUXl4RFFVRkRMRVZCUVVVc1EwRkJReXhIUVVGSExGVkJRVlVzUTBGQlF5eERRVUZETzNkQ1FVTnVReXhaUVVGWkxFZEJRVWNzUTBGQlF5eERRVUZETEVWQlFVVXNRMEZCUXl4SFFVRkhMRlZCUVZVc1EwRkJReXhEUVVGRE8zRkNRVU4wUXp0NVFrRkJUVHQzUWtGRFNDeERRVUZETEVsQlFVa3NSVUZCUlN4SFFVRkhMRWxCUVVrc1EwRkJReXhIUVVGSExFTkJRVU1zUlVGQlJTeERRVUZETEVkQlFVY3NWMEZCVnl4RFFVRkRPM2RDUVVOeVF5eFpRVUZaTEVkQlFVY3NRMEZCUXl4RFFVRkRMRWRCUVVjc1ZVRkJWU3hGUVVGRkxFTkJRVU1zUTBGQlF5eERRVUZETzNkQ1FVTnVReXhaUVVGWkxFZEJRVWNzUTBGQlF5eERRVUZETEVkQlFVY3NWVUZCVlN4RlFVRkZMRU5CUVVNc1EwRkJReXhEUVVGRE8zRkNRVU4wUXp0dlFrRkRSQ3hOUVVGTkxFTkJRVU1zVTBGQlV5eEpRVUZKTEVsQlFVa3NSMEZCUnl4RFFVRkRMRWRCUVVjc1IwRkJSeXhIUVVGSExFTkJRVU1zUjBGQlJ5eEhRVUZITEVOQlFVTTdiMEpCUXpkRExFbEJRVWtzVjBGQlZ5eEhRVUZITEVOQlFVTXNSVUZCUlR0M1FrRkRha0lzVFVGQlRTeERRVUZETEZOQlFWTXNSMEZCUnl4SlFVRkpMRWRCUVVjc1VVRkJVU3hEUVVGRExFTkJRVU1zUTBGQlF5eEhRVUZITEVkQlFVY3NSMEZCUnl4UlFVRlJMRU5CUVVNc1EwRkJReXhEUVVGRExFZEJRVWNzUzBGQlN5eEhRVUZITEZsQlFWa3NRMEZCUXl4RFFVRkRMRU5CUVVNc1IwRkJSeXhIUVVGSExFZEJRVWNzV1VGQldTeERRVUZETEVOQlFVTXNRMEZCUXpzNFFrRkRla2NzUzBGQlN5eEhRVUZITEZsQlFWa3NRMEZCUXl4RFFVRkRMRU5CUVVNc1IwRkJSeXhIUVVGSExFZEJRVWNzV1VGQldTeERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRPM0ZDUVVOeVJEdHBRa0ZEU2p0aFFVTktPMU5CUTBvN1lVRkJUVHRaUVVOSUxFbEJRVWtzUlVGQlJTeEhRVUZITEV0QlFVc3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJRenRaUVVOc1FpeEpRVUZKTEVOQlFVTXNSMEZCUnl4RlFVRkZMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eEZRVUZGTEVOQlFVTXNSMEZCUnl4RlFVRkZMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETzFsQlF6ZENMRWxCUVVrc1JVRkJSU3hIUVVGSExFTkJRVU1zUjBGQlJ5eEZRVUZGTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRE8xbEJRM0pDTEVsQlFVa3NSVUZCUlN4SFFVRkhMRU5CUVVNc1IwRkJSeXhGUVVGRkxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRPMWxCUTNKQ0xFbEJRVWtzVVVGQlVTeEhRVUZITEVOQlFVTXNRMEZCUXl4RlFVRkZMRU5CUVVNc1EwRkJReXhEUVVGRE8xbEJRM1JDTEVsQlFVa3NXVUZCV1N4RlFVRkZMRmxCUVZrc1EwRkJRenRaUVVNdlFpeEpRVUZKTEVsQlFVa3NRMEZCUXl4SFFVRkhMRU5CUVVNc1JVRkJSU3hEUVVGRExFZEJRVWNzUTBGQlF5eEZRVUZGTzJkQ1FVTnNRaXhEUVVGRExFbEJRVWtzUlVGQlJTeEhRVUZITEVsQlFVa3NRMEZCUXl4SFFVRkhMRU5CUVVNc1JVRkJSU3hEUVVGRExFZEJRVWNzVjBGQlZ5eERRVUZETzJkQ1FVTnlReXhaUVVGWkxFZEJRVWNzUTBGQlF5eERRVUZETEVWQlFVVXNRMEZCUXl4SFFVRkhMRlZCUVZVc1EwRkJReXhEUVVGRE8yZENRVU51UXl4WlFVRlpMRWRCUVVjc1EwRkJReXhEUVVGRExFVkJRVVVzUTBGQlF5eEhRVUZITEZWQlFWVXNRMEZCUXl4RFFVRkRPMkZCUTNSRE8ybENRVUZOTzJkQ1FVTklMRU5CUVVNc1NVRkJTU3hGUVVGRkxFZEJRVWNzU1VGQlNTeERRVUZETEVkQlFVY3NRMEZCUXl4RlFVRkZMRU5CUVVNc1IwRkJSeXhYUVVGWExFTkJRVU03WjBKQlEzSkRMRmxCUVZrc1IwRkJSeXhEUVVGRExFTkJRVU1zUjBGQlJ5eFZRVUZWTEVWQlFVVXNRMEZCUXl4RFFVRkRMRU5CUVVNN1owSkJRMjVETEZsQlFWa3NSMEZCUnl4RFFVRkRMRU5CUVVNc1IwRkJSeXhWUVVGVkxFVkJRVVVzUTBGQlF5eERRVUZETEVOQlFVTTdZVUZEZEVNN1dVRkRSQ3hOUVVGTkxFTkJRVU1zVTBGQlV5eEpRVUZKTEVsQlFVa3NSMEZCUnl4RFFVRkRMRWRCUVVjc1IwRkJSeXhIUVVGSExFTkJRVU1zUjBGQlJ5eEhRVUZITEVOQlFVTTdXVUZETjBNc1NVRkJTU3hYUVVGWExFZEJRVWNzUTBGQlF5eEZRVUZGTzJkQ1FVTnFRaXhOUVVGTkxFTkJRVU1zVTBGQlV5eEhRVUZITEVsQlFVa3NSMEZCUnl4UlFVRlJMRU5CUVVNc1EwRkJReXhEUVVGRExFZEJRVWNzUjBGQlJ5eEhRVUZITEZGQlFWRXNRMEZCUXl4RFFVRkRMRU5CUVVNc1IwRkJSeXhMUVVGTExFZEJRVWNzV1VGQldTeERRVUZETEVOQlFVTXNRMEZCUXl4SFFVRkhMRWRCUVVjc1IwRkJSeXhaUVVGWkxFTkJRVU1zUTBGQlF5eERRVUZETzNOQ1FVTjZSeXhMUVVGTExFZEJRVWNzV1VGQldTeERRVUZETEVOQlFVTXNRMEZCUXl4SFFVRkhMRWRCUVVjc1IwRkJSeXhaUVVGWkxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTTdZVUZEY2tRN1UwRkRTanRSUVVORUxFOUJRVThzVFVGQlRTeERRVUZETzBsQlEyeENMRU5CUVVNN1NVRkRUQ3hwUWtGQlF6dEJRVUZFTEVOQlFVTXNRVUY2YkVKRUxFbEJlV3hDUXp0QlFYcHNRbGtzWjBOQlFWVWlmUT09IiwiXCJ1c2Ugc3RyaWN0XCI7XHJcbk9iamVjdC5kZWZpbmVQcm9wZXJ0eShleHBvcnRzLCBcIl9fZXNNb2R1bGVcIiwgeyB2YWx1ZTogdHJ1ZSB9KTtcclxudmFyIHBhY2tpbmdPcHRpb25zID0ge1xyXG4gICAgUEFERElORzogMTAsXHJcbiAgICBHT0xERU5fU0VDVElPTjogKDEgKyBNYXRoLnNxcnQoNSkpIC8gMixcclxuICAgIEZMT0FUX0VQU0lMT046IDAuMDAwMSxcclxuICAgIE1BWF9JTkVSQVRJT05TOiAxMDBcclxufTtcclxuZnVuY3Rpb24gYXBwbHlQYWNraW5nKGdyYXBocywgdywgaCwgbm9kZV9zaXplLCBkZXNpcmVkX3JhdGlvLCBjZW50ZXJHcmFwaCkge1xyXG4gICAgaWYgKGRlc2lyZWRfcmF0aW8gPT09IHZvaWQgMCkgeyBkZXNpcmVkX3JhdGlvID0gMTsgfVxyXG4gICAgaWYgKGNlbnRlckdyYXBoID09PSB2b2lkIDApIHsgY2VudGVyR3JhcGggPSB0cnVlOyB9XHJcbiAgICB2YXIgaW5pdF94ID0gMCwgaW5pdF95ID0gMCwgc3ZnX3dpZHRoID0gdywgc3ZnX2hlaWdodCA9IGgsIGRlc2lyZWRfcmF0aW8gPSB0eXBlb2YgZGVzaXJlZF9yYXRpbyAhPT0gJ3VuZGVmaW5lZCcgPyBkZXNpcmVkX3JhdGlvIDogMSwgbm9kZV9zaXplID0gdHlwZW9mIG5vZGVfc2l6ZSAhPT0gJ3VuZGVmaW5lZCcgPyBub2RlX3NpemUgOiAwLCByZWFsX3dpZHRoID0gMCwgcmVhbF9oZWlnaHQgPSAwLCBtaW5fd2lkdGggPSAwLCBnbG9iYWxfYm90dG9tID0gMCwgbGluZSA9IFtdO1xyXG4gICAgaWYgKGdyYXBocy5sZW5ndGggPT0gMClcclxuICAgICAgICByZXR1cm47XHJcbiAgICBjYWxjdWxhdGVfYmIoZ3JhcGhzKTtcclxuICAgIGFwcGx5KGdyYXBocywgZGVzaXJlZF9yYXRpbyk7XHJcbiAgICBpZiAoY2VudGVyR3JhcGgpIHtcclxuICAgICAgICBwdXRfbm9kZXNfdG9fcmlnaHRfcG9zaXRpb25zKGdyYXBocyk7XHJcbiAgICB9XHJcbiAgICBmdW5jdGlvbiBjYWxjdWxhdGVfYmIoZ3JhcGhzKSB7XHJcbiAgICAgICAgZ3JhcGhzLmZvckVhY2goZnVuY3Rpb24gKGcpIHtcclxuICAgICAgICAgICAgY2FsY3VsYXRlX3NpbmdsZV9iYihnKTtcclxuICAgICAgICB9KTtcclxuICAgICAgICBmdW5jdGlvbiBjYWxjdWxhdGVfc2luZ2xlX2JiKGdyYXBoKSB7XHJcbiAgICAgICAgICAgIHZhciBtaW5feCA9IE51bWJlci5NQVhfVkFMVUUsIG1pbl95ID0gTnVtYmVyLk1BWF9WQUxVRSwgbWF4X3ggPSAwLCBtYXhfeSA9IDA7XHJcbiAgICAgICAgICAgIGdyYXBoLmFycmF5LmZvckVhY2goZnVuY3Rpb24gKHYpIHtcclxuICAgICAgICAgICAgICAgIHZhciB3ID0gdHlwZW9mIHYud2lkdGggIT09ICd1bmRlZmluZWQnID8gdi53aWR0aCA6IG5vZGVfc2l6ZTtcclxuICAgICAgICAgICAgICAgIHZhciBoID0gdHlwZW9mIHYuaGVpZ2h0ICE9PSAndW5kZWZpbmVkJyA/IHYuaGVpZ2h0IDogbm9kZV9zaXplO1xyXG4gICAgICAgICAgICAgICAgdyAvPSAyO1xyXG4gICAgICAgICAgICAgICAgaCAvPSAyO1xyXG4gICAgICAgICAgICAgICAgbWF4X3ggPSBNYXRoLm1heCh2LnggKyB3LCBtYXhfeCk7XHJcbiAgICAgICAgICAgICAgICBtaW5feCA9IE1hdGgubWluKHYueCAtIHcsIG1pbl94KTtcclxuICAgICAgICAgICAgICAgIG1heF95ID0gTWF0aC5tYXgodi55ICsgaCwgbWF4X3kpO1xyXG4gICAgICAgICAgICAgICAgbWluX3kgPSBNYXRoLm1pbih2LnkgLSBoLCBtaW5feSk7XHJcbiAgICAgICAgICAgIH0pO1xyXG4gICAgICAgICAgICBncmFwaC53aWR0aCA9IG1heF94IC0gbWluX3g7XHJcbiAgICAgICAgICAgIGdyYXBoLmhlaWdodCA9IG1heF95IC0gbWluX3k7XHJcbiAgICAgICAgfVxyXG4gICAgfVxyXG4gICAgZnVuY3Rpb24gcHV0X25vZGVzX3RvX3JpZ2h0X3Bvc2l0aW9ucyhncmFwaHMpIHtcclxuICAgICAgICBncmFwaHMuZm9yRWFjaChmdW5jdGlvbiAoZykge1xyXG4gICAgICAgICAgICB2YXIgY2VudGVyID0geyB4OiAwLCB5OiAwIH07XHJcbiAgICAgICAgICAgIGcuYXJyYXkuZm9yRWFjaChmdW5jdGlvbiAobm9kZSkge1xyXG4gICAgICAgICAgICAgICAgY2VudGVyLnggKz0gbm9kZS54O1xyXG4gICAgICAgICAgICAgICAgY2VudGVyLnkgKz0gbm9kZS55O1xyXG4gICAgICAgICAgICB9KTtcclxuICAgICAgICAgICAgY2VudGVyLnggLz0gZy5hcnJheS5sZW5ndGg7XHJcbiAgICAgICAgICAgIGNlbnRlci55IC89IGcuYXJyYXkubGVuZ3RoO1xyXG4gICAgICAgICAgICB2YXIgY29ybmVyID0geyB4OiBjZW50ZXIueCAtIGcud2lkdGggLyAyLCB5OiBjZW50ZXIueSAtIGcuaGVpZ2h0IC8gMiB9O1xyXG4gICAgICAgICAgICB2YXIgb2Zmc2V0ID0geyB4OiBnLnggLSBjb3JuZXIueCArIHN2Z193aWR0aCAvIDIgLSByZWFsX3dpZHRoIC8gMiwgeTogZy55IC0gY29ybmVyLnkgKyBzdmdfaGVpZ2h0IC8gMiAtIHJlYWxfaGVpZ2h0IC8gMiB9O1xyXG4gICAgICAgICAgICBnLmFycmF5LmZvckVhY2goZnVuY3Rpb24gKG5vZGUpIHtcclxuICAgICAgICAgICAgICAgIG5vZGUueCArPSBvZmZzZXQueDtcclxuICAgICAgICAgICAgICAgIG5vZGUueSArPSBvZmZzZXQueTtcclxuICAgICAgICAgICAgfSk7XHJcbiAgICAgICAgfSk7XHJcbiAgICB9XHJcbiAgICBmdW5jdGlvbiBhcHBseShkYXRhLCBkZXNpcmVkX3JhdGlvKSB7XHJcbiAgICAgICAgdmFyIGN1cnJfYmVzdF9mID0gTnVtYmVyLlBPU0lUSVZFX0lORklOSVRZO1xyXG4gICAgICAgIHZhciBjdXJyX2Jlc3QgPSAwO1xyXG4gICAgICAgIGRhdGEuc29ydChmdW5jdGlvbiAoYSwgYikgeyByZXR1cm4gYi5oZWlnaHQgLSBhLmhlaWdodDsgfSk7XHJcbiAgICAgICAgbWluX3dpZHRoID0gZGF0YS5yZWR1Y2UoZnVuY3Rpb24gKGEsIGIpIHtcclxuICAgICAgICAgICAgcmV0dXJuIGEud2lkdGggPCBiLndpZHRoID8gYS53aWR0aCA6IGIud2lkdGg7XHJcbiAgICAgICAgfSk7XHJcbiAgICAgICAgdmFyIGxlZnQgPSB4MSA9IG1pbl93aWR0aDtcclxuICAgICAgICB2YXIgcmlnaHQgPSB4MiA9IGdldF9lbnRpcmVfd2lkdGgoZGF0YSk7XHJcbiAgICAgICAgdmFyIGl0ZXJhdGlvbkNvdW50ZXIgPSAwO1xyXG4gICAgICAgIHZhciBmX3gxID0gTnVtYmVyLk1BWF9WQUxVRTtcclxuICAgICAgICB2YXIgZl94MiA9IE51bWJlci5NQVhfVkFMVUU7XHJcbiAgICAgICAgdmFyIGZsYWcgPSAtMTtcclxuICAgICAgICB2YXIgZHggPSBOdW1iZXIuTUFYX1ZBTFVFO1xyXG4gICAgICAgIHZhciBkZiA9IE51bWJlci5NQVhfVkFMVUU7XHJcbiAgICAgICAgd2hpbGUgKChkeCA+IG1pbl93aWR0aCkgfHwgZGYgPiBwYWNraW5nT3B0aW9ucy5GTE9BVF9FUFNJTE9OKSB7XHJcbiAgICAgICAgICAgIGlmIChmbGFnICE9IDEpIHtcclxuICAgICAgICAgICAgICAgIHZhciB4MSA9IHJpZ2h0IC0gKHJpZ2h0IC0gbGVmdCkgLyBwYWNraW5nT3B0aW9ucy5HT0xERU5fU0VDVElPTjtcclxuICAgICAgICAgICAgICAgIHZhciBmX3gxID0gc3RlcChkYXRhLCB4MSk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgaWYgKGZsYWcgIT0gMCkge1xyXG4gICAgICAgICAgICAgICAgdmFyIHgyID0gbGVmdCArIChyaWdodCAtIGxlZnQpIC8gcGFja2luZ09wdGlvbnMuR09MREVOX1NFQ1RJT047XHJcbiAgICAgICAgICAgICAgICB2YXIgZl94MiA9IHN0ZXAoZGF0YSwgeDIpO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIGR4ID0gTWF0aC5hYnMoeDEgLSB4Mik7XHJcbiAgICAgICAgICAgIGRmID0gTWF0aC5hYnMoZl94MSAtIGZfeDIpO1xyXG4gICAgICAgICAgICBpZiAoZl94MSA8IGN1cnJfYmVzdF9mKSB7XHJcbiAgICAgICAgICAgICAgICBjdXJyX2Jlc3RfZiA9IGZfeDE7XHJcbiAgICAgICAgICAgICAgICBjdXJyX2Jlc3QgPSB4MTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICBpZiAoZl94MiA8IGN1cnJfYmVzdF9mKSB7XHJcbiAgICAgICAgICAgICAgICBjdXJyX2Jlc3RfZiA9IGZfeDI7XHJcbiAgICAgICAgICAgICAgICBjdXJyX2Jlc3QgPSB4MjtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICBpZiAoZl94MSA+IGZfeDIpIHtcclxuICAgICAgICAgICAgICAgIGxlZnQgPSB4MTtcclxuICAgICAgICAgICAgICAgIHgxID0geDI7XHJcbiAgICAgICAgICAgICAgICBmX3gxID0gZl94MjtcclxuICAgICAgICAgICAgICAgIGZsYWcgPSAxO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIGVsc2Uge1xyXG4gICAgICAgICAgICAgICAgcmlnaHQgPSB4MjtcclxuICAgICAgICAgICAgICAgIHgyID0geDE7XHJcbiAgICAgICAgICAgICAgICBmX3gyID0gZl94MTtcclxuICAgICAgICAgICAgICAgIGZsYWcgPSAwO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIGlmIChpdGVyYXRpb25Db3VudGVyKysgPiAxMDApIHtcclxuICAgICAgICAgICAgICAgIGJyZWFrO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHN0ZXAoZGF0YSwgY3Vycl9iZXN0KTtcclxuICAgIH1cclxuICAgIGZ1bmN0aW9uIHN0ZXAoZGF0YSwgbWF4X3dpZHRoKSB7XHJcbiAgICAgICAgbGluZSA9IFtdO1xyXG4gICAgICAgIHJlYWxfd2lkdGggPSAwO1xyXG4gICAgICAgIHJlYWxfaGVpZ2h0ID0gMDtcclxuICAgICAgICBnbG9iYWxfYm90dG9tID0gaW5pdF95O1xyXG4gICAgICAgIGZvciAodmFyIGkgPSAwOyBpIDwgZGF0YS5sZW5ndGg7IGkrKykge1xyXG4gICAgICAgICAgICB2YXIgbyA9IGRhdGFbaV07XHJcbiAgICAgICAgICAgIHB1dF9yZWN0KG8sIG1heF93aWR0aCk7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHJldHVybiBNYXRoLmFicyhnZXRfcmVhbF9yYXRpbygpIC0gZGVzaXJlZF9yYXRpbyk7XHJcbiAgICB9XHJcbiAgICBmdW5jdGlvbiBwdXRfcmVjdChyZWN0LCBtYXhfd2lkdGgpIHtcclxuICAgICAgICB2YXIgcGFyZW50ID0gdW5kZWZpbmVkO1xyXG4gICAgICAgIGZvciAodmFyIGkgPSAwOyBpIDwgbGluZS5sZW5ndGg7IGkrKykge1xyXG4gICAgICAgICAgICBpZiAoKGxpbmVbaV0uc3BhY2VfbGVmdCA+PSByZWN0LmhlaWdodCkgJiYgKGxpbmVbaV0ueCArIGxpbmVbaV0ud2lkdGggKyByZWN0LndpZHRoICsgcGFja2luZ09wdGlvbnMuUEFERElORyAtIG1heF93aWR0aCkgPD0gcGFja2luZ09wdGlvbnMuRkxPQVRfRVBTSUxPTikge1xyXG4gICAgICAgICAgICAgICAgcGFyZW50ID0gbGluZVtpXTtcclxuICAgICAgICAgICAgICAgIGJyZWFrO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG4gICAgICAgIGxpbmUucHVzaChyZWN0KTtcclxuICAgICAgICBpZiAocGFyZW50ICE9PSB1bmRlZmluZWQpIHtcclxuICAgICAgICAgICAgcmVjdC54ID0gcGFyZW50LnggKyBwYXJlbnQud2lkdGggKyBwYWNraW5nT3B0aW9ucy5QQURESU5HO1xyXG4gICAgICAgICAgICByZWN0LnkgPSBwYXJlbnQuYm90dG9tO1xyXG4gICAgICAgICAgICByZWN0LnNwYWNlX2xlZnQgPSByZWN0LmhlaWdodDtcclxuICAgICAgICAgICAgcmVjdC5ib3R0b20gPSByZWN0Lnk7XHJcbiAgICAgICAgICAgIHBhcmVudC5zcGFjZV9sZWZ0IC09IHJlY3QuaGVpZ2h0ICsgcGFja2luZ09wdGlvbnMuUEFERElORztcclxuICAgICAgICAgICAgcGFyZW50LmJvdHRvbSArPSByZWN0LmhlaWdodCArIHBhY2tpbmdPcHRpb25zLlBBRERJTkc7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIGVsc2Uge1xyXG4gICAgICAgICAgICByZWN0LnkgPSBnbG9iYWxfYm90dG9tO1xyXG4gICAgICAgICAgICBnbG9iYWxfYm90dG9tICs9IHJlY3QuaGVpZ2h0ICsgcGFja2luZ09wdGlvbnMuUEFERElORztcclxuICAgICAgICAgICAgcmVjdC54ID0gaW5pdF94O1xyXG4gICAgICAgICAgICByZWN0LmJvdHRvbSA9IHJlY3QueTtcclxuICAgICAgICAgICAgcmVjdC5zcGFjZV9sZWZ0ID0gcmVjdC5oZWlnaHQ7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIGlmIChyZWN0LnkgKyByZWN0LmhlaWdodCAtIHJlYWxfaGVpZ2h0ID4gLXBhY2tpbmdPcHRpb25zLkZMT0FUX0VQU0lMT04pXHJcbiAgICAgICAgICAgIHJlYWxfaGVpZ2h0ID0gcmVjdC55ICsgcmVjdC5oZWlnaHQgLSBpbml0X3k7XHJcbiAgICAgICAgaWYgKHJlY3QueCArIHJlY3Qud2lkdGggLSByZWFsX3dpZHRoID4gLXBhY2tpbmdPcHRpb25zLkZMT0FUX0VQU0lMT04pXHJcbiAgICAgICAgICAgIHJlYWxfd2lkdGggPSByZWN0LnggKyByZWN0LndpZHRoIC0gaW5pdF94O1xyXG4gICAgfVxyXG4gICAgO1xyXG4gICAgZnVuY3Rpb24gZ2V0X2VudGlyZV93aWR0aChkYXRhKSB7XHJcbiAgICAgICAgdmFyIHdpZHRoID0gMDtcclxuICAgICAgICBkYXRhLmZvckVhY2goZnVuY3Rpb24gKGQpIHsgcmV0dXJuIHdpZHRoICs9IGQud2lkdGggKyBwYWNraW5nT3B0aW9ucy5QQURESU5HOyB9KTtcclxuICAgICAgICByZXR1cm4gd2lkdGg7XHJcbiAgICB9XHJcbiAgICBmdW5jdGlvbiBnZXRfcmVhbF9yYXRpbygpIHtcclxuICAgICAgICByZXR1cm4gKHJlYWxfd2lkdGggLyByZWFsX2hlaWdodCk7XHJcbiAgICB9XHJcbn1cclxuZXhwb3J0cy5hcHBseVBhY2tpbmcgPSBhcHBseVBhY2tpbmc7XHJcbmZ1bmN0aW9uIHNlcGFyYXRlR3JhcGhzKG5vZGVzLCBsaW5rcykge1xyXG4gICAgdmFyIG1hcmtzID0ge307XHJcbiAgICB2YXIgd2F5cyA9IHt9O1xyXG4gICAgdmFyIGdyYXBocyA9IFtdO1xyXG4gICAgdmFyIGNsdXN0ZXJzID0gMDtcclxuICAgIGZvciAodmFyIGkgPSAwOyBpIDwgbGlua3MubGVuZ3RoOyBpKyspIHtcclxuICAgICAgICB2YXIgbGluayA9IGxpbmtzW2ldO1xyXG4gICAgICAgIHZhciBuMSA9IGxpbmsuc291cmNlO1xyXG4gICAgICAgIHZhciBuMiA9IGxpbmsudGFyZ2V0O1xyXG4gICAgICAgIGlmICh3YXlzW24xLmluZGV4XSlcclxuICAgICAgICAgICAgd2F5c1tuMS5pbmRleF0ucHVzaChuMik7XHJcbiAgICAgICAgZWxzZVxyXG4gICAgICAgICAgICB3YXlzW24xLmluZGV4XSA9IFtuMl07XHJcbiAgICAgICAgaWYgKHdheXNbbjIuaW5kZXhdKVxyXG4gICAgICAgICAgICB3YXlzW24yLmluZGV4XS5wdXNoKG4xKTtcclxuICAgICAgICBlbHNlXHJcbiAgICAgICAgICAgIHdheXNbbjIuaW5kZXhdID0gW24xXTtcclxuICAgIH1cclxuICAgIGZvciAodmFyIGkgPSAwOyBpIDwgbm9kZXMubGVuZ3RoOyBpKyspIHtcclxuICAgICAgICB2YXIgbm9kZSA9IG5vZGVzW2ldO1xyXG4gICAgICAgIGlmIChtYXJrc1tub2RlLmluZGV4XSlcclxuICAgICAgICAgICAgY29udGludWU7XHJcbiAgICAgICAgZXhwbG9yZV9ub2RlKG5vZGUsIHRydWUpO1xyXG4gICAgfVxyXG4gICAgZnVuY3Rpb24gZXhwbG9yZV9ub2RlKG4sIGlzX25ldykge1xyXG4gICAgICAgIGlmIChtYXJrc1tuLmluZGV4XSAhPT0gdW5kZWZpbmVkKVxyXG4gICAgICAgICAgICByZXR1cm47XHJcbiAgICAgICAgaWYgKGlzX25ldykge1xyXG4gICAgICAgICAgICBjbHVzdGVycysrO1xyXG4gICAgICAgICAgICBncmFwaHMucHVzaCh7IGFycmF5OiBbXSB9KTtcclxuICAgICAgICB9XHJcbiAgICAgICAgbWFya3Nbbi5pbmRleF0gPSBjbHVzdGVycztcclxuICAgICAgICBncmFwaHNbY2x1c3RlcnMgLSAxXS5hcnJheS5wdXNoKG4pO1xyXG4gICAgICAgIHZhciBhZGphY2VudCA9IHdheXNbbi5pbmRleF07XHJcbiAgICAgICAgaWYgKCFhZGphY2VudClcclxuICAgICAgICAgICAgcmV0dXJuO1xyXG4gICAgICAgIGZvciAodmFyIGogPSAwOyBqIDwgYWRqYWNlbnQubGVuZ3RoOyBqKyspIHtcclxuICAgICAgICAgICAgZXhwbG9yZV9ub2RlKGFkamFjZW50W2pdLCBmYWxzZSk7XHJcbiAgICAgICAgfVxyXG4gICAgfVxyXG4gICAgcmV0dXJuIGdyYXBocztcclxufVxyXG5leHBvcnRzLnNlcGFyYXRlR3JhcGhzID0gc2VwYXJhdGVHcmFwaHM7XHJcbi8vIyBzb3VyY2VNYXBwaW5nVVJMPWRhdGE6YXBwbGljYXRpb24vanNvbjtiYXNlNjQsZXlKMlpYSnphVzl1SWpvekxDSm1hV3hsSWpvaWFHRnVaR3hsWkdselkyOXVibVZqZEdWa0xtcHpJaXdpYzI5MWNtTmxVbTl2ZENJNklpSXNJbk52ZFhKalpYTWlPbHNpTGk0dkxpNHZWMlZpUTI5c1lTOXpjbU12YUdGdVpHeGxaR2x6WTI5dWJtVmpkR1ZrTG5SeklsMHNJbTVoYldWeklqcGJYU3dpYldGd2NHbHVaM01pT2lJN08wRkJRVWtzU1VGQlNTeGpRVUZqTEVkQlFVYzdTVUZEYWtJc1QwRkJUeXhGUVVGRkxFVkJRVVU3U1VGRFdDeGpRVUZqTEVWQlFVVXNRMEZCUXl4RFFVRkRMRWRCUVVjc1NVRkJTU3hEUVVGRExFbEJRVWtzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4SFFVRkhMRU5CUVVNN1NVRkRkRU1zWVVGQllTeEZRVUZGTEUxQlFVMDdTVUZEY2tJc1kwRkJZeXhGUVVGRkxFZEJRVWM3UTBGRGRFSXNRMEZCUXp0QlFVZEdMRk5CUVdkQ0xGbEJRVmtzUTBGQlF5eE5RVUZwUWl4RlFVRkZMRU5CUVVNc1JVRkJSU3hEUVVGRExFVkJRVVVzVTBGQlV5eEZRVUZGTEdGQlFXbENMRVZCUVVVc1YwRkJhMEk3U1VGQmNrTXNPRUpCUVVFc1JVRkJRU3hwUWtGQmFVSTdTVUZCUlN3MFFrRkJRU3hGUVVGQkxHdENRVUZyUWp0SlFVVnNSeXhKUVVGSkxFMUJRVTBzUjBGQlJ5eERRVUZETEVWQlExWXNUVUZCVFN4SFFVRkhMRU5CUVVNc1JVRkZWaXhUUVVGVExFZEJRVWNzUTBGQlF5eEZRVU5pTEZWQlFWVXNSMEZCUnl4RFFVRkRMRVZCUldRc1lVRkJZU3hIUVVGSExFOUJRVThzWVVGQllTeExRVUZMTEZkQlFWY3NRMEZCUXl4RFFVRkRMRU5CUVVNc1lVRkJZU3hEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVWQlEzaEZMRk5CUVZNc1IwRkJSeXhQUVVGUExGTkJRVk1zUzBGQlN5eFhRVUZYTEVOQlFVTXNRMEZCUXl4RFFVRkRMRk5CUVZNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eEZRVVUxUkN4VlFVRlZMRWRCUVVjc1EwRkJReXhGUVVOa0xGZEJRVmNzUjBGQlJ5eERRVUZETEVWQlEyWXNVMEZCVXl4SFFVRkhMRU5CUVVNc1JVRkZZaXhoUVVGaExFZEJRVWNzUTBGQlF5eEZRVU5xUWl4SlFVRkpMRWRCUVVjc1JVRkJSU3hEUVVGRE8wbEJSV1FzU1VGQlNTeE5RVUZOTEVOQlFVTXNUVUZCVFN4SlFVRkpMRU5CUVVNN1VVRkRiRUlzVDBGQlR6dEpRVlZZTEZsQlFWa3NRMEZCUXl4TlFVRk5MRU5CUVVNc1EwRkJRenRKUVVOeVFpeExRVUZMTEVOQlFVTXNUVUZCVFN4RlFVRkZMR0ZCUVdFc1EwRkJReXhEUVVGRE8wbEJRemRDTEVsQlFVY3NWMEZCVnl4RlFVRkZPMUZCUTFvc05FSkJRVFJDTEVOQlFVTXNUVUZCVFN4RFFVRkRMRU5CUVVNN1MwRkRlRU03U1VGSFJDeFRRVUZUTEZsQlFWa3NRMEZCUXl4TlFVRk5PMUZCUlhoQ0xFMUJRVTBzUTBGQlF5eFBRVUZQTEVOQlFVTXNWVUZCVlN4RFFVRkRPMWxCUTNSQ0xHMUNRVUZ0UWl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGQk8xRkJRekZDTEVOQlFVTXNRMEZCUXl4RFFVRkRPMUZCUlVnc1UwRkJVeXh0UWtGQmJVSXNRMEZCUXl4TFFVRkxPMWxCUXpsQ0xFbEJRVWtzUzBGQlN5eEhRVUZITEUxQlFVMHNRMEZCUXl4VFFVRlRMRVZCUVVVc1MwRkJTeXhIUVVGSExFMUJRVTBzUTBGQlF5eFRRVUZUTEVWQlEyeEVMRXRCUVVzc1IwRkJSeXhEUVVGRExFVkJRVVVzUzBGQlN5eEhRVUZITEVOQlFVTXNRMEZCUXp0WlFVVjZRaXhMUVVGTExFTkJRVU1zUzBGQlN5eERRVUZETEU5QlFVOHNRMEZCUXl4VlFVRlZMRU5CUVVNN1owSkJRek5DTEVsQlFVa3NRMEZCUXl4SFFVRkhMRTlCUVU4c1EwRkJReXhEUVVGRExFdEJRVXNzUzBGQlN5eFhRVUZYTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhMUVVGTExFTkJRVU1zUTBGQlF5eERRVUZETEZOQlFWTXNRMEZCUXp0blFrRkROMFFzU1VGQlNTeERRVUZETEVkQlFVY3NUMEZCVHl4RFFVRkRMRU5CUVVNc1RVRkJUU3hMUVVGTExGZEJRVmNzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRTFCUVUwc1EwRkJReXhEUVVGRExFTkJRVU1zVTBGQlV5eERRVUZETzJkQ1FVTXZSQ3hEUVVGRExFbEJRVWtzUTBGQlF5eERRVUZETzJkQ1FVTlFMRU5CUVVNc1NVRkJTU3hEUVVGRExFTkJRVU03WjBKQlExQXNTMEZCU3l4SFFVRkhMRWxCUVVrc1EwRkJReXhIUVVGSExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNSMEZCUnl4RFFVRkRMRVZCUVVVc1MwRkJTeXhEUVVGRExFTkJRVU03WjBKQlEycERMRXRCUVVzc1IwRkJSeXhKUVVGSkxFTkJRVU1zUjBGQlJ5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRWRCUVVjc1EwRkJReXhGUVVGRkxFdEJRVXNzUTBGQlF5eERRVUZETzJkQ1FVTnFReXhMUVVGTExFZEJRVWNzU1VGQlNTeERRVUZETEVkQlFVY3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhIUVVGSExFTkJRVU1zUlVGQlJTeExRVUZMTEVOQlFVTXNRMEZCUXp0blFrRkRha01zUzBGQlN5eEhRVUZITEVsQlFVa3NRMEZCUXl4SFFVRkhMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUjBGQlJ5eERRVUZETEVWQlFVVXNTMEZCU3l4RFFVRkRMRU5CUVVNN1dVRkRja01zUTBGQlF5eERRVUZETEVOQlFVTTdXVUZGU0N4TFFVRkxMRU5CUVVNc1MwRkJTeXhIUVVGSExFdEJRVXNzUjBGQlJ5eExRVUZMTEVOQlFVTTdXVUZETlVJc1MwRkJTeXhEUVVGRExFMUJRVTBzUjBGQlJ5eExRVUZMTEVkQlFVY3NTMEZCU3l4RFFVRkRPMUZCUTJwRExFTkJRVU03U1VGRFRDeERRVUZETzBsQmRVTkVMRk5CUVZNc05FSkJRVFJDTEVOQlFVTXNUVUZCVFR0UlFVTjRReXhOUVVGTkxFTkJRVU1zVDBGQlR5eERRVUZETEZWQlFWVXNRMEZCUXp0WlFVVjBRaXhKUVVGSkxFMUJRVTBzUjBGQlJ5eEZRVUZGTEVOQlFVTXNSVUZCUlN4RFFVRkRMRVZCUVVVc1EwRkJReXhGUVVGRkxFTkJRVU1zUlVGQlJTeERRVUZETzFsQlJUVkNMRU5CUVVNc1EwRkJReXhMUVVGTExFTkJRVU1zVDBGQlR5eERRVUZETEZWQlFWVXNTVUZCU1R0blFrRkRNVUlzVFVGQlRTeERRVUZETEVOQlFVTXNTVUZCU1N4SlFVRkpMRU5CUVVNc1EwRkJReXhEUVVGRE8yZENRVU51UWl4TlFVRk5MRU5CUVVNc1EwRkJReXhKUVVGSkxFbEJRVWtzUTBGQlF5eERRVUZETEVOQlFVTTdXVUZEZGtJc1EwRkJReXhEUVVGRExFTkJRVU03V1VGRlNDeE5RVUZOTEVOQlFVTXNRMEZCUXl4SlFVRkpMRU5CUVVNc1EwRkJReXhMUVVGTExFTkJRVU1zVFVGQlRTeERRVUZETzFsQlF6TkNMRTFCUVUwc1EwRkJReXhEUVVGRExFbEJRVWtzUTBGQlF5eERRVUZETEV0QlFVc3NRMEZCUXl4TlFVRk5MRU5CUVVNN1dVRkhNMElzU1VGQlNTeE5RVUZOTEVkQlFVY3NSVUZCUlN4RFFVRkRMRVZCUVVVc1RVRkJUU3hEUVVGRExFTkJRVU1zUjBGQlJ5eERRVUZETEVOQlFVTXNTMEZCU3l4SFFVRkhMRU5CUVVNc1JVRkJSU3hEUVVGRExFVkJRVVVzVFVGQlRTeERRVUZETEVOQlFVTXNSMEZCUnl4RFFVRkRMRU5CUVVNc1RVRkJUU3hIUVVGSExFTkJRVU1zUlVGQlJTeERRVUZETzFsQlEzWkZMRWxCUVVrc1RVRkJUU3hIUVVGSExFVkJRVVVzUTBGQlF5eEZRVUZGTEVOQlFVTXNRMEZCUXl4RFFVRkRMRWRCUVVjc1RVRkJUU3hEUVVGRExFTkJRVU1zUjBGQlJ5eFRRVUZUTEVkQlFVY3NRMEZCUXl4SFFVRkhMRlZCUVZVc1IwRkJSeXhEUVVGRExFVkJRVVVzUTBGQlF5eEZRVUZGTEVOQlFVTXNRMEZCUXl4RFFVRkRMRWRCUVVjc1RVRkJUU3hEUVVGRExFTkJRVU1zUjBGQlJ5eFZRVUZWTEVkQlFVY3NRMEZCUXl4SFFVRkhMRmRCUVZjc1IwRkJSeXhEUVVGRExFVkJRVU1zUTBGQlF6dFpRVWQ2U0N4RFFVRkRMRU5CUVVNc1MwRkJTeXhEUVVGRExFOUJRVThzUTBGQlF5eFZRVUZWTEVsQlFVazdaMEpCUXpGQ0xFbEJRVWtzUTBGQlF5eERRVUZETEVsQlFVa3NUVUZCVFN4RFFVRkRMRU5CUVVNc1EwRkJRenRuUWtGRGJrSXNTVUZCU1N4RFFVRkRMRU5CUVVNc1NVRkJTU3hOUVVGTkxFTkJRVU1zUTBGQlF5eERRVUZETzFsQlEzWkNMRU5CUVVNc1EwRkJReXhEUVVGRE8xRkJRMUFzUTBGQlF5eERRVUZETEVOQlFVTTdTVUZEVUN4RFFVRkRPMGxCU1VRc1UwRkJVeXhMUVVGTExFTkJRVU1zU1VGQlNTeEZRVUZGTEdGQlFXRTdVVUZET1VJc1NVRkJTU3hYUVVGWExFZEJRVWNzVFVGQlRTeERRVUZETEdsQ1FVRnBRaXhEUVVGRE8xRkJRek5ETEVsQlFVa3NVMEZCVXl4SFFVRkhMRU5CUVVNc1EwRkJRenRSUVVOc1FpeEpRVUZKTEVOQlFVTXNTVUZCU1N4RFFVRkRMRlZCUVZVc1EwRkJReXhGUVVGRkxFTkJRVU1zU1VGQlNTeFBRVUZQTEVOQlFVTXNRMEZCUXl4TlFVRk5MRWRCUVVjc1EwRkJReXhEUVVGRExFMUJRVTBzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRPMUZCUlRORUxGTkJRVk1zUjBGQlJ5eEpRVUZKTEVOQlFVTXNUVUZCVFN4RFFVRkRMRlZCUVZVc1EwRkJReXhGUVVGRkxFTkJRVU03V1VGRGJFTXNUMEZCVHl4RFFVRkRMRU5CUVVNc1MwRkJTeXhIUVVGSExFTkJRVU1zUTBGQlF5eExRVUZMTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhMUVVGTExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4TFFVRkxMRU5CUVVNN1VVRkRha1FzUTBGQlF5eERRVUZETEVOQlFVTTdVVUZGU0N4SlFVRkpMRWxCUVVrc1IwRkJSeXhGUVVGRkxFZEJRVWNzVTBGQlV5eERRVUZETzFGQlF6RkNMRWxCUVVrc1MwRkJTeXhIUVVGSExFVkJRVVVzUjBGQlJ5eG5Ra0ZCWjBJc1EwRkJReXhKUVVGSkxFTkJRVU1zUTBGQlF6dFJRVU40UXl4SlFVRkpMR2RDUVVGblFpeEhRVUZITEVOQlFVTXNRMEZCUXp0UlFVVjZRaXhKUVVGSkxFbEJRVWtzUjBGQlJ5eE5RVUZOTEVOQlFVTXNVMEZCVXl4RFFVRkRPMUZCUXpWQ0xFbEJRVWtzU1VGQlNTeEhRVUZITEUxQlFVMHNRMEZCUXl4VFFVRlRMRU5CUVVNN1VVRkROVUlzU1VGQlNTeEpRVUZKTEVkQlFVY3NRMEZCUXl4RFFVRkRMRU5CUVVNN1VVRkhaQ3hKUVVGSkxFVkJRVVVzUjBGQlJ5eE5RVUZOTEVOQlFVTXNVMEZCVXl4RFFVRkRPMUZCUXpGQ0xFbEJRVWtzUlVGQlJTeEhRVUZITEUxQlFVMHNRMEZCUXl4VFFVRlRMRU5CUVVNN1VVRkZNVUlzVDBGQlR5eERRVUZETEVWQlFVVXNSMEZCUnl4VFFVRlRMRU5CUVVNc1NVRkJTU3hGUVVGRkxFZEJRVWNzWTBGQll5eERRVUZETEdGQlFXRXNSVUZCUlR0WlFVVXhSQ3hKUVVGSkxFbEJRVWtzU1VGQlNTeERRVUZETEVWQlFVVTdaMEpCUTFnc1NVRkJTU3hGUVVGRkxFZEJRVWNzUzBGQlN5eEhRVUZITEVOQlFVTXNTMEZCU3l4SFFVRkhMRWxCUVVrc1EwRkJReXhIUVVGSExHTkJRV01zUTBGQlF5eGpRVUZqTEVOQlFVTTdaMEpCUTJoRkxFbEJRVWtzU1VGQlNTeEhRVUZITEVsQlFVa3NRMEZCUXl4SlFVRkpMRVZCUVVVc1JVRkJSU3hEUVVGRExFTkJRVU03WVVGRE4wSTdXVUZEUkN4SlFVRkpMRWxCUVVrc1NVRkJTU3hEUVVGRExFVkJRVVU3WjBKQlExZ3NTVUZCU1N4RlFVRkZMRWRCUVVjc1NVRkJTU3hIUVVGSExFTkJRVU1zUzBGQlN5eEhRVUZITEVsQlFVa3NRMEZCUXl4SFFVRkhMR05CUVdNc1EwRkJReXhqUVVGakxFTkJRVU03WjBKQlF5OUVMRWxCUVVrc1NVRkJTU3hIUVVGSExFbEJRVWtzUTBGQlF5eEpRVUZKTEVWQlFVVXNSVUZCUlN4RFFVRkRMRU5CUVVNN1lVRkROMEk3V1VGRlJDeEZRVUZGTEVkQlFVY3NTVUZCU1N4RFFVRkRMRWRCUVVjc1EwRkJReXhGUVVGRkxFZEJRVWNzUlVGQlJTeERRVUZETEVOQlFVTTdXVUZEZGtJc1JVRkJSU3hIUVVGSExFbEJRVWtzUTBGQlF5eEhRVUZITEVOQlFVTXNTVUZCU1N4SFFVRkhMRWxCUVVrc1EwRkJReXhEUVVGRE8xbEJSVE5DTEVsQlFVa3NTVUZCU1N4SFFVRkhMRmRCUVZjc1JVRkJSVHRuUWtGRGNFSXNWMEZCVnl4SFFVRkhMRWxCUVVrc1EwRkJRenRuUWtGRGJrSXNVMEZCVXl4SFFVRkhMRVZCUVVVc1EwRkJRenRoUVVOc1FqdFpRVVZFTEVsQlFVa3NTVUZCU1N4SFFVRkhMRmRCUVZjc1JVRkJSVHRuUWtGRGNFSXNWMEZCVnl4SFFVRkhMRWxCUVVrc1EwRkJRenRuUWtGRGJrSXNVMEZCVXl4SFFVRkhMRVZCUVVVc1EwRkJRenRoUVVOc1FqdFpRVVZFTEVsQlFVa3NTVUZCU1N4SFFVRkhMRWxCUVVrc1JVRkJSVHRuUWtGRFlpeEpRVUZKTEVkQlFVY3NSVUZCUlN4RFFVRkRPMmRDUVVOV0xFVkJRVVVzUjBGQlJ5eEZRVUZGTEVOQlFVTTdaMEpCUTFJc1NVRkJTU3hIUVVGSExFbEJRVWtzUTBGQlF6dG5Ra0ZEV2l4SlFVRkpMRWRCUVVjc1EwRkJReXhEUVVGRE8yRkJRMW83YVVKQlFVMDdaMEpCUTBnc1MwRkJTeXhIUVVGSExFVkJRVVVzUTBGQlF6dG5Ra0ZEV0N4RlFVRkZMRWRCUVVjc1JVRkJSU3hEUVVGRE8yZENRVU5TTEVsQlFVa3NSMEZCUnl4SlFVRkpMRU5CUVVNN1owSkJRMW9zU1VGQlNTeEhRVUZITEVOQlFVTXNRMEZCUXp0aFFVTmFPMWxCUlVRc1NVRkJTU3huUWtGQlowSXNSVUZCUlN4SFFVRkhMRWRCUVVjc1JVRkJSVHRuUWtGRE1VSXNUVUZCVFR0aFFVTlVPMU5CUTBvN1VVRkZSQ3hKUVVGSkxFTkJRVU1zU1VGQlNTeEZRVUZGTEZOQlFWTXNRMEZCUXl4RFFVRkRPMGxCUXpGQ0xFTkJRVU03U1VGSlJDeFRRVUZUTEVsQlFVa3NRMEZCUXl4SlFVRkpMRVZCUVVVc1UwRkJVenRSUVVONlFpeEpRVUZKTEVkQlFVY3NSVUZCUlN4RFFVRkRPMUZCUTFZc1ZVRkJWU3hIUVVGSExFTkJRVU1zUTBGQlF6dFJRVU5tTEZkQlFWY3NSMEZCUnl4RFFVRkRMRU5CUVVNN1VVRkRhRUlzWVVGQllTeEhRVUZITEUxQlFVMHNRMEZCUXp0UlFVVjJRaXhMUVVGTExFbEJRVWtzUTBGQlF5eEhRVUZITEVOQlFVTXNSVUZCUlN4RFFVRkRMRWRCUVVjc1NVRkJTU3hEUVVGRExFMUJRVTBzUlVGQlJTeERRVUZETEVWQlFVVXNSVUZCUlR0WlFVTnNReXhKUVVGSkxFTkJRVU1zUjBGQlJ5eEpRVUZKTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNN1dVRkRhRUlzVVVGQlVTeERRVUZETEVOQlFVTXNSVUZCUlN4VFFVRlRMRU5CUVVNc1EwRkJRenRUUVVNeFFqdFJRVVZFTEU5QlFVOHNTVUZCU1N4RFFVRkRMRWRCUVVjc1EwRkJReXhqUVVGakxFVkJRVVVzUjBGQlJ5eGhRVUZoTEVOQlFVTXNRMEZCUXp0SlFVTjBSQ3hEUVVGRE8wbEJSMFFzVTBGQlV5eFJRVUZSTEVOQlFVTXNTVUZCU1N4RlFVRkZMRk5CUVZNN1VVRkhOMElzU1VGQlNTeE5RVUZOTEVkQlFVY3NVMEZCVXl4RFFVRkRPMUZCUlhaQ0xFdEJRVXNzU1VGQlNTeERRVUZETEVkQlFVY3NRMEZCUXl4RlFVRkZMRU5CUVVNc1IwRkJSeXhKUVVGSkxFTkJRVU1zVFVGQlRTeEZRVUZGTEVOQlFVTXNSVUZCUlN4RlFVRkZPMWxCUTJ4RExFbEJRVWtzUTBGQlF5eEpRVUZKTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1ZVRkJWU3hKUVVGSkxFbEJRVWtzUTBGQlF5eE5RVUZOTEVOQlFVTXNTVUZCU1N4RFFVRkRMRWxCUVVrc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVkQlFVY3NTVUZCU1N4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFdEJRVXNzUjBGQlJ5eEpRVUZKTEVOQlFVTXNTMEZCU3l4SFFVRkhMR05CUVdNc1EwRkJReXhQUVVGUExFZEJRVWNzVTBGQlV5eERRVUZETEVsQlFVa3NZMEZCWXl4RFFVRkRMR0ZCUVdFc1JVRkJSVHRuUWtGRGRFb3NUVUZCVFN4SFFVRkhMRWxCUVVrc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dG5Ra0ZEYWtJc1RVRkJUVHRoUVVOVU8xTkJRMG83VVVGRlJDeEpRVUZKTEVOQlFVTXNTVUZCU1N4RFFVRkRMRWxCUVVrc1EwRkJReXhEUVVGRE8xRkJSV2hDTEVsQlFVa3NUVUZCVFN4TFFVRkxMRk5CUVZNc1JVRkJSVHRaUVVOMFFpeEpRVUZKTEVOQlFVTXNRMEZCUXl4SFFVRkhMRTFCUVUwc1EwRkJReXhEUVVGRExFZEJRVWNzVFVGQlRTeERRVUZETEV0QlFVc3NSMEZCUnl4alFVRmpMRU5CUVVNc1QwRkJUeXhEUVVGRE8xbEJRekZFTEVsQlFVa3NRMEZCUXl4RFFVRkRMRWRCUVVjc1RVRkJUU3hEUVVGRExFMUJRVTBzUTBGQlF6dFpRVU4yUWl4SlFVRkpMRU5CUVVNc1ZVRkJWU3hIUVVGSExFbEJRVWtzUTBGQlF5eE5RVUZOTEVOQlFVTTdXVUZET1VJc1NVRkJTU3hEUVVGRExFMUJRVTBzUjBGQlJ5eEpRVUZKTEVOQlFVTXNRMEZCUXl4RFFVRkRPMWxCUTNKQ0xFMUJRVTBzUTBGQlF5eFZRVUZWTEVsQlFVa3NTVUZCU1N4RFFVRkRMRTFCUVUwc1IwRkJSeXhqUVVGakxFTkJRVU1zVDBGQlR5eERRVUZETzFsQlF6RkVMRTFCUVUwc1EwRkJReXhOUVVGTkxFbEJRVWtzU1VGQlNTeERRVUZETEUxQlFVMHNSMEZCUnl4alFVRmpMRU5CUVVNc1QwRkJUeXhEUVVGRE8xTkJRM3BFTzJGQlFVMDdXVUZEU0N4SlFVRkpMRU5CUVVNc1EwRkJReXhIUVVGSExHRkJRV0VzUTBGQlF6dFpRVU4yUWl4aFFVRmhMRWxCUVVrc1NVRkJTU3hEUVVGRExFMUJRVTBzUjBGQlJ5eGpRVUZqTEVOQlFVTXNUMEZCVHl4RFFVRkRPMWxCUTNSRUxFbEJRVWtzUTBGQlF5eERRVUZETEVkQlFVY3NUVUZCVFN4RFFVRkRPMWxCUTJoQ0xFbEJRVWtzUTBGQlF5eE5RVUZOTEVkQlFVY3NTVUZCU1N4RFFVRkRMRU5CUVVNc1EwRkJRenRaUVVOeVFpeEpRVUZKTEVOQlFVTXNWVUZCVlN4SFFVRkhMRWxCUVVrc1EwRkJReXhOUVVGTkxFTkJRVU03VTBGRGFrTTdVVUZGUkN4SlFVRkpMRWxCUVVrc1EwRkJReXhEUVVGRExFZEJRVWNzU1VGQlNTeERRVUZETEUxQlFVMHNSMEZCUnl4WFFVRlhMRWRCUVVjc1EwRkJReXhqUVVGakxFTkJRVU1zWVVGQllUdFpRVUZGTEZkQlFWY3NSMEZCUnl4SlFVRkpMRU5CUVVNc1EwRkJReXhIUVVGSExFbEJRVWtzUTBGQlF5eE5RVUZOTEVkQlFVY3NUVUZCVFN4RFFVRkRPMUZCUTNCSUxFbEJRVWtzU1VGQlNTeERRVUZETEVOQlFVTXNSMEZCUnl4SlFVRkpMRU5CUVVNc1MwRkJTeXhIUVVGSExGVkJRVlVzUjBGQlJ5eERRVUZETEdOQlFXTXNRMEZCUXl4aFFVRmhPMWxCUVVVc1ZVRkJWU3hIUVVGSExFbEJRVWtzUTBGQlF5eERRVUZETEVkQlFVY3NTVUZCU1N4RFFVRkRMRXRCUVVzc1IwRkJSeXhOUVVGTkxFTkJRVU03U1VGRGNFZ3NRMEZCUXp0SlFVRkJMRU5CUVVNN1NVRkZSaXhUUVVGVExHZENRVUZuUWl4RFFVRkRMRWxCUVVrN1VVRkRNVUlzU1VGQlNTeExRVUZMTEVkQlFVY3NRMEZCUXl4RFFVRkRPMUZCUTJRc1NVRkJTU3hEUVVGRExFOUJRVThzUTBGQlF5eFZRVUZWTEVOQlFVTXNTVUZCU1N4UFFVRlBMRXRCUVVzc1NVRkJTU3hEUVVGRExFTkJRVU1zUzBGQlN5eEhRVUZITEdOQlFXTXNRMEZCUXl4UFFVRlBMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dFJRVU5xUml4UFFVRlBMRXRCUVVzc1EwRkJRenRKUVVOcVFpeERRVUZETzBsQlJVUXNVMEZCVXl4alFVRmpPMUZCUTI1Q0xFOUJRVThzUTBGQlF5eFZRVUZWTEVkQlFVY3NWMEZCVnl4RFFVRkRMRU5CUVVNN1NVRkRkRU1zUTBGQlF6dEJRVU5NTEVOQlFVTTdRVUV4VUVRc2IwTkJNRkJETzBGQlRVUXNVMEZCWjBJc1kwRkJZeXhEUVVGRExFdEJRVXNzUlVGQlJTeExRVUZMTzBsQlEzWkRMRWxCUVVrc1MwRkJTeXhIUVVGSExFVkJRVVVzUTBGQlF6dEpRVU5tTEVsQlFVa3NTVUZCU1N4SFFVRkhMRVZCUVVVc1EwRkJRenRKUVVOa0xFbEJRVWtzVFVGQlRTeEhRVUZITEVWQlFVVXNRMEZCUXp0SlFVTm9RaXhKUVVGSkxGRkJRVkVzUjBGQlJ5eERRVUZETEVOQlFVTTdTVUZGYWtJc1MwRkJTeXhKUVVGSkxFTkJRVU1zUjBGQlJ5eERRVUZETEVWQlFVVXNRMEZCUXl4SFFVRkhMRXRCUVVzc1EwRkJReXhOUVVGTkxFVkJRVVVzUTBGQlF5eEZRVUZGTEVWQlFVVTdVVUZEYmtNc1NVRkJTU3hKUVVGSkxFZEJRVWNzUzBGQlN5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRPMUZCUTNCQ0xFbEJRVWtzUlVGQlJTeEhRVUZITEVsQlFVa3NRMEZCUXl4TlFVRk5MRU5CUVVNN1VVRkRja0lzU1VGQlNTeEZRVUZGTEVkQlFVY3NTVUZCU1N4RFFVRkRMRTFCUVUwc1EwRkJRenRSUVVOeVFpeEpRVUZKTEVsQlFVa3NRMEZCUXl4RlFVRkZMRU5CUVVNc1MwRkJTeXhEUVVGRE8xbEJRMlFzU1VGQlNTeERRVUZETEVWQlFVVXNRMEZCUXl4TFFVRkxMRU5CUVVNc1EwRkJReXhKUVVGSkxFTkJRVU1zUlVGQlJTeERRVUZETEVOQlFVTTdPMWxCUlhoQ0xFbEJRVWtzUTBGQlF5eEZRVUZGTEVOQlFVTXNTMEZCU3l4RFFVRkRMRWRCUVVjc1EwRkJReXhGUVVGRkxFTkJRVU1zUTBGQlF6dFJRVVV4UWl4SlFVRkpMRWxCUVVrc1EwRkJReXhGUVVGRkxFTkJRVU1zUzBGQlN5eERRVUZETzFsQlEyUXNTVUZCU1N4RFFVRkRMRVZCUVVVc1EwRkJReXhMUVVGTExFTkJRVU1zUTBGQlF5eEpRVUZKTEVOQlFVTXNSVUZCUlN4RFFVRkRMRU5CUVVNN08xbEJSWGhDTEVsQlFVa3NRMEZCUXl4RlFVRkZMRU5CUVVNc1MwRkJTeXhEUVVGRExFZEJRVWNzUTBGQlF5eEZRVUZGTEVOQlFVTXNRMEZCUXp0TFFVTTNRanRKUVVWRUxFdEJRVXNzU1VGQlNTeERRVUZETEVkQlFVY3NRMEZCUXl4RlFVRkZMRU5CUVVNc1IwRkJSeXhMUVVGTExFTkJRVU1zVFVGQlRTeEZRVUZGTEVOQlFVTXNSVUZCUlN4RlFVRkZPMUZCUTI1RExFbEJRVWtzU1VGQlNTeEhRVUZITEV0QlFVc3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJRenRSUVVOd1FpeEpRVUZKTEV0QlFVc3NRMEZCUXl4SlFVRkpMRU5CUVVNc1MwRkJTeXhEUVVGRE8xbEJRVVVzVTBGQlV6dFJRVU5vUXl4WlFVRlpMRU5CUVVNc1NVRkJTU3hGUVVGRkxFbEJRVWtzUTBGQlF5eERRVUZETzB0QlF6VkNPMGxCUlVRc1UwRkJVeXhaUVVGWkxFTkJRVU1zUTBGQlF5eEZRVUZGTEUxQlFVMDdVVUZETTBJc1NVRkJTU3hMUVVGTExFTkJRVU1zUTBGQlF5eERRVUZETEV0QlFVc3NRMEZCUXl4TFFVRkxMRk5CUVZNN1dVRkJSU3hQUVVGUE8xRkJRM3BETEVsQlFVa3NUVUZCVFN4RlFVRkZPMWxCUTFJc1VVRkJVU3hGUVVGRkxFTkJRVU03V1VGRFdDeE5RVUZOTEVOQlFVTXNTVUZCU1N4RFFVRkRMRVZCUVVVc1MwRkJTeXhGUVVGRkxFVkJRVVVzUlVGQlJTeERRVUZETEVOQlFVTTdVMEZET1VJN1VVRkRSQ3hMUVVGTExFTkJRVU1zUTBGQlF5eERRVUZETEV0QlFVc3NRMEZCUXl4SFFVRkhMRkZCUVZFc1EwRkJRenRSUVVNeFFpeE5RVUZOTEVOQlFVTXNVVUZCVVN4SFFVRkhMRU5CUVVNc1EwRkJReXhEUVVGRExFdEJRVXNzUTBGQlF5eEpRVUZKTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNN1VVRkRia01zU1VGQlNTeFJRVUZSTEVkQlFVY3NTVUZCU1N4RFFVRkRMRU5CUVVNc1EwRkJReXhMUVVGTExFTkJRVU1zUTBGQlF6dFJRVU0zUWl4SlFVRkpMRU5CUVVNc1VVRkJVVHRaUVVGRkxFOUJRVTg3VVVGRmRFSXNTMEZCU3l4SlFVRkpMRU5CUVVNc1IwRkJSeXhEUVVGRExFVkJRVVVzUTBGQlF5eEhRVUZITEZGQlFWRXNRMEZCUXl4TlFVRk5MRVZCUVVVc1EwRkJReXhGUVVGRkxFVkJRVVU3V1VGRGRFTXNXVUZCV1N4RFFVRkRMRkZCUVZFc1EwRkJReXhEUVVGRExFTkJRVU1zUlVGQlJTeExRVUZMTEVOQlFVTXNRMEZCUXp0VFFVTndRenRKUVVOTUxFTkJRVU03U1VGRlJDeFBRVUZQTEUxQlFVMHNRMEZCUXp0QlFVTnNRaXhEUVVGRE8wRkJOVU5FTEhkRFFUUkRReUo5IiwiXCJ1c2Ugc3RyaWN0XCI7XHJcbk9iamVjdC5kZWZpbmVQcm9wZXJ0eShleHBvcnRzLCBcIl9fZXNNb2R1bGVcIiwgeyB2YWx1ZTogdHJ1ZSB9KTtcclxudmFyIHBvd2VyZ3JhcGggPSByZXF1aXJlKFwiLi9wb3dlcmdyYXBoXCIpO1xyXG52YXIgbGlua2xlbmd0aHNfMSA9IHJlcXVpcmUoXCIuL2xpbmtsZW5ndGhzXCIpO1xyXG52YXIgZGVzY2VudF8xID0gcmVxdWlyZShcIi4vZGVzY2VudFwiKTtcclxudmFyIHJlY3RhbmdsZV8xID0gcmVxdWlyZShcIi4vcmVjdGFuZ2xlXCIpO1xyXG52YXIgc2hvcnRlc3RwYXRoc18xID0gcmVxdWlyZShcIi4vc2hvcnRlc3RwYXRoc1wiKTtcclxudmFyIGdlb21fMSA9IHJlcXVpcmUoXCIuL2dlb21cIik7XHJcbnZhciBoYW5kbGVkaXNjb25uZWN0ZWRfMSA9IHJlcXVpcmUoXCIuL2hhbmRsZWRpc2Nvbm5lY3RlZFwiKTtcclxudmFyIEV2ZW50VHlwZTtcclxuKGZ1bmN0aW9uIChFdmVudFR5cGUpIHtcclxuICAgIEV2ZW50VHlwZVtFdmVudFR5cGVbXCJzdGFydFwiXSA9IDBdID0gXCJzdGFydFwiO1xyXG4gICAgRXZlbnRUeXBlW0V2ZW50VHlwZVtcInRpY2tcIl0gPSAxXSA9IFwidGlja1wiO1xyXG4gICAgRXZlbnRUeXBlW0V2ZW50VHlwZVtcImVuZFwiXSA9IDJdID0gXCJlbmRcIjtcclxufSkoRXZlbnRUeXBlID0gZXhwb3J0cy5FdmVudFR5cGUgfHwgKGV4cG9ydHMuRXZlbnRUeXBlID0ge30pKTtcclxuO1xyXG5mdW5jdGlvbiBpc0dyb3VwKGcpIHtcclxuICAgIHJldHVybiB0eXBlb2YgZy5sZWF2ZXMgIT09ICd1bmRlZmluZWQnIHx8IHR5cGVvZiBnLmdyb3VwcyAhPT0gJ3VuZGVmaW5lZCc7XHJcbn1cclxudmFyIExheW91dCA9IChmdW5jdGlvbiAoKSB7XHJcbiAgICBmdW5jdGlvbiBMYXlvdXQoKSB7XHJcbiAgICAgICAgdmFyIF90aGlzID0gdGhpcztcclxuICAgICAgICB0aGlzLl9jYW52YXNTaXplID0gWzEsIDFdO1xyXG4gICAgICAgIHRoaXMuX2xpbmtEaXN0YW5jZSA9IDIwO1xyXG4gICAgICAgIHRoaXMuX2RlZmF1bHROb2RlU2l6ZSA9IDEwO1xyXG4gICAgICAgIHRoaXMuX2xpbmtMZW5ndGhDYWxjdWxhdG9yID0gbnVsbDtcclxuICAgICAgICB0aGlzLl9saW5rVHlwZSA9IG51bGw7XHJcbiAgICAgICAgdGhpcy5fYXZvaWRPdmVybGFwcyA9IGZhbHNlO1xyXG4gICAgICAgIHRoaXMuX2hhbmRsZURpc2Nvbm5lY3RlZCA9IHRydWU7XHJcbiAgICAgICAgdGhpcy5fcnVubmluZyA9IGZhbHNlO1xyXG4gICAgICAgIHRoaXMuX25vZGVzID0gW107XHJcbiAgICAgICAgdGhpcy5fZ3JvdXBzID0gW107XHJcbiAgICAgICAgdGhpcy5fcm9vdEdyb3VwID0gbnVsbDtcclxuICAgICAgICB0aGlzLl9saW5rcyA9IFtdO1xyXG4gICAgICAgIHRoaXMuX2NvbnN0cmFpbnRzID0gW107XHJcbiAgICAgICAgdGhpcy5fZGlzdGFuY2VNYXRyaXggPSBudWxsO1xyXG4gICAgICAgIHRoaXMuX2Rlc2NlbnQgPSBudWxsO1xyXG4gICAgICAgIHRoaXMuX2RpcmVjdGVkTGlua0NvbnN0cmFpbnRzID0gbnVsbDtcclxuICAgICAgICB0aGlzLl90aHJlc2hvbGQgPSAwLjAxO1xyXG4gICAgICAgIHRoaXMuX3Zpc2liaWxpdHlHcmFwaCA9IG51bGw7XHJcbiAgICAgICAgdGhpcy5fZ3JvdXBDb21wYWN0bmVzcyA9IDFlLTY7XHJcbiAgICAgICAgdGhpcy5ldmVudCA9IG51bGw7XHJcbiAgICAgICAgdGhpcy5saW5rQWNjZXNzb3IgPSB7XHJcbiAgICAgICAgICAgIGdldFNvdXJjZUluZGV4OiBMYXlvdXQuZ2V0U291cmNlSW5kZXgsXHJcbiAgICAgICAgICAgIGdldFRhcmdldEluZGV4OiBMYXlvdXQuZ2V0VGFyZ2V0SW5kZXgsXHJcbiAgICAgICAgICAgIHNldExlbmd0aDogTGF5b3V0LnNldExpbmtMZW5ndGgsXHJcbiAgICAgICAgICAgIGdldFR5cGU6IGZ1bmN0aW9uIChsKSB7IHJldHVybiB0eXBlb2YgX3RoaXMuX2xpbmtUeXBlID09PSBcImZ1bmN0aW9uXCIgPyBfdGhpcy5fbGlua1R5cGUobCkgOiAwOyB9XHJcbiAgICAgICAgfTtcclxuICAgIH1cclxuICAgIExheW91dC5wcm90b3R5cGUub24gPSBmdW5jdGlvbiAoZSwgbGlzdGVuZXIpIHtcclxuICAgICAgICBpZiAoIXRoaXMuZXZlbnQpXHJcbiAgICAgICAgICAgIHRoaXMuZXZlbnQgPSB7fTtcclxuICAgICAgICBpZiAodHlwZW9mIGUgPT09ICdzdHJpbmcnKSB7XHJcbiAgICAgICAgICAgIHRoaXMuZXZlbnRbRXZlbnRUeXBlW2VdXSA9IGxpc3RlbmVyO1xyXG4gICAgICAgIH1cclxuICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgdGhpcy5ldmVudFtlXSA9IGxpc3RlbmVyO1xyXG4gICAgICAgIH1cclxuICAgICAgICByZXR1cm4gdGhpcztcclxuICAgIH07XHJcbiAgICBMYXlvdXQucHJvdG90eXBlLnRyaWdnZXIgPSBmdW5jdGlvbiAoZSkge1xyXG4gICAgICAgIGlmICh0aGlzLmV2ZW50ICYmIHR5cGVvZiB0aGlzLmV2ZW50W2UudHlwZV0gIT09ICd1bmRlZmluZWQnKSB7XHJcbiAgICAgICAgICAgIHRoaXMuZXZlbnRbZS50eXBlXShlKTtcclxuICAgICAgICB9XHJcbiAgICB9O1xyXG4gICAgTGF5b3V0LnByb3RvdHlwZS5raWNrID0gZnVuY3Rpb24gKCkge1xyXG4gICAgICAgIHdoaWxlICghdGhpcy50aWNrKCkpXHJcbiAgICAgICAgICAgIDtcclxuICAgIH07XHJcbiAgICBMYXlvdXQucHJvdG90eXBlLnRpY2sgPSBmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgaWYgKHRoaXMuX2FscGhhIDwgdGhpcy5fdGhyZXNob2xkKSB7XHJcbiAgICAgICAgICAgIHRoaXMuX3J1bm5pbmcgPSBmYWxzZTtcclxuICAgICAgICAgICAgdGhpcy50cmlnZ2VyKHsgdHlwZTogRXZlbnRUeXBlLmVuZCwgYWxwaGE6IHRoaXMuX2FscGhhID0gMCwgc3RyZXNzOiB0aGlzLl9sYXN0U3RyZXNzIH0pO1xyXG4gICAgICAgICAgICByZXR1cm4gdHJ1ZTtcclxuICAgICAgICB9XHJcbiAgICAgICAgdmFyIG4gPSB0aGlzLl9ub2Rlcy5sZW5ndGgsIG0gPSB0aGlzLl9saW5rcy5sZW5ndGg7XHJcbiAgICAgICAgdmFyIG8sIGk7XHJcbiAgICAgICAgdGhpcy5fZGVzY2VudC5sb2Nrcy5jbGVhcigpO1xyXG4gICAgICAgIGZvciAoaSA9IDA7IGkgPCBuOyArK2kpIHtcclxuICAgICAgICAgICAgbyA9IHRoaXMuX25vZGVzW2ldO1xyXG4gICAgICAgICAgICBpZiAoby5maXhlZCkge1xyXG4gICAgICAgICAgICAgICAgaWYgKHR5cGVvZiBvLnB4ID09PSAndW5kZWZpbmVkJyB8fCB0eXBlb2Ygby5weSA9PT0gJ3VuZGVmaW5lZCcpIHtcclxuICAgICAgICAgICAgICAgICAgICBvLnB4ID0gby54O1xyXG4gICAgICAgICAgICAgICAgICAgIG8ucHkgPSBvLnk7XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICB2YXIgcCA9IFtvLnB4LCBvLnB5XTtcclxuICAgICAgICAgICAgICAgIHRoaXMuX2Rlc2NlbnQubG9ja3MuYWRkKGksIHApO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHZhciBzMSA9IHRoaXMuX2Rlc2NlbnQucnVuZ2VLdXR0YSgpO1xyXG4gICAgICAgIGlmIChzMSA9PT0gMCkge1xyXG4gICAgICAgICAgICB0aGlzLl9hbHBoYSA9IDA7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIGVsc2UgaWYgKHR5cGVvZiB0aGlzLl9sYXN0U3RyZXNzICE9PSAndW5kZWZpbmVkJykge1xyXG4gICAgICAgICAgICB0aGlzLl9hbHBoYSA9IHMxO1xyXG4gICAgICAgIH1cclxuICAgICAgICB0aGlzLl9sYXN0U3RyZXNzID0gczE7XHJcbiAgICAgICAgdGhpcy51cGRhdGVOb2RlUG9zaXRpb25zKCk7XHJcbiAgICAgICAgdGhpcy50cmlnZ2VyKHsgdHlwZTogRXZlbnRUeXBlLnRpY2ssIGFscGhhOiB0aGlzLl9hbHBoYSwgc3RyZXNzOiB0aGlzLl9sYXN0U3RyZXNzIH0pO1xyXG4gICAgICAgIHJldHVybiBmYWxzZTtcclxuICAgIH07XHJcbiAgICBMYXlvdXQucHJvdG90eXBlLnVwZGF0ZU5vZGVQb3NpdGlvbnMgPSBmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgdmFyIHggPSB0aGlzLl9kZXNjZW50LnhbMF0sIHkgPSB0aGlzLl9kZXNjZW50LnhbMV07XHJcbiAgICAgICAgdmFyIG8sIGkgPSB0aGlzLl9ub2Rlcy5sZW5ndGg7XHJcbiAgICAgICAgd2hpbGUgKGktLSkge1xyXG4gICAgICAgICAgICBvID0gdGhpcy5fbm9kZXNbaV07XHJcbiAgICAgICAgICAgIG8ueCA9IHhbaV07XHJcbiAgICAgICAgICAgIG8ueSA9IHlbaV07XHJcbiAgICAgICAgfVxyXG4gICAgfTtcclxuICAgIExheW91dC5wcm90b3R5cGUubm9kZXMgPSBmdW5jdGlvbiAodikge1xyXG4gICAgICAgIGlmICghdikge1xyXG4gICAgICAgICAgICBpZiAodGhpcy5fbm9kZXMubGVuZ3RoID09PSAwICYmIHRoaXMuX2xpbmtzLmxlbmd0aCA+IDApIHtcclxuICAgICAgICAgICAgICAgIHZhciBuID0gMDtcclxuICAgICAgICAgICAgICAgIHRoaXMuX2xpbmtzLmZvckVhY2goZnVuY3Rpb24gKGwpIHtcclxuICAgICAgICAgICAgICAgICAgICBuID0gTWF0aC5tYXgobiwgbC5zb3VyY2UsIGwudGFyZ2V0KTtcclxuICAgICAgICAgICAgICAgIH0pO1xyXG4gICAgICAgICAgICAgICAgdGhpcy5fbm9kZXMgPSBuZXcgQXJyYXkoKytuKTtcclxuICAgICAgICAgICAgICAgIGZvciAodmFyIGkgPSAwOyBpIDwgbjsgKytpKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgdGhpcy5fbm9kZXNbaV0gPSB7fTtcclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICByZXR1cm4gdGhpcy5fbm9kZXM7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHRoaXMuX25vZGVzID0gdjtcclxuICAgICAgICByZXR1cm4gdGhpcztcclxuICAgIH07XHJcbiAgICBMYXlvdXQucHJvdG90eXBlLmdyb3VwcyA9IGZ1bmN0aW9uICh4KSB7XHJcbiAgICAgICAgdmFyIF90aGlzID0gdGhpcztcclxuICAgICAgICBpZiAoIXgpXHJcbiAgICAgICAgICAgIHJldHVybiB0aGlzLl9ncm91cHM7XHJcbiAgICAgICAgdGhpcy5fZ3JvdXBzID0geDtcclxuICAgICAgICB0aGlzLl9yb290R3JvdXAgPSB7fTtcclxuICAgICAgICB0aGlzLl9ncm91cHMuZm9yRWFjaChmdW5jdGlvbiAoZykge1xyXG4gICAgICAgICAgICBpZiAodHlwZW9mIGcucGFkZGluZyA9PT0gXCJ1bmRlZmluZWRcIilcclxuICAgICAgICAgICAgICAgIGcucGFkZGluZyA9IDE7XHJcbiAgICAgICAgICAgIGlmICh0eXBlb2YgZy5sZWF2ZXMgIT09IFwidW5kZWZpbmVkXCIpIHtcclxuICAgICAgICAgICAgICAgIGcubGVhdmVzLmZvckVhY2goZnVuY3Rpb24gKHYsIGkpIHtcclxuICAgICAgICAgICAgICAgICAgICBpZiAodHlwZW9mIHYgPT09ICdudW1iZXInKVxyXG4gICAgICAgICAgICAgICAgICAgICAgICAoZy5sZWF2ZXNbaV0gPSBfdGhpcy5fbm9kZXNbdl0pLnBhcmVudCA9IGc7XHJcbiAgICAgICAgICAgICAgICB9KTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICBpZiAodHlwZW9mIGcuZ3JvdXBzICE9PSBcInVuZGVmaW5lZFwiKSB7XHJcbiAgICAgICAgICAgICAgICBnLmdyb3Vwcy5mb3JFYWNoKGZ1bmN0aW9uIChnaSwgaSkge1xyXG4gICAgICAgICAgICAgICAgICAgIGlmICh0eXBlb2YgZ2kgPT09ICdudW1iZXInKVxyXG4gICAgICAgICAgICAgICAgICAgICAgICAoZy5ncm91cHNbaV0gPSBfdGhpcy5fZ3JvdXBzW2dpXSkucGFyZW50ID0gZztcclxuICAgICAgICAgICAgICAgIH0pO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfSk7XHJcbiAgICAgICAgdGhpcy5fcm9vdEdyb3VwLmxlYXZlcyA9IHRoaXMuX25vZGVzLmZpbHRlcihmdW5jdGlvbiAodikgeyByZXR1cm4gdHlwZW9mIHYucGFyZW50ID09PSAndW5kZWZpbmVkJzsgfSk7XHJcbiAgICAgICAgdGhpcy5fcm9vdEdyb3VwLmdyb3VwcyA9IHRoaXMuX2dyb3Vwcy5maWx0ZXIoZnVuY3Rpb24gKGcpIHsgcmV0dXJuIHR5cGVvZiBnLnBhcmVudCA9PT0gJ3VuZGVmaW5lZCc7IH0pO1xyXG4gICAgICAgIHJldHVybiB0aGlzO1xyXG4gICAgfTtcclxuICAgIExheW91dC5wcm90b3R5cGUucG93ZXJHcmFwaEdyb3VwcyA9IGZ1bmN0aW9uIChmKSB7XHJcbiAgICAgICAgdmFyIGcgPSBwb3dlcmdyYXBoLmdldEdyb3Vwcyh0aGlzLl9ub2RlcywgdGhpcy5fbGlua3MsIHRoaXMubGlua0FjY2Vzc29yLCB0aGlzLl9yb290R3JvdXApO1xyXG4gICAgICAgIHRoaXMuZ3JvdXBzKGcuZ3JvdXBzKTtcclxuICAgICAgICBmKGcpO1xyXG4gICAgICAgIHJldHVybiB0aGlzO1xyXG4gICAgfTtcclxuICAgIExheW91dC5wcm90b3R5cGUuYXZvaWRPdmVybGFwcyA9IGZ1bmN0aW9uICh2KSB7XHJcbiAgICAgICAgaWYgKCFhcmd1bWVudHMubGVuZ3RoKVxyXG4gICAgICAgICAgICByZXR1cm4gdGhpcy5fYXZvaWRPdmVybGFwcztcclxuICAgICAgICB0aGlzLl9hdm9pZE92ZXJsYXBzID0gdjtcclxuICAgICAgICByZXR1cm4gdGhpcztcclxuICAgIH07XHJcbiAgICBMYXlvdXQucHJvdG90eXBlLmhhbmRsZURpc2Nvbm5lY3RlZCA9IGZ1bmN0aW9uICh2KSB7XHJcbiAgICAgICAgaWYgKCFhcmd1bWVudHMubGVuZ3RoKVxyXG4gICAgICAgICAgICByZXR1cm4gdGhpcy5faGFuZGxlRGlzY29ubmVjdGVkO1xyXG4gICAgICAgIHRoaXMuX2hhbmRsZURpc2Nvbm5lY3RlZCA9IHY7XHJcbiAgICAgICAgcmV0dXJuIHRoaXM7XHJcbiAgICB9O1xyXG4gICAgTGF5b3V0LnByb3RvdHlwZS5mbG93TGF5b3V0ID0gZnVuY3Rpb24gKGF4aXMsIG1pblNlcGFyYXRpb24pIHtcclxuICAgICAgICBpZiAoIWFyZ3VtZW50cy5sZW5ndGgpXHJcbiAgICAgICAgICAgIGF4aXMgPSAneSc7XHJcbiAgICAgICAgdGhpcy5fZGlyZWN0ZWRMaW5rQ29uc3RyYWludHMgPSB7XHJcbiAgICAgICAgICAgIGF4aXM6IGF4aXMsXHJcbiAgICAgICAgICAgIGdldE1pblNlcGFyYXRpb246IHR5cGVvZiBtaW5TZXBhcmF0aW9uID09PSAnbnVtYmVyJyA/IGZ1bmN0aW9uICgpIHsgcmV0dXJuIG1pblNlcGFyYXRpb247IH0gOiBtaW5TZXBhcmF0aW9uXHJcbiAgICAgICAgfTtcclxuICAgICAgICByZXR1cm4gdGhpcztcclxuICAgIH07XHJcbiAgICBMYXlvdXQucHJvdG90eXBlLmxpbmtzID0gZnVuY3Rpb24gKHgpIHtcclxuICAgICAgICBpZiAoIWFyZ3VtZW50cy5sZW5ndGgpXHJcbiAgICAgICAgICAgIHJldHVybiB0aGlzLl9saW5rcztcclxuICAgICAgICB0aGlzLl9saW5rcyA9IHg7XHJcbiAgICAgICAgcmV0dXJuIHRoaXM7XHJcbiAgICB9O1xyXG4gICAgTGF5b3V0LnByb3RvdHlwZS5jb25zdHJhaW50cyA9IGZ1bmN0aW9uIChjKSB7XHJcbiAgICAgICAgaWYgKCFhcmd1bWVudHMubGVuZ3RoKVxyXG4gICAgICAgICAgICByZXR1cm4gdGhpcy5fY29uc3RyYWludHM7XHJcbiAgICAgICAgdGhpcy5fY29uc3RyYWludHMgPSBjO1xyXG4gICAgICAgIHJldHVybiB0aGlzO1xyXG4gICAgfTtcclxuICAgIExheW91dC5wcm90b3R5cGUuZGlzdGFuY2VNYXRyaXggPSBmdW5jdGlvbiAoZCkge1xyXG4gICAgICAgIGlmICghYXJndW1lbnRzLmxlbmd0aClcclxuICAgICAgICAgICAgcmV0dXJuIHRoaXMuX2Rpc3RhbmNlTWF0cml4O1xyXG4gICAgICAgIHRoaXMuX2Rpc3RhbmNlTWF0cml4ID0gZDtcclxuICAgICAgICByZXR1cm4gdGhpcztcclxuICAgIH07XHJcbiAgICBMYXlvdXQucHJvdG90eXBlLnNpemUgPSBmdW5jdGlvbiAoeCkge1xyXG4gICAgICAgIGlmICgheClcclxuICAgICAgICAgICAgcmV0dXJuIHRoaXMuX2NhbnZhc1NpemU7XHJcbiAgICAgICAgdGhpcy5fY2FudmFzU2l6ZSA9IHg7XHJcbiAgICAgICAgcmV0dXJuIHRoaXM7XHJcbiAgICB9O1xyXG4gICAgTGF5b3V0LnByb3RvdHlwZS5kZWZhdWx0Tm9kZVNpemUgPSBmdW5jdGlvbiAoeCkge1xyXG4gICAgICAgIGlmICgheClcclxuICAgICAgICAgICAgcmV0dXJuIHRoaXMuX2RlZmF1bHROb2RlU2l6ZTtcclxuICAgICAgICB0aGlzLl9kZWZhdWx0Tm9kZVNpemUgPSB4O1xyXG4gICAgICAgIHJldHVybiB0aGlzO1xyXG4gICAgfTtcclxuICAgIExheW91dC5wcm90b3R5cGUuZ3JvdXBDb21wYWN0bmVzcyA9IGZ1bmN0aW9uICh4KSB7XHJcbiAgICAgICAgaWYgKCF4KVxyXG4gICAgICAgICAgICByZXR1cm4gdGhpcy5fZ3JvdXBDb21wYWN0bmVzcztcclxuICAgICAgICB0aGlzLl9ncm91cENvbXBhY3RuZXNzID0geDtcclxuICAgICAgICByZXR1cm4gdGhpcztcclxuICAgIH07XHJcbiAgICBMYXlvdXQucHJvdG90eXBlLmxpbmtEaXN0YW5jZSA9IGZ1bmN0aW9uICh4KSB7XHJcbiAgICAgICAgaWYgKCF4KSB7XHJcbiAgICAgICAgICAgIHJldHVybiB0aGlzLl9saW5rRGlzdGFuY2U7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHRoaXMuX2xpbmtEaXN0YW5jZSA9IHR5cGVvZiB4ID09PSBcImZ1bmN0aW9uXCIgPyB4IDogK3g7XHJcbiAgICAgICAgdGhpcy5fbGlua0xlbmd0aENhbGN1bGF0b3IgPSBudWxsO1xyXG4gICAgICAgIHJldHVybiB0aGlzO1xyXG4gICAgfTtcclxuICAgIExheW91dC5wcm90b3R5cGUubGlua1R5cGUgPSBmdW5jdGlvbiAoZikge1xyXG4gICAgICAgIHRoaXMuX2xpbmtUeXBlID0gZjtcclxuICAgICAgICByZXR1cm4gdGhpcztcclxuICAgIH07XHJcbiAgICBMYXlvdXQucHJvdG90eXBlLmNvbnZlcmdlbmNlVGhyZXNob2xkID0gZnVuY3Rpb24gKHgpIHtcclxuICAgICAgICBpZiAoIXgpXHJcbiAgICAgICAgICAgIHJldHVybiB0aGlzLl90aHJlc2hvbGQ7XHJcbiAgICAgICAgdGhpcy5fdGhyZXNob2xkID0gdHlwZW9mIHggPT09IFwiZnVuY3Rpb25cIiA/IHggOiAreDtcclxuICAgICAgICByZXR1cm4gdGhpcztcclxuICAgIH07XHJcbiAgICBMYXlvdXQucHJvdG90eXBlLmFscGhhID0gZnVuY3Rpb24gKHgpIHtcclxuICAgICAgICBpZiAoIWFyZ3VtZW50cy5sZW5ndGgpXHJcbiAgICAgICAgICAgIHJldHVybiB0aGlzLl9hbHBoYTtcclxuICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgeCA9ICt4O1xyXG4gICAgICAgICAgICBpZiAodGhpcy5fYWxwaGEpIHtcclxuICAgICAgICAgICAgICAgIGlmICh4ID4gMClcclxuICAgICAgICAgICAgICAgICAgICB0aGlzLl9hbHBoYSA9IHg7XHJcbiAgICAgICAgICAgICAgICBlbHNlXHJcbiAgICAgICAgICAgICAgICAgICAgdGhpcy5fYWxwaGEgPSAwO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIGVsc2UgaWYgKHggPiAwKSB7XHJcbiAgICAgICAgICAgICAgICBpZiAoIXRoaXMuX3J1bm5pbmcpIHtcclxuICAgICAgICAgICAgICAgICAgICB0aGlzLl9ydW5uaW5nID0gdHJ1ZTtcclxuICAgICAgICAgICAgICAgICAgICB0aGlzLnRyaWdnZXIoeyB0eXBlOiBFdmVudFR5cGUuc3RhcnQsIGFscGhhOiB0aGlzLl9hbHBoYSA9IHggfSk7XHJcbiAgICAgICAgICAgICAgICAgICAgdGhpcy5raWNrKCk7XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgcmV0dXJuIHRoaXM7XHJcbiAgICAgICAgfVxyXG4gICAgfTtcclxuICAgIExheW91dC5wcm90b3R5cGUuZ2V0TGlua0xlbmd0aCA9IGZ1bmN0aW9uIChsaW5rKSB7XHJcbiAgICAgICAgcmV0dXJuIHR5cGVvZiB0aGlzLl9saW5rRGlzdGFuY2UgPT09IFwiZnVuY3Rpb25cIiA/ICsodGhpcy5fbGlua0Rpc3RhbmNlKGxpbmspKSA6IHRoaXMuX2xpbmtEaXN0YW5jZTtcclxuICAgIH07XHJcbiAgICBMYXlvdXQuc2V0TGlua0xlbmd0aCA9IGZ1bmN0aW9uIChsaW5rLCBsZW5ndGgpIHtcclxuICAgICAgICBsaW5rLmxlbmd0aCA9IGxlbmd0aDtcclxuICAgIH07XHJcbiAgICBMYXlvdXQucHJvdG90eXBlLmdldExpbmtUeXBlID0gZnVuY3Rpb24gKGxpbmspIHtcclxuICAgICAgICByZXR1cm4gdHlwZW9mIHRoaXMuX2xpbmtUeXBlID09PSBcImZ1bmN0aW9uXCIgPyB0aGlzLl9saW5rVHlwZShsaW5rKSA6IDA7XHJcbiAgICB9O1xyXG4gICAgTGF5b3V0LnByb3RvdHlwZS5zeW1tZXRyaWNEaWZmTGlua0xlbmd0aHMgPSBmdW5jdGlvbiAoaWRlYWxMZW5ndGgsIHcpIHtcclxuICAgICAgICB2YXIgX3RoaXMgPSB0aGlzO1xyXG4gICAgICAgIGlmICh3ID09PSB2b2lkIDApIHsgdyA9IDE7IH1cclxuICAgICAgICB0aGlzLmxpbmtEaXN0YW5jZShmdW5jdGlvbiAobCkgeyByZXR1cm4gaWRlYWxMZW5ndGggKiBsLmxlbmd0aDsgfSk7XHJcbiAgICAgICAgdGhpcy5fbGlua0xlbmd0aENhbGN1bGF0b3IgPSBmdW5jdGlvbiAoKSB7IHJldHVybiBsaW5rbGVuZ3Roc18xLnN5bW1ldHJpY0RpZmZMaW5rTGVuZ3RocyhfdGhpcy5fbGlua3MsIF90aGlzLmxpbmtBY2Nlc3Nvciwgdyk7IH07XHJcbiAgICAgICAgcmV0dXJuIHRoaXM7XHJcbiAgICB9O1xyXG4gICAgTGF5b3V0LnByb3RvdHlwZS5qYWNjYXJkTGlua0xlbmd0aHMgPSBmdW5jdGlvbiAoaWRlYWxMZW5ndGgsIHcpIHtcclxuICAgICAgICB2YXIgX3RoaXMgPSB0aGlzO1xyXG4gICAgICAgIGlmICh3ID09PSB2b2lkIDApIHsgdyA9IDE7IH1cclxuICAgICAgICB0aGlzLmxpbmtEaXN0YW5jZShmdW5jdGlvbiAobCkgeyByZXR1cm4gaWRlYWxMZW5ndGggKiBsLmxlbmd0aDsgfSk7XHJcbiAgICAgICAgdGhpcy5fbGlua0xlbmd0aENhbGN1bGF0b3IgPSBmdW5jdGlvbiAoKSB7IHJldHVybiBsaW5rbGVuZ3Roc18xLmphY2NhcmRMaW5rTGVuZ3RocyhfdGhpcy5fbGlua3MsIF90aGlzLmxpbmtBY2Nlc3Nvciwgdyk7IH07XHJcbiAgICAgICAgcmV0dXJuIHRoaXM7XHJcbiAgICB9O1xyXG4gICAgTGF5b3V0LnByb3RvdHlwZS5zdGFydCA9IGZ1bmN0aW9uIChpbml0aWFsVW5jb25zdHJhaW5lZEl0ZXJhdGlvbnMsIGluaXRpYWxVc2VyQ29uc3RyYWludEl0ZXJhdGlvbnMsIGluaXRpYWxBbGxDb25zdHJhaW50c0l0ZXJhdGlvbnMsIGdyaWRTbmFwSXRlcmF0aW9ucywga2VlcFJ1bm5pbmcsIGNlbnRlckdyYXBoKSB7XHJcbiAgICAgICAgdmFyIF90aGlzID0gdGhpcztcclxuICAgICAgICBpZiAoaW5pdGlhbFVuY29uc3RyYWluZWRJdGVyYXRpb25zID09PSB2b2lkIDApIHsgaW5pdGlhbFVuY29uc3RyYWluZWRJdGVyYXRpb25zID0gMDsgfVxyXG4gICAgICAgIGlmIChpbml0aWFsVXNlckNvbnN0cmFpbnRJdGVyYXRpb25zID09PSB2b2lkIDApIHsgaW5pdGlhbFVzZXJDb25zdHJhaW50SXRlcmF0aW9ucyA9IDA7IH1cclxuICAgICAgICBpZiAoaW5pdGlhbEFsbENvbnN0cmFpbnRzSXRlcmF0aW9ucyA9PT0gdm9pZCAwKSB7IGluaXRpYWxBbGxDb25zdHJhaW50c0l0ZXJhdGlvbnMgPSAwOyB9XHJcbiAgICAgICAgaWYgKGdyaWRTbmFwSXRlcmF0aW9ucyA9PT0gdm9pZCAwKSB7IGdyaWRTbmFwSXRlcmF0aW9ucyA9IDA7IH1cclxuICAgICAgICBpZiAoa2VlcFJ1bm5pbmcgPT09IHZvaWQgMCkgeyBrZWVwUnVubmluZyA9IHRydWU7IH1cclxuICAgICAgICBpZiAoY2VudGVyR3JhcGggPT09IHZvaWQgMCkgeyBjZW50ZXJHcmFwaCA9IHRydWU7IH1cclxuICAgICAgICB2YXIgaSwgaiwgbiA9IHRoaXMubm9kZXMoKS5sZW5ndGgsIE4gPSBuICsgMiAqIHRoaXMuX2dyb3Vwcy5sZW5ndGgsIG0gPSB0aGlzLl9saW5rcy5sZW5ndGgsIHcgPSB0aGlzLl9jYW52YXNTaXplWzBdLCBoID0gdGhpcy5fY2FudmFzU2l6ZVsxXTtcclxuICAgICAgICB2YXIgeCA9IG5ldyBBcnJheShOKSwgeSA9IG5ldyBBcnJheShOKTtcclxuICAgICAgICB2YXIgRyA9IG51bGw7XHJcbiAgICAgICAgdmFyIGFvID0gdGhpcy5fYXZvaWRPdmVybGFwcztcclxuICAgICAgICB0aGlzLl9ub2Rlcy5mb3JFYWNoKGZ1bmN0aW9uICh2LCBpKSB7XHJcbiAgICAgICAgICAgIHYuaW5kZXggPSBpO1xyXG4gICAgICAgICAgICBpZiAodHlwZW9mIHYueCA9PT0gJ3VuZGVmaW5lZCcpIHtcclxuICAgICAgICAgICAgICAgIHYueCA9IHcgLyAyLCB2LnkgPSBoIC8gMjtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICB4W2ldID0gdi54LCB5W2ldID0gdi55O1xyXG4gICAgICAgIH0pO1xyXG4gICAgICAgIGlmICh0aGlzLl9saW5rTGVuZ3RoQ2FsY3VsYXRvcilcclxuICAgICAgICAgICAgdGhpcy5fbGlua0xlbmd0aENhbGN1bGF0b3IoKTtcclxuICAgICAgICB2YXIgZGlzdGFuY2VzO1xyXG4gICAgICAgIGlmICh0aGlzLl9kaXN0YW5jZU1hdHJpeCkge1xyXG4gICAgICAgICAgICBkaXN0YW5jZXMgPSB0aGlzLl9kaXN0YW5jZU1hdHJpeDtcclxuICAgICAgICB9XHJcbiAgICAgICAgZWxzZSB7XHJcbiAgICAgICAgICAgIGRpc3RhbmNlcyA9IChuZXcgc2hvcnRlc3RwYXRoc18xLkNhbGN1bGF0b3IoTiwgdGhpcy5fbGlua3MsIExheW91dC5nZXRTb3VyY2VJbmRleCwgTGF5b3V0LmdldFRhcmdldEluZGV4LCBmdW5jdGlvbiAobCkgeyByZXR1cm4gX3RoaXMuZ2V0TGlua0xlbmd0aChsKTsgfSkpLkRpc3RhbmNlTWF0cml4KCk7XHJcbiAgICAgICAgICAgIEcgPSBkZXNjZW50XzEuRGVzY2VudC5jcmVhdGVTcXVhcmVNYXRyaXgoTiwgZnVuY3Rpb24gKCkgeyByZXR1cm4gMjsgfSk7XHJcbiAgICAgICAgICAgIHRoaXMuX2xpbmtzLmZvckVhY2goZnVuY3Rpb24gKGwpIHtcclxuICAgICAgICAgICAgICAgIGlmICh0eXBlb2YgbC5zb3VyY2UgPT0gXCJudW1iZXJcIilcclxuICAgICAgICAgICAgICAgICAgICBsLnNvdXJjZSA9IF90aGlzLl9ub2Rlc1tsLnNvdXJjZV07XHJcbiAgICAgICAgICAgICAgICBpZiAodHlwZW9mIGwudGFyZ2V0ID09IFwibnVtYmVyXCIpXHJcbiAgICAgICAgICAgICAgICAgICAgbC50YXJnZXQgPSBfdGhpcy5fbm9kZXNbbC50YXJnZXRdO1xyXG4gICAgICAgICAgICB9KTtcclxuICAgICAgICAgICAgdGhpcy5fbGlua3MuZm9yRWFjaChmdW5jdGlvbiAoZSkge1xyXG4gICAgICAgICAgICAgICAgdmFyIHUgPSBMYXlvdXQuZ2V0U291cmNlSW5kZXgoZSksIHYgPSBMYXlvdXQuZ2V0VGFyZ2V0SW5kZXgoZSk7XHJcbiAgICAgICAgICAgICAgICBHW3VdW3ZdID0gR1t2XVt1XSA9IGUud2VpZ2h0IHx8IDE7XHJcbiAgICAgICAgICAgIH0pO1xyXG4gICAgICAgIH1cclxuICAgICAgICB2YXIgRCA9IGRlc2NlbnRfMS5EZXNjZW50LmNyZWF0ZVNxdWFyZU1hdHJpeChOLCBmdW5jdGlvbiAoaSwgaikge1xyXG4gICAgICAgICAgICByZXR1cm4gZGlzdGFuY2VzW2ldW2pdO1xyXG4gICAgICAgIH0pO1xyXG4gICAgICAgIGlmICh0aGlzLl9yb290R3JvdXAgJiYgdHlwZW9mIHRoaXMuX3Jvb3RHcm91cC5ncm91cHMgIT09ICd1bmRlZmluZWQnKSB7XHJcbiAgICAgICAgICAgIHZhciBpID0gbjtcclxuICAgICAgICAgICAgdmFyIGFkZEF0dHJhY3Rpb24gPSBmdW5jdGlvbiAoaSwgaiwgc3RyZW5ndGgsIGlkZWFsRGlzdGFuY2UpIHtcclxuICAgICAgICAgICAgICAgIEdbaV1bal0gPSBHW2pdW2ldID0gc3RyZW5ndGg7XHJcbiAgICAgICAgICAgICAgICBEW2ldW2pdID0gRFtqXVtpXSA9IGlkZWFsRGlzdGFuY2U7XHJcbiAgICAgICAgICAgIH07XHJcbiAgICAgICAgICAgIHRoaXMuX2dyb3Vwcy5mb3JFYWNoKGZ1bmN0aW9uIChnKSB7XHJcbiAgICAgICAgICAgICAgICBhZGRBdHRyYWN0aW9uKGksIGkgKyAxLCBfdGhpcy5fZ3JvdXBDb21wYWN0bmVzcywgMC4xKTtcclxuICAgICAgICAgICAgICAgIHhbaV0gPSAwLCB5W2krK10gPSAwO1xyXG4gICAgICAgICAgICAgICAgeFtpXSA9IDAsIHlbaSsrXSA9IDA7XHJcbiAgICAgICAgICAgIH0pO1xyXG4gICAgICAgIH1cclxuICAgICAgICBlbHNlXHJcbiAgICAgICAgICAgIHRoaXMuX3Jvb3RHcm91cCA9IHsgbGVhdmVzOiB0aGlzLl9ub2RlcywgZ3JvdXBzOiBbXSB9O1xyXG4gICAgICAgIHZhciBjdXJDb25zdHJhaW50cyA9IHRoaXMuX2NvbnN0cmFpbnRzIHx8IFtdO1xyXG4gICAgICAgIGlmICh0aGlzLl9kaXJlY3RlZExpbmtDb25zdHJhaW50cykge1xyXG4gICAgICAgICAgICB0aGlzLmxpbmtBY2Nlc3Nvci5nZXRNaW5TZXBhcmF0aW9uID0gdGhpcy5fZGlyZWN0ZWRMaW5rQ29uc3RyYWludHMuZ2V0TWluU2VwYXJhdGlvbjtcclxuICAgICAgICAgICAgY3VyQ29uc3RyYWludHMgPSBjdXJDb25zdHJhaW50cy5jb25jYXQobGlua2xlbmd0aHNfMS5nZW5lcmF0ZURpcmVjdGVkRWRnZUNvbnN0cmFpbnRzKG4sIHRoaXMuX2xpbmtzLCB0aGlzLl9kaXJlY3RlZExpbmtDb25zdHJhaW50cy5heGlzLCAodGhpcy5saW5rQWNjZXNzb3IpKSk7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHRoaXMuYXZvaWRPdmVybGFwcyhmYWxzZSk7XHJcbiAgICAgICAgdGhpcy5fZGVzY2VudCA9IG5ldyBkZXNjZW50XzEuRGVzY2VudChbeCwgeV0sIEQpO1xyXG4gICAgICAgIHRoaXMuX2Rlc2NlbnQubG9ja3MuY2xlYXIoKTtcclxuICAgICAgICBmb3IgKHZhciBpID0gMDsgaSA8IG47ICsraSkge1xyXG4gICAgICAgICAgICB2YXIgbyA9IHRoaXMuX25vZGVzW2ldO1xyXG4gICAgICAgICAgICBpZiAoby5maXhlZCkge1xyXG4gICAgICAgICAgICAgICAgby5weCA9IG8ueDtcclxuICAgICAgICAgICAgICAgIG8ucHkgPSBvLnk7XHJcbiAgICAgICAgICAgICAgICB2YXIgcCA9IFtvLngsIG8ueV07XHJcbiAgICAgICAgICAgICAgICB0aGlzLl9kZXNjZW50LmxvY2tzLmFkZChpLCBwKTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH1cclxuICAgICAgICB0aGlzLl9kZXNjZW50LnRocmVzaG9sZCA9IHRoaXMuX3RocmVzaG9sZDtcclxuICAgICAgICB0aGlzLmluaXRpYWxMYXlvdXQoaW5pdGlhbFVuY29uc3RyYWluZWRJdGVyYXRpb25zLCB4LCB5KTtcclxuICAgICAgICBpZiAoY3VyQ29uc3RyYWludHMubGVuZ3RoID4gMClcclxuICAgICAgICAgICAgdGhpcy5fZGVzY2VudC5wcm9qZWN0ID0gbmV3IHJlY3RhbmdsZV8xLlByb2plY3Rpb24odGhpcy5fbm9kZXMsIHRoaXMuX2dyb3VwcywgdGhpcy5fcm9vdEdyb3VwLCBjdXJDb25zdHJhaW50cykucHJvamVjdEZ1bmN0aW9ucygpO1xyXG4gICAgICAgIHRoaXMuX2Rlc2NlbnQucnVuKGluaXRpYWxVc2VyQ29uc3RyYWludEl0ZXJhdGlvbnMpO1xyXG4gICAgICAgIHRoaXMuc2VwYXJhdGVPdmVybGFwcGluZ0NvbXBvbmVudHModywgaCwgY2VudGVyR3JhcGgpO1xyXG4gICAgICAgIHRoaXMuYXZvaWRPdmVybGFwcyhhbyk7XHJcbiAgICAgICAgaWYgKGFvKSB7XHJcbiAgICAgICAgICAgIHRoaXMuX25vZGVzLmZvckVhY2goZnVuY3Rpb24gKHYsIGkpIHsgdi54ID0geFtpXSwgdi55ID0geVtpXTsgfSk7XHJcbiAgICAgICAgICAgIHRoaXMuX2Rlc2NlbnQucHJvamVjdCA9IG5ldyByZWN0YW5nbGVfMS5Qcm9qZWN0aW9uKHRoaXMuX25vZGVzLCB0aGlzLl9ncm91cHMsIHRoaXMuX3Jvb3RHcm91cCwgY3VyQ29uc3RyYWludHMsIHRydWUpLnByb2plY3RGdW5jdGlvbnMoKTtcclxuICAgICAgICAgICAgdGhpcy5fbm9kZXMuZm9yRWFjaChmdW5jdGlvbiAodiwgaSkgeyB4W2ldID0gdi54LCB5W2ldID0gdi55OyB9KTtcclxuICAgICAgICB9XHJcbiAgICAgICAgdGhpcy5fZGVzY2VudC5HID0gRztcclxuICAgICAgICB0aGlzLl9kZXNjZW50LnJ1bihpbml0aWFsQWxsQ29uc3RyYWludHNJdGVyYXRpb25zKTtcclxuICAgICAgICBpZiAoZ3JpZFNuYXBJdGVyYXRpb25zKSB7XHJcbiAgICAgICAgICAgIHRoaXMuX2Rlc2NlbnQuc25hcFN0cmVuZ3RoID0gMTAwMDtcclxuICAgICAgICAgICAgdGhpcy5fZGVzY2VudC5zbmFwR3JpZFNpemUgPSB0aGlzLl9ub2Rlc1swXS53aWR0aDtcclxuICAgICAgICAgICAgdGhpcy5fZGVzY2VudC5udW1HcmlkU25hcE5vZGVzID0gbjtcclxuICAgICAgICAgICAgdGhpcy5fZGVzY2VudC5zY2FsZVNuYXBCeU1heEggPSBuICE9IE47XHJcbiAgICAgICAgICAgIHZhciBHMCA9IGRlc2NlbnRfMS5EZXNjZW50LmNyZWF0ZVNxdWFyZU1hdHJpeChOLCBmdW5jdGlvbiAoaSwgaikge1xyXG4gICAgICAgICAgICAgICAgaWYgKGkgPj0gbiB8fCBqID49IG4pXHJcbiAgICAgICAgICAgICAgICAgICAgcmV0dXJuIEdbaV1bal07XHJcbiAgICAgICAgICAgICAgICByZXR1cm4gMDtcclxuICAgICAgICAgICAgfSk7XHJcbiAgICAgICAgICAgIHRoaXMuX2Rlc2NlbnQuRyA9IEcwO1xyXG4gICAgICAgICAgICB0aGlzLl9kZXNjZW50LnJ1bihncmlkU25hcEl0ZXJhdGlvbnMpO1xyXG4gICAgICAgIH1cclxuICAgICAgICB0aGlzLnVwZGF0ZU5vZGVQb3NpdGlvbnMoKTtcclxuICAgICAgICB0aGlzLnNlcGFyYXRlT3ZlcmxhcHBpbmdDb21wb25lbnRzKHcsIGgsIGNlbnRlckdyYXBoKTtcclxuICAgICAgICByZXR1cm4ga2VlcFJ1bm5pbmcgPyB0aGlzLnJlc3VtZSgpIDogdGhpcztcclxuICAgIH07XHJcbiAgICBMYXlvdXQucHJvdG90eXBlLmluaXRpYWxMYXlvdXQgPSBmdW5jdGlvbiAoaXRlcmF0aW9ucywgeCwgeSkge1xyXG4gICAgICAgIGlmICh0aGlzLl9ncm91cHMubGVuZ3RoID4gMCAmJiBpdGVyYXRpb25zID4gMCkge1xyXG4gICAgICAgICAgICB2YXIgbiA9IHRoaXMuX25vZGVzLmxlbmd0aDtcclxuICAgICAgICAgICAgdmFyIGVkZ2VzID0gdGhpcy5fbGlua3MubWFwKGZ1bmN0aW9uIChlKSB7IHJldHVybiAoeyBzb3VyY2U6IGUuc291cmNlLmluZGV4LCB0YXJnZXQ6IGUudGFyZ2V0LmluZGV4IH0pOyB9KTtcclxuICAgICAgICAgICAgdmFyIHZzID0gdGhpcy5fbm9kZXMubWFwKGZ1bmN0aW9uICh2KSB7IHJldHVybiAoeyBpbmRleDogdi5pbmRleCB9KTsgfSk7XHJcbiAgICAgICAgICAgIHRoaXMuX2dyb3Vwcy5mb3JFYWNoKGZ1bmN0aW9uIChnLCBpKSB7XHJcbiAgICAgICAgICAgICAgICB2cy5wdXNoKHsgaW5kZXg6IGcuaW5kZXggPSBuICsgaSB9KTtcclxuICAgICAgICAgICAgfSk7XHJcbiAgICAgICAgICAgIHRoaXMuX2dyb3Vwcy5mb3JFYWNoKGZ1bmN0aW9uIChnLCBpKSB7XHJcbiAgICAgICAgICAgICAgICBpZiAodHlwZW9mIGcubGVhdmVzICE9PSAndW5kZWZpbmVkJylcclxuICAgICAgICAgICAgICAgICAgICBnLmxlYXZlcy5mb3JFYWNoKGZ1bmN0aW9uICh2KSB7IHJldHVybiBlZGdlcy5wdXNoKHsgc291cmNlOiBnLmluZGV4LCB0YXJnZXQ6IHYuaW5kZXggfSk7IH0pO1xyXG4gICAgICAgICAgICAgICAgaWYgKHR5cGVvZiBnLmdyb3VwcyAhPT0gJ3VuZGVmaW5lZCcpXHJcbiAgICAgICAgICAgICAgICAgICAgZy5ncm91cHMuZm9yRWFjaChmdW5jdGlvbiAoZ2cpIHsgcmV0dXJuIGVkZ2VzLnB1c2goeyBzb3VyY2U6IGcuaW5kZXgsIHRhcmdldDogZ2cuaW5kZXggfSk7IH0pO1xyXG4gICAgICAgICAgICB9KTtcclxuICAgICAgICAgICAgbmV3IExheW91dCgpXHJcbiAgICAgICAgICAgICAgICAuc2l6ZSh0aGlzLnNpemUoKSlcclxuICAgICAgICAgICAgICAgIC5ub2Rlcyh2cylcclxuICAgICAgICAgICAgICAgIC5saW5rcyhlZGdlcylcclxuICAgICAgICAgICAgICAgIC5hdm9pZE92ZXJsYXBzKGZhbHNlKVxyXG4gICAgICAgICAgICAgICAgLmxpbmtEaXN0YW5jZSh0aGlzLmxpbmtEaXN0YW5jZSgpKVxyXG4gICAgICAgICAgICAgICAgLnN5bW1ldHJpY0RpZmZMaW5rTGVuZ3Rocyg1KVxyXG4gICAgICAgICAgICAgICAgLmNvbnZlcmdlbmNlVGhyZXNob2xkKDFlLTQpXHJcbiAgICAgICAgICAgICAgICAuc3RhcnQoaXRlcmF0aW9ucywgMCwgMCwgMCwgZmFsc2UpO1xyXG4gICAgICAgICAgICB0aGlzLl9ub2Rlcy5mb3JFYWNoKGZ1bmN0aW9uICh2KSB7XHJcbiAgICAgICAgICAgICAgICB4W3YuaW5kZXhdID0gdnNbdi5pbmRleF0ueDtcclxuICAgICAgICAgICAgICAgIHlbdi5pbmRleF0gPSB2c1t2LmluZGV4XS55O1xyXG4gICAgICAgICAgICB9KTtcclxuICAgICAgICB9XHJcbiAgICAgICAgZWxzZSB7XHJcbiAgICAgICAgICAgIHRoaXMuX2Rlc2NlbnQucnVuKGl0ZXJhdGlvbnMpO1xyXG4gICAgICAgIH1cclxuICAgIH07XHJcbiAgICBMYXlvdXQucHJvdG90eXBlLnNlcGFyYXRlT3ZlcmxhcHBpbmdDb21wb25lbnRzID0gZnVuY3Rpb24gKHdpZHRoLCBoZWlnaHQsIGNlbnRlckdyYXBoKSB7XHJcbiAgICAgICAgdmFyIF90aGlzID0gdGhpcztcclxuICAgICAgICBpZiAoY2VudGVyR3JhcGggPT09IHZvaWQgMCkgeyBjZW50ZXJHcmFwaCA9IHRydWU7IH1cclxuICAgICAgICBpZiAoIXRoaXMuX2Rpc3RhbmNlTWF0cml4ICYmIHRoaXMuX2hhbmRsZURpc2Nvbm5lY3RlZCkge1xyXG4gICAgICAgICAgICB2YXIgeF8xID0gdGhpcy5fZGVzY2VudC54WzBdLCB5XzEgPSB0aGlzLl9kZXNjZW50LnhbMV07XHJcbiAgICAgICAgICAgIHRoaXMuX25vZGVzLmZvckVhY2goZnVuY3Rpb24gKHYsIGkpIHsgdi54ID0geF8xW2ldLCB2LnkgPSB5XzFbaV07IH0pO1xyXG4gICAgICAgICAgICB2YXIgZ3JhcGhzID0gaGFuZGxlZGlzY29ubmVjdGVkXzEuc2VwYXJhdGVHcmFwaHModGhpcy5fbm9kZXMsIHRoaXMuX2xpbmtzKTtcclxuICAgICAgICAgICAgaGFuZGxlZGlzY29ubmVjdGVkXzEuYXBwbHlQYWNraW5nKGdyYXBocywgd2lkdGgsIGhlaWdodCwgdGhpcy5fZGVmYXVsdE5vZGVTaXplLCAxLCBjZW50ZXJHcmFwaCk7XHJcbiAgICAgICAgICAgIHRoaXMuX25vZGVzLmZvckVhY2goZnVuY3Rpb24gKHYsIGkpIHtcclxuICAgICAgICAgICAgICAgIF90aGlzLl9kZXNjZW50LnhbMF1baV0gPSB2LngsIF90aGlzLl9kZXNjZW50LnhbMV1baV0gPSB2Lnk7XHJcbiAgICAgICAgICAgICAgICBpZiAodi5ib3VuZHMpIHtcclxuICAgICAgICAgICAgICAgICAgICB2LmJvdW5kcy5zZXRYQ2VudHJlKHYueCk7XHJcbiAgICAgICAgICAgICAgICAgICAgdi5ib3VuZHMuc2V0WUNlbnRyZSh2LnkpO1xyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICB9KTtcclxuICAgICAgICB9XHJcbiAgICB9O1xyXG4gICAgTGF5b3V0LnByb3RvdHlwZS5yZXN1bWUgPSBmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgcmV0dXJuIHRoaXMuYWxwaGEoMC4xKTtcclxuICAgIH07XHJcbiAgICBMYXlvdXQucHJvdG90eXBlLnN0b3AgPSBmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgcmV0dXJuIHRoaXMuYWxwaGEoMCk7XHJcbiAgICB9O1xyXG4gICAgTGF5b3V0LnByb3RvdHlwZS5wcmVwYXJlRWRnZVJvdXRpbmcgPSBmdW5jdGlvbiAobm9kZU1hcmdpbikge1xyXG4gICAgICAgIGlmIChub2RlTWFyZ2luID09PSB2b2lkIDApIHsgbm9kZU1hcmdpbiA9IDA7IH1cclxuICAgICAgICB0aGlzLl92aXNpYmlsaXR5R3JhcGggPSBuZXcgZ2VvbV8xLlRhbmdlbnRWaXNpYmlsaXR5R3JhcGgodGhpcy5fbm9kZXMubWFwKGZ1bmN0aW9uICh2KSB7XHJcbiAgICAgICAgICAgIHJldHVybiB2LmJvdW5kcy5pbmZsYXRlKC1ub2RlTWFyZ2luKS52ZXJ0aWNlcygpO1xyXG4gICAgICAgIH0pKTtcclxuICAgIH07XHJcbiAgICBMYXlvdXQucHJvdG90eXBlLnJvdXRlRWRnZSA9IGZ1bmN0aW9uIChlZGdlLCBhaCwgZHJhdykge1xyXG4gICAgICAgIGlmIChhaCA9PT0gdm9pZCAwKSB7IGFoID0gNTsgfVxyXG4gICAgICAgIHZhciBsaW5lRGF0YSA9IFtdO1xyXG4gICAgICAgIHZhciB2ZzIgPSBuZXcgZ2VvbV8xLlRhbmdlbnRWaXNpYmlsaXR5R3JhcGgodGhpcy5fdmlzaWJpbGl0eUdyYXBoLlAsIHsgVjogdGhpcy5fdmlzaWJpbGl0eUdyYXBoLlYsIEU6IHRoaXMuX3Zpc2liaWxpdHlHcmFwaC5FIH0pLCBwb3J0MSA9IHsgeDogZWRnZS5zb3VyY2UueCwgeTogZWRnZS5zb3VyY2UueSB9LCBwb3J0MiA9IHsgeDogZWRnZS50YXJnZXQueCwgeTogZWRnZS50YXJnZXQueSB9LCBzdGFydCA9IHZnMi5hZGRQb2ludChwb3J0MSwgZWRnZS5zb3VyY2UuaW5kZXgpLCBlbmQgPSB2ZzIuYWRkUG9pbnQocG9ydDIsIGVkZ2UudGFyZ2V0LmluZGV4KTtcclxuICAgICAgICB2ZzIuYWRkRWRnZUlmVmlzaWJsZShwb3J0MSwgcG9ydDIsIGVkZ2Uuc291cmNlLmluZGV4LCBlZGdlLnRhcmdldC5pbmRleCk7XHJcbiAgICAgICAgaWYgKHR5cGVvZiBkcmF3ICE9PSAndW5kZWZpbmVkJykge1xyXG4gICAgICAgICAgICBkcmF3KHZnMik7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHZhciBzb3VyY2VJbmQgPSBmdW5jdGlvbiAoZSkgeyByZXR1cm4gZS5zb3VyY2UuaWQ7IH0sIHRhcmdldEluZCA9IGZ1bmN0aW9uIChlKSB7IHJldHVybiBlLnRhcmdldC5pZDsgfSwgbGVuZ3RoID0gZnVuY3Rpb24gKGUpIHsgcmV0dXJuIGUubGVuZ3RoKCk7IH0sIHNwQ2FsYyA9IG5ldyBzaG9ydGVzdHBhdGhzXzEuQ2FsY3VsYXRvcih2ZzIuVi5sZW5ndGgsIHZnMi5FLCBzb3VyY2VJbmQsIHRhcmdldEluZCwgbGVuZ3RoKSwgc2hvcnRlc3RQYXRoID0gc3BDYWxjLlBhdGhGcm9tTm9kZVRvTm9kZShzdGFydC5pZCwgZW5kLmlkKTtcclxuICAgICAgICBpZiAoc2hvcnRlc3RQYXRoLmxlbmd0aCA9PT0gMSB8fCBzaG9ydGVzdFBhdGgubGVuZ3RoID09PSB2ZzIuVi5sZW5ndGgpIHtcclxuICAgICAgICAgICAgdmFyIHJvdXRlID0gcmVjdGFuZ2xlXzEubWFrZUVkZ2VCZXR3ZWVuKGVkZ2Uuc291cmNlLmlubmVyQm91bmRzLCBlZGdlLnRhcmdldC5pbm5lckJvdW5kcywgYWgpO1xyXG4gICAgICAgICAgICBsaW5lRGF0YSA9IFtyb3V0ZS5zb3VyY2VJbnRlcnNlY3Rpb24sIHJvdXRlLmFycm93U3RhcnRdO1xyXG4gICAgICAgIH1cclxuICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgdmFyIG4gPSBzaG9ydGVzdFBhdGgubGVuZ3RoIC0gMiwgcCA9IHZnMi5WW3Nob3J0ZXN0UGF0aFtuXV0ucCwgcSA9IHZnMi5WW3Nob3J0ZXN0UGF0aFswXV0ucCwgbGluZURhdGEgPSBbZWRnZS5zb3VyY2UuaW5uZXJCb3VuZHMucmF5SW50ZXJzZWN0aW9uKHAueCwgcC55KV07XHJcbiAgICAgICAgICAgIGZvciAodmFyIGkgPSBuOyBpID49IDA7IC0taSlcclxuICAgICAgICAgICAgICAgIGxpbmVEYXRhLnB1c2godmcyLlZbc2hvcnRlc3RQYXRoW2ldXS5wKTtcclxuICAgICAgICAgICAgbGluZURhdGEucHVzaChyZWN0YW5nbGVfMS5tYWtlRWRnZVRvKHEsIGVkZ2UudGFyZ2V0LmlubmVyQm91bmRzLCBhaCkpO1xyXG4gICAgICAgIH1cclxuICAgICAgICByZXR1cm4gbGluZURhdGE7XHJcbiAgICB9O1xyXG4gICAgTGF5b3V0LmdldFNvdXJjZUluZGV4ID0gZnVuY3Rpb24gKGUpIHtcclxuICAgICAgICByZXR1cm4gdHlwZW9mIGUuc291cmNlID09PSAnbnVtYmVyJyA/IGUuc291cmNlIDogZS5zb3VyY2UuaW5kZXg7XHJcbiAgICB9O1xyXG4gICAgTGF5b3V0LmdldFRhcmdldEluZGV4ID0gZnVuY3Rpb24gKGUpIHtcclxuICAgICAgICByZXR1cm4gdHlwZW9mIGUudGFyZ2V0ID09PSAnbnVtYmVyJyA/IGUudGFyZ2V0IDogZS50YXJnZXQuaW5kZXg7XHJcbiAgICB9O1xyXG4gICAgTGF5b3V0LmxpbmtJZCA9IGZ1bmN0aW9uIChlKSB7XHJcbiAgICAgICAgcmV0dXJuIExheW91dC5nZXRTb3VyY2VJbmRleChlKSArIFwiLVwiICsgTGF5b3V0LmdldFRhcmdldEluZGV4KGUpO1xyXG4gICAgfTtcclxuICAgIExheW91dC5kcmFnU3RhcnQgPSBmdW5jdGlvbiAoZCkge1xyXG4gICAgICAgIGlmIChpc0dyb3VwKGQpKSB7XHJcbiAgICAgICAgICAgIExheW91dC5zdG9yZU9mZnNldChkLCBMYXlvdXQuZHJhZ09yaWdpbihkKSk7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIGVsc2Uge1xyXG4gICAgICAgICAgICBMYXlvdXQuc3RvcE5vZGUoZCk7XHJcbiAgICAgICAgICAgIGQuZml4ZWQgfD0gMjtcclxuICAgICAgICB9XHJcbiAgICB9O1xyXG4gICAgTGF5b3V0LnN0b3BOb2RlID0gZnVuY3Rpb24gKHYpIHtcclxuICAgICAgICB2LnB4ID0gdi54O1xyXG4gICAgICAgIHYucHkgPSB2Lnk7XHJcbiAgICB9O1xyXG4gICAgTGF5b3V0LnN0b3JlT2Zmc2V0ID0gZnVuY3Rpb24gKGQsIG9yaWdpbikge1xyXG4gICAgICAgIGlmICh0eXBlb2YgZC5sZWF2ZXMgIT09ICd1bmRlZmluZWQnKSB7XHJcbiAgICAgICAgICAgIGQubGVhdmVzLmZvckVhY2goZnVuY3Rpb24gKHYpIHtcclxuICAgICAgICAgICAgICAgIHYuZml4ZWQgfD0gMjtcclxuICAgICAgICAgICAgICAgIExheW91dC5zdG9wTm9kZSh2KTtcclxuICAgICAgICAgICAgICAgIHYuX2RyYWdHcm91cE9mZnNldFggPSB2LnggLSBvcmlnaW4ueDtcclxuICAgICAgICAgICAgICAgIHYuX2RyYWdHcm91cE9mZnNldFkgPSB2LnkgLSBvcmlnaW4ueTtcclxuICAgICAgICAgICAgfSk7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIGlmICh0eXBlb2YgZC5ncm91cHMgIT09ICd1bmRlZmluZWQnKSB7XHJcbiAgICAgICAgICAgIGQuZ3JvdXBzLmZvckVhY2goZnVuY3Rpb24gKGcpIHsgcmV0dXJuIExheW91dC5zdG9yZU9mZnNldChnLCBvcmlnaW4pOyB9KTtcclxuICAgICAgICB9XHJcbiAgICB9O1xyXG4gICAgTGF5b3V0LmRyYWdPcmlnaW4gPSBmdW5jdGlvbiAoZCkge1xyXG4gICAgICAgIGlmIChpc0dyb3VwKGQpKSB7XHJcbiAgICAgICAgICAgIHJldHVybiB7XHJcbiAgICAgICAgICAgICAgICB4OiBkLmJvdW5kcy5jeCgpLFxyXG4gICAgICAgICAgICAgICAgeTogZC5ib3VuZHMuY3koKVxyXG4gICAgICAgICAgICB9O1xyXG4gICAgICAgIH1cclxuICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgcmV0dXJuIGQ7XHJcbiAgICAgICAgfVxyXG4gICAgfTtcclxuICAgIExheW91dC5kcmFnID0gZnVuY3Rpb24gKGQsIHBvc2l0aW9uKSB7XHJcbiAgICAgICAgaWYgKGlzR3JvdXAoZCkpIHtcclxuICAgICAgICAgICAgaWYgKHR5cGVvZiBkLmxlYXZlcyAhPT0gJ3VuZGVmaW5lZCcpIHtcclxuICAgICAgICAgICAgICAgIGQubGVhdmVzLmZvckVhY2goZnVuY3Rpb24gKHYpIHtcclxuICAgICAgICAgICAgICAgICAgICBkLmJvdW5kcy5zZXRYQ2VudHJlKHBvc2l0aW9uLngpO1xyXG4gICAgICAgICAgICAgICAgICAgIGQuYm91bmRzLnNldFlDZW50cmUocG9zaXRpb24ueSk7XHJcbiAgICAgICAgICAgICAgICAgICAgdi5weCA9IHYuX2RyYWdHcm91cE9mZnNldFggKyBwb3NpdGlvbi54O1xyXG4gICAgICAgICAgICAgICAgICAgIHYucHkgPSB2Ll9kcmFnR3JvdXBPZmZzZXRZICsgcG9zaXRpb24ueTtcclxuICAgICAgICAgICAgICAgIH0pO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIGlmICh0eXBlb2YgZC5ncm91cHMgIT09ICd1bmRlZmluZWQnKSB7XHJcbiAgICAgICAgICAgICAgICBkLmdyb3Vwcy5mb3JFYWNoKGZ1bmN0aW9uIChnKSB7IHJldHVybiBMYXlvdXQuZHJhZyhnLCBwb3NpdGlvbik7IH0pO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG4gICAgICAgIGVsc2Uge1xyXG4gICAgICAgICAgICBkLnB4ID0gcG9zaXRpb24ueDtcclxuICAgICAgICAgICAgZC5weSA9IHBvc2l0aW9uLnk7XHJcbiAgICAgICAgfVxyXG4gICAgfTtcclxuICAgIExheW91dC5kcmFnRW5kID0gZnVuY3Rpb24gKGQpIHtcclxuICAgICAgICBpZiAoaXNHcm91cChkKSkge1xyXG4gICAgICAgICAgICBpZiAodHlwZW9mIGQubGVhdmVzICE9PSAndW5kZWZpbmVkJykge1xyXG4gICAgICAgICAgICAgICAgZC5sZWF2ZXMuZm9yRWFjaChmdW5jdGlvbiAodikge1xyXG4gICAgICAgICAgICAgICAgICAgIExheW91dC5kcmFnRW5kKHYpO1xyXG4gICAgICAgICAgICAgICAgICAgIGRlbGV0ZSB2Ll9kcmFnR3JvdXBPZmZzZXRYO1xyXG4gICAgICAgICAgICAgICAgICAgIGRlbGV0ZSB2Ll9kcmFnR3JvdXBPZmZzZXRZO1xyXG4gICAgICAgICAgICAgICAgfSk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgaWYgKHR5cGVvZiBkLmdyb3VwcyAhPT0gJ3VuZGVmaW5lZCcpIHtcclxuICAgICAgICAgICAgICAgIGQuZ3JvdXBzLmZvckVhY2goTGF5b3V0LmRyYWdFbmQpO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG4gICAgICAgIGVsc2Uge1xyXG4gICAgICAgICAgICBkLmZpeGVkICY9IH42O1xyXG4gICAgICAgIH1cclxuICAgIH07XHJcbiAgICBMYXlvdXQubW91c2VPdmVyID0gZnVuY3Rpb24gKGQpIHtcclxuICAgICAgICBkLmZpeGVkIHw9IDQ7XHJcbiAgICAgICAgZC5weCA9IGQueCwgZC5weSA9IGQueTtcclxuICAgIH07XHJcbiAgICBMYXlvdXQubW91c2VPdXQgPSBmdW5jdGlvbiAoZCkge1xyXG4gICAgICAgIGQuZml4ZWQgJj0gfjQ7XHJcbiAgICB9O1xyXG4gICAgcmV0dXJuIExheW91dDtcclxufSgpKTtcclxuZXhwb3J0cy5MYXlvdXQgPSBMYXlvdXQ7XHJcbi8vIyBzb3VyY2VNYXBwaW5nVVJMPWRhdGE6YXBwbGljYXRpb24vanNvbjtiYXNlNjQsZXlKMlpYSnphVzl1SWpvekxDSm1hV3hsSWpvaWJHRjViM1YwTG1weklpd2ljMjkxY21ObFVtOXZkQ0k2SWlJc0luTnZkWEpqWlhNaU9sc2lMaTR2TGk0dlYyVmlRMjlzWVM5emNtTXZiR0Y1YjNWMExuUnpJbDBzSW01aGJXVnpJanBiWFN3aWJXRndjR2x1WjNNaU9pSTdPMEZCUVVFc2VVTkJRVEJETzBGQlF6RkRMRFpEUVVFclNEdEJRVU12U0N4eFEwRkJhVU03UVVGRGFrTXNlVU5CUVRoRk8wRkJRemxGTEdsRVFVRXdRenRCUVVNeFF5d3JRa0ZCZFVRN1FVRkRka1FzTWtSQlFXbEZPMEZCVHpkRUxFbEJRVmtzVTBGQk9FSTdRVUZCTVVNc1YwRkJXU3hUUVVGVE8wbEJRVWNzTWtOQlFVc3NRMEZCUVR0SlFVRkZMSGxEUVVGSkxFTkJRVUU3U1VGQlJTeDFRMEZCUnl4RFFVRkJPMEZCUVVNc1EwRkJReXhGUVVFNVFpeFRRVUZUTEVkQlFWUXNhVUpCUVZNc1MwRkJWQ3hwUWtGQlV5eFJRVUZ4UWp0QlFVRkJMRU5CUVVNN1FVRXJRek5ETEZOQlFWTXNUMEZCVHl4RFFVRkRMRU5CUVUwN1NVRkRia0lzVDBGQlR5eFBRVUZQTEVOQlFVTXNRMEZCUXl4TlFVRk5MRXRCUVVzc1YwRkJWeXhKUVVGSkxFOUJRVThzUTBGQlF5eERRVUZETEUxQlFVMHNTMEZCU3l4WFFVRlhMRU5CUVVNN1FVRkRPVVVzUTBGQlF6dEJRWGRDUkR0SlFVRkJPMUZCUVVFc2FVSkJhM2xDUXp0UlFXcDVRbGNzWjBKQlFWY3NSMEZCUnl4RFFVRkRMRU5CUVVNc1JVRkJSU3hEUVVGRExFTkJRVU1zUTBGQlF6dFJRVU55UWl4clFrRkJZU3hIUVVGNVF5eEZRVUZGTEVOQlFVTTdVVUZEZWtRc2NVSkJRV2RDTEVkQlFWY3NSVUZCUlN4RFFVRkRPMUZCUXpsQ0xEQkNRVUZ4UWl4SFFVRkhMRWxCUVVrc1EwRkJRenRSUVVNM1FpeGpRVUZUTEVkQlFVY3NTVUZCU1N4RFFVRkRPMUZCUTJwQ0xHMUNRVUZqTEVkQlFVY3NTMEZCU3l4RFFVRkRPMUZCUTNaQ0xIZENRVUZ0UWl4SFFVRkhMRWxCUVVrc1EwRkJRenRSUVVjelFpeGhRVUZSTEVkQlFVY3NTMEZCU3l4RFFVRkRPMUZCUTJwQ0xGZEJRVTBzUjBGQlJ5eEZRVUZGTEVOQlFVTTdVVUZEV2l4WlFVRlBMRWRCUVVjc1JVRkJSU3hEUVVGRE8xRkJRMklzWlVGQlZTeEhRVUZITEVsQlFVa3NRMEZCUXp0UlFVTnNRaXhYUVVGTkxFZEJRVEJDTEVWQlFVVXNRMEZCUXp0UlFVTnVReXhwUWtGQldTeEhRVUZITEVWQlFVVXNRMEZCUXp0UlFVTnNRaXh2UWtGQlpTeEhRVUZITEVsQlFVa3NRMEZCUXp0UlFVTjJRaXhoUVVGUkxFZEJRVmtzU1VGQlNTeERRVUZETzFGQlEzcENMRFpDUVVGM1FpeEhRVUZITEVsQlFVa3NRMEZCUXp0UlFVTm9ReXhsUVVGVkxFZEJRVWNzU1VGQlNTeERRVUZETzFGQlEyeENMSEZDUVVGblFpeEhRVUZITEVsQlFVa3NRMEZCUXp0UlFVTjRRaXh6UWtGQmFVSXNSMEZCUnl4SlFVRkpMRU5CUVVNN1VVRkhka0lzVlVGQlN5eEhRVUZITEVsQlFVa3NRMEZCUXp0UlFXdFdka0lzYVVKQlFWa3NSMEZCTWtJN1dVRkRia01zWTBGQll5eEZRVUZGTEUxQlFVMHNRMEZCUXl4alFVRmpPMWxCUTNKRExHTkJRV01zUlVGQlJTeE5RVUZOTEVOQlFVTXNZMEZCWXp0WlFVTnlReXhUUVVGVExFVkJRVVVzVFVGQlRTeERRVUZETEdGQlFXRTdXVUZETDBJc1QwRkJUeXhGUVVGRkxGVkJRVUVzUTBGQlF5eEpRVUZKTEU5QlFVRXNUMEZCVHl4TFFVRkpMRU5CUVVNc1UwRkJVeXhMUVVGTExGVkJRVlVzUTBGQlF5eERRVUZETEVOQlFVTXNTMEZCU1N4RFFVRkRMRk5CUVZNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RlFVRTFSQ3hEUVVFMFJEdFRRVU0zUlN4RFFVRkRPMGxCYldKT0xFTkJRVU03U1VGMGQwSlZMRzFDUVVGRkxFZEJRVlFzVlVGQlZTeERRVUZ4UWl4RlFVRkZMRkZCUVdsRE8xRkJSVGxFTEVsQlFVa3NRMEZCUXl4SlFVRkpMRU5CUVVNc1MwRkJTenRaUVVGRkxFbEJRVWtzUTBGQlF5eExRVUZMTEVkQlFVY3NSVUZCUlN4RFFVRkRPMUZCUTJwRExFbEJRVWtzVDBGQlR5eERRVUZETEV0QlFVc3NVVUZCVVN4RlFVRkZPMWxCUTNaQ0xFbEJRVWtzUTBGQlF5eExRVUZMTEVOQlFVTXNVMEZCVXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFZEJRVWNzVVVGQlVTeERRVUZETzFOQlEzWkRPMkZCUVUwN1dVRkRTQ3hKUVVGSkxFTkJRVU1zUzBGQlN5eERRVUZETEVOQlFVTXNRMEZCUXl4SFFVRkhMRkZCUVZFc1EwRkJRenRUUVVNMVFqdFJRVU5FTEU5QlFVOHNTVUZCU1N4RFFVRkRPMGxCUTJoQ0xFTkJRVU03U1VGSlV5eDNRa0ZCVHl4SFFVRnFRaXhWUVVGclFpeERRVUZSTzFGQlEzUkNMRWxCUVVrc1NVRkJTU3hEUVVGRExFdEJRVXNzU1VGQlNTeFBRVUZQTEVsQlFVa3NRMEZCUXl4TFFVRkxMRU5CUVVNc1EwRkJReXhEUVVGRExFbEJRVWtzUTBGQlF5eExRVUZMTEZkQlFWY3NSVUZCUlR0WlFVTjZSQ3hKUVVGSkxFTkJRVU1zUzBGQlN5eERRVUZETEVOQlFVTXNRMEZCUXl4SlFVRkpMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dFRRVU42UWp0SlFVTk1MRU5CUVVNN1NVRkxVeXh4UWtGQlNTeEhRVUZrTzFGQlEwa3NUMEZCVHl4RFFVRkRMRWxCUVVrc1EwRkJReXhKUVVGSkxFVkJRVVU3V1VGQlF5eERRVUZETzBsQlEzcENMRU5CUVVNN1NVRkxVeXh4UWtGQlNTeEhRVUZrTzFGQlEwa3NTVUZCU1N4SlFVRkpMRU5CUVVNc1RVRkJUU3hIUVVGSExFbEJRVWtzUTBGQlF5eFZRVUZWTEVWQlFVVTdXVUZETDBJc1NVRkJTU3hEUVVGRExGRkJRVkVzUjBGQlJ5eExRVUZMTEVOQlFVTTdXVUZEZEVJc1NVRkJTU3hEUVVGRExFOUJRVThzUTBGQlF5eEZRVUZGTEVsQlFVa3NSVUZCUlN4VFFVRlRMRU5CUVVNc1IwRkJSeXhGUVVGRkxFdEJRVXNzUlVGQlJTeEpRVUZKTEVOQlFVTXNUVUZCVFN4SFFVRkhMRU5CUVVNc1JVRkJSU3hOUVVGTkxFVkJRVVVzU1VGQlNTeERRVUZETEZkQlFWY3NSVUZCUlN4RFFVRkRMRU5CUVVNN1dVRkRlRVlzVDBGQlR5eEpRVUZKTEVOQlFVTTdVMEZEWmp0UlFVTkVMRWxCUVUwc1EwRkJReXhIUVVGSExFbEJRVWtzUTBGQlF5eE5RVUZOTEVOQlFVTXNUVUZCVFN4RlFVTjBRaXhEUVVGRExFZEJRVWNzU1VGQlNTeERRVUZETEUxQlFVMHNRMEZCUXl4TlFVRk5MRU5CUVVNN1VVRkROMElzU1VGQlNTeERRVUZETEVWQlFVVXNRMEZCUXl4RFFVRkRPMUZCUlZRc1NVRkJTU3hEUVVGRExGRkJRVkVzUTBGQlF5eExRVUZMTEVOQlFVTXNTMEZCU3l4RlFVRkZMRU5CUVVNN1VVRkROVUlzUzBGQlN5eERRVUZETEVkQlFVY3NRMEZCUXl4RlFVRkZMRU5CUVVNc1IwRkJSeXhEUVVGRExFVkJRVVVzUlVGQlJTeERRVUZETEVWQlFVVTdXVUZEY0VJc1EwRkJReXhIUVVGSExFbEJRVWtzUTBGQlF5eE5RVUZOTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNN1dVRkRia0lzU1VGQlNTeERRVUZETEVOQlFVTXNTMEZCU3l4RlFVRkZPMmRDUVVOVUxFbEJRVWtzVDBGQlR5eERRVUZETEVOQlFVTXNSVUZCUlN4TFFVRkxMRmRCUVZjc1NVRkJTU3hQUVVGUExFTkJRVU1zUTBGQlF5eEZRVUZGTEV0QlFVc3NWMEZCVnl4RlFVRkZPMjlDUVVNMVJDeERRVUZETEVOQlFVTXNSVUZCUlN4SFFVRkhMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU03YjBKQlExZ3NRMEZCUXl4RFFVRkRMRVZCUVVVc1IwRkJSeXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETzJsQ1FVTmtPMmRDUVVORUxFbEJRVWtzUTBGQlF5eEhRVUZITEVOQlFVTXNRMEZCUXl4RFFVRkRMRVZCUVVVc1JVRkJSU3hEUVVGRExFTkJRVU1zUlVGQlJTeERRVUZETEVOQlFVTTdaMEpCUTNKQ0xFbEJRVWtzUTBGQlF5eFJRVUZSTEVOQlFVTXNTMEZCU3l4RFFVRkRMRWRCUVVjc1EwRkJReXhEUVVGRExFVkJRVVVzUTBGQlF5eERRVUZETEVOQlFVTTdZVUZEYWtNN1UwRkRTanRSUVVWRUxFbEJRVWtzUlVGQlJTeEhRVUZITEVsQlFVa3NRMEZCUXl4UlFVRlJMRU5CUVVNc1ZVRkJWU3hGUVVGRkxFTkJRVU03VVVGRmNFTXNTVUZCU1N4RlFVRkZMRXRCUVVzc1EwRkJReXhGUVVGRk8xbEJRMVlzU1VGQlNTeERRVUZETEUxQlFVMHNSMEZCUnl4RFFVRkRMRU5CUVVNN1UwRkRia0k3WVVGQlRTeEpRVUZKTEU5QlFVOHNTVUZCU1N4RFFVRkRMRmRCUVZjc1MwRkJTeXhYUVVGWExFVkJRVVU3V1VGRGFFUXNTVUZCU1N4RFFVRkRMRTFCUVUwc1IwRkJSeXhGUVVGRkxFTkJRVU03VTBGRGNFSTdVVUZEUkN4SlFVRkpMRU5CUVVNc1YwRkJWeXhIUVVGSExFVkJRVVVzUTBGQlF6dFJRVVYwUWl4SlFVRkpMRU5CUVVNc2JVSkJRVzFDTEVWQlFVVXNRMEZCUXp0UlFVVXpRaXhKUVVGSkxFTkJRVU1zVDBGQlR5eERRVUZETEVWQlFVVXNTVUZCU1N4RlFVRkZMRk5CUVZNc1EwRkJReXhKUVVGSkxFVkJRVVVzUzBGQlN5eEZRVUZGTEVsQlFVa3NRMEZCUXl4TlFVRk5MRVZCUVVVc1RVRkJUU3hGUVVGRkxFbEJRVWtzUTBGQlF5eFhRVUZYTEVWQlFVVXNRMEZCUXl4RFFVRkRPMUZCUTNKR0xFOUJRVThzUzBGQlN5eERRVUZETzBsQlEycENMRU5CUVVNN1NVRkhUeXh2UTBGQmJVSXNSMEZCTTBJN1VVRkRTU3hKUVVGTkxFTkJRVU1zUjBGQlJ5eEpRVUZKTEVOQlFVTXNVVUZCVVN4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUlVGQlJTeERRVUZETEVkQlFVY3NTVUZCU1N4RFFVRkRMRkZCUVZFc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTTdVVUZEY2tRc1NVRkJTU3hEUVVGRExFVkJRVVVzUTBGQlF5eEhRVUZITEVsQlFVa3NRMEZCUXl4TlFVRk5MRU5CUVVNc1RVRkJUU3hEUVVGRE8xRkJRemxDTEU5QlFVOHNRMEZCUXl4RlFVRkZMRVZCUVVVN1dVRkRVaXhEUVVGRExFZEJRVWNzU1VGQlNTeERRVUZETEUxQlFVMHNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJRenRaUVVOdVFpeERRVUZETEVOQlFVTXNRMEZCUXl4SFFVRkhMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dFpRVU5ZTEVOQlFVTXNRMEZCUXl4RFFVRkRMRWRCUVVjc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETzFOQlEyUTdTVUZEVEN4RFFVRkRPMGxCVjBRc2MwSkJRVXNzUjBGQlRDeFZRVUZOTEVOQlFVODdVVUZEVkN4SlFVRkpMRU5CUVVNc1EwRkJReXhGUVVGRk8xbEJRMG9zU1VGQlNTeEpRVUZKTEVOQlFVTXNUVUZCVFN4RFFVRkRMRTFCUVUwc1MwRkJTeXhEUVVGRExFbEJRVWtzU1VGQlNTeERRVUZETEUxQlFVMHNRMEZCUXl4TlFVRk5MRWRCUVVjc1EwRkJReXhGUVVGRk8yZENRVWR3UkN4SlFVRkpMRU5CUVVNc1IwRkJSeXhEUVVGRExFTkJRVU03WjBKQlExWXNTVUZCU1N4RFFVRkRMRTFCUVUwc1EwRkJReXhQUVVGUExFTkJRVU1zVlVGQlZTeERRVUZETzI5Q1FVTXpRaXhEUVVGRExFZEJRVWNzU1VGQlNTeERRVUZETEVkQlFVY3NRMEZCUXl4RFFVRkRMRVZCUVZVc1EwRkJReXhEUVVGRExFMUJRVTBzUlVGQlZTeERRVUZETEVOQlFVTXNUVUZCVFN4RFFVRkRMRU5CUVVNN1owSkJRM2hFTEVOQlFVTXNRMEZCUXl4RFFVRkRPMmRDUVVOSUxFbEJRVWtzUTBGQlF5eE5RVUZOTEVkQlFVY3NTVUZCU1N4TFFVRkxMRU5CUVVNc1JVRkJSU3hEUVVGRExFTkJRVU1zUTBGQlF6dG5Ra0ZETjBJc1MwRkJTeXhKUVVGSkxFTkJRVU1zUjBGQlJ5eERRVUZETEVWQlFVVXNRMEZCUXl4SFFVRkhMRU5CUVVNc1JVRkJSU3hGUVVGRkxFTkJRVU1zUlVGQlJUdHZRa0ZEZUVJc1NVRkJTU3hEUVVGRExFMUJRVTBzUTBGQlF5eERRVUZETEVOQlFVTXNSMEZCUnl4RlFVRkZMRU5CUVVNN2FVSkJRM1pDTzJGQlEwbzdXVUZEUkN4UFFVRlBMRWxCUVVrc1EwRkJReXhOUVVGTkxFTkJRVU03VTBGRGRFSTdVVUZEUkN4SlFVRkpMRU5CUVVNc1RVRkJUU3hIUVVGSExFTkJRVU1zUTBGQlF6dFJRVU5vUWl4UFFVRlBMRWxCUVVrc1EwRkJRenRKUVVOb1FpeERRVUZETzBsQlUwUXNkVUpCUVUwc1IwRkJUaXhWUVVGUExFTkJRV2RDTzFGQlFYWkNMR2xDUVhWQ1F6dFJRWFJDUnl4SlFVRkpMRU5CUVVNc1EwRkJRenRaUVVGRkxFOUJRVThzU1VGQlNTeERRVUZETEU5QlFVOHNRMEZCUXp0UlFVTTFRaXhKUVVGSkxFTkJRVU1zVDBGQlR5eEhRVUZITEVOQlFVTXNRMEZCUXp0UlFVTnFRaXhKUVVGSkxFTkJRVU1zVlVGQlZTeEhRVUZITEVWQlFVVXNRMEZCUXp0UlFVTnlRaXhKUVVGSkxFTkJRVU1zVDBGQlR5eERRVUZETEU5QlFVOHNRMEZCUXl4VlFVRkJMRU5CUVVNN1dVRkRiRUlzU1VGQlNTeFBRVUZQTEVOQlFVTXNRMEZCUXl4UFFVRlBMRXRCUVVzc1YwRkJWenRuUWtGRGFFTXNRMEZCUXl4RFFVRkRMRTlCUVU4c1IwRkJSeXhEUVVGRExFTkJRVU03V1VGRGJFSXNTVUZCU1N4UFFVRlBMRU5CUVVNc1EwRkJReXhOUVVGTkxFdEJRVXNzVjBGQlZ5eEZRVUZGTzJkQ1FVTnFReXhEUVVGRExFTkJRVU1zVFVGQlRTeERRVUZETEU5QlFVOHNRMEZCUXl4VlFVRkRMRU5CUVVNc1JVRkJSU3hEUVVGRE8yOUNRVU5zUWl4SlFVRkpMRTlCUVU4c1EwRkJReXhMUVVGTExGRkJRVkU3ZDBKQlEzSkNMRU5CUVVNc1EwRkJReXhEUVVGRExFMUJRVTBzUTBGQlF5eERRVUZETEVOQlFVTXNSMEZCUnl4TFFVRkpMRU5CUVVNc1RVRkJUU3hEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNUVUZCVFN4SFFVRkhMRU5CUVVNc1EwRkJRVHRuUWtGRGFrUXNRMEZCUXl4RFFVRkRMRU5CUVVNN1lVRkRUanRaUVVORUxFbEJRVWtzVDBGQlR5eERRVUZETEVOQlFVTXNUVUZCVFN4TFFVRkxMRmRCUVZjc1JVRkJSVHRuUWtGRGFrTXNRMEZCUXl4RFFVRkRMRTFCUVUwc1EwRkJReXhQUVVGUExFTkJRVU1zVlVGQlF5eEZRVUZGTEVWQlFVVXNRMEZCUXp0dlFrRkRia0lzU1VGQlNTeFBRVUZQTEVWQlFVVXNTMEZCU3l4UlFVRlJPM2RDUVVOMFFpeERRVUZETEVOQlFVTXNRMEZCUXl4TlFVRk5MRU5CUVVNc1EwRkJReXhEUVVGRExFZEJRVWNzUzBGQlNTeERRVUZETEU5QlFVOHNRMEZCUXl4RlFVRkZMRU5CUVVNc1EwRkJReXhEUVVGRExFMUJRVTBzUjBGQlJ5eERRVUZETEVOQlFVRTdaMEpCUTI1RUxFTkJRVU1zUTBGQlF5eERRVUZETzJGQlEwNDdVVUZEVEN4RFFVRkRMRU5CUVVNc1EwRkJRenRSUVVOSUxFbEJRVWtzUTBGQlF5eFZRVUZWTEVOQlFVTXNUVUZCVFN4SFFVRkhMRWxCUVVrc1EwRkJReXhOUVVGTkxFTkJRVU1zVFVGQlRTeERRVUZETEZWQlFVRXNRMEZCUXl4SlFVRkpMRTlCUVVFc1QwRkJUeXhEUVVGRExFTkJRVU1zVFVGQlRTeExRVUZMTEZkQlFWY3NSVUZCTDBJc1EwRkJLMElzUTBGQlF5eERRVUZETzFGQlEyeEdMRWxCUVVrc1EwRkJReXhWUVVGVkxFTkJRVU1zVFVGQlRTeEhRVUZITEVsQlFVa3NRMEZCUXl4UFFVRlBMRU5CUVVNc1RVRkJUU3hEUVVGRExGVkJRVUVzUTBGQlF5eEpRVUZKTEU5QlFVRXNUMEZCVHl4RFFVRkRMRU5CUVVNc1RVRkJUU3hMUVVGTExGZEJRVmNzUlVGQkwwSXNRMEZCSzBJc1EwRkJReXhEUVVGRE8xRkJRMjVHTEU5QlFVOHNTVUZCU1N4RFFVRkRPMGxCUTJoQ0xFTkJRVU03U1VGRlJDeHBRMEZCWjBJc1IwRkJhRUlzVlVGQmFVSXNRMEZCVnp0UlFVTjRRaXhKUVVGSkxFTkJRVU1zUjBGQlJ5eFZRVUZWTEVOQlFVTXNVMEZCVXl4RFFVRkRMRWxCUVVrc1EwRkJReXhOUVVGTkxFVkJRVVVzU1VGQlNTeERRVUZETEUxQlFVMHNSVUZCUlN4SlFVRkpMRU5CUVVNc1dVRkJXU3hGUVVGRkxFbEJRVWtzUTBGQlF5eFZRVUZWTEVOQlFVTXNRMEZCUXp0UlFVTXpSaXhKUVVGSkxFTkJRVU1zVFVGQlRTeERRVUZETEVOQlFVTXNRMEZCUXl4TlFVRk5MRU5CUVVNc1EwRkJRenRSUVVOMFFpeERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNN1VVRkRUQ3hQUVVGUExFbEJRVWtzUTBGQlF6dEpRVU5vUWl4RFFVRkRPMGxCVlVRc09FSkJRV0VzUjBGQllpeFZRVUZqTEVOQlFWYzdVVUZEY2tJc1NVRkJTU3hEUVVGRExGTkJRVk1zUTBGQlF5eE5RVUZOTzFsQlFVVXNUMEZCVHl4SlFVRkpMRU5CUVVNc1kwRkJZeXhEUVVGRE8xRkJRMnhFTEVsQlFVa3NRMEZCUXl4alFVRmpMRWRCUVVjc1EwRkJReXhEUVVGRE8xRkJRM2hDTEU5QlFVOHNTVUZCU1N4RFFVRkRPMGxCUTJoQ0xFTkJRVU03U1VGWlJDeHRRMEZCYTBJc1IwRkJiRUlzVlVGQmJVSXNRMEZCVnp0UlFVTXhRaXhKUVVGSkxFTkJRVU1zVTBGQlV5eERRVUZETEUxQlFVMDdXVUZCUlN4UFFVRlBMRWxCUVVrc1EwRkJReXh0UWtGQmJVSXNRMEZCUXp0UlFVTjJSQ3hKUVVGSkxFTkJRVU1zYlVKQlFXMUNMRWRCUVVjc1EwRkJReXhEUVVGRE8xRkJRemRDTEU5QlFVOHNTVUZCU1N4RFFVRkRPMGxCUTJoQ0xFTkJRVU03U1VGUlJDd3lRa0ZCVlN4SFFVRldMRlZCUVZjc1NVRkJXU3hGUVVGRkxHRkJRWGRETzFGQlF6ZEVMRWxCUVVrc1EwRkJReXhUUVVGVExFTkJRVU1zVFVGQlRUdFpRVUZGTEVsQlFVa3NSMEZCUnl4SFFVRkhMRU5CUVVNN1VVRkRiRU1zU1VGQlNTeERRVUZETEhkQ1FVRjNRaXhIUVVGSE8xbEJRelZDTEVsQlFVa3NSVUZCUlN4SlFVRkpPMWxCUTFZc1owSkJRV2RDTEVWQlFVVXNUMEZCVHl4aFFVRmhMRXRCUVVzc1VVRkJVU3hEUVVGRExFTkJRVU1zUTBGQlF5eGpRVUZqTEU5QlFVOHNZVUZCWVN4RFFVRkJMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eGhRVUZoTzFOQlF6ZEhMRU5CUVVNN1VVRkRSaXhQUVVGUExFbEJRVWtzUTBGQlF6dEpRVU5vUWl4RFFVRkRPMGxCVTBRc2MwSkJRVXNzUjBGQlRDeFZRVUZOTEVOQlFUUkNPMUZCUXpsQ0xFbEJRVWtzUTBGQlF5eFRRVUZUTEVOQlFVTXNUVUZCVFR0WlFVRkZMRTlCUVU4c1NVRkJTU3hEUVVGRExFMUJRVTBzUTBGQlF6dFJRVU14UXl4SlFVRkpMRU5CUVVNc1RVRkJUU3hIUVVGSExFTkJRVU1zUTBGQlF6dFJRVU5vUWl4UFFVRlBMRWxCUVVrc1EwRkJRenRKUVVOb1FpeERRVUZETzBsQlZVUXNORUpCUVZjc1IwRkJXQ3hWUVVGWkxFTkJRV003VVVGRGRFSXNTVUZCU1N4RFFVRkRMRk5CUVZNc1EwRkJReXhOUVVGTk8xbEJRVVVzVDBGQlR5eEpRVUZKTEVOQlFVTXNXVUZCV1N4RFFVRkRPMUZCUTJoRUxFbEJRVWtzUTBGQlF5eFpRVUZaTEVkQlFVY3NRMEZCUXl4RFFVRkRPMUZCUTNSQ0xFOUJRVThzU1VGQlNTeERRVUZETzBsQlEyaENMRU5CUVVNN1NVRlhSQ3dyUWtGQll5eEhRVUZrTEZWQlFXVXNRMEZCVHp0UlFVTnNRaXhKUVVGSkxFTkJRVU1zVTBGQlV5eERRVUZETEUxQlFVMDdXVUZCUlN4UFFVRlBMRWxCUVVrc1EwRkJReXhsUVVGbExFTkJRVU03VVVGRGJrUXNTVUZCU1N4RFFVRkRMR1ZCUVdVc1IwRkJSeXhEUVVGRExFTkJRVU03VVVGRGVrSXNUMEZCVHl4SlFVRkpMRU5CUVVNN1NVRkRhRUlzUTBGQlF6dEpRVlZFTEhGQ1FVRkpMRWRCUVVvc1ZVRkJTeXhEUVVGcFFqdFJRVU5zUWl4SlFVRkpMRU5CUVVNc1EwRkJRenRaUVVGRkxFOUJRVThzU1VGQlNTeERRVUZETEZkQlFWY3NRMEZCUXp0UlFVTm9ReXhKUVVGSkxFTkJRVU1zVjBGQlZ5eEhRVUZITEVOQlFVTXNRMEZCUXp0UlFVTnlRaXhQUVVGUExFbEJRVWtzUTBGQlF6dEpRVU5vUWl4RFFVRkRPMGxCVTBRc1owTkJRV1VzUjBGQlppeFZRVUZuUWl4RFFVRlBPMUZCUTI1Q0xFbEJRVWtzUTBGQlF5eERRVUZETzFsQlFVVXNUMEZCVHl4SlFVRkpMRU5CUVVNc1owSkJRV2RDTEVOQlFVTTdVVUZEY2tNc1NVRkJTU3hEUVVGRExHZENRVUZuUWl4SFFVRkhMRU5CUVVNc1EwRkJRenRSUVVNeFFpeFBRVUZQTEVsQlFVa3NRMEZCUXp0SlFVTm9RaXhEUVVGRE8wbEJVMFFzYVVOQlFXZENMRWRCUVdoQ0xGVkJRV2xDTEVOQlFVODdVVUZEY0VJc1NVRkJTU3hEUVVGRExFTkJRVU03V1VGQlJTeFBRVUZQTEVsQlFVa3NRMEZCUXl4cFFrRkJhVUlzUTBGQlF6dFJRVU4wUXl4SlFVRkpMRU5CUVVNc2FVSkJRV2xDTEVkQlFVY3NRMEZCUXl4RFFVRkRPMUZCUXpOQ0xFOUJRVThzU1VGQlNTeERRVUZETzBsQlEyaENMRU5CUVVNN1NVRlRSQ3cyUWtGQldTeEhRVUZhTEZWQlFXRXNRMEZCVHp0UlFVTm9RaXhKUVVGSkxFTkJRVU1zUTBGQlF5eEZRVUZGTzFsQlEwb3NUMEZCVHl4SlFVRkpMRU5CUVVNc1lVRkJZU3hEUVVGRE8xTkJRemRDTzFGQlEwUXNTVUZCU1N4RFFVRkRMR0ZCUVdFc1IwRkJSeXhQUVVGUExFTkJRVU1zUzBGQlN5eFZRVUZWTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTTdVVUZEZEVRc1NVRkJTU3hEUVVGRExIRkNRVUZ4UWl4SFFVRkhMRWxCUVVrc1EwRkJRenRSUVVOc1F5eFBRVUZQTEVsQlFVa3NRMEZCUXp0SlFVTm9RaXhEUVVGRE8wbEJSVVFzZVVKQlFWRXNSMEZCVWl4VlFVRlRMRU5CUVc5Q08xRkJRM3BDTEVsQlFVa3NRMEZCUXl4VFFVRlRMRWRCUVVjc1EwRkJReXhEUVVGRE8xRkJRMjVDTEU5QlFVOHNTVUZCU1N4RFFVRkRPMGxCUTJoQ0xFTkJRVU03U1VGSlJDeHhRMEZCYjBJc1IwRkJjRUlzVlVGQmNVSXNRMEZCVlR0UlFVTXpRaXhKUVVGSkxFTkJRVU1zUTBGQlF6dFpRVUZGTEU5QlFVOHNTVUZCU1N4RFFVRkRMRlZCUVZVc1EwRkJRenRSUVVNdlFpeEpRVUZKTEVOQlFVTXNWVUZCVlN4SFFVRkhMRTlCUVU4c1EwRkJReXhMUVVGTExGVkJRVlVzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dFJRVU51UkN4UFFVRlBMRWxCUVVrc1EwRkJRenRKUVVOb1FpeERRVUZETzBsQlNVUXNjMEpCUVVzc1IwRkJUQ3hWUVVGTkxFTkJRVlU3VVVGRFdpeEpRVUZKTEVOQlFVTXNVMEZCVXl4RFFVRkRMRTFCUVUwN1dVRkJSU3hQUVVGUExFbEJRVWtzUTBGQlF5eE5RVUZOTEVOQlFVTTdZVUZEY2tNN1dVRkRSQ3hEUVVGRExFZEJRVWNzUTBGQlF5eERRVUZETEVOQlFVTTdXVUZEVUN4SlFVRkpMRWxCUVVrc1EwRkJReXhOUVVGTkxFVkJRVVU3WjBKQlEySXNTVUZCU1N4RFFVRkRMRWRCUVVjc1EwRkJRenR2UWtGQlJTeEpRVUZKTEVOQlFVTXNUVUZCVFN4SFFVRkhMRU5CUVVNc1EwRkJRenM3YjBKQlEzUkNMRWxCUVVrc1EwRkJReXhOUVVGTkxFZEJRVWNzUTBGQlF5eERRVUZETzJGQlEzaENPMmxDUVVGTkxFbEJRVWtzUTBGQlF5eEhRVUZITEVOQlFVTXNSVUZCUlR0blFrRkRaQ3hKUVVGSkxFTkJRVU1zU1VGQlNTeERRVUZETEZGQlFWRXNSVUZCUlR0dlFrRkRhRUlzU1VGQlNTeERRVUZETEZGQlFWRXNSMEZCUnl4SlFVRkpMRU5CUVVNN2IwSkJRM0pDTEVsQlFVa3NRMEZCUXl4UFFVRlBMRU5CUVVNc1JVRkJSU3hKUVVGSkxFVkJRVVVzVTBGQlV5eERRVUZETEV0QlFVc3NSVUZCUlN4TFFVRkxMRVZCUVVVc1NVRkJTU3hEUVVGRExFMUJRVTBzUjBGQlJ5eERRVUZETEVWQlFVTXNRMEZCUXl4RFFVRkRPMjlDUVVNdlJDeEpRVUZKTEVOQlFVTXNTVUZCU1N4RlFVRkZMRU5CUVVNN2FVSkJRMlk3WVVGRFNqdFpRVU5FTEU5QlFVOHNTVUZCU1N4RFFVRkRPMU5CUTJZN1NVRkRUQ3hEUVVGRE8wbEJSVVFzT0VKQlFXRXNSMEZCWWl4VlFVRmpMRWxCUVhsQ08xRkJRMjVETEU5QlFVOHNUMEZCVHl4SlFVRkpMRU5CUVVNc1lVRkJZU3hMUVVGTExGVkJRVlVzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRXJRaXhKUVVGSkxFTkJRVU1zWVVGQll5eERRVUZETEVsQlFVa3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGVExFbEJRVWtzUTBGQlF5eGhRVUZoTEVOQlFVTTdTVUZET1Vrc1EwRkJRenRKUVVWTkxHOUNRVUZoTEVkQlFYQkNMRlZCUVhGQ0xFbEJRWFZDTEVWQlFVVXNUVUZCWXp0UlFVTjRSQ3hKUVVGSkxFTkJRVU1zVFVGQlRTeEhRVUZITEUxQlFVMHNRMEZCUXp0SlFVTjZRaXhEUVVGRE8wbEJSVVFzTkVKQlFWY3NSMEZCV0N4VlFVRlpMRWxCUVhsQ08xRkJRMnBETEU5QlFVOHNUMEZCVHl4SlFVRkpMRU5CUVVNc1UwRkJVeXhMUVVGTExGVkJRVlVzUTBGQlF5eERRVUZETEVOQlFVTXNTVUZCU1N4RFFVRkRMRk5CUVZNc1EwRkJReXhKUVVGSkxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRPMGxCUXpORkxFTkJRVU03U1VGdFFrUXNlVU5CUVhkQ0xFZEJRWGhDTEZWQlFYbENMRmRCUVcxQ0xFVkJRVVVzUTBGQllUdFJRVUV6UkN4cFFrRkpRenRSUVVvMlF5eHJRa0ZCUVN4RlFVRkJMRXRCUVdFN1VVRkRka1FzU1VGQlNTeERRVUZETEZsQlFWa3NRMEZCUXl4VlFVRkJMRU5CUVVNc1NVRkJTU3hQUVVGQkxGZEJRVmNzUjBGQlJ5eERRVUZETEVOQlFVTXNUVUZCVFN4RlFVRjBRaXhEUVVGelFpeERRVUZETEVOQlFVTTdVVUZETDBNc1NVRkJTU3hEUVVGRExIRkNRVUZ4UWl4SFFVRkhMR05CUVUwc1QwRkJRU3h6UTBGQmQwSXNRMEZCUXl4TFFVRkpMRU5CUVVNc1RVRkJUU3hGUVVGRkxFdEJRVWtzUTBGQlF5eFpRVUZaTEVWQlFVVXNRMEZCUXl4RFFVRkRMRVZCUVRORUxFTkJRVEpFTEVOQlFVTTdVVUZETDBZc1QwRkJUeXhKUVVGSkxFTkJRVU03U1VGRGFFSXNRMEZCUXp0SlFWbEVMRzFEUVVGclFpeEhRVUZzUWl4VlFVRnRRaXhYUVVGdFFpeEZRVUZGTEVOQlFXRTdVVUZCY2tRc2FVSkJTVU03VVVGS2RVTXNhMEpCUVVFc1JVRkJRU3hMUVVGaE8xRkJRMnBFTEVsQlFVa3NRMEZCUXl4WlFVRlpMRU5CUVVNc1ZVRkJRU3hEUVVGRExFbEJRVWtzVDBGQlFTeFhRVUZYTEVkQlFVY3NRMEZCUXl4RFFVRkRMRTFCUVUwc1JVRkJkRUlzUTBGQmMwSXNRMEZCUXl4RFFVRkRPMUZCUXk5RExFbEJRVWtzUTBGQlF5eHhRa0ZCY1VJc1IwRkJSeXhqUVVGTkxFOUJRVUVzWjBOQlFXdENMRU5CUVVNc1MwRkJTU3hEUVVGRExFMUJRVTBzUlVGQlJTeExRVUZKTEVOQlFVTXNXVUZCV1N4RlFVRkZMRU5CUVVNc1EwRkJReXhGUVVGeVJDeERRVUZ4UkN4RFFVRkRPMUZCUTNwR0xFOUJRVThzU1VGQlNTeERRVUZETzBsQlEyaENMRU5CUVVNN1NVRlpSQ3h6UWtGQlN5eEhRVUZNTEZWQlEwa3NPRUpCUVRCRExFVkJRekZETEN0Q1FVRXlReXhGUVVNelF5d3JRa0ZCTWtNc1JVRkRNME1zYTBKQlFUaENMRVZCUXpsQ0xGZEJRV3RDTEVWQlEyeENMRmRCUVd0Q08xRkJUblJDTEdsQ1FYTktRenRSUVhKS1J5d3JRMEZCUVN4RlFVRkJMR3REUVVFd1F6dFJRVU14UXl4blJFRkJRU3hGUVVGQkxHMURRVUV5UXp0UlFVTXpReXhuUkVGQlFTeEZRVUZCTEcxRFFVRXlRenRSUVVNelF5eHRRMEZCUVN4RlFVRkJMSE5DUVVFNFFqdFJRVU01UWl3MFFrRkJRU3hGUVVGQkxHdENRVUZyUWp0UlFVTnNRaXcwUWtGQlFTeEZRVUZCTEd0Q1FVRnJRanRSUVVWc1FpeEpRVUZKTEVOQlFWTXNSVUZEVkN4RFFVRlRMRVZCUTFRc1EwRkJReXhIUVVGblFpeEpRVUZKTEVOQlFVTXNTMEZCU3l4RlFVRkhMRU5CUVVNc1RVRkJUU3hGUVVOeVF5eERRVUZETEVkQlFVY3NRMEZCUXl4SFFVRkhMRU5CUVVNc1IwRkJSeXhKUVVGSkxFTkJRVU1zVDBGQlR5eERRVUZETEUxQlFVMHNSVUZETDBJc1EwRkJReXhIUVVGSExFbEJRVWtzUTBGQlF5eE5RVUZOTEVOQlFVTXNUVUZCVFN4RlFVTjBRaXhEUVVGRExFZEJRVWNzU1VGQlNTeERRVUZETEZkQlFWY3NRMEZCUXl4RFFVRkRMRU5CUVVNc1JVRkRka0lzUTBGQlF5eEhRVUZITEVsQlFVa3NRMEZCUXl4WFFVRlhMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU03VVVGRk5VSXNTVUZCU1N4RFFVRkRMRWRCUVVjc1NVRkJTU3hMUVVGTExFTkJRVU1zUTBGQlF5eERRVUZETEVWQlFVVXNRMEZCUXl4SFFVRkhMRWxCUVVrc1MwRkJTeXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETzFGQlJYWkRMRWxCUVVrc1EwRkJReXhIUVVGSExFbEJRVWtzUTBGQlF6dFJRVVZpTEVsQlFVa3NSVUZCUlN4SFFVRkhMRWxCUVVrc1EwRkJReXhqUVVGakxFTkJRVU03VVVGRk4wSXNTVUZCU1N4RFFVRkRMRTFCUVUwc1EwRkJReXhQUVVGUExFTkJRVU1zVlVGQlF5eERRVUZETEVWQlFVVXNRMEZCUXp0WlFVTnlRaXhEUVVGRExFTkJRVU1zUzBGQlN5eEhRVUZITEVOQlFVTXNRMEZCUXp0WlFVTmFMRWxCUVVrc1QwRkJUeXhEUVVGRExFTkJRVU1zUTBGQlF5eExRVUZMTEZkQlFWY3NSVUZCUlR0blFrRkROVUlzUTBGQlF5eERRVUZETEVOQlFVTXNSMEZCUnl4RFFVRkRMRWRCUVVjc1EwRkJReXhGUVVGRkxFTkJRVU1zUTBGQlF5eERRVUZETEVkQlFVY3NRMEZCUXl4SFFVRkhMRU5CUVVNc1EwRkJRenRoUVVNMVFqdFpRVU5FTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1IwRkJSeXhEUVVGRExFTkJRVU1zUTBGQlF5eEZRVUZGTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1IwRkJSeXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETzFGQlF6TkNMRU5CUVVNc1EwRkJReXhEUVVGRE8xRkJSVWdzU1VGQlNTeEpRVUZKTEVOQlFVTXNjVUpCUVhGQ08xbEJRVVVzU1VGQlNTeERRVUZETEhGQ1FVRnhRaXhGUVVGRkxFTkJRVU03VVVGTE4wUXNTVUZCU1N4VFFVRlRMRU5CUVVNN1VVRkRaQ3hKUVVGSkxFbEJRVWtzUTBGQlF5eGxRVUZsTEVWQlFVVTdXVUZGZEVJc1UwRkJVeXhIUVVGSExFbEJRVWtzUTBGQlF5eGxRVUZsTEVOQlFVTTdVMEZEY0VNN1lVRkJUVHRaUVVWSUxGTkJRVk1zUjBGQlJ5eERRVUZETEVsQlFVa3NNRUpCUVZVc1EwRkJReXhEUVVGRExFVkJRVVVzU1VGQlNTeERRVUZETEUxQlFVMHNSVUZCUlN4TlFVRk5MRU5CUVVNc1kwRkJZeXhGUVVGRkxFMUJRVTBzUTBGQlF5eGpRVUZqTEVWQlFVVXNWVUZCUVN4RFFVRkRMRWxCUVVjc1QwRkJRU3hMUVVGSkxFTkJRVU1zWVVGQllTeERRVUZETEVOQlFVTXNRMEZCUXl4RlFVRnlRaXhEUVVGeFFpeERRVUZETEVOQlFVTXNRMEZCUXl4alFVRmpMRVZCUVVVc1EwRkJRenRaUVVsMlNTeERRVUZETEVkQlFVY3NhVUpCUVU4c1EwRkJReXhyUWtGQmEwSXNRMEZCUXl4RFFVRkRMRVZCUVVVc1kwRkJUU3hQUVVGQkxFTkJRVU1zUlVGQlJDeERRVUZETEVOQlFVTXNRMEZCUXp0WlFVTXpReXhKUVVGSkxFTkJRVU1zVFVGQlRTeERRVUZETEU5QlFVOHNRMEZCUXl4VlFVRkJMRU5CUVVNN1owSkJRMnBDTEVsQlFVa3NUMEZCVHl4RFFVRkRMRU5CUVVNc1RVRkJUU3hKUVVGSkxGRkJRVkU3YjBKQlFVVXNRMEZCUXl4RFFVRkRMRTFCUVUwc1IwRkJSeXhMUVVGSkxFTkJRVU1zVFVGQlRTeERRVUZUTEVOQlFVTXNRMEZCUXl4TlFVRk5MRU5CUVVNc1EwRkJRenRuUWtGRE1VVXNTVUZCU1N4UFFVRlBMRU5CUVVNc1EwRkJReXhOUVVGTkxFbEJRVWtzVVVGQlVUdHZRa0ZCUlN4RFFVRkRMRU5CUVVNc1RVRkJUU3hIUVVGSExFdEJRVWtzUTBGQlF5eE5RVUZOTEVOQlFWTXNRMEZCUXl4RFFVRkRMRTFCUVUwc1EwRkJReXhEUVVGRE8xbEJRemxGTEVOQlFVTXNRMEZCUXl4RFFVRkRPMWxCUTBnc1NVRkJTU3hEUVVGRExFMUJRVTBzUTBGQlF5eFBRVUZQTEVOQlFVTXNWVUZCUVN4RFFVRkRPMmRDUVVOcVFpeEpRVUZOTEVOQlFVTXNSMEZCUnl4TlFVRk5MRU5CUVVNc1kwRkJZeXhEUVVGRExFTkJRVU1zUTBGQlF5eEZRVUZGTEVOQlFVTXNSMEZCUnl4TlFVRk5MRU5CUVVNc1kwRkJZeXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETzJkQ1FVTnFSU3hEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRWRCUVVjc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4SFFVRkhMRU5CUVVNc1EwRkJReXhOUVVGTkxFbEJRVWtzUTBGQlF5eERRVUZETzFsQlEzUkRMRU5CUVVNc1EwRkJReXhEUVVGRE8xTkJRMDQ3VVVGRlJDeEpRVUZKTEVOQlFVTXNSMEZCUnl4cFFrRkJUeXhEUVVGRExHdENRVUZyUWl4RFFVRkRMRU5CUVVNc1JVRkJSU3hWUVVGVkxFTkJRVU1zUlVGQlJTeERRVUZETzFsQlEyaEVMRTlCUVU4c1UwRkJVeXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRPMUZCUXpOQ0xFTkJRVU1zUTBGQlF5eERRVUZETzFGQlJVZ3NTVUZCU1N4SlFVRkpMRU5CUVVNc1ZVRkJWU3hKUVVGSkxFOUJRVThzU1VGQlNTeERRVUZETEZWQlFWVXNRMEZCUXl4TlFVRk5MRXRCUVVzc1YwRkJWeXhGUVVGRk8xbEJRMnhGTEVsQlFVa3NRMEZCUXl4SFFVRkhMRU5CUVVNc1EwRkJRenRaUVVOV0xFbEJRVWtzWVVGQllTeEhRVUZITEZWQlFVTXNRMEZCUXl4RlFVRkZMRU5CUVVNc1JVRkJSU3hSUVVGUkxFVkJRVVVzWVVGQllUdG5Ra0ZET1VNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4SFFVRkhMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNSMEZCUnl4UlFVRlJMRU5CUVVNN1owSkJRemRDTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUjBGQlJ5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFZEJRVWNzWVVGQllTeERRVUZETzFsQlEzUkRMRU5CUVVNc1EwRkJRenRaUVVOR0xFbEJRVWtzUTBGQlF5eFBRVUZQTEVOQlFVTXNUMEZCVHl4RFFVRkRMRlZCUVVFc1EwRkJRenRuUWtGRGJFSXNZVUZCWVN4RFFVRkRMRU5CUVVNc1JVRkJSU3hEUVVGRExFZEJRVWNzUTBGQlF5eEZRVUZGTEV0QlFVa3NRMEZCUXl4cFFrRkJhVUlzUlVGQlJTeEhRVUZITEVOQlFVTXNRMEZCUXp0blFrRnBRbkpFTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1IwRkJSeXhEUVVGRExFVkJRVVVzUTBGQlF5eERRVUZETEVOQlFVTXNSVUZCUlN4RFFVRkRMRWRCUVVjc1EwRkJReXhEUVVGRE8yZENRVU55UWl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFZEJRVWNzUTBGQlF5eEZRVUZGTEVOQlFVTXNRMEZCUXl4RFFVRkRMRVZCUVVVc1EwRkJReXhIUVVGSExFTkJRVU1zUTBGQlF6dFpRVU42UWl4RFFVRkRMRU5CUVVNc1EwRkJRenRUUVVOT096dFpRVUZOTEVsQlFVa3NRMEZCUXl4VlFVRlZMRWRCUVVjc1JVRkJSU3hOUVVGTkxFVkJRVVVzU1VGQlNTeERRVUZETEUxQlFVMHNSVUZCUlN4TlFVRk5MRVZCUVVVc1JVRkJSU3hGUVVGRkxFTkJRVU03VVVGRk4wUXNTVUZCU1N4alFVRmpMRWRCUVVjc1NVRkJTU3hEUVVGRExGbEJRVmtzU1VGQlNTeEZRVUZGTEVOQlFVTTdVVUZETjBNc1NVRkJTU3hKUVVGSkxFTkJRVU1zZDBKQlFYZENMRVZCUVVVN1dVRkRla0lzU1VGQlNTeERRVUZETEZsQlFXRXNRMEZCUXl4blFrRkJaMElzUjBGQlJ5eEpRVUZKTEVOQlFVTXNkMEpCUVhkQ0xFTkJRVU1zWjBKQlFXZENMRU5CUVVNN1dVRkRNMFlzWTBGQll5eEhRVUZITEdOQlFXTXNRMEZCUXl4TlFVRk5MRU5CUVVNc05rTkJRU3RDTEVOQlFVTXNRMEZCUXl4RlFVRkZMRWxCUVVrc1EwRkJReXhOUVVGTkxFVkJRVVVzU1VGQlNTeERRVUZETEhkQ1FVRjNRaXhEUVVGRExFbEJRVWtzUlVGQlR5eERRVUZETEVsQlFVa3NRMEZCUXl4WlFVRlpMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU03VTBGSGVrbzdVVUZGUkN4SlFVRkpMRU5CUVVNc1lVRkJZU3hEUVVGRExFdEJRVXNzUTBGQlF5eERRVUZETzFGQlF6RkNMRWxCUVVrc1EwRkJReXhSUVVGUkxFZEJRVWNzU1VGQlNTeHBRa0ZCVHl4RFFVRkRMRU5CUVVNc1EwRkJReXhGUVVGRkxFTkJRVU1zUTBGQlF5eEZRVUZGTEVOQlFVTXNRMEZCUXl4RFFVRkRPMUZCUlhaRExFbEJRVWtzUTBGQlF5eFJRVUZSTEVOQlFVTXNTMEZCU3l4RFFVRkRMRXRCUVVzc1JVRkJSU3hEUVVGRE8xRkJRelZDTEV0QlFVc3NTVUZCU1N4RFFVRkRMRWRCUVVjc1EwRkJReXhGUVVGRkxFTkJRVU1zUjBGQlJ5eERRVUZETEVWQlFVVXNSVUZCUlN4RFFVRkRMRVZCUVVVN1dVRkRlRUlzU1VGQlNTeERRVUZETEVkQlFVY3NTVUZCU1N4RFFVRkRMRTFCUVUwc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dFpRVU4yUWl4SlFVRkpMRU5CUVVNc1EwRkJReXhMUVVGTExFVkJRVVU3WjBKQlExUXNRMEZCUXl4RFFVRkRMRVZCUVVVc1IwRkJSeXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETzJkQ1FVTllMRU5CUVVNc1EwRkJReXhGUVVGRkxFZEJRVWNzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXp0blFrRkRXQ3hKUVVGSkxFTkJRVU1zUjBGQlJ5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRVZCUVVVc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETzJkQ1FVTnVRaXhKUVVGSkxFTkJRVU1zVVVGQlVTeERRVUZETEV0QlFVc3NRMEZCUXl4SFFVRkhMRU5CUVVNc1EwRkJReXhGUVVGRkxFTkJRVU1zUTBGQlF5eERRVUZETzJGQlEycERPMU5CUTBvN1VVRkRSQ3hKUVVGSkxFTkJRVU1zVVVGQlVTeERRVUZETEZOQlFWTXNSMEZCUnl4SlFVRkpMRU5CUVVNc1ZVRkJWU3hEUVVGRE8xRkJTekZETEVsQlFVa3NRMEZCUXl4aFFVRmhMRU5CUVVNc09FSkJRVGhDTEVWQlFVVXNRMEZCUXl4RlFVRkZMRU5CUVVNc1EwRkJReXhEUVVGRE8xRkJSM3BFTEVsQlFVa3NZMEZCWXl4RFFVRkRMRTFCUVUwc1IwRkJSeXhEUVVGRE8xbEJRVVVzU1VGQlNTeERRVUZETEZGQlFWRXNRMEZCUXl4UFFVRlBMRWRCUVVjc1NVRkJTU3h6UWtGQlZTeERRVUZETEVsQlFVa3NRMEZCUXl4TlFVRk5MRVZCUVVVc1NVRkJTU3hEUVVGRExFOUJRVThzUlVGQlJTeEpRVUZKTEVOQlFVTXNWVUZCVlN4RlFVRkZMR05CUVdNc1EwRkJReXhEUVVGRExHZENRVUZuUWl4RlFVRkZMRU5CUVVNN1VVRkRja29zU1VGQlNTeERRVUZETEZGQlFWRXNRMEZCUXl4SFFVRkhMRU5CUVVNc0swSkJRU3RDTEVOQlFVTXNRMEZCUXp0UlFVTnVSQ3hKUVVGSkxFTkJRVU1zTmtKQlFUWkNMRU5CUVVNc1EwRkJReXhGUVVGRkxFTkJRVU1zUlVGQlJTeFhRVUZYTEVOQlFVTXNRMEZCUXp0UlFVZDBSQ3hKUVVGSkxFTkJRVU1zWVVGQllTeERRVUZETEVWQlFVVXNRMEZCUXl4RFFVRkRPMUZCUTNaQ0xFbEJRVWtzUlVGQlJTeEZRVUZGTzFsQlEwb3NTVUZCU1N4RFFVRkRMRTFCUVUwc1EwRkJReXhQUVVGUExFTkJRVU1zVlVGQlZTeERRVUZETEVWQlFVVXNRMEZCUXl4SlFVRkpMRU5CUVVNc1EwRkJReXhEUVVGRExFZEJRVWNzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RlFVRkZMRU5CUVVNc1EwRkJReXhEUVVGRExFZEJRVWNzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU03V1VGRGFrVXNTVUZCU1N4RFFVRkRMRkZCUVZFc1EwRkJReXhQUVVGUExFZEJRVWNzU1VGQlNTeHpRa0ZCVlN4RFFVRkRMRWxCUVVrc1EwRkJReXhOUVVGTkxFVkJRVVVzU1VGQlNTeERRVUZETEU5QlFVOHNSVUZCUlN4SlFVRkpMRU5CUVVNc1ZVRkJWU3hGUVVGRkxHTkJRV01zUlVGQlJTeEpRVUZKTEVOQlFVTXNRMEZCUXl4blFrRkJaMElzUlVGQlJTeERRVUZETzFsQlF6VklMRWxCUVVrc1EwRkJReXhOUVVGTkxFTkJRVU1zVDBGQlR5eERRVUZETEZWQlFWVXNRMEZCUXl4RlFVRkZMRU5CUVVNc1NVRkJTU3hEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVkQlFVY3NRMEZCUXl4RFFVRkRMRU5CUVVNc1JVRkJSU3hEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVkQlFVY3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETzFOQlEzQkZPMUZCUjBRc1NVRkJTU3hEUVVGRExGRkJRVkVzUTBGQlF5eERRVUZETEVkQlFVY3NRMEZCUXl4RFFVRkRPMUZCUTNCQ0xFbEJRVWtzUTBGQlF5eFJRVUZSTEVOQlFVTXNSMEZCUnl4RFFVRkRMQ3RDUVVFclFpeERRVUZETEVOQlFVTTdVVUZGYmtRc1NVRkJTU3hyUWtGQmEwSXNSVUZCUlR0WlFVTndRaXhKUVVGSkxFTkJRVU1zVVVGQlVTeERRVUZETEZsQlFWa3NSMEZCUnl4SlFVRkpMRU5CUVVNN1dVRkRiRU1zU1VGQlNTeERRVUZETEZGQlFWRXNRMEZCUXl4WlFVRlpMRWRCUVVjc1NVRkJTU3hEUVVGRExFMUJRVTBzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4TFFVRkxMRU5CUVVNN1dVRkRiRVFzU1VGQlNTeERRVUZETEZGQlFWRXNRMEZCUXl4blFrRkJaMElzUjBGQlJ5eERRVUZETEVOQlFVTTdXVUZEYmtNc1NVRkJTU3hEUVVGRExGRkJRVkVzUTBGQlF5eGxRVUZsTEVkQlFVY3NRMEZCUXl4SlFVRkpMRU5CUVVNc1EwRkJRenRaUVVOMlF5eEpRVUZKTEVWQlFVVXNSMEZCUnl4cFFrRkJUeXhEUVVGRExHdENRVUZyUWl4RFFVRkRMRU5CUVVNc1JVRkJReXhWUVVGRExFTkJRVU1zUlVGQlJTeERRVUZETzJkQ1FVTjJReXhKUVVGSkxFTkJRVU1zU1VGQlNTeERRVUZETEVsQlFVa3NRMEZCUXl4SlFVRkpMRU5CUVVNN2IwSkJRVVVzVDBGQlR5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU03WjBKQlEzSkRMRTlCUVU4c1EwRkJReXhEUVVGQk8xbEJRMW9zUTBGQlF5eERRVUZETEVOQlFVTTdXVUZEU0N4SlFVRkpMRU5CUVVNc1VVRkJVU3hEUVVGRExFTkJRVU1zUjBGQlJ5eEZRVUZGTEVOQlFVTTdXVUZEY2tJc1NVRkJTU3hEUVVGRExGRkJRVkVzUTBGQlF5eEhRVUZITEVOQlFVTXNhMEpCUVd0Q0xFTkJRVU1zUTBGQlF6dFRRVU42UXp0UlFVVkVMRWxCUVVrc1EwRkJReXh0UWtGQmJVSXNSVUZCUlN4RFFVRkRPMUZCUXpOQ0xFbEJRVWtzUTBGQlF5dzJRa0ZCTmtJc1EwRkJReXhEUVVGRExFVkJRVVVzUTBGQlF5eEZRVUZGTEZkQlFWY3NRMEZCUXl4RFFVRkRPMUZCUTNSRUxFOUJRVThzVjBGQlZ5eERRVUZETEVOQlFVTXNRMEZCUXl4SlFVRkpMRU5CUVVNc1RVRkJUU3hGUVVGRkxFTkJRVU1zUTBGQlF5eERRVUZETEVsQlFVa3NRMEZCUXp0SlFVTTVReXhEUVVGRE8wbEJSVThzT0VKQlFXRXNSMEZCY2tJc1ZVRkJjMElzVlVGQmEwSXNSVUZCUlN4RFFVRlhMRVZCUVVVc1EwRkJWenRSUVVNNVJDeEpRVUZKTEVsQlFVa3NRMEZCUXl4UFFVRlBMRU5CUVVNc1RVRkJUU3hIUVVGSExFTkJRVU1zU1VGQlNTeFZRVUZWTEVkQlFVY3NRMEZCUXl4RlFVRkZPMWxCUnpORExFbEJRVWtzUTBGQlF5eEhRVUZITEVsQlFVa3NRMEZCUXl4TlFVRk5MRU5CUVVNc1RVRkJUU3hEUVVGRE8xbEJRek5DTEVsQlFVa3NTMEZCU3l4SFFVRkhMRWxCUVVrc1EwRkJReXhOUVVGTkxFTkJRVU1zUjBGQlJ5eERRVUZETEZWQlFVRXNRMEZCUXl4SlFVRkpMRTlCUVVFc1EwRkJTeXhGUVVGRkxFMUJRVTBzUlVGQlV5eERRVUZETEVOQlFVTXNUVUZCVHl4RFFVRkRMRXRCUVVzc1JVRkJSU3hOUVVGTkxFVkJRVk1zUTBGQlF5eERRVUZETEUxQlFVOHNRMEZCUXl4TFFVRkxMRVZCUVVVc1EwRkJRU3hGUVVGMlJTeERRVUYxUlN4RFFVRkRMRU5CUVVNN1dVRkRNVWNzU1VGQlNTeEZRVUZGTEVkQlFVY3NTVUZCU1N4RFFVRkRMRTFCUVUwc1EwRkJReXhIUVVGSExFTkJRVU1zVlVGQlFTeERRVUZETEVsQlFVa3NUMEZCUVN4RFFVRkxMRVZCUVVVc1MwRkJTeXhGUVVGRkxFTkJRVU1zUTBGQlF5eExRVUZMTEVWQlFVVXNRMEZCUVN4RlFVRjJRaXhEUVVGMVFpeERRVUZETEVOQlFVTTdXVUZEZGtRc1NVRkJTU3hEUVVGRExFOUJRVThzUTBGQlF5eFBRVUZQTEVOQlFVTXNWVUZCUXl4RFFVRkRMRVZCUVVVc1EwRkJRenRuUWtGRGRFSXNSVUZCUlN4RFFVRkRMRWxCUVVrc1EwRkJUU3hGUVVGRkxFdEJRVXNzUlVGQlJTeERRVUZETEVOQlFVTXNTMEZCU3l4SFFVRkhMRU5CUVVNc1IwRkJSeXhEUVVGRExFVkJRVVVzUTBGQlF5eERRVUZETzFsQlF6ZERMRU5CUVVNc1EwRkJReXhEUVVGRE8xbEJRMGdzU1VGQlNTeERRVUZETEU5QlFVOHNRMEZCUXl4UFFVRlBMRU5CUVVNc1ZVRkJReXhEUVVGRExFVkJRVVVzUTBGQlF6dG5Ra0ZEZEVJc1NVRkJTU3hQUVVGUExFTkJRVU1zUTBGQlF5eE5RVUZOTEV0QlFVc3NWMEZCVnp0dlFrRkRMMElzUTBGQlF5eERRVUZETEUxQlFVMHNRMEZCUXl4UFFVRlBMRU5CUVVNc1ZVRkJRU3hEUVVGRExFbEJRVWtzVDBGQlFTeExRVUZMTEVOQlFVTXNTVUZCU1N4RFFVRkRMRVZCUVVVc1RVRkJUU3hGUVVGRkxFTkJRVU1zUTBGQlF5eExRVUZMTEVWQlFVVXNUVUZCVFN4RlFVRkZMRU5CUVVNc1EwRkJReXhMUVVGTExFVkJRVVVzUTBGQlF5eEZRVUZvUkN4RFFVRm5SQ3hEUVVGRExFTkJRVU03WjBKQlF6VkZMRWxCUVVrc1QwRkJUeXhEUVVGRExFTkJRVU1zVFVGQlRTeExRVUZMTEZkQlFWYzdiMEpCUXk5Q0xFTkJRVU1zUTBGQlF5eE5RVUZOTEVOQlFVTXNUMEZCVHl4RFFVRkRMRlZCUVVFc1JVRkJSU3hKUVVGSkxFOUJRVUVzUzBGQlN5eERRVUZETEVsQlFVa3NRMEZCUXl4RlFVRkZMRTFCUVUwc1JVRkJSU3hEUVVGRExFTkJRVU1zUzBGQlN5eEZRVUZGTEUxQlFVMHNSVUZCUlN4RlFVRkZMRU5CUVVNc1MwRkJTeXhGUVVGRkxFTkJRVU1zUlVGQmFrUXNRMEZCYVVRc1EwRkJReXhEUVVGRE8xbEJRMnhHTEVOQlFVTXNRMEZCUXl4RFFVRkRPMWxCUjBnc1NVRkJTU3hOUVVGTkxFVkJRVVU3YVVKQlExQXNTVUZCU1N4RFFVRkRMRWxCUVVrc1EwRkJReXhKUVVGSkxFVkJRVVVzUTBGQlF6dHBRa0ZEYWtJc1MwRkJTeXhEUVVGRExFVkJRVVVzUTBGQlF6dHBRa0ZEVkN4TFFVRkxMRU5CUVVNc1MwRkJTeXhEUVVGRE8ybENRVU5hTEdGQlFXRXNRMEZCUXl4TFFVRkxMRU5CUVVNN2FVSkJRM0JDTEZsQlFWa3NRMEZCUXl4SlFVRkpMRU5CUVVNc1dVRkJXU3hGUVVGRkxFTkJRVU03YVVKQlEycERMSGRDUVVGM1FpeERRVUZETEVOQlFVTXNRMEZCUXp0cFFrRkRNMElzYjBKQlFXOUNMRU5CUVVNc1NVRkJTU3hEUVVGRE8ybENRVU14UWl4TFFVRkxMRU5CUVVNc1ZVRkJWU3hGUVVGRkxFTkJRVU1zUlVGQlJTeERRVUZETEVWQlFVVXNRMEZCUXl4RlFVRkZMRXRCUVVzc1EwRkJReXhEUVVGRE8xbEJSWFpETEVsQlFVa3NRMEZCUXl4TlFVRk5MRU5CUVVNc1QwRkJUeXhEUVVGRExGVkJRVUVzUTBGQlF6dG5Ra0ZEYWtJc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eExRVUZMTEVOQlFVTXNSMEZCUnl4RlFVRkZMRU5CUVVNc1EwRkJReXhEUVVGRExFdEJRVXNzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXp0blFrRkRNMElzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4TFFVRkxMRU5CUVVNc1IwRkJSeXhGUVVGRkxFTkJRVU1zUTBGQlF5eERRVUZETEV0QlFVc3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJRenRaUVVNdlFpeERRVUZETEVOQlFVTXNRMEZCUXp0VFFVTk9PMkZCUVUwN1dVRkRTQ3hKUVVGSkxFTkJRVU1zVVVGQlVTeERRVUZETEVkQlFVY3NRMEZCUXl4VlFVRlZMRU5CUVVNc1EwRkJRenRUUVVOcVF6dEpRVU5NTEVOQlFVTTdTVUZIVHl3NFEwRkJOa0lzUjBGQmNrTXNWVUZCYzBNc1MwRkJZU3hGUVVGRkxFMUJRV01zUlVGQlJTeFhRVUV5UWp0UlFVRm9SeXhwUWtGbFF6dFJRV1p2UlN3MFFrRkJRU3hGUVVGQkxHdENRVUV5UWp0UlFVVTFSaXhKUVVGSkxFTkJRVU1zU1VGQlNTeERRVUZETEdWQlFXVXNTVUZCU1N4SlFVRkpMRU5CUVVNc2JVSkJRVzFDTEVWQlFVVTdXVUZEYmtRc1NVRkJTU3hIUVVGRExFZEJRVWNzU1VGQlNTeERRVUZETEZGQlFWRXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFVkJRVVVzUjBGQlF5eEhRVUZITEVsQlFVa3NRMEZCUXl4UlFVRlJMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETzFsQlEyNUVMRWxCUVVrc1EwRkJReXhOUVVGTkxFTkJRVU1zVDBGQlR5eERRVUZETEZWQlFWVXNRMEZCUXl4RlFVRkZMRU5CUVVNc1NVRkJTU3hEUVVGRExFTkJRVU1zUTBGQlF5eEhRVUZITEVkQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1JVRkJSU3hEUVVGRExFTkJRVU1zUTBGQlF5eEhRVUZITEVkQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETzFsQlEycEZMRWxCUVVrc1RVRkJUU3hIUVVGSExHMURRVUZqTEVOQlFVTXNTVUZCU1N4RFFVRkRMRTFCUVUwc1JVRkJSU3hKUVVGSkxFTkJRVU1zVFVGQlRTeERRVUZETEVOQlFVTTdXVUZEZEVRc2FVTkJRVmtzUTBGQlF5eE5RVUZOTEVWQlFVVXNTMEZCU3l4RlFVRkZMRTFCUVUwc1JVRkJSU3hKUVVGSkxFTkJRVU1zWjBKQlFXZENMRVZCUVVVc1EwRkJReXhGUVVGRkxGZEJRVmNzUTBGQlF5eERRVUZETzFsQlF6TkZMRWxCUVVrc1EwRkJReXhOUVVGTkxFTkJRVU1zVDBGQlR5eERRVUZETEZWQlFVTXNRMEZCUXl4RlFVRkZMRU5CUVVNN1owSkJRM0pDTEV0QlFVa3NRMEZCUXl4UlFVRlJMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4SFFVRkhMRU5CUVVNc1EwRkJReXhEUVVGRExFVkJRVVVzUzBGQlNTeERRVUZETEZGQlFWRXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVkQlFVY3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJRenRuUWtGRGVrUXNTVUZCU1N4RFFVRkRMRU5CUVVNc1RVRkJUU3hGUVVGRk8yOUNRVU5XTEVOQlFVTXNRMEZCUXl4TlFVRk5MRU5CUVVNc1ZVRkJWU3hEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXp0dlFrRkRla0lzUTBGQlF5eERRVUZETEUxQlFVMHNRMEZCUXl4VlFVRlZMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETzJsQ1FVTTFRanRaUVVOTUxFTkJRVU1zUTBGQlF5eERRVUZETzFOQlEwNDdTVUZEVEN4RFFVRkRPMGxCUlVRc2RVSkJRVTBzUjBGQlRqdFJRVU5KTEU5QlFVOHNTVUZCU1N4RFFVRkRMRXRCUVVzc1EwRkJReXhIUVVGSExFTkJRVU1zUTBGQlF6dEpRVU16UWl4RFFVRkRPMGxCUlVRc2NVSkJRVWtzUjBGQlNqdFJRVU5KTEU5QlFVOHNTVUZCU1N4RFFVRkRMRXRCUVVzc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dEpRVU42UWl4RFFVRkRPMGxCU1VRc2JVTkJRV3RDTEVkQlFXeENMRlZCUVcxQ0xGVkJRWE5DTzFGQlFYUkNMREpDUVVGQkxFVkJRVUVzWTBGQmMwSTdVVUZEY2tNc1NVRkJTU3hEUVVGRExHZENRVUZuUWl4SFFVRkhMRWxCUVVrc05rSkJRWE5DTEVOQlF6bERMRWxCUVVrc1EwRkJReXhOUVVGTkxFTkJRVU1zUjBGQlJ5eERRVUZETEZWQlFWVXNRMEZCUXp0WlFVTjJRaXhQUVVGUExFTkJRVU1zUTBGQlF5eE5RVUZOTEVOQlFVTXNUMEZCVHl4RFFVRkRMRU5CUVVNc1ZVRkJWU3hEUVVGRExFTkJRVU1zVVVGQlVTeEZRVUZGTEVOQlFVTTdVVUZEY0VRc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dEpRVU5hTEVOQlFVTTdTVUZYUkN3d1FrRkJVeXhIUVVGVUxGVkJRVlVzU1VGQlNTeEZRVUZGTEVWQlFXTXNSVUZCUlN4SlFVRkpPMUZCUVhCQ0xHMUNRVUZCTEVWQlFVRXNUVUZCWXp0UlFVTXhRaXhKUVVGSkxGRkJRVkVzUjBGQlJ5eEZRVUZGTEVOQlFVTTdVVUZKYkVJc1NVRkJTU3hIUVVGSExFZEJRVWNzU1VGQlNTdzJRa0ZCYzBJc1EwRkJReXhKUVVGSkxFTkJRVU1zWjBKQlFXZENMRU5CUVVNc1EwRkJReXhGUVVGRkxFVkJRVVVzUTBGQlF5eEZRVUZGTEVsQlFVa3NRMEZCUXl4blFrRkJaMElzUTBGQlF5eERRVUZETEVWQlFVVXNRMEZCUXl4RlFVRkZMRWxCUVVrc1EwRkJReXhuUWtGQlowSXNRMEZCUXl4RFFVRkRMRVZCUVVVc1EwRkJReXhGUVVOeVNDeExRVUZMTEVkQlFXRXNSVUZCUlN4RFFVRkRMRVZCUVVVc1NVRkJTU3hEUVVGRExFMUJRVTBzUTBGQlF5eERRVUZETEVWQlFVVXNRMEZCUXl4RlFVRkZMRWxCUVVrc1EwRkJReXhOUVVGTkxFTkJRVU1zUTBGQlF5eEZRVUZGTEVWQlEzaEVMRXRCUVVzc1IwRkJZU3hGUVVGRkxFTkJRVU1zUlVGQlJTeEpRVUZKTEVOQlFVTXNUVUZCVFN4RFFVRkRMRU5CUVVNc1JVRkJSU3hEUVVGRExFVkJRVVVzU1VGQlNTeERRVUZETEUxQlFVMHNRMEZCUXl4RFFVRkRMRVZCUVVVc1JVRkRlRVFzUzBGQlN5eEhRVUZITEVkQlFVY3NRMEZCUXl4UlFVRlJMRU5CUVVNc1MwRkJTeXhGUVVGRkxFbEJRVWtzUTBGQlF5eE5RVUZOTEVOQlFVTXNTMEZCU3l4RFFVRkRMRVZCUXpsRExFZEJRVWNzUjBGQlJ5eEhRVUZITEVOQlFVTXNVVUZCVVN4RFFVRkRMRXRCUVVzc1JVRkJSU3hKUVVGSkxFTkJRVU1zVFVGQlRTeERRVUZETEV0QlFVc3NRMEZCUXl4RFFVRkRPMUZCUTJwRUxFZEJRVWNzUTBGQlF5eG5Ra0ZCWjBJc1EwRkJReXhMUVVGTExFVkJRVVVzUzBGQlN5eEZRVUZGTEVsQlFVa3NRMEZCUXl4TlFVRk5MRU5CUVVNc1MwRkJTeXhGUVVGRkxFbEJRVWtzUTBGQlF5eE5RVUZOTEVOQlFVTXNTMEZCU3l4RFFVRkRMRU5CUVVNN1VVRkRla1VzU1VGQlNTeFBRVUZQTEVsQlFVa3NTMEZCU3l4WFFVRlhMRVZCUVVVN1dVRkROMElzU1VGQlNTeERRVUZETEVkQlFVY3NRMEZCUXl4RFFVRkRPMU5CUTJJN1VVRkRSQ3hKUVVGSkxGTkJRVk1zUjBGQlJ5eFZRVUZCTEVOQlFVTXNTVUZCU1N4UFFVRkJMRU5CUVVNc1EwRkJReXhOUVVGTkxFTkJRVU1zUlVGQlJTeEZRVUZZTEVOQlFWY3NSVUZCUlN4VFFVRlRMRWRCUVVjc1ZVRkJRU3hEUVVGRExFbEJRVWtzVDBGQlFTeERRVUZETEVOQlFVTXNUVUZCVFN4RFFVRkRMRVZCUVVVc1JVRkJXQ3hEUVVGWExFVkJRVVVzVFVGQlRTeEhRVUZITEZWQlFVRXNRMEZCUXl4SlFVRkpMRTlCUVVFc1EwRkJReXhEUVVGRExFMUJRVTBzUlVGQlJTeEZRVUZXTEVOQlFWVXNSVUZEY0VZc1RVRkJUU3hIUVVGSExFbEJRVWtzTUVKQlFWVXNRMEZCUXl4SFFVRkhMRU5CUVVNc1EwRkJReXhEUVVGRExFMUJRVTBzUlVGQlJTeEhRVUZITEVOQlFVTXNRMEZCUXl4RlFVRkZMRk5CUVZNc1JVRkJSU3hUUVVGVExFVkJRVVVzVFVGQlRTeERRVUZETEVWQlF6RkZMRmxCUVZrc1IwRkJSeXhOUVVGTkxFTkJRVU1zYTBKQlFXdENMRU5CUVVNc1MwRkJTeXhEUVVGRExFVkJRVVVzUlVGQlJTeEhRVUZITEVOQlFVTXNSVUZCUlN4RFFVRkRMRU5CUVVNN1VVRkRMMFFzU1VGQlNTeFpRVUZaTEVOQlFVTXNUVUZCVFN4TFFVRkxMRU5CUVVNc1NVRkJTU3haUVVGWkxFTkJRVU1zVFVGQlRTeExRVUZMTEVkQlFVY3NRMEZCUXl4RFFVRkRMRU5CUVVNc1RVRkJUU3hGUVVGRk8xbEJRMjVGTEVsQlFVa3NTMEZCU3l4SFFVRkhMREpDUVVGbExFTkJRVU1zU1VGQlNTeERRVUZETEUxQlFVMHNRMEZCUXl4WFFVRlhMRVZCUVVVc1NVRkJTU3hEUVVGRExFMUJRVTBzUTBGQlF5eFhRVUZYTEVWQlFVVXNSVUZCUlN4RFFVRkRMRU5CUVVNN1dVRkRiRVlzVVVGQlVTeEhRVUZITEVOQlFVTXNTMEZCU3l4RFFVRkRMR3RDUVVGclFpeEZRVUZGTEV0QlFVc3NRMEZCUXl4VlFVRlZMRU5CUVVNc1EwRkJRenRUUVVNelJEdGhRVUZOTzFsQlEwZ3NTVUZCU1N4RFFVRkRMRWRCUVVjc1dVRkJXU3hEUVVGRExFMUJRVTBzUjBGQlJ5eERRVUZETEVWQlF6TkNMRU5CUVVNc1IwRkJSeXhIUVVGSExFTkJRVU1zUTBGQlF5eERRVUZETEZsQlFWa3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUlVGRE5VSXNRMEZCUXl4SFFVRkhMRWRCUVVjc1EwRkJReXhEUVVGRExFTkJRVU1zV1VGQldTeERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhGUVVNMVFpeFJRVUZSTEVkQlFVY3NRMEZCUXl4SlFVRkpMRU5CUVVNc1RVRkJUU3hEUVVGRExGZEJRVmNzUTBGQlF5eGxRVUZsTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1JVRkJSU3hEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXp0WlFVTnVSU3hMUVVGTExFbEJRVWtzUTBGQlF5eEhRVUZITEVOQlFVTXNSVUZCUlN4RFFVRkRMRWxCUVVrc1EwRkJReXhGUVVGRkxFVkJRVVVzUTBGQlF6dG5Ra0ZEZGtJc1VVRkJVU3hEUVVGRExFbEJRVWtzUTBGQlF5eEhRVUZITEVOQlFVTXNRMEZCUXl4RFFVRkRMRmxCUVZrc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRPMWxCUXpWRExGRkJRVkVzUTBGQlF5eEpRVUZKTEVOQlFVTXNjMEpCUVZVc1EwRkJReXhEUVVGRExFVkJRVVVzU1VGQlNTeERRVUZETEUxQlFVMHNRMEZCUXl4WFFVRlhMRVZCUVVVc1JVRkJSU3hEUVVGRExFTkJRVU1zUTBGQlF6dFRRVU0zUkR0UlFXRkVMRTlCUVU4c1VVRkJVU3hEUVVGRE8wbEJRM0JDTEVOQlFVTTdTVUZIVFN4eFFrRkJZeXhIUVVGeVFpeFZRVUZ6UWl4RFFVRnpRanRSUVVONFF5eFBRVUZQTEU5QlFVOHNRMEZCUXl4RFFVRkRMRTFCUVUwc1MwRkJTeXhSUVVGUkxFTkJRVU1zUTBGQlF5eERRVUZUTEVOQlFVTXNRMEZCUXl4TlFVRk5MRU5CUVVNc1EwRkJReXhEUVVGUkxFTkJRVU1zUTBGQlF5eE5RVUZQTEVOQlFVTXNTMEZCU3l4RFFVRkRPMGxCUTNCR0xFTkJRVU03U1VGSFRTeHhRa0ZCWXl4SFFVRnlRaXhWUVVGelFpeERRVUZ6UWp0UlFVTjRReXhQUVVGUExFOUJRVThzUTBGQlF5eERRVUZETEUxQlFVMHNTMEZCU3l4UlFVRlJMRU5CUVVNc1EwRkJReXhEUVVGVExFTkJRVU1zUTBGQlF5eE5RVUZOTEVOQlFVTXNRMEZCUXl4RFFVRlJMRU5CUVVNc1EwRkJReXhOUVVGUExFTkJRVU1zUzBGQlN5eERRVUZETzBsQlEzQkdMRU5CUVVNN1NVRkhUU3hoUVVGTkxFZEJRV0lzVlVGQll5eERRVUZ6UWp0UlFVTm9ReXhQUVVGUExFMUJRVTBzUTBGQlF5eGpRVUZqTEVOQlFVTXNRMEZCUXl4RFFVRkRMRWRCUVVjc1IwRkJSeXhIUVVGSExFMUJRVTBzUTBGQlF5eGpRVUZqTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNN1NVRkRja1VzUTBGQlF6dEpRVTFOTEdkQ1FVRlRMRWRCUVdoQ0xGVkJRV2xDTEVOQlFXVTdVVUZETlVJc1NVRkJTU3hQUVVGUExFTkJRVU1zUTBGQlF5eERRVUZETEVWQlFVVTdXVUZEV2l4TlFVRk5MRU5CUVVNc1YwRkJWeXhEUVVGRExFTkJRVU1zUlVGQlJTeE5RVUZOTEVOQlFVTXNWVUZCVlN4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU03VTBGREwwTTdZVUZCVFR0WlFVTklMRTFCUVUwc1EwRkJReXhSUVVGUkxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTTdXVUZEYmtJc1EwRkJReXhEUVVGRExFdEJRVXNzU1VGQlNTeERRVUZETEVOQlFVTTdVMEZEYUVJN1NVRkRUQ3hEUVVGRE8wbEJTV01zWlVGQlVTeEhRVUYyUWl4VlFVRjNRaXhEUVVGUE8xRkJRM0pDTEVOQlFVVXNRMEZCUXl4RlFVRkZMRWRCUVVjc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dFJRVU5hTEVOQlFVVXNRMEZCUXl4RlFVRkZMRWRCUVVjc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dEpRVU4wUWl4RFFVRkRPMGxCU1dNc2EwSkJRVmNzUjBGQk1VSXNWVUZCTWtJc1EwRkJVU3hGUVVGRkxFMUJRV2RETzFGQlEycEZMRWxCUVVrc1QwRkJUeXhEUVVGRExFTkJRVU1zVFVGQlRTeExRVUZMTEZkQlFWY3NSVUZCUlR0WlFVTnFReXhEUVVGRExFTkJRVU1zVFVGQlRTeERRVUZETEU5QlFVOHNRMEZCUXl4VlFVRkJMRU5CUVVNN1owSkJRMlFzUTBGQlF5eERRVUZETEV0QlFVc3NTVUZCU1N4RFFVRkRMRU5CUVVNN1owSkJRMklzVFVGQlRTeERRVUZETEZGQlFWRXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJRenRuUWtGRFlpeERRVUZGTEVOQlFVTXNhVUpCUVdsQ0xFZEJRVWNzUTBGQlF5eERRVUZETEVOQlFVTXNSMEZCUnl4TlFVRk5MRU5CUVVNc1EwRkJReXhEUVVGRE8yZENRVU4wUXl4RFFVRkZMRU5CUVVNc2FVSkJRV2xDTEVkQlFVY3NRMEZCUXl4RFFVRkRMRU5CUVVNc1IwRkJSeXhOUVVGTkxFTkJRVU1zUTBGQlF5eERRVUZETzFsQlEyaEVMRU5CUVVNc1EwRkJReXhEUVVGRE8xTkJRMDQ3VVVGRFJDeEpRVUZKTEU5QlFVOHNRMEZCUXl4RFFVRkRMRTFCUVUwc1MwRkJTeXhYUVVGWExFVkJRVVU3V1VGRGFrTXNRMEZCUXl4RFFVRkRMRTFCUVUwc1EwRkJReXhQUVVGUExFTkJRVU1zVlVGQlFTeERRVUZETEVsQlFVa3NUMEZCUVN4TlFVRk5MRU5CUVVNc1YwRkJWeXhEUVVGRExFTkJRVU1zUlVGQlJTeE5RVUZOTEVOQlFVTXNSVUZCTjBJc1EwRkJOa0lzUTBGQlF5eERRVUZETzFOQlEzaEVPMGxCUTB3c1EwRkJRenRKUVVkTkxHbENRVUZWTEVkQlFXcENMRlZCUVd0Q0xFTkJRV1U3VVVGRE4wSXNTVUZCU1N4UFFVRlBMRU5CUVVNc1EwRkJReXhEUVVGRExFVkJRVVU3V1VGRFdpeFBRVUZQTzJkQ1FVTklMRU5CUVVNc1JVRkJSU3hEUVVGRExFTkJRVU1zVFVGQlRTeERRVUZETEVWQlFVVXNSVUZCUlR0blFrRkRhRUlzUTBGQlF5eEZRVUZGTEVOQlFVTXNRMEZCUXl4TlFVRk5MRU5CUVVNc1JVRkJSU3hGUVVGRk8yRkJRMjVDTEVOQlFVTTdVMEZEVER0aFFVRk5PMWxCUTBnc1QwRkJUeXhEUVVGRExFTkJRVU03VTBGRFdqdEpRVU5NTEVOQlFVTTdTVUZKVFN4WFFVRkpMRWRCUVZnc1ZVRkJXU3hEUVVGbExFVkJRVVVzVVVGQmEwTTdVVUZETTBRc1NVRkJTU3hQUVVGUExFTkJRVU1zUTBGQlF5eERRVUZETEVWQlFVVTdXVUZEV2l4SlFVRkpMRTlCUVU4c1EwRkJReXhEUVVGRExFMUJRVTBzUzBGQlN5eFhRVUZYTEVWQlFVVTdaMEpCUTJwRExFTkJRVU1zUTBGQlF5eE5RVUZOTEVOQlFVTXNUMEZCVHl4RFFVRkRMRlZCUVVFc1EwRkJRenR2UWtGRFpDeERRVUZETEVOQlFVTXNUVUZCVFN4RFFVRkRMRlZCUVZVc1EwRkJReXhSUVVGUkxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTTdiMEpCUTJoRExFTkJRVU1zUTBGQlF5eE5RVUZOTEVOQlFVTXNWVUZCVlN4RFFVRkRMRkZCUVZFc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dHZRa0ZETVVJc1EwRkJSU3hEUVVGRExFVkJRVVVzUjBGQlV5eERRVUZGTEVOQlFVTXNhVUpCUVdsQ0xFZEJRVWNzVVVGQlVTeERRVUZETEVOQlFVTXNRMEZCUXp0dlFrRkRhRVFzUTBGQlJTeERRVUZETEVWQlFVVXNSMEZCVXl4RFFVRkZMRU5CUVVNc2FVSkJRV2xDTEVkQlFVY3NVVUZCVVN4RFFVRkRMRU5CUVVNc1EwRkJRenRuUWtGRE1VUXNRMEZCUXl4RFFVRkRMRU5CUVVNN1lVRkRUanRaUVVORUxFbEJRVWtzVDBGQlR5eERRVUZETEVOQlFVTXNUVUZCVFN4TFFVRkxMRmRCUVZjc1JVRkJSVHRuUWtGRGFrTXNRMEZCUXl4RFFVRkRMRTFCUVUwc1EwRkJReXhQUVVGUExFTkJRVU1zVlVGQlFTeERRVUZETEVsQlFVa3NUMEZCUVN4TlFVRk5MRU5CUVVNc1NVRkJTU3hEUVVGRExFTkJRVU1zUlVGQlJTeFJRVUZSTEVOQlFVTXNSVUZCZUVJc1EwRkJkMElzUTBGQlF5eERRVUZETzJGQlEyNUVPMU5CUTBvN1lVRkJUVHRaUVVOSExFTkJRVVVzUTBGQlF5eEZRVUZGTEVkQlFVY3NVVUZCVVN4RFFVRkRMRU5CUVVNc1EwRkJRenRaUVVOdVFpeERRVUZGTEVOQlFVTXNSVUZCUlN4SFFVRkhMRkZCUVZFc1EwRkJReXhEUVVGRExFTkJRVU03VTBGRE5VSTdTVUZEVEN4RFFVRkRPMGxCU1Uwc1kwRkJUeXhIUVVGa0xGVkJRV1VzUTBGQlF6dFJRVU5hTEVsQlFVa3NUMEZCVHl4RFFVRkRMRU5CUVVNc1EwRkJReXhGUVVGRk8xbEJRMW9zU1VGQlNTeFBRVUZQTEVOQlFVTXNRMEZCUXl4TlFVRk5MRXRCUVVzc1YwRkJWeXhGUVVGRk8yZENRVU5xUXl4RFFVRkRMRU5CUVVNc1RVRkJUU3hEUVVGRExFOUJRVThzUTBGQlF5eFZRVUZCTEVOQlFVTTdiMEpCUTJRc1RVRkJUU3hEUVVGRExFOUJRVThzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXp0dlFrRkRiRUlzVDBGQllTeERRVUZGTEVOQlFVTXNhVUpCUVdsQ0xFTkJRVU03YjBKQlEyeERMRTlCUVdFc1EwRkJSU3hEUVVGRExHbENRVUZwUWl4RFFVRkRPMmRDUVVOMFF5eERRVUZETEVOQlFVTXNRMEZCUXp0aFFVTk9PMWxCUTBRc1NVRkJTU3hQUVVGUExFTkJRVU1zUTBGQlF5eE5RVUZOTEV0QlFVc3NWMEZCVnl4RlFVRkZPMmRDUVVOcVF5eERRVUZETEVOQlFVTXNUVUZCVFN4RFFVRkRMRTlCUVU4c1EwRkJReXhOUVVGTkxFTkJRVU1zVDBGQlR5eERRVUZETEVOQlFVTTdZVUZEY0VNN1UwRkRTanRoUVVGTk8xbEJRMGdzUTBGQlF5eERRVUZETEV0QlFVc3NTVUZCU1N4RFFVRkRMRU5CUVVNc1EwRkJRenRUUVVWcVFqdEpRVU5NTEVOQlFVTTdTVUZIVFN4blFrRkJVeXhIUVVGb1FpeFZRVUZwUWl4RFFVRkRPMUZCUTJRc1EwRkJReXhEUVVGRExFdEJRVXNzU1VGQlNTeERRVUZETEVOQlFVTTdVVUZEWWl4RFFVRkRMRU5CUVVNc1JVRkJSU3hIUVVGSExFTkJRVU1zUTBGQlF5eERRVUZETEVWQlFVVXNRMEZCUXl4RFFVRkRMRVZCUVVVc1IwRkJSeXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETzBsQlF6TkNMRU5CUVVNN1NVRkhUU3hsUVVGUkxFZEJRV1lzVlVGQlowSXNRMEZCUXp0UlFVTmlMRU5CUVVNc1EwRkJReXhMUVVGTExFbEJRVWtzUTBGQlF5eERRVUZETEVOQlFVTTdTVUZEYkVJc1EwRkJRenRKUVVOTUxHRkJRVU03UVVGQlJDeERRVUZETEVGQmJIbENSQ3hKUVd0NVFrTTdRVUZzZVVKWkxIZENRVUZOSW4wPSIsIlwidXNlIHN0cmljdFwiO1xyXG5PYmplY3QuZGVmaW5lUHJvcGVydHkoZXhwb3J0cywgXCJfX2VzTW9kdWxlXCIsIHsgdmFsdWU6IHRydWUgfSk7XHJcbnZhciBzaG9ydGVzdHBhdGhzXzEgPSByZXF1aXJlKFwiLi9zaG9ydGVzdHBhdGhzXCIpO1xyXG52YXIgZGVzY2VudF8xID0gcmVxdWlyZShcIi4vZGVzY2VudFwiKTtcclxudmFyIHJlY3RhbmdsZV8xID0gcmVxdWlyZShcIi4vcmVjdGFuZ2xlXCIpO1xyXG52YXIgbGlua2xlbmd0aHNfMSA9IHJlcXVpcmUoXCIuL2xpbmtsZW5ndGhzXCIpO1xyXG52YXIgTGluazNEID0gKGZ1bmN0aW9uICgpIHtcclxuICAgIGZ1bmN0aW9uIExpbmszRChzb3VyY2UsIHRhcmdldCkge1xyXG4gICAgICAgIHRoaXMuc291cmNlID0gc291cmNlO1xyXG4gICAgICAgIHRoaXMudGFyZ2V0ID0gdGFyZ2V0O1xyXG4gICAgfVxyXG4gICAgTGluazNELnByb3RvdHlwZS5hY3R1YWxMZW5ndGggPSBmdW5jdGlvbiAoeCkge1xyXG4gICAgICAgIHZhciBfdGhpcyA9IHRoaXM7XHJcbiAgICAgICAgcmV0dXJuIE1hdGguc3FydCh4LnJlZHVjZShmdW5jdGlvbiAoYywgdikge1xyXG4gICAgICAgICAgICB2YXIgZHggPSB2W190aGlzLnRhcmdldF0gLSB2W190aGlzLnNvdXJjZV07XHJcbiAgICAgICAgICAgIHJldHVybiBjICsgZHggKiBkeDtcclxuICAgICAgICB9LCAwKSk7XHJcbiAgICB9O1xyXG4gICAgcmV0dXJuIExpbmszRDtcclxufSgpKTtcclxuZXhwb3J0cy5MaW5rM0QgPSBMaW5rM0Q7XHJcbnZhciBOb2RlM0QgPSAoZnVuY3Rpb24gKCkge1xyXG4gICAgZnVuY3Rpb24gTm9kZTNEKHgsIHksIHopIHtcclxuICAgICAgICBpZiAoeCA9PT0gdm9pZCAwKSB7IHggPSAwOyB9XHJcbiAgICAgICAgaWYgKHkgPT09IHZvaWQgMCkgeyB5ID0gMDsgfVxyXG4gICAgICAgIGlmICh6ID09PSB2b2lkIDApIHsgeiA9IDA7IH1cclxuICAgICAgICB0aGlzLnggPSB4O1xyXG4gICAgICAgIHRoaXMueSA9IHk7XHJcbiAgICAgICAgdGhpcy56ID0gejtcclxuICAgIH1cclxuICAgIHJldHVybiBOb2RlM0Q7XHJcbn0oKSk7XHJcbmV4cG9ydHMuTm9kZTNEID0gTm9kZTNEO1xyXG52YXIgTGF5b3V0M0QgPSAoZnVuY3Rpb24gKCkge1xyXG4gICAgZnVuY3Rpb24gTGF5b3V0M0Qobm9kZXMsIGxpbmtzLCBpZGVhbExpbmtMZW5ndGgpIHtcclxuICAgICAgICB2YXIgX3RoaXMgPSB0aGlzO1xyXG4gICAgICAgIGlmIChpZGVhbExpbmtMZW5ndGggPT09IHZvaWQgMCkgeyBpZGVhbExpbmtMZW5ndGggPSAxOyB9XHJcbiAgICAgICAgdGhpcy5ub2RlcyA9IG5vZGVzO1xyXG4gICAgICAgIHRoaXMubGlua3MgPSBsaW5rcztcclxuICAgICAgICB0aGlzLmlkZWFsTGlua0xlbmd0aCA9IGlkZWFsTGlua0xlbmd0aDtcclxuICAgICAgICB0aGlzLmNvbnN0cmFpbnRzID0gbnVsbDtcclxuICAgICAgICB0aGlzLnVzZUphY2NhcmRMaW5rTGVuZ3RocyA9IHRydWU7XHJcbiAgICAgICAgdGhpcy5yZXN1bHQgPSBuZXcgQXJyYXkoTGF5b3V0M0Quayk7XHJcbiAgICAgICAgZm9yICh2YXIgaSA9IDA7IGkgPCBMYXlvdXQzRC5rOyArK2kpIHtcclxuICAgICAgICAgICAgdGhpcy5yZXN1bHRbaV0gPSBuZXcgQXJyYXkobm9kZXMubGVuZ3RoKTtcclxuICAgICAgICB9XHJcbiAgICAgICAgbm9kZXMuZm9yRWFjaChmdW5jdGlvbiAodiwgaSkge1xyXG4gICAgICAgICAgICBmb3IgKHZhciBfaSA9IDAsIF9hID0gTGF5b3V0M0QuZGltczsgX2kgPCBfYS5sZW5ndGg7IF9pKyspIHtcclxuICAgICAgICAgICAgICAgIHZhciBkaW0gPSBfYVtfaV07XHJcbiAgICAgICAgICAgICAgICBpZiAodHlwZW9mIHZbZGltXSA9PSAndW5kZWZpbmVkJylcclxuICAgICAgICAgICAgICAgICAgICB2W2RpbV0gPSBNYXRoLnJhbmRvbSgpO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIF90aGlzLnJlc3VsdFswXVtpXSA9IHYueDtcclxuICAgICAgICAgICAgX3RoaXMucmVzdWx0WzFdW2ldID0gdi55O1xyXG4gICAgICAgICAgICBfdGhpcy5yZXN1bHRbMl1baV0gPSB2Lno7XHJcbiAgICAgICAgfSk7XHJcbiAgICB9XHJcbiAgICA7XHJcbiAgICBMYXlvdXQzRC5wcm90b3R5cGUubGlua0xlbmd0aCA9IGZ1bmN0aW9uIChsKSB7XHJcbiAgICAgICAgcmV0dXJuIGwuYWN0dWFsTGVuZ3RoKHRoaXMucmVzdWx0KTtcclxuICAgIH07XHJcbiAgICBMYXlvdXQzRC5wcm90b3R5cGUuc3RhcnQgPSBmdW5jdGlvbiAoaXRlcmF0aW9ucykge1xyXG4gICAgICAgIHZhciBfdGhpcyA9IHRoaXM7XHJcbiAgICAgICAgaWYgKGl0ZXJhdGlvbnMgPT09IHZvaWQgMCkgeyBpdGVyYXRpb25zID0gMTAwOyB9XHJcbiAgICAgICAgdmFyIG4gPSB0aGlzLm5vZGVzLmxlbmd0aDtcclxuICAgICAgICB2YXIgbGlua0FjY2Vzc29yID0gbmV3IExpbmtBY2Nlc3NvcigpO1xyXG4gICAgICAgIGlmICh0aGlzLnVzZUphY2NhcmRMaW5rTGVuZ3RocylcclxuICAgICAgICAgICAgbGlua2xlbmd0aHNfMS5qYWNjYXJkTGlua0xlbmd0aHModGhpcy5saW5rcywgbGlua0FjY2Vzc29yLCAxLjUpO1xyXG4gICAgICAgIHRoaXMubGlua3MuZm9yRWFjaChmdW5jdGlvbiAoZSkgeyByZXR1cm4gZS5sZW5ndGggKj0gX3RoaXMuaWRlYWxMaW5rTGVuZ3RoOyB9KTtcclxuICAgICAgICB2YXIgZGlzdGFuY2VNYXRyaXggPSAobmV3IHNob3J0ZXN0cGF0aHNfMS5DYWxjdWxhdG9yKG4sIHRoaXMubGlua3MsIGZ1bmN0aW9uIChlKSB7IHJldHVybiBlLnNvdXJjZTsgfSwgZnVuY3Rpb24gKGUpIHsgcmV0dXJuIGUudGFyZ2V0OyB9LCBmdW5jdGlvbiAoZSkgeyByZXR1cm4gZS5sZW5ndGg7IH0pKS5EaXN0YW5jZU1hdHJpeCgpO1xyXG4gICAgICAgIHZhciBEID0gZGVzY2VudF8xLkRlc2NlbnQuY3JlYXRlU3F1YXJlTWF0cml4KG4sIGZ1bmN0aW9uIChpLCBqKSB7IHJldHVybiBkaXN0YW5jZU1hdHJpeFtpXVtqXTsgfSk7XHJcbiAgICAgICAgdmFyIEcgPSBkZXNjZW50XzEuRGVzY2VudC5jcmVhdGVTcXVhcmVNYXRyaXgobiwgZnVuY3Rpb24gKCkgeyByZXR1cm4gMjsgfSk7XHJcbiAgICAgICAgdGhpcy5saW5rcy5mb3JFYWNoKGZ1bmN0aW9uIChfYSkge1xyXG4gICAgICAgICAgICB2YXIgc291cmNlID0gX2Euc291cmNlLCB0YXJnZXQgPSBfYS50YXJnZXQ7XHJcbiAgICAgICAgICAgIHJldHVybiBHW3NvdXJjZV1bdGFyZ2V0XSA9IEdbdGFyZ2V0XVtzb3VyY2VdID0gMTtcclxuICAgICAgICB9KTtcclxuICAgICAgICB0aGlzLmRlc2NlbnQgPSBuZXcgZGVzY2VudF8xLkRlc2NlbnQodGhpcy5yZXN1bHQsIEQpO1xyXG4gICAgICAgIHRoaXMuZGVzY2VudC50aHJlc2hvbGQgPSAxZS0zO1xyXG4gICAgICAgIHRoaXMuZGVzY2VudC5HID0gRztcclxuICAgICAgICBpZiAodGhpcy5jb25zdHJhaW50cylcclxuICAgICAgICAgICAgdGhpcy5kZXNjZW50LnByb2plY3QgPSBuZXcgcmVjdGFuZ2xlXzEuUHJvamVjdGlvbih0aGlzLm5vZGVzLCBudWxsLCBudWxsLCB0aGlzLmNvbnN0cmFpbnRzKS5wcm9qZWN0RnVuY3Rpb25zKCk7XHJcbiAgICAgICAgZm9yICh2YXIgaSA9IDA7IGkgPCB0aGlzLm5vZGVzLmxlbmd0aDsgaSsrKSB7XHJcbiAgICAgICAgICAgIHZhciB2ID0gdGhpcy5ub2Rlc1tpXTtcclxuICAgICAgICAgICAgaWYgKHYuZml4ZWQpIHtcclxuICAgICAgICAgICAgICAgIHRoaXMuZGVzY2VudC5sb2Nrcy5hZGQoaSwgW3YueCwgdi55LCB2LnpdKTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH1cclxuICAgICAgICB0aGlzLmRlc2NlbnQucnVuKGl0ZXJhdGlvbnMpO1xyXG4gICAgICAgIHJldHVybiB0aGlzO1xyXG4gICAgfTtcclxuICAgIExheW91dDNELnByb3RvdHlwZS50aWNrID0gZnVuY3Rpb24gKCkge1xyXG4gICAgICAgIHRoaXMuZGVzY2VudC5sb2Nrcy5jbGVhcigpO1xyXG4gICAgICAgIGZvciAodmFyIGkgPSAwOyBpIDwgdGhpcy5ub2Rlcy5sZW5ndGg7IGkrKykge1xyXG4gICAgICAgICAgICB2YXIgdiA9IHRoaXMubm9kZXNbaV07XHJcbiAgICAgICAgICAgIGlmICh2LmZpeGVkKSB7XHJcbiAgICAgICAgICAgICAgICB0aGlzLmRlc2NlbnQubG9ja3MuYWRkKGksIFt2LngsIHYueSwgdi56XSk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICAgICAgcmV0dXJuIHRoaXMuZGVzY2VudC5ydW5nZUt1dHRhKCk7XHJcbiAgICB9O1xyXG4gICAgTGF5b3V0M0QuZGltcyA9IFsneCcsICd5JywgJ3onXTtcclxuICAgIExheW91dDNELmsgPSBMYXlvdXQzRC5kaW1zLmxlbmd0aDtcclxuICAgIHJldHVybiBMYXlvdXQzRDtcclxufSgpKTtcclxuZXhwb3J0cy5MYXlvdXQzRCA9IExheW91dDNEO1xyXG52YXIgTGlua0FjY2Vzc29yID0gKGZ1bmN0aW9uICgpIHtcclxuICAgIGZ1bmN0aW9uIExpbmtBY2Nlc3NvcigpIHtcclxuICAgIH1cclxuICAgIExpbmtBY2Nlc3Nvci5wcm90b3R5cGUuZ2V0U291cmNlSW5kZXggPSBmdW5jdGlvbiAoZSkgeyByZXR1cm4gZS5zb3VyY2U7IH07XHJcbiAgICBMaW5rQWNjZXNzb3IucHJvdG90eXBlLmdldFRhcmdldEluZGV4ID0gZnVuY3Rpb24gKGUpIHsgcmV0dXJuIGUudGFyZ2V0OyB9O1xyXG4gICAgTGlua0FjY2Vzc29yLnByb3RvdHlwZS5nZXRMZW5ndGggPSBmdW5jdGlvbiAoZSkgeyByZXR1cm4gZS5sZW5ndGg7IH07XHJcbiAgICBMaW5rQWNjZXNzb3IucHJvdG90eXBlLnNldExlbmd0aCA9IGZ1bmN0aW9uIChlLCBsKSB7IGUubGVuZ3RoID0gbDsgfTtcclxuICAgIHJldHVybiBMaW5rQWNjZXNzb3I7XHJcbn0oKSk7XHJcbi8vIyBzb3VyY2VNYXBwaW5nVVJMPWRhdGE6YXBwbGljYXRpb24vanNvbjtiYXNlNjQsZXlKMlpYSnphVzl1SWpvekxDSm1hV3hsSWpvaWJHRjViM1YwTTJRdWFuTWlMQ0p6YjNWeVkyVlNiMjkwSWpvaUlpd2ljMjkxY21ObGN5STZXeUl1TGk4dUxpOVhaV0pEYjJ4aEwzTnlZeTlzWVhsdmRYUXpaQzUwY3lKZExDSnVZVzFsY3lJNlcxMHNJbTFoY0hCcGJtZHpJam9pT3p0QlFVRkJMR2xFUVVFd1F6dEJRVU14UXl4eFEwRkJhVU03UVVGRGFrTXNlVU5CUVRSRU8wRkJSVFZFTERaRFFVRnZSVHRCUVVWd1JUdEpRVVZSTEdkQ1FVRnRRaXhOUVVGakxFVkJRVk1zVFVGQll6dFJRVUZ5UXl4WFFVRk5MRWRCUVU0c1RVRkJUU3hEUVVGUk8xRkJRVk1zVjBGQlRTeEhRVUZPTEUxQlFVMHNRMEZCVVR0SlFVRkpMRU5CUVVNN1NVRkROMFFzTmtKQlFWa3NSMEZCV2l4VlFVRmhMRU5CUVdFN1VVRkJNVUlzYVVKQlRVTTdVVUZNUnl4UFFVRlBMRWxCUVVrc1EwRkJReXhKUVVGSkxFTkJRMW9zUTBGQlF5eERRVUZETEUxQlFVMHNRMEZCUXl4VlFVRkRMRU5CUVZNc1JVRkJSU3hEUVVGWE8xbEJRelZDTEVsQlFVMHNSVUZCUlN4SFFVRkhMRU5CUVVNc1EwRkJReXhMUVVGSkxFTkJRVU1zVFVGQlRTeERRVUZETEVkQlFVY3NRMEZCUXl4RFFVRkRMRXRCUVVrc1EwRkJReXhOUVVGTkxFTkJRVU1zUTBGQlF6dFpRVU16UXl4UFFVRlBMRU5CUVVNc1IwRkJSeXhGUVVGRkxFZEJRVWNzUlVGQlJTeERRVUZETzFGQlEzWkNMRU5CUVVNc1JVRkJSU3hEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETzBsQlEyWXNRMEZCUXp0SlFVTk1MR0ZCUVVNN1FVRkJSQ3hEUVVGRExFRkJWa3dzU1VGVlN6dEJRVlpSTEhkQ1FVRk5PMEZCVjJZN1NVRlRTU3huUWtGRFZ5eERRVUZoTEVWQlEySXNRMEZCWVN4RlFVTmlMRU5CUVdFN1VVRkdZaXhyUWtGQlFTeEZRVUZCTEV0QlFXRTdVVUZEWWl4clFrRkJRU3hGUVVGQkxFdEJRV0U3VVVGRFlpeHJRa0ZCUVN4RlFVRkJMRXRCUVdFN1VVRkdZaXhOUVVGRExFZEJRVVFzUTBGQlF5eERRVUZaTzFGQlEySXNUVUZCUXl4SFFVRkVMRU5CUVVNc1EwRkJXVHRSUVVOaUxFMUJRVU1zUjBGQlJDeERRVUZETEVOQlFWazdTVUZCU1N4RFFVRkRPMGxCUTJwRExHRkJRVU03UVVGQlJDeERRVUZETEVGQllrUXNTVUZoUXp0QlFXSlpMSGRDUVVGTk8wRkJZMjVDTzBsQlRVa3NhMEpCUVcxQ0xFdEJRV1VzUlVGQlV5eExRVUZsTEVWQlFWTXNaVUZCTWtJN1VVRkJPVVlzYVVKQllVTTdVVUZpYTBVc1owTkJRVUVzUlVGQlFTeHRRa0ZCTWtJN1VVRkJNMFVzVlVGQlN5eEhRVUZNTEV0QlFVc3NRMEZCVlR0UlFVRlRMRlZCUVVzc1IwRkJUQ3hMUVVGTExFTkJRVlU3VVVGQlV5eHZRa0ZCWlN4SFFVRm1MR1ZCUVdVc1EwRkJXVHRSUVVZNVJpeG5Ra0ZCVnl4SFFVRlZMRWxCUVVrc1EwRkJRenRSUVhGQ01VSXNNRUpCUVhGQ0xFZEJRVmtzU1VGQlNTeERRVUZETzFGQmJFSnNReXhKUVVGSkxFTkJRVU1zVFVGQlRTeEhRVUZITEVsQlFVa3NTMEZCU3l4RFFVRkRMRkZCUVZFc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dFJRVU53UXl4TFFVRkxMRWxCUVVrc1EwRkJReXhIUVVGSExFTkJRVU1zUlVGQlJTeERRVUZETEVkQlFVY3NVVUZCVVN4RFFVRkRMRU5CUVVNc1JVRkJSU3hGUVVGRkxFTkJRVU1zUlVGQlJUdFpRVU5xUXl4SlFVRkpMRU5CUVVNc1RVRkJUU3hEUVVGRExFTkJRVU1zUTBGQlF5eEhRVUZITEVsQlFVa3NTMEZCU3l4RFFVRkRMRXRCUVVzc1EwRkJReXhOUVVGTkxFTkJRVU1zUTBGQlF6dFRRVU0xUXp0UlFVTkVMRXRCUVVzc1EwRkJReXhQUVVGUExFTkJRVU1zVlVGQlF5eERRVUZETEVWQlFVVXNRMEZCUXp0WlFVTm1MRXRCUVdkQ0xGVkJRV0VzUlVGQllpeExRVUZCTEZGQlFWRXNRMEZCUXl4SlFVRkpMRVZCUVdJc1kwRkJZU3hGUVVGaUxFbEJRV0VzUlVGQlJUdG5Ra0ZCTVVJc1NVRkJTU3hIUVVGSExGTkJRVUU3WjBKQlExSXNTVUZCU1N4UFFVRlBMRU5CUVVNc1EwRkJReXhIUVVGSExFTkJRVU1zU1VGQlNTeFhRVUZYTzI5Q1FVRkZMRU5CUVVNc1EwRkJReXhIUVVGSExFTkJRVU1zUjBGQlJ5eEpRVUZKTEVOQlFVTXNUVUZCVFN4RlFVRkZMRU5CUVVNN1lVRkROVVE3V1VGRFJDeExRVUZKTEVOQlFVTXNUVUZCVFN4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eEhRVUZITEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNN1dVRkRlRUlzUzBGQlNTeERRVUZETEUxQlFVMHNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUjBGQlJ5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRPMWxCUTNoQ0xFdEJRVWtzUTBGQlF5eE5RVUZOTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFZEJRVWNzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXp0UlFVTTFRaXhEUVVGRExFTkJRVU1zUTBGQlF6dEpRVU5RTEVOQlFVTTdTVUZCUVN4RFFVRkRPMGxCUlVZc05rSkJRVlVzUjBGQlZpeFZRVUZYTEVOQlFWTTdVVUZEYUVJc1QwRkJUeXhEUVVGRExFTkJRVU1zV1VGQldTeERRVUZETEVsQlFVa3NRMEZCUXl4TlFVRk5MRU5CUVVNc1EwRkJRenRKUVVOMlF5eERRVUZETzBsQlMwUXNkMEpCUVVzc1IwRkJUQ3hWUVVGTkxGVkJRWGRDTzFGQlFUbENMR2xDUVhWRFF6dFJRWFpEU3l3eVFrRkJRU3hGUVVGQkxHZENRVUYzUWp0UlFVTXhRaXhKUVVGTkxFTkJRVU1zUjBGQlJ5eEpRVUZKTEVOQlFVTXNTMEZCU3l4RFFVRkRMRTFCUVUwc1EwRkJRenRSUVVVMVFpeEpRVUZKTEZsQlFWa3NSMEZCUnl4SlFVRkpMRmxCUVZrc1JVRkJSU3hEUVVGRE8xRkJSWFJETEVsQlFVa3NTVUZCU1N4RFFVRkRMSEZDUVVGeFFqdFpRVU14UWl4blEwRkJhMElzUTBGQlF5eEpRVUZKTEVOQlFVTXNTMEZCU3l4RlFVRkZMRmxCUVZrc1JVRkJSU3hIUVVGSExFTkJRVU1zUTBGQlF6dFJRVVYwUkN4SlFVRkpMRU5CUVVNc1MwRkJTeXhEUVVGRExFOUJRVThzUTBGQlF5eFZRVUZCTEVOQlFVTXNTVUZCU1N4UFFVRkJMRU5CUVVNc1EwRkJReXhOUVVGTkxFbEJRVWtzUzBGQlNTeERRVUZETEdWQlFXVXNSVUZCYUVNc1EwRkJaME1zUTBGQlF5eERRVUZETzFGQlJ6RkVMRWxCUVUwc1kwRkJZeXhIUVVGSExFTkJRVU1zU1VGQlNTd3dRa0ZCVlN4RFFVRkRMRU5CUVVNc1JVRkJSU3hKUVVGSkxFTkJRVU1zUzBGQlN5eEZRVU5vUkN4VlFVRkJMRU5CUVVNc1NVRkJSeXhQUVVGQkxFTkJRVU1zUTBGQlF5eE5RVUZOTEVWQlFWSXNRMEZCVVN4RlFVRkZMRlZCUVVFc1EwRkJReXhKUVVGSExFOUJRVUVzUTBGQlF5eERRVUZETEUxQlFVMHNSVUZCVWl4RFFVRlJMRVZCUVVVc1ZVRkJRU3hEUVVGRExFbEJRVWtzVDBGQlFTeERRVUZETEVOQlFVTXNUVUZCVFN4RlFVRlNMRU5CUVZFc1EwRkJReXhEUVVGRExFTkJRVU1zWTBGQll5eEZRVUZGTEVOQlFVTTdVVUZGYWtVc1NVRkJUU3hEUVVGRExFZEJRVWNzYVVKQlFVOHNRMEZCUXl4clFrRkJhMElzUTBGQlF5eERRVUZETEVWQlFVVXNWVUZCUXl4RFFVRkRMRVZCUVVVc1EwRkJReXhKUVVGTExFOUJRVUVzWTBGQll5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhGUVVGd1FpeERRVUZ2UWl4RFFVRkRMRU5CUVVNN1VVRkplRVVzU1VGQlNTeERRVUZETEVkQlFVY3NhVUpCUVU4c1EwRkJReXhyUWtGQmEwSXNRMEZCUXl4RFFVRkRMRVZCUVVVc1kwRkJZeXhQUVVGUExFTkJRVU1zUTBGQlFTeERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRPMUZCUTJoRkxFbEJRVWtzUTBGQlF5eExRVUZMTEVOQlFVTXNUMEZCVHl4RFFVRkRMRlZCUVVNc1JVRkJhMEk3WjBKQlFXaENMR3RDUVVGTkxFVkJRVVVzYTBKQlFVMDdXVUZCVHl4UFFVRkJMRU5CUVVNc1EwRkJReXhOUVVGTkxFTkJRVU1zUTBGQlF5eE5RVUZOTEVOQlFVTXNSMEZCUnl4RFFVRkRMRU5CUVVNc1RVRkJUU3hEUVVGRExFTkJRVU1zVFVGQlRTeERRVUZETEVkQlFVY3NRMEZCUXp0UlFVRjZReXhEUVVGNVF5eERRVUZETEVOQlFVTTdVVUZGZEVZc1NVRkJTU3hEUVVGRExFOUJRVThzUjBGQlJ5eEpRVUZKTEdsQ1FVRlBMRU5CUVVNc1NVRkJTU3hEUVVGRExFMUJRVTBzUlVGQlJTeERRVUZETEVOQlFVTXNRMEZCUXp0UlFVTXpReXhKUVVGSkxFTkJRVU1zVDBGQlR5eERRVUZETEZOQlFWTXNSMEZCUnl4SlFVRkpMRU5CUVVNN1VVRkRPVUlzU1VGQlNTeERRVUZETEU5QlFVOHNRMEZCUXl4RFFVRkRMRWRCUVVjc1EwRkJReXhEUVVGRE8xRkJTVzVDTEVsQlFVa3NTVUZCU1N4RFFVRkRMRmRCUVZjN1dVRkRhRUlzU1VGQlNTeERRVUZETEU5QlFVOHNRMEZCUXl4UFFVRlBMRWRCUVVjc1NVRkJTU3h6UWtGQlZTeERRVUZqTEVsQlFVa3NRMEZCUXl4TFFVRkxMRVZCUVVVc1NVRkJTU3hGUVVGRkxFbEJRVWtzUlVGQlJTeEpRVUZKTEVOQlFVTXNWMEZCVnl4RFFVRkRMRU5CUVVNc1owSkJRV2RDTEVWQlFVVXNRMEZCUXp0UlFVVndTQ3hMUVVGTExFbEJRVWtzUTBGQlF5eEhRVUZITEVOQlFVTXNSVUZCUlN4RFFVRkRMRWRCUVVjc1NVRkJTU3hEUVVGRExFdEJRVXNzUTBGQlF5eE5RVUZOTEVWQlFVVXNRMEZCUXl4RlFVRkZMRVZCUVVVN1dVRkRlRU1zU1VGQlNTeERRVUZETEVkQlFVY3NTVUZCU1N4RFFVRkRMRXRCUVVzc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dFpRVU4wUWl4SlFVRkpMRU5CUVVNc1EwRkJReXhMUVVGTExFVkJRVVU3WjBKQlExUXNTVUZCU1N4RFFVRkRMRTlCUVU4c1EwRkJReXhMUVVGTExFTkJRVU1zUjBGQlJ5eERRVUZETEVOQlFVTXNSVUZCUlN4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFVkJRVVVzUTBGQlF5eERRVUZETEVOQlFVTXNSVUZCUlN4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dGhRVU01UXp0VFFVTktPMUZCUlVRc1NVRkJTU3hEUVVGRExFOUJRVThzUTBGQlF5eEhRVUZITEVOQlFVTXNWVUZCVlN4RFFVRkRMRU5CUVVNN1VVRkROMElzVDBGQlR5eEpRVUZKTEVOQlFVTTdTVUZEYUVJc1EwRkJRenRKUVVWRUxIVkNRVUZKTEVkQlFVbzdVVUZEU1N4SlFVRkpMRU5CUVVNc1QwRkJUeXhEUVVGRExFdEJRVXNzUTBGQlF5eExRVUZMTEVWQlFVVXNRMEZCUXp0UlFVTXpRaXhMUVVGTExFbEJRVWtzUTBGQlF5eEhRVUZITEVOQlFVTXNSVUZCUlN4RFFVRkRMRWRCUVVjc1NVRkJTU3hEUVVGRExFdEJRVXNzUTBGQlF5eE5RVUZOTEVWQlFVVXNRMEZCUXl4RlFVRkZMRVZCUVVVN1dVRkRlRU1zU1VGQlNTeERRVUZETEVkQlFVY3NTVUZCU1N4RFFVRkRMRXRCUVVzc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dFpRVU4wUWl4SlFVRkpMRU5CUVVNc1EwRkJReXhMUVVGTExFVkJRVVU3WjBKQlExUXNTVUZCU1N4RFFVRkRMRTlCUVU4c1EwRkJReXhMUVVGTExFTkJRVU1zUjBGQlJ5eERRVUZETEVOQlFVTXNSVUZCUlN4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFVkJRVVVzUTBGQlF5eERRVUZETEVOQlFVTXNSVUZCUlN4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dGhRVU01UXp0VFFVTktPMUZCUTBRc1QwRkJUeXhKUVVGSkxFTkJRVU1zVDBGQlR5eERRVUZETEZWQlFWVXNSVUZCUlN4RFFVRkRPMGxCUTNKRExFTkJRVU03U1VFM1JVMHNZVUZCU1N4SFFVRkhMRU5CUVVNc1IwRkJSeXhGUVVGRkxFZEJRVWNzUlVGQlJTeEhRVUZITEVOQlFVTXNRMEZCUXp0SlFVTjJRaXhWUVVGRExFZEJRVWNzVVVGQlVTeERRVUZETEVsQlFVa3NRMEZCUXl4TlFVRk5MRU5CUVVNN1NVRTJSWEJETEdWQlFVTTdRMEZCUVN4QlFTOUZSQ3hKUVN0RlF6dEJRUzlGV1N3MFFrRkJVVHRCUVdsR2NrSTdTVUZCUVR0SlFVdEJMRU5CUVVNN1NVRktSeXh4UTBGQll5eEhRVUZrTEZWQlFXVXNRMEZCVFN4SlFVRlpMRTlCUVU4c1EwRkJReXhEUVVGRExFMUJRVTBzUTBGQlF5eERRVUZETEVOQlFVTTdTVUZEYmtRc2NVTkJRV01zUjBGQlpDeFZRVUZsTEVOQlFVMHNTVUZCV1N4UFFVRlBMRU5CUVVNc1EwRkJReXhOUVVGTkxFTkJRVU1zUTBGQlF5eERRVUZETzBsQlEyNUVMR2REUVVGVExFZEJRVlFzVlVGQlZTeERRVUZOTEVsQlFWa3NUMEZCVHl4RFFVRkRMRU5CUVVNc1RVRkJUU3hEUVVGRExFTkJRVU1zUTBGQlF6dEpRVU01UXl4blEwRkJVeXhIUVVGVUxGVkJRVlVzUTBGQlRTeEZRVUZGTEVOQlFWTXNTVUZCU1N4RFFVRkRMRU5CUVVNc1RVRkJUU3hIUVVGSExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTTdTVUZEYkVRc2JVSkJRVU03UVVGQlJDeERRVUZETEVGQlRFUXNTVUZMUXlKOSIsIlwidXNlIHN0cmljdFwiO1xyXG5PYmplY3QuZGVmaW5lUHJvcGVydHkoZXhwb3J0cywgXCJfX2VzTW9kdWxlXCIsIHsgdmFsdWU6IHRydWUgfSk7XHJcbmZ1bmN0aW9uIHVuaW9uQ291bnQoYSwgYikge1xyXG4gICAgdmFyIHUgPSB7fTtcclxuICAgIGZvciAodmFyIGkgaW4gYSlcclxuICAgICAgICB1W2ldID0ge307XHJcbiAgICBmb3IgKHZhciBpIGluIGIpXHJcbiAgICAgICAgdVtpXSA9IHt9O1xyXG4gICAgcmV0dXJuIE9iamVjdC5rZXlzKHUpLmxlbmd0aDtcclxufVxyXG5mdW5jdGlvbiBpbnRlcnNlY3Rpb25Db3VudChhLCBiKSB7XHJcbiAgICB2YXIgbiA9IDA7XHJcbiAgICBmb3IgKHZhciBpIGluIGEpXHJcbiAgICAgICAgaWYgKHR5cGVvZiBiW2ldICE9PSAndW5kZWZpbmVkJylcclxuICAgICAgICAgICAgKytuO1xyXG4gICAgcmV0dXJuIG47XHJcbn1cclxuZnVuY3Rpb24gZ2V0TmVpZ2hib3VycyhsaW5rcywgbGEpIHtcclxuICAgIHZhciBuZWlnaGJvdXJzID0ge307XHJcbiAgICB2YXIgYWRkTmVpZ2hib3VycyA9IGZ1bmN0aW9uICh1LCB2KSB7XHJcbiAgICAgICAgaWYgKHR5cGVvZiBuZWlnaGJvdXJzW3VdID09PSAndW5kZWZpbmVkJylcclxuICAgICAgICAgICAgbmVpZ2hib3Vyc1t1XSA9IHt9O1xyXG4gICAgICAgIG5laWdoYm91cnNbdV1bdl0gPSB7fTtcclxuICAgIH07XHJcbiAgICBsaW5rcy5mb3JFYWNoKGZ1bmN0aW9uIChlKSB7XHJcbiAgICAgICAgdmFyIHUgPSBsYS5nZXRTb3VyY2VJbmRleChlKSwgdiA9IGxhLmdldFRhcmdldEluZGV4KGUpO1xyXG4gICAgICAgIGFkZE5laWdoYm91cnModSwgdik7XHJcbiAgICAgICAgYWRkTmVpZ2hib3Vycyh2LCB1KTtcclxuICAgIH0pO1xyXG4gICAgcmV0dXJuIG5laWdoYm91cnM7XHJcbn1cclxuZnVuY3Rpb24gY29tcHV0ZUxpbmtMZW5ndGhzKGxpbmtzLCB3LCBmLCBsYSkge1xyXG4gICAgdmFyIG5laWdoYm91cnMgPSBnZXROZWlnaGJvdXJzKGxpbmtzLCBsYSk7XHJcbiAgICBsaW5rcy5mb3JFYWNoKGZ1bmN0aW9uIChsKSB7XHJcbiAgICAgICAgdmFyIGEgPSBuZWlnaGJvdXJzW2xhLmdldFNvdXJjZUluZGV4KGwpXTtcclxuICAgICAgICB2YXIgYiA9IG5laWdoYm91cnNbbGEuZ2V0VGFyZ2V0SW5kZXgobCldO1xyXG4gICAgICAgIGxhLnNldExlbmd0aChsLCAxICsgdyAqIGYoYSwgYikpO1xyXG4gICAgfSk7XHJcbn1cclxuZnVuY3Rpb24gc3ltbWV0cmljRGlmZkxpbmtMZW5ndGhzKGxpbmtzLCBsYSwgdykge1xyXG4gICAgaWYgKHcgPT09IHZvaWQgMCkgeyB3ID0gMTsgfVxyXG4gICAgY29tcHV0ZUxpbmtMZW5ndGhzKGxpbmtzLCB3LCBmdW5jdGlvbiAoYSwgYikgeyByZXR1cm4gTWF0aC5zcXJ0KHVuaW9uQ291bnQoYSwgYikgLSBpbnRlcnNlY3Rpb25Db3VudChhLCBiKSk7IH0sIGxhKTtcclxufVxyXG5leHBvcnRzLnN5bW1ldHJpY0RpZmZMaW5rTGVuZ3RocyA9IHN5bW1ldHJpY0RpZmZMaW5rTGVuZ3RocztcclxuZnVuY3Rpb24gamFjY2FyZExpbmtMZW5ndGhzKGxpbmtzLCBsYSwgdykge1xyXG4gICAgaWYgKHcgPT09IHZvaWQgMCkgeyB3ID0gMTsgfVxyXG4gICAgY29tcHV0ZUxpbmtMZW5ndGhzKGxpbmtzLCB3LCBmdW5jdGlvbiAoYSwgYikge1xyXG4gICAgICAgIHJldHVybiBNYXRoLm1pbihPYmplY3Qua2V5cyhhKS5sZW5ndGgsIE9iamVjdC5rZXlzKGIpLmxlbmd0aCkgPCAxLjEgPyAwIDogaW50ZXJzZWN0aW9uQ291bnQoYSwgYikgLyB1bmlvbkNvdW50KGEsIGIpO1xyXG4gICAgfSwgbGEpO1xyXG59XHJcbmV4cG9ydHMuamFjY2FyZExpbmtMZW5ndGhzID0gamFjY2FyZExpbmtMZW5ndGhzO1xyXG5mdW5jdGlvbiBnZW5lcmF0ZURpcmVjdGVkRWRnZUNvbnN0cmFpbnRzKG4sIGxpbmtzLCBheGlzLCBsYSkge1xyXG4gICAgdmFyIGNvbXBvbmVudHMgPSBzdHJvbmdseUNvbm5lY3RlZENvbXBvbmVudHMobiwgbGlua3MsIGxhKTtcclxuICAgIHZhciBub2RlcyA9IHt9O1xyXG4gICAgY29tcG9uZW50cy5mb3JFYWNoKGZ1bmN0aW9uIChjLCBpKSB7XHJcbiAgICAgICAgcmV0dXJuIGMuZm9yRWFjaChmdW5jdGlvbiAodikgeyByZXR1cm4gbm9kZXNbdl0gPSBpOyB9KTtcclxuICAgIH0pO1xyXG4gICAgdmFyIGNvbnN0cmFpbnRzID0gW107XHJcbiAgICBsaW5rcy5mb3JFYWNoKGZ1bmN0aW9uIChsKSB7XHJcbiAgICAgICAgdmFyIHVpID0gbGEuZ2V0U291cmNlSW5kZXgobCksIHZpID0gbGEuZ2V0VGFyZ2V0SW5kZXgobCksIHUgPSBub2Rlc1t1aV0sIHYgPSBub2Rlc1t2aV07XHJcbiAgICAgICAgaWYgKHUgIT09IHYpIHtcclxuICAgICAgICAgICAgY29uc3RyYWludHMucHVzaCh7XHJcbiAgICAgICAgICAgICAgICBheGlzOiBheGlzLFxyXG4gICAgICAgICAgICAgICAgbGVmdDogdWksXHJcbiAgICAgICAgICAgICAgICByaWdodDogdmksXHJcbiAgICAgICAgICAgICAgICBnYXA6IGxhLmdldE1pblNlcGFyYXRpb24obClcclxuICAgICAgICAgICAgfSk7XHJcbiAgICAgICAgfVxyXG4gICAgfSk7XHJcbiAgICByZXR1cm4gY29uc3RyYWludHM7XHJcbn1cclxuZXhwb3J0cy5nZW5lcmF0ZURpcmVjdGVkRWRnZUNvbnN0cmFpbnRzID0gZ2VuZXJhdGVEaXJlY3RlZEVkZ2VDb25zdHJhaW50cztcclxuZnVuY3Rpb24gc3Ryb25nbHlDb25uZWN0ZWRDb21wb25lbnRzKG51bVZlcnRpY2VzLCBlZGdlcywgbGEpIHtcclxuICAgIHZhciBub2RlcyA9IFtdO1xyXG4gICAgdmFyIGluZGV4ID0gMDtcclxuICAgIHZhciBzdGFjayA9IFtdO1xyXG4gICAgdmFyIGNvbXBvbmVudHMgPSBbXTtcclxuICAgIGZ1bmN0aW9uIHN0cm9uZ0Nvbm5lY3Qodikge1xyXG4gICAgICAgIHYuaW5kZXggPSB2Lmxvd2xpbmsgPSBpbmRleCsrO1xyXG4gICAgICAgIHN0YWNrLnB1c2godik7XHJcbiAgICAgICAgdi5vblN0YWNrID0gdHJ1ZTtcclxuICAgICAgICBmb3IgKHZhciBfaSA9IDAsIF9hID0gdi5vdXQ7IF9pIDwgX2EubGVuZ3RoOyBfaSsrKSB7XHJcbiAgICAgICAgICAgIHZhciB3ID0gX2FbX2ldO1xyXG4gICAgICAgICAgICBpZiAodHlwZW9mIHcuaW5kZXggPT09ICd1bmRlZmluZWQnKSB7XHJcbiAgICAgICAgICAgICAgICBzdHJvbmdDb25uZWN0KHcpO1xyXG4gICAgICAgICAgICAgICAgdi5sb3dsaW5rID0gTWF0aC5taW4odi5sb3dsaW5rLCB3Lmxvd2xpbmspO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIGVsc2UgaWYgKHcub25TdGFjaykge1xyXG4gICAgICAgICAgICAgICAgdi5sb3dsaW5rID0gTWF0aC5taW4odi5sb3dsaW5rLCB3LmluZGV4KTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH1cclxuICAgICAgICBpZiAodi5sb3dsaW5rID09PSB2LmluZGV4KSB7XHJcbiAgICAgICAgICAgIHZhciBjb21wb25lbnQgPSBbXTtcclxuICAgICAgICAgICAgd2hpbGUgKHN0YWNrLmxlbmd0aCkge1xyXG4gICAgICAgICAgICAgICAgdyA9IHN0YWNrLnBvcCgpO1xyXG4gICAgICAgICAgICAgICAgdy5vblN0YWNrID0gZmFsc2U7XHJcbiAgICAgICAgICAgICAgICBjb21wb25lbnQucHVzaCh3KTtcclxuICAgICAgICAgICAgICAgIGlmICh3ID09PSB2KVxyXG4gICAgICAgICAgICAgICAgICAgIGJyZWFrO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIGNvbXBvbmVudHMucHVzaChjb21wb25lbnQubWFwKGZ1bmN0aW9uICh2KSB7IHJldHVybiB2LmlkOyB9KSk7XHJcbiAgICAgICAgfVxyXG4gICAgfVxyXG4gICAgZm9yICh2YXIgaSA9IDA7IGkgPCBudW1WZXJ0aWNlczsgaSsrKSB7XHJcbiAgICAgICAgbm9kZXMucHVzaCh7IGlkOiBpLCBvdXQ6IFtdIH0pO1xyXG4gICAgfVxyXG4gICAgZm9yICh2YXIgX2kgPSAwLCBlZGdlc18xID0gZWRnZXM7IF9pIDwgZWRnZXNfMS5sZW5ndGg7IF9pKyspIHtcclxuICAgICAgICB2YXIgZSA9IGVkZ2VzXzFbX2ldO1xyXG4gICAgICAgIHZhciB2XzEgPSBub2Rlc1tsYS5nZXRTb3VyY2VJbmRleChlKV0sIHcgPSBub2Rlc1tsYS5nZXRUYXJnZXRJbmRleChlKV07XHJcbiAgICAgICAgdl8xLm91dC5wdXNoKHcpO1xyXG4gICAgfVxyXG4gICAgZm9yICh2YXIgX2EgPSAwLCBub2Rlc18xID0gbm9kZXM7IF9hIDwgbm9kZXNfMS5sZW5ndGg7IF9hKyspIHtcclxuICAgICAgICB2YXIgdiA9IG5vZGVzXzFbX2FdO1xyXG4gICAgICAgIGlmICh0eXBlb2Ygdi5pbmRleCA9PT0gJ3VuZGVmaW5lZCcpXHJcbiAgICAgICAgICAgIHN0cm9uZ0Nvbm5lY3Qodik7XHJcbiAgICB9XHJcbiAgICByZXR1cm4gY29tcG9uZW50cztcclxufVxyXG5leHBvcnRzLnN0cm9uZ2x5Q29ubmVjdGVkQ29tcG9uZW50cyA9IHN0cm9uZ2x5Q29ubmVjdGVkQ29tcG9uZW50cztcclxuLy8jIHNvdXJjZU1hcHBpbmdVUkw9ZGF0YTphcHBsaWNhdGlvbi9qc29uO2Jhc2U2NCxleUoyWlhKemFXOXVJam96TENKbWFXeGxJam9pYkdsdWEyeGxibWQwYUhNdWFuTWlMQ0p6YjNWeVkyVlNiMjkwSWpvaUlpd2ljMjkxY21ObGN5STZXeUl1TGk4dUxpOVhaV0pEYjJ4aEwzTnlZeTlzYVc1cmJHVnVaM1JvY3k1MGN5SmRMQ0p1WVcxbGN5STZXMTBzSW0xaGNIQnBibWR6SWpvaU96dEJRVlZKTEZOQlFWTXNWVUZCVlN4RFFVRkRMRU5CUVUwc1JVRkJSU3hEUVVGTk8wbEJRemxDTEVsQlFVa3NRMEZCUXl4SFFVRkhMRVZCUVVVc1EwRkJRenRKUVVOWUxFdEJRVXNzU1VGQlNTeERRVUZETEVsQlFVa3NRMEZCUXp0UlFVRkZMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUjBGQlJ5eEZRVUZGTEVOQlFVTTdTVUZETTBJc1MwRkJTeXhKUVVGSkxFTkJRVU1zU1VGQlNTeERRVUZETzFGQlFVVXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhIUVVGSExFVkJRVVVzUTBGQlF6dEpRVU16UWl4UFFVRlBMRTFCUVUwc1EwRkJReXhKUVVGSkxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNUVUZCVFN4RFFVRkRPMEZCUTJwRExFTkJRVU03UVVGSFJDeFRRVUZUTEdsQ1FVRnBRaXhEUVVGRExFTkJRVmNzUlVGQlJTeERRVUZYTzBsQlF5OURMRWxCUVVrc1EwRkJReXhIUVVGSExFTkJRVU1zUTBGQlF6dEpRVU5XTEV0QlFVc3NTVUZCU1N4RFFVRkRMRWxCUVVrc1EwRkJRenRSUVVGRkxFbEJRVWtzVDBGQlR5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRXRCUVVzc1YwRkJWenRaUVVGRkxFVkJRVVVzUTBGQlF5eERRVUZETzBsQlEzUkVMRTlCUVU4c1EwRkJReXhEUVVGRE8wRkJRMklzUTBGQlF6dEJRVVZFTEZOQlFWTXNZVUZCWVN4RFFVRlBMRXRCUVdFc1JVRkJSU3hGUVVGelFqdEpRVU01UkN4SlFVRkpMRlZCUVZVc1IwRkJSeXhGUVVGRkxFTkJRVU03U1VGRGNFSXNTVUZCU1N4aFFVRmhMRWRCUVVjc1ZVRkJReXhEUVVGRExFVkJRVVVzUTBGQlF6dFJRVU55UWl4SlFVRkpMRTlCUVU4c1ZVRkJWU3hEUVVGRExFTkJRVU1zUTBGQlF5eExRVUZMTEZkQlFWYzdXVUZEY0VNc1ZVRkJWU3hEUVVGRExFTkJRVU1zUTBGQlF5eEhRVUZITEVWQlFVVXNRMEZCUXp0UlFVTjJRaXhWUVVGVkxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRWRCUVVjc1JVRkJSU3hEUVVGRE8wbEJRekZDTEVOQlFVTXNRMEZCUXp0SlFVTkdMRXRCUVVzc1EwRkJReXhQUVVGUExFTkJRVU1zVlVGQlFTeERRVUZETzFGQlExZ3NTVUZCU1N4RFFVRkRMRWRCUVVjc1JVRkJSU3hEUVVGRExHTkJRV01zUTBGQlF5eERRVUZETEVOQlFVTXNSVUZCUlN4RFFVRkRMRWRCUVVjc1JVRkJSU3hEUVVGRExHTkJRV01zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXp0UlFVTjJSQ3hoUVVGaExFTkJRVU1zUTBGQlF5eEZRVUZGTEVOQlFVTXNRMEZCUXl4RFFVRkRPMUZCUTNCQ0xHRkJRV0VzUTBGQlF5eERRVUZETEVWQlFVVXNRMEZCUXl4RFFVRkRMRU5CUVVNN1NVRkRlRUlzUTBGQlF5eERRVUZETEVOQlFVTTdTVUZEU0N4UFFVRlBMRlZCUVZVc1EwRkJRenRCUVVOMFFpeERRVUZETzBGQlIwUXNVMEZCVXl4clFrRkJhMElzUTBGQlR5eExRVUZoTEVWQlFVVXNRMEZCVXl4RlFVRkZMRU5CUVRaQ0xFVkJRVVVzUlVGQk5FSTdTVUZEYmtnc1NVRkJTU3hWUVVGVkxFZEJRVWNzWVVGQllTeERRVUZETEV0QlFVc3NSVUZCUlN4RlFVRkZMRU5CUVVNc1EwRkJRenRKUVVNeFF5eExRVUZMTEVOQlFVTXNUMEZCVHl4RFFVRkRMRlZCUVVFc1EwRkJRenRSUVVOWUxFbEJRVWtzUTBGQlF5eEhRVUZITEZWQlFWVXNRMEZCUXl4RlFVRkZMRU5CUVVNc1kwRkJZeXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTTdVVUZEZWtNc1NVRkJTU3hEUVVGRExFZEJRVWNzVlVGQlZTeERRVUZETEVWQlFVVXNRMEZCUXl4alFVRmpMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dFJRVU42UXl4RlFVRkZMRU5CUVVNc1UwRkJVeXhEUVVGRExFTkJRVU1zUlVGQlJTeERRVUZETEVkQlFVY3NRMEZCUXl4SFFVRkhMRU5CUVVNc1EwRkJReXhEUVVGRExFVkJRVVVzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXp0SlFVTnlReXhEUVVGRExFTkJRVU1zUTBGQlF6dEJRVU5RTEVOQlFVTTdRVUZMUkN4VFFVRm5RaXgzUWtGQmQwSXNRMEZCVHl4TFFVRmhMRVZCUVVVc1JVRkJORUlzUlVGQlJTeERRVUZoTzBsQlFXSXNhMEpCUVVFc1JVRkJRU3hMUVVGaE8wbEJRM0pITEd0Q1FVRnJRaXhEUVVGRExFdEJRVXNzUlVGQlJTeERRVUZETEVWQlFVVXNWVUZCUXl4RFFVRkRMRVZCUVVVc1EwRkJReXhKUVVGTExFOUJRVUVzU1VGQlNTeERRVUZETEVsQlFVa3NRMEZCUXl4VlFVRlZMRU5CUVVNc1EwRkJReXhGUVVGRkxFTkJRVU1zUTBGQlF5eEhRVUZITEdsQ1FVRnBRaXhEUVVGRExFTkJRVU1zUlVGQlJTeERRVUZETEVOQlFVTXNRMEZCUXl4RlFVRnlSQ3hEUVVGeFJDeEZRVUZGTEVWQlFVVXNRMEZCUXl4RFFVRkRPMEZCUTNSSExFTkJRVU03UVVGR1JDdzBSRUZGUXp0QlFVdEVMRk5CUVdkQ0xHdENRVUZyUWl4RFFVRlBMRXRCUVdFc1JVRkJSU3hGUVVFMFFpeEZRVUZGTEVOQlFXRTdTVUZCWWl4clFrRkJRU3hGUVVGQkxFdEJRV0U3U1VGREwwWXNhMEpCUVd0Q0xFTkJRVU1zUzBGQlN5eEZRVUZGTEVOQlFVTXNSVUZCUlN4VlFVRkRMRU5CUVVNc1JVRkJSU3hEUVVGRE8xRkJRemxDTEU5QlFVRXNTVUZCU1N4RFFVRkRMRWRCUVVjc1EwRkJReXhOUVVGTkxFTkJRVU1zU1VGQlNTeERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRTFCUVUwc1JVRkJSU3hOUVVGTkxFTkJRVU1zU1VGQlNTeERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRTFCUVUwc1EwRkJReXhIUVVGSExFZEJRVWNzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhwUWtGQmFVSXNRMEZCUXl4RFFVRkRMRVZCUVVVc1EwRkJReXhEUVVGRExFZEJRVWNzVlVGQlZTeERRVUZETEVOQlFVTXNSVUZCUlN4RFFVRkRMRU5CUVVNN1NVRkJOMGNzUTBGQk5rY3NSVUZETTBjc1JVRkJSU3hEUVVGRExFTkJRVU03UVVGRFpDeERRVUZETzBGQlNrUXNaMFJCU1VNN1FVRnZRa1FzVTBGQlowSXNLMEpCUVN0Q0xFTkJRVThzUTBGQlV5eEZRVUZGTEV0QlFXRXNSVUZCUlN4SlFVRlpMRVZCUTNoR0xFVkJRWGxDTzBsQlJYcENMRWxCUVVrc1ZVRkJWU3hIUVVGSExESkNRVUV5UWl4RFFVRkRMRU5CUVVNc1JVRkJSU3hMUVVGTExFVkJRVVVzUlVGQlJTeERRVUZETEVOQlFVTTdTVUZETTBRc1NVRkJTU3hMUVVGTExFZEJRVWNzUlVGQlJTeERRVUZETzBsQlEyWXNWVUZCVlN4RFFVRkRMRTlCUVU4c1EwRkJReXhWUVVGRExFTkJRVU1zUlVGQlF5eERRVUZETzFGQlEyNUNMRTlCUVVFc1EwRkJReXhEUVVGRExFOUJRVThzUTBGQlF5eFZRVUZCTEVOQlFVTXNTVUZCU1N4UFFVRkJMRXRCUVVzc1EwRkJReXhEUVVGRExFTkJRVU1zUjBGQlJ5eERRVUZETEVWQlFWb3NRMEZCV1N4RFFVRkRPMGxCUVRWQ0xFTkJRVFJDTEVOQlF5OUNMRU5CUVVNN1NVRkRSaXhKUVVGSkxGZEJRVmNzUjBGQlZTeEZRVUZGTEVOQlFVTTdTVUZETlVJc1MwRkJTeXhEUVVGRExFOUJRVThzUTBGQlF5eFZRVUZCTEVOQlFVTTdVVUZEV0N4SlFVRkpMRVZCUVVVc1IwRkJSeXhGUVVGRkxFTkJRVU1zWTBGQll5eERRVUZETEVOQlFVTXNRMEZCUXl4RlFVRkZMRVZCUVVVc1IwRkJSeXhGUVVGRkxFTkJRVU1zWTBGQll5eERRVUZETEVOQlFVTXNRMEZCUXl4RlFVTndSQ3hEUVVGRExFZEJRVWNzUzBGQlN5eERRVUZETEVWQlFVVXNRMEZCUXl4RlFVRkZMRU5CUVVNc1IwRkJSeXhMUVVGTExFTkJRVU1zUlVGQlJTeERRVUZETEVOQlFVTTdVVUZEYWtNc1NVRkJTU3hEUVVGRExFdEJRVXNzUTBGQlF5eEZRVUZGTzFsQlExUXNWMEZCVnl4RFFVRkRMRWxCUVVrc1EwRkJRenRuUWtGRFlpeEpRVUZKTEVWQlFVVXNTVUZCU1R0blFrRkRWaXhKUVVGSkxFVkJRVVVzUlVGQlJUdG5Ra0ZEVWl4TFFVRkxMRVZCUVVVc1JVRkJSVHRuUWtGRFZDeEhRVUZITEVWQlFVVXNSVUZCUlN4RFFVRkRMR2RDUVVGblFpeERRVUZETEVOQlFVTXNRMEZCUXp0aFFVTTVRaXhEUVVGRExFTkJRVU03VTBGRFRqdEpRVU5NTEVOQlFVTXNRMEZCUXl4RFFVRkRPMGxCUTBnc1QwRkJUeXhYUVVGWExFTkJRVU03UVVGRGRrSXNRMEZCUXp0QlFYUkNSQ3d3UlVGelFrTTdRVUZSUkN4VFFVRm5RaXd5UWtGQk1rSXNRMEZCVHl4WFFVRnRRaXhGUVVGRkxFdEJRV0VzUlVGQlJTeEZRVUZ6UWp0SlFVTjRSeXhKUVVGSkxFdEJRVXNzUjBGQlJ5eEZRVUZGTEVOQlFVTTdTVUZEWml4SlFVRkpMRXRCUVVzc1IwRkJSeXhEUVVGRExFTkJRVU03U1VGRFpDeEpRVUZKTEV0QlFVc3NSMEZCUnl4RlFVRkZMRU5CUVVNN1NVRkRaaXhKUVVGSkxGVkJRVlVzUjBGQlJ5eEZRVUZGTEVOQlFVTTdTVUZEY0VJc1UwRkJVeXhoUVVGaExFTkJRVU1zUTBGQlF6dFJRVVZ3UWl4RFFVRkRMRU5CUVVNc1MwRkJTeXhIUVVGSExFTkJRVU1zUTBGQlF5eFBRVUZQTEVkQlFVY3NTMEZCU3l4RlFVRkZMRU5CUVVNN1VVRkRPVUlzUzBGQlN5eERRVUZETEVsQlFVa3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJRenRSUVVOa0xFTkJRVU1zUTBGQlF5eFBRVUZQTEVkQlFVY3NTVUZCU1N4RFFVRkRPMUZCUjJwQ0xFdEJRV01zVlVGQlN5eEZRVUZNTEV0QlFVRXNRMEZCUXl4RFFVRkRMRWRCUVVjc1JVRkJUQ3hqUVVGTExFVkJRVXdzU1VGQlN5eEZRVUZGTzFsQlFXaENMRWxCUVVrc1EwRkJReXhUUVVGQk8xbEJRMDRzU1VGQlNTeFBRVUZQTEVOQlFVTXNRMEZCUXl4TFFVRkxMRXRCUVVzc1YwRkJWeXhGUVVGRk8yZENRVVZvUXl4aFFVRmhMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU03WjBKQlEycENMRU5CUVVNc1EwRkJReXhQUVVGUExFZEJRVWNzU1VGQlNTeERRVUZETEVkQlFVY3NRMEZCUXl4RFFVRkRMRU5CUVVNc1QwRkJUeXhGUVVGRkxFTkJRVU1zUTBGQlF5eFBRVUZQTEVOQlFVTXNRMEZCUXp0aFFVTTVRenRwUWtGQlRTeEpRVUZKTEVOQlFVTXNRMEZCUXl4UFFVRlBMRVZCUVVVN1owSkJSV3hDTEVOQlFVTXNRMEZCUXl4UFFVRlBMRWRCUVVjc1NVRkJTU3hEUVVGRExFZEJRVWNzUTBGQlF5eERRVUZETEVOQlFVTXNUMEZCVHl4RlFVRkZMRU5CUVVNc1EwRkJReXhMUVVGTExFTkJRVU1zUTBGQlF6dGhRVU0xUXp0VFFVTktPMUZCUjBRc1NVRkJTU3hEUVVGRExFTkJRVU1zVDBGQlR5eExRVUZMTEVOQlFVTXNRMEZCUXl4TFFVRkxMRVZCUVVVN1dVRkZka0lzU1VGQlNTeFRRVUZUTEVkQlFVY3NSVUZCUlN4RFFVRkRPMWxCUTI1Q0xFOUJRVThzUzBGQlN5eERRVUZETEUxQlFVMHNSVUZCUlR0blFrRkRha0lzUTBGQlF5eEhRVUZITEV0QlFVc3NRMEZCUXl4SFFVRkhMRVZCUVVVc1EwRkJRenRuUWtGRGFFSXNRMEZCUXl4RFFVRkRMRTlCUVU4c1IwRkJSeXhMUVVGTExFTkJRVU03WjBKQlJXeENMRk5CUVZNc1EwRkJReXhKUVVGSkxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTTdaMEpCUTJ4Q0xFbEJRVWtzUTBGQlF5eExRVUZMTEVOQlFVTTdiMEpCUVVVc1RVRkJUVHRoUVVOMFFqdFpRVVZFTEZWQlFWVXNRMEZCUXl4SlFVRkpMRU5CUVVNc1UwRkJVeXhEUVVGRExFZEJRVWNzUTBGQlF5eFZRVUZCTEVOQlFVTXNTVUZCU1N4UFFVRkJMRU5CUVVNc1EwRkJReXhGUVVGRkxFVkJRVW9zUTBGQlNTeERRVUZETEVOQlFVTXNRMEZCUXp0VFFVTTNRenRKUVVOTUxFTkJRVU03U1VGRFJDeExRVUZMTEVsQlFVa3NRMEZCUXl4SFFVRkhMRU5CUVVNc1JVRkJSU3hEUVVGRExFZEJRVWNzVjBGQlZ5eEZRVUZGTEVOQlFVTXNSVUZCUlN4RlFVRkZPMUZCUTJ4RExFdEJRVXNzUTBGQlF5eEpRVUZKTEVOQlFVTXNSVUZCUXl4RlFVRkZMRVZCUVVVc1EwRkJReXhGUVVGRkxFZEJRVWNzUlVGQlJTeEZRVUZGTEVWQlFVTXNRMEZCUXl4RFFVRkRPMHRCUTJoRE8wbEJRMFFzUzBGQll5eFZRVUZMTEVWQlFVd3NaVUZCU3l4RlFVRk1MRzFDUVVGTExFVkJRVXdzU1VGQlN5eEZRVUZGTzFGQlFXaENMRWxCUVVrc1EwRkJReXhqUVVGQk8xRkJRMDRzU1VGQlNTeEhRVUZETEVkQlFVY3NTMEZCU3l4RFFVRkRMRVZCUVVVc1EwRkJReXhqUVVGakxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNSVUZETDBJc1EwRkJReXhIUVVGSExFdEJRVXNzUTBGQlF5eEZRVUZGTEVOQlFVTXNZMEZCWXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU03VVVGRGNFTXNSMEZCUXl4RFFVRkRMRWRCUVVjc1EwRkJReXhKUVVGSkxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTTdTMEZEYWtJN1NVRkRSQ3hMUVVGakxGVkJRVXNzUlVGQlRDeGxRVUZMTEVWQlFVd3NiVUpCUVVzc1JVRkJUQ3hKUVVGTE8xRkJRV1FzU1VGQlNTeERRVUZETEdOQlFVRTdVVUZCVnl4SlFVRkpMRTlCUVU4c1EwRkJReXhEUVVGRExFdEJRVXNzUzBGQlN5eFhRVUZYTzFsQlFVVXNZVUZCWVN4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRE8wdEJRVUU3U1VGRE1VVXNUMEZCVHl4VlFVRlZMRU5CUVVNN1FVRkRkRUlzUTBGQlF6dEJRV2hFUkN4clJVRm5SRU1pZlE9PSIsIlwidXNlIHN0cmljdFwiO1xyXG5PYmplY3QuZGVmaW5lUHJvcGVydHkoZXhwb3J0cywgXCJfX2VzTW9kdWxlXCIsIHsgdmFsdWU6IHRydWUgfSk7XHJcbnZhciBQb3dlckVkZ2UgPSAoZnVuY3Rpb24gKCkge1xyXG4gICAgZnVuY3Rpb24gUG93ZXJFZGdlKHNvdXJjZSwgdGFyZ2V0LCB0eXBlKSB7XHJcbiAgICAgICAgdGhpcy5zb3VyY2UgPSBzb3VyY2U7XHJcbiAgICAgICAgdGhpcy50YXJnZXQgPSB0YXJnZXQ7XHJcbiAgICAgICAgdGhpcy50eXBlID0gdHlwZTtcclxuICAgIH1cclxuICAgIHJldHVybiBQb3dlckVkZ2U7XHJcbn0oKSk7XHJcbmV4cG9ydHMuUG93ZXJFZGdlID0gUG93ZXJFZGdlO1xyXG52YXIgQ29uZmlndXJhdGlvbiA9IChmdW5jdGlvbiAoKSB7XHJcbiAgICBmdW5jdGlvbiBDb25maWd1cmF0aW9uKG4sIGVkZ2VzLCBsaW5rQWNjZXNzb3IsIHJvb3RHcm91cCkge1xyXG4gICAgICAgIHZhciBfdGhpcyA9IHRoaXM7XHJcbiAgICAgICAgdGhpcy5saW5rQWNjZXNzb3IgPSBsaW5rQWNjZXNzb3I7XHJcbiAgICAgICAgdGhpcy5tb2R1bGVzID0gbmV3IEFycmF5KG4pO1xyXG4gICAgICAgIHRoaXMucm9vdHMgPSBbXTtcclxuICAgICAgICBpZiAocm9vdEdyb3VwKSB7XHJcbiAgICAgICAgICAgIHRoaXMuaW5pdE1vZHVsZXNGcm9tR3JvdXAocm9vdEdyb3VwKTtcclxuICAgICAgICB9XHJcbiAgICAgICAgZWxzZSB7XHJcbiAgICAgICAgICAgIHRoaXMucm9vdHMucHVzaChuZXcgTW9kdWxlU2V0KCkpO1xyXG4gICAgICAgICAgICBmb3IgKHZhciBpID0gMDsgaSA8IG47ICsraSlcclxuICAgICAgICAgICAgICAgIHRoaXMucm9vdHNbMF0uYWRkKHRoaXMubW9kdWxlc1tpXSA9IG5ldyBNb2R1bGUoaSkpO1xyXG4gICAgICAgIH1cclxuICAgICAgICB0aGlzLlIgPSBlZGdlcy5sZW5ndGg7XHJcbiAgICAgICAgZWRnZXMuZm9yRWFjaChmdW5jdGlvbiAoZSkge1xyXG4gICAgICAgICAgICB2YXIgcyA9IF90aGlzLm1vZHVsZXNbbGlua0FjY2Vzc29yLmdldFNvdXJjZUluZGV4KGUpXSwgdCA9IF90aGlzLm1vZHVsZXNbbGlua0FjY2Vzc29yLmdldFRhcmdldEluZGV4KGUpXSwgdHlwZSA9IGxpbmtBY2Nlc3Nvci5nZXRUeXBlKGUpO1xyXG4gICAgICAgICAgICBzLm91dGdvaW5nLmFkZCh0eXBlLCB0KTtcclxuICAgICAgICAgICAgdC5pbmNvbWluZy5hZGQodHlwZSwgcyk7XHJcbiAgICAgICAgfSk7XHJcbiAgICB9XHJcbiAgICBDb25maWd1cmF0aW9uLnByb3RvdHlwZS5pbml0TW9kdWxlc0Zyb21Hcm91cCA9IGZ1bmN0aW9uIChncm91cCkge1xyXG4gICAgICAgIHZhciBtb2R1bGVTZXQgPSBuZXcgTW9kdWxlU2V0KCk7XHJcbiAgICAgICAgdGhpcy5yb290cy5wdXNoKG1vZHVsZVNldCk7XHJcbiAgICAgICAgZm9yICh2YXIgaSA9IDA7IGkgPCBncm91cC5sZWF2ZXMubGVuZ3RoOyArK2kpIHtcclxuICAgICAgICAgICAgdmFyIG5vZGUgPSBncm91cC5sZWF2ZXNbaV07XHJcbiAgICAgICAgICAgIHZhciBtb2R1bGUgPSBuZXcgTW9kdWxlKG5vZGUuaWQpO1xyXG4gICAgICAgICAgICB0aGlzLm1vZHVsZXNbbm9kZS5pZF0gPSBtb2R1bGU7XHJcbiAgICAgICAgICAgIG1vZHVsZVNldC5hZGQobW9kdWxlKTtcclxuICAgICAgICB9XHJcbiAgICAgICAgaWYgKGdyb3VwLmdyb3Vwcykge1xyXG4gICAgICAgICAgICBmb3IgKHZhciBqID0gMDsgaiA8IGdyb3VwLmdyb3Vwcy5sZW5ndGg7ICsraikge1xyXG4gICAgICAgICAgICAgICAgdmFyIGNoaWxkID0gZ3JvdXAuZ3JvdXBzW2pdO1xyXG4gICAgICAgICAgICAgICAgdmFyIGRlZmluaXRpb24gPSB7fTtcclxuICAgICAgICAgICAgICAgIGZvciAodmFyIHByb3AgaW4gY2hpbGQpXHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKHByb3AgIT09IFwibGVhdmVzXCIgJiYgcHJvcCAhPT0gXCJncm91cHNcIiAmJiBjaGlsZC5oYXNPd25Qcm9wZXJ0eShwcm9wKSlcclxuICAgICAgICAgICAgICAgICAgICAgICAgZGVmaW5pdGlvbltwcm9wXSA9IGNoaWxkW3Byb3BdO1xyXG4gICAgICAgICAgICAgICAgbW9kdWxlU2V0LmFkZChuZXcgTW9kdWxlKC0xIC0gaiwgbmV3IExpbmtTZXRzKCksIG5ldyBMaW5rU2V0cygpLCB0aGlzLmluaXRNb2R1bGVzRnJvbUdyb3VwKGNoaWxkKSwgZGVmaW5pdGlvbikpO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHJldHVybiBtb2R1bGVTZXQ7XHJcbiAgICB9O1xyXG4gICAgQ29uZmlndXJhdGlvbi5wcm90b3R5cGUubWVyZ2UgPSBmdW5jdGlvbiAoYSwgYiwgaykge1xyXG4gICAgICAgIGlmIChrID09PSB2b2lkIDApIHsgayA9IDA7IH1cclxuICAgICAgICB2YXIgaW5JbnQgPSBhLmluY29taW5nLmludGVyc2VjdGlvbihiLmluY29taW5nKSwgb3V0SW50ID0gYS5vdXRnb2luZy5pbnRlcnNlY3Rpb24oYi5vdXRnb2luZyk7XHJcbiAgICAgICAgdmFyIGNoaWxkcmVuID0gbmV3IE1vZHVsZVNldCgpO1xyXG4gICAgICAgIGNoaWxkcmVuLmFkZChhKTtcclxuICAgICAgICBjaGlsZHJlbi5hZGQoYik7XHJcbiAgICAgICAgdmFyIG0gPSBuZXcgTW9kdWxlKHRoaXMubW9kdWxlcy5sZW5ndGgsIG91dEludCwgaW5JbnQsIGNoaWxkcmVuKTtcclxuICAgICAgICB0aGlzLm1vZHVsZXMucHVzaChtKTtcclxuICAgICAgICB2YXIgdXBkYXRlID0gZnVuY3Rpb24gKHMsIGksIG8pIHtcclxuICAgICAgICAgICAgcy5mb3JBbGwoZnVuY3Rpb24gKG1zLCBsaW5rdHlwZSkge1xyXG4gICAgICAgICAgICAgICAgbXMuZm9yQWxsKGZ1bmN0aW9uIChuKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgdmFyIG5scyA9IG5baV07XHJcbiAgICAgICAgICAgICAgICAgICAgbmxzLmFkZChsaW5rdHlwZSwgbSk7XHJcbiAgICAgICAgICAgICAgICAgICAgbmxzLnJlbW92ZShsaW5rdHlwZSwgYSk7XHJcbiAgICAgICAgICAgICAgICAgICAgbmxzLnJlbW92ZShsaW5rdHlwZSwgYik7XHJcbiAgICAgICAgICAgICAgICAgICAgYVtvXS5yZW1vdmUobGlua3R5cGUsIG4pO1xyXG4gICAgICAgICAgICAgICAgICAgIGJbb10ucmVtb3ZlKGxpbmt0eXBlLCBuKTtcclxuICAgICAgICAgICAgICAgIH0pO1xyXG4gICAgICAgICAgICB9KTtcclxuICAgICAgICB9O1xyXG4gICAgICAgIHVwZGF0ZShvdXRJbnQsIFwiaW5jb21pbmdcIiwgXCJvdXRnb2luZ1wiKTtcclxuICAgICAgICB1cGRhdGUoaW5JbnQsIFwib3V0Z29pbmdcIiwgXCJpbmNvbWluZ1wiKTtcclxuICAgICAgICB0aGlzLlIgLT0gaW5JbnQuY291bnQoKSArIG91dEludC5jb3VudCgpO1xyXG4gICAgICAgIHRoaXMucm9vdHNba10ucmVtb3ZlKGEpO1xyXG4gICAgICAgIHRoaXMucm9vdHNba10ucmVtb3ZlKGIpO1xyXG4gICAgICAgIHRoaXMucm9vdHNba10uYWRkKG0pO1xyXG4gICAgICAgIHJldHVybiBtO1xyXG4gICAgfTtcclxuICAgIENvbmZpZ3VyYXRpb24ucHJvdG90eXBlLnJvb3RNZXJnZXMgPSBmdW5jdGlvbiAoaykge1xyXG4gICAgICAgIGlmIChrID09PSB2b2lkIDApIHsgayA9IDA7IH1cclxuICAgICAgICB2YXIgcnMgPSB0aGlzLnJvb3RzW2tdLm1vZHVsZXMoKTtcclxuICAgICAgICB2YXIgbiA9IHJzLmxlbmd0aDtcclxuICAgICAgICB2YXIgbWVyZ2VzID0gbmV3IEFycmF5KG4gKiAobiAtIDEpKTtcclxuICAgICAgICB2YXIgY3RyID0gMDtcclxuICAgICAgICBmb3IgKHZhciBpID0gMCwgaV8gPSBuIC0gMTsgaSA8IGlfOyArK2kpIHtcclxuICAgICAgICAgICAgZm9yICh2YXIgaiA9IGkgKyAxOyBqIDwgbjsgKytqKSB7XHJcbiAgICAgICAgICAgICAgICB2YXIgYSA9IHJzW2ldLCBiID0gcnNbal07XHJcbiAgICAgICAgICAgICAgICBtZXJnZXNbY3RyXSA9IHsgaWQ6IGN0ciwgbkVkZ2VzOiB0aGlzLm5FZGdlcyhhLCBiKSwgYTogYSwgYjogYiB9O1xyXG4gICAgICAgICAgICAgICAgY3RyKys7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICAgICAgcmV0dXJuIG1lcmdlcztcclxuICAgIH07XHJcbiAgICBDb25maWd1cmF0aW9uLnByb3RvdHlwZS5ncmVlZHlNZXJnZSA9IGZ1bmN0aW9uICgpIHtcclxuICAgICAgICBmb3IgKHZhciBpID0gMDsgaSA8IHRoaXMucm9vdHMubGVuZ3RoOyArK2kpIHtcclxuICAgICAgICAgICAgaWYgKHRoaXMucm9vdHNbaV0ubW9kdWxlcygpLmxlbmd0aCA8IDIpXHJcbiAgICAgICAgICAgICAgICBjb250aW51ZTtcclxuICAgICAgICAgICAgdmFyIG1zID0gdGhpcy5yb290TWVyZ2VzKGkpLnNvcnQoZnVuY3Rpb24gKGEsIGIpIHsgcmV0dXJuIGEubkVkZ2VzID09IGIubkVkZ2VzID8gYS5pZCAtIGIuaWQgOiBhLm5FZGdlcyAtIGIubkVkZ2VzOyB9KTtcclxuICAgICAgICAgICAgdmFyIG0gPSBtc1swXTtcclxuICAgICAgICAgICAgaWYgKG0ubkVkZ2VzID49IHRoaXMuUilcclxuICAgICAgICAgICAgICAgIGNvbnRpbnVlO1xyXG4gICAgICAgICAgICB0aGlzLm1lcmdlKG0uYSwgbS5iLCBpKTtcclxuICAgICAgICAgICAgcmV0dXJuIHRydWU7XHJcbiAgICAgICAgfVxyXG4gICAgfTtcclxuICAgIENvbmZpZ3VyYXRpb24ucHJvdG90eXBlLm5FZGdlcyA9IGZ1bmN0aW9uIChhLCBiKSB7XHJcbiAgICAgICAgdmFyIGluSW50ID0gYS5pbmNvbWluZy5pbnRlcnNlY3Rpb24oYi5pbmNvbWluZyksIG91dEludCA9IGEub3V0Z29pbmcuaW50ZXJzZWN0aW9uKGIub3V0Z29pbmcpO1xyXG4gICAgICAgIHJldHVybiB0aGlzLlIgLSBpbkludC5jb3VudCgpIC0gb3V0SW50LmNvdW50KCk7XHJcbiAgICB9O1xyXG4gICAgQ29uZmlndXJhdGlvbi5wcm90b3R5cGUuZ2V0R3JvdXBIaWVyYXJjaHkgPSBmdW5jdGlvbiAocmV0YXJnZXRlZEVkZ2VzKSB7XHJcbiAgICAgICAgdmFyIF90aGlzID0gdGhpcztcclxuICAgICAgICB2YXIgZ3JvdXBzID0gW107XHJcbiAgICAgICAgdmFyIHJvb3QgPSB7fTtcclxuICAgICAgICB0b0dyb3Vwcyh0aGlzLnJvb3RzWzBdLCByb290LCBncm91cHMpO1xyXG4gICAgICAgIHZhciBlcyA9IHRoaXMuYWxsRWRnZXMoKTtcclxuICAgICAgICBlcy5mb3JFYWNoKGZ1bmN0aW9uIChlKSB7XHJcbiAgICAgICAgICAgIHZhciBhID0gX3RoaXMubW9kdWxlc1tlLnNvdXJjZV07XHJcbiAgICAgICAgICAgIHZhciBiID0gX3RoaXMubW9kdWxlc1tlLnRhcmdldF07XHJcbiAgICAgICAgICAgIHJldGFyZ2V0ZWRFZGdlcy5wdXNoKG5ldyBQb3dlckVkZ2UodHlwZW9mIGEuZ2lkID09PSBcInVuZGVmaW5lZFwiID8gZS5zb3VyY2UgOiBncm91cHNbYS5naWRdLCB0eXBlb2YgYi5naWQgPT09IFwidW5kZWZpbmVkXCIgPyBlLnRhcmdldCA6IGdyb3Vwc1tiLmdpZF0sIGUudHlwZSkpO1xyXG4gICAgICAgIH0pO1xyXG4gICAgICAgIHJldHVybiBncm91cHM7XHJcbiAgICB9O1xyXG4gICAgQ29uZmlndXJhdGlvbi5wcm90b3R5cGUuYWxsRWRnZXMgPSBmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgdmFyIGVzID0gW107XHJcbiAgICAgICAgQ29uZmlndXJhdGlvbi5nZXRFZGdlcyh0aGlzLnJvb3RzWzBdLCBlcyk7XHJcbiAgICAgICAgcmV0dXJuIGVzO1xyXG4gICAgfTtcclxuICAgIENvbmZpZ3VyYXRpb24uZ2V0RWRnZXMgPSBmdW5jdGlvbiAobW9kdWxlcywgZXMpIHtcclxuICAgICAgICBtb2R1bGVzLmZvckFsbChmdW5jdGlvbiAobSkge1xyXG4gICAgICAgICAgICBtLmdldEVkZ2VzKGVzKTtcclxuICAgICAgICAgICAgQ29uZmlndXJhdGlvbi5nZXRFZGdlcyhtLmNoaWxkcmVuLCBlcyk7XHJcbiAgICAgICAgfSk7XHJcbiAgICB9O1xyXG4gICAgcmV0dXJuIENvbmZpZ3VyYXRpb247XHJcbn0oKSk7XHJcbmV4cG9ydHMuQ29uZmlndXJhdGlvbiA9IENvbmZpZ3VyYXRpb247XHJcbmZ1bmN0aW9uIHRvR3JvdXBzKG1vZHVsZXMsIGdyb3VwLCBncm91cHMpIHtcclxuICAgIG1vZHVsZXMuZm9yQWxsKGZ1bmN0aW9uIChtKSB7XHJcbiAgICAgICAgaWYgKG0uaXNMZWFmKCkpIHtcclxuICAgICAgICAgICAgaWYgKCFncm91cC5sZWF2ZXMpXHJcbiAgICAgICAgICAgICAgICBncm91cC5sZWF2ZXMgPSBbXTtcclxuICAgICAgICAgICAgZ3JvdXAubGVhdmVzLnB1c2gobS5pZCk7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIGVsc2Uge1xyXG4gICAgICAgICAgICB2YXIgZyA9IGdyb3VwO1xyXG4gICAgICAgICAgICBtLmdpZCA9IGdyb3Vwcy5sZW5ndGg7XHJcbiAgICAgICAgICAgIGlmICghbS5pc0lzbGFuZCgpIHx8IG0uaXNQcmVkZWZpbmVkKCkpIHtcclxuICAgICAgICAgICAgICAgIGcgPSB7IGlkOiBtLmdpZCB9O1xyXG4gICAgICAgICAgICAgICAgaWYgKG0uaXNQcmVkZWZpbmVkKCkpXHJcbiAgICAgICAgICAgICAgICAgICAgZm9yICh2YXIgcHJvcCBpbiBtLmRlZmluaXRpb24pXHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGdbcHJvcF0gPSBtLmRlZmluaXRpb25bcHJvcF07XHJcbiAgICAgICAgICAgICAgICBpZiAoIWdyb3VwLmdyb3VwcylcclxuICAgICAgICAgICAgICAgICAgICBncm91cC5ncm91cHMgPSBbXTtcclxuICAgICAgICAgICAgICAgIGdyb3VwLmdyb3Vwcy5wdXNoKG0uZ2lkKTtcclxuICAgICAgICAgICAgICAgIGdyb3Vwcy5wdXNoKGcpO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIHRvR3JvdXBzKG0uY2hpbGRyZW4sIGcsIGdyb3Vwcyk7XHJcbiAgICAgICAgfVxyXG4gICAgfSk7XHJcbn1cclxudmFyIE1vZHVsZSA9IChmdW5jdGlvbiAoKSB7XHJcbiAgICBmdW5jdGlvbiBNb2R1bGUoaWQsIG91dGdvaW5nLCBpbmNvbWluZywgY2hpbGRyZW4sIGRlZmluaXRpb24pIHtcclxuICAgICAgICBpZiAob3V0Z29pbmcgPT09IHZvaWQgMCkgeyBvdXRnb2luZyA9IG5ldyBMaW5rU2V0cygpOyB9XHJcbiAgICAgICAgaWYgKGluY29taW5nID09PSB2b2lkIDApIHsgaW5jb21pbmcgPSBuZXcgTGlua1NldHMoKTsgfVxyXG4gICAgICAgIGlmIChjaGlsZHJlbiA9PT0gdm9pZCAwKSB7IGNoaWxkcmVuID0gbmV3IE1vZHVsZVNldCgpOyB9XHJcbiAgICAgICAgdGhpcy5pZCA9IGlkO1xyXG4gICAgICAgIHRoaXMub3V0Z29pbmcgPSBvdXRnb2luZztcclxuICAgICAgICB0aGlzLmluY29taW5nID0gaW5jb21pbmc7XHJcbiAgICAgICAgdGhpcy5jaGlsZHJlbiA9IGNoaWxkcmVuO1xyXG4gICAgICAgIHRoaXMuZGVmaW5pdGlvbiA9IGRlZmluaXRpb247XHJcbiAgICB9XHJcbiAgICBNb2R1bGUucHJvdG90eXBlLmdldEVkZ2VzID0gZnVuY3Rpb24gKGVzKSB7XHJcbiAgICAgICAgdmFyIF90aGlzID0gdGhpcztcclxuICAgICAgICB0aGlzLm91dGdvaW5nLmZvckFsbChmdW5jdGlvbiAobXMsIGVkZ2V0eXBlKSB7XHJcbiAgICAgICAgICAgIG1zLmZvckFsbChmdW5jdGlvbiAodGFyZ2V0KSB7XHJcbiAgICAgICAgICAgICAgICBlcy5wdXNoKG5ldyBQb3dlckVkZ2UoX3RoaXMuaWQsIHRhcmdldC5pZCwgZWRnZXR5cGUpKTtcclxuICAgICAgICAgICAgfSk7XHJcbiAgICAgICAgfSk7XHJcbiAgICB9O1xyXG4gICAgTW9kdWxlLnByb3RvdHlwZS5pc0xlYWYgPSBmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgcmV0dXJuIHRoaXMuY2hpbGRyZW4uY291bnQoKSA9PT0gMDtcclxuICAgIH07XHJcbiAgICBNb2R1bGUucHJvdG90eXBlLmlzSXNsYW5kID0gZnVuY3Rpb24gKCkge1xyXG4gICAgICAgIHJldHVybiB0aGlzLm91dGdvaW5nLmNvdW50KCkgPT09IDAgJiYgdGhpcy5pbmNvbWluZy5jb3VudCgpID09PSAwO1xyXG4gICAgfTtcclxuICAgIE1vZHVsZS5wcm90b3R5cGUuaXNQcmVkZWZpbmVkID0gZnVuY3Rpb24gKCkge1xyXG4gICAgICAgIHJldHVybiB0eXBlb2YgdGhpcy5kZWZpbml0aW9uICE9PSBcInVuZGVmaW5lZFwiO1xyXG4gICAgfTtcclxuICAgIHJldHVybiBNb2R1bGU7XHJcbn0oKSk7XHJcbmV4cG9ydHMuTW9kdWxlID0gTW9kdWxlO1xyXG5mdW5jdGlvbiBpbnRlcnNlY3Rpb24obSwgbikge1xyXG4gICAgdmFyIGkgPSB7fTtcclxuICAgIGZvciAodmFyIHYgaW4gbSlcclxuICAgICAgICBpZiAodiBpbiBuKVxyXG4gICAgICAgICAgICBpW3ZdID0gbVt2XTtcclxuICAgIHJldHVybiBpO1xyXG59XHJcbnZhciBNb2R1bGVTZXQgPSAoZnVuY3Rpb24gKCkge1xyXG4gICAgZnVuY3Rpb24gTW9kdWxlU2V0KCkge1xyXG4gICAgICAgIHRoaXMudGFibGUgPSB7fTtcclxuICAgIH1cclxuICAgIE1vZHVsZVNldC5wcm90b3R5cGUuY291bnQgPSBmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgcmV0dXJuIE9iamVjdC5rZXlzKHRoaXMudGFibGUpLmxlbmd0aDtcclxuICAgIH07XHJcbiAgICBNb2R1bGVTZXQucHJvdG90eXBlLmludGVyc2VjdGlvbiA9IGZ1bmN0aW9uIChvdGhlcikge1xyXG4gICAgICAgIHZhciByZXN1bHQgPSBuZXcgTW9kdWxlU2V0KCk7XHJcbiAgICAgICAgcmVzdWx0LnRhYmxlID0gaW50ZXJzZWN0aW9uKHRoaXMudGFibGUsIG90aGVyLnRhYmxlKTtcclxuICAgICAgICByZXR1cm4gcmVzdWx0O1xyXG4gICAgfTtcclxuICAgIE1vZHVsZVNldC5wcm90b3R5cGUuaW50ZXJzZWN0aW9uQ291bnQgPSBmdW5jdGlvbiAob3RoZXIpIHtcclxuICAgICAgICByZXR1cm4gdGhpcy5pbnRlcnNlY3Rpb24ob3RoZXIpLmNvdW50KCk7XHJcbiAgICB9O1xyXG4gICAgTW9kdWxlU2V0LnByb3RvdHlwZS5jb250YWlucyA9IGZ1bmN0aW9uIChpZCkge1xyXG4gICAgICAgIHJldHVybiBpZCBpbiB0aGlzLnRhYmxlO1xyXG4gICAgfTtcclxuICAgIE1vZHVsZVNldC5wcm90b3R5cGUuYWRkID0gZnVuY3Rpb24gKG0pIHtcclxuICAgICAgICB0aGlzLnRhYmxlW20uaWRdID0gbTtcclxuICAgIH07XHJcbiAgICBNb2R1bGVTZXQucHJvdG90eXBlLnJlbW92ZSA9IGZ1bmN0aW9uIChtKSB7XHJcbiAgICAgICAgZGVsZXRlIHRoaXMudGFibGVbbS5pZF07XHJcbiAgICB9O1xyXG4gICAgTW9kdWxlU2V0LnByb3RvdHlwZS5mb3JBbGwgPSBmdW5jdGlvbiAoZikge1xyXG4gICAgICAgIGZvciAodmFyIG1pZCBpbiB0aGlzLnRhYmxlKSB7XHJcbiAgICAgICAgICAgIGYodGhpcy50YWJsZVttaWRdKTtcclxuICAgICAgICB9XHJcbiAgICB9O1xyXG4gICAgTW9kdWxlU2V0LnByb3RvdHlwZS5tb2R1bGVzID0gZnVuY3Rpb24gKCkge1xyXG4gICAgICAgIHZhciB2cyA9IFtdO1xyXG4gICAgICAgIHRoaXMuZm9yQWxsKGZ1bmN0aW9uIChtKSB7XHJcbiAgICAgICAgICAgIGlmICghbS5pc1ByZWRlZmluZWQoKSlcclxuICAgICAgICAgICAgICAgIHZzLnB1c2gobSk7XHJcbiAgICAgICAgfSk7XHJcbiAgICAgICAgcmV0dXJuIHZzO1xyXG4gICAgfTtcclxuICAgIHJldHVybiBNb2R1bGVTZXQ7XHJcbn0oKSk7XHJcbmV4cG9ydHMuTW9kdWxlU2V0ID0gTW9kdWxlU2V0O1xyXG52YXIgTGlua1NldHMgPSAoZnVuY3Rpb24gKCkge1xyXG4gICAgZnVuY3Rpb24gTGlua1NldHMoKSB7XHJcbiAgICAgICAgdGhpcy5zZXRzID0ge307XHJcbiAgICAgICAgdGhpcy5uID0gMDtcclxuICAgIH1cclxuICAgIExpbmtTZXRzLnByb3RvdHlwZS5jb3VudCA9IGZ1bmN0aW9uICgpIHtcclxuICAgICAgICByZXR1cm4gdGhpcy5uO1xyXG4gICAgfTtcclxuICAgIExpbmtTZXRzLnByb3RvdHlwZS5jb250YWlucyA9IGZ1bmN0aW9uIChpZCkge1xyXG4gICAgICAgIHZhciByZXN1bHQgPSBmYWxzZTtcclxuICAgICAgICB0aGlzLmZvckFsbE1vZHVsZXMoZnVuY3Rpb24gKG0pIHtcclxuICAgICAgICAgICAgaWYgKCFyZXN1bHQgJiYgbS5pZCA9PSBpZCkge1xyXG4gICAgICAgICAgICAgICAgcmVzdWx0ID0gdHJ1ZTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH0pO1xyXG4gICAgICAgIHJldHVybiByZXN1bHQ7XHJcbiAgICB9O1xyXG4gICAgTGlua1NldHMucHJvdG90eXBlLmFkZCA9IGZ1bmN0aW9uIChsaW5rdHlwZSwgbSkge1xyXG4gICAgICAgIHZhciBzID0gbGlua3R5cGUgaW4gdGhpcy5zZXRzID8gdGhpcy5zZXRzW2xpbmt0eXBlXSA6IHRoaXMuc2V0c1tsaW5rdHlwZV0gPSBuZXcgTW9kdWxlU2V0KCk7XHJcbiAgICAgICAgcy5hZGQobSk7XHJcbiAgICAgICAgKyt0aGlzLm47XHJcbiAgICB9O1xyXG4gICAgTGlua1NldHMucHJvdG90eXBlLnJlbW92ZSA9IGZ1bmN0aW9uIChsaW5rdHlwZSwgbSkge1xyXG4gICAgICAgIHZhciBtcyA9IHRoaXMuc2V0c1tsaW5rdHlwZV07XHJcbiAgICAgICAgbXMucmVtb3ZlKG0pO1xyXG4gICAgICAgIGlmIChtcy5jb3VudCgpID09PSAwKSB7XHJcbiAgICAgICAgICAgIGRlbGV0ZSB0aGlzLnNldHNbbGlua3R5cGVdO1xyXG4gICAgICAgIH1cclxuICAgICAgICAtLXRoaXMubjtcclxuICAgIH07XHJcbiAgICBMaW5rU2V0cy5wcm90b3R5cGUuZm9yQWxsID0gZnVuY3Rpb24gKGYpIHtcclxuICAgICAgICBmb3IgKHZhciBsaW5rdHlwZSBpbiB0aGlzLnNldHMpIHtcclxuICAgICAgICAgICAgZih0aGlzLnNldHNbbGlua3R5cGVdLCBOdW1iZXIobGlua3R5cGUpKTtcclxuICAgICAgICB9XHJcbiAgICB9O1xyXG4gICAgTGlua1NldHMucHJvdG90eXBlLmZvckFsbE1vZHVsZXMgPSBmdW5jdGlvbiAoZikge1xyXG4gICAgICAgIHRoaXMuZm9yQWxsKGZ1bmN0aW9uIChtcywgbHQpIHsgcmV0dXJuIG1zLmZvckFsbChmKTsgfSk7XHJcbiAgICB9O1xyXG4gICAgTGlua1NldHMucHJvdG90eXBlLmludGVyc2VjdGlvbiA9IGZ1bmN0aW9uIChvdGhlcikge1xyXG4gICAgICAgIHZhciByZXN1bHQgPSBuZXcgTGlua1NldHMoKTtcclxuICAgICAgICB0aGlzLmZvckFsbChmdW5jdGlvbiAobXMsIGx0KSB7XHJcbiAgICAgICAgICAgIGlmIChsdCBpbiBvdGhlci5zZXRzKSB7XHJcbiAgICAgICAgICAgICAgICB2YXIgaSA9IG1zLmludGVyc2VjdGlvbihvdGhlci5zZXRzW2x0XSksIG4gPSBpLmNvdW50KCk7XHJcbiAgICAgICAgICAgICAgICBpZiAobiA+IDApIHtcclxuICAgICAgICAgICAgICAgICAgICByZXN1bHQuc2V0c1tsdF0gPSBpO1xyXG4gICAgICAgICAgICAgICAgICAgIHJlc3VsdC5uICs9IG47XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9KTtcclxuICAgICAgICByZXR1cm4gcmVzdWx0O1xyXG4gICAgfTtcclxuICAgIHJldHVybiBMaW5rU2V0cztcclxufSgpKTtcclxuZXhwb3J0cy5MaW5rU2V0cyA9IExpbmtTZXRzO1xyXG5mdW5jdGlvbiBpbnRlcnNlY3Rpb25Db3VudChtLCBuKSB7XHJcbiAgICByZXR1cm4gT2JqZWN0LmtleXMoaW50ZXJzZWN0aW9uKG0sIG4pKS5sZW5ndGg7XHJcbn1cclxuZnVuY3Rpb24gZ2V0R3JvdXBzKG5vZGVzLCBsaW5rcywgbGEsIHJvb3RHcm91cCkge1xyXG4gICAgdmFyIG4gPSBub2Rlcy5sZW5ndGgsIGMgPSBuZXcgQ29uZmlndXJhdGlvbihuLCBsaW5rcywgbGEsIHJvb3RHcm91cCk7XHJcbiAgICB3aGlsZSAoYy5ncmVlZHlNZXJnZSgpKVxyXG4gICAgICAgIDtcclxuICAgIHZhciBwb3dlckVkZ2VzID0gW107XHJcbiAgICB2YXIgZyA9IGMuZ2V0R3JvdXBIaWVyYXJjaHkocG93ZXJFZGdlcyk7XHJcbiAgICBwb3dlckVkZ2VzLmZvckVhY2goZnVuY3Rpb24gKGUpIHtcclxuICAgICAgICB2YXIgZiA9IGZ1bmN0aW9uIChlbmQpIHtcclxuICAgICAgICAgICAgdmFyIGcgPSBlW2VuZF07XHJcbiAgICAgICAgICAgIGlmICh0eXBlb2YgZyA9PSBcIm51bWJlclwiKVxyXG4gICAgICAgICAgICAgICAgZVtlbmRdID0gbm9kZXNbZ107XHJcbiAgICAgICAgfTtcclxuICAgICAgICBmKFwic291cmNlXCIpO1xyXG4gICAgICAgIGYoXCJ0YXJnZXRcIik7XHJcbiAgICB9KTtcclxuICAgIHJldHVybiB7IGdyb3VwczogZywgcG93ZXJFZGdlczogcG93ZXJFZGdlcyB9O1xyXG59XHJcbmV4cG9ydHMuZ2V0R3JvdXBzID0gZ2V0R3JvdXBzO1xyXG4vLyMgc291cmNlTWFwcGluZ1VSTD1kYXRhOmFwcGxpY2F0aW9uL2pzb247YmFzZTY0LGV5SjJaWEp6YVc5dUlqb3pMQ0ptYVd4bElqb2ljRzkzWlhKbmNtRndhQzVxY3lJc0luTnZkWEpqWlZKdmIzUWlPaUlpTENKemIzVnlZMlZ6SWpwYklpNHVMeTR1TDFkbFlrTnZiR0V2YzNKakwzQnZkMlZ5WjNKaGNHZ3VkSE1pWFN3aWJtRnRaWE1pT2x0ZExDSnRZWEJ3YVc1bmN5STZJanM3UVVGUFNUdEpRVU5KTEcxQ1FVTlhMRTFCUVZjc1JVRkRXQ3hOUVVGWExFVkJRMWdzU1VGQldUdFJRVVphTEZkQlFVMHNSMEZCVGl4TlFVRk5MRU5CUVVzN1VVRkRXQ3hYUVVGTkxFZEJRVTRzVFVGQlRTeERRVUZMTzFGQlExZ3NVMEZCU1N4SFFVRktMRWxCUVVrc1EwRkJVVHRKUVVGSkxFTkJRVU03U1VGRGFFTXNaMEpCUVVNN1FVRkJSQ3hEUVVGRExFRkJURVFzU1VGTFF6dEJRVXhaTERoQ1FVRlRPMEZCVDNSQ08wbEJVMGtzZFVKQlFWa3NRMEZCVXl4RlFVRkZMRXRCUVdFc1JVRkJWU3haUVVGdlF5eEZRVUZGTEZOQlFXbENPMUZCUVhKSExHbENRV3RDUXp0UlFXeENOa01zYVVKQlFWa3NSMEZCV2l4WlFVRlpMRU5CUVhkQ08xRkJRemxGTEVsQlFVa3NRMEZCUXl4UFFVRlBMRWRCUVVjc1NVRkJTU3hMUVVGTExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTTdVVUZETlVJc1NVRkJTU3hEUVVGRExFdEJRVXNzUjBGQlJ5eEZRVUZGTEVOQlFVTTdVVUZEYUVJc1NVRkJTU3hUUVVGVExFVkJRVVU3V1VGRFdDeEpRVUZKTEVOQlFVTXNiMEpCUVc5Q0xFTkJRVU1zVTBGQlV5eERRVUZETEVOQlFVTTdVMEZEZUVNN1lVRkJUVHRaUVVOSUxFbEJRVWtzUTBGQlF5eExRVUZMTEVOQlFVTXNTVUZCU1N4RFFVRkRMRWxCUVVrc1UwRkJVeXhGUVVGRkxFTkJRVU1zUTBGQlF6dFpRVU5xUXl4TFFVRkxMRWxCUVVrc1EwRkJReXhIUVVGSExFTkJRVU1zUlVGQlJTeERRVUZETEVkQlFVY3NRMEZCUXl4RlFVRkZMRVZCUVVVc1EwRkJRenRuUWtGRGRFSXNTVUZCU1N4RFFVRkRMRXRCUVVzc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eEhRVUZITEVOQlFVTXNTVUZCU1N4RFFVRkRMRTlCUVU4c1EwRkJReXhEUVVGRExFTkJRVU1zUjBGQlJ5eEpRVUZKTEUxQlFVMHNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRE8xTkJRekZFTzFGQlEwUXNTVUZCU1N4RFFVRkRMRU5CUVVNc1IwRkJSeXhMUVVGTExFTkJRVU1zVFVGQlRTeERRVUZETzFGQlEzUkNMRXRCUVVzc1EwRkJReXhQUVVGUExFTkJRVU1zVlVGQlFTeERRVUZETzFsQlExZ3NTVUZCU1N4RFFVRkRMRWRCUVVjc1MwRkJTU3hEUVVGRExFOUJRVThzUTBGQlF5eFpRVUZaTEVOQlFVTXNZMEZCWXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFVkJRMmhFTEVOQlFVTXNSMEZCUnl4TFFVRkpMRU5CUVVNc1QwRkJUeXhEUVVGRExGbEJRVmtzUTBGQlF5eGpRVUZqTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1JVRkRhRVFzU1VGQlNTeEhRVUZITEZsQlFWa3NRMEZCUXl4UFFVRlBMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU03V1VGRGJrTXNRMEZCUXl4RFFVRkRMRkZCUVZFc1EwRkJReXhIUVVGSExFTkJRVU1zU1VGQlNTeEZRVUZGTEVOQlFVTXNRMEZCUXl4RFFVRkRPMWxCUTNoQ0xFTkJRVU1zUTBGQlF5eFJRVUZSTEVOQlFVTXNSMEZCUnl4RFFVRkRMRWxCUVVrc1JVRkJSU3hEUVVGRExFTkJRVU1zUTBGQlF6dFJRVU0xUWl4RFFVRkRMRU5CUVVNc1EwRkJRenRKUVVOUUxFTkJRVU03U1VGRlR5dzBRMEZCYjBJc1IwRkJOVUlzVlVGQk5rSXNTMEZCU3p0UlFVTTVRaXhKUVVGSkxGTkJRVk1zUjBGQlJ5eEpRVUZKTEZOQlFWTXNSVUZCUlN4RFFVRkRPMUZCUTJoRExFbEJRVWtzUTBGQlF5eExRVUZMTEVOQlFVTXNTVUZCU1N4RFFVRkRMRk5CUVZNc1EwRkJReXhEUVVGRE8xRkJRek5DTEV0QlFVc3NTVUZCU1N4RFFVRkRMRWRCUVVjc1EwRkJReXhGUVVGRkxFTkJRVU1zUjBGQlJ5eExRVUZMTEVOQlFVTXNUVUZCVFN4RFFVRkRMRTFCUVUwc1JVRkJSU3hGUVVGRkxFTkJRVU1zUlVGQlJUdFpRVU14UXl4SlFVRkpMRWxCUVVrc1IwRkJSeXhMUVVGTExFTkJRVU1zVFVGQlRTeERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRPMWxCUXpOQ0xFbEJRVWtzVFVGQlRTeEhRVUZITEVsQlFVa3NUVUZCVFN4RFFVRkRMRWxCUVVrc1EwRkJReXhGUVVGRkxFTkJRVU1zUTBGQlF6dFpRVU5xUXl4SlFVRkpMRU5CUVVNc1QwRkJUeXhEUVVGRExFbEJRVWtzUTBGQlF5eEZRVUZGTEVOQlFVTXNSMEZCUnl4TlFVRk5MRU5CUVVNN1dVRkRMMElzVTBGQlV5eERRVUZETEVkQlFVY3NRMEZCUXl4TlFVRk5MRU5CUVVNc1EwRkJRenRUUVVONlFqdFJRVU5FTEVsQlFVa3NTMEZCU3l4RFFVRkRMRTFCUVUwc1JVRkJSVHRaUVVOa0xFdEJRVXNzU1VGQlNTeERRVUZETEVkQlFVY3NRMEZCUXl4RlFVRkZMRU5CUVVNc1IwRkJSeXhMUVVGTExFTkJRVU1zVFVGQlRTeERRVUZETEUxQlFVMHNSVUZCUlN4RlFVRkZMRU5CUVVNc1JVRkJSVHRuUWtGRE1VTXNTVUZCU1N4TFFVRkxMRWRCUVVjc1MwRkJTeXhEUVVGRExFMUJRVTBzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXp0blFrRkZOVUlzU1VGQlNTeFZRVUZWTEVkQlFVY3NSVUZCUlN4RFFVRkRPMmRDUVVOd1FpeExRVUZMTEVsQlFVa3NTVUZCU1N4SlFVRkpMRXRCUVVzN2IwSkJRMnhDTEVsQlFVa3NTVUZCU1N4TFFVRkxMRkZCUVZFc1NVRkJTU3hKUVVGSkxFdEJRVXNzVVVGQlVTeEpRVUZKTEV0QlFVc3NRMEZCUXl4alFVRmpMRU5CUVVNc1NVRkJTU3hEUVVGRE8zZENRVU53UlN4VlFVRlZMRU5CUVVNc1NVRkJTU3hEUVVGRExFZEJRVWNzUzBGQlN5eERRVUZETEVsQlFVa3NRMEZCUXl4RFFVRkRPMmRDUVVWMlF5eFRRVUZUTEVOQlFVTXNSMEZCUnl4RFFVRkRMRWxCUVVrc1RVRkJUU3hEUVVGRExFTkJRVU1zUTBGQlF5eEhRVUZETEVOQlFVTXNSVUZCUlN4SlFVRkpMRkZCUVZFc1JVRkJSU3hGUVVGRkxFbEJRVWtzVVVGQlVTeEZRVUZGTEVWQlFVVXNTVUZCU1N4RFFVRkRMRzlDUVVGdlFpeERRVUZETEV0QlFVc3NRMEZCUXl4RlFVRkZMRlZCUVZVc1EwRkJReXhEUVVGRExFTkJRVU03WVVGRGFrZzdVMEZEU2p0UlFVTkVMRTlCUVU4c1UwRkJVeXhEUVVGRE8wbEJRM0JDTEVOQlFVTTdTVUZIUml3MlFrRkJTeXhIUVVGTUxGVkJRVTBzUTBGQlV5eEZRVUZGTEVOQlFWTXNSVUZCUlN4RFFVRmhPMUZCUVdJc2EwSkJRVUVzUlVGQlFTeExRVUZoTzFGQlEzSkRMRWxCUVVrc1MwRkJTeXhIUVVGSExFTkJRVU1zUTBGQlF5eFJRVUZSTEVOQlFVTXNXVUZCV1N4RFFVRkRMRU5CUVVNc1EwRkJReXhSUVVGUkxFTkJRVU1zUlVGRE0wTXNUVUZCVFN4SFFVRkhMRU5CUVVNc1EwRkJReXhSUVVGUkxFTkJRVU1zV1VGQldTeERRVUZETEVOQlFVTXNRMEZCUXl4UlFVRlJMRU5CUVVNc1EwRkJRenRSUVVOcVJDeEpRVUZKTEZGQlFWRXNSMEZCUnl4SlFVRkpMRk5CUVZNc1JVRkJSU3hEUVVGRE8xRkJReTlDTEZGQlFWRXNRMEZCUXl4SFFVRkhMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU03VVVGRGFFSXNVVUZCVVN4RFFVRkRMRWRCUVVjc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dFJRVU5vUWl4SlFVRkpMRU5CUVVNc1IwRkJSeXhKUVVGSkxFMUJRVTBzUTBGQlF5eEpRVUZKTEVOQlFVTXNUMEZCVHl4RFFVRkRMRTFCUVUwc1JVRkJSU3hOUVVGTkxFVkJRVVVzUzBGQlN5eEZRVUZGTEZGQlFWRXNRMEZCUXl4RFFVRkRPMUZCUTJwRkxFbEJRVWtzUTBGQlF5eFBRVUZQTEVOQlFVTXNTVUZCU1N4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRE8xRkJRM0pDTEVsQlFVa3NUVUZCVFN4SFFVRkhMRlZCUVVNc1EwRkJWeXhGUVVGRkxFTkJRVk1zUlVGQlJTeERRVUZUTzFsQlF6TkRMRU5CUVVNc1EwRkJReXhOUVVGTkxFTkJRVU1zVlVGQlF5eEZRVUZGTEVWQlFVVXNVVUZCVVR0blFrRkRiRUlzUlVGQlJTeERRVUZETEUxQlFVMHNRMEZCUXl4VlFVRkJMRU5CUVVNN2IwSkJRMUFzU1VGQlNTeEhRVUZITEVkQlFXRXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRE8yOUNRVU42UWl4SFFVRkhMRU5CUVVNc1IwRkJSeXhEUVVGRExGRkJRVkVzUlVGQlJTeERRVUZETEVOQlFVTXNRMEZCUXp0dlFrRkRja0lzUjBGQlJ5eERRVUZETEUxQlFVMHNRMEZCUXl4UlFVRlJMRVZCUVVVc1EwRkJReXhEUVVGRExFTkJRVU03YjBKQlEzaENMRWRCUVVjc1EwRkJReXhOUVVGTkxFTkJRVU1zVVVGQlVTeEZRVUZGTEVOQlFVTXNRMEZCUXl4RFFVRkRPMjlDUVVOaUxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVVXNRMEZCUXl4TlFVRk5MRU5CUVVNc1VVRkJVU3hGUVVGRkxFTkJRVU1zUTBGQlF5eERRVUZETzI5Q1FVTXhRaXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZGTEVOQlFVTXNUVUZCVFN4RFFVRkRMRkZCUVZFc1JVRkJSU3hEUVVGRExFTkJRVU1zUTBGQlF6dG5Ra0ZEZWtNc1EwRkJReXhEUVVGRExFTkJRVU03V1VGRFVDeERRVUZETEVOQlFVTXNRMEZCUXp0UlFVTlFMRU5CUVVNc1EwRkJRenRSUVVOR0xFMUJRVTBzUTBGQlF5eE5RVUZOTEVWQlFVVXNWVUZCVlN4RlFVRkZMRlZCUVZVc1EwRkJReXhEUVVGRE8xRkJRM1pETEUxQlFVMHNRMEZCUXl4TFFVRkxMRVZCUVVVc1ZVRkJWU3hGUVVGRkxGVkJRVlVzUTBGQlF5eERRVUZETzFGQlEzUkRMRWxCUVVrc1EwRkJReXhEUVVGRExFbEJRVWtzUzBGQlN5eERRVUZETEV0QlFVc3NSVUZCUlN4SFFVRkhMRTFCUVUwc1EwRkJReXhMUVVGTExFVkJRVVVzUTBGQlF6dFJRVU42UXl4SlFVRkpMRU5CUVVNc1MwRkJTeXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEUxQlFVMHNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJRenRSUVVONFFpeEpRVUZKTEVOQlFVTXNTMEZCU3l4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFMUJRVTBzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXp0UlFVTjRRaXhKUVVGSkxFTkJRVU1zUzBGQlN5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRWRCUVVjc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dFJRVU55UWl4UFFVRlBMRU5CUVVNc1EwRkJRenRKUVVOaUxFTkJRVU03U1VGRlR5eHJRMEZCVlN4SFFVRnNRaXhWUVVGdFFpeERRVUZoTzFGQlFXSXNhMEpCUVVFc1JVRkJRU3hMUVVGaE8xRkJUVFZDTEVsQlFVa3NSVUZCUlN4SFFVRkhMRWxCUVVrc1EwRkJReXhMUVVGTExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNUMEZCVHl4RlFVRkZMRU5CUVVNN1VVRkRha01zU1VGQlNTeERRVUZETEVkQlFVY3NSVUZCUlN4RFFVRkRMRTFCUVUwc1EwRkJRenRSUVVOc1FpeEpRVUZKTEUxQlFVMHNSMEZCUnl4SlFVRkpMRXRCUVVzc1EwRkJReXhEUVVGRExFZEJRVWNzUTBGQlF5eERRVUZETEVkQlFVY3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJRenRSUVVOd1F5eEpRVUZKTEVkQlFVY3NSMEZCUnl4RFFVRkRMRU5CUVVNN1VVRkRXaXhMUVVGTExFbEJRVWtzUTBGQlF5eEhRVUZITEVOQlFVTXNSVUZCUlN4RlFVRkZMRWRCUVVjc1EwRkJReXhIUVVGSExFTkJRVU1zUlVGQlJTeERRVUZETEVkQlFVY3NSVUZCUlN4RlFVRkZMRVZCUVVVc1EwRkJReXhGUVVGRk8xbEJRM0pETEV0QlFVc3NTVUZCU1N4RFFVRkRMRWRCUVVjc1EwRkJReXhIUVVGRExFTkJRVU1zUlVGQlJTeERRVUZETEVkQlFVY3NRMEZCUXl4RlFVRkZMRVZCUVVVc1EwRkJReXhGUVVGRk8yZENRVU14UWl4SlFVRkpMRU5CUVVNc1IwRkJSeXhGUVVGRkxFTkJRVU1zUTBGQlF5eERRVUZETEVWQlFVVXNRMEZCUXl4SFFVRkhMRVZCUVVVc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dG5Ra0ZEZWtJc1RVRkJUU3hEUVVGRExFZEJRVWNzUTBGQlF5eEhRVUZITEVWQlFVVXNSVUZCUlN4RlFVRkZMRWRCUVVjc1JVRkJSU3hOUVVGTkxFVkJRVVVzU1VGQlNTeERRVUZETEUxQlFVMHNRMEZCUXl4RFFVRkRMRVZCUVVVc1EwRkJReXhEUVVGRExFVkJRVVVzUTBGQlF5eEZRVUZGTEVOQlFVTXNSVUZCUlN4RFFVRkRMRVZCUVVVc1EwRkJReXhGUVVGRkxFTkJRVU03WjBKQlEycEZMRWRCUVVjc1JVRkJSU3hEUVVGRE8yRkJRMVE3VTBGRFNqdFJRVU5FTEU5QlFVOHNUVUZCVFN4RFFVRkRPMGxCUTJ4Q0xFTkJRVU03U1VGRlJDeHRRMEZCVnl4SFFVRllPMUZCUTBrc1MwRkJTeXhKUVVGSkxFTkJRVU1zUjBGQlJ5eERRVUZETEVWQlFVVXNRMEZCUXl4SFFVRkhMRWxCUVVrc1EwRkJReXhMUVVGTExFTkJRVU1zVFVGQlRTeEZRVUZGTEVWQlFVVXNRMEZCUXl4RlFVRkZPMWxCUlhoRExFbEJRVWtzU1VGQlNTeERRVUZETEV0QlFVc3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhQUVVGUExFVkJRVVVzUTBGQlF5eE5RVUZOTEVkQlFVY3NRMEZCUXp0blFrRkJSU3hUUVVGVE8xbEJSMnBFTEVsQlFVa3NSVUZCUlN4SFFVRkhMRWxCUVVrc1EwRkJReXhWUVVGVkxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNTVUZCU1N4RFFVRkRMRlZCUVVNc1EwRkJReXhGUVVGRkxFTkJRVU1zU1VGQlN5eFBRVUZCTEVOQlFVTXNRMEZCUXl4TlFVRk5MRWxCUVVrc1EwRkJReXhEUVVGRExFMUJRVTBzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRVZCUVVVc1IwRkJSeXhEUVVGRExFTkJRVU1zUlVGQlJTeERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1RVRkJUU3hIUVVGSExFTkJRVU1zUTBGQlF5eE5RVUZOTEVWQlFYaEVMRU5CUVhkRUxFTkJRVU1zUTBGQlF6dFpRVU55Unl4SlFVRkpMRU5CUVVNc1IwRkJSeXhGUVVGRkxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTTdXVUZEWkN4SlFVRkpMRU5CUVVNc1EwRkJReXhOUVVGTkxFbEJRVWtzU1VGQlNTeERRVUZETEVOQlFVTTdaMEpCUVVVc1UwRkJVenRaUVVOcVF5eEpRVUZKTEVOQlFVTXNTMEZCU3l4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFVkJRVVVzUTBGQlF5eERRVUZETEVOQlFVTXNSVUZCUlN4RFFVRkRMRU5CUVVNc1EwRkJRenRaUVVONFFpeFBRVUZQTEVsQlFVa3NRMEZCUXp0VFFVTm1PMGxCUTB3c1EwRkJRenRKUVVWUExEaENRVUZOTEVkQlFXUXNWVUZCWlN4RFFVRlRMRVZCUVVVc1EwRkJVenRSUVVNdlFpeEpRVUZKTEV0QlFVc3NSMEZCUnl4RFFVRkRMRU5CUVVNc1VVRkJVU3hEUVVGRExGbEJRVmtzUTBGQlF5eERRVUZETEVOQlFVTXNVVUZCVVN4RFFVRkRMRVZCUXpORExFMUJRVTBzUjBGQlJ5eERRVUZETEVOQlFVTXNVVUZCVVN4RFFVRkRMRmxCUVZrc1EwRkJReXhEUVVGRExFTkJRVU1zVVVGQlVTeERRVUZETEVOQlFVTTdVVUZEYWtRc1QwRkJUeXhKUVVGSkxFTkJRVU1zUTBGQlF5eEhRVUZITEV0QlFVc3NRMEZCUXl4TFFVRkxMRVZCUVVVc1IwRkJSeXhOUVVGTkxFTkJRVU1zUzBGQlN5eEZRVUZGTEVOQlFVTTdTVUZEYmtRc1EwRkJRenRKUVVWRUxIbERRVUZwUWl4SFFVRnFRaXhWUVVGclFpeGxRVUUwUWp0UlFVRTVReXhwUWtGbFF6dFJRV1JITEVsQlFVa3NUVUZCVFN4SFFVRkhMRVZCUVVVc1EwRkJRenRSUVVOb1FpeEpRVUZKTEVsQlFVa3NSMEZCUnl4RlFVRkZMRU5CUVVNN1VVRkRaQ3hSUVVGUkxFTkJRVU1zU1VGQlNTeERRVUZETEV0QlFVc3NRMEZCUXl4RFFVRkRMRU5CUVVNc1JVRkJSU3hKUVVGSkxFVkJRVVVzVFVGQlRTeERRVUZETEVOQlFVTTdVVUZEZEVNc1NVRkJTU3hGUVVGRkxFZEJRVWNzU1VGQlNTeERRVUZETEZGQlFWRXNSVUZCUlN4RFFVRkRPMUZCUTNwQ0xFVkJRVVVzUTBGQlF5eFBRVUZQTEVOQlFVTXNWVUZCUVN4RFFVRkRPMWxCUTFJc1NVRkJTU3hEUVVGRExFZEJRVWNzUzBGQlNTeERRVUZETEU5QlFVOHNRMEZCUXl4RFFVRkRMRU5CUVVNc1RVRkJUU3hEUVVGRExFTkJRVU03V1VGREwwSXNTVUZCU1N4RFFVRkRMRWRCUVVjc1MwRkJTU3hEUVVGRExFOUJRVThzUTBGQlF5eERRVUZETEVOQlFVTXNUVUZCVFN4RFFVRkRMRU5CUVVNN1dVRkRMMElzWlVGQlpTeERRVUZETEVsQlFVa3NRMEZCUXl4SlFVRkpMRk5CUVZNc1EwRkRPVUlzVDBGQlR5eERRVUZETEVOQlFVTXNSMEZCUnl4TFFVRkxMRmRCUVZjc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEUxQlFVMHNRMEZCUXl4RFFVRkRMRU5CUVVNc1RVRkJUU3hEUVVGRExFTkJRVU1zUTBGQlF5eEhRVUZITEVOQlFVTXNSVUZEZGtRc1QwRkJUeXhEUVVGRExFTkJRVU1zUjBGQlJ5eExRVUZMTEZkQlFWY3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFMUJRVTBzUTBGQlF5eERRVUZETEVOQlFVTXNUVUZCVFN4RFFVRkRMRU5CUVVNc1EwRkJReXhIUVVGSExFTkJRVU1zUlVGRGRrUXNRMEZCUXl4RFFVRkRMRWxCUVVrc1EwRkRWQ3hEUVVGRExFTkJRVU03VVVGRFVDeERRVUZETEVOQlFVTXNRMEZCUXp0UlFVTklMRTlCUVU4c1RVRkJUU3hEUVVGRE8wbEJRMnhDTEVOQlFVTTdTVUZGUkN4blEwRkJVU3hIUVVGU08xRkJRMGtzU1VGQlNTeEZRVUZGTEVkQlFVY3NSVUZCUlN4RFFVRkRPMUZCUTFvc1lVRkJZU3hEUVVGRExGRkJRVkVzUTBGQlF5eEpRVUZKTEVOQlFVTXNTMEZCU3l4RFFVRkRMRU5CUVVNc1EwRkJReXhGUVVGRkxFVkJRVVVzUTBGQlF5eERRVUZETzFGQlF6RkRMRTlCUVU4c1JVRkJSU3hEUVVGRE8wbEJRMlFzUTBGQlF6dEpRVVZOTEhOQ1FVRlJMRWRCUVdZc1ZVRkJaMElzVDBGQmEwSXNSVUZCUlN4RlFVRmxPMUZCUXk5RExFOUJRVThzUTBGQlF5eE5RVUZOTEVOQlFVTXNWVUZCUVN4RFFVRkRPMWxCUTFvc1EwRkJReXhEUVVGRExGRkJRVkVzUTBGQlF5eEZRVUZGTEVOQlFVTXNRMEZCUXp0WlFVTm1MR0ZCUVdFc1EwRkJReXhSUVVGUkxFTkJRVU1zUTBGQlF5eERRVUZETEZGQlFWRXNSVUZCUlN4RlFVRkZMRU5CUVVNc1EwRkJRenRSUVVNelF5eERRVUZETEVOQlFVTXNRMEZCUXp0SlFVTlFMRU5CUVVNN1NVRkRUQ3h2UWtGQlF6dEJRVUZFTEVOQlFVTXNRVUY0U2tRc1NVRjNTa003UVVGNFNsa3NjME5CUVdFN1FVRXdTakZDTEZOQlFWTXNVVUZCVVN4RFFVRkRMRTlCUVd0Q0xFVkJRVVVzUzBGQlN5eEZRVUZGTEUxQlFVMDdTVUZETDBNc1QwRkJUeXhEUVVGRExFMUJRVTBzUTBGQlF5eFZRVUZCTEVOQlFVTTdVVUZEV2l4SlFVRkpMRU5CUVVNc1EwRkJReXhOUVVGTkxFVkJRVVVzUlVGQlJUdFpRVU5hTEVsQlFVa3NRMEZCUXl4TFFVRkxMRU5CUVVNc1RVRkJUVHRuUWtGQlJTeExRVUZMTEVOQlFVTXNUVUZCVFN4SFFVRkhMRVZCUVVVc1EwRkJRenRaUVVOeVF5eExRVUZMTEVOQlFVTXNUVUZCVFN4RFFVRkRMRWxCUVVrc1EwRkJReXhEUVVGRExFTkJRVU1zUlVGQlJTeERRVUZETEVOQlFVTTdVMEZETTBJN1lVRkJUVHRaUVVOSUxFbEJRVWtzUTBGQlF5eEhRVUZITEV0QlFVc3NRMEZCUXp0WlFVTmtMRU5CUVVNc1EwRkJReXhIUVVGSExFZEJRVWNzVFVGQlRTeERRVUZETEUxQlFVMHNRMEZCUXp0WlFVTjBRaXhKUVVGSkxFTkJRVU1zUTBGQlF5eERRVUZETEZGQlFWRXNSVUZCUlN4SlFVRkpMRU5CUVVNc1EwRkJReXhaUVVGWkxFVkJRVVVzUlVGQlJUdG5Ra0ZEYmtNc1EwRkJReXhIUVVGSExFVkJRVVVzUlVGQlJTeEZRVUZGTEVOQlFVTXNRMEZCUXl4SFFVRkhMRVZCUVVVc1EwRkJRenRuUWtGRGJFSXNTVUZCU1N4RFFVRkRMRU5CUVVNc1dVRkJXU3hGUVVGRk8yOUNRVVZvUWl4TFFVRkxMRWxCUVVrc1NVRkJTU3hKUVVGSkxFTkJRVU1zUTBGQlF5eFZRVUZWTzNkQ1FVTjZRaXhEUVVGRExFTkJRVU1zU1VGQlNTeERRVUZETEVkQlFVY3NRMEZCUXl4RFFVRkRMRlZCUVZVc1EwRkJReXhKUVVGSkxFTkJRVU1zUTBGQlF6dG5Ra0ZEY2tNc1NVRkJTU3hEUVVGRExFdEJRVXNzUTBGQlF5eE5RVUZOTzI5Q1FVRkZMRXRCUVVzc1EwRkJReXhOUVVGTkxFZEJRVWNzUlVGQlJTeERRVUZETzJkQ1FVTnlReXhMUVVGTExFTkJRVU1zVFVGQlRTeERRVUZETEVsQlFVa3NRMEZCUXl4RFFVRkRMRU5CUVVNc1IwRkJSeXhEUVVGRExFTkJRVU03WjBKQlEzcENMRTFCUVUwc1EwRkJReXhKUVVGSkxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTTdZVUZEYkVJN1dVRkRSQ3hSUVVGUkxFTkJRVU1zUTBGQlF5eERRVUZETEZGQlFWRXNSVUZCUlN4RFFVRkRMRVZCUVVVc1RVRkJUU3hEUVVGRExFTkJRVU03VTBGRGJrTTdTVUZEVEN4RFFVRkRMRU5CUVVNc1EwRkJRenRCUVVOUUxFTkJRVU03UVVGRlJEdEpRVWRKTEdkQ1FVTlhMRVZCUVZVc1JVRkRWaXhSUVVGdFF5eEZRVU51UXl4UlFVRnRReXhGUVVOdVF5eFJRVUZ4UXl4RlFVTnlReXhWUVVGblFqdFJRVWhvUWl4NVFrRkJRU3hGUVVGQkxHVkJRWGxDTEZGQlFWRXNSVUZCUlR0UlFVTnVReXg1UWtGQlFTeEZRVUZCTEdWQlFYbENMRkZCUVZFc1JVRkJSVHRSUVVOdVF5eDVRa0ZCUVN4RlFVRkJMR1ZCUVRCQ0xGTkJRVk1zUlVGQlJUdFJRVWh5UXl4UFFVRkZMRWRCUVVZc1JVRkJSU3hEUVVGUk8xRkJRMVlzWVVGQlVTeEhRVUZTTEZGQlFWRXNRMEZCTWtJN1VVRkRia01zWVVGQlVTeEhRVUZTTEZGQlFWRXNRMEZCTWtJN1VVRkRia01zWVVGQlVTeEhRVUZTTEZGQlFWRXNRMEZCTmtJN1VVRkRja01zWlVGQlZTeEhRVUZXTEZWQlFWVXNRMEZCVFR0SlFVRkpMRU5CUVVNN1NVRkZhRU1zZVVKQlFWRXNSMEZCVWl4VlFVRlRMRVZCUVdVN1VVRkJlRUlzYVVKQlRVTTdVVUZNUnl4SlFVRkpMRU5CUVVNc1VVRkJVU3hEUVVGRExFMUJRVTBzUTBGQlF5eFZRVUZETEVWQlFVVXNSVUZCUlN4UlFVRlJPMWxCUXpsQ0xFVkJRVVVzUTBGQlF5eE5RVUZOTEVOQlFVTXNWVUZCUVN4TlFVRk5PMmRDUVVOYUxFVkJRVVVzUTBGQlF5eEpRVUZKTEVOQlFVTXNTVUZCU1N4VFFVRlRMRU5CUVVNc1MwRkJTU3hEUVVGRExFVkJRVVVzUlVGQlJTeE5RVUZOTEVOQlFVTXNSVUZCUlN4RlFVRkZMRkZCUVZFc1EwRkJReXhEUVVGRExFTkJRVU03V1VGRGVrUXNRMEZCUXl4RFFVRkRMRU5CUVVNN1VVRkRVQ3hEUVVGRExFTkJRVU1zUTBGQlF6dEpRVU5RTEVOQlFVTTdTVUZGUkN4MVFrRkJUU3hIUVVGT08xRkJRMGtzVDBGQlR5eEpRVUZKTEVOQlFVTXNVVUZCVVN4RFFVRkRMRXRCUVVzc1JVRkJSU3hMUVVGTExFTkJRVU1zUTBGQlF6dEpRVU4yUXl4RFFVRkRPMGxCUlVRc2VVSkJRVkVzUjBGQlVqdFJRVU5KTEU5QlFVOHNTVUZCU1N4RFFVRkRMRkZCUVZFc1EwRkJReXhMUVVGTExFVkJRVVVzUzBGQlN5eERRVUZETEVsQlFVa3NTVUZCU1N4RFFVRkRMRkZCUVZFc1EwRkJReXhMUVVGTExFVkJRVVVzUzBGQlN5eERRVUZETEVOQlFVTTdTVUZEZEVVc1EwRkJRenRKUVVWRUxEWkNRVUZaTEVkQlFWbzdVVUZEU1N4UFFVRlBMRTlCUVU4c1NVRkJTU3hEUVVGRExGVkJRVlVzUzBGQlN5eFhRVUZYTEVOQlFVTTdTVUZEYkVRc1EwRkJRenRKUVVOTUxHRkJRVU03UVVGQlJDeERRVUZETEVGQk4wSkVMRWxCTmtKRE8wRkJOMEpaTEhkQ1FVRk5PMEZCSzBKdVFpeFRRVUZUTEZsQlFWa3NRMEZCUXl4RFFVRk5MRVZCUVVVc1EwRkJUVHRKUVVOb1F5eEpRVUZKTEVOQlFVTXNSMEZCUnl4RlFVRkZMRU5CUVVNN1NVRkRXQ3hMUVVGTExFbEJRVWtzUTBGQlF5eEpRVUZKTEVOQlFVTTdVVUZCUlN4SlFVRkpMRU5CUVVNc1NVRkJTU3hEUVVGRE8xbEJRVVVzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4SFFVRkhMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dEpRVU42UXl4UFFVRlBMRU5CUVVNc1EwRkJRenRCUVVOaUxFTkJRVU03UVVGRlJEdEpRVUZCTzFGQlEwa3NWVUZCU3l4SFFVRlJMRVZCUVVVc1EwRkJRenRKUVd0RGNFSXNRMEZCUXp0SlFXcERSeXg1UWtGQlN5eEhRVUZNTzFGQlEwa3NUMEZCVHl4TlFVRk5MRU5CUVVNc1NVRkJTU3hEUVVGRExFbEJRVWtzUTBGQlF5eExRVUZMTEVOQlFVTXNRMEZCUXl4TlFVRk5MRU5CUVVNN1NVRkRNVU1zUTBGQlF6dEpRVU5FTEdkRFFVRlpMRWRCUVZvc1ZVRkJZU3hMUVVGblFqdFJRVU42UWl4SlFVRkpMRTFCUVUwc1IwRkJSeXhKUVVGSkxGTkJRVk1zUlVGQlJTeERRVUZETzFGQlF6ZENMRTFCUVUwc1EwRkJReXhMUVVGTExFZEJRVWNzV1VGQldTeERRVUZETEVsQlFVa3NRMEZCUXl4TFFVRkxMRVZCUVVVc1MwRkJTeXhEUVVGRExFdEJRVXNzUTBGQlF5eERRVUZETzFGQlEzSkVMRTlCUVU4c1RVRkJUU3hEUVVGRE8wbEJRMnhDTEVOQlFVTTdTVUZEUkN4eFEwRkJhVUlzUjBGQmFrSXNWVUZCYTBJc1MwRkJaMEk3VVVGRE9VSXNUMEZCVHl4SlFVRkpMRU5CUVVNc1dVRkJXU3hEUVVGRExFdEJRVXNzUTBGQlF5eERRVUZETEV0QlFVc3NSVUZCUlN4RFFVRkRPMGxCUXpWRExFTkJRVU03U1VGRFJDdzBRa0ZCVVN4SFFVRlNMRlZCUVZNc1JVRkJWVHRSUVVObUxFOUJRVThzUlVGQlJTeEpRVUZKTEVsQlFVa3NRMEZCUXl4TFFVRkxMRU5CUVVNN1NVRkROVUlzUTBGQlF6dEpRVU5FTEhWQ1FVRkhMRWRCUVVnc1ZVRkJTU3hEUVVGVE8xRkJRMVFzU1VGQlNTeERRVUZETEV0QlFVc3NRMEZCUXl4RFFVRkRMRU5CUVVNc1JVRkJSU3hEUVVGRExFZEJRVWNzUTBGQlF5eERRVUZETzBsQlEzcENMRU5CUVVNN1NVRkRSQ3d3UWtGQlRTeEhRVUZPTEZWQlFVOHNRMEZCVXp0UlFVTmFMRTlCUVU4c1NVRkJTU3hEUVVGRExFdEJRVXNzUTBGQlF5eERRVUZETEVOQlFVTXNSVUZCUlN4RFFVRkRMRU5CUVVNN1NVRkROVUlzUTBGQlF6dEpRVU5FTERCQ1FVRk5MRWRCUVU0c1ZVRkJUeXhEUVVGelFqdFJRVU42UWl4TFFVRkxMRWxCUVVrc1IwRkJSeXhKUVVGSkxFbEJRVWtzUTBGQlF5eExRVUZMTEVWQlFVVTdXVUZEZUVJc1EwRkJReXhEUVVGRExFbEJRVWtzUTBGQlF5eExRVUZMTEVOQlFVTXNSMEZCUnl4RFFVRkRMRU5CUVVNc1EwRkJRenRUUVVOMFFqdEpRVU5NTEVOQlFVTTdTVUZEUkN3eVFrRkJUeXhIUVVGUU8xRkJRMGtzU1VGQlNTeEZRVUZGTEVkQlFVY3NSVUZCUlN4RFFVRkRPMUZCUTFvc1NVRkJTU3hEUVVGRExFMUJRVTBzUTBGQlF5eFZRVUZCTEVOQlFVTTdXVUZEVkN4SlFVRkpMRU5CUVVNc1EwRkJReXhEUVVGRExGbEJRVmtzUlVGQlJUdG5Ra0ZEYWtJc1JVRkJSU3hEUVVGRExFbEJRVWtzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXp0UlFVTnVRaXhEUVVGRExFTkJRVU1zUTBGQlF6dFJRVU5JTEU5QlFVOHNSVUZCUlN4RFFVRkRPMGxCUTJRc1EwRkJRenRKUVVOTUxHZENRVUZETzBGQlFVUXNRMEZCUXl4QlFXNURSQ3hKUVcxRFF6dEJRVzVEV1N3NFFrRkJVenRCUVhGRGRFSTdTVUZCUVR0UlFVTkpMRk5CUVVrc1IwRkJVU3hGUVVGRkxFTkJRVU03VVVGRFppeE5RVUZETEVkQlFWY3NRMEZCUXl4RFFVRkRPMGxCWjBSc1FpeERRVUZETzBsQkwwTkhMSGRDUVVGTExFZEJRVXc3VVVGRFNTeFBRVUZQTEVsQlFVa3NRMEZCUXl4RFFVRkRMRU5CUVVNN1NVRkRiRUlzUTBGQlF6dEpRVU5FTERKQ1FVRlJMRWRCUVZJc1ZVRkJVeXhGUVVGVk8xRkJRMllzU1VGQlNTeE5RVUZOTEVkQlFVY3NTMEZCU3l4RFFVRkRPMUZCUTI1Q0xFbEJRVWtzUTBGQlF5eGhRVUZoTEVOQlFVTXNWVUZCUVN4RFFVRkRPMWxCUTJoQ0xFbEJRVWtzUTBGQlF5eE5RVUZOTEVsQlFVa3NRMEZCUXl4RFFVRkRMRVZCUVVVc1NVRkJTU3hGUVVGRkxFVkJRVVU3WjBKQlEzWkNMRTFCUVUwc1IwRkJSeXhKUVVGSkxFTkJRVU03WVVGRGFrSTdVVUZEVEN4RFFVRkRMRU5CUVVNc1EwRkJRenRSUVVOSUxFOUJRVThzVFVGQlRTeERRVUZETzBsQlEyeENMRU5CUVVNN1NVRkRSQ3h6UWtGQlJ5eEhRVUZJTEZWQlFVa3NVVUZCWjBJc1JVRkJSU3hEUVVGVE8xRkJRek5DTEVsQlFVa3NRMEZCUXl4SFFVRmpMRkZCUVZFc1NVRkJTU3hKUVVGSkxFTkJRVU1zU1VGQlNTeERRVUZETEVOQlFVTXNRMEZCUXl4SlFVRkpMRU5CUVVNc1NVRkJTU3hEUVVGRExGRkJRVkVzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4SlFVRkpMRU5CUVVNc1NVRkJTU3hEUVVGRExGRkJRVkVzUTBGQlF5eEhRVUZITEVsQlFVa3NVMEZCVXl4RlFVRkZMRU5CUVVNN1VVRkRka2NzUTBGQlF5eERRVUZETEVkQlFVY3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJRenRSUVVOVUxFVkJRVVVzU1VGQlNTeERRVUZETEVOQlFVTXNRMEZCUXp0SlFVTmlMRU5CUVVNN1NVRkRSQ3g1UWtGQlRTeEhRVUZPTEZWQlFVOHNVVUZCWjBJc1JVRkJSU3hEUVVGVE8xRkJRemxDTEVsQlFVa3NSVUZCUlN4SFFVRmpMRWxCUVVrc1EwRkJReXhKUVVGSkxFTkJRVU1zVVVGQlVTeERRVUZETEVOQlFVTTdVVUZEZUVNc1JVRkJSU3hEUVVGRExFMUJRVTBzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXp0UlFVTmlMRWxCUVVrc1JVRkJSU3hEUVVGRExFdEJRVXNzUlVGQlJTeExRVUZMTEVOQlFVTXNSVUZCUlR0WlFVTnNRaXhQUVVGUExFbEJRVWtzUTBGQlF5eEpRVUZKTEVOQlFVTXNVVUZCVVN4RFFVRkRMRU5CUVVNN1UwRkRPVUk3VVVGRFJDeEZRVUZGTEVsQlFVa3NRMEZCUXl4RFFVRkRMRU5CUVVNN1NVRkRZaXhEUVVGRE8wbEJRMFFzZVVKQlFVMHNSMEZCVGl4VlFVRlBMRU5CUVRSRE8xRkJReTlETEV0QlFVc3NTVUZCU1N4UlFVRlJMRWxCUVVrc1NVRkJTU3hEUVVGRExFbEJRVWtzUlVGQlJUdFpRVU0xUWl4RFFVRkRMRU5CUVZrc1NVRkJTU3hEUVVGRExFbEJRVWtzUTBGQlF5eFJRVUZSTEVOQlFVTXNSVUZCUlN4TlFVRk5MRU5CUVVNc1VVRkJVU3hEUVVGRExFTkJRVU1zUTBGQlF6dFRRVU4yUkR0SlFVTk1MRU5CUVVNN1NVRkRSQ3huUTBGQllTeEhRVUZpTEZWQlFXTXNRMEZCYzBJN1VVRkRhRU1zU1VGQlNTeERRVUZETEUxQlFVMHNRMEZCUXl4VlFVRkRMRVZCUVVVc1JVRkJSU3hGUVVGRkxFbEJRVXNzVDBGQlFTeEZRVUZGTEVOQlFVTXNUVUZCVFN4RFFVRkRMRU5CUVVNc1EwRkJReXhGUVVGYUxFTkJRVmtzUTBGQlF5eERRVUZETzBsQlF6RkRMRU5CUVVNN1NVRkRSQ3dyUWtGQldTeEhRVUZhTEZWQlFXRXNTMEZCWlR0UlFVTjRRaXhKUVVGSkxFMUJRVTBzUjBGQllTeEpRVUZKTEZGQlFWRXNSVUZCUlN4RFFVRkRPMUZCUTNSRExFbEJRVWtzUTBGQlF5eE5RVUZOTEVOQlFVTXNWVUZCUXl4RlFVRkZMRVZCUVVVc1JVRkJSVHRaUVVObUxFbEJRVWtzUlVGQlJTeEpRVUZKTEV0QlFVc3NRMEZCUXl4SlFVRkpMRVZCUVVVN1owSkJRMnhDTEVsQlFVa3NRMEZCUXl4SFFVRkhMRVZCUVVVc1EwRkJReXhaUVVGWkxFTkJRVU1zUzBGQlN5eERRVUZETEVsQlFVa3NRMEZCUXl4RlFVRkZMRU5CUVVNc1EwRkJReXhGUVVOdVF5eERRVUZETEVkQlFVY3NRMEZCUXl4RFFVRkRMRXRCUVVzc1JVRkJSU3hEUVVGRE8yZENRVU5zUWl4SlFVRkpMRU5CUVVNc1IwRkJSeXhEUVVGRExFVkJRVVU3YjBKQlExQXNUVUZCVFN4RFFVRkRMRWxCUVVrc1EwRkJReXhGUVVGRkxFTkJRVU1zUjBGQlJ5eERRVUZETEVOQlFVTTdiMEpCUTNCQ0xFMUJRVTBzUTBGQlF5eERRVUZETEVsQlFVa3NRMEZCUXl4RFFVRkRPMmxDUVVOcVFqdGhRVU5LTzFGQlEwd3NRMEZCUXl4RFFVRkRMRU5CUVVNN1VVRkRTQ3hQUVVGUExFMUJRVTBzUTBGQlF6dEpRVU5zUWl4RFFVRkRPMGxCUTB3c1pVRkJRenRCUVVGRUxFTkJRVU1zUVVGc1JFUXNTVUZyUkVNN1FVRnNSRmtzTkVKQlFWRTdRVUZ2UkhKQ0xGTkJRVk1zYVVKQlFXbENMRU5CUVVNc1EwRkJUU3hGUVVGRkxFTkJRVTA3U1VGRGNrTXNUMEZCVHl4TlFVRk5MRU5CUVVNc1NVRkJTU3hEUVVGRExGbEJRVmtzUTBGQlF5eERRVUZETEVWQlFVVXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhOUVVGTkxFTkJRVUU3UVVGRGFrUXNRMEZCUXp0QlFVVkVMRk5CUVdkQ0xGTkJRVk1zUTBGQlR5eExRVUZaTEVWQlFVVXNTMEZCWVN4RlFVRkZMRVZCUVRCQ0xFVkJRVVVzVTBGQmFVSTdTVUZEZEVjc1NVRkJTU3hEUVVGRExFZEJRVWNzUzBGQlN5eERRVUZETEUxQlFVMHNSVUZEYUVJc1EwRkJReXhIUVVGSExFbEJRVWtzWVVGQllTeERRVUZETEVOQlFVTXNSVUZCUlN4TFFVRkxMRVZCUVVVc1JVRkJSU3hGUVVGRkxGTkJRVk1zUTBGQlF5eERRVUZETzBsQlEyNUVMRTlCUVU4c1EwRkJReXhEUVVGRExGZEJRVmNzUlVGQlJUdFJRVUZETEVOQlFVTTdTVUZEZUVJc1NVRkJTU3hWUVVGVkxFZEJRV2RDTEVWQlFVVXNRMEZCUXp0SlFVTnFReXhKUVVGSkxFTkJRVU1zUjBGQlJ5eERRVUZETEVOQlFVTXNhVUpCUVdsQ0xFTkJRVU1zVlVGQlZTeERRVUZETEVOQlFVTTdTVUZEZUVNc1ZVRkJWU3hEUVVGRExFOUJRVThzUTBGQlF5eFZRVUZWTEVOQlFVTTdVVUZETVVJc1NVRkJTU3hEUVVGRExFZEJRVWNzVlVGQlF5eEhRVUZITzFsQlExSXNTVUZCU1N4RFFVRkRMRWRCUVVjc1EwRkJReXhEUVVGRExFZEJRVWNzUTBGQlF5eERRVUZETzFsQlEyWXNTVUZCU1N4UFFVRlBMRU5CUVVNc1NVRkJTU3hSUVVGUk8yZENRVUZGTEVOQlFVTXNRMEZCUXl4SFFVRkhMRU5CUVVNc1IwRkJSeXhMUVVGTExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTTdVVUZEYUVRc1EwRkJReXhEUVVGRE8xRkJRMFlzUTBGQlF5eERRVUZETEZGQlFWRXNRMEZCUXl4RFFVRkRPMUZCUTFvc1EwRkJReXhEUVVGRExGRkJRVkVzUTBGQlF5eERRVUZETzBsQlEyaENMRU5CUVVNc1EwRkJReXhEUVVGRE8wbEJRMGdzVDBGQlR5eEZRVUZGTEUxQlFVMHNSVUZCUlN4RFFVRkRMRVZCUVVVc1ZVRkJWU3hGUVVGRkxGVkJRVlVzUlVGQlJTeERRVUZETzBGQlEycEVMRU5CUVVNN1FVRm1SQ3c0UWtGbFF5SjkiLCJcInVzZSBzdHJpY3RcIjtcclxuT2JqZWN0LmRlZmluZVByb3BlcnR5KGV4cG9ydHMsIFwiX19lc01vZHVsZVwiLCB7IHZhbHVlOiB0cnVlIH0pO1xyXG52YXIgUGFpcmluZ0hlYXAgPSAoZnVuY3Rpb24gKCkge1xyXG4gICAgZnVuY3Rpb24gUGFpcmluZ0hlYXAoZWxlbSkge1xyXG4gICAgICAgIHRoaXMuZWxlbSA9IGVsZW07XHJcbiAgICAgICAgdGhpcy5zdWJoZWFwcyA9IFtdO1xyXG4gICAgfVxyXG4gICAgUGFpcmluZ0hlYXAucHJvdG90eXBlLnRvU3RyaW5nID0gZnVuY3Rpb24gKHNlbGVjdG9yKSB7XHJcbiAgICAgICAgdmFyIHN0ciA9IFwiXCIsIG5lZWRDb21tYSA9IGZhbHNlO1xyXG4gICAgICAgIGZvciAodmFyIGkgPSAwOyBpIDwgdGhpcy5zdWJoZWFwcy5sZW5ndGg7ICsraSkge1xyXG4gICAgICAgICAgICB2YXIgc3ViaGVhcCA9IHRoaXMuc3ViaGVhcHNbaV07XHJcbiAgICAgICAgICAgIGlmICghc3ViaGVhcC5lbGVtKSB7XHJcbiAgICAgICAgICAgICAgICBuZWVkQ29tbWEgPSBmYWxzZTtcclxuICAgICAgICAgICAgICAgIGNvbnRpbnVlO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIGlmIChuZWVkQ29tbWEpIHtcclxuICAgICAgICAgICAgICAgIHN0ciA9IHN0ciArIFwiLFwiO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIHN0ciA9IHN0ciArIHN1YmhlYXAudG9TdHJpbmcoc2VsZWN0b3IpO1xyXG4gICAgICAgICAgICBuZWVkQ29tbWEgPSB0cnVlO1xyXG4gICAgICAgIH1cclxuICAgICAgICBpZiAoc3RyICE9PSBcIlwiKSB7XHJcbiAgICAgICAgICAgIHN0ciA9IFwiKFwiICsgc3RyICsgXCIpXCI7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHJldHVybiAodGhpcy5lbGVtID8gc2VsZWN0b3IodGhpcy5lbGVtKSA6IFwiXCIpICsgc3RyO1xyXG4gICAgfTtcclxuICAgIFBhaXJpbmdIZWFwLnByb3RvdHlwZS5mb3JFYWNoID0gZnVuY3Rpb24gKGYpIHtcclxuICAgICAgICBpZiAoIXRoaXMuZW1wdHkoKSkge1xyXG4gICAgICAgICAgICBmKHRoaXMuZWxlbSwgdGhpcyk7XHJcbiAgICAgICAgICAgIHRoaXMuc3ViaGVhcHMuZm9yRWFjaChmdW5jdGlvbiAocykgeyByZXR1cm4gcy5mb3JFYWNoKGYpOyB9KTtcclxuICAgICAgICB9XHJcbiAgICB9O1xyXG4gICAgUGFpcmluZ0hlYXAucHJvdG90eXBlLmNvdW50ID0gZnVuY3Rpb24gKCkge1xyXG4gICAgICAgIHJldHVybiB0aGlzLmVtcHR5KCkgPyAwIDogMSArIHRoaXMuc3ViaGVhcHMucmVkdWNlKGZ1bmN0aW9uIChuLCBoKSB7XHJcbiAgICAgICAgICAgIHJldHVybiBuICsgaC5jb3VudCgpO1xyXG4gICAgICAgIH0sIDApO1xyXG4gICAgfTtcclxuICAgIFBhaXJpbmdIZWFwLnByb3RvdHlwZS5taW4gPSBmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgcmV0dXJuIHRoaXMuZWxlbTtcclxuICAgIH07XHJcbiAgICBQYWlyaW5nSGVhcC5wcm90b3R5cGUuZW1wdHkgPSBmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgcmV0dXJuIHRoaXMuZWxlbSA9PSBudWxsO1xyXG4gICAgfTtcclxuICAgIFBhaXJpbmdIZWFwLnByb3RvdHlwZS5jb250YWlucyA9IGZ1bmN0aW9uIChoKSB7XHJcbiAgICAgICAgaWYgKHRoaXMgPT09IGgpXHJcbiAgICAgICAgICAgIHJldHVybiB0cnVlO1xyXG4gICAgICAgIGZvciAodmFyIGkgPSAwOyBpIDwgdGhpcy5zdWJoZWFwcy5sZW5ndGg7IGkrKykge1xyXG4gICAgICAgICAgICBpZiAodGhpcy5zdWJoZWFwc1tpXS5jb250YWlucyhoKSlcclxuICAgICAgICAgICAgICAgIHJldHVybiB0cnVlO1xyXG4gICAgICAgIH1cclxuICAgICAgICByZXR1cm4gZmFsc2U7XHJcbiAgICB9O1xyXG4gICAgUGFpcmluZ0hlYXAucHJvdG90eXBlLmlzSGVhcCA9IGZ1bmN0aW9uIChsZXNzVGhhbikge1xyXG4gICAgICAgIHZhciBfdGhpcyA9IHRoaXM7XHJcbiAgICAgICAgcmV0dXJuIHRoaXMuc3ViaGVhcHMuZXZlcnkoZnVuY3Rpb24gKGgpIHsgcmV0dXJuIGxlc3NUaGFuKF90aGlzLmVsZW0sIGguZWxlbSkgJiYgaC5pc0hlYXAobGVzc1RoYW4pOyB9KTtcclxuICAgIH07XHJcbiAgICBQYWlyaW5nSGVhcC5wcm90b3R5cGUuaW5zZXJ0ID0gZnVuY3Rpb24gKG9iaiwgbGVzc1RoYW4pIHtcclxuICAgICAgICByZXR1cm4gdGhpcy5tZXJnZShuZXcgUGFpcmluZ0hlYXAob2JqKSwgbGVzc1RoYW4pO1xyXG4gICAgfTtcclxuICAgIFBhaXJpbmdIZWFwLnByb3RvdHlwZS5tZXJnZSA9IGZ1bmN0aW9uIChoZWFwMiwgbGVzc1RoYW4pIHtcclxuICAgICAgICBpZiAodGhpcy5lbXB0eSgpKVxyXG4gICAgICAgICAgICByZXR1cm4gaGVhcDI7XHJcbiAgICAgICAgZWxzZSBpZiAoaGVhcDIuZW1wdHkoKSlcclxuICAgICAgICAgICAgcmV0dXJuIHRoaXM7XHJcbiAgICAgICAgZWxzZSBpZiAobGVzc1RoYW4odGhpcy5lbGVtLCBoZWFwMi5lbGVtKSkge1xyXG4gICAgICAgICAgICB0aGlzLnN1YmhlYXBzLnB1c2goaGVhcDIpO1xyXG4gICAgICAgICAgICByZXR1cm4gdGhpcztcclxuICAgICAgICB9XHJcbiAgICAgICAgZWxzZSB7XHJcbiAgICAgICAgICAgIGhlYXAyLnN1YmhlYXBzLnB1c2godGhpcyk7XHJcbiAgICAgICAgICAgIHJldHVybiBoZWFwMjtcclxuICAgICAgICB9XHJcbiAgICB9O1xyXG4gICAgUGFpcmluZ0hlYXAucHJvdG90eXBlLnJlbW92ZU1pbiA9IGZ1bmN0aW9uIChsZXNzVGhhbikge1xyXG4gICAgICAgIGlmICh0aGlzLmVtcHR5KCkpXHJcbiAgICAgICAgICAgIHJldHVybiBudWxsO1xyXG4gICAgICAgIGVsc2VcclxuICAgICAgICAgICAgcmV0dXJuIHRoaXMubWVyZ2VQYWlycyhsZXNzVGhhbik7XHJcbiAgICB9O1xyXG4gICAgUGFpcmluZ0hlYXAucHJvdG90eXBlLm1lcmdlUGFpcnMgPSBmdW5jdGlvbiAobGVzc1RoYW4pIHtcclxuICAgICAgICBpZiAodGhpcy5zdWJoZWFwcy5sZW5ndGggPT0gMClcclxuICAgICAgICAgICAgcmV0dXJuIG5ldyBQYWlyaW5nSGVhcChudWxsKTtcclxuICAgICAgICBlbHNlIGlmICh0aGlzLnN1YmhlYXBzLmxlbmd0aCA9PSAxKSB7XHJcbiAgICAgICAgICAgIHJldHVybiB0aGlzLnN1YmhlYXBzWzBdO1xyXG4gICAgICAgIH1cclxuICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgdmFyIGZpcnN0UGFpciA9IHRoaXMuc3ViaGVhcHMucG9wKCkubWVyZ2UodGhpcy5zdWJoZWFwcy5wb3AoKSwgbGVzc1RoYW4pO1xyXG4gICAgICAgICAgICB2YXIgcmVtYWluaW5nID0gdGhpcy5tZXJnZVBhaXJzKGxlc3NUaGFuKTtcclxuICAgICAgICAgICAgcmV0dXJuIGZpcnN0UGFpci5tZXJnZShyZW1haW5pbmcsIGxlc3NUaGFuKTtcclxuICAgICAgICB9XHJcbiAgICB9O1xyXG4gICAgUGFpcmluZ0hlYXAucHJvdG90eXBlLmRlY3JlYXNlS2V5ID0gZnVuY3Rpb24gKHN1YmhlYXAsIG5ld1ZhbHVlLCBzZXRIZWFwTm9kZSwgbGVzc1RoYW4pIHtcclxuICAgICAgICB2YXIgbmV3SGVhcCA9IHN1YmhlYXAucmVtb3ZlTWluKGxlc3NUaGFuKTtcclxuICAgICAgICBzdWJoZWFwLmVsZW0gPSBuZXdIZWFwLmVsZW07XHJcbiAgICAgICAgc3ViaGVhcC5zdWJoZWFwcyA9IG5ld0hlYXAuc3ViaGVhcHM7XHJcbiAgICAgICAgaWYgKHNldEhlYXBOb2RlICE9PSBudWxsICYmIG5ld0hlYXAuZWxlbSAhPT0gbnVsbCkge1xyXG4gICAgICAgICAgICBzZXRIZWFwTm9kZShzdWJoZWFwLmVsZW0sIHN1YmhlYXApO1xyXG4gICAgICAgIH1cclxuICAgICAgICB2YXIgcGFpcmluZ05vZGUgPSBuZXcgUGFpcmluZ0hlYXAobmV3VmFsdWUpO1xyXG4gICAgICAgIGlmIChzZXRIZWFwTm9kZSAhPT0gbnVsbCkge1xyXG4gICAgICAgICAgICBzZXRIZWFwTm9kZShuZXdWYWx1ZSwgcGFpcmluZ05vZGUpO1xyXG4gICAgICAgIH1cclxuICAgICAgICByZXR1cm4gdGhpcy5tZXJnZShwYWlyaW5nTm9kZSwgbGVzc1RoYW4pO1xyXG4gICAgfTtcclxuICAgIHJldHVybiBQYWlyaW5nSGVhcDtcclxufSgpKTtcclxuZXhwb3J0cy5QYWlyaW5nSGVhcCA9IFBhaXJpbmdIZWFwO1xyXG52YXIgUHJpb3JpdHlRdWV1ZSA9IChmdW5jdGlvbiAoKSB7XHJcbiAgICBmdW5jdGlvbiBQcmlvcml0eVF1ZXVlKGxlc3NUaGFuKSB7XHJcbiAgICAgICAgdGhpcy5sZXNzVGhhbiA9IGxlc3NUaGFuO1xyXG4gICAgfVxyXG4gICAgUHJpb3JpdHlRdWV1ZS5wcm90b3R5cGUudG9wID0gZnVuY3Rpb24gKCkge1xyXG4gICAgICAgIGlmICh0aGlzLmVtcHR5KCkpIHtcclxuICAgICAgICAgICAgcmV0dXJuIG51bGw7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHJldHVybiB0aGlzLnJvb3QuZWxlbTtcclxuICAgIH07XHJcbiAgICBQcmlvcml0eVF1ZXVlLnByb3RvdHlwZS5wdXNoID0gZnVuY3Rpb24gKCkge1xyXG4gICAgICAgIHZhciBhcmdzID0gW107XHJcbiAgICAgICAgZm9yICh2YXIgX2kgPSAwOyBfaSA8IGFyZ3VtZW50cy5sZW5ndGg7IF9pKyspIHtcclxuICAgICAgICAgICAgYXJnc1tfaV0gPSBhcmd1bWVudHNbX2ldO1xyXG4gICAgICAgIH1cclxuICAgICAgICB2YXIgcGFpcmluZ05vZGU7XHJcbiAgICAgICAgZm9yICh2YXIgaSA9IDAsIGFyZzsgYXJnID0gYXJnc1tpXTsgKytpKSB7XHJcbiAgICAgICAgICAgIHBhaXJpbmdOb2RlID0gbmV3IFBhaXJpbmdIZWFwKGFyZyk7XHJcbiAgICAgICAgICAgIHRoaXMucm9vdCA9IHRoaXMuZW1wdHkoKSA/XHJcbiAgICAgICAgICAgICAgICBwYWlyaW5nTm9kZSA6IHRoaXMucm9vdC5tZXJnZShwYWlyaW5nTm9kZSwgdGhpcy5sZXNzVGhhbik7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHJldHVybiBwYWlyaW5nTm9kZTtcclxuICAgIH07XHJcbiAgICBQcmlvcml0eVF1ZXVlLnByb3RvdHlwZS5lbXB0eSA9IGZ1bmN0aW9uICgpIHtcclxuICAgICAgICByZXR1cm4gIXRoaXMucm9vdCB8fCAhdGhpcy5yb290LmVsZW07XHJcbiAgICB9O1xyXG4gICAgUHJpb3JpdHlRdWV1ZS5wcm90b3R5cGUuaXNIZWFwID0gZnVuY3Rpb24gKCkge1xyXG4gICAgICAgIHJldHVybiB0aGlzLnJvb3QuaXNIZWFwKHRoaXMubGVzc1RoYW4pO1xyXG4gICAgfTtcclxuICAgIFByaW9yaXR5UXVldWUucHJvdG90eXBlLmZvckVhY2ggPSBmdW5jdGlvbiAoZikge1xyXG4gICAgICAgIHRoaXMucm9vdC5mb3JFYWNoKGYpO1xyXG4gICAgfTtcclxuICAgIFByaW9yaXR5UXVldWUucHJvdG90eXBlLnBvcCA9IGZ1bmN0aW9uICgpIHtcclxuICAgICAgICBpZiAodGhpcy5lbXB0eSgpKSB7XHJcbiAgICAgICAgICAgIHJldHVybiBudWxsO1xyXG4gICAgICAgIH1cclxuICAgICAgICB2YXIgb2JqID0gdGhpcy5yb290Lm1pbigpO1xyXG4gICAgICAgIHRoaXMucm9vdCA9IHRoaXMucm9vdC5yZW1vdmVNaW4odGhpcy5sZXNzVGhhbik7XHJcbiAgICAgICAgcmV0dXJuIG9iajtcclxuICAgIH07XHJcbiAgICBQcmlvcml0eVF1ZXVlLnByb3RvdHlwZS5yZWR1Y2VLZXkgPSBmdW5jdGlvbiAoaGVhcE5vZGUsIG5ld0tleSwgc2V0SGVhcE5vZGUpIHtcclxuICAgICAgICBpZiAoc2V0SGVhcE5vZGUgPT09IHZvaWQgMCkgeyBzZXRIZWFwTm9kZSA9IG51bGw7IH1cclxuICAgICAgICB0aGlzLnJvb3QgPSB0aGlzLnJvb3QuZGVjcmVhc2VLZXkoaGVhcE5vZGUsIG5ld0tleSwgc2V0SGVhcE5vZGUsIHRoaXMubGVzc1RoYW4pO1xyXG4gICAgfTtcclxuICAgIFByaW9yaXR5UXVldWUucHJvdG90eXBlLnRvU3RyaW5nID0gZnVuY3Rpb24gKHNlbGVjdG9yKSB7XHJcbiAgICAgICAgcmV0dXJuIHRoaXMucm9vdC50b1N0cmluZyhzZWxlY3Rvcik7XHJcbiAgICB9O1xyXG4gICAgUHJpb3JpdHlRdWV1ZS5wcm90b3R5cGUuY291bnQgPSBmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgcmV0dXJuIHRoaXMucm9vdC5jb3VudCgpO1xyXG4gICAgfTtcclxuICAgIHJldHVybiBQcmlvcml0eVF1ZXVlO1xyXG59KCkpO1xyXG5leHBvcnRzLlByaW9yaXR5UXVldWUgPSBQcmlvcml0eVF1ZXVlO1xyXG4vLyMgc291cmNlTWFwcGluZ1VSTD1kYXRhOmFwcGxpY2F0aW9uL2pzb247YmFzZTY0LGV5SjJaWEp6YVc5dUlqb3pMQ0ptYVd4bElqb2ljSEYxWlhWbExtcHpJaXdpYzI5MWNtTmxVbTl2ZENJNklpSXNJbk52ZFhKalpYTWlPbHNpTGk0dkxpNHZWMlZpUTI5c1lTOXpjbU12Y0hGMVpYVmxMblJ6SWwwc0ltNWhiV1Z6SWpwYlhTd2liV0Z3Y0dsdVozTWlPaUk3TzBGQlEwRTdTVUZKU1N4eFFrRkJiVUlzU1VGQlR6dFJRVUZRTEZOQlFVa3NSMEZCU2l4SlFVRkpMRU5CUVVjN1VVRkRkRUlzU1VGQlNTeERRVUZETEZGQlFWRXNSMEZCUnl4RlFVRkZMRU5CUVVNN1NVRkRka0lzUTBGQlF6dEpRVVZOTERoQ1FVRlJMRWRCUVdZc1ZVRkJaMElzVVVGQlVUdFJRVU53UWl4SlFVRkpMRWRCUVVjc1IwRkJSeXhGUVVGRkxFVkJRVVVzVTBGQlV5eEhRVUZITEV0QlFVc3NRMEZCUXp0UlFVTm9ReXhMUVVGTExFbEJRVWtzUTBGQlF5eEhRVUZITEVOQlFVTXNSVUZCUlN4RFFVRkRMRWRCUVVjc1NVRkJTU3hEUVVGRExGRkJRVkVzUTBGQlF5eE5RVUZOTEVWQlFVVXNSVUZCUlN4RFFVRkRMRVZCUVVVN1dVRkRNME1zU1VGQlNTeFBRVUZQTEVkQlFXMUNMRWxCUVVrc1EwRkJReXhSUVVGUkxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTTdXVUZETDBNc1NVRkJTU3hEUVVGRExFOUJRVThzUTBGQlF5eEpRVUZKTEVWQlFVVTdaMEpCUTJZc1UwRkJVeXhIUVVGSExFdEJRVXNzUTBGQlF6dG5Ra0ZEYkVJc1UwRkJVenRoUVVOYU8xbEJRMFFzU1VGQlNTeFRRVUZUTEVWQlFVVTdaMEpCUTFnc1IwRkJSeXhIUVVGSExFZEJRVWNzUjBGQlJ5eEhRVUZITEVOQlFVTTdZVUZEYmtJN1dVRkRSQ3hIUVVGSExFZEJRVWNzUjBGQlJ5eEhRVUZITEU5QlFVOHNRMEZCUXl4UlFVRlJMRU5CUVVNc1VVRkJVU3hEUVVGRExFTkJRVU03V1VGRGRrTXNVMEZCVXl4SFFVRkhMRWxCUVVrc1EwRkJRenRUUVVOd1FqdFJRVU5FTEVsQlFVa3NSMEZCUnl4TFFVRkxMRVZCUVVVc1JVRkJSVHRaUVVOYUxFZEJRVWNzUjBGQlJ5eEhRVUZITEVkQlFVY3NSMEZCUnl4SFFVRkhMRWRCUVVjc1EwRkJRenRUUVVONlFqdFJRVU5FTEU5QlFVOHNRMEZCUXl4SlFVRkpMRU5CUVVNc1NVRkJTU3hEUVVGRExFTkJRVU1zUTBGQlF5eFJRVUZSTEVOQlFVTXNTVUZCU1N4RFFVRkRMRWxCUVVrc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eEZRVUZGTEVOQlFVTXNSMEZCUnl4SFFVRkhMRU5CUVVNN1NVRkRlRVFzUTBGQlF6dEpRVVZOTERaQ1FVRlBMRWRCUVdRc1ZVRkJaU3hEUVVGRE8xRkJRMW9zU1VGQlNTeERRVUZETEVsQlFVa3NRMEZCUXl4TFFVRkxMRVZCUVVVc1JVRkJSVHRaUVVObUxFTkJRVU1zUTBGQlF5eEpRVUZKTEVOQlFVTXNTVUZCU1N4RlFVRkZMRWxCUVVrc1EwRkJReXhEUVVGRE8xbEJRMjVDTEVsQlFVa3NRMEZCUXl4UlFVRlJMRU5CUVVNc1QwRkJUeXhEUVVGRExGVkJRVUVzUTBGQlF5eEpRVUZKTEU5QlFVRXNRMEZCUXl4RFFVRkRMRTlCUVU4c1EwRkJReXhEUVVGRExFTkJRVU1zUlVGQldpeERRVUZaTEVOQlFVTXNRMEZCUXp0VFFVTTFRenRKUVVOTUxFTkJRVU03U1VGRlRTd3lRa0ZCU3l4SFFVRmFPMUZCUTBrc1QwRkJUeXhKUVVGSkxFTkJRVU1zUzBGQlN5eEZRVUZGTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eEhRVUZITEVsQlFVa3NRMEZCUXl4UlFVRlJMRU5CUVVNc1RVRkJUU3hEUVVGRExGVkJRVU1zUTBGQlV5eEZRVUZGTEVOQlFXbENPMWxCUXpWRkxFOUJRVThzUTBGQlF5eEhRVUZITEVOQlFVTXNRMEZCUXl4TFFVRkxMRVZCUVVVc1EwRkJRenRSUVVONlFpeERRVUZETEVWQlFVVXNRMEZCUXl4RFFVRkRMRU5CUVVNN1NVRkRWaXhEUVVGRE8wbEJSVTBzZVVKQlFVY3NSMEZCVmp0UlFVTkpMRTlCUVU4c1NVRkJTU3hEUVVGRExFbEJRVWtzUTBGQlF6dEpRVU55UWl4RFFVRkRPMGxCUlUwc01rSkJRVXNzUjBGQldqdFJRVU5KTEU5QlFVOHNTVUZCU1N4RFFVRkRMRWxCUVVrc1NVRkJTU3hKUVVGSkxFTkJRVU03U1VGRE4wSXNRMEZCUXp0SlFVVk5MRGhDUVVGUkxFZEJRV1lzVlVGQlowSXNRMEZCYVVJN1VVRkROMElzU1VGQlNTeEpRVUZKTEV0QlFVc3NRMEZCUXp0WlFVRkZMRTlCUVU4c1NVRkJTU3hEUVVGRE8xRkJRelZDTEV0QlFVc3NTVUZCU1N4RFFVRkRMRWRCUVVjc1EwRkJReXhGUVVGRkxFTkJRVU1zUjBGQlJ5eEpRVUZKTEVOQlFVTXNVVUZCVVN4RFFVRkRMRTFCUVUwc1JVRkJSU3hEUVVGRExFVkJRVVVzUlVGQlJUdFpRVU16UXl4SlFVRkpMRWxCUVVrc1EwRkJReXhSUVVGUkxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNVVUZCVVN4RFFVRkRMRU5CUVVNc1EwRkJRenRuUWtGQlJTeFBRVUZQTEVsQlFVa3NRMEZCUXp0VFFVTnFSRHRSUVVORUxFOUJRVThzUzBGQlN5eERRVUZETzBsQlEycENMRU5CUVVNN1NVRkZUU3cwUWtGQlRTeEhRVUZpTEZWQlFXTXNVVUZCYVVNN1VVRkJMME1zYVVKQlJVTTdVVUZFUnl4UFFVRlBMRWxCUVVrc1EwRkJReXhSUVVGUkxFTkJRVU1zUzBGQlN5eERRVUZETEZWQlFVRXNRMEZCUXl4SlFVRkhMRTlCUVVFc1VVRkJVU3hEUVVGRExFdEJRVWtzUTBGQlF5eEpRVUZKTEVWQlFVVXNRMEZCUXl4RFFVRkRMRWxCUVVrc1EwRkJReXhKUVVGSkxFTkJRVU1zUTBGQlF5eE5RVUZOTEVOQlFVTXNVVUZCVVN4RFFVRkRMRVZCUVdwRUxFTkJRV2xFTEVOQlFVTXNRMEZCUXp0SlFVTjBSaXhEUVVGRE8wbEJSVTBzTkVKQlFVMHNSMEZCWWl4VlFVRmpMRWRCUVU4c1JVRkJSU3hSUVVGUk8xRkJRek5DTEU5QlFVOHNTVUZCU1N4RFFVRkRMRXRCUVVzc1EwRkJReXhKUVVGSkxGZEJRVmNzUTBGQlNTeEhRVUZITEVOQlFVTXNSVUZCUlN4UlFVRlJMRU5CUVVNc1EwRkJRenRKUVVONlJDeERRVUZETzBsQlJVMHNNa0pCUVVzc1IwRkJXaXhWUVVGaExFdEJRWEZDTEVWQlFVVXNVVUZCVVR0UlFVTjRReXhKUVVGSkxFbEJRVWtzUTBGQlF5eExRVUZMTEVWQlFVVTdXVUZCUlN4UFFVRlBMRXRCUVVzc1EwRkJRenRoUVVNeFFpeEpRVUZKTEV0QlFVc3NRMEZCUXl4TFFVRkxMRVZCUVVVN1dVRkJSU3hQUVVGUExFbEJRVWtzUTBGQlF6dGhRVU12UWl4SlFVRkpMRkZCUVZFc1EwRkJReXhKUVVGSkxFTkJRVU1zU1VGQlNTeEZRVUZGTEV0QlFVc3NRMEZCUXl4SlFVRkpMRU5CUVVNc1JVRkJSVHRaUVVOMFF5eEpRVUZKTEVOQlFVTXNVVUZCVVN4RFFVRkRMRWxCUVVrc1EwRkJReXhMUVVGTExFTkJRVU1zUTBGQlF6dFpRVU14UWl4UFFVRlBMRWxCUVVrc1EwRkJRenRUUVVObU8yRkJRVTA3V1VGRFNDeExRVUZMTEVOQlFVTXNVVUZCVVN4RFFVRkRMRWxCUVVrc1EwRkJReXhKUVVGSkxFTkJRVU1zUTBGQlF6dFpRVU14UWl4UFFVRlBMRXRCUVVzc1EwRkJRenRUUVVOb1FqdEpRVU5NTEVOQlFVTTdTVUZGVFN3clFrRkJVeXhIUVVGb1FpeFZRVUZwUWl4UlFVRnBRenRSUVVNNVF5eEpRVUZKTEVsQlFVa3NRMEZCUXl4TFFVRkxMRVZCUVVVN1dVRkJSU3hQUVVGUExFbEJRVWtzUTBGQlF6czdXVUZEZWtJc1QwRkJUeXhKUVVGSkxFTkJRVU1zVlVGQlZTeERRVUZETEZGQlFWRXNRMEZCUXl4RFFVRkRPMGxCUXpGRExFTkJRVU03U1VGRlRTeG5RMEZCVlN4SFFVRnFRaXhWUVVGclFpeFJRVUZwUXp0UlFVTXZReXhKUVVGSkxFbEJRVWtzUTBGQlF5eFJRVUZSTEVOQlFVTXNUVUZCVFN4SlFVRkpMRU5CUVVNN1dVRkJSU3hQUVVGUExFbEJRVWtzVjBGQlZ5eERRVUZKTEVsQlFVa3NRMEZCUXl4RFFVRkRPMkZCUXpGRUxFbEJRVWtzU1VGQlNTeERRVUZETEZGQlFWRXNRMEZCUXl4TlFVRk5MRWxCUVVrc1EwRkJReXhGUVVGRk8xbEJRVVVzVDBGQlR5eEpRVUZKTEVOQlFVTXNVVUZCVVN4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRE8xTkJRVVU3WVVGRE0wUTdXVUZEUkN4SlFVRkpMRk5CUVZNc1IwRkJSeXhKUVVGSkxFTkJRVU1zVVVGQlVTeERRVUZETEVkQlFVY3NSVUZCUlN4RFFVRkRMRXRCUVVzc1EwRkJReXhKUVVGSkxFTkJRVU1zVVVGQlVTeERRVUZETEVkQlFVY3NSVUZCUlN4RlFVRkZMRkZCUVZFc1EwRkJReXhEUVVGRE8xbEJRM3BGTEVsQlFVa3NVMEZCVXl4SFFVRkhMRWxCUVVrc1EwRkJReXhWUVVGVkxFTkJRVU1zVVVGQlVTeERRVUZETEVOQlFVTTdXVUZETVVNc1QwRkJUeXhUUVVGVExFTkJRVU1zUzBGQlN5eERRVUZETEZOQlFWTXNSVUZCUlN4UlFVRlJMRU5CUVVNc1EwRkJRenRUUVVNdlF6dEpRVU5NTEVOQlFVTTdTVUZEVFN4cFEwRkJWeXhIUVVGc1FpeFZRVUZ0UWl4UFFVRjFRaXhGUVVGRkxGRkJRVmNzUlVGQlJTeFhRVUUwUXl4RlFVRkZMRkZCUVdsRE8xRkJRM0JKTEVsQlFVa3NUMEZCVHl4SFFVRkhMRTlCUVU4c1EwRkJReXhUUVVGVExFTkJRVU1zVVVGQlVTeERRVUZETEVOQlFVTTdVVUZGTVVNc1QwRkJUeXhEUVVGRExFbEJRVWtzUjBGQlJ5eFBRVUZQTEVOQlFVTXNTVUZCU1N4RFFVRkRPMUZCUXpWQ0xFOUJRVThzUTBGQlF5eFJRVUZSTEVkQlFVY3NUMEZCVHl4RFFVRkRMRkZCUVZFc1EwRkJRenRSUVVOd1F5eEpRVUZKTEZkQlFWY3NTMEZCU3l4SlFVRkpMRWxCUVVrc1QwRkJUeXhEUVVGRExFbEJRVWtzUzBGQlN5eEpRVUZKTEVWQlFVVTdXVUZETDBNc1YwRkJWeXhEUVVGRExFOUJRVThzUTBGQlF5eEpRVUZKTEVWQlFVVXNUMEZCVHl4RFFVRkRMRU5CUVVNN1UwRkRkRU03VVVGRFJDeEpRVUZKTEZkQlFWY3NSMEZCUnl4SlFVRkpMRmRCUVZjc1EwRkJReXhSUVVGUkxFTkJRVU1zUTBGQlF6dFJRVU0xUXl4SlFVRkpMRmRCUVZjc1MwRkJTeXhKUVVGSkxFVkJRVVU3V1VGRGRFSXNWMEZCVnl4RFFVRkRMRkZCUVZFc1JVRkJSU3hYUVVGWExFTkJRVU1zUTBGQlF6dFRRVU4wUXp0UlFVTkVMRTlCUVU4c1NVRkJTU3hEUVVGRExFdEJRVXNzUTBGQlF5eFhRVUZYTEVWQlFVVXNVVUZCVVN4RFFVRkRMRU5CUVVNN1NVRkROME1zUTBGQlF6dEpRVU5NTEd0Q1FVRkRPMEZCUVVRc1EwRkJReXhCUVhwSFJDeEpRWGxIUXp0QlFYcEhXU3hyUTBGQlZ6dEJRVGhIZUVJN1NVRkZTU3gxUWtGQmIwSXNVVUZCYVVNN1VVRkJha01zWVVGQlVTeEhRVUZTTEZGQlFWRXNRMEZCZVVJN1NVRkJTU3hEUVVGRE8wbEJTMjVFTERKQ1FVRkhMRWRCUVZZN1VVRkRTU3hKUVVGSkxFbEJRVWtzUTBGQlF5eExRVUZMTEVWQlFVVXNSVUZCUlR0WlFVRkZMRTlCUVU4c1NVRkJTU3hEUVVGRE8xTkJRVVU3VVVGRGJFTXNUMEZCVHl4SlFVRkpMRU5CUVVNc1NVRkJTU3hEUVVGRExFbEJRVWtzUTBGQlF6dEpRVU14UWl4RFFVRkRPMGxCUzAwc05FSkJRVWtzUjBGQldEdFJRVUZaTEdOQlFWazdZVUZCV2l4VlFVRlpMRVZCUVZvc2NVSkJRVmtzUlVGQldpeEpRVUZaTzFsQlFWb3NlVUpCUVZrN08xRkJRM0JDTEVsQlFVa3NWMEZCVnl4RFFVRkRPMUZCUTJoQ0xFdEJRVXNzU1VGQlNTeERRVUZETEVkQlFVY3NRMEZCUXl4RlFVRkZMRWRCUVVjc1JVRkJSU3hIUVVGSExFZEJRVU1zU1VGQlNTeERRVUZETEVOQlFVTXNRMEZCUXl4RlFVRkZMRVZCUVVVc1EwRkJReXhGUVVGRk8xbEJRMjVETEZkQlFWY3NSMEZCUnl4SlFVRkpMRmRCUVZjc1EwRkJReXhIUVVGSExFTkJRVU1zUTBGQlF6dFpRVU51UXl4SlFVRkpMRU5CUVVNc1NVRkJTU3hIUVVGSExFbEJRVWtzUTBGQlF5eExRVUZMTEVWQlFVVXNRMEZCUXl4RFFVRkRPMmRDUVVOMFFpeFhRVUZYTEVOQlFVTXNRMEZCUXl4RFFVRkRMRWxCUVVrc1EwRkJReXhKUVVGSkxFTkJRVU1zUzBGQlN5eERRVUZETEZkQlFWY3NSVUZCUlN4SlFVRkpMRU5CUVVNc1VVRkJVU3hEUVVGRExFTkJRVU03VTBGRGFrVTdVVUZEUkN4UFFVRlBMRmRCUVZjc1EwRkJRenRKUVVOMlFpeERRVUZETzBsQlMwMHNOa0pCUVVzc1IwRkJXanRSUVVOSkxFOUJRVThzUTBGQlF5eEpRVUZKTEVOQlFVTXNTVUZCU1N4SlFVRkpMRU5CUVVNc1NVRkJTU3hEUVVGRExFbEJRVWtzUTBGQlF5eEpRVUZKTEVOQlFVTTdTVUZEZWtNc1EwRkJRenRKUVV0TkxEaENRVUZOTEVkQlFXSTdVVUZEU1N4UFFVRlBMRWxCUVVrc1EwRkJReXhKUVVGSkxFTkJRVU1zVFVGQlRTeERRVUZETEVsQlFVa3NRMEZCUXl4UlFVRlJMRU5CUVVNc1EwRkJRenRKUVVNelF5eERRVUZETzBsQlMwMHNLMEpCUVU4c1IwRkJaQ3hWUVVGbExFTkJRVU03VVVGRFdpeEpRVUZKTEVOQlFVTXNTVUZCU1N4RFFVRkRMRTlCUVU4c1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dEpRVU42UWl4RFFVRkRPMGxCU1Uwc01rSkJRVWNzUjBGQlZqdFJRVU5KTEVsQlFVa3NTVUZCU1N4RFFVRkRMRXRCUVVzc1JVRkJSU3hGUVVGRk8xbEJRMlFzVDBGQlR5eEpRVUZKTEVOQlFVTTdVMEZEWmp0UlFVTkVMRWxCUVVrc1IwRkJSeXhIUVVGSExFbEJRVWtzUTBGQlF5eEpRVUZKTEVOQlFVTXNSMEZCUnl4RlFVRkZMRU5CUVVNN1VVRkRNVUlzU1VGQlNTeERRVUZETEVsQlFVa3NSMEZCUnl4SlFVRkpMRU5CUVVNc1NVRkJTU3hEUVVGRExGTkJRVk1zUTBGQlF5eEpRVUZKTEVOQlFVTXNVVUZCVVN4RFFVRkRMRU5CUVVNN1VVRkRMME1zVDBGQlR5eEhRVUZITEVOQlFVTTdTVUZEWml4RFFVRkRPMGxCU1Uwc2FVTkJRVk1zUjBGQmFFSXNWVUZCYVVJc1VVRkJkMElzUlVGQlJTeE5RVUZUTEVWQlFVVXNWMEZCYlVRN1VVRkJia1FzTkVKQlFVRXNSVUZCUVN4clFrRkJiVVE3VVVGRGNrY3NTVUZCU1N4RFFVRkRMRWxCUVVrc1IwRkJSeXhKUVVGSkxFTkJRVU1zU1VGQlNTeERRVUZETEZkQlFWY3NRMEZCUXl4UlFVRlJMRVZCUVVVc1RVRkJUU3hGUVVGRkxGZEJRVmNzUlVGQlJTeEpRVUZKTEVOQlFVTXNVVUZCVVN4RFFVRkRMRU5CUVVNN1NVRkRjRVlzUTBGQlF6dEpRVU5OTEdkRFFVRlJMRWRCUVdZc1ZVRkJaMElzVVVGQlVUdFJRVU53UWl4UFFVRlBMRWxCUVVrc1EwRkJReXhKUVVGSkxFTkJRVU1zVVVGQlVTeERRVUZETEZGQlFWRXNRMEZCUXl4RFFVRkRPMGxCUTNoRExFTkJRVU03U1VGTFRTdzJRa0ZCU3l4SFFVRmFPMUZCUTBrc1QwRkJUeXhKUVVGSkxFTkJRVU1zU1VGQlNTeERRVUZETEV0QlFVc3NSVUZCUlN4RFFVRkRPMGxCUXpkQ0xFTkJRVU03U1VGRFRDeHZRa0ZCUXp0QlFVRkVMRU5CUVVNc1FVRjRSVVFzU1VGM1JVTTdRVUY0UlZrc2MwTkJRV0VpZlE9PSIsIlwidXNlIHN0cmljdFwiO1xyXG52YXIgX19leHRlbmRzID0gKHRoaXMgJiYgdGhpcy5fX2V4dGVuZHMpIHx8IChmdW5jdGlvbiAoKSB7XHJcbiAgICB2YXIgZXh0ZW5kU3RhdGljcyA9IGZ1bmN0aW9uIChkLCBiKSB7XHJcbiAgICAgICAgZXh0ZW5kU3RhdGljcyA9IE9iamVjdC5zZXRQcm90b3R5cGVPZiB8fFxyXG4gICAgICAgICAgICAoeyBfX3Byb3RvX186IFtdIH0gaW5zdGFuY2VvZiBBcnJheSAmJiBmdW5jdGlvbiAoZCwgYikgeyBkLl9fcHJvdG9fXyA9IGI7IH0pIHx8XHJcbiAgICAgICAgICAgIGZ1bmN0aW9uIChkLCBiKSB7IGZvciAodmFyIHAgaW4gYikgaWYgKGIuaGFzT3duUHJvcGVydHkocCkpIGRbcF0gPSBiW3BdOyB9O1xyXG4gICAgICAgIHJldHVybiBleHRlbmRTdGF0aWNzKGQsIGIpO1xyXG4gICAgfTtcclxuICAgIHJldHVybiBmdW5jdGlvbiAoZCwgYikge1xyXG4gICAgICAgIGV4dGVuZFN0YXRpY3MoZCwgYik7XHJcbiAgICAgICAgZnVuY3Rpb24gX18oKSB7IHRoaXMuY29uc3RydWN0b3IgPSBkOyB9XHJcbiAgICAgICAgZC5wcm90b3R5cGUgPSBiID09PSBudWxsID8gT2JqZWN0LmNyZWF0ZShiKSA6IChfXy5wcm90b3R5cGUgPSBiLnByb3RvdHlwZSwgbmV3IF9fKCkpO1xyXG4gICAgfTtcclxufSkoKTtcclxuT2JqZWN0LmRlZmluZVByb3BlcnR5KGV4cG9ydHMsIFwiX19lc01vZHVsZVwiLCB7IHZhbHVlOiB0cnVlIH0pO1xyXG52YXIgVHJlZUJhc2UgPSAoZnVuY3Rpb24gKCkge1xyXG4gICAgZnVuY3Rpb24gVHJlZUJhc2UoKSB7XHJcbiAgICAgICAgdGhpcy5maW5kSXRlciA9IGZ1bmN0aW9uIChkYXRhKSB7XHJcbiAgICAgICAgICAgIHZhciByZXMgPSB0aGlzLl9yb290O1xyXG4gICAgICAgICAgICB2YXIgaXRlciA9IHRoaXMuaXRlcmF0b3IoKTtcclxuICAgICAgICAgICAgd2hpbGUgKHJlcyAhPT0gbnVsbCkge1xyXG4gICAgICAgICAgICAgICAgdmFyIGMgPSB0aGlzLl9jb21wYXJhdG9yKGRhdGEsIHJlcy5kYXRhKTtcclxuICAgICAgICAgICAgICAgIGlmIChjID09PSAwKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgaXRlci5fY3Vyc29yID0gcmVzO1xyXG4gICAgICAgICAgICAgICAgICAgIHJldHVybiBpdGVyO1xyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgZWxzZSB7XHJcbiAgICAgICAgICAgICAgICAgICAgaXRlci5fYW5jZXN0b3JzLnB1c2gocmVzKTtcclxuICAgICAgICAgICAgICAgICAgICByZXMgPSByZXMuZ2V0X2NoaWxkKGMgPiAwKTtcclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICByZXR1cm4gbnVsbDtcclxuICAgICAgICB9O1xyXG4gICAgfVxyXG4gICAgVHJlZUJhc2UucHJvdG90eXBlLmNsZWFyID0gZnVuY3Rpb24gKCkge1xyXG4gICAgICAgIHRoaXMuX3Jvb3QgPSBudWxsO1xyXG4gICAgICAgIHRoaXMuc2l6ZSA9IDA7XHJcbiAgICB9O1xyXG4gICAgO1xyXG4gICAgVHJlZUJhc2UucHJvdG90eXBlLmZpbmQgPSBmdW5jdGlvbiAoZGF0YSkge1xyXG4gICAgICAgIHZhciByZXMgPSB0aGlzLl9yb290O1xyXG4gICAgICAgIHdoaWxlIChyZXMgIT09IG51bGwpIHtcclxuICAgICAgICAgICAgdmFyIGMgPSB0aGlzLl9jb21wYXJhdG9yKGRhdGEsIHJlcy5kYXRhKTtcclxuICAgICAgICAgICAgaWYgKGMgPT09IDApIHtcclxuICAgICAgICAgICAgICAgIHJldHVybiByZXMuZGF0YTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgICAgIHJlcyA9IHJlcy5nZXRfY2hpbGQoYyA+IDApO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHJldHVybiBudWxsO1xyXG4gICAgfTtcclxuICAgIDtcclxuICAgIFRyZWVCYXNlLnByb3RvdHlwZS5sb3dlckJvdW5kID0gZnVuY3Rpb24gKGRhdGEpIHtcclxuICAgICAgICByZXR1cm4gdGhpcy5fYm91bmQoZGF0YSwgdGhpcy5fY29tcGFyYXRvcik7XHJcbiAgICB9O1xyXG4gICAgO1xyXG4gICAgVHJlZUJhc2UucHJvdG90eXBlLnVwcGVyQm91bmQgPSBmdW5jdGlvbiAoZGF0YSkge1xyXG4gICAgICAgIHZhciBjbXAgPSB0aGlzLl9jb21wYXJhdG9yO1xyXG4gICAgICAgIGZ1bmN0aW9uIHJldmVyc2VfY21wKGEsIGIpIHtcclxuICAgICAgICAgICAgcmV0dXJuIGNtcChiLCBhKTtcclxuICAgICAgICB9XHJcbiAgICAgICAgcmV0dXJuIHRoaXMuX2JvdW5kKGRhdGEsIHJldmVyc2VfY21wKTtcclxuICAgIH07XHJcbiAgICA7XHJcbiAgICBUcmVlQmFzZS5wcm90b3R5cGUubWluID0gZnVuY3Rpb24gKCkge1xyXG4gICAgICAgIHZhciByZXMgPSB0aGlzLl9yb290O1xyXG4gICAgICAgIGlmIChyZXMgPT09IG51bGwpIHtcclxuICAgICAgICAgICAgcmV0dXJuIG51bGw7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHdoaWxlIChyZXMubGVmdCAhPT0gbnVsbCkge1xyXG4gICAgICAgICAgICByZXMgPSByZXMubGVmdDtcclxuICAgICAgICB9XHJcbiAgICAgICAgcmV0dXJuIHJlcy5kYXRhO1xyXG4gICAgfTtcclxuICAgIDtcclxuICAgIFRyZWVCYXNlLnByb3RvdHlwZS5tYXggPSBmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgdmFyIHJlcyA9IHRoaXMuX3Jvb3Q7XHJcbiAgICAgICAgaWYgKHJlcyA9PT0gbnVsbCkge1xyXG4gICAgICAgICAgICByZXR1cm4gbnVsbDtcclxuICAgICAgICB9XHJcbiAgICAgICAgd2hpbGUgKHJlcy5yaWdodCAhPT0gbnVsbCkge1xyXG4gICAgICAgICAgICByZXMgPSByZXMucmlnaHQ7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHJldHVybiByZXMuZGF0YTtcclxuICAgIH07XHJcbiAgICA7XHJcbiAgICBUcmVlQmFzZS5wcm90b3R5cGUuaXRlcmF0b3IgPSBmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgcmV0dXJuIG5ldyBJdGVyYXRvcih0aGlzKTtcclxuICAgIH07XHJcbiAgICA7XHJcbiAgICBUcmVlQmFzZS5wcm90b3R5cGUuZWFjaCA9IGZ1bmN0aW9uIChjYikge1xyXG4gICAgICAgIHZhciBpdCA9IHRoaXMuaXRlcmF0b3IoKSwgZGF0YTtcclxuICAgICAgICB3aGlsZSAoKGRhdGEgPSBpdC5uZXh0KCkpICE9PSBudWxsKSB7XHJcbiAgICAgICAgICAgIGNiKGRhdGEpO1xyXG4gICAgICAgIH1cclxuICAgIH07XHJcbiAgICA7XHJcbiAgICBUcmVlQmFzZS5wcm90b3R5cGUucmVhY2ggPSBmdW5jdGlvbiAoY2IpIHtcclxuICAgICAgICB2YXIgaXQgPSB0aGlzLml0ZXJhdG9yKCksIGRhdGE7XHJcbiAgICAgICAgd2hpbGUgKChkYXRhID0gaXQucHJldigpKSAhPT0gbnVsbCkge1xyXG4gICAgICAgICAgICBjYihkYXRhKTtcclxuICAgICAgICB9XHJcbiAgICB9O1xyXG4gICAgO1xyXG4gICAgVHJlZUJhc2UucHJvdG90eXBlLl9ib3VuZCA9IGZ1bmN0aW9uIChkYXRhLCBjbXApIHtcclxuICAgICAgICB2YXIgY3VyID0gdGhpcy5fcm9vdDtcclxuICAgICAgICB2YXIgaXRlciA9IHRoaXMuaXRlcmF0b3IoKTtcclxuICAgICAgICB3aGlsZSAoY3VyICE9PSBudWxsKSB7XHJcbiAgICAgICAgICAgIHZhciBjID0gdGhpcy5fY29tcGFyYXRvcihkYXRhLCBjdXIuZGF0YSk7XHJcbiAgICAgICAgICAgIGlmIChjID09PSAwKSB7XHJcbiAgICAgICAgICAgICAgICBpdGVyLl9jdXJzb3IgPSBjdXI7XHJcbiAgICAgICAgICAgICAgICByZXR1cm4gaXRlcjtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICBpdGVyLl9hbmNlc3RvcnMucHVzaChjdXIpO1xyXG4gICAgICAgICAgICBjdXIgPSBjdXIuZ2V0X2NoaWxkKGMgPiAwKTtcclxuICAgICAgICB9XHJcbiAgICAgICAgZm9yICh2YXIgaSA9IGl0ZXIuX2FuY2VzdG9ycy5sZW5ndGggLSAxOyBpID49IDA7IC0taSkge1xyXG4gICAgICAgICAgICBjdXIgPSBpdGVyLl9hbmNlc3RvcnNbaV07XHJcbiAgICAgICAgICAgIGlmIChjbXAoZGF0YSwgY3VyLmRhdGEpID4gMCkge1xyXG4gICAgICAgICAgICAgICAgaXRlci5fY3Vyc29yID0gY3VyO1xyXG4gICAgICAgICAgICAgICAgaXRlci5fYW5jZXN0b3JzLmxlbmd0aCA9IGk7XHJcbiAgICAgICAgICAgICAgICByZXR1cm4gaXRlcjtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH1cclxuICAgICAgICBpdGVyLl9hbmNlc3RvcnMubGVuZ3RoID0gMDtcclxuICAgICAgICByZXR1cm4gaXRlcjtcclxuICAgIH07XHJcbiAgICA7XHJcbiAgICByZXR1cm4gVHJlZUJhc2U7XHJcbn0oKSk7XHJcbmV4cG9ydHMuVHJlZUJhc2UgPSBUcmVlQmFzZTtcclxudmFyIEl0ZXJhdG9yID0gKGZ1bmN0aW9uICgpIHtcclxuICAgIGZ1bmN0aW9uIEl0ZXJhdG9yKHRyZWUpIHtcclxuICAgICAgICB0aGlzLl90cmVlID0gdHJlZTtcclxuICAgICAgICB0aGlzLl9hbmNlc3RvcnMgPSBbXTtcclxuICAgICAgICB0aGlzLl9jdXJzb3IgPSBudWxsO1xyXG4gICAgfVxyXG4gICAgSXRlcmF0b3IucHJvdG90eXBlLmRhdGEgPSBmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgcmV0dXJuIHRoaXMuX2N1cnNvciAhPT0gbnVsbCA/IHRoaXMuX2N1cnNvci5kYXRhIDogbnVsbDtcclxuICAgIH07XHJcbiAgICA7XHJcbiAgICBJdGVyYXRvci5wcm90b3R5cGUubmV4dCA9IGZ1bmN0aW9uICgpIHtcclxuICAgICAgICBpZiAodGhpcy5fY3Vyc29yID09PSBudWxsKSB7XHJcbiAgICAgICAgICAgIHZhciByb290ID0gdGhpcy5fdHJlZS5fcm9vdDtcclxuICAgICAgICAgICAgaWYgKHJvb3QgIT09IG51bGwpIHtcclxuICAgICAgICAgICAgICAgIHRoaXMuX21pbk5vZGUocm9vdCk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICAgICAgZWxzZSB7XHJcbiAgICAgICAgICAgIGlmICh0aGlzLl9jdXJzb3IucmlnaHQgPT09IG51bGwpIHtcclxuICAgICAgICAgICAgICAgIHZhciBzYXZlO1xyXG4gICAgICAgICAgICAgICAgZG8ge1xyXG4gICAgICAgICAgICAgICAgICAgIHNhdmUgPSB0aGlzLl9jdXJzb3I7XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKHRoaXMuX2FuY2VzdG9ycy5sZW5ndGgpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgdGhpcy5fY3Vyc29yID0gdGhpcy5fYW5jZXN0b3JzLnBvcCgpO1xyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgdGhpcy5fY3Vyc29yID0gbnVsbDtcclxuICAgICAgICAgICAgICAgICAgICAgICAgYnJlYWs7XHJcbiAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgfSB3aGlsZSAodGhpcy5fY3Vyc29yLnJpZ2h0ID09PSBzYXZlKTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgICAgIHRoaXMuX2FuY2VzdG9ycy5wdXNoKHRoaXMuX2N1cnNvcik7XHJcbiAgICAgICAgICAgICAgICB0aGlzLl9taW5Ob2RlKHRoaXMuX2N1cnNvci5yaWdodCk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICAgICAgcmV0dXJuIHRoaXMuX2N1cnNvciAhPT0gbnVsbCA/IHRoaXMuX2N1cnNvci5kYXRhIDogbnVsbDtcclxuICAgIH07XHJcbiAgICA7XHJcbiAgICBJdGVyYXRvci5wcm90b3R5cGUucHJldiA9IGZ1bmN0aW9uICgpIHtcclxuICAgICAgICBpZiAodGhpcy5fY3Vyc29yID09PSBudWxsKSB7XHJcbiAgICAgICAgICAgIHZhciByb290ID0gdGhpcy5fdHJlZS5fcm9vdDtcclxuICAgICAgICAgICAgaWYgKHJvb3QgIT09IG51bGwpIHtcclxuICAgICAgICAgICAgICAgIHRoaXMuX21heE5vZGUocm9vdCk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICAgICAgZWxzZSB7XHJcbiAgICAgICAgICAgIGlmICh0aGlzLl9jdXJzb3IubGVmdCA9PT0gbnVsbCkge1xyXG4gICAgICAgICAgICAgICAgdmFyIHNhdmU7XHJcbiAgICAgICAgICAgICAgICBkbyB7XHJcbiAgICAgICAgICAgICAgICAgICAgc2F2ZSA9IHRoaXMuX2N1cnNvcjtcclxuICAgICAgICAgICAgICAgICAgICBpZiAodGhpcy5fYW5jZXN0b3JzLmxlbmd0aCkge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICB0aGlzLl9jdXJzb3IgPSB0aGlzLl9hbmNlc3RvcnMucG9wKCk7XHJcbiAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgIGVsc2Uge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICB0aGlzLl9jdXJzb3IgPSBudWxsO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICBicmVhaztcclxuICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICB9IHdoaWxlICh0aGlzLl9jdXJzb3IubGVmdCA9PT0gc2F2ZSk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgZWxzZSB7XHJcbiAgICAgICAgICAgICAgICB0aGlzLl9hbmNlc3RvcnMucHVzaCh0aGlzLl9jdXJzb3IpO1xyXG4gICAgICAgICAgICAgICAgdGhpcy5fbWF4Tm9kZSh0aGlzLl9jdXJzb3IubGVmdCk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICAgICAgcmV0dXJuIHRoaXMuX2N1cnNvciAhPT0gbnVsbCA/IHRoaXMuX2N1cnNvci5kYXRhIDogbnVsbDtcclxuICAgIH07XHJcbiAgICA7XHJcbiAgICBJdGVyYXRvci5wcm90b3R5cGUuX21pbk5vZGUgPSBmdW5jdGlvbiAoc3RhcnQpIHtcclxuICAgICAgICB3aGlsZSAoc3RhcnQubGVmdCAhPT0gbnVsbCkge1xyXG4gICAgICAgICAgICB0aGlzLl9hbmNlc3RvcnMucHVzaChzdGFydCk7XHJcbiAgICAgICAgICAgIHN0YXJ0ID0gc3RhcnQubGVmdDtcclxuICAgICAgICB9XHJcbiAgICAgICAgdGhpcy5fY3Vyc29yID0gc3RhcnQ7XHJcbiAgICB9O1xyXG4gICAgO1xyXG4gICAgSXRlcmF0b3IucHJvdG90eXBlLl9tYXhOb2RlID0gZnVuY3Rpb24gKHN0YXJ0KSB7XHJcbiAgICAgICAgd2hpbGUgKHN0YXJ0LnJpZ2h0ICE9PSBudWxsKSB7XHJcbiAgICAgICAgICAgIHRoaXMuX2FuY2VzdG9ycy5wdXNoKHN0YXJ0KTtcclxuICAgICAgICAgICAgc3RhcnQgPSBzdGFydC5yaWdodDtcclxuICAgICAgICB9XHJcbiAgICAgICAgdGhpcy5fY3Vyc29yID0gc3RhcnQ7XHJcbiAgICB9O1xyXG4gICAgO1xyXG4gICAgcmV0dXJuIEl0ZXJhdG9yO1xyXG59KCkpO1xyXG5leHBvcnRzLkl0ZXJhdG9yID0gSXRlcmF0b3I7XHJcbnZhciBOb2RlID0gKGZ1bmN0aW9uICgpIHtcclxuICAgIGZ1bmN0aW9uIE5vZGUoZGF0YSkge1xyXG4gICAgICAgIHRoaXMuZGF0YSA9IGRhdGE7XHJcbiAgICAgICAgdGhpcy5sZWZ0ID0gbnVsbDtcclxuICAgICAgICB0aGlzLnJpZ2h0ID0gbnVsbDtcclxuICAgICAgICB0aGlzLnJlZCA9IHRydWU7XHJcbiAgICB9XHJcbiAgICBOb2RlLnByb3RvdHlwZS5nZXRfY2hpbGQgPSBmdW5jdGlvbiAoZGlyKSB7XHJcbiAgICAgICAgcmV0dXJuIGRpciA/IHRoaXMucmlnaHQgOiB0aGlzLmxlZnQ7XHJcbiAgICB9O1xyXG4gICAgO1xyXG4gICAgTm9kZS5wcm90b3R5cGUuc2V0X2NoaWxkID0gZnVuY3Rpb24gKGRpciwgdmFsKSB7XHJcbiAgICAgICAgaWYgKGRpcikge1xyXG4gICAgICAgICAgICB0aGlzLnJpZ2h0ID0gdmFsO1xyXG4gICAgICAgIH1cclxuICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgdGhpcy5sZWZ0ID0gdmFsO1xyXG4gICAgICAgIH1cclxuICAgIH07XHJcbiAgICA7XHJcbiAgICByZXR1cm4gTm9kZTtcclxufSgpKTtcclxudmFyIFJCVHJlZSA9IChmdW5jdGlvbiAoX3N1cGVyKSB7XHJcbiAgICBfX2V4dGVuZHMoUkJUcmVlLCBfc3VwZXIpO1xyXG4gICAgZnVuY3Rpb24gUkJUcmVlKGNvbXBhcmF0b3IpIHtcclxuICAgICAgICB2YXIgX3RoaXMgPSBfc3VwZXIuY2FsbCh0aGlzKSB8fCB0aGlzO1xyXG4gICAgICAgIF90aGlzLl9yb290ID0gbnVsbDtcclxuICAgICAgICBfdGhpcy5fY29tcGFyYXRvciA9IGNvbXBhcmF0b3I7XHJcbiAgICAgICAgX3RoaXMuc2l6ZSA9IDA7XHJcbiAgICAgICAgcmV0dXJuIF90aGlzO1xyXG4gICAgfVxyXG4gICAgUkJUcmVlLnByb3RvdHlwZS5pbnNlcnQgPSBmdW5jdGlvbiAoZGF0YSkge1xyXG4gICAgICAgIHZhciByZXQgPSBmYWxzZTtcclxuICAgICAgICBpZiAodGhpcy5fcm9vdCA9PT0gbnVsbCkge1xyXG4gICAgICAgICAgICB0aGlzLl9yb290ID0gbmV3IE5vZGUoZGF0YSk7XHJcbiAgICAgICAgICAgIHJldCA9IHRydWU7XHJcbiAgICAgICAgICAgIHRoaXMuc2l6ZSsrO1xyXG4gICAgICAgIH1cclxuICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgdmFyIGhlYWQgPSBuZXcgTm9kZSh1bmRlZmluZWQpO1xyXG4gICAgICAgICAgICB2YXIgZGlyID0gZmFsc2U7XHJcbiAgICAgICAgICAgIHZhciBsYXN0ID0gZmFsc2U7XHJcbiAgICAgICAgICAgIHZhciBncCA9IG51bGw7XHJcbiAgICAgICAgICAgIHZhciBnZ3AgPSBoZWFkO1xyXG4gICAgICAgICAgICB2YXIgcCA9IG51bGw7XHJcbiAgICAgICAgICAgIHZhciBub2RlID0gdGhpcy5fcm9vdDtcclxuICAgICAgICAgICAgZ2dwLnJpZ2h0ID0gdGhpcy5fcm9vdDtcclxuICAgICAgICAgICAgd2hpbGUgKHRydWUpIHtcclxuICAgICAgICAgICAgICAgIGlmIChub2RlID09PSBudWxsKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgbm9kZSA9IG5ldyBOb2RlKGRhdGEpO1xyXG4gICAgICAgICAgICAgICAgICAgIHAuc2V0X2NoaWxkKGRpciwgbm9kZSk7XHJcbiAgICAgICAgICAgICAgICAgICAgcmV0ID0gdHJ1ZTtcclxuICAgICAgICAgICAgICAgICAgICB0aGlzLnNpemUrKztcclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIGVsc2UgaWYgKFJCVHJlZS5pc19yZWQobm9kZS5sZWZ0KSAmJiBSQlRyZWUuaXNfcmVkKG5vZGUucmlnaHQpKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgbm9kZS5yZWQgPSB0cnVlO1xyXG4gICAgICAgICAgICAgICAgICAgIG5vZGUubGVmdC5yZWQgPSBmYWxzZTtcclxuICAgICAgICAgICAgICAgICAgICBub2RlLnJpZ2h0LnJlZCA9IGZhbHNlO1xyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgaWYgKFJCVHJlZS5pc19yZWQobm9kZSkgJiYgUkJUcmVlLmlzX3JlZChwKSkge1xyXG4gICAgICAgICAgICAgICAgICAgIHZhciBkaXIyID0gZ2dwLnJpZ2h0ID09PSBncDtcclxuICAgICAgICAgICAgICAgICAgICBpZiAobm9kZSA9PT0gcC5nZXRfY2hpbGQobGFzdCkpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgZ2dwLnNldF9jaGlsZChkaXIyLCBSQlRyZWUuc2luZ2xlX3JvdGF0ZShncCwgIWxhc3QpKTtcclxuICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgZWxzZSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGdncC5zZXRfY2hpbGQoZGlyMiwgUkJUcmVlLmRvdWJsZV9yb3RhdGUoZ3AsICFsYXN0KSk7XHJcbiAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgdmFyIGNtcCA9IHRoaXMuX2NvbXBhcmF0b3Iobm9kZS5kYXRhLCBkYXRhKTtcclxuICAgICAgICAgICAgICAgIGlmIChjbXAgPT09IDApIHtcclxuICAgICAgICAgICAgICAgICAgICBicmVhaztcclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIGxhc3QgPSBkaXI7XHJcbiAgICAgICAgICAgICAgICBkaXIgPSBjbXAgPCAwO1xyXG4gICAgICAgICAgICAgICAgaWYgKGdwICE9PSBudWxsKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgZ2dwID0gZ3A7XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICBncCA9IHA7XHJcbiAgICAgICAgICAgICAgICBwID0gbm9kZTtcclxuICAgICAgICAgICAgICAgIG5vZGUgPSBub2RlLmdldF9jaGlsZChkaXIpO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIHRoaXMuX3Jvb3QgPSBoZWFkLnJpZ2h0O1xyXG4gICAgICAgIH1cclxuICAgICAgICB0aGlzLl9yb290LnJlZCA9IGZhbHNlO1xyXG4gICAgICAgIHJldHVybiByZXQ7XHJcbiAgICB9O1xyXG4gICAgO1xyXG4gICAgUkJUcmVlLnByb3RvdHlwZS5yZW1vdmUgPSBmdW5jdGlvbiAoZGF0YSkge1xyXG4gICAgICAgIGlmICh0aGlzLl9yb290ID09PSBudWxsKSB7XHJcbiAgICAgICAgICAgIHJldHVybiBmYWxzZTtcclxuICAgICAgICB9XHJcbiAgICAgICAgdmFyIGhlYWQgPSBuZXcgTm9kZSh1bmRlZmluZWQpO1xyXG4gICAgICAgIHZhciBub2RlID0gaGVhZDtcclxuICAgICAgICBub2RlLnJpZ2h0ID0gdGhpcy5fcm9vdDtcclxuICAgICAgICB2YXIgcCA9IG51bGw7XHJcbiAgICAgICAgdmFyIGdwID0gbnVsbDtcclxuICAgICAgICB2YXIgZm91bmQgPSBudWxsO1xyXG4gICAgICAgIHZhciBkaXIgPSB0cnVlO1xyXG4gICAgICAgIHdoaWxlIChub2RlLmdldF9jaGlsZChkaXIpICE9PSBudWxsKSB7XHJcbiAgICAgICAgICAgIHZhciBsYXN0ID0gZGlyO1xyXG4gICAgICAgICAgICBncCA9IHA7XHJcbiAgICAgICAgICAgIHAgPSBub2RlO1xyXG4gICAgICAgICAgICBub2RlID0gbm9kZS5nZXRfY2hpbGQoZGlyKTtcclxuICAgICAgICAgICAgdmFyIGNtcCA9IHRoaXMuX2NvbXBhcmF0b3IoZGF0YSwgbm9kZS5kYXRhKTtcclxuICAgICAgICAgICAgZGlyID0gY21wID4gMDtcclxuICAgICAgICAgICAgaWYgKGNtcCA9PT0gMCkge1xyXG4gICAgICAgICAgICAgICAgZm91bmQgPSBub2RlO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIGlmICghUkJUcmVlLmlzX3JlZChub2RlKSAmJiAhUkJUcmVlLmlzX3JlZChub2RlLmdldF9jaGlsZChkaXIpKSkge1xyXG4gICAgICAgICAgICAgICAgaWYgKFJCVHJlZS5pc19yZWQobm9kZS5nZXRfY2hpbGQoIWRpcikpKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgdmFyIHNyID0gUkJUcmVlLnNpbmdsZV9yb3RhdGUobm9kZSwgZGlyKTtcclxuICAgICAgICAgICAgICAgICAgICBwLnNldF9jaGlsZChsYXN0LCBzcik7XHJcbiAgICAgICAgICAgICAgICAgICAgcCA9IHNyO1xyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgZWxzZSBpZiAoIVJCVHJlZS5pc19yZWQobm9kZS5nZXRfY2hpbGQoIWRpcikpKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgdmFyIHNpYmxpbmcgPSBwLmdldF9jaGlsZCghbGFzdCk7XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKHNpYmxpbmcgIT09IG51bGwpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgaWYgKCFSQlRyZWUuaXNfcmVkKHNpYmxpbmcuZ2V0X2NoaWxkKCFsYXN0KSkgJiYgIVJCVHJlZS5pc19yZWQoc2libGluZy5nZXRfY2hpbGQobGFzdCkpKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBwLnJlZCA9IGZhbHNlO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgc2libGluZy5yZWQgPSB0cnVlO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgbm9kZS5yZWQgPSB0cnVlO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGVsc2Uge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgdmFyIGRpcjIgPSBncC5yaWdodCA9PT0gcDtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGlmIChSQlRyZWUuaXNfcmVkKHNpYmxpbmcuZ2V0X2NoaWxkKGxhc3QpKSkge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGdwLnNldF9jaGlsZChkaXIyLCBSQlRyZWUuZG91YmxlX3JvdGF0ZShwLCBsYXN0KSk7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBlbHNlIGlmIChSQlRyZWUuaXNfcmVkKHNpYmxpbmcuZ2V0X2NoaWxkKCFsYXN0KSkpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBncC5zZXRfY2hpbGQoZGlyMiwgUkJUcmVlLnNpbmdsZV9yb3RhdGUocCwgbGFzdCkpO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgdmFyIGdwYyA9IGdwLmdldF9jaGlsZChkaXIyKTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGdwYy5yZWQgPSB0cnVlO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgbm9kZS5yZWQgPSB0cnVlO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgZ3BjLmxlZnQucmVkID0gZmFsc2U7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBncGMucmlnaHQucmVkID0gZmFsc2U7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICAgICAgaWYgKGZvdW5kICE9PSBudWxsKSB7XHJcbiAgICAgICAgICAgIGZvdW5kLmRhdGEgPSBub2RlLmRhdGE7XHJcbiAgICAgICAgICAgIHAuc2V0X2NoaWxkKHAucmlnaHQgPT09IG5vZGUsIG5vZGUuZ2V0X2NoaWxkKG5vZGUubGVmdCA9PT0gbnVsbCkpO1xyXG4gICAgICAgICAgICB0aGlzLnNpemUtLTtcclxuICAgICAgICB9XHJcbiAgICAgICAgdGhpcy5fcm9vdCA9IGhlYWQucmlnaHQ7XHJcbiAgICAgICAgaWYgKHRoaXMuX3Jvb3QgIT09IG51bGwpIHtcclxuICAgICAgICAgICAgdGhpcy5fcm9vdC5yZWQgPSBmYWxzZTtcclxuICAgICAgICB9XHJcbiAgICAgICAgcmV0dXJuIGZvdW5kICE9PSBudWxsO1xyXG4gICAgfTtcclxuICAgIDtcclxuICAgIFJCVHJlZS5pc19yZWQgPSBmdW5jdGlvbiAobm9kZSkge1xyXG4gICAgICAgIHJldHVybiBub2RlICE9PSBudWxsICYmIG5vZGUucmVkO1xyXG4gICAgfTtcclxuICAgIFJCVHJlZS5zaW5nbGVfcm90YXRlID0gZnVuY3Rpb24gKHJvb3QsIGRpcikge1xyXG4gICAgICAgIHZhciBzYXZlID0gcm9vdC5nZXRfY2hpbGQoIWRpcik7XHJcbiAgICAgICAgcm9vdC5zZXRfY2hpbGQoIWRpciwgc2F2ZS5nZXRfY2hpbGQoZGlyKSk7XHJcbiAgICAgICAgc2F2ZS5zZXRfY2hpbGQoZGlyLCByb290KTtcclxuICAgICAgICByb290LnJlZCA9IHRydWU7XHJcbiAgICAgICAgc2F2ZS5yZWQgPSBmYWxzZTtcclxuICAgICAgICByZXR1cm4gc2F2ZTtcclxuICAgIH07XHJcbiAgICBSQlRyZWUuZG91YmxlX3JvdGF0ZSA9IGZ1bmN0aW9uIChyb290LCBkaXIpIHtcclxuICAgICAgICByb290LnNldF9jaGlsZCghZGlyLCBSQlRyZWUuc2luZ2xlX3JvdGF0ZShyb290LmdldF9jaGlsZCghZGlyKSwgIWRpcikpO1xyXG4gICAgICAgIHJldHVybiBSQlRyZWUuc2luZ2xlX3JvdGF0ZShyb290LCBkaXIpO1xyXG4gICAgfTtcclxuICAgIHJldHVybiBSQlRyZWU7XHJcbn0oVHJlZUJhc2UpKTtcclxuZXhwb3J0cy5SQlRyZWUgPSBSQlRyZWU7XHJcbi8vIyBzb3VyY2VNYXBwaW5nVVJMPWRhdGE6YXBwbGljYXRpb24vanNvbjtiYXNlNjQsZXlKMlpYSnphVzl1SWpvekxDSm1hV3hsSWpvaWNtSjBjbVZsTG1weklpd2ljMjkxY21ObFVtOXZkQ0k2SWlJc0luTnZkWEpqWlhNaU9sc2lMaTR2TGk0dlYyVmlRMjlzWVM5emNtTXZjbUowY21WbExuUnpJbDBzSW01aGJXVnpJanBiWFN3aWJXRndjR2x1WjNNaU9pSTdPenM3T3pzN096czdPenM3T3p0QlFYVkNTVHRKUVVGQk8xRkJORUpKTEdGQlFWRXNSMEZCUnl4VlFVRlZMRWxCUVVrN1dVRkRja0lzU1VGQlNTeEhRVUZITEVkQlFVY3NTVUZCU1N4RFFVRkRMRXRCUVVzc1EwRkJRenRaUVVOeVFpeEpRVUZKTEVsQlFVa3NSMEZCUnl4SlFVRkpMRU5CUVVNc1VVRkJVU3hGUVVGRkxFTkJRVU03V1VGRk0wSXNUMEZCVHl4SFFVRkhMRXRCUVVzc1NVRkJTU3hGUVVGRk8yZENRVU5xUWl4SlFVRkpMRU5CUVVNc1IwRkJSeXhKUVVGSkxFTkJRVU1zVjBGQlZ5eERRVUZETEVsQlFVa3NSVUZCUlN4SFFVRkhMRU5CUVVNc1NVRkJTU3hEUVVGRExFTkJRVU03WjBKQlEzcERMRWxCUVVrc1EwRkJReXhMUVVGTExFTkJRVU1zUlVGQlJUdHZRa0ZEVkN4SlFVRkpMRU5CUVVNc1QwRkJUeXhIUVVGSExFZEJRVWNzUTBGQlF6dHZRa0ZEYmtJc1QwRkJUeXhKUVVGSkxFTkJRVU03YVVKQlEyWTdjVUpCUTBrN2IwSkJRMFFzU1VGQlNTeERRVUZETEZWQlFWVXNRMEZCUXl4SlFVRkpMRU5CUVVNc1IwRkJSeXhEUVVGRExFTkJRVU03YjBKQlF6RkNMRWRCUVVjc1IwRkJSeXhIUVVGSExFTkJRVU1zVTBGQlV5eERRVUZETEVOQlFVTXNSMEZCUnl4RFFVRkRMRU5CUVVNc1EwRkJRenRwUWtGRE9VSTdZVUZEU2p0WlFVVkVMRTlCUVU4c1NVRkJTU3hEUVVGRE8xRkJRMmhDTEVOQlFVTXNRMEZCUXp0SlFTdEdUaXhEUVVGRE8wbEJka2xITEhkQ1FVRkxMRWRCUVV3N1VVRkRTU3hKUVVGSkxFTkJRVU1zUzBGQlN5eEhRVUZITEVsQlFVa3NRMEZCUXp0UlFVTnNRaXhKUVVGSkxFTkJRVU1zU1VGQlNTeEhRVUZITEVOQlFVTXNRMEZCUXp0SlFVTnNRaXhEUVVGRE8wbEJRVUVzUTBGQlF6dEpRVWRHTEhWQ1FVRkpMRWRCUVVvc1ZVRkJTeXhKUVVGSk8xRkJRMHdzU1VGQlNTeEhRVUZITEVkQlFVY3NTVUZCU1N4RFFVRkRMRXRCUVVzc1EwRkJRenRSUVVWeVFpeFBRVUZQTEVkQlFVY3NTMEZCU3l4SlFVRkpMRVZCUVVVN1dVRkRha0lzU1VGQlNTeERRVUZETEVkQlFVY3NTVUZCU1N4RFFVRkRMRmRCUVZjc1EwRkJReXhKUVVGSkxFVkJRVVVzUjBGQlJ5eERRVUZETEVsQlFVa3NRMEZCUXl4RFFVRkRPMWxCUTNwRExFbEJRVWtzUTBGQlF5eExRVUZMTEVOQlFVTXNSVUZCUlR0blFrRkRWQ3hQUVVGUExFZEJRVWNzUTBGQlF5eEpRVUZKTEVOQlFVTTdZVUZEYmtJN2FVSkJRMGs3WjBKQlEwUXNSMEZCUnl4SFFVRkhMRWRCUVVjc1EwRkJReXhUUVVGVExFTkJRVU1zUTBGQlF5eEhRVUZITEVOQlFVTXNRMEZCUXl4RFFVRkRPMkZCUXpsQ08xTkJRMG83VVVGRlJDeFBRVUZQTEVsQlFVa3NRMEZCUXp0SlFVTm9RaXhEUVVGRE8wbEJRVUVzUTBGQlF6dEpRWFZDUml3MlFrRkJWU3hIUVVGV0xGVkJRVmNzU1VGQlNUdFJRVU5ZTEU5QlFVOHNTVUZCU1N4RFFVRkRMRTFCUVUwc1EwRkJReXhKUVVGSkxFVkJRVVVzU1VGQlNTeERRVUZETEZkQlFWY3NRMEZCUXl4RFFVRkRPMGxCUXk5RExFTkJRVU03U1VGQlFTeERRVUZETzBsQlIwWXNOa0pCUVZVc1IwRkJWaXhWUVVGWExFbEJRVWs3VVVGRFdDeEpRVUZKTEVkQlFVY3NSMEZCUnl4SlFVRkpMRU5CUVVNc1YwRkJWeXhEUVVGRE8xRkJSVE5DTEZOQlFWTXNWMEZCVnl4RFFVRkRMRU5CUVVNc1JVRkJSU3hEUVVGRE8xbEJRM0pDTEU5QlFVOHNSMEZCUnl4RFFVRkRMRU5CUVVNc1JVRkJSU3hEUVVGRExFTkJRVU1zUTBGQlF6dFJRVU55UWl4RFFVRkRPMUZCUlVRc1QwRkJUeXhKUVVGSkxFTkJRVU1zVFVGQlRTeERRVUZETEVsQlFVa3NSVUZCUlN4WFFVRlhMRU5CUVVNc1EwRkJRenRKUVVNeFF5eERRVUZETzBsQlFVRXNRMEZCUXp0SlFVZEdMSE5DUVVGSExFZEJRVWc3VVVGRFNTeEpRVUZKTEVkQlFVY3NSMEZCUnl4SlFVRkpMRU5CUVVNc1MwRkJTeXhEUVVGRE8xRkJRM0pDTEVsQlFVa3NSMEZCUnl4TFFVRkxMRWxCUVVrc1JVRkJSVHRaUVVOa0xFOUJRVThzU1VGQlNTeERRVUZETzFOQlEyWTdVVUZGUkN4UFFVRlBMRWRCUVVjc1EwRkJReXhKUVVGSkxFdEJRVXNzU1VGQlNTeEZRVUZGTzFsQlEzUkNMRWRCUVVjc1IwRkJSeXhIUVVGSExFTkJRVU1zU1VGQlNTeERRVUZETzFOQlEyeENPMUZCUlVRc1QwRkJUeXhIUVVGSExFTkJRVU1zU1VGQlNTeERRVUZETzBsQlEzQkNMRU5CUVVNN1NVRkJRU3hEUVVGRE8wbEJSMFlzYzBKQlFVY3NSMEZCU0R0UlFVTkpMRWxCUVVrc1IwRkJSeXhIUVVGSExFbEJRVWtzUTBGQlF5eExRVUZMTEVOQlFVTTdVVUZEY2tJc1NVRkJTU3hIUVVGSExFdEJRVXNzU1VGQlNTeEZRVUZGTzFsQlEyUXNUMEZCVHl4SlFVRkpMRU5CUVVNN1UwRkRaanRSUVVWRUxFOUJRVThzUjBGQlJ5eERRVUZETEV0QlFVc3NTMEZCU3l4SlFVRkpMRVZCUVVVN1dVRkRka0lzUjBGQlJ5eEhRVUZITEVkQlFVY3NRMEZCUXl4TFFVRkxMRU5CUVVNN1UwRkRia0k3VVVGRlJDeFBRVUZQTEVkQlFVY3NRMEZCUXl4SlFVRkpMRU5CUVVNN1NVRkRjRUlzUTBGQlF6dEpRVUZCTEVOQlFVTTdTVUZKUml3eVFrRkJVU3hIUVVGU08xRkJRMGtzVDBGQlR5eEpRVUZKTEZGQlFWRXNRMEZCUXl4SlFVRkpMRU5CUVVNc1EwRkJRenRKUVVNNVFpeERRVUZETzBsQlFVRXNRMEZCUXp0SlFVZEdMSFZDUVVGSkxFZEJRVW9zVlVGQlN5eEZRVUZGTzFGQlEwZ3NTVUZCU1N4RlFVRkZMRWRCUVVjc1NVRkJTU3hEUVVGRExGRkJRVkVzUlVGQlJTeEZRVUZGTEVsQlFVa3NRMEZCUXp0UlFVTXZRaXhQUVVGUExFTkJRVU1zU1VGQlNTeEhRVUZITEVWQlFVVXNRMEZCUXl4SlFVRkpMRVZCUVVVc1EwRkJReXhMUVVGTExFbEJRVWtzUlVGQlJUdFpRVU5vUXl4RlFVRkZMRU5CUVVNc1NVRkJTU3hEUVVGRExFTkJRVU03VTBGRFdqdEpRVU5NTEVOQlFVTTdTVUZCUVN4RFFVRkRPMGxCUjBZc2QwSkJRVXNzUjBGQlRDeFZRVUZOTEVWQlFVVTdVVUZEU2l4SlFVRkpMRVZCUVVVc1IwRkJSeXhKUVVGSkxFTkJRVU1zVVVGQlVTeEZRVUZGTEVWQlFVVXNTVUZCU1N4RFFVRkRPMUZCUXk5Q0xFOUJRVThzUTBGQlF5eEpRVUZKTEVkQlFVY3NSVUZCUlN4RFFVRkRMRWxCUVVrc1JVRkJSU3hEUVVGRExFdEJRVXNzU1VGQlNTeEZRVUZGTzFsQlEyaERMRVZCUVVVc1EwRkJReXhKUVVGSkxFTkJRVU1zUTBGQlF6dFRRVU5hTzBsQlEwd3NRMEZCUXp0SlFVRkJMRU5CUVVNN1NVRkhSaXg1UWtGQlRTeEhRVUZPTEZWQlFVOHNTVUZCU1N4RlFVRkZMRWRCUVVjN1VVRkRXaXhKUVVGSkxFZEJRVWNzUjBGQlJ5eEpRVUZKTEVOQlFVTXNTMEZCU3l4RFFVRkRPMUZCUTNKQ0xFbEJRVWtzU1VGQlNTeEhRVUZITEVsQlFVa3NRMEZCUXl4UlFVRlJMRVZCUVVVc1EwRkJRenRSUVVVelFpeFBRVUZQTEVkQlFVY3NTMEZCU3l4SlFVRkpMRVZCUVVVN1dVRkRha0lzU1VGQlNTeERRVUZETEVkQlFVY3NTVUZCU1N4RFFVRkRMRmRCUVZjc1EwRkJReXhKUVVGSkxFVkJRVVVzUjBGQlJ5eERRVUZETEVsQlFVa3NRMEZCUXl4RFFVRkRPMWxCUTNwRExFbEJRVWtzUTBGQlF5eExRVUZMTEVOQlFVTXNSVUZCUlR0blFrRkRWQ3hKUVVGSkxFTkJRVU1zVDBGQlR5eEhRVUZITEVkQlFVY3NRMEZCUXp0blFrRkRia0lzVDBGQlR5eEpRVUZKTEVOQlFVTTdZVUZEWmp0WlFVTkVMRWxCUVVrc1EwRkJReXhWUVVGVkxFTkJRVU1zU1VGQlNTeERRVUZETEVkQlFVY3NRMEZCUXl4RFFVRkRPMWxCUXpGQ0xFZEJRVWNzUjBGQlJ5eEhRVUZITEVOQlFVTXNVMEZCVXl4RFFVRkRMRU5CUVVNc1IwRkJSeXhEUVVGRExFTkJRVU1zUTBGQlF6dFRRVU01UWp0UlFVVkVMRXRCUVVzc1NVRkJTU3hEUVVGRExFZEJRVWNzU1VGQlNTeERRVUZETEZWQlFWVXNRMEZCUXl4TlFVRk5MRWRCUVVjc1EwRkJReXhGUVVGRkxFTkJRVU1zU1VGQlNTeERRVUZETEVWQlFVVXNSVUZCUlN4RFFVRkRMRVZCUVVVN1dVRkRiRVFzUjBGQlJ5eEhRVUZITEVsQlFVa3NRMEZCUXl4VlFVRlZMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU03V1VGRGVrSXNTVUZCU1N4SFFVRkhMRU5CUVVNc1NVRkJTU3hGUVVGRkxFZEJRVWNzUTBGQlF5eEpRVUZKTEVOQlFVTXNSMEZCUnl4RFFVRkRMRVZCUVVVN1owSkJRM3BDTEVsQlFVa3NRMEZCUXl4UFFVRlBMRWRCUVVjc1IwRkJSeXhEUVVGRE8yZENRVU51UWl4SlFVRkpMRU5CUVVNc1ZVRkJWU3hEUVVGRExFMUJRVTBzUjBGQlJ5eERRVUZETEVOQlFVTTdaMEpCUXpOQ0xFOUJRVThzU1VGQlNTeERRVUZETzJGQlEyWTdVMEZEU2p0UlFVVkVMRWxCUVVrc1EwRkJReXhWUVVGVkxFTkJRVU1zVFVGQlRTeEhRVUZITEVOQlFVTXNRMEZCUXp0UlFVTXpRaXhQUVVGUExFbEJRVWtzUTBGQlF6dEpRVU5vUWl4RFFVRkRPMGxCUVVFc1EwRkJRenRKUVVOT0xHVkJRVU03UVVGQlJDeERRVUZETEVGQk5VbEVMRWxCTkVsRE8wRkJOVWxaTERSQ1FVRlJPMEZCTmtseVFqdEpRVWxKTEd0Q1FVRlpMRWxCUVVrN1VVRkRXaXhKUVVGSkxFTkJRVU1zUzBGQlN5eEhRVUZITEVsQlFVa3NRMEZCUXp0UlFVTnNRaXhKUVVGSkxFTkJRVU1zVlVGQlZTeEhRVUZITEVWQlFVVXNRMEZCUXp0UlFVTnlRaXhKUVVGSkxFTkJRVU1zVDBGQlR5eEhRVUZITEVsQlFVa3NRMEZCUXp0SlFVTjRRaXhEUVVGRE8wbEJSVVFzZFVKQlFVa3NSMEZCU2p0UlFVTkpMRTlCUVU4c1NVRkJTU3hEUVVGRExFOUJRVThzUzBGQlN5eEpRVUZKTEVOQlFVTXNRMEZCUXl4RFFVRkRMRWxCUVVrc1EwRkJReXhQUVVGUExFTkJRVU1zU1VGQlNTeERRVUZETEVOQlFVTXNRMEZCUXl4SlFVRkpMRU5CUVVNN1NVRkROVVFzUTBGQlF6dEpRVUZCTEVOQlFVTTdTVUZKUml4MVFrRkJTU3hIUVVGS08xRkJRMGtzU1VGQlNTeEpRVUZKTEVOQlFVTXNUMEZCVHl4TFFVRkxMRWxCUVVrc1JVRkJSVHRaUVVOMlFpeEpRVUZKTEVsQlFVa3NSMEZCUnl4SlFVRkpMRU5CUVVNc1MwRkJTeXhEUVVGRExFdEJRVXNzUTBGQlF6dFpRVU0xUWl4SlFVRkpMRWxCUVVrc1MwRkJTeXhKUVVGSkxFVkJRVVU3WjBKQlEyWXNTVUZCU1N4RFFVRkRMRkZCUVZFc1EwRkJReXhKUVVGSkxFTkJRVU1zUTBGQlF6dGhRVU4yUWp0VFFVTktPMkZCUTBrN1dVRkRSQ3hKUVVGSkxFbEJRVWtzUTBGQlF5eFBRVUZQTEVOQlFVTXNTMEZCU3l4TFFVRkxMRWxCUVVrc1JVRkJSVHRuUWtGSE4wSXNTVUZCU1N4SlFVRkpMRU5CUVVNN1owSkJRMVFzUjBGQlJ6dHZRa0ZEUXl4SlFVRkpMRWRCUVVjc1NVRkJTU3hEUVVGRExFOUJRVThzUTBGQlF6dHZRa0ZEY0VJc1NVRkJTU3hKUVVGSkxFTkJRVU1zVlVGQlZTeERRVUZETEUxQlFVMHNSVUZCUlR0M1FrRkRlRUlzU1VGQlNTeERRVUZETEU5QlFVOHNSMEZCUnl4SlFVRkpMRU5CUVVNc1ZVRkJWU3hEUVVGRExFZEJRVWNzUlVGQlJTeERRVUZETzNGQ1FVTjRRenQ1UWtGRFNUdDNRa0ZEUkN4SlFVRkpMRU5CUVVNc1QwRkJUeXhIUVVGSExFbEJRVWtzUTBGQlF6dDNRa0ZEY0VJc1RVRkJUVHR4UWtGRFZEdHBRa0ZEU2l4UlFVRlJMRWxCUVVrc1EwRkJReXhQUVVGUExFTkJRVU1zUzBGQlN5eExRVUZMTEVsQlFVa3NSVUZCUlR0aFFVTjZRenRwUWtGRFNUdG5Ra0ZGUkN4SlFVRkpMRU5CUVVNc1ZVRkJWU3hEUVVGRExFbEJRVWtzUTBGQlF5eEpRVUZKTEVOQlFVTXNUMEZCVHl4RFFVRkRMRU5CUVVNN1owSkJRMjVETEVsQlFVa3NRMEZCUXl4UlFVRlJMRU5CUVVNc1NVRkJTU3hEUVVGRExFOUJRVThzUTBGQlF5eExRVUZMTEVOQlFVTXNRMEZCUXp0aFFVTnlRenRUUVVOS08xRkJRMFFzVDBGQlR5eEpRVUZKTEVOQlFVTXNUMEZCVHl4TFFVRkxMRWxCUVVrc1EwRkJReXhEUVVGRExFTkJRVU1zU1VGQlNTeERRVUZETEU5QlFVOHNRMEZCUXl4SlFVRkpMRU5CUVVNc1EwRkJReXhEUVVGRExFbEJRVWtzUTBGQlF6dEpRVU0xUkN4RFFVRkRPMGxCUVVFc1EwRkJRenRKUVVsR0xIVkNRVUZKTEVkQlFVbzdVVUZEU1N4SlFVRkpMRWxCUVVrc1EwRkJReXhQUVVGUExFdEJRVXNzU1VGQlNTeEZRVUZGTzFsQlEzWkNMRWxCUVVrc1NVRkJTU3hIUVVGSExFbEJRVWtzUTBGQlF5eExRVUZMTEVOQlFVTXNTMEZCU3l4RFFVRkRPMWxCUXpWQ0xFbEJRVWtzU1VGQlNTeExRVUZMTEVsQlFVa3NSVUZCUlR0blFrRkRaaXhKUVVGSkxFTkJRVU1zVVVGQlVTeERRVUZETEVsQlFVa3NRMEZCUXl4RFFVRkRPMkZCUTNaQ08xTkJRMG83WVVGRFNUdFpRVU5FTEVsQlFVa3NTVUZCU1N4RFFVRkRMRTlCUVU4c1EwRkJReXhKUVVGSkxFdEJRVXNzU1VGQlNTeEZRVUZGTzJkQ1FVTTFRaXhKUVVGSkxFbEJRVWtzUTBGQlF6dG5Ra0ZEVkN4SFFVRkhPMjlDUVVORExFbEJRVWtzUjBGQlJ5eEpRVUZKTEVOQlFVTXNUMEZCVHl4RFFVRkRPMjlDUVVOd1FpeEpRVUZKTEVsQlFVa3NRMEZCUXl4VlFVRlZMRU5CUVVNc1RVRkJUU3hGUVVGRk8zZENRVU40UWl4SlFVRkpMRU5CUVVNc1QwRkJUeXhIUVVGSExFbEJRVWtzUTBGQlF5eFZRVUZWTEVOQlFVTXNSMEZCUnl4RlFVRkZMRU5CUVVNN2NVSkJRM2hETzNsQ1FVTkpPM2RDUVVORUxFbEJRVWtzUTBGQlF5eFBRVUZQTEVkQlFVY3NTVUZCU1N4RFFVRkRPM2RDUVVOd1FpeE5RVUZOTzNGQ1FVTlVPMmxDUVVOS0xGRkJRVkVzU1VGQlNTeERRVUZETEU5QlFVOHNRMEZCUXl4SlFVRkpMRXRCUVVzc1NVRkJTU3hGUVVGRk8yRkJRM2hETzJsQ1FVTkpPMmRDUVVORUxFbEJRVWtzUTBGQlF5eFZRVUZWTEVOQlFVTXNTVUZCU1N4RFFVRkRMRWxCUVVrc1EwRkJReXhQUVVGUExFTkJRVU1zUTBGQlF6dG5Ra0ZEYmtNc1NVRkJTU3hEUVVGRExGRkJRVkVzUTBGQlF5eEpRVUZKTEVOQlFVTXNUMEZCVHl4RFFVRkRMRWxCUVVrc1EwRkJReXhEUVVGRE8yRkJRM0JETzFOQlEwbzdVVUZEUkN4UFFVRlBMRWxCUVVrc1EwRkJReXhQUVVGUExFdEJRVXNzU1VGQlNTeERRVUZETEVOQlFVTXNRMEZCUXl4SlFVRkpMRU5CUVVNc1QwRkJUeXhEUVVGRExFbEJRVWtzUTBGQlF5eERRVUZETEVOQlFVTXNTVUZCU1N4RFFVRkRPMGxCUXpWRUxFTkJRVU03U1VGQlFTeERRVUZETzBsQlJVWXNNa0pCUVZFc1IwRkJVaXhWUVVGVExFdEJRVXM3VVVGRFZpeFBRVUZQTEV0QlFVc3NRMEZCUXl4SlFVRkpMRXRCUVVzc1NVRkJTU3hGUVVGRk8xbEJRM2hDTEVsQlFVa3NRMEZCUXl4VlFVRlZMRU5CUVVNc1NVRkJTU3hEUVVGRExFdEJRVXNzUTBGQlF5eERRVUZETzFsQlF6VkNMRXRCUVVzc1IwRkJSeXhMUVVGTExFTkJRVU1zU1VGQlNTeERRVUZETzFOQlEzUkNPMUZCUTBRc1NVRkJTU3hEUVVGRExFOUJRVThzUjBGQlJ5eExRVUZMTEVOQlFVTTdTVUZEZWtJc1EwRkJRenRKUVVGQkxFTkJRVU03U1VGRlJpd3lRa0ZCVVN4SFFVRlNMRlZCUVZNc1MwRkJTenRSUVVOV0xFOUJRVThzUzBGQlN5eERRVUZETEV0QlFVc3NTMEZCU3l4SlFVRkpMRVZCUVVVN1dVRkRla0lzU1VGQlNTeERRVUZETEZWQlFWVXNRMEZCUXl4SlFVRkpMRU5CUVVNc1MwRkJTeXhEUVVGRExFTkJRVU03V1VGRE5VSXNTMEZCU3l4SFFVRkhMRXRCUVVzc1EwRkJReXhMUVVGTExFTkJRVU03VTBGRGRrSTdVVUZEUkN4SlFVRkpMRU5CUVVNc1QwRkJUeXhIUVVGSExFdEJRVXNzUTBGQlF6dEpRVU42UWl4RFFVRkRPMGxCUVVFc1EwRkJRenRKUVVOT0xHVkJRVU03UVVGQlJDeERRVUZETEVGQk9VWkVMRWxCT0VaRE8wRkJPVVpaTERSQ1FVRlJPMEZCWjBkeVFqdEpRVXRKTEdOQlFWa3NTVUZCU1R0UlFVTmFMRWxCUVVrc1EwRkJReXhKUVVGSkxFZEJRVWNzU1VGQlNTeERRVUZETzFGQlEycENMRWxCUVVrc1EwRkJReXhKUVVGSkxFZEJRVWNzU1VGQlNTeERRVUZETzFGQlEycENMRWxCUVVrc1EwRkJReXhMUVVGTExFZEJRVWNzU1VGQlNTeERRVUZETzFGQlEyeENMRWxCUVVrc1EwRkJReXhIUVVGSExFZEJRVWNzU1VGQlNTeERRVUZETzBsQlEzQkNMRU5CUVVNN1NVRkZSQ3gzUWtGQlV5eEhRVUZVTEZWQlFWVXNSMEZCUnp0UlFVTlVMRTlCUVU4c1IwRkJSeXhEUVVGRExFTkJRVU1zUTBGQlF5eEpRVUZKTEVOQlFVTXNTMEZCU3l4RFFVRkRMRU5CUVVNc1EwRkJReXhKUVVGSkxFTkJRVU1zU1VGQlNTeERRVUZETzBsQlEzaERMRU5CUVVNN1NVRkJRU3hEUVVGRE8wbEJSVVlzZDBKQlFWTXNSMEZCVkN4VlFVRlZMRWRCUVVjc1JVRkJSU3hIUVVGSE8xRkJRMlFzU1VGQlNTeEhRVUZITEVWQlFVVTdXVUZEVEN4SlFVRkpMRU5CUVVNc1MwRkJTeXhIUVVGSExFZEJRVWNzUTBGQlF6dFRRVU53UWp0aFFVTkpPMWxCUTBRc1NVRkJTU3hEUVVGRExFbEJRVWtzUjBGQlJ5eEhRVUZITEVOQlFVTTdVMEZEYmtJN1NVRkRUQ3hEUVVGRE8wbEJRVUVzUTBGQlF6dEpRVU5PTEZkQlFVTTdRVUZCUkN4RFFVRkRMRUZCZUVKRUxFbEJkMEpETzBGQlJVUTdTVUZCSzBJc01FSkJRVkU3U1VGTGJrTXNaMEpCUVZrc1ZVRkJhME03VVVGQk9VTXNXVUZEU1N4cFFrRkJUeXhUUVVsV08xRkJTRWNzUzBGQlNTeERRVUZETEV0QlFVc3NSMEZCUnl4SlFVRkpMRU5CUVVNN1VVRkRiRUlzUzBGQlNTeERRVUZETEZkQlFWY3NSMEZCUnl4VlFVRlZMRU5CUVVNN1VVRkRPVUlzUzBGQlNTeERRVUZETEVsQlFVa3NSMEZCUnl4RFFVRkRMRU5CUVVNN08wbEJRMnhDTEVOQlFVTTdTVUZIUkN4MVFrRkJUU3hIUVVGT0xGVkJRVThzU1VGQlNUdFJRVU5RTEVsQlFVa3NSMEZCUnl4SFFVRkhMRXRCUVVzc1EwRkJRenRSUVVWb1FpeEpRVUZKTEVsQlFVa3NRMEZCUXl4TFFVRkxMRXRCUVVzc1NVRkJTU3hGUVVGRk8xbEJSWEpDTEVsQlFVa3NRMEZCUXl4TFFVRkxMRWRCUVVjc1NVRkJTU3hKUVVGSkxFTkJRVU1zU1VGQlNTeERRVUZETEVOQlFVTTdXVUZETlVJc1IwRkJSeXhIUVVGSExFbEJRVWtzUTBGQlF6dFpRVU5ZTEVsQlFVa3NRMEZCUXl4SlFVRkpMRVZCUVVVc1EwRkJRenRUUVVObU8yRkJRMGs3V1VGRFJDeEpRVUZKTEVsQlFVa3NSMEZCUnl4SlFVRkpMRWxCUVVrc1EwRkJReXhUUVVGVExFTkJRVU1zUTBGQlF6dFpRVVV2UWl4SlFVRkpMRWRCUVVjc1IwRkJSeXhMUVVGTExFTkJRVU03V1VGRGFFSXNTVUZCU1N4SlFVRkpMRWRCUVVjc1MwRkJTeXhEUVVGRE8xbEJSMnBDTEVsQlFVa3NSVUZCUlN4SFFVRkhMRWxCUVVrc1EwRkJRenRaUVVOa0xFbEJRVWtzUjBGQlJ5eEhRVUZITEVsQlFVa3NRMEZCUXp0WlFVTm1MRWxCUVVrc1EwRkJReXhIUVVGSExFbEJRVWtzUTBGQlF6dFpRVU5pTEVsQlFVa3NTVUZCU1N4SFFVRkhMRWxCUVVrc1EwRkJReXhMUVVGTExFTkJRVU03V1VGRGRFSXNSMEZCUnl4RFFVRkRMRXRCUVVzc1IwRkJSeXhKUVVGSkxFTkJRVU1zUzBGQlN5eERRVUZETzFsQlIzWkNMRTlCUVU4c1NVRkJTU3hGUVVGRk8yZENRVU5VTEVsQlFVa3NTVUZCU1N4TFFVRkxMRWxCUVVrc1JVRkJSVHR2UWtGRlppeEpRVUZKTEVkQlFVY3NTVUZCU1N4SlFVRkpMRU5CUVVNc1NVRkJTU3hEUVVGRExFTkJRVU03YjBKQlEzUkNMRU5CUVVNc1EwRkJReXhUUVVGVExFTkJRVU1zUjBGQlJ5eEZRVUZGTEVsQlFVa3NRMEZCUXl4RFFVRkRPMjlDUVVOMlFpeEhRVUZITEVkQlFVY3NTVUZCU1N4RFFVRkRPMjlDUVVOWUxFbEJRVWtzUTBGQlF5eEpRVUZKTEVWQlFVVXNRMEZCUXp0cFFrRkRaanR4UWtGRFNTeEpRVUZKTEUxQlFVMHNRMEZCUXl4TlFVRk5MRU5CUVVNc1NVRkJTU3hEUVVGRExFbEJRVWtzUTBGQlF5eEpRVUZKTEUxQlFVMHNRMEZCUXl4TlFVRk5MRU5CUVVNc1NVRkJTU3hEUVVGRExFdEJRVXNzUTBGQlF5eEZRVUZGTzI5Q1FVVTFSQ3hKUVVGSkxFTkJRVU1zUjBGQlJ5eEhRVUZITEVsQlFVa3NRMEZCUXp0dlFrRkRhRUlzU1VGQlNTeERRVUZETEVsQlFVa3NRMEZCUXl4SFFVRkhMRWRCUVVjc1MwRkJTeXhEUVVGRE8yOUNRVU4wUWl4SlFVRkpMRU5CUVVNc1MwRkJTeXhEUVVGRExFZEJRVWNzUjBGQlJ5eExRVUZMTEVOQlFVTTdhVUpCUXpGQ08yZENRVWRFTEVsQlFVa3NUVUZCVFN4RFFVRkRMRTFCUVUwc1EwRkJReXhKUVVGSkxFTkJRVU1zU1VGQlNTeE5RVUZOTEVOQlFVTXNUVUZCVFN4RFFVRkRMRU5CUVVNc1EwRkJReXhGUVVGRk8yOUNRVU42UXl4SlFVRkpMRWxCUVVrc1IwRkJSeXhIUVVGSExFTkJRVU1zUzBGQlN5eExRVUZMTEVWQlFVVXNRMEZCUXp0dlFrRkZOVUlzU1VGQlNTeEpRVUZKTEV0QlFVc3NRMEZCUXl4RFFVRkRMRk5CUVZNc1EwRkJReXhKUVVGSkxFTkJRVU1zUlVGQlJUdDNRa0ZETlVJc1IwRkJSeXhEUVVGRExGTkJRVk1zUTBGQlF5eEpRVUZKTEVWQlFVVXNUVUZCVFN4RFFVRkRMR0ZCUVdFc1EwRkJReXhGUVVGRkxFVkJRVVVzUTBGQlF5eEpRVUZKTEVOQlFVTXNRMEZCUXl4RFFVRkRPM0ZDUVVONFJEdDVRa0ZEU1R0M1FrRkRSQ3hIUVVGSExFTkJRVU1zVTBGQlV5eERRVUZETEVsQlFVa3NSVUZCUlN4TlFVRk5MRU5CUVVNc1lVRkJZU3hEUVVGRExFVkJRVVVzUlVGQlJTeERRVUZETEVsQlFVa3NRMEZCUXl4RFFVRkRMRU5CUVVNN2NVSkJRM2hFTzJsQ1FVTktPMmRDUVVWRUxFbEJRVWtzUjBGQlJ5eEhRVUZITEVsQlFVa3NRMEZCUXl4WFFVRlhMRU5CUVVNc1NVRkJTU3hEUVVGRExFbEJRVWtzUlVGQlJTeEpRVUZKTEVOQlFVTXNRMEZCUXp0blFrRkhOVU1zU1VGQlNTeEhRVUZITEV0QlFVc3NRMEZCUXl4RlFVRkZPMjlDUVVOWUxFMUJRVTA3YVVKQlExUTdaMEpCUlVRc1NVRkJTU3hIUVVGSExFZEJRVWNzUTBGQlF6dG5Ra0ZEV0N4SFFVRkhMRWRCUVVjc1IwRkJSeXhIUVVGSExFTkJRVU1zUTBGQlF6dG5Ra0ZIWkN4SlFVRkpMRVZCUVVVc1MwRkJTeXhKUVVGSkxFVkJRVVU3YjBKQlEySXNSMEZCUnl4SFFVRkhMRVZCUVVVc1EwRkJRenRwUWtGRFdqdG5Ra0ZEUkN4RlFVRkZMRWRCUVVjc1EwRkJReXhEUVVGRE8yZENRVU5RTEVOQlFVTXNSMEZCUnl4SlFVRkpMRU5CUVVNN1owSkJRMVFzU1VGQlNTeEhRVUZITEVsQlFVa3NRMEZCUXl4VFFVRlRMRU5CUVVNc1IwRkJSeXhEUVVGRExFTkJRVU03WVVGRE9VSTdXVUZIUkN4SlFVRkpMRU5CUVVNc1MwRkJTeXhIUVVGSExFbEJRVWtzUTBGQlF5eExRVUZMTEVOQlFVTTdVMEZETTBJN1VVRkhSQ3hKUVVGSkxFTkJRVU1zUzBGQlN5eERRVUZETEVkQlFVY3NSMEZCUnl4TFFVRkxMRU5CUVVNN1VVRkZka0lzVDBGQlR5eEhRVUZITEVOQlFVTTdTVUZEWml4RFFVRkRPMGxCUVVFc1EwRkJRenRKUVVkR0xIVkNRVUZOTEVkQlFVNHNWVUZCVHl4SlFVRkpPMUZCUTFBc1NVRkJTU3hKUVVGSkxFTkJRVU1zUzBGQlN5eExRVUZMTEVsQlFVa3NSVUZCUlR0WlFVTnlRaXhQUVVGUExFdEJRVXNzUTBGQlF6dFRRVU5vUWp0UlFVVkVMRWxCUVVrc1NVRkJTU3hIUVVGSExFbEJRVWtzU1VGQlNTeERRVUZETEZOQlFWTXNRMEZCUXl4RFFVRkRPMUZCUXk5Q0xFbEJRVWtzU1VGQlNTeEhRVUZITEVsQlFVa3NRMEZCUXp0UlFVTm9RaXhKUVVGSkxFTkJRVU1zUzBGQlN5eEhRVUZITEVsQlFVa3NRMEZCUXl4TFFVRkxMRU5CUVVNN1VVRkRlRUlzU1VGQlNTeERRVUZETEVkQlFVY3NTVUZCU1N4RFFVRkRPMUZCUTJJc1NVRkJTU3hGUVVGRkxFZEJRVWNzU1VGQlNTeERRVUZETzFGQlEyUXNTVUZCU1N4TFFVRkxMRWRCUVVjc1NVRkJTU3hEUVVGRE8xRkJRMnBDTEVsQlFVa3NSMEZCUnl4SFFVRkhMRWxCUVVrc1EwRkJRenRSUVVWbUxFOUJRVThzU1VGQlNTeERRVUZETEZOQlFWTXNRMEZCUXl4SFFVRkhMRU5CUVVNc1MwRkJTeXhKUVVGSkxFVkJRVVU3V1VGRGFrTXNTVUZCU1N4SlFVRkpMRWRCUVVjc1IwRkJSeXhEUVVGRE8xbEJSMllzUlVGQlJTeEhRVUZITEVOQlFVTXNRMEZCUXp0WlFVTlFMRU5CUVVNc1IwRkJSeXhKUVVGSkxFTkJRVU03V1VGRFZDeEpRVUZKTEVkQlFVY3NTVUZCU1N4RFFVRkRMRk5CUVZNc1EwRkJReXhIUVVGSExFTkJRVU1zUTBGQlF6dFpRVVV6UWl4SlFVRkpMRWRCUVVjc1IwRkJSeXhKUVVGSkxFTkJRVU1zVjBGQlZ5eERRVUZETEVsQlFVa3NSVUZCUlN4SlFVRkpMRU5CUVVNc1NVRkJTU3hEUVVGRExFTkJRVU03V1VGRk5VTXNSMEZCUnl4SFFVRkhMRWRCUVVjc1IwRkJSeXhEUVVGRExFTkJRVU03V1VGSFpDeEpRVUZKTEVkQlFVY3NTMEZCU3l4RFFVRkRMRVZCUVVVN1owSkJRMWdzUzBGQlN5eEhRVUZITEVsQlFVa3NRMEZCUXp0aFFVTm9RanRaUVVkRUxFbEJRVWtzUTBGQlF5eE5RVUZOTEVOQlFVTXNUVUZCVFN4RFFVRkRMRWxCUVVrc1EwRkJReXhKUVVGSkxFTkJRVU1zVFVGQlRTeERRVUZETEUxQlFVMHNRMEZCUXl4SlFVRkpMRU5CUVVNc1UwRkJVeXhEUVVGRExFZEJRVWNzUTBGQlF5eERRVUZETEVWQlFVVTdaMEpCUXpkRUxFbEJRVWtzVFVGQlRTeERRVUZETEUxQlFVMHNRMEZCUXl4SlFVRkpMRU5CUVVNc1UwRkJVeXhEUVVGRExFTkJRVU1zUjBGQlJ5eERRVUZETEVOQlFVTXNSVUZCUlR0dlFrRkRja01zU1VGQlNTeEZRVUZGTEVkQlFVY3NUVUZCVFN4RFFVRkRMR0ZCUVdFc1EwRkJReXhKUVVGSkxFVkJRVVVzUjBGQlJ5eERRVUZETEVOQlFVTTdiMEpCUTNwRExFTkJRVU1zUTBGQlF5eFRRVUZUTEVOQlFVTXNTVUZCU1N4RlFVRkZMRVZCUVVVc1EwRkJReXhEUVVGRE8yOUNRVU4wUWl4RFFVRkRMRWRCUVVjc1JVRkJSU3hEUVVGRE8ybENRVU5XTzNGQ1FVTkpMRWxCUVVrc1EwRkJReXhOUVVGTkxFTkJRVU1zVFVGQlRTeERRVUZETEVsQlFVa3NRMEZCUXl4VFFVRlRMRU5CUVVNc1EwRkJReXhIUVVGSExFTkJRVU1zUTBGQlF5eEZRVUZGTzI5Q1FVTXpReXhKUVVGSkxFOUJRVThzUjBGQlJ5eERRVUZETEVOQlFVTXNVMEZCVXl4RFFVRkRMRU5CUVVNc1NVRkJTU3hEUVVGRExFTkJRVU03YjBKQlEycERMRWxCUVVrc1QwRkJUeXhMUVVGTExFbEJRVWtzUlVGQlJUdDNRa0ZEYkVJc1NVRkJTU3hEUVVGRExFMUJRVTBzUTBGQlF5eE5RVUZOTEVOQlFVTXNUMEZCVHl4RFFVRkRMRk5CUVZNc1EwRkJReXhEUVVGRExFbEJRVWtzUTBGQlF5eERRVUZETEVsQlFVa3NRMEZCUXl4TlFVRk5MRU5CUVVNc1RVRkJUU3hEUVVGRExFOUJRVThzUTBGQlF5eFRRVUZUTEVOQlFVTXNTVUZCU1N4RFFVRkRMRU5CUVVNc1JVRkJSVHMwUWtGRmNrWXNRMEZCUXl4RFFVRkRMRWRCUVVjc1IwRkJSeXhMUVVGTExFTkJRVU03TkVKQlEyUXNUMEZCVHl4RFFVRkRMRWRCUVVjc1IwRkJSeXhKUVVGSkxFTkJRVU03TkVKQlEyNUNMRWxCUVVrc1EwRkJReXhIUVVGSExFZEJRVWNzU1VGQlNTeERRVUZETzNsQ1FVTnVRanMyUWtGRFNUczBRa0ZEUkN4SlFVRkpMRWxCUVVrc1IwRkJSeXhGUVVGRkxFTkJRVU1zUzBGQlN5eExRVUZMTEVOQlFVTXNRMEZCUXpzMFFrRkZNVUlzU1VGQlNTeE5RVUZOTEVOQlFVTXNUVUZCVFN4RFFVRkRMRTlCUVU4c1EwRkJReXhUUVVGVExFTkJRVU1zU1VGQlNTeERRVUZETEVOQlFVTXNSVUZCUlR0blEwRkRlRU1zUlVGQlJTeERRVUZETEZOQlFWTXNRMEZCUXl4SlFVRkpMRVZCUVVVc1RVRkJUU3hEUVVGRExHRkJRV0VzUTBGQlF5eERRVUZETEVWQlFVVXNTVUZCU1N4RFFVRkRMRU5CUVVNc1EwRkJRenMyUWtGRGNrUTdhVU5CUTBrc1NVRkJTU3hOUVVGTkxFTkJRVU1zVFVGQlRTeERRVUZETEU5QlFVOHNRMEZCUXl4VFFVRlRMRU5CUVVNc1EwRkJReXhKUVVGSkxFTkJRVU1zUTBGQlF5eEZRVUZGTzJkRFFVTTVReXhGUVVGRkxFTkJRVU1zVTBGQlV5eERRVUZETEVsQlFVa3NSVUZCUlN4TlFVRk5MRU5CUVVNc1lVRkJZU3hEUVVGRExFTkJRVU1zUlVGQlJTeEpRVUZKTEVOQlFVTXNRMEZCUXl4RFFVRkRPelpDUVVOeVJEczBRa0ZIUkN4SlFVRkpMRWRCUVVjc1IwRkJSeXhGUVVGRkxFTkJRVU1zVTBGQlV5eERRVUZETEVsQlFVa3NRMEZCUXl4RFFVRkRPelJDUVVNM1FpeEhRVUZITEVOQlFVTXNSMEZCUnl4SFFVRkhMRWxCUVVrc1EwRkJRenMwUWtGRFppeEpRVUZKTEVOQlFVTXNSMEZCUnl4SFFVRkhMRWxCUVVrc1EwRkJRenMwUWtGRGFFSXNSMEZCUnl4RFFVRkRMRWxCUVVrc1EwRkJReXhIUVVGSExFZEJRVWNzUzBGQlN5eERRVUZET3pSQ1FVTnlRaXhIUVVGSExFTkJRVU1zUzBGQlN5eERRVUZETEVkQlFVY3NSMEZCUnl4TFFVRkxMRU5CUVVNN2VVSkJRM3BDTzNGQ1FVTktPMmxDUVVOS08yRkJRMG83VTBGRFNqdFJRVWRFTEVsQlFVa3NTMEZCU3l4TFFVRkxMRWxCUVVrc1JVRkJSVHRaUVVOb1FpeExRVUZMTEVOQlFVTXNTVUZCU1N4SFFVRkhMRWxCUVVrc1EwRkJReXhKUVVGSkxFTkJRVU03V1VGRGRrSXNRMEZCUXl4RFFVRkRMRk5CUVZNc1EwRkJReXhEUVVGRExFTkJRVU1zUzBGQlN5eExRVUZMTEVsQlFVa3NSVUZCUlN4SlFVRkpMRU5CUVVNc1UwRkJVeXhEUVVGRExFbEJRVWtzUTBGQlF5eEpRVUZKTEV0QlFVc3NTVUZCU1N4RFFVRkRMRU5CUVVNc1EwRkJRenRaUVVOc1JTeEpRVUZKTEVOQlFVTXNTVUZCU1N4RlFVRkZMRU5CUVVNN1UwRkRaanRSUVVkRUxFbEJRVWtzUTBGQlF5eExRVUZMTEVkQlFVY3NTVUZCU1N4RFFVRkRMRXRCUVVzc1EwRkJRenRSUVVONFFpeEpRVUZKTEVsQlFVa3NRMEZCUXl4TFFVRkxMRXRCUVVzc1NVRkJTU3hGUVVGRk8xbEJRM0pDTEVsQlFVa3NRMEZCUXl4TFFVRkxMRU5CUVVNc1IwRkJSeXhIUVVGSExFdEJRVXNzUTBGQlF6dFRRVU14UWp0UlFVVkVMRTlCUVU4c1MwRkJTeXhMUVVGTExFbEJRVWtzUTBGQlF6dEpRVU14UWl4RFFVRkRPMGxCUVVFc1EwRkJRenRKUVVWTExHRkJRVTBzUjBGQllpeFZRVUZqTEVsQlFVazdVVUZEWkN4UFFVRlBMRWxCUVVrc1MwRkJTeXhKUVVGSkxFbEJRVWtzU1VGQlNTeERRVUZETEVkQlFVY3NRMEZCUXp0SlFVTnlReXhEUVVGRE8wbEJSVTBzYjBKQlFXRXNSMEZCY0VJc1ZVRkJjVUlzU1VGQlNTeEZRVUZGTEVkQlFVYzdVVUZETVVJc1NVRkJTU3hKUVVGSkxFZEJRVWNzU1VGQlNTeERRVUZETEZOQlFWTXNRMEZCUXl4RFFVRkRMRWRCUVVjc1EwRkJReXhEUVVGRE8xRkJSV2hETEVsQlFVa3NRMEZCUXl4VFFVRlRMRU5CUVVNc1EwRkJReXhIUVVGSExFVkJRVVVzU1VGQlNTeERRVUZETEZOQlFWTXNRMEZCUXl4SFFVRkhMRU5CUVVNc1EwRkJReXhEUVVGRE8xRkJRekZETEVsQlFVa3NRMEZCUXl4VFFVRlRMRU5CUVVNc1IwRkJSeXhGUVVGRkxFbEJRVWtzUTBGQlF5eERRVUZETzFGQlJURkNMRWxCUVVrc1EwRkJReXhIUVVGSExFZEJRVWNzU1VGQlNTeERRVUZETzFGQlEyaENMRWxCUVVrc1EwRkJReXhIUVVGSExFZEJRVWNzUzBGQlN5eERRVUZETzFGQlJXcENMRTlCUVU4c1NVRkJTU3hEUVVGRE8wbEJRMmhDTEVOQlFVTTdTVUZGVFN4dlFrRkJZU3hIUVVGd1FpeFZRVUZ4UWl4SlFVRkpMRVZCUVVVc1IwRkJSenRSUVVNeFFpeEpRVUZKTEVOQlFVTXNVMEZCVXl4RFFVRkRMRU5CUVVNc1IwRkJSeXhGUVVGRkxFMUJRVTBzUTBGQlF5eGhRVUZoTEVOQlFVTXNTVUZCU1N4RFFVRkRMRk5CUVZNc1EwRkJReXhEUVVGRExFZEJRVWNzUTBGQlF5eEZRVUZGTEVOQlFVTXNSMEZCUnl4RFFVRkRMRU5CUVVNc1EwRkJRenRSUVVOMlJTeFBRVUZQTEUxQlFVMHNRMEZCUXl4aFFVRmhMRU5CUVVNc1NVRkJTU3hGUVVGRkxFZEJRVWNzUTBGQlF5eERRVUZETzBsQlF6TkRMRU5CUVVNN1NVRkRUQ3hoUVVGRE8wRkJRVVFzUTBGQlF5eEJRWEpOUkN4RFFVRXJRaXhSUVVGUkxFZEJjVTEwUXp0QlFYSk5XU3gzUWtGQlRTSjkiLCJcInVzZSBzdHJpY3RcIjtcclxudmFyIF9fZXh0ZW5kcyA9ICh0aGlzICYmIHRoaXMuX19leHRlbmRzKSB8fCAoZnVuY3Rpb24gKCkge1xyXG4gICAgdmFyIGV4dGVuZFN0YXRpY3MgPSBmdW5jdGlvbiAoZCwgYikge1xyXG4gICAgICAgIGV4dGVuZFN0YXRpY3MgPSBPYmplY3Quc2V0UHJvdG90eXBlT2YgfHxcclxuICAgICAgICAgICAgKHsgX19wcm90b19fOiBbXSB9IGluc3RhbmNlb2YgQXJyYXkgJiYgZnVuY3Rpb24gKGQsIGIpIHsgZC5fX3Byb3RvX18gPSBiOyB9KSB8fFxyXG4gICAgICAgICAgICBmdW5jdGlvbiAoZCwgYikgeyBmb3IgKHZhciBwIGluIGIpIGlmIChiLmhhc093blByb3BlcnR5KHApKSBkW3BdID0gYltwXTsgfTtcclxuICAgICAgICByZXR1cm4gZXh0ZW5kU3RhdGljcyhkLCBiKTtcclxuICAgIH07XHJcbiAgICByZXR1cm4gZnVuY3Rpb24gKGQsIGIpIHtcclxuICAgICAgICBleHRlbmRTdGF0aWNzKGQsIGIpO1xyXG4gICAgICAgIGZ1bmN0aW9uIF9fKCkgeyB0aGlzLmNvbnN0cnVjdG9yID0gZDsgfVxyXG4gICAgICAgIGQucHJvdG90eXBlID0gYiA9PT0gbnVsbCA/IE9iamVjdC5jcmVhdGUoYikgOiAoX18ucHJvdG90eXBlID0gYi5wcm90b3R5cGUsIG5ldyBfXygpKTtcclxuICAgIH07XHJcbn0pKCk7XHJcbk9iamVjdC5kZWZpbmVQcm9wZXJ0eShleHBvcnRzLCBcIl9fZXNNb2R1bGVcIiwgeyB2YWx1ZTogdHJ1ZSB9KTtcclxudmFyIHZwc2NfMSA9IHJlcXVpcmUoXCIuL3Zwc2NcIik7XHJcbnZhciByYnRyZWVfMSA9IHJlcXVpcmUoXCIuL3JidHJlZVwiKTtcclxuZnVuY3Rpb24gY29tcHV0ZUdyb3VwQm91bmRzKGcpIHtcclxuICAgIGcuYm91bmRzID0gdHlwZW9mIGcubGVhdmVzICE9PSBcInVuZGVmaW5lZFwiID9cclxuICAgICAgICBnLmxlYXZlcy5yZWR1Y2UoZnVuY3Rpb24gKHIsIGMpIHsgcmV0dXJuIGMuYm91bmRzLnVuaW9uKHIpOyB9LCBSZWN0YW5nbGUuZW1wdHkoKSkgOlxyXG4gICAgICAgIFJlY3RhbmdsZS5lbXB0eSgpO1xyXG4gICAgaWYgKHR5cGVvZiBnLmdyb3VwcyAhPT0gXCJ1bmRlZmluZWRcIilcclxuICAgICAgICBnLmJvdW5kcyA9IGcuZ3JvdXBzLnJlZHVjZShmdW5jdGlvbiAociwgYykgeyByZXR1cm4gY29tcHV0ZUdyb3VwQm91bmRzKGMpLnVuaW9uKHIpOyB9LCBnLmJvdW5kcyk7XHJcbiAgICBnLmJvdW5kcyA9IGcuYm91bmRzLmluZmxhdGUoZy5wYWRkaW5nKTtcclxuICAgIHJldHVybiBnLmJvdW5kcztcclxufVxyXG5leHBvcnRzLmNvbXB1dGVHcm91cEJvdW5kcyA9IGNvbXB1dGVHcm91cEJvdW5kcztcclxudmFyIFJlY3RhbmdsZSA9IChmdW5jdGlvbiAoKSB7XHJcbiAgICBmdW5jdGlvbiBSZWN0YW5nbGUoeCwgWCwgeSwgWSkge1xyXG4gICAgICAgIHRoaXMueCA9IHg7XHJcbiAgICAgICAgdGhpcy5YID0gWDtcclxuICAgICAgICB0aGlzLnkgPSB5O1xyXG4gICAgICAgIHRoaXMuWSA9IFk7XHJcbiAgICB9XHJcbiAgICBSZWN0YW5nbGUuZW1wdHkgPSBmdW5jdGlvbiAoKSB7IHJldHVybiBuZXcgUmVjdGFuZ2xlKE51bWJlci5QT1NJVElWRV9JTkZJTklUWSwgTnVtYmVyLk5FR0FUSVZFX0lORklOSVRZLCBOdW1iZXIuUE9TSVRJVkVfSU5GSU5JVFksIE51bWJlci5ORUdBVElWRV9JTkZJTklUWSk7IH07XHJcbiAgICBSZWN0YW5nbGUucHJvdG90eXBlLmN4ID0gZnVuY3Rpb24gKCkgeyByZXR1cm4gKHRoaXMueCArIHRoaXMuWCkgLyAyOyB9O1xyXG4gICAgUmVjdGFuZ2xlLnByb3RvdHlwZS5jeSA9IGZ1bmN0aW9uICgpIHsgcmV0dXJuICh0aGlzLnkgKyB0aGlzLlkpIC8gMjsgfTtcclxuICAgIFJlY3RhbmdsZS5wcm90b3R5cGUub3ZlcmxhcFggPSBmdW5jdGlvbiAocikge1xyXG4gICAgICAgIHZhciB1eCA9IHRoaXMuY3goKSwgdnggPSByLmN4KCk7XHJcbiAgICAgICAgaWYgKHV4IDw9IHZ4ICYmIHIueCA8IHRoaXMuWClcclxuICAgICAgICAgICAgcmV0dXJuIHRoaXMuWCAtIHIueDtcclxuICAgICAgICBpZiAodnggPD0gdXggJiYgdGhpcy54IDwgci5YKVxyXG4gICAgICAgICAgICByZXR1cm4gci5YIC0gdGhpcy54O1xyXG4gICAgICAgIHJldHVybiAwO1xyXG4gICAgfTtcclxuICAgIFJlY3RhbmdsZS5wcm90b3R5cGUub3ZlcmxhcFkgPSBmdW5jdGlvbiAocikge1xyXG4gICAgICAgIHZhciB1eSA9IHRoaXMuY3koKSwgdnkgPSByLmN5KCk7XHJcbiAgICAgICAgaWYgKHV5IDw9IHZ5ICYmIHIueSA8IHRoaXMuWSlcclxuICAgICAgICAgICAgcmV0dXJuIHRoaXMuWSAtIHIueTtcclxuICAgICAgICBpZiAodnkgPD0gdXkgJiYgdGhpcy55IDwgci5ZKVxyXG4gICAgICAgICAgICByZXR1cm4gci5ZIC0gdGhpcy55O1xyXG4gICAgICAgIHJldHVybiAwO1xyXG4gICAgfTtcclxuICAgIFJlY3RhbmdsZS5wcm90b3R5cGUuc2V0WENlbnRyZSA9IGZ1bmN0aW9uIChjeCkge1xyXG4gICAgICAgIHZhciBkeCA9IGN4IC0gdGhpcy5jeCgpO1xyXG4gICAgICAgIHRoaXMueCArPSBkeDtcclxuICAgICAgICB0aGlzLlggKz0gZHg7XHJcbiAgICB9O1xyXG4gICAgUmVjdGFuZ2xlLnByb3RvdHlwZS5zZXRZQ2VudHJlID0gZnVuY3Rpb24gKGN5KSB7XHJcbiAgICAgICAgdmFyIGR5ID0gY3kgLSB0aGlzLmN5KCk7XHJcbiAgICAgICAgdGhpcy55ICs9IGR5O1xyXG4gICAgICAgIHRoaXMuWSArPSBkeTtcclxuICAgIH07XHJcbiAgICBSZWN0YW5nbGUucHJvdG90eXBlLndpZHRoID0gZnVuY3Rpb24gKCkge1xyXG4gICAgICAgIHJldHVybiB0aGlzLlggLSB0aGlzLng7XHJcbiAgICB9O1xyXG4gICAgUmVjdGFuZ2xlLnByb3RvdHlwZS5oZWlnaHQgPSBmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgcmV0dXJuIHRoaXMuWSAtIHRoaXMueTtcclxuICAgIH07XHJcbiAgICBSZWN0YW5nbGUucHJvdG90eXBlLnVuaW9uID0gZnVuY3Rpb24gKHIpIHtcclxuICAgICAgICByZXR1cm4gbmV3IFJlY3RhbmdsZShNYXRoLm1pbih0aGlzLngsIHIueCksIE1hdGgubWF4KHRoaXMuWCwgci5YKSwgTWF0aC5taW4odGhpcy55LCByLnkpLCBNYXRoLm1heCh0aGlzLlksIHIuWSkpO1xyXG4gICAgfTtcclxuICAgIFJlY3RhbmdsZS5wcm90b3R5cGUubGluZUludGVyc2VjdGlvbnMgPSBmdW5jdGlvbiAoeDEsIHkxLCB4MiwgeTIpIHtcclxuICAgICAgICB2YXIgc2lkZXMgPSBbW3RoaXMueCwgdGhpcy55LCB0aGlzLlgsIHRoaXMueV0sXHJcbiAgICAgICAgICAgIFt0aGlzLlgsIHRoaXMueSwgdGhpcy5YLCB0aGlzLlldLFxyXG4gICAgICAgICAgICBbdGhpcy5YLCB0aGlzLlksIHRoaXMueCwgdGhpcy5ZXSxcclxuICAgICAgICAgICAgW3RoaXMueCwgdGhpcy5ZLCB0aGlzLngsIHRoaXMueV1dO1xyXG4gICAgICAgIHZhciBpbnRlcnNlY3Rpb25zID0gW107XHJcbiAgICAgICAgZm9yICh2YXIgaSA9IDA7IGkgPCA0OyArK2kpIHtcclxuICAgICAgICAgICAgdmFyIHIgPSBSZWN0YW5nbGUubGluZUludGVyc2VjdGlvbih4MSwgeTEsIHgyLCB5Miwgc2lkZXNbaV1bMF0sIHNpZGVzW2ldWzFdLCBzaWRlc1tpXVsyXSwgc2lkZXNbaV1bM10pO1xyXG4gICAgICAgICAgICBpZiAociAhPT0gbnVsbClcclxuICAgICAgICAgICAgICAgIGludGVyc2VjdGlvbnMucHVzaCh7IHg6IHIueCwgeTogci55IH0pO1xyXG4gICAgICAgIH1cclxuICAgICAgICByZXR1cm4gaW50ZXJzZWN0aW9ucztcclxuICAgIH07XHJcbiAgICBSZWN0YW5nbGUucHJvdG90eXBlLnJheUludGVyc2VjdGlvbiA9IGZ1bmN0aW9uICh4MiwgeTIpIHtcclxuICAgICAgICB2YXIgaW50cyA9IHRoaXMubGluZUludGVyc2VjdGlvbnModGhpcy5jeCgpLCB0aGlzLmN5KCksIHgyLCB5Mik7XHJcbiAgICAgICAgcmV0dXJuIGludHMubGVuZ3RoID4gMCA/IGludHNbMF0gOiBudWxsO1xyXG4gICAgfTtcclxuICAgIFJlY3RhbmdsZS5wcm90b3R5cGUudmVydGljZXMgPSBmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgcmV0dXJuIFtcclxuICAgICAgICAgICAgeyB4OiB0aGlzLngsIHk6IHRoaXMueSB9LFxyXG4gICAgICAgICAgICB7IHg6IHRoaXMuWCwgeTogdGhpcy55IH0sXHJcbiAgICAgICAgICAgIHsgeDogdGhpcy5YLCB5OiB0aGlzLlkgfSxcclxuICAgICAgICAgICAgeyB4OiB0aGlzLngsIHk6IHRoaXMuWSB9XHJcbiAgICAgICAgXTtcclxuICAgIH07XHJcbiAgICBSZWN0YW5nbGUubGluZUludGVyc2VjdGlvbiA9IGZ1bmN0aW9uICh4MSwgeTEsIHgyLCB5MiwgeDMsIHkzLCB4NCwgeTQpIHtcclxuICAgICAgICB2YXIgZHgxMiA9IHgyIC0geDEsIGR4MzQgPSB4NCAtIHgzLCBkeTEyID0geTIgLSB5MSwgZHkzNCA9IHk0IC0geTMsIGRlbm9taW5hdG9yID0gZHkzNCAqIGR4MTIgLSBkeDM0ICogZHkxMjtcclxuICAgICAgICBpZiAoZGVub21pbmF0b3IgPT0gMClcclxuICAgICAgICAgICAgcmV0dXJuIG51bGw7XHJcbiAgICAgICAgdmFyIGR4MzEgPSB4MSAtIHgzLCBkeTMxID0geTEgLSB5MywgbnVtYSA9IGR4MzQgKiBkeTMxIC0gZHkzNCAqIGR4MzEsIGEgPSBudW1hIC8gZGVub21pbmF0b3IsIG51bWIgPSBkeDEyICogZHkzMSAtIGR5MTIgKiBkeDMxLCBiID0gbnVtYiAvIGRlbm9taW5hdG9yO1xyXG4gICAgICAgIGlmIChhID49IDAgJiYgYSA8PSAxICYmIGIgPj0gMCAmJiBiIDw9IDEpIHtcclxuICAgICAgICAgICAgcmV0dXJuIHtcclxuICAgICAgICAgICAgICAgIHg6IHgxICsgYSAqIGR4MTIsXHJcbiAgICAgICAgICAgICAgICB5OiB5MSArIGEgKiBkeTEyXHJcbiAgICAgICAgICAgIH07XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHJldHVybiBudWxsO1xyXG4gICAgfTtcclxuICAgIFJlY3RhbmdsZS5wcm90b3R5cGUuaW5mbGF0ZSA9IGZ1bmN0aW9uIChwYWQpIHtcclxuICAgICAgICByZXR1cm4gbmV3IFJlY3RhbmdsZSh0aGlzLnggLSBwYWQsIHRoaXMuWCArIHBhZCwgdGhpcy55IC0gcGFkLCB0aGlzLlkgKyBwYWQpO1xyXG4gICAgfTtcclxuICAgIHJldHVybiBSZWN0YW5nbGU7XHJcbn0oKSk7XHJcbmV4cG9ydHMuUmVjdGFuZ2xlID0gUmVjdGFuZ2xlO1xyXG5mdW5jdGlvbiBtYWtlRWRnZUJldHdlZW4oc291cmNlLCB0YXJnZXQsIGFoKSB7XHJcbiAgICB2YXIgc2kgPSBzb3VyY2UucmF5SW50ZXJzZWN0aW9uKHRhcmdldC5jeCgpLCB0YXJnZXQuY3koKSkgfHwgeyB4OiBzb3VyY2UuY3goKSwgeTogc291cmNlLmN5KCkgfSwgdGkgPSB0YXJnZXQucmF5SW50ZXJzZWN0aW9uKHNvdXJjZS5jeCgpLCBzb3VyY2UuY3koKSkgfHwgeyB4OiB0YXJnZXQuY3goKSwgeTogdGFyZ2V0LmN5KCkgfSwgZHggPSB0aS54IC0gc2kueCwgZHkgPSB0aS55IC0gc2kueSwgbCA9IE1hdGguc3FydChkeCAqIGR4ICsgZHkgKiBkeSksIGFsID0gbCAtIGFoO1xyXG4gICAgcmV0dXJuIHtcclxuICAgICAgICBzb3VyY2VJbnRlcnNlY3Rpb246IHNpLFxyXG4gICAgICAgIHRhcmdldEludGVyc2VjdGlvbjogdGksXHJcbiAgICAgICAgYXJyb3dTdGFydDogeyB4OiBzaS54ICsgYWwgKiBkeCAvIGwsIHk6IHNpLnkgKyBhbCAqIGR5IC8gbCB9XHJcbiAgICB9O1xyXG59XHJcbmV4cG9ydHMubWFrZUVkZ2VCZXR3ZWVuID0gbWFrZUVkZ2VCZXR3ZWVuO1xyXG5mdW5jdGlvbiBtYWtlRWRnZVRvKHMsIHRhcmdldCwgYWgpIHtcclxuICAgIHZhciB0aSA9IHRhcmdldC5yYXlJbnRlcnNlY3Rpb24ocy54LCBzLnkpO1xyXG4gICAgaWYgKCF0aSlcclxuICAgICAgICB0aSA9IHsgeDogdGFyZ2V0LmN4KCksIHk6IHRhcmdldC5jeSgpIH07XHJcbiAgICB2YXIgZHggPSB0aS54IC0gcy54LCBkeSA9IHRpLnkgLSBzLnksIGwgPSBNYXRoLnNxcnQoZHggKiBkeCArIGR5ICogZHkpO1xyXG4gICAgcmV0dXJuIHsgeDogdGkueCAtIGFoICogZHggLyBsLCB5OiB0aS55IC0gYWggKiBkeSAvIGwgfTtcclxufVxyXG5leHBvcnRzLm1ha2VFZGdlVG8gPSBtYWtlRWRnZVRvO1xyXG52YXIgTm9kZSA9IChmdW5jdGlvbiAoKSB7XHJcbiAgICBmdW5jdGlvbiBOb2RlKHYsIHIsIHBvcykge1xyXG4gICAgICAgIHRoaXMudiA9IHY7XHJcbiAgICAgICAgdGhpcy5yID0gcjtcclxuICAgICAgICB0aGlzLnBvcyA9IHBvcztcclxuICAgICAgICB0aGlzLnByZXYgPSBtYWtlUkJUcmVlKCk7XHJcbiAgICAgICAgdGhpcy5uZXh0ID0gbWFrZVJCVHJlZSgpO1xyXG4gICAgfVxyXG4gICAgcmV0dXJuIE5vZGU7XHJcbn0oKSk7XHJcbnZhciBFdmVudCA9IChmdW5jdGlvbiAoKSB7XHJcbiAgICBmdW5jdGlvbiBFdmVudChpc09wZW4sIHYsIHBvcykge1xyXG4gICAgICAgIHRoaXMuaXNPcGVuID0gaXNPcGVuO1xyXG4gICAgICAgIHRoaXMudiA9IHY7XHJcbiAgICAgICAgdGhpcy5wb3MgPSBwb3M7XHJcbiAgICB9XHJcbiAgICByZXR1cm4gRXZlbnQ7XHJcbn0oKSk7XHJcbmZ1bmN0aW9uIGNvbXBhcmVFdmVudHMoYSwgYikge1xyXG4gICAgaWYgKGEucG9zID4gYi5wb3MpIHtcclxuICAgICAgICByZXR1cm4gMTtcclxuICAgIH1cclxuICAgIGlmIChhLnBvcyA8IGIucG9zKSB7XHJcbiAgICAgICAgcmV0dXJuIC0xO1xyXG4gICAgfVxyXG4gICAgaWYgKGEuaXNPcGVuKSB7XHJcbiAgICAgICAgcmV0dXJuIC0xO1xyXG4gICAgfVxyXG4gICAgaWYgKGIuaXNPcGVuKSB7XHJcbiAgICAgICAgcmV0dXJuIDE7XHJcbiAgICB9XHJcbiAgICByZXR1cm4gMDtcclxufVxyXG5mdW5jdGlvbiBtYWtlUkJUcmVlKCkge1xyXG4gICAgcmV0dXJuIG5ldyByYnRyZWVfMS5SQlRyZWUoZnVuY3Rpb24gKGEsIGIpIHsgcmV0dXJuIGEucG9zIC0gYi5wb3M7IH0pO1xyXG59XHJcbnZhciB4UmVjdCA9IHtcclxuICAgIGdldENlbnRyZTogZnVuY3Rpb24gKHIpIHsgcmV0dXJuIHIuY3goKTsgfSxcclxuICAgIGdldE9wZW46IGZ1bmN0aW9uIChyKSB7IHJldHVybiByLnk7IH0sXHJcbiAgICBnZXRDbG9zZTogZnVuY3Rpb24gKHIpIHsgcmV0dXJuIHIuWTsgfSxcclxuICAgIGdldFNpemU6IGZ1bmN0aW9uIChyKSB7IHJldHVybiByLndpZHRoKCk7IH0sXHJcbiAgICBtYWtlUmVjdDogZnVuY3Rpb24gKG9wZW4sIGNsb3NlLCBjZW50ZXIsIHNpemUpIHsgcmV0dXJuIG5ldyBSZWN0YW5nbGUoY2VudGVyIC0gc2l6ZSAvIDIsIGNlbnRlciArIHNpemUgLyAyLCBvcGVuLCBjbG9zZSk7IH0sXHJcbiAgICBmaW5kTmVpZ2hib3VyczogZmluZFhOZWlnaGJvdXJzXHJcbn07XHJcbnZhciB5UmVjdCA9IHtcclxuICAgIGdldENlbnRyZTogZnVuY3Rpb24gKHIpIHsgcmV0dXJuIHIuY3koKTsgfSxcclxuICAgIGdldE9wZW46IGZ1bmN0aW9uIChyKSB7IHJldHVybiByLng7IH0sXHJcbiAgICBnZXRDbG9zZTogZnVuY3Rpb24gKHIpIHsgcmV0dXJuIHIuWDsgfSxcclxuICAgIGdldFNpemU6IGZ1bmN0aW9uIChyKSB7IHJldHVybiByLmhlaWdodCgpOyB9LFxyXG4gICAgbWFrZVJlY3Q6IGZ1bmN0aW9uIChvcGVuLCBjbG9zZSwgY2VudGVyLCBzaXplKSB7IHJldHVybiBuZXcgUmVjdGFuZ2xlKG9wZW4sIGNsb3NlLCBjZW50ZXIgLSBzaXplIC8gMiwgY2VudGVyICsgc2l6ZSAvIDIpOyB9LFxyXG4gICAgZmluZE5laWdoYm91cnM6IGZpbmRZTmVpZ2hib3Vyc1xyXG59O1xyXG5mdW5jdGlvbiBnZW5lcmF0ZUdyb3VwQ29uc3RyYWludHMocm9vdCwgZiwgbWluU2VwLCBpc0NvbnRhaW5lZCkge1xyXG4gICAgaWYgKGlzQ29udGFpbmVkID09PSB2b2lkIDApIHsgaXNDb250YWluZWQgPSBmYWxzZTsgfVxyXG4gICAgdmFyIHBhZGRpbmcgPSByb290LnBhZGRpbmcsIGduID0gdHlwZW9mIHJvb3QuZ3JvdXBzICE9PSAndW5kZWZpbmVkJyA/IHJvb3QuZ3JvdXBzLmxlbmd0aCA6IDAsIGxuID0gdHlwZW9mIHJvb3QubGVhdmVzICE9PSAndW5kZWZpbmVkJyA/IHJvb3QubGVhdmVzLmxlbmd0aCA6IDAsIGNoaWxkQ29uc3RyYWludHMgPSAhZ24gPyBbXVxyXG4gICAgICAgIDogcm9vdC5ncm91cHMucmVkdWNlKGZ1bmN0aW9uIChjY3MsIGcpIHsgcmV0dXJuIGNjcy5jb25jYXQoZ2VuZXJhdGVHcm91cENvbnN0cmFpbnRzKGcsIGYsIG1pblNlcCwgdHJ1ZSkpOyB9LCBbXSksIG4gPSAoaXNDb250YWluZWQgPyAyIDogMCkgKyBsbiArIGduLCB2cyA9IG5ldyBBcnJheShuKSwgcnMgPSBuZXcgQXJyYXkobiksIGkgPSAwLCBhZGQgPSBmdW5jdGlvbiAociwgdikgeyByc1tpXSA9IHI7IHZzW2krK10gPSB2OyB9O1xyXG4gICAgaWYgKGlzQ29udGFpbmVkKSB7XHJcbiAgICAgICAgdmFyIGIgPSByb290LmJvdW5kcywgYyA9IGYuZ2V0Q2VudHJlKGIpLCBzID0gZi5nZXRTaXplKGIpIC8gMiwgb3BlbiA9IGYuZ2V0T3BlbihiKSwgY2xvc2UgPSBmLmdldENsb3NlKGIpLCBtaW4gPSBjIC0gcyArIHBhZGRpbmcgLyAyLCBtYXggPSBjICsgcyAtIHBhZGRpbmcgLyAyO1xyXG4gICAgICAgIHJvb3QubWluVmFyLmRlc2lyZWRQb3NpdGlvbiA9IG1pbjtcclxuICAgICAgICBhZGQoZi5tYWtlUmVjdChvcGVuLCBjbG9zZSwgbWluLCBwYWRkaW5nKSwgcm9vdC5taW5WYXIpO1xyXG4gICAgICAgIHJvb3QubWF4VmFyLmRlc2lyZWRQb3NpdGlvbiA9IG1heDtcclxuICAgICAgICBhZGQoZi5tYWtlUmVjdChvcGVuLCBjbG9zZSwgbWF4LCBwYWRkaW5nKSwgcm9vdC5tYXhWYXIpO1xyXG4gICAgfVxyXG4gICAgaWYgKGxuKVxyXG4gICAgICAgIHJvb3QubGVhdmVzLmZvckVhY2goZnVuY3Rpb24gKGwpIHsgcmV0dXJuIGFkZChsLmJvdW5kcywgbC52YXJpYWJsZSk7IH0pO1xyXG4gICAgaWYgKGduKVxyXG4gICAgICAgIHJvb3QuZ3JvdXBzLmZvckVhY2goZnVuY3Rpb24gKGcpIHtcclxuICAgICAgICAgICAgdmFyIGIgPSBnLmJvdW5kcztcclxuICAgICAgICAgICAgYWRkKGYubWFrZVJlY3QoZi5nZXRPcGVuKGIpLCBmLmdldENsb3NlKGIpLCBmLmdldENlbnRyZShiKSwgZi5nZXRTaXplKGIpKSwgZy5taW5WYXIpO1xyXG4gICAgICAgIH0pO1xyXG4gICAgdmFyIGNzID0gZ2VuZXJhdGVDb25zdHJhaW50cyhycywgdnMsIGYsIG1pblNlcCk7XHJcbiAgICBpZiAoZ24pIHtcclxuICAgICAgICB2cy5mb3JFYWNoKGZ1bmN0aW9uICh2KSB7IHYuY091dCA9IFtdLCB2LmNJbiA9IFtdOyB9KTtcclxuICAgICAgICBjcy5mb3JFYWNoKGZ1bmN0aW9uIChjKSB7IGMubGVmdC5jT3V0LnB1c2goYyksIGMucmlnaHQuY0luLnB1c2goYyk7IH0pO1xyXG4gICAgICAgIHJvb3QuZ3JvdXBzLmZvckVhY2goZnVuY3Rpb24gKGcpIHtcclxuICAgICAgICAgICAgdmFyIGdhcEFkanVzdG1lbnQgPSAoZy5wYWRkaW5nIC0gZi5nZXRTaXplKGcuYm91bmRzKSkgLyAyO1xyXG4gICAgICAgICAgICBnLm1pblZhci5jSW4uZm9yRWFjaChmdW5jdGlvbiAoYykgeyByZXR1cm4gYy5nYXAgKz0gZ2FwQWRqdXN0bWVudDsgfSk7XHJcbiAgICAgICAgICAgIGcubWluVmFyLmNPdXQuZm9yRWFjaChmdW5jdGlvbiAoYykgeyBjLmxlZnQgPSBnLm1heFZhcjsgYy5nYXAgKz0gZ2FwQWRqdXN0bWVudDsgfSk7XHJcbiAgICAgICAgfSk7XHJcbiAgICB9XHJcbiAgICByZXR1cm4gY2hpbGRDb25zdHJhaW50cy5jb25jYXQoY3MpO1xyXG59XHJcbmZ1bmN0aW9uIGdlbmVyYXRlQ29uc3RyYWludHMocnMsIHZhcnMsIHJlY3QsIG1pblNlcCkge1xyXG4gICAgdmFyIGksIG4gPSBycy5sZW5ndGg7XHJcbiAgICB2YXIgTiA9IDIgKiBuO1xyXG4gICAgY29uc29sZS5hc3NlcnQodmFycy5sZW5ndGggPj0gbik7XHJcbiAgICB2YXIgZXZlbnRzID0gbmV3IEFycmF5KE4pO1xyXG4gICAgZm9yIChpID0gMDsgaSA8IG47ICsraSkge1xyXG4gICAgICAgIHZhciByID0gcnNbaV07XHJcbiAgICAgICAgdmFyIHYgPSBuZXcgTm9kZSh2YXJzW2ldLCByLCByZWN0LmdldENlbnRyZShyKSk7XHJcbiAgICAgICAgZXZlbnRzW2ldID0gbmV3IEV2ZW50KHRydWUsIHYsIHJlY3QuZ2V0T3BlbihyKSk7XHJcbiAgICAgICAgZXZlbnRzW2kgKyBuXSA9IG5ldyBFdmVudChmYWxzZSwgdiwgcmVjdC5nZXRDbG9zZShyKSk7XHJcbiAgICB9XHJcbiAgICBldmVudHMuc29ydChjb21wYXJlRXZlbnRzKTtcclxuICAgIHZhciBjcyA9IG5ldyBBcnJheSgpO1xyXG4gICAgdmFyIHNjYW5saW5lID0gbWFrZVJCVHJlZSgpO1xyXG4gICAgZm9yIChpID0gMDsgaSA8IE47ICsraSkge1xyXG4gICAgICAgIHZhciBlID0gZXZlbnRzW2ldO1xyXG4gICAgICAgIHZhciB2ID0gZS52O1xyXG4gICAgICAgIGlmIChlLmlzT3Blbikge1xyXG4gICAgICAgICAgICBzY2FubGluZS5pbnNlcnQodik7XHJcbiAgICAgICAgICAgIHJlY3QuZmluZE5laWdoYm91cnModiwgc2NhbmxpbmUpO1xyXG4gICAgICAgIH1cclxuICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgc2NhbmxpbmUucmVtb3ZlKHYpO1xyXG4gICAgICAgICAgICB2YXIgbWFrZUNvbnN0cmFpbnQgPSBmdW5jdGlvbiAobCwgcikge1xyXG4gICAgICAgICAgICAgICAgdmFyIHNlcCA9IChyZWN0LmdldFNpemUobC5yKSArIHJlY3QuZ2V0U2l6ZShyLnIpKSAvIDIgKyBtaW5TZXA7XHJcbiAgICAgICAgICAgICAgICBjcy5wdXNoKG5ldyB2cHNjXzEuQ29uc3RyYWludChsLnYsIHIudiwgc2VwKSk7XHJcbiAgICAgICAgICAgIH07XHJcbiAgICAgICAgICAgIHZhciB2aXNpdE5laWdoYm91cnMgPSBmdW5jdGlvbiAoZm9yd2FyZCwgcmV2ZXJzZSwgbWtjb24pIHtcclxuICAgICAgICAgICAgICAgIHZhciB1LCBpdCA9IHZbZm9yd2FyZF0uaXRlcmF0b3IoKTtcclxuICAgICAgICAgICAgICAgIHdoaWxlICgodSA9IGl0W2ZvcndhcmRdKCkpICE9PSBudWxsKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgbWtjb24odSwgdik7XHJcbiAgICAgICAgICAgICAgICAgICAgdVtyZXZlcnNlXS5yZW1vdmUodik7XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIH07XHJcbiAgICAgICAgICAgIHZpc2l0TmVpZ2hib3VycyhcInByZXZcIiwgXCJuZXh0XCIsIGZ1bmN0aW9uICh1LCB2KSB7IHJldHVybiBtYWtlQ29uc3RyYWludCh1LCB2KTsgfSk7XHJcbiAgICAgICAgICAgIHZpc2l0TmVpZ2hib3VycyhcIm5leHRcIiwgXCJwcmV2XCIsIGZ1bmN0aW9uICh1LCB2KSB7IHJldHVybiBtYWtlQ29uc3RyYWludCh2LCB1KTsgfSk7XHJcbiAgICAgICAgfVxyXG4gICAgfVxyXG4gICAgY29uc29sZS5hc3NlcnQoc2NhbmxpbmUuc2l6ZSA9PT0gMCk7XHJcbiAgICByZXR1cm4gY3M7XHJcbn1cclxuZnVuY3Rpb24gZmluZFhOZWlnaGJvdXJzKHYsIHNjYW5saW5lKSB7XHJcbiAgICB2YXIgZiA9IGZ1bmN0aW9uIChmb3J3YXJkLCByZXZlcnNlKSB7XHJcbiAgICAgICAgdmFyIGl0ID0gc2NhbmxpbmUuZmluZEl0ZXIodik7XHJcbiAgICAgICAgdmFyIHU7XHJcbiAgICAgICAgd2hpbGUgKCh1ID0gaXRbZm9yd2FyZF0oKSkgIT09IG51bGwpIHtcclxuICAgICAgICAgICAgdmFyIHVvdmVydlggPSB1LnIub3ZlcmxhcFgodi5yKTtcclxuICAgICAgICAgICAgaWYgKHVvdmVydlggPD0gMCB8fCB1b3ZlcnZYIDw9IHUuci5vdmVybGFwWSh2LnIpKSB7XHJcbiAgICAgICAgICAgICAgICB2W2ZvcndhcmRdLmluc2VydCh1KTtcclxuICAgICAgICAgICAgICAgIHVbcmV2ZXJzZV0uaW5zZXJ0KHYpO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIGlmICh1b3ZlcnZYIDw9IDApIHtcclxuICAgICAgICAgICAgICAgIGJyZWFrO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG4gICAgfTtcclxuICAgIGYoXCJuZXh0XCIsIFwicHJldlwiKTtcclxuICAgIGYoXCJwcmV2XCIsIFwibmV4dFwiKTtcclxufVxyXG5mdW5jdGlvbiBmaW5kWU5laWdoYm91cnModiwgc2NhbmxpbmUpIHtcclxuICAgIHZhciBmID0gZnVuY3Rpb24gKGZvcndhcmQsIHJldmVyc2UpIHtcclxuICAgICAgICB2YXIgdSA9IHNjYW5saW5lLmZpbmRJdGVyKHYpW2ZvcndhcmRdKCk7XHJcbiAgICAgICAgaWYgKHUgIT09IG51bGwgJiYgdS5yLm92ZXJsYXBYKHYucikgPiAwKSB7XHJcbiAgICAgICAgICAgIHZbZm9yd2FyZF0uaW5zZXJ0KHUpO1xyXG4gICAgICAgICAgICB1W3JldmVyc2VdLmluc2VydCh2KTtcclxuICAgICAgICB9XHJcbiAgICB9O1xyXG4gICAgZihcIm5leHRcIiwgXCJwcmV2XCIpO1xyXG4gICAgZihcInByZXZcIiwgXCJuZXh0XCIpO1xyXG59XHJcbmZ1bmN0aW9uIGdlbmVyYXRlWENvbnN0cmFpbnRzKHJzLCB2YXJzKSB7XHJcbiAgICByZXR1cm4gZ2VuZXJhdGVDb25zdHJhaW50cyhycywgdmFycywgeFJlY3QsIDFlLTYpO1xyXG59XHJcbmV4cG9ydHMuZ2VuZXJhdGVYQ29uc3RyYWludHMgPSBnZW5lcmF0ZVhDb25zdHJhaW50cztcclxuZnVuY3Rpb24gZ2VuZXJhdGVZQ29uc3RyYWludHMocnMsIHZhcnMpIHtcclxuICAgIHJldHVybiBnZW5lcmF0ZUNvbnN0cmFpbnRzKHJzLCB2YXJzLCB5UmVjdCwgMWUtNik7XHJcbn1cclxuZXhwb3J0cy5nZW5lcmF0ZVlDb25zdHJhaW50cyA9IGdlbmVyYXRlWUNvbnN0cmFpbnRzO1xyXG5mdW5jdGlvbiBnZW5lcmF0ZVhHcm91cENvbnN0cmFpbnRzKHJvb3QpIHtcclxuICAgIHJldHVybiBnZW5lcmF0ZUdyb3VwQ29uc3RyYWludHMocm9vdCwgeFJlY3QsIDFlLTYpO1xyXG59XHJcbmV4cG9ydHMuZ2VuZXJhdGVYR3JvdXBDb25zdHJhaW50cyA9IGdlbmVyYXRlWEdyb3VwQ29uc3RyYWludHM7XHJcbmZ1bmN0aW9uIGdlbmVyYXRlWUdyb3VwQ29uc3RyYWludHMocm9vdCkge1xyXG4gICAgcmV0dXJuIGdlbmVyYXRlR3JvdXBDb25zdHJhaW50cyhyb290LCB5UmVjdCwgMWUtNik7XHJcbn1cclxuZXhwb3J0cy5nZW5lcmF0ZVlHcm91cENvbnN0cmFpbnRzID0gZ2VuZXJhdGVZR3JvdXBDb25zdHJhaW50cztcclxuZnVuY3Rpb24gcmVtb3ZlT3ZlcmxhcHMocnMpIHtcclxuICAgIHZhciB2cyA9IHJzLm1hcChmdW5jdGlvbiAocikgeyByZXR1cm4gbmV3IHZwc2NfMS5WYXJpYWJsZShyLmN4KCkpOyB9KTtcclxuICAgIHZhciBjcyA9IGdlbmVyYXRlWENvbnN0cmFpbnRzKHJzLCB2cyk7XHJcbiAgICB2YXIgc29sdmVyID0gbmV3IHZwc2NfMS5Tb2x2ZXIodnMsIGNzKTtcclxuICAgIHNvbHZlci5zb2x2ZSgpO1xyXG4gICAgdnMuZm9yRWFjaChmdW5jdGlvbiAodiwgaSkgeyByZXR1cm4gcnNbaV0uc2V0WENlbnRyZSh2LnBvc2l0aW9uKCkpOyB9KTtcclxuICAgIHZzID0gcnMubWFwKGZ1bmN0aW9uIChyKSB7IHJldHVybiBuZXcgdnBzY18xLlZhcmlhYmxlKHIuY3koKSk7IH0pO1xyXG4gICAgY3MgPSBnZW5lcmF0ZVlDb25zdHJhaW50cyhycywgdnMpO1xyXG4gICAgc29sdmVyID0gbmV3IHZwc2NfMS5Tb2x2ZXIodnMsIGNzKTtcclxuICAgIHNvbHZlci5zb2x2ZSgpO1xyXG4gICAgdnMuZm9yRWFjaChmdW5jdGlvbiAodiwgaSkgeyByZXR1cm4gcnNbaV0uc2V0WUNlbnRyZSh2LnBvc2l0aW9uKCkpOyB9KTtcclxufVxyXG5leHBvcnRzLnJlbW92ZU92ZXJsYXBzID0gcmVtb3ZlT3ZlcmxhcHM7XHJcbnZhciBJbmRleGVkVmFyaWFibGUgPSAoZnVuY3Rpb24gKF9zdXBlcikge1xyXG4gICAgX19leHRlbmRzKEluZGV4ZWRWYXJpYWJsZSwgX3N1cGVyKTtcclxuICAgIGZ1bmN0aW9uIEluZGV4ZWRWYXJpYWJsZShpbmRleCwgdykge1xyXG4gICAgICAgIHZhciBfdGhpcyA9IF9zdXBlci5jYWxsKHRoaXMsIDAsIHcpIHx8IHRoaXM7XHJcbiAgICAgICAgX3RoaXMuaW5kZXggPSBpbmRleDtcclxuICAgICAgICByZXR1cm4gX3RoaXM7XHJcbiAgICB9XHJcbiAgICByZXR1cm4gSW5kZXhlZFZhcmlhYmxlO1xyXG59KHZwc2NfMS5WYXJpYWJsZSkpO1xyXG5leHBvcnRzLkluZGV4ZWRWYXJpYWJsZSA9IEluZGV4ZWRWYXJpYWJsZTtcclxudmFyIFByb2plY3Rpb24gPSAoZnVuY3Rpb24gKCkge1xyXG4gICAgZnVuY3Rpb24gUHJvamVjdGlvbihub2RlcywgZ3JvdXBzLCByb290R3JvdXAsIGNvbnN0cmFpbnRzLCBhdm9pZE92ZXJsYXBzKSB7XHJcbiAgICAgICAgdmFyIF90aGlzID0gdGhpcztcclxuICAgICAgICBpZiAocm9vdEdyb3VwID09PSB2b2lkIDApIHsgcm9vdEdyb3VwID0gbnVsbDsgfVxyXG4gICAgICAgIGlmIChjb25zdHJhaW50cyA9PT0gdm9pZCAwKSB7IGNvbnN0cmFpbnRzID0gbnVsbDsgfVxyXG4gICAgICAgIGlmIChhdm9pZE92ZXJsYXBzID09PSB2b2lkIDApIHsgYXZvaWRPdmVybGFwcyA9IGZhbHNlOyB9XHJcbiAgICAgICAgdGhpcy5ub2RlcyA9IG5vZGVzO1xyXG4gICAgICAgIHRoaXMuZ3JvdXBzID0gZ3JvdXBzO1xyXG4gICAgICAgIHRoaXMucm9vdEdyb3VwID0gcm9vdEdyb3VwO1xyXG4gICAgICAgIHRoaXMuYXZvaWRPdmVybGFwcyA9IGF2b2lkT3ZlcmxhcHM7XHJcbiAgICAgICAgdGhpcy52YXJpYWJsZXMgPSBub2Rlcy5tYXAoZnVuY3Rpb24gKHYsIGkpIHtcclxuICAgICAgICAgICAgcmV0dXJuIHYudmFyaWFibGUgPSBuZXcgSW5kZXhlZFZhcmlhYmxlKGksIDEpO1xyXG4gICAgICAgIH0pO1xyXG4gICAgICAgIGlmIChjb25zdHJhaW50cylcclxuICAgICAgICAgICAgdGhpcy5jcmVhdGVDb25zdHJhaW50cyhjb25zdHJhaW50cyk7XHJcbiAgICAgICAgaWYgKGF2b2lkT3ZlcmxhcHMgJiYgcm9vdEdyb3VwICYmIHR5cGVvZiByb290R3JvdXAuZ3JvdXBzICE9PSAndW5kZWZpbmVkJykge1xyXG4gICAgICAgICAgICBub2Rlcy5mb3JFYWNoKGZ1bmN0aW9uICh2KSB7XHJcbiAgICAgICAgICAgICAgICBpZiAoIXYud2lkdGggfHwgIXYuaGVpZ2h0KSB7XHJcbiAgICAgICAgICAgICAgICAgICAgdi5ib3VuZHMgPSBuZXcgUmVjdGFuZ2xlKHYueCwgdi54LCB2LnksIHYueSk7XHJcbiAgICAgICAgICAgICAgICAgICAgcmV0dXJuO1xyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgdmFyIHcyID0gdi53aWR0aCAvIDIsIGgyID0gdi5oZWlnaHQgLyAyO1xyXG4gICAgICAgICAgICAgICAgdi5ib3VuZHMgPSBuZXcgUmVjdGFuZ2xlKHYueCAtIHcyLCB2LnggKyB3Miwgdi55IC0gaDIsIHYueSArIGgyKTtcclxuICAgICAgICAgICAgfSk7XHJcbiAgICAgICAgICAgIGNvbXB1dGVHcm91cEJvdW5kcyhyb290R3JvdXApO1xyXG4gICAgICAgICAgICB2YXIgaSA9IG5vZGVzLmxlbmd0aDtcclxuICAgICAgICAgICAgZ3JvdXBzLmZvckVhY2goZnVuY3Rpb24gKGcpIHtcclxuICAgICAgICAgICAgICAgIF90aGlzLnZhcmlhYmxlc1tpXSA9IGcubWluVmFyID0gbmV3IEluZGV4ZWRWYXJpYWJsZShpKyssIHR5cGVvZiBnLnN0aWZmbmVzcyAhPT0gXCJ1bmRlZmluZWRcIiA/IGcuc3RpZmZuZXNzIDogMC4wMSk7XHJcbiAgICAgICAgICAgICAgICBfdGhpcy52YXJpYWJsZXNbaV0gPSBnLm1heFZhciA9IG5ldyBJbmRleGVkVmFyaWFibGUoaSsrLCB0eXBlb2YgZy5zdGlmZm5lc3MgIT09IFwidW5kZWZpbmVkXCIgPyBnLnN0aWZmbmVzcyA6IDAuMDEpO1xyXG4gICAgICAgICAgICB9KTtcclxuICAgICAgICB9XHJcbiAgICB9XHJcbiAgICBQcm9qZWN0aW9uLnByb3RvdHlwZS5jcmVhdGVTZXBhcmF0aW9uID0gZnVuY3Rpb24gKGMpIHtcclxuICAgICAgICByZXR1cm4gbmV3IHZwc2NfMS5Db25zdHJhaW50KHRoaXMubm9kZXNbYy5sZWZ0XS52YXJpYWJsZSwgdGhpcy5ub2Rlc1tjLnJpZ2h0XS52YXJpYWJsZSwgYy5nYXAsIHR5cGVvZiBjLmVxdWFsaXR5ICE9PSBcInVuZGVmaW5lZFwiID8gYy5lcXVhbGl0eSA6IGZhbHNlKTtcclxuICAgIH07XHJcbiAgICBQcm9qZWN0aW9uLnByb3RvdHlwZS5tYWtlRmVhc2libGUgPSBmdW5jdGlvbiAoYykge1xyXG4gICAgICAgIHZhciBfdGhpcyA9IHRoaXM7XHJcbiAgICAgICAgaWYgKCF0aGlzLmF2b2lkT3ZlcmxhcHMpXHJcbiAgICAgICAgICAgIHJldHVybjtcclxuICAgICAgICB2YXIgYXhpcyA9ICd4JywgZGltID0gJ3dpZHRoJztcclxuICAgICAgICBpZiAoYy5heGlzID09PSAneCcpXHJcbiAgICAgICAgICAgIGF4aXMgPSAneScsIGRpbSA9ICdoZWlnaHQnO1xyXG4gICAgICAgIHZhciB2cyA9IGMub2Zmc2V0cy5tYXAoZnVuY3Rpb24gKG8pIHsgcmV0dXJuIF90aGlzLm5vZGVzW28ubm9kZV07IH0pLnNvcnQoZnVuY3Rpb24gKGEsIGIpIHsgcmV0dXJuIGFbYXhpc10gLSBiW2F4aXNdOyB9KTtcclxuICAgICAgICB2YXIgcCA9IG51bGw7XHJcbiAgICAgICAgdnMuZm9yRWFjaChmdW5jdGlvbiAodikge1xyXG4gICAgICAgICAgICBpZiAocCkge1xyXG4gICAgICAgICAgICAgICAgdmFyIG5leHRQb3MgPSBwW2F4aXNdICsgcFtkaW1dO1xyXG4gICAgICAgICAgICAgICAgaWYgKG5leHRQb3MgPiB2W2F4aXNdKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgdltheGlzXSA9IG5leHRQb3M7XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgcCA9IHY7XHJcbiAgICAgICAgfSk7XHJcbiAgICB9O1xyXG4gICAgUHJvamVjdGlvbi5wcm90b3R5cGUuY3JlYXRlQWxpZ25tZW50ID0gZnVuY3Rpb24gKGMpIHtcclxuICAgICAgICB2YXIgX3RoaXMgPSB0aGlzO1xyXG4gICAgICAgIHZhciB1ID0gdGhpcy5ub2Rlc1tjLm9mZnNldHNbMF0ubm9kZV0udmFyaWFibGU7XHJcbiAgICAgICAgdGhpcy5tYWtlRmVhc2libGUoYyk7XHJcbiAgICAgICAgdmFyIGNzID0gYy5heGlzID09PSAneCcgPyB0aGlzLnhDb25zdHJhaW50cyA6IHRoaXMueUNvbnN0cmFpbnRzO1xyXG4gICAgICAgIGMub2Zmc2V0cy5zbGljZSgxKS5mb3JFYWNoKGZ1bmN0aW9uIChvKSB7XHJcbiAgICAgICAgICAgIHZhciB2ID0gX3RoaXMubm9kZXNbby5ub2RlXS52YXJpYWJsZTtcclxuICAgICAgICAgICAgY3MucHVzaChuZXcgdnBzY18xLkNvbnN0cmFpbnQodSwgdiwgby5vZmZzZXQsIHRydWUpKTtcclxuICAgICAgICB9KTtcclxuICAgIH07XHJcbiAgICBQcm9qZWN0aW9uLnByb3RvdHlwZS5jcmVhdGVDb25zdHJhaW50cyA9IGZ1bmN0aW9uIChjb25zdHJhaW50cykge1xyXG4gICAgICAgIHZhciBfdGhpcyA9IHRoaXM7XHJcbiAgICAgICAgdmFyIGlzU2VwID0gZnVuY3Rpb24gKGMpIHsgcmV0dXJuIHR5cGVvZiBjLnR5cGUgPT09ICd1bmRlZmluZWQnIHx8IGMudHlwZSA9PT0gJ3NlcGFyYXRpb24nOyB9O1xyXG4gICAgICAgIHRoaXMueENvbnN0cmFpbnRzID0gY29uc3RyYWludHNcclxuICAgICAgICAgICAgLmZpbHRlcihmdW5jdGlvbiAoYykgeyByZXR1cm4gYy5heGlzID09PSBcInhcIiAmJiBpc1NlcChjKTsgfSlcclxuICAgICAgICAgICAgLm1hcChmdW5jdGlvbiAoYykgeyByZXR1cm4gX3RoaXMuY3JlYXRlU2VwYXJhdGlvbihjKTsgfSk7XHJcbiAgICAgICAgdGhpcy55Q29uc3RyYWludHMgPSBjb25zdHJhaW50c1xyXG4gICAgICAgICAgICAuZmlsdGVyKGZ1bmN0aW9uIChjKSB7IHJldHVybiBjLmF4aXMgPT09IFwieVwiICYmIGlzU2VwKGMpOyB9KVxyXG4gICAgICAgICAgICAubWFwKGZ1bmN0aW9uIChjKSB7IHJldHVybiBfdGhpcy5jcmVhdGVTZXBhcmF0aW9uKGMpOyB9KTtcclxuICAgICAgICBjb25zdHJhaW50c1xyXG4gICAgICAgICAgICAuZmlsdGVyKGZ1bmN0aW9uIChjKSB7IHJldHVybiBjLnR5cGUgPT09ICdhbGlnbm1lbnQnOyB9KVxyXG4gICAgICAgICAgICAuZm9yRWFjaChmdW5jdGlvbiAoYykgeyByZXR1cm4gX3RoaXMuY3JlYXRlQWxpZ25tZW50KGMpOyB9KTtcclxuICAgIH07XHJcbiAgICBQcm9qZWN0aW9uLnByb3RvdHlwZS5zZXR1cFZhcmlhYmxlc0FuZEJvdW5kcyA9IGZ1bmN0aW9uICh4MCwgeTAsIGRlc2lyZWQsIGdldERlc2lyZWQpIHtcclxuICAgICAgICB0aGlzLm5vZGVzLmZvckVhY2goZnVuY3Rpb24gKHYsIGkpIHtcclxuICAgICAgICAgICAgaWYgKHYuZml4ZWQpIHtcclxuICAgICAgICAgICAgICAgIHYudmFyaWFibGUud2VpZ2h0ID0gdi5maXhlZFdlaWdodCA/IHYuZml4ZWRXZWlnaHQgOiAxMDAwO1xyXG4gICAgICAgICAgICAgICAgZGVzaXJlZFtpXSA9IGdldERlc2lyZWQodik7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgZWxzZSB7XHJcbiAgICAgICAgICAgICAgICB2LnZhcmlhYmxlLndlaWdodCA9IDE7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgdmFyIHcgPSAodi53aWR0aCB8fCAwKSAvIDIsIGggPSAodi5oZWlnaHQgfHwgMCkgLyAyO1xyXG4gICAgICAgICAgICB2YXIgaXggPSB4MFtpXSwgaXkgPSB5MFtpXTtcclxuICAgICAgICAgICAgdi5ib3VuZHMgPSBuZXcgUmVjdGFuZ2xlKGl4IC0gdywgaXggKyB3LCBpeSAtIGgsIGl5ICsgaCk7XHJcbiAgICAgICAgfSk7XHJcbiAgICB9O1xyXG4gICAgUHJvamVjdGlvbi5wcm90b3R5cGUueFByb2plY3QgPSBmdW5jdGlvbiAoeDAsIHkwLCB4KSB7XHJcbiAgICAgICAgaWYgKCF0aGlzLnJvb3RHcm91cCAmJiAhKHRoaXMuYXZvaWRPdmVybGFwcyB8fCB0aGlzLnhDb25zdHJhaW50cykpXHJcbiAgICAgICAgICAgIHJldHVybjtcclxuICAgICAgICB0aGlzLnByb2plY3QoeDAsIHkwLCB4MCwgeCwgZnVuY3Rpb24gKHYpIHsgcmV0dXJuIHYucHg7IH0sIHRoaXMueENvbnN0cmFpbnRzLCBnZW5lcmF0ZVhHcm91cENvbnN0cmFpbnRzLCBmdW5jdGlvbiAodikgeyByZXR1cm4gdi5ib3VuZHMuc2V0WENlbnRyZSh4W3YudmFyaWFibGUuaW5kZXhdID0gdi52YXJpYWJsZS5wb3NpdGlvbigpKTsgfSwgZnVuY3Rpb24gKGcpIHtcclxuICAgICAgICAgICAgdmFyIHhtaW4gPSB4W2cubWluVmFyLmluZGV4XSA9IGcubWluVmFyLnBvc2l0aW9uKCk7XHJcbiAgICAgICAgICAgIHZhciB4bWF4ID0geFtnLm1heFZhci5pbmRleF0gPSBnLm1heFZhci5wb3NpdGlvbigpO1xyXG4gICAgICAgICAgICB2YXIgcDIgPSBnLnBhZGRpbmcgLyAyO1xyXG4gICAgICAgICAgICBnLmJvdW5kcy54ID0geG1pbiAtIHAyO1xyXG4gICAgICAgICAgICBnLmJvdW5kcy5YID0geG1heCArIHAyO1xyXG4gICAgICAgIH0pO1xyXG4gICAgfTtcclxuICAgIFByb2plY3Rpb24ucHJvdG90eXBlLnlQcm9qZWN0ID0gZnVuY3Rpb24gKHgwLCB5MCwgeSkge1xyXG4gICAgICAgIGlmICghdGhpcy5yb290R3JvdXAgJiYgIXRoaXMueUNvbnN0cmFpbnRzKVxyXG4gICAgICAgICAgICByZXR1cm47XHJcbiAgICAgICAgdGhpcy5wcm9qZWN0KHgwLCB5MCwgeTAsIHksIGZ1bmN0aW9uICh2KSB7IHJldHVybiB2LnB5OyB9LCB0aGlzLnlDb25zdHJhaW50cywgZ2VuZXJhdGVZR3JvdXBDb25zdHJhaW50cywgZnVuY3Rpb24gKHYpIHsgcmV0dXJuIHYuYm91bmRzLnNldFlDZW50cmUoeVt2LnZhcmlhYmxlLmluZGV4XSA9IHYudmFyaWFibGUucG9zaXRpb24oKSk7IH0sIGZ1bmN0aW9uIChnKSB7XHJcbiAgICAgICAgICAgIHZhciB5bWluID0geVtnLm1pblZhci5pbmRleF0gPSBnLm1pblZhci5wb3NpdGlvbigpO1xyXG4gICAgICAgICAgICB2YXIgeW1heCA9IHlbZy5tYXhWYXIuaW5kZXhdID0gZy5tYXhWYXIucG9zaXRpb24oKTtcclxuICAgICAgICAgICAgdmFyIHAyID0gZy5wYWRkaW5nIC8gMjtcclxuICAgICAgICAgICAgZy5ib3VuZHMueSA9IHltaW4gLSBwMjtcclxuICAgICAgICAgICAgO1xyXG4gICAgICAgICAgICBnLmJvdW5kcy5ZID0geW1heCArIHAyO1xyXG4gICAgICAgIH0pO1xyXG4gICAgfTtcclxuICAgIFByb2plY3Rpb24ucHJvdG90eXBlLnByb2plY3RGdW5jdGlvbnMgPSBmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgdmFyIF90aGlzID0gdGhpcztcclxuICAgICAgICByZXR1cm4gW1xyXG4gICAgICAgICAgICBmdW5jdGlvbiAoeDAsIHkwLCB4KSB7IHJldHVybiBfdGhpcy54UHJvamVjdCh4MCwgeTAsIHgpOyB9LFxyXG4gICAgICAgICAgICBmdW5jdGlvbiAoeDAsIHkwLCB5KSB7IHJldHVybiBfdGhpcy55UHJvamVjdCh4MCwgeTAsIHkpOyB9XHJcbiAgICAgICAgXTtcclxuICAgIH07XHJcbiAgICBQcm9qZWN0aW9uLnByb3RvdHlwZS5wcm9qZWN0ID0gZnVuY3Rpb24gKHgwLCB5MCwgc3RhcnQsIGRlc2lyZWQsIGdldERlc2lyZWQsIGNzLCBnZW5lcmF0ZUNvbnN0cmFpbnRzLCB1cGRhdGVOb2RlQm91bmRzLCB1cGRhdGVHcm91cEJvdW5kcykge1xyXG4gICAgICAgIHRoaXMuc2V0dXBWYXJpYWJsZXNBbmRCb3VuZHMoeDAsIHkwLCBkZXNpcmVkLCBnZXREZXNpcmVkKTtcclxuICAgICAgICBpZiAodGhpcy5yb290R3JvdXAgJiYgdGhpcy5hdm9pZE92ZXJsYXBzKSB7XHJcbiAgICAgICAgICAgIGNvbXB1dGVHcm91cEJvdW5kcyh0aGlzLnJvb3RHcm91cCk7XHJcbiAgICAgICAgICAgIGNzID0gY3MuY29uY2F0KGdlbmVyYXRlQ29uc3RyYWludHModGhpcy5yb290R3JvdXApKTtcclxuICAgICAgICB9XHJcbiAgICAgICAgdGhpcy5zb2x2ZSh0aGlzLnZhcmlhYmxlcywgY3MsIHN0YXJ0LCBkZXNpcmVkKTtcclxuICAgICAgICB0aGlzLm5vZGVzLmZvckVhY2godXBkYXRlTm9kZUJvdW5kcyk7XHJcbiAgICAgICAgaWYgKHRoaXMucm9vdEdyb3VwICYmIHRoaXMuYXZvaWRPdmVybGFwcykge1xyXG4gICAgICAgICAgICB0aGlzLmdyb3Vwcy5mb3JFYWNoKHVwZGF0ZUdyb3VwQm91bmRzKTtcclxuICAgICAgICAgICAgY29tcHV0ZUdyb3VwQm91bmRzKHRoaXMucm9vdEdyb3VwKTtcclxuICAgICAgICB9XHJcbiAgICB9O1xyXG4gICAgUHJvamVjdGlvbi5wcm90b3R5cGUuc29sdmUgPSBmdW5jdGlvbiAodnMsIGNzLCBzdGFydGluZywgZGVzaXJlZCkge1xyXG4gICAgICAgIHZhciBzb2x2ZXIgPSBuZXcgdnBzY18xLlNvbHZlcih2cywgY3MpO1xyXG4gICAgICAgIHNvbHZlci5zZXRTdGFydGluZ1Bvc2l0aW9ucyhzdGFydGluZyk7XHJcbiAgICAgICAgc29sdmVyLnNldERlc2lyZWRQb3NpdGlvbnMoZGVzaXJlZCk7XHJcbiAgICAgICAgc29sdmVyLnNvbHZlKCk7XHJcbiAgICB9O1xyXG4gICAgcmV0dXJuIFByb2plY3Rpb247XHJcbn0oKSk7XHJcbmV4cG9ydHMuUHJvamVjdGlvbiA9IFByb2plY3Rpb247XHJcbi8vIyBzb3VyY2VNYXBwaW5nVVJMPWRhdGE6YXBwbGljYXRpb24vanNvbjtiYXNlNjQsZXlKMlpYSnphVzl1SWpvekxDSm1hV3hsSWpvaWNtVmpkR0Z1WjJ4bExtcHpJaXdpYzI5MWNtTmxVbTl2ZENJNklpSXNJbk52ZFhKalpYTWlPbHNpTGk0dkxpNHZWMlZpUTI5c1lTOXpjbU12Y21WamRHRnVaMnhsTG5SeklsMHNJbTVoYldWeklqcGJYU3dpYldGd2NHbHVaM01pT2lJN096czdPenM3T3pzN096czdPenRCUVVGQkxDdENRVUZ0UkR0QlFVTnVSQ3h0UTBGQkswSTdRVUZyUWpOQ0xGTkJRV2RDTEd0Q1FVRnJRaXhEUVVGRExFTkJRV3RDTzBsQlEycEVMRU5CUVVNc1EwRkJReXhOUVVGTkxFZEJRVWNzVDBGQlR5eERRVUZETEVOQlFVTXNUVUZCVFN4TFFVRkxMRmRCUVZjc1EwRkJReXhEUVVGRE8xRkJRM2hETEVOQlFVTXNRMEZCUXl4TlFVRk5MRU5CUVVNc1RVRkJUU3hEUVVGRExGVkJRVU1zUTBGQldTeEZRVUZGTEVOQlFVTXNTVUZCU3l4UFFVRkJMRU5CUVVNc1EwRkJReXhOUVVGTkxFTkJRVU1zUzBGQlN5eERRVUZETEVOQlFVTXNRMEZCUXl4RlFVRnFRaXhEUVVGcFFpeEZRVUZGTEZOQlFWTXNRMEZCUXl4TFFVRkxMRVZCUVVVc1EwRkJReXhEUVVGRExFTkJRVU03VVVGRE5VVXNVMEZCVXl4RFFVRkRMRXRCUVVzc1JVRkJSU3hEUVVGRE8wbEJRM1JDTEVsQlFVa3NUMEZCVHl4RFFVRkRMRU5CUVVNc1RVRkJUU3hMUVVGTExGZEJRVmM3VVVGREwwSXNRMEZCUXl4RFFVRkRMRTFCUVUwc1IwRkJZeXhEUVVGRExFTkJRVU1zVFVGQlRTeERRVUZETEUxQlFVMHNRMEZCUXl4VlFVRkRMRU5CUVZrc1JVRkJSU3hEUVVGRExFbEJRVXNzVDBGQlFTeHJRa0ZCYTBJc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eExRVUZMTEVOQlFVTXNRMEZCUXl4RFFVRkRMRVZCUVRsQ0xFTkJRVGhDTEVWQlFVVXNRMEZCUXl4RFFVRkRMRTFCUVUwc1EwRkJReXhEUVVGRE8wbEJRM3BITEVOQlFVTXNRMEZCUXl4TlFVRk5MRWRCUVVjc1EwRkJReXhEUVVGRExFMUJRVTBzUTBGQlF5eFBRVUZQTEVOQlFVTXNRMEZCUXl4RFFVRkRMRTlCUVU4c1EwRkJReXhEUVVGRE8wbEJRM1pETEU5QlFVOHNRMEZCUXl4RFFVRkRMRTFCUVUwc1EwRkJRenRCUVVOd1FpeERRVUZETzBGQlVrUXNaMFJCVVVNN1FVRkZSRHRKUVVOSkxHMUNRVU5YTEVOQlFWTXNSVUZEVkN4RFFVRlRMRVZCUTFRc1EwRkJVeXhGUVVOVUxFTkJRVk03VVVGSVZDeE5RVUZETEVkQlFVUXNRMEZCUXl4RFFVRlJPMUZCUTFRc1RVRkJReXhIUVVGRUxFTkJRVU1zUTBGQlVUdFJRVU5VTEUxQlFVTXNSMEZCUkN4RFFVRkRMRU5CUVZFN1VVRkRWQ3hOUVVGRExFZEJRVVFzUTBGQlF5eERRVUZSTzBsQlFVa3NRMEZCUXp0SlFVVnNRaXhsUVVGTExFZEJRVm9zWTBGQk5FSXNUMEZCVHl4SlFVRkpMRk5CUVZNc1EwRkJReXhOUVVGTkxFTkJRVU1zYVVKQlFXbENMRVZCUVVVc1RVRkJUU3hEUVVGRExHbENRVUZwUWl4RlFVRkZMRTFCUVUwc1EwRkJReXhwUWtGQmFVSXNSVUZCUlN4TlFVRk5MRU5CUVVNc2FVSkJRV2xDTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNN1NVRkZNMG9zYzBKQlFVVXNSMEZCUml4alFVRmxMRTlCUVU4c1EwRkJReXhKUVVGSkxFTkJRVU1zUTBGQlF5eEhRVUZITEVsQlFVa3NRMEZCUXl4RFFVRkRMRU5CUVVNc1IwRkJSeXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETzBsQlJUbERMSE5DUVVGRkxFZEJRVVlzWTBGQlpTeFBRVUZQTEVOQlFVTXNTVUZCU1N4RFFVRkRMRU5CUVVNc1IwRkJSeXhKUVVGSkxFTkJRVU1zUTBGQlF5eERRVUZETEVkQlFVY3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJRenRKUVVVNVF5dzBRa0ZCVVN4SFFVRlNMRlZCUVZNc1EwRkJXVHRSUVVOcVFpeEpRVUZKTEVWQlFVVXNSMEZCUnl4SlFVRkpMRU5CUVVNc1JVRkJSU3hGUVVGRkxFVkJRVVVzUlVGQlJTeEhRVUZITEVOQlFVTXNRMEZCUXl4RlFVRkZMRVZCUVVVc1EwRkJRenRSUVVOb1F5eEpRVUZKTEVWQlFVVXNTVUZCU1N4RlFVRkZMRWxCUVVrc1EwRkJReXhEUVVGRExFTkJRVU1zUjBGQlJ5eEpRVUZKTEVOQlFVTXNRMEZCUXp0WlFVRkZMRTlCUVU4c1NVRkJTU3hEUVVGRExFTkJRVU1zUjBGQlJ5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRPMUZCUTJ4RUxFbEJRVWtzUlVGQlJTeEpRVUZKTEVWQlFVVXNTVUZCU1N4SlFVRkpMRU5CUVVNc1EwRkJReXhIUVVGSExFTkJRVU1zUTBGQlF5eERRVUZETzFsQlFVVXNUMEZCVHl4RFFVRkRMRU5CUVVNc1EwRkJReXhIUVVGSExFbEJRVWtzUTBGQlF5eERRVUZETEVOQlFVTTdVVUZEYkVRc1QwRkJUeXhEUVVGRExFTkJRVU03U1VGRFlpeERRVUZETzBsQlJVUXNORUpCUVZFc1IwRkJVaXhWUVVGVExFTkJRVms3VVVGRGFrSXNTVUZCU1N4RlFVRkZMRWRCUVVjc1NVRkJTU3hEUVVGRExFVkJRVVVzUlVGQlJTeEZRVUZGTEVWQlFVVXNSMEZCUnl4RFFVRkRMRU5CUVVNc1JVRkJSU3hGUVVGRkxFTkJRVU03VVVGRGFFTXNTVUZCU1N4RlFVRkZMRWxCUVVrc1JVRkJSU3hKUVVGSkxFTkJRVU1zUTBGQlF5eERRVUZETEVkQlFVY3NTVUZCU1N4RFFVRkRMRU5CUVVNN1dVRkJSU3hQUVVGUExFbEJRVWtzUTBGQlF5eERRVUZETEVkQlFVY3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJRenRSUVVOc1JDeEpRVUZKTEVWQlFVVXNTVUZCU1N4RlFVRkZMRWxCUVVrc1NVRkJTU3hEUVVGRExFTkJRVU1zUjBGQlJ5eERRVUZETEVOQlFVTXNRMEZCUXp0WlFVRkZMRTlCUVU4c1EwRkJReXhEUVVGRExFTkJRVU1zUjBGQlJ5eEpRVUZKTEVOQlFVTXNRMEZCUXl4RFFVRkRPMUZCUTJ4RUxFOUJRVThzUTBGQlF5eERRVUZETzBsQlEySXNRMEZCUXp0SlFVVkVMRGhDUVVGVkxFZEJRVllzVlVGQlZ5eEZRVUZWTzFGQlEycENMRWxCUVVrc1JVRkJSU3hIUVVGSExFVkJRVVVzUjBGQlJ5eEpRVUZKTEVOQlFVTXNSVUZCUlN4RlFVRkZMRU5CUVVNN1VVRkRlRUlzU1VGQlNTeERRVUZETEVOQlFVTXNTVUZCU1N4RlFVRkZMRU5CUVVNN1VVRkRZaXhKUVVGSkxFTkJRVU1zUTBGQlF5eEpRVUZKTEVWQlFVVXNRMEZCUXp0SlFVTnFRaXhEUVVGRE8wbEJSVVFzT0VKQlFWVXNSMEZCVml4VlFVRlhMRVZCUVZVN1VVRkRha0lzU1VGQlNTeEZRVUZGTEVkQlFVY3NSVUZCUlN4SFFVRkhMRWxCUVVrc1EwRkJReXhGUVVGRkxFVkJRVVVzUTBGQlF6dFJRVU40UWl4SlFVRkpMRU5CUVVNc1EwRkJReXhKUVVGSkxFVkJRVVVzUTBGQlF6dFJRVU5pTEVsQlFVa3NRMEZCUXl4RFFVRkRMRWxCUVVrc1JVRkJSU3hEUVVGRE8wbEJRMnBDTEVOQlFVTTdTVUZGUkN4NVFrRkJTeXhIUVVGTU8xRkJRMGtzVDBGQlR5eEpRVUZKTEVOQlFVTXNRMEZCUXl4SFFVRkhMRWxCUVVrc1EwRkJReXhEUVVGRExFTkJRVU03U1VGRE0wSXNRMEZCUXp0SlFVVkVMREJDUVVGTkxFZEJRVTQ3VVVGRFNTeFBRVUZQTEVsQlFVa3NRMEZCUXl4RFFVRkRMRWRCUVVjc1NVRkJTU3hEUVVGRExFTkJRVU1zUTBGQlF6dEpRVU16UWl4RFFVRkRPMGxCUlVRc2VVSkJRVXNzUjBGQlRDeFZRVUZOTEVOQlFWazdVVUZEWkN4UFFVRlBMRWxCUVVrc1UwRkJVeXhEUVVGRExFbEJRVWtzUTBGQlF5eEhRVUZITEVOQlFVTXNTVUZCU1N4RFFVRkRMRU5CUVVNc1JVRkJSU3hEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVWQlFVVXNTVUZCU1N4RFFVRkRMRWRCUVVjc1EwRkJReXhKUVVGSkxFTkJRVU1zUTBGQlF5eEZRVUZGTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1JVRkJSU3hKUVVGSkxFTkJRVU1zUjBGQlJ5eERRVUZETEVsQlFVa3NRMEZCUXl4RFFVRkRMRVZCUVVVc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eEZRVUZGTEVsQlFVa3NRMEZCUXl4SFFVRkhMRU5CUVVNc1NVRkJTU3hEUVVGRExFTkJRVU1zUlVGQlJTeERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJRenRKUVVOeVNDeERRVUZETzBsQlYwUXNjVU5CUVdsQ0xFZEJRV3BDTEZWQlFXdENMRVZCUVZVc1JVRkJSU3hGUVVGVkxFVkJRVVVzUlVGQlZTeEZRVUZGTEVWQlFWVTdVVUZETlVRc1NVRkJTU3hMUVVGTExFZEJRVWNzUTBGQlF5eERRVUZETEVsQlFVa3NRMEZCUXl4RFFVRkRMRVZCUVVVc1NVRkJTU3hEUVVGRExFTkJRVU1zUlVGQlJTeEpRVUZKTEVOQlFVTXNRMEZCUXl4RlFVRkZMRWxCUVVrc1EwRkJReXhEUVVGRExFTkJRVU03V1VGRGNrTXNRMEZCUXl4SlFVRkpMRU5CUVVNc1EwRkJReXhGUVVGRkxFbEJRVWtzUTBGQlF5eERRVUZETEVWQlFVVXNTVUZCU1N4RFFVRkRMRU5CUVVNc1JVRkJSU3hKUVVGSkxFTkJRVU1zUTBGQlF5eERRVUZETzFsQlEyaERMRU5CUVVNc1NVRkJTU3hEUVVGRExFTkJRVU1zUlVGQlJTeEpRVUZKTEVOQlFVTXNRMEZCUXl4RlFVRkZMRWxCUVVrc1EwRkJReXhEUVVGRExFVkJRVVVzU1VGQlNTeERRVUZETEVOQlFVTXNRMEZCUXp0WlFVTndReXhEUVVGRExFbEJRVWtzUTBGQlF5eERRVUZETEVWQlFVVXNTVUZCU1N4RFFVRkRMRU5CUVVNc1JVRkJSU3hKUVVGSkxFTkJRVU1zUTBGQlF5eEZRVUZGTEVsQlFVa3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRE8xRkJRM1JETEVsQlFVa3NZVUZCWVN4SFFVRkhMRVZCUVVVc1EwRkJRenRSUVVOMlFpeExRVUZMTEVsQlFVa3NRMEZCUXl4SFFVRkhMRU5CUVVNc1JVRkJSU3hEUVVGRExFZEJRVWNzUTBGQlF5eEZRVUZGTEVWQlFVVXNRMEZCUXl4RlFVRkZPMWxCUTNoQ0xFbEJRVWtzUTBGQlF5eEhRVUZITEZOQlFWTXNRMEZCUXl4blFrRkJaMElzUTBGQlF5eEZRVUZGTEVWQlFVVXNSVUZCUlN4RlFVRkZMRVZCUVVVc1JVRkJSU3hGUVVGRkxFVkJRVVVzUzBGQlN5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhGUVVGRkxFdEJRVXNzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1JVRkJSU3hMUVVGTExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRVZCUVVVc1MwRkJTeXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNN1dVRkRka2NzU1VGQlNTeERRVUZETEV0QlFVc3NTVUZCU1R0blFrRkJSU3hoUVVGaExFTkJRVU1zU1VGQlNTeERRVUZETEVWQlFVVXNRMEZCUXl4RlFVRkZMRU5CUVVNc1EwRkJReXhEUVVGRExFVkJRVVVzUTBGQlF5eEZRVUZGTEVOQlFVTXNRMEZCUXl4RFFVRkRMRVZCUVVVc1EwRkJReXhEUVVGRE8xTkJRekZFTzFGQlEwUXNUMEZCVHl4aFFVRmhMRU5CUVVNN1NVRkRla0lzUTBGQlF6dEpRVlZFTEcxRFFVRmxMRWRCUVdZc1ZVRkJaMElzUlVGQlZTeEZRVUZGTEVWQlFWVTdVVUZEYkVNc1NVRkJTU3hKUVVGSkxFZEJRVWNzU1VGQlNTeERRVUZETEdsQ1FVRnBRaXhEUVVGRExFbEJRVWtzUTBGQlF5eEZRVUZGTEVWQlFVVXNSVUZCUlN4SlFVRkpMRU5CUVVNc1JVRkJSU3hGUVVGRkxFVkJRVVVzUlVGQlJTeEZRVUZGTEVWQlFVVXNRMEZCUXl4RFFVRkRPMUZCUTJoRkxFOUJRVThzU1VGQlNTeERRVUZETEUxQlFVMHNSMEZCUnl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFbEJRVWtzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1NVRkJTU3hEUVVGRE8wbEJRelZETEVOQlFVTTdTVUZGUkN3MFFrRkJVU3hIUVVGU08xRkJRMGtzVDBGQlR6dFpRVU5JTEVWQlFVVXNRMEZCUXl4RlFVRkZMRWxCUVVrc1EwRkJReXhEUVVGRExFVkJRVVVzUTBGQlF5eEZRVUZGTEVsQlFVa3NRMEZCUXl4RFFVRkRMRVZCUVVVN1dVRkRlRUlzUlVGQlJTeERRVUZETEVWQlFVVXNTVUZCU1N4RFFVRkRMRU5CUVVNc1JVRkJSU3hEUVVGRExFVkJRVVVzU1VGQlNTeERRVUZETEVOQlFVTXNSVUZCUlR0WlFVTjRRaXhGUVVGRkxFTkJRVU1zUlVGQlJTeEpRVUZKTEVOQlFVTXNRMEZCUXl4RlFVRkZMRU5CUVVNc1JVRkJSU3hKUVVGSkxFTkJRVU1zUTBGQlF5eEZRVUZGTzFsQlEzaENMRVZCUVVVc1EwRkJReXhGUVVGRkxFbEJRVWtzUTBGQlF5eERRVUZETEVWQlFVVXNRMEZCUXl4RlFVRkZMRWxCUVVrc1EwRkJReXhEUVVGRExFVkJRVVU3VTBGQlF5eERRVUZETzBsQlEyeERMRU5CUVVNN1NVRkZUU3d3UWtGQlowSXNSMEZCZGtJc1ZVRkRTU3hGUVVGVkxFVkJRVVVzUlVGQlZTeEZRVU4wUWl4RlFVRlZMRVZCUVVVc1JVRkJWU3hGUVVOMFFpeEZRVUZWTEVWQlFVVXNSVUZCVlN4RlFVTjBRaXhGUVVGVkxFVkJRVVVzUlVGQlZUdFJRVU4wUWl4SlFVRkpMRWxCUVVrc1IwRkJSeXhGUVVGRkxFZEJRVWNzUlVGQlJTeEZRVUZGTEVsQlFVa3NSMEZCUnl4RlFVRkZMRWRCUVVjc1JVRkJSU3hGUVVNNVFpeEpRVUZKTEVkQlFVY3NSVUZCUlN4SFFVRkhMRVZCUVVVc1JVRkJSU3hKUVVGSkxFZEJRVWNzUlVGQlJTeEhRVUZITEVWQlFVVXNSVUZET1VJc1YwRkJWeXhIUVVGSExFbEJRVWtzUjBGQlJ5eEpRVUZKTEVkQlFVY3NTVUZCU1N4SFFVRkhMRWxCUVVrc1EwRkJRenRSUVVNMVF5eEpRVUZKTEZkQlFWY3NTVUZCU1N4RFFVRkRPMWxCUVVVc1QwRkJUeXhKUVVGSkxFTkJRVU03VVVGRGJFTXNTVUZCU1N4SlFVRkpMRWRCUVVjc1JVRkJSU3hIUVVGSExFVkJRVVVzUlVGQlJTeEpRVUZKTEVkQlFVY3NSVUZCUlN4SFFVRkhMRVZCUVVVc1JVRkRPVUlzU1VGQlNTeEhRVUZITEVsQlFVa3NSMEZCUnl4SlFVRkpMRWRCUVVjc1NVRkJTU3hIUVVGSExFbEJRVWtzUlVGRGFFTXNRMEZCUXl4SFFVRkhMRWxCUVVrc1IwRkJSeXhYUVVGWExFVkJRM1JDTEVsQlFVa3NSMEZCUnl4SlFVRkpMRWRCUVVjc1NVRkJTU3hIUVVGSExFbEJRVWtzUjBGQlJ5eEpRVUZKTEVWQlEyaERMRU5CUVVNc1IwRkJSeXhKUVVGSkxFZEJRVWNzVjBGQlZ5eERRVUZETzFGQlF6TkNMRWxCUVVrc1EwRkJReXhKUVVGSkxFTkJRVU1zU1VGQlNTeERRVUZETEVsQlFVa3NRMEZCUXl4SlFVRkpMRU5CUVVNc1NVRkJTU3hEUVVGRExFbEJRVWtzUTBGQlF5eEpRVUZKTEVOQlFVTXNSVUZCUlR0WlFVTjBReXhQUVVGUE8yZENRVU5JTEVOQlFVTXNSVUZCUlN4RlFVRkZMRWRCUVVjc1EwRkJReXhIUVVGSExFbEJRVWs3WjBKQlEyaENMRU5CUVVNc1JVRkJSU3hGUVVGRkxFZEJRVWNzUTBGQlF5eEhRVUZITEVsQlFVazdZVUZEYmtJc1EwRkJRenRUUVVOTU8xRkJRMFFzVDBGQlR5eEpRVUZKTEVOQlFVTTdTVUZEYUVJc1EwRkJRenRKUVVWRUxESkNRVUZQTEVkQlFWQXNWVUZCVVN4SFFVRlhPMUZCUTJZc1QwRkJUeXhKUVVGSkxGTkJRVk1zUTBGQlF5eEpRVUZKTEVOQlFVTXNRMEZCUXl4SFFVRkhMRWRCUVVjc1JVRkJSU3hKUVVGSkxFTkJRVU1zUTBGQlF5eEhRVUZITEVkQlFVY3NSVUZCUlN4SlFVRkpMRU5CUVVNc1EwRkJReXhIUVVGSExFZEJRVWNzUlVGQlJTeEpRVUZKTEVOQlFVTXNRMEZCUXl4SFFVRkhMRWRCUVVjc1EwRkJReXhEUVVGRE8wbEJRMnBHTEVOQlFVTTdTVUZEVEN4blFrRkJRenRCUVVGRUxFTkJRVU1zUVVGNFNFUXNTVUYzU0VNN1FVRjRTRmtzT0VKQlFWTTdRVUZ4U1hSQ0xGTkJRV2RDTEdWQlFXVXNRMEZCUXl4TlFVRnBRaXhGUVVGRkxFMUJRV2xDTEVWQlFVVXNSVUZCVlR0SlFVVTFSU3hKUVVGTkxFVkJRVVVzUjBGQlJ5eE5RVUZOTEVOQlFVTXNaVUZCWlN4RFFVRkRMRTFCUVUwc1EwRkJReXhGUVVGRkxFVkJRVVVzUlVGQlJTeE5RVUZOTEVOQlFVTXNSVUZCUlN4RlFVRkZMRU5CUVVNc1NVRkJTU3hGUVVGRkxFTkJRVU1zUlVGQlJTeE5RVUZOTEVOQlFVTXNSVUZCUlN4RlFVRkZMRVZCUVVVc1EwRkJReXhGUVVGRkxFMUJRVTBzUTBGQlF5eEZRVUZGTEVWQlFVVXNSVUZCUlN4RlFVTTNSaXhGUVVGRkxFZEJRVWNzVFVGQlRTeERRVUZETEdWQlFXVXNRMEZCUXl4TlFVRk5MRU5CUVVNc1JVRkJSU3hGUVVGRkxFVkJRVVVzVFVGQlRTeERRVUZETEVWQlFVVXNSVUZCUlN4RFFVRkRMRWxCUVVrc1JVRkJSU3hEUVVGRExFVkJRVVVzVFVGQlRTeERRVUZETEVWQlFVVXNSVUZCUlN4RlFVRkZMRU5CUVVNc1JVRkJSU3hOUVVGTkxFTkJRVU1zUlVGQlJTeEZRVUZGTEVWQlFVVXNSVUZETTBZc1JVRkJSU3hIUVVGSExFVkJRVVVzUTBGQlF5eERRVUZETEVkQlFVY3NSVUZCUlN4RFFVRkRMRU5CUVVNc1JVRkRhRUlzUlVGQlJTeEhRVUZITEVWQlFVVXNRMEZCUXl4RFFVRkRMRWRCUVVjc1JVRkJSU3hEUVVGRExFTkJRVU1zUlVGRGFFSXNRMEZCUXl4SFFVRkhMRWxCUVVrc1EwRkJReXhKUVVGSkxFTkJRVU1zUlVGQlJTeEhRVUZITEVWQlFVVXNSMEZCUnl4RlFVRkZMRWRCUVVjc1JVRkJSU3hEUVVGRExFVkJRVVVzUlVGQlJTeEhRVUZITEVOQlFVTXNSMEZCUnl4RlFVRkZMRU5CUVVNN1NVRkRiRVFzVDBGQlR6dFJRVU5JTEd0Q1FVRnJRaXhGUVVGRkxFVkJRVVU3VVVGRGRFSXNhMEpCUVd0Q0xFVkJRVVVzUlVGQlJUdFJRVU4wUWl4VlFVRlZMRVZCUVVVc1JVRkJSU3hEUVVGRExFVkJRVVVzUlVGQlJTeERRVUZETEVOQlFVTXNSMEZCUnl4RlFVRkZMRWRCUVVjc1JVRkJSU3hIUVVGSExFTkJRVU1zUlVGQlJTeERRVUZETEVWQlFVVXNSVUZCUlN4RFFVRkRMRU5CUVVNc1IwRkJSeXhGUVVGRkxFZEJRVWNzUlVGQlJTeEhRVUZITEVOQlFVTXNSVUZCUlR0TFFVTXZSQ3hEUVVGQk8wRkJRMHdzUTBGQlF6dEJRVnBFTERCRFFWbERPMEZCVjBRc1UwRkJaMElzVlVGQlZTeERRVUZETEVOQlFUSkNMRVZCUVVVc1RVRkJhVUlzUlVGQlJTeEZRVUZWTzBsQlEycEdMRWxCUVVrc1JVRkJSU3hIUVVGSExFMUJRVTBzUTBGQlF5eGxRVUZsTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1JVRkJSU3hEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTTdTVUZETVVNc1NVRkJTU3hEUVVGRExFVkJRVVU3VVVGQlJTeEZRVUZGTEVkQlFVY3NSVUZCUlN4RFFVRkRMRVZCUVVVc1RVRkJUU3hEUVVGRExFVkJRVVVzUlVGQlJTeEZRVUZGTEVOQlFVTXNSVUZCUlN4TlFVRk5MRU5CUVVNc1JVRkJSU3hGUVVGRkxFVkJRVVVzUTBGQlF6dEpRVU5xUkN4SlFVRkpMRVZCUVVVc1IwRkJSeXhGUVVGRkxFTkJRVU1zUTBGQlF5eEhRVUZITEVOQlFVTXNRMEZCUXl4RFFVRkRMRVZCUTJZc1JVRkJSU3hIUVVGSExFVkJRVVVzUTBGQlF5eERRVUZETEVkQlFVY3NRMEZCUXl4RFFVRkRMRU5CUVVNc1JVRkRaaXhEUVVGRExFZEJRVWNzU1VGQlNTeERRVUZETEVsQlFVa3NRMEZCUXl4RlFVRkZMRWRCUVVjc1JVRkJSU3hIUVVGSExFVkJRVVVzUjBGQlJ5eEZRVUZGTEVOQlFVTXNRMEZCUXp0SlFVTnlReXhQUVVGUExFVkJRVVVzUTBGQlF5eEZRVUZGTEVWQlFVVXNRMEZCUXl4RFFVRkRMRWRCUVVjc1JVRkJSU3hIUVVGSExFVkJRVVVzUjBGQlJ5eERRVUZETEVWQlFVVXNRMEZCUXl4RlFVRkZMRVZCUVVVc1EwRkJReXhEUVVGRExFZEJRVWNzUlVGQlJTeEhRVUZITEVWQlFVVXNSMEZCUnl4RFFVRkRMRVZCUVVVc1EwRkJRenRCUVVNMVJDeERRVUZETzBGQlVFUXNaME5CVDBNN1FVRkZSRHRKUVVsSkxHTkJRVzFDTEVOQlFWY3NSVUZCVXl4RFFVRlpMRVZCUVZNc1IwRkJWenRSUVVGd1JDeE5RVUZETEVkQlFVUXNRMEZCUXl4RFFVRlZPMUZCUVZNc1RVRkJReXhIUVVGRUxFTkJRVU1zUTBGQlZ6dFJRVUZUTEZGQlFVY3NSMEZCU0N4SFFVRkhMRU5CUVZFN1VVRkRia1VzU1VGQlNTeERRVUZETEVsQlFVa3NSMEZCUnl4VlFVRlZMRVZCUVVVc1EwRkJRenRSUVVONlFpeEpRVUZKTEVOQlFVTXNTVUZCU1N4SFFVRkhMRlZCUVZVc1JVRkJSU3hEUVVGRE8wbEJRemRDTEVOQlFVTTdTVUZEVEN4WFFVRkRPMEZCUVVRc1EwRkJReXhCUVZKRUxFbEJVVU03UVVGRlJEdEpRVU5KTEdWQlFXMUNMRTFCUVdVc1JVRkJVeXhEUVVGUExFVkJRVk1zUjBGQlZ6dFJRVUZ1UkN4WFFVRk5MRWRCUVU0c1RVRkJUU3hEUVVGVE8xRkJRVk1zVFVGQlF5eEhRVUZFTEVOQlFVTXNRMEZCVFR0UlFVRlRMRkZCUVVjc1IwRkJTQ3hIUVVGSExFTkJRVkU3U1VGQlJ5eERRVUZETzBsQlF6bEZMRmxCUVVNN1FVRkJSQ3hEUVVGRExFRkJSa1FzU1VGRlF6dEJRVVZFTEZOQlFWTXNZVUZCWVN4RFFVRkRMRU5CUVZFc1JVRkJSU3hEUVVGUk8wbEJRM0pETEVsQlFVa3NRMEZCUXl4RFFVRkRMRWRCUVVjc1IwRkJSeXhEUVVGRExFTkJRVU1zUjBGQlJ5eEZRVUZGTzFGQlEyWXNUMEZCVHl4RFFVRkRMRU5CUVVNN1MwRkRXanRKUVVORUxFbEJRVWtzUTBGQlF5eERRVUZETEVkQlFVY3NSMEZCUnl4RFFVRkRMRU5CUVVNc1IwRkJSeXhGUVVGRk8xRkJRMllzVDBGQlR5eERRVUZETEVOQlFVTXNRMEZCUXp0TFFVTmlPMGxCUTBRc1NVRkJTU3hEUVVGRExFTkJRVU1zVFVGQlRTeEZRVUZGTzFGQlJWWXNUMEZCVHl4RFFVRkRMRU5CUVVNc1EwRkJRenRMUVVOaU8wbEJRMFFzU1VGQlNTeERRVUZETEVOQlFVTXNUVUZCVFN4RlFVRkZPMUZCUlZZc1QwRkJUeXhEUVVGRExFTkJRVU03UzBGRFdqdEpRVU5FTEU5QlFVOHNRMEZCUXl4RFFVRkRPMEZCUTJJc1EwRkJRenRCUVVWRUxGTkJRVk1zVlVGQlZUdEpRVU5tTEU5QlFVOHNTVUZCU1N4bFFVRk5MRU5CUVU4c1ZVRkJReXhEUVVGRExFVkJRVVVzUTBGQlF5eEpRVUZMTEU5QlFVRXNRMEZCUXl4RFFVRkRMRWRCUVVjc1IwRkJSeXhEUVVGRExFTkJRVU1zUjBGQlJ5eEZRVUZpTEVOQlFXRXNRMEZCUXl4RFFVRkRPMEZCUTNKRUxFTkJRVU03UVVGWFJDeEpRVUZKTEV0QlFVc3NSMEZCYTBJN1NVRkRka0lzVTBGQlV5eEZRVUZGTEZWQlFVRXNRMEZCUXl4SlFVRkhMRTlCUVVFc1EwRkJReXhEUVVGRExFVkJRVVVzUlVGQlJTeEZRVUZPTEVOQlFVMDdTVUZEY2tJc1QwRkJUeXhGUVVGRkxGVkJRVUVzUTBGQlF5eEpRVUZITEU5QlFVRXNRMEZCUXl4RFFVRkRMRU5CUVVNc1JVRkJTQ3hEUVVGSE8wbEJRMmhDTEZGQlFWRXNSVUZCUlN4VlFVRkJMRU5CUVVNc1NVRkJSeXhQUVVGQkxFTkJRVU1zUTBGQlF5eERRVUZETEVWQlFVZ3NRMEZCUnp0SlFVTnFRaXhQUVVGUExFVkJRVVVzVlVGQlFTeERRVUZETEVsQlFVY3NUMEZCUVN4RFFVRkRMRU5CUVVNc1MwRkJTeXhGUVVGRkxFVkJRVlFzUTBGQlV6dEpRVU4wUWl4UlFVRlJMRVZCUVVVc1ZVRkJReXhKUVVGSkxFVkJRVVVzUzBGQlN5eEZRVUZGTEUxQlFVMHNSVUZCUlN4SlFVRkpMRWxCUVVzc1QwRkJRU3hKUVVGSkxGTkJRVk1zUTBGQlF5eE5RVUZOTEVkQlFVY3NTVUZCU1N4SFFVRkhMRU5CUVVNc1JVRkJSU3hOUVVGTkxFZEJRVWNzU1VGQlNTeEhRVUZITEVOQlFVTXNSVUZCUlN4SlFVRkpMRVZCUVVVc1MwRkJTeXhEUVVGRExFVkJRV2hGTEVOQlFXZEZPMGxCUTNwSExHTkJRV01zUlVGQlJTeGxRVUZsTzBOQlEyeERMRU5CUVVNN1FVRkZSaXhKUVVGSkxFdEJRVXNzUjBGQmEwSTdTVUZEZGtJc1UwRkJVeXhGUVVGRkxGVkJRVUVzUTBGQlF5eEpRVUZITEU5QlFVRXNRMEZCUXl4RFFVRkRMRVZCUVVVc1JVRkJSU3hGUVVGT0xFTkJRVTA3U1VGRGNrSXNUMEZCVHl4RlFVRkZMRlZCUVVFc1EwRkJReXhKUVVGSExFOUJRVUVzUTBGQlF5eERRVUZETEVOQlFVTXNSVUZCU0N4RFFVRkhPMGxCUTJoQ0xGRkJRVkVzUlVGQlJTeFZRVUZCTEVOQlFVTXNTVUZCUnl4UFFVRkJMRU5CUVVNc1EwRkJReXhEUVVGRExFVkJRVWdzUTBGQlJ6dEpRVU5xUWl4UFFVRlBMRVZCUVVVc1ZVRkJRU3hEUVVGRExFbEJRVWNzVDBGQlFTeERRVUZETEVOQlFVTXNUVUZCVFN4RlFVRkZMRVZCUVZZc1EwRkJWVHRKUVVOMlFpeFJRVUZSTEVWQlFVVXNWVUZCUXl4SlFVRkpMRVZCUVVVc1MwRkJTeXhGUVVGRkxFMUJRVTBzUlVGQlJTeEpRVUZKTEVsQlFVc3NUMEZCUVN4SlFVRkpMRk5CUVZNc1EwRkJReXhKUVVGSkxFVkJRVVVzUzBGQlN5eEZRVUZGTEUxQlFVMHNSMEZCUnl4SlFVRkpMRWRCUVVjc1EwRkJReXhGUVVGRkxFMUJRVTBzUjBGQlJ5eEpRVUZKTEVkQlFVY3NRMEZCUXl4RFFVRkRMRVZCUVdoRkxFTkJRV2RGTzBsQlEzcEhMR05CUVdNc1JVRkJSU3hsUVVGbE8wTkJRMnhETEVOQlFVTTdRVUZGUml4VFFVRlRMSGRDUVVGM1FpeERRVUZETEVsQlFYRkNMRVZCUVVVc1EwRkJaMElzUlVGQlJTeE5RVUZqTEVWQlFVVXNWMEZCTkVJN1NVRkJOVUlzTkVKQlFVRXNSVUZCUVN4dFFrRkJORUk3U1VGRmJrZ3NTVUZCU1N4UFFVRlBMRWRCUVVjc1NVRkJTU3hEUVVGRExFOUJRVThzUlVGRGRFSXNSVUZCUlN4SFFVRkhMRTlCUVU4c1NVRkJTU3hEUVVGRExFMUJRVTBzUzBGQlN5eFhRVUZYTEVOQlFVTXNRMEZCUXl4RFFVRkRMRWxCUVVrc1EwRkJReXhOUVVGTkxFTkJRVU1zVFVGQlRTeERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRVZCUTJoRkxFVkJRVVVzUjBGQlJ5eFBRVUZQTEVsQlFVa3NRMEZCUXl4TlFVRk5MRXRCUVVzc1YwRkJWeXhEUVVGRExFTkJRVU1zUTBGQlF5eEpRVUZKTEVOQlFVTXNUVUZCVFN4RFFVRkRMRTFCUVUwc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eEZRVU5vUlN4blFrRkJaMElzUjBGQmFVSXNRMEZCUXl4RlFVRkZMRU5CUVVNc1EwRkJReXhEUVVGRExFVkJRVVU3VVVGRGVrTXNRMEZCUXl4RFFVRkRMRWxCUVVrc1EwRkJReXhOUVVGTkxFTkJRVU1zVFVGQlRTeERRVUZETEZWQlFVTXNSMEZCYVVJc1JVRkJSU3hEUVVGRExFbEJRVXNzVDBGQlFTeEhRVUZITEVOQlFVTXNUVUZCVFN4RFFVRkRMSGRDUVVGM1FpeERRVUZETEVOQlFVTXNSVUZCUlN4RFFVRkRMRVZCUVVVc1RVRkJUU3hGUVVGRkxFbEJRVWtzUTBGQlF5eERRVUZETEVWQlFYaEVMRU5CUVhkRUxFVkJRVVVzUlVGQlJTeERRVUZETEVWQlF6VkhMRU5CUVVNc1IwRkJSeXhEUVVGRExGZEJRVmNzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUjBGQlJ5eEZRVUZGTEVkQlFVY3NSVUZCUlN4RlFVTnVReXhGUVVGRkxFZEJRV1VzU1VGQlNTeExRVUZMTEVOQlFVTXNRMEZCUXl4RFFVRkRMRVZCUXpkQ0xFVkJRVVVzUjBGQlowSXNTVUZCU1N4TFFVRkxMRU5CUVVNc1EwRkJReXhEUVVGRExFVkJRemxDTEVOQlFVTXNSMEZCUnl4RFFVRkRMRVZCUTB3c1IwRkJSeXhIUVVGSExGVkJRVU1zUTBGQlF5eEZRVUZGTEVOQlFVTXNTVUZCVHl4RlFVRkZMRU5CUVVNc1EwRkJReXhEUVVGRExFZEJRVWNzUTBGQlF5eERRVUZETEVOQlFVTXNSVUZCUlN4RFFVRkRMRU5CUVVNc1JVRkJSU3hEUVVGRExFZEJRVWNzUTBGQlF5eERRVUZCTEVOQlFVTXNRMEZCUXl4RFFVRkRPMGxCUXk5RExFbEJRVWtzVjBGQlZ5eEZRVUZGTzFGQlJXSXNTVUZCU1N4RFFVRkRMRWRCUVdNc1NVRkJTU3hEUVVGRExFMUJRVTBzUlVGRE1VSXNRMEZCUXl4SFFVRkhMRU5CUVVNc1EwRkJReXhUUVVGVExFTkJRVU1zUTBGQlF5eERRVUZETEVWQlFVVXNRMEZCUXl4SFFVRkhMRU5CUVVNc1EwRkJReXhQUVVGUExFTkJRVU1zUTBGQlF5eERRVUZETEVkQlFVY3NRMEZCUXl4RlFVTjRReXhKUVVGSkxFZEJRVWNzUTBGQlF5eERRVUZETEU5QlFVOHNRMEZCUXl4RFFVRkRMRU5CUVVNc1JVRkJSU3hMUVVGTExFZEJRVWNzUTBGQlF5eERRVUZETEZGQlFWRXNRMEZCUXl4RFFVRkRMRU5CUVVNc1JVRkRNVU1zUjBGQlJ5eEhRVUZITEVOQlFVTXNSMEZCUnl4RFFVRkRMRWRCUVVjc1QwRkJUeXhIUVVGSExFTkJRVU1zUlVGQlJTeEhRVUZITEVkQlFVY3NRMEZCUXl4SFFVRkhMRU5CUVVNc1IwRkJSeXhQUVVGUExFZEJRVWNzUTBGQlF5eERRVUZETzFGQlEzcEVMRWxCUVVrc1EwRkJReXhOUVVGTkxFTkJRVU1zWlVGQlpTeEhRVUZITEVkQlFVY3NRMEZCUXp0UlFVTnNReXhIUVVGSExFTkJRVU1zUTBGQlF5eERRVUZETEZGQlFWRXNRMEZCUXl4SlFVRkpMRVZCUVVVc1MwRkJTeXhGUVVGRkxFZEJRVWNzUlVGQlJTeFBRVUZQTEVOQlFVTXNSVUZCUlN4SlFVRkpMRU5CUVVNc1RVRkJUU3hEUVVGRExFTkJRVU03VVVGRGVFUXNTVUZCU1N4RFFVRkRMRTFCUVUwc1EwRkJReXhsUVVGbExFZEJRVWNzUjBGQlJ5eERRVUZETzFGQlEyeERMRWRCUVVjc1EwRkJReXhEUVVGRExFTkJRVU1zVVVGQlVTeERRVUZETEVsQlFVa3NSVUZCUlN4TFFVRkxMRVZCUVVVc1IwRkJSeXhGUVVGRkxFOUJRVThzUTBGQlF5eEZRVUZGTEVsQlFVa3NRMEZCUXl4TlFVRk5MRU5CUVVNc1EwRkJRenRMUVVNelJEdEpRVU5FTEVsQlFVa3NSVUZCUlR0UlFVRkZMRWxCUVVrc1EwRkJReXhOUVVGTkxFTkJRVU1zVDBGQlR5eERRVUZETEZWQlFVRXNRMEZCUXl4SlFVRkpMRTlCUVVFc1IwRkJSeXhEUVVGRExFTkJRVU1zUTBGQlF5eE5RVUZOTEVWQlFVVXNRMEZCUXl4RFFVRkRMRkZCUVZFc1EwRkJReXhGUVVGNlFpeERRVUY1UWl4RFFVRkRMRU5CUVVNN1NVRkROVVFzU1VGQlNTeEZRVUZGTzFGQlFVVXNTVUZCU1N4RFFVRkRMRTFCUVUwc1EwRkJReXhQUVVGUExFTkJRVU1zVlVGQlFTeERRVUZETzFsQlEzcENMRWxCUVVrc1EwRkJReXhIUVVGakxFTkJRVU1zUTBGQlF5eE5RVUZOTEVOQlFVTTdXVUZETlVJc1IwRkJSeXhEUVVGRExFTkJRVU1zUTBGQlF5eFJRVUZSTEVOQlFVTXNRMEZCUXl4RFFVRkRMRTlCUVU4c1EwRkJReXhEUVVGRExFTkJRVU1zUlVGQlJTeERRVUZETEVOQlFVTXNVVUZCVVN4RFFVRkRMRU5CUVVNc1EwRkJReXhGUVVGRkxFTkJRVU1zUTBGQlF5eFRRVUZUTEVOQlFVTXNRMEZCUXl4RFFVRkRMRVZCUVVVc1EwRkJReXhEUVVGRExFOUJRVThzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RlFVRkZMRU5CUVVNc1EwRkJReXhOUVVGTkxFTkJRVU1zUTBGQlF6dFJRVU42Uml4RFFVRkRMRU5CUVVNc1EwRkJRenRKUVVOSUxFbEJRVWtzUlVGQlJTeEhRVUZITEcxQ1FVRnRRaXhEUVVGRExFVkJRVVVzUlVGQlJTeEZRVUZGTEVWQlFVVXNRMEZCUXl4RlFVRkZMRTFCUVUwc1EwRkJReXhEUVVGRE8wbEJRMmhFTEVsQlFVa3NSVUZCUlN4RlFVRkZPMUZCUTBvc1JVRkJSU3hEUVVGRExFOUJRVThzUTBGQlF5eFZRVUZCTEVOQlFVTXNTVUZCVFN4RFFVRkRMRU5CUVVNc1NVRkJTU3hIUVVGSExFVkJRVVVzUlVGQlJTeERRVUZETEVOQlFVTXNSMEZCUnl4SFFVRkhMRVZCUVVVc1EwRkJRU3hEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETzFGQlF6ZERMRVZCUVVVc1EwRkJReXhQUVVGUExFTkJRVU1zVlVGQlFTeERRVUZETEVsQlFVMHNRMEZCUXl4RFFVRkRMRWxCUVVrc1EwRkJReXhKUVVGSkxFTkJRVU1zU1VGQlNTeERRVUZETEVOQlFVTXNRMEZCUXl4RlFVRkZMRU5CUVVNc1EwRkJReXhMUVVGTExFTkJRVU1zUjBGQlJ5eERRVUZETEVsQlFVa3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJRU3hEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETzFGQlF6bEVMRWxCUVVrc1EwRkJReXhOUVVGTkxFTkJRVU1zVDBGQlR5eERRVUZETEZWQlFVRXNRMEZCUXp0WlFVTnFRaXhKUVVGSkxHRkJRV0VzUjBGQlJ5eERRVUZETEVOQlFVTXNRMEZCUXl4UFFVRlBMRWRCUVVjc1EwRkJReXhEUVVGRExFOUJRVThzUTBGQlF5eERRVUZETEVOQlFVTXNUVUZCVFN4RFFVRkRMRU5CUVVNc1IwRkJSeXhEUVVGRExFTkJRVU03V1VGRE1VUXNRMEZCUXl4RFFVRkRMRTFCUVUwc1EwRkJReXhIUVVGSExFTkJRVU1zVDBGQlR5eERRVUZETEZWQlFVRXNRMEZCUXl4SlFVRkpMRTlCUVVFc1EwRkJReXhEUVVGRExFZEJRVWNzU1VGQlNTeGhRVUZoTEVWQlFYUkNMRU5CUVhOQ0xFTkJRVU1zUTBGQlF6dFpRVU5zUkN4RFFVRkRMRU5CUVVNc1RVRkJUU3hEUVVGRExFbEJRVWtzUTBGQlF5eFBRVUZQTEVOQlFVTXNWVUZCUVN4RFFVRkRMRWxCUVUwc1EwRkJReXhEUVVGRExFbEJRVWtzUjBGQlJ5eERRVUZETEVOQlFVTXNUVUZCVFN4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFZEJRVWNzU1VGQlNTeGhRVUZoTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJRenRSUVVNdlJTeERRVUZETEVOQlFVTXNRMEZCUXp0TFFVTk9PMGxCUTBRc1QwRkJUeXhuUWtGQlowSXNRMEZCUXl4TlFVRk5MRU5CUVVNc1JVRkJSU3hEUVVGRExFTkJRVU03UVVGRGRrTXNRMEZCUXp0QlFVVkVMRk5CUVZNc2JVSkJRVzFDTEVOQlFVTXNSVUZCWlN4RlFVRkZMRWxCUVdkQ0xFVkJRekZFTEVsQlFXMUNMRVZCUVVVc1RVRkJZenRKUVVWdVF5eEpRVUZKTEVOQlFVTXNSVUZCUlN4RFFVRkRMRWRCUVVjc1JVRkJSU3hEUVVGRExFMUJRVTBzUTBGQlF6dEpRVU55UWl4SlFVRkpMRU5CUVVNc1IwRkJSeXhEUVVGRExFZEJRVWNzUTBGQlF5eERRVUZETzBsQlEyUXNUMEZCVHl4RFFVRkRMRTFCUVUwc1EwRkJReXhKUVVGSkxFTkJRVU1zVFVGQlRTeEpRVUZKTEVOQlFVTXNRMEZCUXl4RFFVRkRPMGxCUTJwRExFbEJRVWtzVFVGQlRTeEhRVUZITEVsQlFVa3NTMEZCU3l4RFFVRlJMRU5CUVVNc1EwRkJReXhEUVVGRE8wbEJRMnBETEV0QlFVc3NRMEZCUXl4SFFVRkhMRU5CUVVNc1JVRkJSU3hEUVVGRExFZEJRVWNzUTBGQlF5eEZRVUZGTEVWQlFVVXNRMEZCUXl4RlFVRkZPMUZCUTNCQ0xFbEJRVWtzUTBGQlF5eEhRVUZITEVWQlFVVXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJRenRSUVVOa0xFbEJRVWtzUTBGQlF5eEhRVUZITEVsQlFVa3NTVUZCU1N4RFFVRkRMRWxCUVVrc1EwRkJReXhEUVVGRExFTkJRVU1zUlVGQlJTeERRVUZETEVWQlFVVXNTVUZCU1N4RFFVRkRMRk5CUVZNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETzFGQlEyaEVMRTFCUVUwc1EwRkJReXhEUVVGRExFTkJRVU1zUjBGQlJ5eEpRVUZKTEV0QlFVc3NRMEZCUXl4SlFVRkpMRVZCUVVVc1EwRkJReXhGUVVGRkxFbEJRVWtzUTBGQlF5eFBRVUZQTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJRenRSUVVOb1JDeE5RVUZOTEVOQlFVTXNRMEZCUXl4SFFVRkhMRU5CUVVNc1EwRkJReXhIUVVGSExFbEJRVWtzUzBGQlN5eERRVUZETEV0QlFVc3NSVUZCUlN4RFFVRkRMRVZCUVVVc1NVRkJTU3hEUVVGRExGRkJRVkVzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRPMHRCUTNwRU8wbEJRMFFzVFVGQlRTeERRVUZETEVsQlFVa3NRMEZCUXl4aFFVRmhMRU5CUVVNc1EwRkJRenRKUVVNelFpeEpRVUZKTEVWQlFVVXNSMEZCUnl4SlFVRkpMRXRCUVVzc1JVRkJZeXhEUVVGRE8wbEJRMnBETEVsQlFVa3NVVUZCVVN4SFFVRkhMRlZCUVZVc1JVRkJSU3hEUVVGRE8wbEJRelZDTEV0QlFVc3NRMEZCUXl4SFFVRkhMRU5CUVVNc1JVRkJSU3hEUVVGRExFZEJRVWNzUTBGQlF5eEZRVUZGTEVWQlFVVXNRMEZCUXl4RlFVRkZPMUZCUTNCQ0xFbEJRVWtzUTBGQlF5eEhRVUZITEUxQlFVMHNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJRenRSUVVOc1FpeEpRVUZKTEVOQlFVTXNSMEZCUnl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRE8xRkJRMW9zU1VGQlNTeERRVUZETEVOQlFVTXNUVUZCVFN4RlFVRkZPMWxCUTFZc1VVRkJVU3hEUVVGRExFMUJRVTBzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXp0WlFVTnVRaXhKUVVGSkxFTkJRVU1zWTBGQll5eERRVUZETEVOQlFVTXNSVUZCUlN4UlFVRlJMRU5CUVVNc1EwRkJRenRUUVVOd1F6dGhRVUZOTzFsQlJVZ3NVVUZCVVN4RFFVRkRMRTFCUVUwc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dFpRVU51UWl4SlFVRkpMR05CUVdNc1IwRkJSeXhWUVVGRExFTkJRVU1zUlVGQlJTeERRVUZETzJkQ1FVTjBRaXhKUVVGSkxFZEJRVWNzUjBGQlJ5eERRVUZETEVsQlFVa3NRMEZCUXl4UFFVRlBMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eEhRVUZITEVsQlFVa3NRMEZCUXl4UFFVRlBMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVkQlFVY3NRMEZCUXl4SFFVRkhMRTFCUVUwc1EwRkJRenRuUWtGREwwUXNSVUZCUlN4RFFVRkRMRWxCUVVrc1EwRkJReXhKUVVGSkxHbENRVUZWTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1JVRkJSU3hEUVVGRExFTkJRVU1zUTBGQlF5eEZRVUZGTEVkQlFVY3NRMEZCUXl4RFFVRkRMRU5CUVVNN1dVRkRNME1zUTBGQlF5eERRVUZETzFsQlEwWXNTVUZCU1N4bFFVRmxMRWRCUVVjc1ZVRkJReXhQUVVGUExFVkJRVVVzVDBGQlR5eEZRVUZGTEV0QlFVczdaMEpCUXpGRExFbEJRVWtzUTBGQlF5eEZRVUZGTEVWQlFVVXNSMEZCUnl4RFFVRkRMRU5CUVVNc1QwRkJUeXhEUVVGRExFTkJRVU1zVVVGQlVTeEZRVUZGTEVOQlFVTTdaMEpCUTJ4RExFOUJRVThzUTBGQlF5eERRVUZETEVkQlFVY3NSVUZCUlN4RFFVRkRMRTlCUVU4c1EwRkJReXhGUVVGRkxFTkJRVU1zUzBGQlN5eEpRVUZKTEVWQlFVVTdiMEpCUTJwRExFdEJRVXNzUTBGQlF5eERRVUZETEVWQlFVVXNRMEZCUXl4RFFVRkRMRU5CUVVNN2IwSkJRMW9zUTBGQlF5eERRVUZETEU5QlFVOHNRMEZCUXl4RFFVRkRMRTFCUVUwc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dHBRa0ZEZUVJN1dVRkRUQ3hEUVVGRExFTkJRVU03V1VGRFJpeGxRVUZsTEVOQlFVTXNUVUZCVFN4RlFVRkZMRTFCUVUwc1JVRkJSU3hWUVVGRExFTkJRVU1zUlVGQlJTeERRVUZETEVsQlFVc3NUMEZCUVN4alFVRmpMRU5CUVVNc1EwRkJReXhGUVVGRkxFTkJRVU1zUTBGQlF5eEZRVUZ3UWl4RFFVRnZRaXhEUVVGRExFTkJRVU03V1VGRGFFVXNaVUZCWlN4RFFVRkRMRTFCUVUwc1JVRkJSU3hOUVVGTkxFVkJRVVVzVlVGQlF5eERRVUZETEVWQlFVVXNRMEZCUXl4SlFVRkxMRTlCUVVFc1kwRkJZeXhEUVVGRExFTkJRVU1zUlVGQlJTeERRVUZETEVOQlFVTXNSVUZCY0VJc1EwRkJiMElzUTBGQlF5eERRVUZETzFOQlEyNUZPMHRCUTBvN1NVRkRSQ3hQUVVGUExFTkJRVU1zVFVGQlRTeERRVUZETEZGQlFWRXNRMEZCUXl4SlFVRkpMRXRCUVVzc1EwRkJReXhEUVVGRExFTkJRVU03U1VGRGNFTXNUMEZCVHl4RlFVRkZMRU5CUVVNN1FVRkRaQ3hEUVVGRE8wRkJSVVFzVTBGQlV5eGxRVUZsTEVOQlFVTXNRMEZCVHl4RlFVRkZMRkZCUVhOQ08wbEJRM0JFTEVsQlFVa3NRMEZCUXl4SFFVRkhMRlZCUVVNc1QwRkJUeXhGUVVGRkxFOUJRVTg3VVVGRGNrSXNTVUZCU1N4RlFVRkZMRWRCUVVjc1VVRkJVU3hEUVVGRExGRkJRVkVzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXp0UlFVTTVRaXhKUVVGSkxFTkJRVU1zUTBGQlF6dFJRVU5PTEU5QlFVOHNRMEZCUXl4RFFVRkRMRWRCUVVjc1JVRkJSU3hEUVVGRExFOUJRVThzUTBGQlF5eEZRVUZGTEVOQlFVTXNTMEZCU3l4SlFVRkpMRVZCUVVVN1dVRkRha01zU1VGQlNTeFBRVUZQTEVkQlFVY3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhSUVVGUkxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRPMWxCUTJoRExFbEJRVWtzVDBGQlR5eEpRVUZKTEVOQlFVTXNTVUZCU1N4UFFVRlBMRWxCUVVrc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eFJRVUZSTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhGUVVGRk8yZENRVU01UXl4RFFVRkRMRU5CUVVNc1QwRkJUeXhEUVVGRExFTkJRVU1zVFVGQlRTeERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRPMmRDUVVOeVFpeERRVUZETEVOQlFVTXNUMEZCVHl4RFFVRkRMRU5CUVVNc1RVRkJUU3hEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETzJGQlEzaENPMWxCUTBRc1NVRkJTU3hQUVVGUExFbEJRVWtzUTBGQlF5eEZRVUZGTzJkQ1FVTmtMRTFCUVUwN1lVRkRWRHRUUVVOS08wbEJRMHdzUTBGQlF5eERRVUZCTzBsQlEwUXNRMEZCUXl4RFFVRkRMRTFCUVUwc1JVRkJSU3hOUVVGTkxFTkJRVU1zUTBGQlF6dEpRVU5zUWl4RFFVRkRMRU5CUVVNc1RVRkJUU3hGUVVGRkxFMUJRVTBzUTBGQlF5eERRVUZETzBGQlEzUkNMRU5CUVVNN1FVRkZSQ3hUUVVGVExHVkJRV1VzUTBGQlF5eERRVUZQTEVWQlFVVXNVVUZCYzBJN1NVRkRjRVFzU1VGQlNTeERRVUZETEVkQlFVY3NWVUZCUXl4UFFVRlBMRVZCUVVVc1QwRkJUenRSUVVOeVFpeEpRVUZKTEVOQlFVTXNSMEZCUnl4UlFVRlJMRU5CUVVNc1VVRkJVU3hEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEU5QlFVOHNRMEZCUXl4RlFVRkZMRU5CUVVNN1VVRkRlRU1zU1VGQlNTeERRVUZETEV0QlFVc3NTVUZCU1N4SlFVRkpMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zVVVGQlVTeERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1IwRkJSeXhEUVVGRExFVkJRVVU3V1VGRGNrTXNRMEZCUXl4RFFVRkRMRTlCUVU4c1EwRkJReXhEUVVGRExFMUJRVTBzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXp0WlFVTnlRaXhEUVVGRExFTkJRVU1zVDBGQlR5eERRVUZETEVOQlFVTXNUVUZCVFN4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRE8xTkJRM2hDTzBsQlEwd3NRMEZCUXl4RFFVRkJPMGxCUTBRc1EwRkJReXhEUVVGRExFMUJRVTBzUlVGQlJTeE5RVUZOTEVOQlFVTXNRMEZCUXp0SlFVTnNRaXhEUVVGRExFTkJRVU1zVFVGQlRTeEZRVUZGTEUxQlFVMHNRMEZCUXl4RFFVRkRPMEZCUTNSQ0xFTkJRVU03UVVGRlJDeFRRVUZuUWl4dlFrRkJiMElzUTBGQlF5eEZRVUZsTEVWQlFVVXNTVUZCWjBJN1NVRkRiRVVzVDBGQlR5eHRRa0ZCYlVJc1EwRkJReXhGUVVGRkxFVkJRVVVzU1VGQlNTeEZRVUZGTEV0QlFVc3NSVUZCUlN4SlFVRkpMRU5CUVVNc1EwRkJRenRCUVVOMFJDeERRVUZETzBGQlJrUXNiMFJCUlVNN1FVRkZSQ3hUUVVGblFpeHZRa0ZCYjBJc1EwRkJReXhGUVVGbExFVkJRVVVzU1VGQlowSTdTVUZEYkVVc1QwRkJUeXh0UWtGQmJVSXNRMEZCUXl4RlFVRkZMRVZCUVVVc1NVRkJTU3hGUVVGRkxFdEJRVXNzUlVGQlJTeEpRVUZKTEVOQlFVTXNRMEZCUXp0QlFVTjBSQ3hEUVVGRE8wRkJSa1FzYjBSQlJVTTdRVUZGUkN4VFFVRm5RaXg1UWtGQmVVSXNRMEZCUXl4SlFVRnhRanRKUVVNelJDeFBRVUZQTEhkQ1FVRjNRaXhEUVVGRExFbEJRVWtzUlVGQlJTeExRVUZMTEVWQlFVVXNTVUZCU1N4RFFVRkRMRU5CUVVNN1FVRkRka1FzUTBGQlF6dEJRVVpFTERoRVFVVkRPMEZCUlVRc1UwRkJaMElzZVVKQlFYbENMRU5CUVVNc1NVRkJjVUk3U1VGRE0wUXNUMEZCVHl4M1FrRkJkMElzUTBGQlF5eEpRVUZKTEVWQlFVVXNTMEZCU3l4RlFVRkZMRWxCUVVrc1EwRkJReXhEUVVGRE8wRkJRM1pFTEVOQlFVTTdRVUZHUkN3NFJFRkZRenRCUVVWRUxGTkJRV2RDTEdOQlFXTXNRMEZCUXl4RlFVRmxPMGxCUXpGRExFbEJRVWtzUlVGQlJTeEhRVUZITEVWQlFVVXNRMEZCUXl4SFFVRkhMRU5CUVVNc1ZVRkJRU3hEUVVGRExFbEJRVWtzVDBGQlFTeEpRVUZKTEdWQlFWRXNRMEZCUXl4RFFVRkRMRU5CUVVNc1JVRkJSU3hGUVVGRkxFTkJRVU1zUlVGQmNFSXNRMEZCYjBJc1EwRkJReXhEUVVGRE8wbEJRek5ETEVsQlFVa3NSVUZCUlN4SFFVRkhMRzlDUVVGdlFpeERRVUZETEVWQlFVVXNSVUZCUlN4RlFVRkZMRU5CUVVNc1EwRkJRenRKUVVOMFF5eEpRVUZKTEUxQlFVMHNSMEZCUnl4SlFVRkpMR0ZCUVUwc1EwRkJReXhGUVVGRkxFVkJRVVVzUlVGQlJTeERRVUZETEVOQlFVTTdTVUZEYUVNc1RVRkJUU3hEUVVGRExFdEJRVXNzUlVGQlJTeERRVUZETzBsQlEyWXNSVUZCUlN4RFFVRkRMRTlCUVU4c1EwRkJReXhWUVVGRExFTkJRVU1zUlVGQlJTeERRVUZETEVsQlFVc3NUMEZCUVN4RlFVRkZMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zVlVGQlZTeERRVUZETEVOQlFVTXNRMEZCUXl4UlFVRlJMRVZCUVVVc1EwRkJReXhGUVVFNVFpeERRVUU0UWl4RFFVRkRMRU5CUVVNN1NVRkRja1FzUlVGQlJTeEhRVUZITEVWQlFVVXNRMEZCUXl4SFFVRkhMRU5CUVVNc1ZVRkJRU3hEUVVGRExFbEJRVWNzVDBGQlFTeEpRVUZKTEdWQlFWRXNRMEZCUXl4RFFVRkRMRU5CUVVNc1JVRkJSU3hGUVVGRkxFTkJRVU1zUlVGQmNFSXNRMEZCYjBJc1EwRkJReXhEUVVGRE8wbEJRM1JETEVWQlFVVXNSMEZCUnl4dlFrRkJiMElzUTBGQlF5eEZRVUZGTEVWQlFVVXNSVUZCUlN4RFFVRkRMRU5CUVVNN1NVRkRiRU1zVFVGQlRTeEhRVUZITEVsQlFVa3NZVUZCVFN4RFFVRkRMRVZCUVVVc1JVRkJSU3hGUVVGRkxFTkJRVU1zUTBGQlF6dEpRVU0xUWl4TlFVRk5MRU5CUVVNc1MwRkJTeXhGUVVGRkxFTkJRVU03U1VGRFppeEZRVUZGTEVOQlFVTXNUMEZCVHl4RFFVRkRMRlZCUVVNc1EwRkJReXhGUVVGRkxFTkJRVU1zU1VGQlN5eFBRVUZCTEVWQlFVVXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhWUVVGVkxFTkJRVU1zUTBGQlF5eERRVUZETEZGQlFWRXNSVUZCUlN4RFFVRkRMRVZCUVRsQ0xFTkJRVGhDTEVOQlFVTXNRMEZCUXp0QlFVTjZSQ3hEUVVGRE8wRkJXRVFzZDBOQlYwTTdRVUZoUkR0SlFVRnhReXh0UTBGQlVUdEpRVU42UXl4NVFrRkJiVUlzUzBGQllTeEZRVUZGTEVOQlFWTTdVVUZCTTBNc1dVRkRTU3hyUWtGQlRTeERRVUZETEVWQlFVVXNRMEZCUXl4RFFVRkRMRk5CUTJRN1VVRkdhMElzVjBGQlN5eEhRVUZNTEV0QlFVc3NRMEZCVVRzN1NVRkZhRU1zUTBGQlF6dEpRVU5NTEhOQ1FVRkRPMEZCUVVRc1EwRkJReXhCUVVwRUxFTkJRWEZETEdWQlFWRXNSMEZKTlVNN1FVRktXU3d3UTBGQlpUdEJRVTAxUWp0SlFVdEpMRzlDUVVGdlFpeExRVUZyUWl4RlFVTXhRaXhOUVVGNVFpeEZRVU42UWl4VFFVRnBReXhGUVVONlF5eFhRVUYzUWl4RlFVTm9RaXhoUVVFNFFqdFJRVW94UXl4cFFrRTRRa003VVVFMVFsY3NNRUpCUVVFc1JVRkJRU3huUWtGQmFVTTdVVUZEZWtNc05FSkJRVUVzUlVGQlFTeHJRa0ZCZDBJN1VVRkRhRUlzT0VKQlFVRXNSVUZCUVN4eFFrRkJPRUk3VVVGS2RFSXNWVUZCU3l4SFFVRk1MRXRCUVVzc1EwRkJZVHRSUVVNeFFpeFhRVUZOTEVkQlFVNHNUVUZCVFN4RFFVRnRRanRSUVVONlFpeGpRVUZUTEVkQlFWUXNVMEZCVXl4RFFVRjNRanRSUVVWcVF5eHJRa0ZCWVN4SFFVRmlMR0ZCUVdFc1EwRkJhVUk3VVVGRmRFTXNTVUZCU1N4RFFVRkRMRk5CUVZNc1IwRkJSeXhMUVVGTExFTkJRVU1zUjBGQlJ5eERRVUZETEZWQlFVTXNRMEZCUXl4RlFVRkZMRU5CUVVNN1dVRkROVUlzVDBGQlR5eERRVUZETEVOQlFVTXNVVUZCVVN4SFFVRkhMRWxCUVVrc1pVRkJaU3hEUVVGRExFTkJRVU1zUlVGQlJTeERRVUZETEVOQlFVTXNRMEZCUXp0UlFVTnNSQ3hEUVVGRExFTkJRVU1zUTBGQlF6dFJRVVZJTEVsQlFVa3NWMEZCVnp0WlFVRkZMRWxCUVVrc1EwRkJReXhwUWtGQmFVSXNRMEZCUXl4WFFVRlhMRU5CUVVNc1EwRkJRenRSUVVWeVJDeEpRVUZKTEdGQlFXRXNTVUZCU1N4VFFVRlRMRWxCUVVrc1QwRkJUeXhUUVVGVExFTkJRVU1zVFVGQlRTeExRVUZMTEZkQlFWY3NSVUZCUlR0WlFVTjJSU3hMUVVGTExFTkJRVU1zVDBGQlR5eERRVUZETEZWQlFVRXNRMEZCUXp0blFrRkRNVUlzU1VGQlNTeERRVUZETEVOQlFVTXNRMEZCUXl4TFFVRkxMRWxCUVVrc1EwRkJReXhEUVVGRExFTkJRVU1zVFVGQlRTeEZRVU42UWp0dlFrRkZReXhEUVVGRExFTkJRVU1zVFVGQlRTeEhRVUZITEVsQlFVa3NVMEZCVXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFVkJRVVVzUTBGQlF5eERRVUZETEVOQlFVTXNSVUZCUlN4RFFVRkRMRU5CUVVNc1EwRkJReXhGUVVGRkxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXp0dlFrRkROME1zVDBGQlR6dHBRa0ZEVUR0blFrRkRZeXhKUVVGSkxFVkJRVVVzUjBGQlJ5eERRVUZETEVOQlFVTXNTMEZCU3l4SFFVRkhMRU5CUVVNc1JVRkJSU3hGUVVGRkxFZEJRVWNzUTBGQlF5eERRVUZETEUxQlFVMHNSMEZCUnl4RFFVRkRMRU5CUVVNN1owSkJRM2hETEVOQlFVTXNRMEZCUXl4TlFVRk5MRWRCUVVjc1NVRkJTU3hUUVVGVExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTXNSMEZCUnl4RlFVRkZMRVZCUVVVc1EwRkJReXhEUVVGRExFTkJRVU1zUjBGQlJ5eEZRVUZGTEVWQlFVVXNRMEZCUXl4RFFVRkRMRU5CUVVNc1IwRkJSeXhGUVVGRkxFVkJRVVVzUTBGQlF5eERRVUZETEVOQlFVTXNSMEZCUnl4RlFVRkZMRU5CUVVNc1EwRkJRenRaUVVOeVJTeERRVUZETEVOQlFVTXNRMEZCUXp0WlFVTklMR3RDUVVGclFpeERRVUZETEZOQlFWTXNRMEZCUXl4RFFVRkRPMWxCUXpsQ0xFbEJRVWtzUTBGQlF5eEhRVUZITEV0QlFVc3NRMEZCUXl4TlFVRk5MRU5CUVVNN1dVRkRja0lzVFVGQlRTeERRVUZETEU5QlFVOHNRMEZCUXl4VlFVRkJMRU5CUVVNN1owSkJRMW9zUzBGQlNTeERRVUZETEZOQlFWTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1IwRkJSeXhEUVVGRExFTkJRVU1zVFVGQlRTeEhRVUZITEVsQlFVa3NaVUZCWlN4RFFVRkRMRU5CUVVNc1JVRkJSU3hGUVVGRkxFOUJRVThzUTBGQlF5eERRVUZETEZOQlFWTXNTMEZCU3l4WFFVRlhMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eFRRVUZUTEVOQlFVTXNRMEZCUXl4RFFVRkRMRWxCUVVrc1EwRkJReXhEUVVGRE8yZENRVU5xU0N4TFFVRkpMRU5CUVVNc1UwRkJVeXhEUVVGRExFTkJRVU1zUTBGQlF5eEhRVUZITEVOQlFVTXNRMEZCUXl4TlFVRk5MRWRCUVVjc1NVRkJTU3hsUVVGbExFTkJRVU1zUTBGQlF5eEZRVUZGTEVWQlFVVXNUMEZCVHl4RFFVRkRMRU5CUVVNc1UwRkJVeXhMUVVGTExGZEJRVmNzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRk5CUVZNc1EwRkJReXhEUVVGRExFTkJRVU1zU1VGQlNTeERRVUZETEVOQlFVTTdXVUZEY2tnc1EwRkJReXhEUVVGRExFTkJRVU03VTBGRFRqdEpRVU5NTEVOQlFVTTdTVUZIVHl4eFEwRkJaMElzUjBGQmVFSXNWVUZCZVVJc1EwRkJUVHRSUVVNelFpeFBRVUZQTEVsQlFVa3NhVUpCUVZVc1EwRkRha0lzU1VGQlNTeERRVUZETEV0QlFVc3NRMEZCUXl4RFFVRkRMRU5CUVVNc1NVRkJTU3hEUVVGRExFTkJRVU1zVVVGQlVTeEZRVU16UWl4SlFVRkpMRU5CUVVNc1MwRkJTeXhEUVVGRExFTkJRVU1zUTBGQlF5eExRVUZMTEVOQlFVTXNRMEZCUXl4UlFVRlJMRVZCUXpWQ0xFTkJRVU1zUTBGQlF5eEhRVUZITEVWQlEwd3NUMEZCVHl4RFFVRkRMRU5CUVVNc1VVRkJVU3hMUVVGTExGZEJRVmNzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRkZCUVZFc1EwRkJReXhEUVVGRExFTkJRVU1zUzBGQlN5eERRVUZETEVOQlFVTTdTVUZEYUVVc1EwRkJRenRKUVVkUExHbERRVUZaTEVkQlFYQkNMRlZCUVhGQ0xFTkJRVTA3VVVGQk0wSXNhVUpCYVVKRE8xRkJhRUpITEVsQlFVa3NRMEZCUXl4SlFVRkpMRU5CUVVNc1lVRkJZVHRaUVVGRkxFOUJRVTg3VVVGRmFFTXNTVUZCU1N4SlFVRkpMRWRCUVVjc1IwRkJSeXhGUVVGRkxFZEJRVWNzUjBGQlJ5eFBRVUZQTEVOQlFVTTdVVUZET1VJc1NVRkJTU3hEUVVGRExFTkJRVU1zU1VGQlNTeExRVUZMTEVkQlFVYzdXVUZCUlN4SlFVRkpMRWRCUVVjc1IwRkJSeXhGUVVGRkxFZEJRVWNzUjBGQlJ5eFJRVUZSTEVOQlFVTTdVVUZETDBNc1NVRkJTU3hGUVVGRkxFZEJRV2RDTEVOQlFVTXNRMEZCUXl4UFFVRlBMRU5CUVVNc1IwRkJSeXhEUVVGRExGVkJRVUVzUTBGQlF5eEpRVUZKTEU5QlFVRXNTMEZCU1N4RFFVRkRMRXRCUVVzc1EwRkJReXhEUVVGRExFTkJRVU1zU1VGQlNTeERRVUZETEVWQlFXeENMRU5CUVd0Q0xFTkJRVU1zUTBGQlF5eEpRVUZKTEVOQlFVTXNWVUZCUXl4RFFVRkRMRVZCUVVVc1EwRkJReXhKUVVGTExFOUJRVUVzUTBGQlF5eERRVUZETEVsQlFVa3NRMEZCUXl4SFFVRkhMRU5CUVVNc1EwRkJReXhKUVVGSkxFTkJRVU1zUlVGQmFrSXNRMEZCYVVJc1EwRkJReXhEUVVGRE8xRkJReTlHTEVsQlFVa3NRMEZCUXl4SFFVRmpMRWxCUVVrc1EwRkJRenRSUVVONFFpeEZRVUZGTEVOQlFVTXNUMEZCVHl4RFFVRkRMRlZCUVVFc1EwRkJRenRaUVVWU0xFbEJRVWtzUTBGQlF5eEZRVUZGTzJkQ1FVTklMRWxCUVVrc1QwRkJUeXhIUVVGSExFTkJRVU1zUTBGQlF5eEpRVUZKTEVOQlFVTXNSMEZCUnl4RFFVRkRMRU5CUVVNc1IwRkJSeXhEUVVGRExFTkJRVU03WjBKQlF5OUNMRWxCUVVrc1QwRkJUeXhIUVVGSExFTkJRVU1zUTBGQlF5eEpRVUZKTEVOQlFVTXNSVUZCUlR0dlFrRkRia0lzUTBGQlF5eERRVUZETEVsQlFVa3NRMEZCUXl4SFFVRkhMRTlCUVU4c1EwRkJRenRwUWtGRGNrSTdZVUZEU2p0WlFVTkVMRU5CUVVNc1IwRkJSeXhEUVVGRExFTkJRVU03VVVGRFZpeERRVUZETEVOQlFVTXNRMEZCUXp0SlFVTlFMRU5CUVVNN1NVRkZUeXh2UTBGQlpTeEhRVUYyUWl4VlFVRjNRaXhEUVVGTk8xRkJRVGxDTEdsQ1FWRkRPMUZCVUVjc1NVRkJTU3hEUVVGRExFZEJRVWNzU1VGQlNTeERRVUZETEV0QlFVc3NRMEZCUXl4RFFVRkRMRU5CUVVNc1QwRkJUeXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVsQlFVa3NRMEZCUXl4RFFVRkRMRkZCUVZFc1EwRkJRenRSUVVNdlF5eEpRVUZKTEVOQlFVTXNXVUZCV1N4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRE8xRkJRM0pDTEVsQlFVa3NSVUZCUlN4SFFVRkhMRU5CUVVNc1EwRkJReXhKUVVGSkxFdEJRVXNzUjBGQlJ5eERRVUZETEVOQlFVTXNRMEZCUXl4SlFVRkpMRU5CUVVNc1dVRkJXU3hEUVVGRExFTkJRVU1zUTBGQlF5eEpRVUZKTEVOQlFVTXNXVUZCV1N4RFFVRkRPMUZCUTJoRkxFTkJRVU1zUTBGQlF5eFBRVUZQTEVOQlFVTXNTMEZCU3l4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFOUJRVThzUTBGQlF5eFZRVUZCTEVOQlFVTTdXVUZEZUVJc1NVRkJTU3hEUVVGRExFZEJRVWNzUzBGQlNTeERRVUZETEV0QlFVc3NRMEZCUXl4RFFVRkRMRU5CUVVNc1NVRkJTU3hEUVVGRExFTkJRVU1zVVVGQlVTeERRVUZETzFsQlEzQkRMRVZCUVVVc1EwRkJReXhKUVVGSkxFTkJRVU1zU1VGQlNTeHBRa0ZCVlN4RFFVRkRMRU5CUVVNc1JVRkJSU3hEUVVGRExFVkJRVVVzUTBGQlF5eERRVUZETEUxQlFVMHNSVUZCUlN4SlFVRkpMRU5CUVVNc1EwRkJReXhEUVVGRE8xRkJRMnhFTEVOQlFVTXNRMEZCUXl4RFFVRkRPMGxCUTFBc1EwRkJRenRKUVVWUExITkRRVUZwUWl4SFFVRjZRaXhWUVVFd1FpeFhRVUZyUWp0UlFVRTFReXhwUWtGWFF6dFJRVlpITEVsQlFVa3NTMEZCU3l4SFFVRkhMRlZCUVVFc1EwRkJReXhKUVVGSkxFOUJRVUVzVDBGQlR5eERRVUZETEVOQlFVTXNTVUZCU1N4TFFVRkxMRmRCUVZjc1NVRkJTU3hEUVVGRExFTkJRVU1zU1VGQlNTeExRVUZMTEZsQlFWa3NSVUZCZUVRc1EwRkJkMFFzUTBGQlF6dFJRVU14UlN4SlFVRkpMRU5CUVVNc1dVRkJXU3hIUVVGSExGZEJRVmM3WVVGRE1VSXNUVUZCVFN4RFFVRkRMRlZCUVVFc1EwRkJReXhKUVVGSkxFOUJRVUVzUTBGQlF5eERRVUZETEVsQlFVa3NTMEZCU3l4SFFVRkhMRWxCUVVrc1MwRkJTeXhEUVVGRExFTkJRVU1zUTBGQlF5eEZRVUV4UWl4RFFVRXdRaXhEUVVGRE8yRkJRM1pETEVkQlFVY3NRMEZCUXl4VlFVRkJMRU5CUVVNc1NVRkJTU3hQUVVGQkxFdEJRVWtzUTBGQlF5eG5Ra0ZCWjBJc1EwRkJReXhEUVVGRExFTkJRVU1zUlVGQmVFSXNRMEZCZDBJc1EwRkJReXhEUVVGRE8xRkJRM2hETEVsQlFVa3NRMEZCUXl4WlFVRlpMRWRCUVVjc1YwRkJWenRoUVVNeFFpeE5RVUZOTEVOQlFVTXNWVUZCUVN4RFFVRkRMRWxCUVVrc1QwRkJRU3hEUVVGRExFTkJRVU1zU1VGQlNTeExRVUZMTEVkQlFVY3NTVUZCU1N4TFFVRkxMRU5CUVVNc1EwRkJReXhEUVVGRExFVkJRVEZDTEVOQlFUQkNMRU5CUVVNN1lVRkRka01zUjBGQlJ5eERRVUZETEZWQlFVRXNRMEZCUXl4SlFVRkpMRTlCUVVFc1MwRkJTU3hEUVVGRExHZENRVUZuUWl4RFFVRkRMRU5CUVVNc1EwRkJReXhGUVVGNFFpeERRVUYzUWl4RFFVRkRMRU5CUVVNN1VVRkRlRU1zVjBGQlZ6dGhRVU5PTEUxQlFVMHNRMEZCUXl4VlFVRkJMRU5CUVVNc1NVRkJTU3hQUVVGQkxFTkJRVU1zUTBGQlF5eEpRVUZKTEV0QlFVc3NWMEZCVnl4RlFVRjBRaXhEUVVGelFpeERRVUZETzJGQlEyNURMRTlCUVU4c1EwRkJReXhWUVVGQkxFTkJRVU1zU1VGQlNTeFBRVUZCTEV0QlFVa3NRMEZCUXl4bFFVRmxMRU5CUVVNc1EwRkJReXhEUVVGRExFVkJRWFpDTEVOQlFYVkNMRU5CUVVNc1EwRkJRenRKUVVNdlF5eERRVUZETzBsQlJVOHNORU5CUVhWQ0xFZEJRUzlDTEZWQlFXZERMRVZCUVZrc1JVRkJSU3hGUVVGWkxFVkJRVVVzVDBGQmFVSXNSVUZCUlN4VlFVRnZRenRSUVVNdlJ5eEpRVUZKTEVOQlFVTXNTMEZCU3l4RFFVRkRMRTlCUVU4c1EwRkJReXhWUVVGRExFTkJRVU1zUlVGQlJTeERRVUZETzFsQlEzQkNMRWxCUVVrc1EwRkJReXhEUVVGRExFdEJRVXNzUlVGQlJUdG5Ra0ZEVkN4RFFVRkRMRU5CUVVNc1VVRkJVU3hEUVVGRExFMUJRVTBzUjBGQlJ5eERRVUZETEVOQlFVTXNWMEZCVnl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zVjBGQlZ5eERRVUZETEVOQlFVTXNRMEZCUXl4SlFVRkpMRU5CUVVNN1owSkJRM3BFTEU5QlFVOHNRMEZCUXl4RFFVRkRMRU5CUVVNc1IwRkJSeXhWUVVGVkxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTTdZVUZET1VJN2FVSkJRVTA3WjBKQlEwZ3NRMEZCUXl4RFFVRkRMRkZCUVZFc1EwRkJReXhOUVVGTkxFZEJRVWNzUTBGQlF5eERRVUZETzJGQlEzcENPMWxCUTBRc1NVRkJTU3hEUVVGRExFZEJRVWNzUTBGQlF5eERRVUZETEVOQlFVTXNTMEZCU3l4SlFVRkpMRU5CUVVNc1EwRkJReXhIUVVGSExFTkJRVU1zUlVGQlJTeERRVUZETEVkQlFVY3NRMEZCUXl4RFFVRkRMRU5CUVVNc1RVRkJUU3hKUVVGSkxFTkJRVU1zUTBGQlF5eEhRVUZITEVOQlFVTXNRMEZCUXp0WlFVTndSQ3hKUVVGSkxFVkJRVVVzUjBGQlJ5eEZRVUZGTEVOQlFVTXNRMEZCUXl4RFFVRkRMRVZCUVVVc1JVRkJSU3hIUVVGSExFVkJRVVVzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXp0WlFVTXpRaXhEUVVGRExFTkJRVU1zVFVGQlRTeEhRVUZITEVsQlFVa3NVMEZCVXl4RFFVRkRMRVZCUVVVc1IwRkJSeXhEUVVGRExFVkJRVVVzUlVGQlJTeEhRVUZITEVOQlFVTXNSVUZCUlN4RlFVRkZMRWRCUVVjc1EwRkJReXhGUVVGRkxFVkJRVVVzUjBGQlJ5eERRVUZETEVOQlFVTXNRMEZCUXp0UlFVTTNSQ3hEUVVGRExFTkJRVU1zUTBGQlF6dEpRVU5RTEVOQlFVTTdTVUZGUkN3MlFrRkJVU3hIUVVGU0xGVkJRVk1zUlVGQldTeEZRVUZGTEVWQlFWa3NSVUZCUlN4RFFVRlhPMUZCUXpWRExFbEJRVWtzUTBGQlF5eEpRVUZKTEVOQlFVTXNVMEZCVXl4SlFVRkpMRU5CUVVNc1EwRkJReXhKUVVGSkxFTkJRVU1zWVVGQllTeEpRVUZKTEVsQlFVa3NRMEZCUXl4WlFVRlpMRU5CUVVNN1dVRkJSU3hQUVVGUE8xRkJRekZGTEVsQlFVa3NRMEZCUXl4UFFVRlBMRU5CUVVNc1JVRkJSU3hGUVVGRkxFVkJRVVVzUlVGQlJTeEZRVUZGTEVWQlFVVXNRMEZCUXl4RlFVRkZMRlZCUVVFc1EwRkJReXhKUVVGSExFOUJRVUVzUTBGQlF5eERRVUZETEVWQlFVVXNSVUZCU2l4RFFVRkpMRVZCUVVVc1NVRkJTU3hEUVVGRExGbEJRVmtzUlVGQlJTeDVRa0ZCZVVJc1JVRkRPVVVzVlVGQlFTeERRVUZETEVsQlFVa3NUMEZCUVN4RFFVRkRMRU5CUVVNc1RVRkJUU3hEUVVGRExGVkJRVlVzUTBGQlF5eERRVUZETEVOQlFXMUNMRU5CUVVNc1EwRkJReXhSUVVGVExFTkJRVU1zUzBGQlN5eERRVUZETEVkQlFVY3NRMEZCUXl4RFFVRkRMRkZCUVZFc1EwRkJReXhSUVVGUkxFVkJRVVVzUTBGQlF5eEZRVUZ1Uml4RFFVRnRSaXhGUVVONFJpeFZRVUZCTEVOQlFVTTdXVUZEUnl4SlFVRkpMRWxCUVVrc1IwRkJSeXhEUVVGRExFTkJRVzFDTEVOQlFVTXNRMEZCUXl4TlFVRlBMRU5CUVVNc1MwRkJTeXhEUVVGRExFZEJRVWNzUTBGQlF5eERRVUZETEUxQlFVMHNRMEZCUXl4UlFVRlJMRVZCUVVVc1EwRkJRenRaUVVOMFJTeEpRVUZKTEVsQlFVa3NSMEZCUnl4RFFVRkRMRU5CUVcxQ0xFTkJRVU1zUTBGQlF5eE5RVUZQTEVOQlFVTXNTMEZCU3l4RFFVRkRMRWRCUVVjc1EwRkJReXhEUVVGRExFMUJRVTBzUTBGQlF5eFJRVUZSTEVWQlFVVXNRMEZCUXp0WlFVTjBSU3hKUVVGSkxFVkJRVVVzUjBGQlJ5eERRVUZETEVOQlFVTXNUMEZCVHl4SFFVRkhMRU5CUVVNc1EwRkJRenRaUVVOMlFpeERRVUZETEVOQlFVTXNUVUZCVFN4RFFVRkRMRU5CUVVNc1IwRkJSeXhKUVVGSkxFZEJRVWNzUlVGQlJTeERRVUZETzFsQlEzWkNMRU5CUVVNc1EwRkJReXhOUVVGTkxFTkJRVU1zUTBGQlF5eEhRVUZITEVsQlFVa3NSMEZCUnl4RlFVRkZMRU5CUVVNN1VVRkRNMElzUTBGQlF5eERRVUZETEVOQlFVTTdTVUZEV0N4RFFVRkRPMGxCUlVRc05rSkJRVkVzUjBGQlVpeFZRVUZUTEVWQlFWa3NSVUZCUlN4RlFVRlpMRVZCUVVVc1EwRkJWenRSUVVNMVF5eEpRVUZKTEVOQlFVTXNTVUZCU1N4RFFVRkRMRk5CUVZNc1NVRkJTU3hEUVVGRExFbEJRVWtzUTBGQlF5eFpRVUZaTzFsQlFVVXNUMEZCVHp0UlFVTnNSQ3hKUVVGSkxFTkJRVU1zVDBGQlR5eERRVUZETEVWQlFVVXNSVUZCUlN4RlFVRkZMRVZCUVVVc1JVRkJSU3hGUVVGRkxFTkJRVU1zUlVGQlJTeFZRVUZCTEVOQlFVTXNTVUZCUnl4UFFVRkJMRU5CUVVNc1EwRkJReXhGUVVGRkxFVkJRVW9zUTBGQlNTeEZRVUZGTEVsQlFVa3NRMEZCUXl4WlFVRlpMRVZCUVVVc2VVSkJRWGxDTEVWQlF6bEZMRlZCUVVFc1EwRkJReXhKUVVGSkxFOUJRVUVzUTBGQlF5eERRVUZETEUxQlFVMHNRMEZCUXl4VlFVRlZMRU5CUVVNc1EwRkJReXhEUVVGdFFpeERRVUZETEVOQlFVTXNVVUZCVXl4RFFVRkRMRXRCUVVzc1EwRkJReXhIUVVGSExFTkJRVU1zUTBGQlF5eFJRVUZSTEVOQlFVTXNVVUZCVVN4RlFVRkZMRU5CUVVNc1JVRkJia1lzUTBGQmJVWXNSVUZEZUVZc1ZVRkJRU3hEUVVGRE8xbEJRMGNzU1VGQlNTeEpRVUZKTEVkQlFVY3NRMEZCUXl4RFFVRnRRaXhEUVVGRExFTkJRVU1zVFVGQlR5eERRVUZETEV0QlFVc3NRMEZCUXl4SFFVRkhMRU5CUVVNc1EwRkJReXhOUVVGTkxFTkJRVU1zVVVGQlVTeEZRVUZGTEVOQlFVTTdXVUZEZEVVc1NVRkJTU3hKUVVGSkxFZEJRVWNzUTBGQlF5eERRVUZ0UWl4RFFVRkRMRU5CUVVNc1RVRkJUeXhEUVVGRExFdEJRVXNzUTBGQlF5eEhRVUZITEVOQlFVTXNRMEZCUXl4TlFVRk5MRU5CUVVNc1VVRkJVU3hGUVVGRkxFTkJRVU03V1VGRGRFVXNTVUZCU1N4RlFVRkZMRWRCUVVjc1EwRkJReXhEUVVGRExFOUJRVThzUjBGQlJ5eERRVUZETEVOQlFVTTdXVUZEZGtJc1EwRkJReXhEUVVGRExFMUJRVTBzUTBGQlF5eERRVUZETEVkQlFVY3NTVUZCU1N4SFFVRkhMRVZCUVVVc1EwRkJRenRaUVVGQkxFTkJRVU03V1VGRGVFSXNRMEZCUXl4RFFVRkRMRTFCUVUwc1EwRkJReXhEUVVGRExFZEJRVWNzU1VGQlNTeEhRVUZITEVWQlFVVXNRMEZCUXp0UlFVTXpRaXhEUVVGRExFTkJRVU1zUTBGQlF6dEpRVU5ZTEVOQlFVTTdTVUZGUkN4eFEwRkJaMElzUjBGQmFFSTdVVUZCUVN4cFFrRkxRenRSUVVwSExFOUJRVTg3V1VGRFNDeFZRVUZETEVWQlFVVXNSVUZCUlN4RlFVRkZMRVZCUVVVc1EwRkJReXhKUVVGTExFOUJRVUVzUzBGQlNTeERRVUZETEZGQlFWRXNRMEZCUXl4RlFVRkZMRVZCUVVVc1JVRkJSU3hGUVVGRkxFTkJRVU1zUTBGQlF5eEZRVUY0UWl4RFFVRjNRanRaUVVOMlF5eFZRVUZETEVWQlFVVXNSVUZCUlN4RlFVRkZMRVZCUVVVc1EwRkJReXhKUVVGTExFOUJRVUVzUzBGQlNTeERRVUZETEZGQlFWRXNRMEZCUXl4RlFVRkZMRVZCUVVVc1JVRkJSU3hGUVVGRkxFTkJRVU1zUTBGQlF5eEZRVUY0UWl4RFFVRjNRanRUUVVNeFF5eERRVUZETzBsQlEwNHNRMEZCUXp0SlFVVlBMRFJDUVVGUExFZEJRV1lzVlVGQlowSXNSVUZCV1N4RlFVRkZMRVZCUVZrc1JVRkJSU3hMUVVGbExFVkJRVVVzVDBGQmFVSXNSVUZETVVVc1ZVRkJiME1zUlVGRGNFTXNSVUZCWjBJc1JVRkRhRUlzYlVKQlFYbEVMRVZCUTNwRUxHZENRVUYxUXl4RlFVTjJReXhwUWtGQk9FTTdVVUZGT1VNc1NVRkJTU3hEUVVGRExIVkNRVUYxUWl4RFFVRkRMRVZCUVVVc1JVRkJSU3hGUVVGRkxFVkJRVVVzVDBGQlR5eEZRVUZGTEZWQlFWVXNRMEZCUXl4RFFVRkRPMUZCUXpGRUxFbEJRVWtzU1VGQlNTeERRVUZETEZOQlFWTXNTVUZCU1N4SlFVRkpMRU5CUVVNc1lVRkJZU3hGUVVGRk8xbEJRM1JETEd0Q1FVRnJRaXhEUVVGRExFbEJRVWtzUTBGQlF5eFRRVUZUTEVOQlFVTXNRMEZCUXp0WlFVTnVReXhGUVVGRkxFZEJRVWNzUlVGQlJTeERRVUZETEUxQlFVMHNRMEZCUXl4dFFrRkJiVUlzUTBGQlF5eEpRVUZKTEVOQlFVTXNVMEZCVXl4RFFVRkRMRU5CUVVNc1EwRkJRenRUUVVOMlJEdFJRVU5FTEVsQlFVa3NRMEZCUXl4TFFVRkxMRU5CUVVNc1NVRkJTU3hEUVVGRExGTkJRVk1zUlVGQlJTeEZRVUZGTEVWQlFVVXNTMEZCU3l4RlFVRkZMRTlCUVU4c1EwRkJReXhEUVVGRE8xRkJReTlETEVsQlFVa3NRMEZCUXl4TFFVRkxMRU5CUVVNc1QwRkJUeXhEUVVGRExHZENRVUZuUWl4RFFVRkRMRU5CUVVNN1VVRkRja01zU1VGQlNTeEpRVUZKTEVOQlFVTXNVMEZCVXl4SlFVRkpMRWxCUVVrc1EwRkJReXhoUVVGaExFVkJRVVU3V1VGRGRFTXNTVUZCU1N4RFFVRkRMRTFCUVUwc1EwRkJReXhQUVVGUExFTkJRVU1zYVVKQlFXbENMRU5CUVVNc1EwRkJRenRaUVVOMlF5eHJRa0ZCYTBJc1EwRkJReXhKUVVGSkxFTkJRVU1zVTBGQlV5eERRVUZETEVOQlFVTTdVMEZEZEVNN1NVRkRUQ3hEUVVGRE8wbEJSVThzTUVKQlFVc3NSMEZCWWl4VlFVRmpMRVZCUVdNc1JVRkJSU3hGUVVGblFpeEZRVUZGTEZGQlFXdENMRVZCUVVVc1QwRkJhVUk3VVVGRGFrWXNTVUZCU1N4TlFVRk5MRWRCUVVjc1NVRkJTU3hoUVVGTkxFTkJRVU1zUlVGQlJTeEZRVUZGTEVWQlFVVXNRMEZCUXl4RFFVRkRPMUZCUTJoRExFMUJRVTBzUTBGQlF5eHZRa0ZCYjBJc1EwRkJReXhSUVVGUkxFTkJRVU1zUTBGQlF6dFJRVU4wUXl4TlFVRk5MRU5CUVVNc2JVSkJRVzFDTEVOQlFVTXNUMEZCVHl4RFFVRkRMRU5CUVVNN1VVRkRjRU1zVFVGQlRTeERRVUZETEV0QlFVc3NSVUZCUlN4RFFVRkRPMGxCUTI1Q0xFTkJRVU03U1VGRFRDeHBRa0ZCUXp0QlFVRkVMRU5CUVVNc1FVRnNTMFFzU1VGclMwTTdRVUZzUzFrc1owTkJRVlVpZlE9PSIsIlwidXNlIHN0cmljdFwiO1xyXG5PYmplY3QuZGVmaW5lUHJvcGVydHkoZXhwb3J0cywgXCJfX2VzTW9kdWxlXCIsIHsgdmFsdWU6IHRydWUgfSk7XHJcbnZhciBwcXVldWVfMSA9IHJlcXVpcmUoXCIuL3BxdWV1ZVwiKTtcclxudmFyIE5laWdoYm91ciA9IChmdW5jdGlvbiAoKSB7XHJcbiAgICBmdW5jdGlvbiBOZWlnaGJvdXIoaWQsIGRpc3RhbmNlKSB7XHJcbiAgICAgICAgdGhpcy5pZCA9IGlkO1xyXG4gICAgICAgIHRoaXMuZGlzdGFuY2UgPSBkaXN0YW5jZTtcclxuICAgIH1cclxuICAgIHJldHVybiBOZWlnaGJvdXI7XHJcbn0oKSk7XHJcbnZhciBOb2RlID0gKGZ1bmN0aW9uICgpIHtcclxuICAgIGZ1bmN0aW9uIE5vZGUoaWQpIHtcclxuICAgICAgICB0aGlzLmlkID0gaWQ7XHJcbiAgICAgICAgdGhpcy5uZWlnaGJvdXJzID0gW107XHJcbiAgICB9XHJcbiAgICByZXR1cm4gTm9kZTtcclxufSgpKTtcclxudmFyIFF1ZXVlRW50cnkgPSAoZnVuY3Rpb24gKCkge1xyXG4gICAgZnVuY3Rpb24gUXVldWVFbnRyeShub2RlLCBwcmV2LCBkKSB7XHJcbiAgICAgICAgdGhpcy5ub2RlID0gbm9kZTtcclxuICAgICAgICB0aGlzLnByZXYgPSBwcmV2O1xyXG4gICAgICAgIHRoaXMuZCA9IGQ7XHJcbiAgICB9XHJcbiAgICByZXR1cm4gUXVldWVFbnRyeTtcclxufSgpKTtcclxudmFyIENhbGN1bGF0b3IgPSAoZnVuY3Rpb24gKCkge1xyXG4gICAgZnVuY3Rpb24gQ2FsY3VsYXRvcihuLCBlcywgZ2V0U291cmNlSW5kZXgsIGdldFRhcmdldEluZGV4LCBnZXRMZW5ndGgpIHtcclxuICAgICAgICB0aGlzLm4gPSBuO1xyXG4gICAgICAgIHRoaXMuZXMgPSBlcztcclxuICAgICAgICB0aGlzLm5laWdoYm91cnMgPSBuZXcgQXJyYXkodGhpcy5uKTtcclxuICAgICAgICB2YXIgaSA9IHRoaXMubjtcclxuICAgICAgICB3aGlsZSAoaS0tKVxyXG4gICAgICAgICAgICB0aGlzLm5laWdoYm91cnNbaV0gPSBuZXcgTm9kZShpKTtcclxuICAgICAgICBpID0gdGhpcy5lcy5sZW5ndGg7XHJcbiAgICAgICAgd2hpbGUgKGktLSkge1xyXG4gICAgICAgICAgICB2YXIgZSA9IHRoaXMuZXNbaV07XHJcbiAgICAgICAgICAgIHZhciB1ID0gZ2V0U291cmNlSW5kZXgoZSksIHYgPSBnZXRUYXJnZXRJbmRleChlKTtcclxuICAgICAgICAgICAgdmFyIGQgPSBnZXRMZW5ndGgoZSk7XHJcbiAgICAgICAgICAgIHRoaXMubmVpZ2hib3Vyc1t1XS5uZWlnaGJvdXJzLnB1c2gobmV3IE5laWdoYm91cih2LCBkKSk7XHJcbiAgICAgICAgICAgIHRoaXMubmVpZ2hib3Vyc1t2XS5uZWlnaGJvdXJzLnB1c2gobmV3IE5laWdoYm91cih1LCBkKSk7XHJcbiAgICAgICAgfVxyXG4gICAgfVxyXG4gICAgQ2FsY3VsYXRvci5wcm90b3R5cGUuRGlzdGFuY2VNYXRyaXggPSBmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgdmFyIEQgPSBuZXcgQXJyYXkodGhpcy5uKTtcclxuICAgICAgICBmb3IgKHZhciBpID0gMDsgaSA8IHRoaXMubjsgKytpKSB7XHJcbiAgICAgICAgICAgIERbaV0gPSB0aGlzLmRpamtzdHJhTmVpZ2hib3VycyhpKTtcclxuICAgICAgICB9XHJcbiAgICAgICAgcmV0dXJuIEQ7XHJcbiAgICB9O1xyXG4gICAgQ2FsY3VsYXRvci5wcm90b3R5cGUuRGlzdGFuY2VzRnJvbU5vZGUgPSBmdW5jdGlvbiAoc3RhcnQpIHtcclxuICAgICAgICByZXR1cm4gdGhpcy5kaWprc3RyYU5laWdoYm91cnMoc3RhcnQpO1xyXG4gICAgfTtcclxuICAgIENhbGN1bGF0b3IucHJvdG90eXBlLlBhdGhGcm9tTm9kZVRvTm9kZSA9IGZ1bmN0aW9uIChzdGFydCwgZW5kKSB7XHJcbiAgICAgICAgcmV0dXJuIHRoaXMuZGlqa3N0cmFOZWlnaGJvdXJzKHN0YXJ0LCBlbmQpO1xyXG4gICAgfTtcclxuICAgIENhbGN1bGF0b3IucHJvdG90eXBlLlBhdGhGcm9tTm9kZVRvTm9kZVdpdGhQcmV2Q29zdCA9IGZ1bmN0aW9uIChzdGFydCwgZW5kLCBwcmV2Q29zdCkge1xyXG4gICAgICAgIHZhciBxID0gbmV3IHBxdWV1ZV8xLlByaW9yaXR5UXVldWUoZnVuY3Rpb24gKGEsIGIpIHsgcmV0dXJuIGEuZCA8PSBiLmQ7IH0pLCB1ID0gdGhpcy5uZWlnaGJvdXJzW3N0YXJ0XSwgcXUgPSBuZXcgUXVldWVFbnRyeSh1LCBudWxsLCAwKSwgdmlzaXRlZEZyb20gPSB7fTtcclxuICAgICAgICBxLnB1c2gocXUpO1xyXG4gICAgICAgIHdoaWxlICghcS5lbXB0eSgpKSB7XHJcbiAgICAgICAgICAgIHF1ID0gcS5wb3AoKTtcclxuICAgICAgICAgICAgdSA9IHF1Lm5vZGU7XHJcbiAgICAgICAgICAgIGlmICh1LmlkID09PSBlbmQpIHtcclxuICAgICAgICAgICAgICAgIGJyZWFrO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIHZhciBpID0gdS5uZWlnaGJvdXJzLmxlbmd0aDtcclxuICAgICAgICAgICAgd2hpbGUgKGktLSkge1xyXG4gICAgICAgICAgICAgICAgdmFyIG5laWdoYm91ciA9IHUubmVpZ2hib3Vyc1tpXSwgdiA9IHRoaXMubmVpZ2hib3Vyc1tuZWlnaGJvdXIuaWRdO1xyXG4gICAgICAgICAgICAgICAgaWYgKHF1LnByZXYgJiYgdi5pZCA9PT0gcXUucHJldi5ub2RlLmlkKVxyXG4gICAgICAgICAgICAgICAgICAgIGNvbnRpbnVlO1xyXG4gICAgICAgICAgICAgICAgdmFyIHZpZHVpZCA9IHYuaWQgKyAnLCcgKyB1LmlkO1xyXG4gICAgICAgICAgICAgICAgaWYgKHZpZHVpZCBpbiB2aXNpdGVkRnJvbSAmJiB2aXNpdGVkRnJvbVt2aWR1aWRdIDw9IHF1LmQpXHJcbiAgICAgICAgICAgICAgICAgICAgY29udGludWU7XHJcbiAgICAgICAgICAgICAgICB2YXIgY2MgPSBxdS5wcmV2ID8gcHJldkNvc3QocXUucHJldi5ub2RlLmlkLCB1LmlkLCB2LmlkKSA6IDAsIHQgPSBxdS5kICsgbmVpZ2hib3VyLmRpc3RhbmNlICsgY2M7XHJcbiAgICAgICAgICAgICAgICB2aXNpdGVkRnJvbVt2aWR1aWRdID0gdDtcclxuICAgICAgICAgICAgICAgIHEucHVzaChuZXcgUXVldWVFbnRyeSh2LCBxdSwgdCkpO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHZhciBwYXRoID0gW107XHJcbiAgICAgICAgd2hpbGUgKHF1LnByZXYpIHtcclxuICAgICAgICAgICAgcXUgPSBxdS5wcmV2O1xyXG4gICAgICAgICAgICBwYXRoLnB1c2gocXUubm9kZS5pZCk7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHJldHVybiBwYXRoO1xyXG4gICAgfTtcclxuICAgIENhbGN1bGF0b3IucHJvdG90eXBlLmRpamtzdHJhTmVpZ2hib3VycyA9IGZ1bmN0aW9uIChzdGFydCwgZGVzdCkge1xyXG4gICAgICAgIGlmIChkZXN0ID09PSB2b2lkIDApIHsgZGVzdCA9IC0xOyB9XHJcbiAgICAgICAgdmFyIHEgPSBuZXcgcHF1ZXVlXzEuUHJpb3JpdHlRdWV1ZShmdW5jdGlvbiAoYSwgYikgeyByZXR1cm4gYS5kIDw9IGIuZDsgfSksIGkgPSB0aGlzLm5laWdoYm91cnMubGVuZ3RoLCBkID0gbmV3IEFycmF5KGkpO1xyXG4gICAgICAgIHdoaWxlIChpLS0pIHtcclxuICAgICAgICAgICAgdmFyIG5vZGUgPSB0aGlzLm5laWdoYm91cnNbaV07XHJcbiAgICAgICAgICAgIG5vZGUuZCA9IGkgPT09IHN0YXJ0ID8gMCA6IE51bWJlci5QT1NJVElWRV9JTkZJTklUWTtcclxuICAgICAgICAgICAgbm9kZS5xID0gcS5wdXNoKG5vZGUpO1xyXG4gICAgICAgIH1cclxuICAgICAgICB3aGlsZSAoIXEuZW1wdHkoKSkge1xyXG4gICAgICAgICAgICB2YXIgdSA9IHEucG9wKCk7XHJcbiAgICAgICAgICAgIGRbdS5pZF0gPSB1LmQ7XHJcbiAgICAgICAgICAgIGlmICh1LmlkID09PSBkZXN0KSB7XHJcbiAgICAgICAgICAgICAgICB2YXIgcGF0aCA9IFtdO1xyXG4gICAgICAgICAgICAgICAgdmFyIHYgPSB1O1xyXG4gICAgICAgICAgICAgICAgd2hpbGUgKHR5cGVvZiB2LnByZXYgIT09ICd1bmRlZmluZWQnKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgcGF0aC5wdXNoKHYucHJldi5pZCk7XHJcbiAgICAgICAgICAgICAgICAgICAgdiA9IHYucHJldjtcclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIHJldHVybiBwYXRoO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIGkgPSB1Lm5laWdoYm91cnMubGVuZ3RoO1xyXG4gICAgICAgICAgICB3aGlsZSAoaS0tKSB7XHJcbiAgICAgICAgICAgICAgICB2YXIgbmVpZ2hib3VyID0gdS5uZWlnaGJvdXJzW2ldO1xyXG4gICAgICAgICAgICAgICAgdmFyIHYgPSB0aGlzLm5laWdoYm91cnNbbmVpZ2hib3VyLmlkXTtcclxuICAgICAgICAgICAgICAgIHZhciB0ID0gdS5kICsgbmVpZ2hib3VyLmRpc3RhbmNlO1xyXG4gICAgICAgICAgICAgICAgaWYgKHUuZCAhPT0gTnVtYmVyLk1BWF9WQUxVRSAmJiB2LmQgPiB0KSB7XHJcbiAgICAgICAgICAgICAgICAgICAgdi5kID0gdDtcclxuICAgICAgICAgICAgICAgICAgICB2LnByZXYgPSB1O1xyXG4gICAgICAgICAgICAgICAgICAgIHEucmVkdWNlS2V5KHYucSwgdiwgZnVuY3Rpb24gKGUsIHEpIHsgcmV0dXJuIGUucSA9IHE7IH0pO1xyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHJldHVybiBkO1xyXG4gICAgfTtcclxuICAgIHJldHVybiBDYWxjdWxhdG9yO1xyXG59KCkpO1xyXG5leHBvcnRzLkNhbGN1bGF0b3IgPSBDYWxjdWxhdG9yO1xyXG4vLyMgc291cmNlTWFwcGluZ1VSTD1kYXRhOmFwcGxpY2F0aW9uL2pzb247YmFzZTY0LGV5SjJaWEp6YVc5dUlqb3pMQ0ptYVd4bElqb2ljMmh2Y25SbGMzUndZWFJvY3k1cWN5SXNJbk52ZFhKalpWSnZiM1FpT2lJaUxDSnpiM1Z5WTJWeklqcGJJaTR1THk0dUwxZGxZa052YkdFdmMzSmpMM05vYjNKMFpYTjBjR0YwYUhNdWRITWlYU3dpYm1GdFpYTWlPbHRkTENKdFlYQndhVzVuY3lJNklqczdRVUZCUVN4dFEwRkJiVVE3UVVGRmJrUTdTVUZEU1N4dFFrRkJiVUlzUlVGQlZTeEZRVUZUTEZGQlFXZENPMUZCUVc1RExFOUJRVVVzUjBGQlJpeEZRVUZGTEVOQlFWRTdVVUZCVXl4aFFVRlJMRWRCUVZJc1VVRkJVU3hEUVVGUk8wbEJRVWtzUTBGQlF6dEpRVU12UkN4blFrRkJRenRCUVVGRUxFTkJRVU1zUVVGR1JDeEpRVVZETzBGQlJVUTdTVUZEU1N4alFVRnRRaXhGUVVGVk8xRkJRVllzVDBGQlJTeEhRVUZHTEVWQlFVVXNRMEZCVVR0UlFVTjZRaXhKUVVGSkxFTkJRVU1zVlVGQlZTeEhRVUZITEVWQlFVVXNRMEZCUXp0SlFVTjZRaXhEUVVGRE8wbEJTMHdzVjBGQlF6dEJRVUZFTEVOQlFVTXNRVUZTUkN4SlFWRkRPMEZCUlVRN1NVRkRTU3h2UWtGQmJVSXNTVUZCVlN4RlFVRlRMRWxCUVdkQ0xFVkJRVk1zUTBGQlV6dFJRVUZ5UkN4VFFVRkpMRWRCUVVvc1NVRkJTU3hEUVVGTk8xRkJRVk1zVTBGQlNTeEhRVUZLTEVsQlFVa3NRMEZCV1R0UlFVRlRMRTFCUVVNc1IwRkJSQ3hEUVVGRExFTkJRVkU3U1VGQlJ5eERRVUZETzBsQlEyaEdMR2xDUVVGRE8wRkJRVVFzUTBGQlF5eEJRVVpFTEVsQlJVTTdRVUZUUkR0SlFVZEpMRzlDUVVGdFFpeERRVUZUTEVWQlFWTXNSVUZCVlN4RlFVRkZMR05CUVcxRExFVkJRVVVzWTBGQmJVTXNSVUZCUlN4VFFVRTRRanRSUVVGMFNTeE5RVUZETEVkQlFVUXNRMEZCUXl4RFFVRlJPMUZCUVZNc1QwRkJSU3hIUVVGR0xFVkJRVVVzUTBGQlVUdFJRVU16UXl4SlFVRkpMRU5CUVVNc1ZVRkJWU3hIUVVGSExFbEJRVWtzUzBGQlN5eERRVUZETEVsQlFVa3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJRenRSUVVOd1F5eEpRVUZKTEVOQlFVTXNSMEZCUnl4SlFVRkpMRU5CUVVNc1EwRkJReXhEUVVGRE8xRkJRVU1zVDBGQlR5eERRVUZETEVWQlFVVTdXVUZCUlN4SlFVRkpMRU5CUVVNc1ZVRkJWU3hEUVVGRExFTkJRVU1zUTBGQlF5eEhRVUZITEVsQlFVa3NTVUZCU1N4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRE8xRkJSVGRFTEVOQlFVTXNSMEZCUnl4SlFVRkpMRU5CUVVNc1JVRkJSU3hEUVVGRExFMUJRVTBzUTBGQlF6dFJRVUZETEU5QlFVOHNRMEZCUXl4RlFVRkZMRVZCUVVVN1dVRkROVUlzU1VGQlNTeERRVUZETEVkQlFVY3NTVUZCU1N4RFFVRkRMRVZCUVVVc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dFpRVU51UWl4SlFVRkpMRU5CUVVNc1IwRkJWeXhqUVVGakxFTkJRVU1zUTBGQlF5eERRVUZETEVWQlFVVXNRMEZCUXl4SFFVRlhMR05CUVdNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dFpRVU5xUlN4SlFVRkpMRU5CUVVNc1IwRkJSeXhUUVVGVExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTTdXVUZEY2tJc1NVRkJTU3hEUVVGRExGVkJRVlVzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4VlFVRlZMRU5CUVVNc1NVRkJTU3hEUVVGRExFbEJRVWtzVTBGQlV5eERRVUZETEVOQlFVTXNSVUZCUlN4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRE8xbEJRM2hFTEVsQlFVa3NRMEZCUXl4VlFVRlZMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zVlVGQlZTeERRVUZETEVsQlFVa3NRMEZCUXl4SlFVRkpMRk5CUVZNc1EwRkJReXhEUVVGRExFVkJRVVVzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXp0VFFVTXpSRHRKUVVOTUxFTkJRVU03U1VGVlJDeHRRMEZCWXl4SFFVRmtPMUZCUTBrc1NVRkJTU3hEUVVGRExFZEJRVWNzU1VGQlNTeExRVUZMTEVOQlFVTXNTVUZCU1N4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRE8xRkJRekZDTEV0QlFVc3NTVUZCU1N4RFFVRkRMRWRCUVVjc1EwRkJReXhGUVVGRkxFTkJRVU1zUjBGQlJ5eEpRVUZKTEVOQlFVTXNRMEZCUXl4RlFVRkZMRVZCUVVVc1EwRkJReXhGUVVGRk8xbEJRemRDTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1IwRkJSeXhKUVVGSkxFTkJRVU1zYTBKQlFXdENMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU03VTBGRGNrTTdVVUZEUkN4UFFVRlBMRU5CUVVNc1EwRkJRenRKUVVOaUxFTkJRVU03U1VGUlJDeHpRMEZCYVVJc1IwRkJha0lzVlVGQmEwSXNTMEZCWVR0UlFVTXpRaXhQUVVGUExFbEJRVWtzUTBGQlF5eHJRa0ZCYTBJc1EwRkJReXhMUVVGTExFTkJRVU1zUTBGQlF6dEpRVU14UXl4RFFVRkRPMGxCUlVRc2RVTkJRV3RDTEVkQlFXeENMRlZCUVcxQ0xFdEJRV0VzUlVGQlJTeEhRVUZYTzFGQlEzcERMRTlCUVU4c1NVRkJTU3hEUVVGRExHdENRVUZyUWl4RFFVRkRMRXRCUVVzc1JVRkJSU3hIUVVGSExFTkJRVU1zUTBGQlF6dEpRVU12UXl4RFFVRkRPMGxCUzBRc2JVUkJRVGhDTEVkQlFUbENMRlZCUTBrc1MwRkJZU3hGUVVOaUxFZEJRVmNzUlVGRFdDeFJRVUU0UXp0UlFVVTVReXhKUVVGSkxFTkJRVU1zUjBGQlJ5eEpRVUZKTEhOQ1FVRmhMRU5CUVdFc1ZVRkJReXhEUVVGRExFVkJRVVVzUTBGQlF5eEpRVUZMTEU5QlFVRXNRMEZCUXl4RFFVRkRMRU5CUVVNc1NVRkJTU3hEUVVGRExFTkJRVU1zUTBGQlF5eEZRVUZXTEVOQlFWVXNRMEZCUXl4RlFVTjJSQ3hEUVVGRExFZEJRVk1zU1VGQlNTeERRVUZETEZWQlFWVXNRMEZCUXl4TFFVRkxMRU5CUVVNc1JVRkRhRU1zUlVGQlJTeEhRVUZsTEVsQlFVa3NWVUZCVlN4RFFVRkRMRU5CUVVNc1JVRkJReXhKUVVGSkxFVkJRVU1zUTBGQlF5eERRVUZETEVWQlEzcERMRmRCUVZjc1IwRkJSeXhGUVVGRkxFTkJRVU03VVVGRGNrSXNRMEZCUXl4RFFVRkRMRWxCUVVrc1EwRkJReXhGUVVGRkxFTkJRVU1zUTBGQlF6dFJRVU5ZTEU5QlFVMHNRMEZCUXl4RFFVRkRMRU5CUVVNc1MwRkJTeXhGUVVGRkxFVkJRVVU3V1VGRFpDeEZRVUZGTEVkQlFVY3NRMEZCUXl4RFFVRkRMRWRCUVVjc1JVRkJSU3hEUVVGRE8xbEJRMklzUTBGQlF5eEhRVUZITEVWQlFVVXNRMEZCUXl4SlFVRkpMRU5CUVVNN1dVRkRXaXhKUVVGSkxFTkJRVU1zUTBGQlF5eEZRVUZGTEV0QlFVc3NSMEZCUnl4RlFVRkZPMmRDUVVOa0xFMUJRVTA3WVVGRFZEdFpRVU5FTEVsQlFVa3NRMEZCUXl4SFFVRkhMRU5CUVVNc1EwRkJReXhWUVVGVkxFTkJRVU1zVFVGQlRTeERRVUZETzFsQlFVTXNUMEZCVHl4RFFVRkRMRVZCUVVVc1JVRkJSVHRuUWtGRGNrTXNTVUZCU1N4VFFVRlRMRWRCUVVjc1EwRkJReXhEUVVGRExGVkJRVlVzUTBGQlF5eERRVUZETEVOQlFVTXNSVUZETTBJc1EwRkJReXhIUVVGSExFbEJRVWtzUTBGQlF5eFZRVUZWTEVOQlFVTXNVMEZCVXl4RFFVRkRMRVZCUVVVc1EwRkJReXhEUVVGRE8yZENRVWQwUXl4SlFVRkpMRVZCUVVVc1EwRkJReXhKUVVGSkxFbEJRVWtzUTBGQlF5eERRVUZETEVWQlFVVXNTMEZCU3l4RlFVRkZMRU5CUVVNc1NVRkJTU3hEUVVGRExFbEJRVWtzUTBGQlF5eEZRVUZGTzI5Q1FVRkZMRk5CUVZNN1owSkJTV3hFTEVsQlFVa3NUVUZCVFN4SFFVRkhMRU5CUVVNc1EwRkJReXhGUVVGRkxFZEJRVWNzUjBGQlJ5eEhRVUZITEVOQlFVTXNRMEZCUXl4RlFVRkZMRU5CUVVNN1owSkJReTlDTEVsQlFVY3NUVUZCVFN4SlFVRkpMRmRCUVZjc1NVRkJTU3hYUVVGWExFTkJRVU1zVFVGQlRTeERRVUZETEVsQlFVa3NSVUZCUlN4RFFVRkRMRU5CUVVNN2IwSkJRMjVFTEZOQlFWTTdaMEpCUldJc1NVRkJTU3hGUVVGRkxFZEJRVWNzUlVGQlJTeERRVUZETEVsQlFVa3NRMEZCUXl4RFFVRkRMRU5CUVVNc1VVRkJVU3hEUVVGRExFVkJRVVVzUTBGQlF5eEpRVUZKTEVOQlFVTXNTVUZCU1N4RFFVRkRMRVZCUVVVc1JVRkJSU3hEUVVGRExFTkJRVU1zUlVGQlJTeEZRVUZGTEVOQlFVTXNRMEZCUXl4RlFVRkZMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eEZRVU40UkN4RFFVRkRMRWRCUVVjc1JVRkJSU3hEUVVGRExFTkJRVU1zUjBGQlJ5eFRRVUZUTEVOQlFVTXNVVUZCVVN4SFFVRkhMRVZCUVVVc1EwRkJRenRuUWtGSGRrTXNWMEZCVnl4RFFVRkRMRTFCUVUwc1EwRkJReXhIUVVGSExFTkJRVU1zUTBGQlF6dG5Ra0ZEZUVJc1EwRkJReXhEUVVGRExFbEJRVWtzUTBGQlF5eEpRVUZKTEZWQlFWVXNRMEZCUXl4RFFVRkRMRVZCUVVVc1JVRkJSU3hGUVVGRkxFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTTdZVUZEY0VNN1UwRkRTanRSUVVORUxFbEJRVWtzU1VGQlNTeEhRVUZaTEVWQlFVVXNRMEZCUXp0UlFVTjJRaXhQUVVGUExFVkJRVVVzUTBGQlF5eEpRVUZKTEVWQlFVVTdXVUZEV2l4RlFVRkZMRWRCUVVjc1JVRkJSU3hEUVVGRExFbEJRVWtzUTBGQlF6dFpRVU5pTEVsQlFVa3NRMEZCUXl4SlFVRkpMRU5CUVVNc1JVRkJSU3hEUVVGRExFbEJRVWtzUTBGQlF5eEZRVUZGTEVOQlFVTXNRMEZCUXp0VFFVTjZRanRSUVVORUxFOUJRVThzU1VGQlNTeERRVUZETzBsQlEyaENMRU5CUVVNN1NVRkZUeXgxUTBGQmEwSXNSMEZCTVVJc1ZVRkJNa0lzUzBGQllTeEZRVUZGTEVsQlFXbENPMUZCUVdwQ0xIRkNRVUZCTEVWQlFVRXNVVUZCWjBJc1EwRkJRenRSUVVOMlJDeEpRVUZKTEVOQlFVTXNSMEZCUnl4SlFVRkpMSE5DUVVGaExFTkJRVThzVlVGQlF5eERRVUZETEVWQlFVVXNRMEZCUXl4SlFVRkxMRTlCUVVFc1EwRkJReXhEUVVGRExFTkJRVU1zU1VGQlNTeERRVUZETEVOQlFVTXNRMEZCUXl4RlFVRldMRU5CUVZVc1EwRkJReXhGUVVOcVJDeERRVUZETEVkQlFVY3NTVUZCU1N4RFFVRkRMRlZCUVZVc1EwRkJReXhOUVVGTkxFVkJRekZDTEVOQlFVTXNSMEZCWVN4SlFVRkpMRXRCUVVzc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dFJRVU12UWl4UFFVRlBMRU5CUVVNc1JVRkJSU3hGUVVGRk8xbEJRMUlzU1VGQlNTeEpRVUZKTEVkQlFWTXNTVUZCU1N4RFFVRkRMRlZCUVZVc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dFpRVU53UXl4SlFVRkpMRU5CUVVNc1EwRkJReXhIUVVGSExFTkJRVU1zUzBGQlN5eExRVUZMTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zVFVGQlRTeERRVUZETEdsQ1FVRnBRaXhEUVVGRE8xbEJRM0JFTEVsQlFVa3NRMEZCUXl4RFFVRkRMRWRCUVVjc1EwRkJReXhEUVVGRExFbEJRVWtzUTBGQlF5eEpRVUZKTEVOQlFVTXNRMEZCUXp0VFFVTjZRanRSUVVORUxFOUJRVThzUTBGQlF5eERRVUZETEVOQlFVTXNTMEZCU3l4RlFVRkZMRVZCUVVVN1dVRkZaaXhKUVVGSkxFTkJRVU1zUjBGQlJ5eERRVUZETEVOQlFVTXNSMEZCUnl4RlFVRkZMRU5CUVVNN1dVRkRhRUlzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RlFVRkZMRU5CUVVNc1IwRkJSeXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETzFsQlEyUXNTVUZCU1N4RFFVRkRMRU5CUVVNc1JVRkJSU3hMUVVGTExFbEJRVWtzUlVGQlJUdG5Ra0ZEWml4SlFVRkpMRWxCUVVrc1IwRkJZU3hGUVVGRkxFTkJRVU03WjBKQlEzaENMRWxCUVVrc1EwRkJReXhIUVVGSExFTkJRVU1zUTBGQlF6dG5Ra0ZEVml4UFFVRlBMRTlCUVU4c1EwRkJReXhEUVVGRExFbEJRVWtzUzBGQlN5eFhRVUZYTEVWQlFVVTdiMEpCUTJ4RExFbEJRVWtzUTBGQlF5eEpRVUZKTEVOQlFVTXNRMEZCUXl4RFFVRkRMRWxCUVVrc1EwRkJReXhGUVVGRkxFTkJRVU1zUTBGQlF6dHZRa0ZEY2tJc1EwRkJReXhIUVVGSExFTkJRVU1zUTBGQlF5eEpRVUZKTEVOQlFVTTdhVUpCUTJRN1owSkJRMFFzVDBGQlR5eEpRVUZKTEVOQlFVTTdZVUZEWmp0WlFVTkVMRU5CUVVNc1IwRkJSeXhEUVVGRExFTkJRVU1zVlVGQlZTeERRVUZETEUxQlFVMHNRMEZCUXp0WlFVRkRMRTlCUVU4c1EwRkJReXhGUVVGRkxFVkJRVVU3WjBKQlEycERMRWxCUVVrc1UwRkJVeXhIUVVGSExFTkJRVU1zUTBGQlF5eFZRVUZWTEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNN1owSkJRMmhETEVsQlFVa3NRMEZCUXl4SFFVRkhMRWxCUVVrc1EwRkJReXhWUVVGVkxFTkJRVU1zVTBGQlV5eERRVUZETEVWQlFVVXNRMEZCUXl4RFFVRkRPMmRDUVVOMFF5eEpRVUZKTEVOQlFVTXNSMEZCUnl4RFFVRkRMRU5CUVVNc1EwRkJReXhIUVVGSExGTkJRVk1zUTBGQlF5eFJRVUZSTEVOQlFVTTdaMEpCUTJwRExFbEJRVWtzUTBGQlF5eERRVUZETEVOQlFVTXNTMEZCU3l4TlFVRk5MRU5CUVVNc1UwRkJVeXhKUVVGSkxFTkJRVU1zUTBGQlF5eERRVUZETEVkQlFVY3NRMEZCUXl4RlFVRkZPMjlDUVVOeVF5eERRVUZETEVOQlFVTXNRMEZCUXl4SFFVRkhMRU5CUVVNc1EwRkJRenR2UWtGRFVpeERRVUZETEVOQlFVTXNTVUZCU1N4SFFVRkhMRU5CUVVNc1EwRkJRenR2UWtGRFdDeERRVUZETEVOQlFVTXNVMEZCVXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFVkJRVVVzUTBGQlF5eEZRVUZGTEZWQlFVTXNRMEZCUXl4RlFVRkRMRU5CUVVNc1NVRkJSeXhQUVVGQkxFTkJRVU1zUTBGQlF5eERRVUZETEVkQlFVY3NRMEZCUXl4RlFVRlFMRU5CUVU4c1EwRkJReXhEUVVGRE8ybENRVU4yUXp0aFFVTktPMU5CUTBvN1VVRkRSQ3hQUVVGUExFTkJRVU1zUTBGQlF6dEpRVU5pTEVOQlFVTTdTVUZEVEN4cFFrRkJRenRCUVVGRUxFTkJRVU1zUVVGcVNVUXNTVUZwU1VNN1FVRnFTVmtzWjBOQlFWVWlmUT09IiwiXCJ1c2Ugc3RyaWN0XCI7XHJcbk9iamVjdC5kZWZpbmVQcm9wZXJ0eShleHBvcnRzLCBcIl9fZXNNb2R1bGVcIiwgeyB2YWx1ZTogdHJ1ZSB9KTtcclxudmFyIFBvc2l0aW9uU3RhdHMgPSAoZnVuY3Rpb24gKCkge1xyXG4gICAgZnVuY3Rpb24gUG9zaXRpb25TdGF0cyhzY2FsZSkge1xyXG4gICAgICAgIHRoaXMuc2NhbGUgPSBzY2FsZTtcclxuICAgICAgICB0aGlzLkFCID0gMDtcclxuICAgICAgICB0aGlzLkFEID0gMDtcclxuICAgICAgICB0aGlzLkEyID0gMDtcclxuICAgIH1cclxuICAgIFBvc2l0aW9uU3RhdHMucHJvdG90eXBlLmFkZFZhcmlhYmxlID0gZnVuY3Rpb24gKHYpIHtcclxuICAgICAgICB2YXIgYWkgPSB0aGlzLnNjYWxlIC8gdi5zY2FsZTtcclxuICAgICAgICB2YXIgYmkgPSB2Lm9mZnNldCAvIHYuc2NhbGU7XHJcbiAgICAgICAgdmFyIHdpID0gdi53ZWlnaHQ7XHJcbiAgICAgICAgdGhpcy5BQiArPSB3aSAqIGFpICogYmk7XHJcbiAgICAgICAgdGhpcy5BRCArPSB3aSAqIGFpICogdi5kZXNpcmVkUG9zaXRpb247XHJcbiAgICAgICAgdGhpcy5BMiArPSB3aSAqIGFpICogYWk7XHJcbiAgICB9O1xyXG4gICAgUG9zaXRpb25TdGF0cy5wcm90b3R5cGUuZ2V0UG9zbiA9IGZ1bmN0aW9uICgpIHtcclxuICAgICAgICByZXR1cm4gKHRoaXMuQUQgLSB0aGlzLkFCKSAvIHRoaXMuQTI7XHJcbiAgICB9O1xyXG4gICAgcmV0dXJuIFBvc2l0aW9uU3RhdHM7XHJcbn0oKSk7XHJcbmV4cG9ydHMuUG9zaXRpb25TdGF0cyA9IFBvc2l0aW9uU3RhdHM7XHJcbnZhciBDb25zdHJhaW50ID0gKGZ1bmN0aW9uICgpIHtcclxuICAgIGZ1bmN0aW9uIENvbnN0cmFpbnQobGVmdCwgcmlnaHQsIGdhcCwgZXF1YWxpdHkpIHtcclxuICAgICAgICBpZiAoZXF1YWxpdHkgPT09IHZvaWQgMCkgeyBlcXVhbGl0eSA9IGZhbHNlOyB9XHJcbiAgICAgICAgdGhpcy5sZWZ0ID0gbGVmdDtcclxuICAgICAgICB0aGlzLnJpZ2h0ID0gcmlnaHQ7XHJcbiAgICAgICAgdGhpcy5nYXAgPSBnYXA7XHJcbiAgICAgICAgdGhpcy5lcXVhbGl0eSA9IGVxdWFsaXR5O1xyXG4gICAgICAgIHRoaXMuYWN0aXZlID0gZmFsc2U7XHJcbiAgICAgICAgdGhpcy51bnNhdGlzZmlhYmxlID0gZmFsc2U7XHJcbiAgICAgICAgdGhpcy5sZWZ0ID0gbGVmdDtcclxuICAgICAgICB0aGlzLnJpZ2h0ID0gcmlnaHQ7XHJcbiAgICAgICAgdGhpcy5nYXAgPSBnYXA7XHJcbiAgICAgICAgdGhpcy5lcXVhbGl0eSA9IGVxdWFsaXR5O1xyXG4gICAgfVxyXG4gICAgQ29uc3RyYWludC5wcm90b3R5cGUuc2xhY2sgPSBmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgcmV0dXJuIHRoaXMudW5zYXRpc2ZpYWJsZSA/IE51bWJlci5NQVhfVkFMVUVcclxuICAgICAgICAgICAgOiB0aGlzLnJpZ2h0LnNjYWxlICogdGhpcy5yaWdodC5wb3NpdGlvbigpIC0gdGhpcy5nYXBcclxuICAgICAgICAgICAgICAgIC0gdGhpcy5sZWZ0LnNjYWxlICogdGhpcy5sZWZ0LnBvc2l0aW9uKCk7XHJcbiAgICB9O1xyXG4gICAgcmV0dXJuIENvbnN0cmFpbnQ7XHJcbn0oKSk7XHJcbmV4cG9ydHMuQ29uc3RyYWludCA9IENvbnN0cmFpbnQ7XHJcbnZhciBWYXJpYWJsZSA9IChmdW5jdGlvbiAoKSB7XHJcbiAgICBmdW5jdGlvbiBWYXJpYWJsZShkZXNpcmVkUG9zaXRpb24sIHdlaWdodCwgc2NhbGUpIHtcclxuICAgICAgICBpZiAod2VpZ2h0ID09PSB2b2lkIDApIHsgd2VpZ2h0ID0gMTsgfVxyXG4gICAgICAgIGlmIChzY2FsZSA9PT0gdm9pZCAwKSB7IHNjYWxlID0gMTsgfVxyXG4gICAgICAgIHRoaXMuZGVzaXJlZFBvc2l0aW9uID0gZGVzaXJlZFBvc2l0aW9uO1xyXG4gICAgICAgIHRoaXMud2VpZ2h0ID0gd2VpZ2h0O1xyXG4gICAgICAgIHRoaXMuc2NhbGUgPSBzY2FsZTtcclxuICAgICAgICB0aGlzLm9mZnNldCA9IDA7XHJcbiAgICB9XHJcbiAgICBWYXJpYWJsZS5wcm90b3R5cGUuZGZkdiA9IGZ1bmN0aW9uICgpIHtcclxuICAgICAgICByZXR1cm4gMi4wICogdGhpcy53ZWlnaHQgKiAodGhpcy5wb3NpdGlvbigpIC0gdGhpcy5kZXNpcmVkUG9zaXRpb24pO1xyXG4gICAgfTtcclxuICAgIFZhcmlhYmxlLnByb3RvdHlwZS5wb3NpdGlvbiA9IGZ1bmN0aW9uICgpIHtcclxuICAgICAgICByZXR1cm4gKHRoaXMuYmxvY2sucHMuc2NhbGUgKiB0aGlzLmJsb2NrLnBvc24gKyB0aGlzLm9mZnNldCkgLyB0aGlzLnNjYWxlO1xyXG4gICAgfTtcclxuICAgIFZhcmlhYmxlLnByb3RvdHlwZS52aXNpdE5laWdoYm91cnMgPSBmdW5jdGlvbiAocHJldiwgZikge1xyXG4gICAgICAgIHZhciBmZiA9IGZ1bmN0aW9uIChjLCBuZXh0KSB7IHJldHVybiBjLmFjdGl2ZSAmJiBwcmV2ICE9PSBuZXh0ICYmIGYoYywgbmV4dCk7IH07XHJcbiAgICAgICAgdGhpcy5jT3V0LmZvckVhY2goZnVuY3Rpb24gKGMpIHsgcmV0dXJuIGZmKGMsIGMucmlnaHQpOyB9KTtcclxuICAgICAgICB0aGlzLmNJbi5mb3JFYWNoKGZ1bmN0aW9uIChjKSB7IHJldHVybiBmZihjLCBjLmxlZnQpOyB9KTtcclxuICAgIH07XHJcbiAgICByZXR1cm4gVmFyaWFibGU7XHJcbn0oKSk7XHJcbmV4cG9ydHMuVmFyaWFibGUgPSBWYXJpYWJsZTtcclxudmFyIEJsb2NrID0gKGZ1bmN0aW9uICgpIHtcclxuICAgIGZ1bmN0aW9uIEJsb2NrKHYpIHtcclxuICAgICAgICB0aGlzLnZhcnMgPSBbXTtcclxuICAgICAgICB2Lm9mZnNldCA9IDA7XHJcbiAgICAgICAgdGhpcy5wcyA9IG5ldyBQb3NpdGlvblN0YXRzKHYuc2NhbGUpO1xyXG4gICAgICAgIHRoaXMuYWRkVmFyaWFibGUodik7XHJcbiAgICB9XHJcbiAgICBCbG9jay5wcm90b3R5cGUuYWRkVmFyaWFibGUgPSBmdW5jdGlvbiAodikge1xyXG4gICAgICAgIHYuYmxvY2sgPSB0aGlzO1xyXG4gICAgICAgIHRoaXMudmFycy5wdXNoKHYpO1xyXG4gICAgICAgIHRoaXMucHMuYWRkVmFyaWFibGUodik7XHJcbiAgICAgICAgdGhpcy5wb3NuID0gdGhpcy5wcy5nZXRQb3NuKCk7XHJcbiAgICB9O1xyXG4gICAgQmxvY2sucHJvdG90eXBlLnVwZGF0ZVdlaWdodGVkUG9zaXRpb24gPSBmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgdGhpcy5wcy5BQiA9IHRoaXMucHMuQUQgPSB0aGlzLnBzLkEyID0gMDtcclxuICAgICAgICBmb3IgKHZhciBpID0gMCwgbiA9IHRoaXMudmFycy5sZW5ndGg7IGkgPCBuOyArK2kpXHJcbiAgICAgICAgICAgIHRoaXMucHMuYWRkVmFyaWFibGUodGhpcy52YXJzW2ldKTtcclxuICAgICAgICB0aGlzLnBvc24gPSB0aGlzLnBzLmdldFBvc24oKTtcclxuICAgIH07XHJcbiAgICBCbG9jay5wcm90b3R5cGUuY29tcHV0ZV9sbSA9IGZ1bmN0aW9uICh2LCB1LCBwb3N0QWN0aW9uKSB7XHJcbiAgICAgICAgdmFyIF90aGlzID0gdGhpcztcclxuICAgICAgICB2YXIgZGZkdiA9IHYuZGZkdigpO1xyXG4gICAgICAgIHYudmlzaXROZWlnaGJvdXJzKHUsIGZ1bmN0aW9uIChjLCBuZXh0KSB7XHJcbiAgICAgICAgICAgIHZhciBfZGZkdiA9IF90aGlzLmNvbXB1dGVfbG0obmV4dCwgdiwgcG9zdEFjdGlvbik7XHJcbiAgICAgICAgICAgIGlmIChuZXh0ID09PSBjLnJpZ2h0KSB7XHJcbiAgICAgICAgICAgICAgICBkZmR2ICs9IF9kZmR2ICogYy5sZWZ0LnNjYWxlO1xyXG4gICAgICAgICAgICAgICAgYy5sbSA9IF9kZmR2O1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIGVsc2Uge1xyXG4gICAgICAgICAgICAgICAgZGZkdiArPSBfZGZkdiAqIGMucmlnaHQuc2NhbGU7XHJcbiAgICAgICAgICAgICAgICBjLmxtID0gLV9kZmR2O1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIHBvc3RBY3Rpb24oYyk7XHJcbiAgICAgICAgfSk7XHJcbiAgICAgICAgcmV0dXJuIGRmZHYgLyB2LnNjYWxlO1xyXG4gICAgfTtcclxuICAgIEJsb2NrLnByb3RvdHlwZS5wb3B1bGF0ZVNwbGl0QmxvY2sgPSBmdW5jdGlvbiAodiwgcHJldikge1xyXG4gICAgICAgIHZhciBfdGhpcyA9IHRoaXM7XHJcbiAgICAgICAgdi52aXNpdE5laWdoYm91cnMocHJldiwgZnVuY3Rpb24gKGMsIG5leHQpIHtcclxuICAgICAgICAgICAgbmV4dC5vZmZzZXQgPSB2Lm9mZnNldCArIChuZXh0ID09PSBjLnJpZ2h0ID8gYy5nYXAgOiAtYy5nYXApO1xyXG4gICAgICAgICAgICBfdGhpcy5hZGRWYXJpYWJsZShuZXh0KTtcclxuICAgICAgICAgICAgX3RoaXMucG9wdWxhdGVTcGxpdEJsb2NrKG5leHQsIHYpO1xyXG4gICAgICAgIH0pO1xyXG4gICAgfTtcclxuICAgIEJsb2NrLnByb3RvdHlwZS50cmF2ZXJzZSA9IGZ1bmN0aW9uICh2aXNpdCwgYWNjLCB2LCBwcmV2KSB7XHJcbiAgICAgICAgdmFyIF90aGlzID0gdGhpcztcclxuICAgICAgICBpZiAodiA9PT0gdm9pZCAwKSB7IHYgPSB0aGlzLnZhcnNbMF07IH1cclxuICAgICAgICBpZiAocHJldiA9PT0gdm9pZCAwKSB7IHByZXYgPSBudWxsOyB9XHJcbiAgICAgICAgdi52aXNpdE5laWdoYm91cnMocHJldiwgZnVuY3Rpb24gKGMsIG5leHQpIHtcclxuICAgICAgICAgICAgYWNjLnB1c2godmlzaXQoYykpO1xyXG4gICAgICAgICAgICBfdGhpcy50cmF2ZXJzZSh2aXNpdCwgYWNjLCBuZXh0LCB2KTtcclxuICAgICAgICB9KTtcclxuICAgIH07XHJcbiAgICBCbG9jay5wcm90b3R5cGUuZmluZE1pbkxNID0gZnVuY3Rpb24gKCkge1xyXG4gICAgICAgIHZhciBtID0gbnVsbDtcclxuICAgICAgICB0aGlzLmNvbXB1dGVfbG0odGhpcy52YXJzWzBdLCBudWxsLCBmdW5jdGlvbiAoYykge1xyXG4gICAgICAgICAgICBpZiAoIWMuZXF1YWxpdHkgJiYgKG0gPT09IG51bGwgfHwgYy5sbSA8IG0ubG0pKVxyXG4gICAgICAgICAgICAgICAgbSA9IGM7XHJcbiAgICAgICAgfSk7XHJcbiAgICAgICAgcmV0dXJuIG07XHJcbiAgICB9O1xyXG4gICAgQmxvY2sucHJvdG90eXBlLmZpbmRNaW5MTUJldHdlZW4gPSBmdW5jdGlvbiAobHYsIHJ2KSB7XHJcbiAgICAgICAgdGhpcy5jb21wdXRlX2xtKGx2LCBudWxsLCBmdW5jdGlvbiAoKSB7IH0pO1xyXG4gICAgICAgIHZhciBtID0gbnVsbDtcclxuICAgICAgICB0aGlzLmZpbmRQYXRoKGx2LCBudWxsLCBydiwgZnVuY3Rpb24gKGMsIG5leHQpIHtcclxuICAgICAgICAgICAgaWYgKCFjLmVxdWFsaXR5ICYmIGMucmlnaHQgPT09IG5leHQgJiYgKG0gPT09IG51bGwgfHwgYy5sbSA8IG0ubG0pKVxyXG4gICAgICAgICAgICAgICAgbSA9IGM7XHJcbiAgICAgICAgfSk7XHJcbiAgICAgICAgcmV0dXJuIG07XHJcbiAgICB9O1xyXG4gICAgQmxvY2sucHJvdG90eXBlLmZpbmRQYXRoID0gZnVuY3Rpb24gKHYsIHByZXYsIHRvLCB2aXNpdCkge1xyXG4gICAgICAgIHZhciBfdGhpcyA9IHRoaXM7XHJcbiAgICAgICAgdmFyIGVuZEZvdW5kID0gZmFsc2U7XHJcbiAgICAgICAgdi52aXNpdE5laWdoYm91cnMocHJldiwgZnVuY3Rpb24gKGMsIG5leHQpIHtcclxuICAgICAgICAgICAgaWYgKCFlbmRGb3VuZCAmJiAobmV4dCA9PT0gdG8gfHwgX3RoaXMuZmluZFBhdGgobmV4dCwgdiwgdG8sIHZpc2l0KSkpIHtcclxuICAgICAgICAgICAgICAgIGVuZEZvdW5kID0gdHJ1ZTtcclxuICAgICAgICAgICAgICAgIHZpc2l0KGMsIG5leHQpO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfSk7XHJcbiAgICAgICAgcmV0dXJuIGVuZEZvdW5kO1xyXG4gICAgfTtcclxuICAgIEJsb2NrLnByb3RvdHlwZS5pc0FjdGl2ZURpcmVjdGVkUGF0aEJldHdlZW4gPSBmdW5jdGlvbiAodSwgdikge1xyXG4gICAgICAgIGlmICh1ID09PSB2KVxyXG4gICAgICAgICAgICByZXR1cm4gdHJ1ZTtcclxuICAgICAgICB2YXIgaSA9IHUuY091dC5sZW5ndGg7XHJcbiAgICAgICAgd2hpbGUgKGktLSkge1xyXG4gICAgICAgICAgICB2YXIgYyA9IHUuY091dFtpXTtcclxuICAgICAgICAgICAgaWYgKGMuYWN0aXZlICYmIHRoaXMuaXNBY3RpdmVEaXJlY3RlZFBhdGhCZXR3ZWVuKGMucmlnaHQsIHYpKVxyXG4gICAgICAgICAgICAgICAgcmV0dXJuIHRydWU7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHJldHVybiBmYWxzZTtcclxuICAgIH07XHJcbiAgICBCbG9jay5zcGxpdCA9IGZ1bmN0aW9uIChjKSB7XHJcbiAgICAgICAgYy5hY3RpdmUgPSBmYWxzZTtcclxuICAgICAgICByZXR1cm4gW0Jsb2NrLmNyZWF0ZVNwbGl0QmxvY2soYy5sZWZ0KSwgQmxvY2suY3JlYXRlU3BsaXRCbG9jayhjLnJpZ2h0KV07XHJcbiAgICB9O1xyXG4gICAgQmxvY2suY3JlYXRlU3BsaXRCbG9jayA9IGZ1bmN0aW9uIChzdGFydFZhcikge1xyXG4gICAgICAgIHZhciBiID0gbmV3IEJsb2NrKHN0YXJ0VmFyKTtcclxuICAgICAgICBiLnBvcHVsYXRlU3BsaXRCbG9jayhzdGFydFZhciwgbnVsbCk7XHJcbiAgICAgICAgcmV0dXJuIGI7XHJcbiAgICB9O1xyXG4gICAgQmxvY2sucHJvdG90eXBlLnNwbGl0QmV0d2VlbiA9IGZ1bmN0aW9uICh2bCwgdnIpIHtcclxuICAgICAgICB2YXIgYyA9IHRoaXMuZmluZE1pbkxNQmV0d2Vlbih2bCwgdnIpO1xyXG4gICAgICAgIGlmIChjICE9PSBudWxsKSB7XHJcbiAgICAgICAgICAgIHZhciBicyA9IEJsb2NrLnNwbGl0KGMpO1xyXG4gICAgICAgICAgICByZXR1cm4geyBjb25zdHJhaW50OiBjLCBsYjogYnNbMF0sIHJiOiBic1sxXSB9O1xyXG4gICAgICAgIH1cclxuICAgICAgICByZXR1cm4gbnVsbDtcclxuICAgIH07XHJcbiAgICBCbG9jay5wcm90b3R5cGUubWVyZ2VBY3Jvc3MgPSBmdW5jdGlvbiAoYiwgYywgZGlzdCkge1xyXG4gICAgICAgIGMuYWN0aXZlID0gdHJ1ZTtcclxuICAgICAgICBmb3IgKHZhciBpID0gMCwgbiA9IGIudmFycy5sZW5ndGg7IGkgPCBuOyArK2kpIHtcclxuICAgICAgICAgICAgdmFyIHYgPSBiLnZhcnNbaV07XHJcbiAgICAgICAgICAgIHYub2Zmc2V0ICs9IGRpc3Q7XHJcbiAgICAgICAgICAgIHRoaXMuYWRkVmFyaWFibGUodik7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHRoaXMucG9zbiA9IHRoaXMucHMuZ2V0UG9zbigpO1xyXG4gICAgfTtcclxuICAgIEJsb2NrLnByb3RvdHlwZS5jb3N0ID0gZnVuY3Rpb24gKCkge1xyXG4gICAgICAgIHZhciBzdW0gPSAwLCBpID0gdGhpcy52YXJzLmxlbmd0aDtcclxuICAgICAgICB3aGlsZSAoaS0tKSB7XHJcbiAgICAgICAgICAgIHZhciB2ID0gdGhpcy52YXJzW2ldLCBkID0gdi5wb3NpdGlvbigpIC0gdi5kZXNpcmVkUG9zaXRpb247XHJcbiAgICAgICAgICAgIHN1bSArPSBkICogZCAqIHYud2VpZ2h0O1xyXG4gICAgICAgIH1cclxuICAgICAgICByZXR1cm4gc3VtO1xyXG4gICAgfTtcclxuICAgIHJldHVybiBCbG9jaztcclxufSgpKTtcclxuZXhwb3J0cy5CbG9jayA9IEJsb2NrO1xyXG52YXIgQmxvY2tzID0gKGZ1bmN0aW9uICgpIHtcclxuICAgIGZ1bmN0aW9uIEJsb2Nrcyh2cykge1xyXG4gICAgICAgIHRoaXMudnMgPSB2cztcclxuICAgICAgICB2YXIgbiA9IHZzLmxlbmd0aDtcclxuICAgICAgICB0aGlzLmxpc3QgPSBuZXcgQXJyYXkobik7XHJcbiAgICAgICAgd2hpbGUgKG4tLSkge1xyXG4gICAgICAgICAgICB2YXIgYiA9IG5ldyBCbG9jayh2c1tuXSk7XHJcbiAgICAgICAgICAgIHRoaXMubGlzdFtuXSA9IGI7XHJcbiAgICAgICAgICAgIGIuYmxvY2tJbmQgPSBuO1xyXG4gICAgICAgIH1cclxuICAgIH1cclxuICAgIEJsb2Nrcy5wcm90b3R5cGUuY29zdCA9IGZ1bmN0aW9uICgpIHtcclxuICAgICAgICB2YXIgc3VtID0gMCwgaSA9IHRoaXMubGlzdC5sZW5ndGg7XHJcbiAgICAgICAgd2hpbGUgKGktLSlcclxuICAgICAgICAgICAgc3VtICs9IHRoaXMubGlzdFtpXS5jb3N0KCk7XHJcbiAgICAgICAgcmV0dXJuIHN1bTtcclxuICAgIH07XHJcbiAgICBCbG9ja3MucHJvdG90eXBlLmluc2VydCA9IGZ1bmN0aW9uIChiKSB7XHJcbiAgICAgICAgYi5ibG9ja0luZCA9IHRoaXMubGlzdC5sZW5ndGg7XHJcbiAgICAgICAgdGhpcy5saXN0LnB1c2goYik7XHJcbiAgICB9O1xyXG4gICAgQmxvY2tzLnByb3RvdHlwZS5yZW1vdmUgPSBmdW5jdGlvbiAoYikge1xyXG4gICAgICAgIHZhciBsYXN0ID0gdGhpcy5saXN0Lmxlbmd0aCAtIDE7XHJcbiAgICAgICAgdmFyIHN3YXBCbG9jayA9IHRoaXMubGlzdFtsYXN0XTtcclxuICAgICAgICB0aGlzLmxpc3QubGVuZ3RoID0gbGFzdDtcclxuICAgICAgICBpZiAoYiAhPT0gc3dhcEJsb2NrKSB7XHJcbiAgICAgICAgICAgIHRoaXMubGlzdFtiLmJsb2NrSW5kXSA9IHN3YXBCbG9jaztcclxuICAgICAgICAgICAgc3dhcEJsb2NrLmJsb2NrSW5kID0gYi5ibG9ja0luZDtcclxuICAgICAgICB9XHJcbiAgICB9O1xyXG4gICAgQmxvY2tzLnByb3RvdHlwZS5tZXJnZSA9IGZ1bmN0aW9uIChjKSB7XHJcbiAgICAgICAgdmFyIGwgPSBjLmxlZnQuYmxvY2ssIHIgPSBjLnJpZ2h0LmJsb2NrO1xyXG4gICAgICAgIHZhciBkaXN0ID0gYy5yaWdodC5vZmZzZXQgLSBjLmxlZnQub2Zmc2V0IC0gYy5nYXA7XHJcbiAgICAgICAgaWYgKGwudmFycy5sZW5ndGggPCByLnZhcnMubGVuZ3RoKSB7XHJcbiAgICAgICAgICAgIHIubWVyZ2VBY3Jvc3MobCwgYywgZGlzdCk7XHJcbiAgICAgICAgICAgIHRoaXMucmVtb3ZlKGwpO1xyXG4gICAgICAgIH1cclxuICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgbC5tZXJnZUFjcm9zcyhyLCBjLCAtZGlzdCk7XHJcbiAgICAgICAgICAgIHRoaXMucmVtb3ZlKHIpO1xyXG4gICAgICAgIH1cclxuICAgIH07XHJcbiAgICBCbG9ja3MucHJvdG90eXBlLmZvckVhY2ggPSBmdW5jdGlvbiAoZikge1xyXG4gICAgICAgIHRoaXMubGlzdC5mb3JFYWNoKGYpO1xyXG4gICAgfTtcclxuICAgIEJsb2Nrcy5wcm90b3R5cGUudXBkYXRlQmxvY2tQb3NpdGlvbnMgPSBmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgdGhpcy5saXN0LmZvckVhY2goZnVuY3Rpb24gKGIpIHsgcmV0dXJuIGIudXBkYXRlV2VpZ2h0ZWRQb3NpdGlvbigpOyB9KTtcclxuICAgIH07XHJcbiAgICBCbG9ja3MucHJvdG90eXBlLnNwbGl0ID0gZnVuY3Rpb24gKGluYWN0aXZlKSB7XHJcbiAgICAgICAgdmFyIF90aGlzID0gdGhpcztcclxuICAgICAgICB0aGlzLnVwZGF0ZUJsb2NrUG9zaXRpb25zKCk7XHJcbiAgICAgICAgdGhpcy5saXN0LmZvckVhY2goZnVuY3Rpb24gKGIpIHtcclxuICAgICAgICAgICAgdmFyIHYgPSBiLmZpbmRNaW5MTSgpO1xyXG4gICAgICAgICAgICBpZiAodiAhPT0gbnVsbCAmJiB2LmxtIDwgU29sdmVyLkxBR1JBTkdJQU5fVE9MRVJBTkNFKSB7XHJcbiAgICAgICAgICAgICAgICBiID0gdi5sZWZ0LmJsb2NrO1xyXG4gICAgICAgICAgICAgICAgQmxvY2suc3BsaXQodikuZm9yRWFjaChmdW5jdGlvbiAobmIpIHsgcmV0dXJuIF90aGlzLmluc2VydChuYik7IH0pO1xyXG4gICAgICAgICAgICAgICAgX3RoaXMucmVtb3ZlKGIpO1xyXG4gICAgICAgICAgICAgICAgaW5hY3RpdmUucHVzaCh2KTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH0pO1xyXG4gICAgfTtcclxuICAgIHJldHVybiBCbG9ja3M7XHJcbn0oKSk7XHJcbmV4cG9ydHMuQmxvY2tzID0gQmxvY2tzO1xyXG52YXIgU29sdmVyID0gKGZ1bmN0aW9uICgpIHtcclxuICAgIGZ1bmN0aW9uIFNvbHZlcih2cywgY3MpIHtcclxuICAgICAgICB0aGlzLnZzID0gdnM7XHJcbiAgICAgICAgdGhpcy5jcyA9IGNzO1xyXG4gICAgICAgIHRoaXMudnMgPSB2cztcclxuICAgICAgICB2cy5mb3JFYWNoKGZ1bmN0aW9uICh2KSB7XHJcbiAgICAgICAgICAgIHYuY0luID0gW10sIHYuY091dCA9IFtdO1xyXG4gICAgICAgIH0pO1xyXG4gICAgICAgIHRoaXMuY3MgPSBjcztcclxuICAgICAgICBjcy5mb3JFYWNoKGZ1bmN0aW9uIChjKSB7XHJcbiAgICAgICAgICAgIGMubGVmdC5jT3V0LnB1c2goYyk7XHJcbiAgICAgICAgICAgIGMucmlnaHQuY0luLnB1c2goYyk7XHJcbiAgICAgICAgfSk7XHJcbiAgICAgICAgdGhpcy5pbmFjdGl2ZSA9IGNzLm1hcChmdW5jdGlvbiAoYykgeyBjLmFjdGl2ZSA9IGZhbHNlOyByZXR1cm4gYzsgfSk7XHJcbiAgICAgICAgdGhpcy5icyA9IG51bGw7XHJcbiAgICB9XHJcbiAgICBTb2x2ZXIucHJvdG90eXBlLmNvc3QgPSBmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgcmV0dXJuIHRoaXMuYnMuY29zdCgpO1xyXG4gICAgfTtcclxuICAgIFNvbHZlci5wcm90b3R5cGUuc2V0U3RhcnRpbmdQb3NpdGlvbnMgPSBmdW5jdGlvbiAocHMpIHtcclxuICAgICAgICB0aGlzLmluYWN0aXZlID0gdGhpcy5jcy5tYXAoZnVuY3Rpb24gKGMpIHsgYy5hY3RpdmUgPSBmYWxzZTsgcmV0dXJuIGM7IH0pO1xyXG4gICAgICAgIHRoaXMuYnMgPSBuZXcgQmxvY2tzKHRoaXMudnMpO1xyXG4gICAgICAgIHRoaXMuYnMuZm9yRWFjaChmdW5jdGlvbiAoYiwgaSkgeyByZXR1cm4gYi5wb3NuID0gcHNbaV07IH0pO1xyXG4gICAgfTtcclxuICAgIFNvbHZlci5wcm90b3R5cGUuc2V0RGVzaXJlZFBvc2l0aW9ucyA9IGZ1bmN0aW9uIChwcykge1xyXG4gICAgICAgIHRoaXMudnMuZm9yRWFjaChmdW5jdGlvbiAodiwgaSkgeyByZXR1cm4gdi5kZXNpcmVkUG9zaXRpb24gPSBwc1tpXTsgfSk7XHJcbiAgICB9O1xyXG4gICAgU29sdmVyLnByb3RvdHlwZS5tb3N0VmlvbGF0ZWQgPSBmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgdmFyIG1pblNsYWNrID0gTnVtYmVyLk1BWF9WQUxVRSwgdiA9IG51bGwsIGwgPSB0aGlzLmluYWN0aXZlLCBuID0gbC5sZW5ndGgsIGRlbGV0ZVBvaW50ID0gbjtcclxuICAgICAgICBmb3IgKHZhciBpID0gMDsgaSA8IG47ICsraSkge1xyXG4gICAgICAgICAgICB2YXIgYyA9IGxbaV07XHJcbiAgICAgICAgICAgIGlmIChjLnVuc2F0aXNmaWFibGUpXHJcbiAgICAgICAgICAgICAgICBjb250aW51ZTtcclxuICAgICAgICAgICAgdmFyIHNsYWNrID0gYy5zbGFjaygpO1xyXG4gICAgICAgICAgICBpZiAoYy5lcXVhbGl0eSB8fCBzbGFjayA8IG1pblNsYWNrKSB7XHJcbiAgICAgICAgICAgICAgICBtaW5TbGFjayA9IHNsYWNrO1xyXG4gICAgICAgICAgICAgICAgdiA9IGM7XHJcbiAgICAgICAgICAgICAgICBkZWxldGVQb2ludCA9IGk7XHJcbiAgICAgICAgICAgICAgICBpZiAoYy5lcXVhbGl0eSlcclxuICAgICAgICAgICAgICAgICAgICBicmVhaztcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH1cclxuICAgICAgICBpZiAoZGVsZXRlUG9pbnQgIT09IG4gJiZcclxuICAgICAgICAgICAgKG1pblNsYWNrIDwgU29sdmVyLlpFUk9fVVBQRVJCT1VORCAmJiAhdi5hY3RpdmUgfHwgdi5lcXVhbGl0eSkpIHtcclxuICAgICAgICAgICAgbFtkZWxldGVQb2ludF0gPSBsW24gLSAxXTtcclxuICAgICAgICAgICAgbC5sZW5ndGggPSBuIC0gMTtcclxuICAgICAgICB9XHJcbiAgICAgICAgcmV0dXJuIHY7XHJcbiAgICB9O1xyXG4gICAgU29sdmVyLnByb3RvdHlwZS5zYXRpc2Z5ID0gZnVuY3Rpb24gKCkge1xyXG4gICAgICAgIGlmICh0aGlzLmJzID09IG51bGwpIHtcclxuICAgICAgICAgICAgdGhpcy5icyA9IG5ldyBCbG9ja3ModGhpcy52cyk7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHRoaXMuYnMuc3BsaXQodGhpcy5pbmFjdGl2ZSk7XHJcbiAgICAgICAgdmFyIHYgPSBudWxsO1xyXG4gICAgICAgIHdoaWxlICgodiA9IHRoaXMubW9zdFZpb2xhdGVkKCkpICYmICh2LmVxdWFsaXR5IHx8IHYuc2xhY2soKSA8IFNvbHZlci5aRVJPX1VQUEVSQk9VTkQgJiYgIXYuYWN0aXZlKSkge1xyXG4gICAgICAgICAgICB2YXIgbGIgPSB2LmxlZnQuYmxvY2ssIHJiID0gdi5yaWdodC5ibG9jaztcclxuICAgICAgICAgICAgaWYgKGxiICE9PSByYikge1xyXG4gICAgICAgICAgICAgICAgdGhpcy5icy5tZXJnZSh2KTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgICAgIGlmIChsYi5pc0FjdGl2ZURpcmVjdGVkUGF0aEJldHdlZW4odi5yaWdodCwgdi5sZWZ0KSkge1xyXG4gICAgICAgICAgICAgICAgICAgIHYudW5zYXRpc2ZpYWJsZSA9IHRydWU7XHJcbiAgICAgICAgICAgICAgICAgICAgY29udGludWU7XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICB2YXIgc3BsaXQgPSBsYi5zcGxpdEJldHdlZW4odi5sZWZ0LCB2LnJpZ2h0KTtcclxuICAgICAgICAgICAgICAgIGlmIChzcGxpdCAhPT0gbnVsbCkge1xyXG4gICAgICAgICAgICAgICAgICAgIHRoaXMuYnMuaW5zZXJ0KHNwbGl0LmxiKTtcclxuICAgICAgICAgICAgICAgICAgICB0aGlzLmJzLmluc2VydChzcGxpdC5yYik7XHJcbiAgICAgICAgICAgICAgICAgICAgdGhpcy5icy5yZW1vdmUobGIpO1xyXG4gICAgICAgICAgICAgICAgICAgIHRoaXMuaW5hY3RpdmUucHVzaChzcGxpdC5jb25zdHJhaW50KTtcclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIGVsc2Uge1xyXG4gICAgICAgICAgICAgICAgICAgIHYudW5zYXRpc2ZpYWJsZSA9IHRydWU7XHJcbiAgICAgICAgICAgICAgICAgICAgY29udGludWU7XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICBpZiAodi5zbGFjaygpID49IDApIHtcclxuICAgICAgICAgICAgICAgICAgICB0aGlzLmluYWN0aXZlLnB1c2godik7XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgICAgICAgICB0aGlzLmJzLm1lcmdlKHYpO1xyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG4gICAgfTtcclxuICAgIFNvbHZlci5wcm90b3R5cGUuc29sdmUgPSBmdW5jdGlvbiAoKSB7XHJcbiAgICAgICAgdGhpcy5zYXRpc2Z5KCk7XHJcbiAgICAgICAgdmFyIGxhc3Rjb3N0ID0gTnVtYmVyLk1BWF9WQUxVRSwgY29zdCA9IHRoaXMuYnMuY29zdCgpO1xyXG4gICAgICAgIHdoaWxlIChNYXRoLmFicyhsYXN0Y29zdCAtIGNvc3QpID4gMC4wMDAxKSB7XHJcbiAgICAgICAgICAgIHRoaXMuc2F0aXNmeSgpO1xyXG4gICAgICAgICAgICBsYXN0Y29zdCA9IGNvc3Q7XHJcbiAgICAgICAgICAgIGNvc3QgPSB0aGlzLmJzLmNvc3QoKTtcclxuICAgICAgICB9XHJcbiAgICAgICAgcmV0dXJuIGNvc3Q7XHJcbiAgICB9O1xyXG4gICAgU29sdmVyLkxBR1JBTkdJQU5fVE9MRVJBTkNFID0gLTFlLTQ7XHJcbiAgICBTb2x2ZXIuWkVST19VUFBFUkJPVU5EID0gLTFlLTEwO1xyXG4gICAgcmV0dXJuIFNvbHZlcjtcclxufSgpKTtcclxuZXhwb3J0cy5Tb2x2ZXIgPSBTb2x2ZXI7XHJcbmZ1bmN0aW9uIHJlbW92ZU92ZXJsYXBJbk9uZURpbWVuc2lvbihzcGFucywgbG93ZXJCb3VuZCwgdXBwZXJCb3VuZCkge1xyXG4gICAgdmFyIHZzID0gc3BhbnMubWFwKGZ1bmN0aW9uIChzKSB7IHJldHVybiBuZXcgVmFyaWFibGUocy5kZXNpcmVkQ2VudGVyKTsgfSk7XHJcbiAgICB2YXIgY3MgPSBbXTtcclxuICAgIHZhciBuID0gc3BhbnMubGVuZ3RoO1xyXG4gICAgZm9yICh2YXIgaSA9IDA7IGkgPCBuIC0gMTsgaSsrKSB7XHJcbiAgICAgICAgdmFyIGxlZnQgPSBzcGFuc1tpXSwgcmlnaHQgPSBzcGFuc1tpICsgMV07XHJcbiAgICAgICAgY3MucHVzaChuZXcgQ29uc3RyYWludCh2c1tpXSwgdnNbaSArIDFdLCAobGVmdC5zaXplICsgcmlnaHQuc2l6ZSkgLyAyKSk7XHJcbiAgICB9XHJcbiAgICB2YXIgbGVmdE1vc3QgPSB2c1swXSwgcmlnaHRNb3N0ID0gdnNbbiAtIDFdLCBsZWZ0TW9zdFNpemUgPSBzcGFuc1swXS5zaXplIC8gMiwgcmlnaHRNb3N0U2l6ZSA9IHNwYW5zW24gLSAxXS5zaXplIC8gMjtcclxuICAgIHZhciB2TG93ZXIgPSBudWxsLCB2VXBwZXIgPSBudWxsO1xyXG4gICAgaWYgKGxvd2VyQm91bmQpIHtcclxuICAgICAgICB2TG93ZXIgPSBuZXcgVmFyaWFibGUobG93ZXJCb3VuZCwgbGVmdE1vc3Qud2VpZ2h0ICogMTAwMCk7XHJcbiAgICAgICAgdnMucHVzaCh2TG93ZXIpO1xyXG4gICAgICAgIGNzLnB1c2gobmV3IENvbnN0cmFpbnQodkxvd2VyLCBsZWZ0TW9zdCwgbGVmdE1vc3RTaXplKSk7XHJcbiAgICB9XHJcbiAgICBpZiAodXBwZXJCb3VuZCkge1xyXG4gICAgICAgIHZVcHBlciA9IG5ldyBWYXJpYWJsZSh1cHBlckJvdW5kLCByaWdodE1vc3Qud2VpZ2h0ICogMTAwMCk7XHJcbiAgICAgICAgdnMucHVzaCh2VXBwZXIpO1xyXG4gICAgICAgIGNzLnB1c2gobmV3IENvbnN0cmFpbnQocmlnaHRNb3N0LCB2VXBwZXIsIHJpZ2h0TW9zdFNpemUpKTtcclxuICAgIH1cclxuICAgIHZhciBzb2x2ZXIgPSBuZXcgU29sdmVyKHZzLCBjcyk7XHJcbiAgICBzb2x2ZXIuc29sdmUoKTtcclxuICAgIHJldHVybiB7XHJcbiAgICAgICAgbmV3Q2VudGVyczogdnMuc2xpY2UoMCwgc3BhbnMubGVuZ3RoKS5tYXAoZnVuY3Rpb24gKHYpIHsgcmV0dXJuIHYucG9zaXRpb24oKTsgfSksXHJcbiAgICAgICAgbG93ZXJCb3VuZDogdkxvd2VyID8gdkxvd2VyLnBvc2l0aW9uKCkgOiBsZWZ0TW9zdC5wb3NpdGlvbigpIC0gbGVmdE1vc3RTaXplLFxyXG4gICAgICAgIHVwcGVyQm91bmQ6IHZVcHBlciA/IHZVcHBlci5wb3NpdGlvbigpIDogcmlnaHRNb3N0LnBvc2l0aW9uKCkgKyByaWdodE1vc3RTaXplXHJcbiAgICB9O1xyXG59XHJcbmV4cG9ydHMucmVtb3ZlT3ZlcmxhcEluT25lRGltZW5zaW9uID0gcmVtb3ZlT3ZlcmxhcEluT25lRGltZW5zaW9uO1xyXG4vLyMgc291cmNlTWFwcGluZ1VSTD1kYXRhOmFwcGxpY2F0aW9uL2pzb247YmFzZTY0LGV5SjJaWEp6YVc5dUlqb3pMQ0ptYVd4bElqb2lkbkJ6WXk1cWN5SXNJbk52ZFhKalpWSnZiM1FpT2lJaUxDSnpiM1Z5WTJWeklqcGJJaTR1THk0dUwxZGxZa052YkdFdmMzSmpMM1p3YzJNdWRITWlYU3dpYm1GdFpYTWlPbHRkTENKdFlYQndhVzVuY3lJNklqczdRVUZCU1R0SlFVdEpMSFZDUVVGdFFpeExRVUZoTzFGQlFXSXNWVUZCU3l4SFFVRk1MRXRCUVVzc1EwRkJVVHRSUVVwb1F5eFBRVUZGTEVkQlFWY3NRMEZCUXl4RFFVRkRPMUZCUTJZc1QwRkJSU3hIUVVGWExFTkJRVU1zUTBGQlF6dFJRVU5tTEU5QlFVVXNSMEZCVnl4RFFVRkRMRU5CUVVNN1NVRkZiMElzUTBGQlF6dEpRVVZ3UXl4dFEwRkJWeXhIUVVGWUxGVkJRVmtzUTBGQlZ6dFJRVU51UWl4SlFVRkpMRVZCUVVVc1IwRkJSeXhKUVVGSkxFTkJRVU1zUzBGQlN5eEhRVUZITEVOQlFVTXNRMEZCUXl4TFFVRkxMRU5CUVVNN1VVRkRPVUlzU1VGQlNTeEZRVUZGTEVkQlFVY3NRMEZCUXl4RFFVRkRMRTFCUVUwc1IwRkJSeXhEUVVGRExFTkJRVU1zUzBGQlN5eERRVUZETzFGQlF6VkNMRWxCUVVrc1JVRkJSU3hIUVVGSExFTkJRVU1zUTBGQlF5eE5RVUZOTEVOQlFVTTdVVUZEYkVJc1NVRkJTU3hEUVVGRExFVkJRVVVzU1VGQlNTeEZRVUZGTEVkQlFVY3NSVUZCUlN4SFFVRkhMRVZCUVVVc1EwRkJRenRSUVVONFFpeEpRVUZKTEVOQlFVTXNSVUZCUlN4SlFVRkpMRVZCUVVVc1IwRkJSeXhGUVVGRkxFZEJRVWNzUTBGQlF5eERRVUZETEdWQlFXVXNRMEZCUXp0UlFVTjJReXhKUVVGSkxFTkJRVU1zUlVGQlJTeEpRVUZKTEVWQlFVVXNSMEZCUnl4RlFVRkZMRWRCUVVjc1JVRkJSU3hEUVVGRE8wbEJRelZDTEVOQlFVTTdTVUZGUkN3clFrRkJUeXhIUVVGUU8xRkJRMGtzVDBGQlR5eERRVUZETEVsQlFVa3NRMEZCUXl4RlFVRkZMRWRCUVVjc1NVRkJTU3hEUVVGRExFVkJRVVVzUTBGQlF5eEhRVUZITEVsQlFVa3NRMEZCUXl4RlFVRkZMRU5CUVVNN1NVRkRla01zUTBGQlF6dEpRVU5NTEc5Q1FVRkRPMEZCUVVRc1EwRkJReXhCUVc1Q1JDeEpRVzFDUXp0QlFXNUNXU3h6UTBGQllUdEJRWEZDTVVJN1NVRkxTU3h2UWtGQmJVSXNTVUZCWXl4RlFVRlRMRXRCUVdVc1JVRkJVeXhIUVVGWExFVkJRVk1zVVVGQmVVSTdVVUZCZWtJc2VVSkJRVUVzUlVGQlFTeG5Ra0ZCZVVJN1VVRkJOVVlzVTBGQlNTeEhRVUZLTEVsQlFVa3NRMEZCVlR0UlFVRlRMRlZCUVVzc1IwRkJUQ3hMUVVGTExFTkJRVlU3VVVGQlV5eFJRVUZITEVkQlFVZ3NSMEZCUnl4RFFVRlJPMUZCUVZNc1lVRkJVU3hIUVVGU0xGRkJRVkVzUTBGQmFVSTdVVUZJTDBjc1YwRkJUU3hIUVVGWkxFdEJRVXNzUTBGQlF6dFJRVU40UWl4clFrRkJZU3hIUVVGWkxFdEJRVXNzUTBGQlF6dFJRVWN6UWl4SlFVRkpMRU5CUVVNc1NVRkJTU3hIUVVGSExFbEJRVWtzUTBGQlF6dFJRVU5xUWl4SlFVRkpMRU5CUVVNc1MwRkJTeXhIUVVGSExFdEJRVXNzUTBGQlF6dFJRVU51UWl4SlFVRkpMRU5CUVVNc1IwRkJSeXhIUVVGSExFZEJRVWNzUTBGQlF6dFJRVU5tTEVsQlFVa3NRMEZCUXl4UlFVRlJMRWRCUVVjc1VVRkJVU3hEUVVGRE8wbEJRemRDTEVOQlFVTTdTVUZGUkN3d1FrRkJTeXhIUVVGTU8xRkJRMGtzVDBGQlR5eEpRVUZKTEVOQlFVTXNZVUZCWVN4RFFVRkRMRU5CUVVNc1EwRkJReXhOUVVGTkxFTkJRVU1zVTBGQlV6dFpRVU40UXl4RFFVRkRMRU5CUVVNc1NVRkJTU3hEUVVGRExFdEJRVXNzUTBGQlF5eExRVUZMTEVkQlFVY3NTVUZCU1N4RFFVRkRMRXRCUVVzc1EwRkJReXhSUVVGUkxFVkJRVVVzUjBGQlJ5eEpRVUZKTEVOQlFVTXNSMEZCUnp0clFrRkRia1FzU1VGQlNTeERRVUZETEVsQlFVa3NRMEZCUXl4TFFVRkxMRWRCUVVjc1NVRkJTU3hEUVVGRExFbEJRVWtzUTBGQlF5eFJRVUZSTEVWQlFVVXNRMEZCUXp0SlFVTnFSQ3hEUVVGRE8wbEJRMHdzYVVKQlFVTTdRVUZCUkN4RFFVRkRMRUZCYWtKRUxFbEJhVUpETzBGQmFrSlpMR2REUVVGVk8wRkJiVUoyUWp0SlFVMUpMR3RDUVVGdFFpeGxRVUYxUWl4RlFVRlRMRTFCUVd0Q0xFVkJRVk1zUzBGQmFVSTdVVUZCTlVNc2RVSkJRVUVzUlVGQlFTeFZRVUZyUWp0UlFVRlRMSE5DUVVGQkxFVkJRVUVzVTBGQmFVSTdVVUZCTlVVc2IwSkJRV1VzUjBGQlppeGxRVUZsTEVOQlFWRTdVVUZCVXl4WFFVRk5MRWRCUVU0c1RVRkJUU3hEUVVGWk8xRkJRVk1zVlVGQlN5eEhRVUZNTEV0QlFVc3NRMEZCV1R0UlFVd3ZSaXhYUVVGTkxFZEJRVmNzUTBGQlF5eERRVUZETzBsQlN5dEZMRU5CUVVNN1NVRkZia2NzZFVKQlFVa3NSMEZCU2p0UlFVTkpMRTlCUVU4c1IwRkJSeXhIUVVGSExFbEJRVWtzUTBGQlF5eE5RVUZOTEVkQlFVY3NRMEZCUXl4SlFVRkpMRU5CUVVNc1VVRkJVU3hGUVVGRkxFZEJRVWNzU1VGQlNTeERRVUZETEdWQlFXVXNRMEZCUXl4RFFVRkRPMGxCUTNoRkxFTkJRVU03U1VGRlJDd3lRa0ZCVVN4SFFVRlNPMUZCUTBrc1QwRkJUeXhEUVVGRExFbEJRVWtzUTBGQlF5eExRVUZMTEVOQlFVTXNSVUZCUlN4RFFVRkRMRXRCUVVzc1IwRkJSeXhKUVVGSkxFTkJRVU1zUzBGQlN5eERRVUZETEVsQlFVa3NSMEZCUnl4SlFVRkpMRU5CUVVNc1RVRkJUU3hEUVVGRExFZEJRVWNzU1VGQlNTeERRVUZETEV0QlFVc3NRMEZCUXp0SlFVTTVSU3hEUVVGRE8wbEJSMFFzYTBOQlFXVXNSMEZCWml4VlFVRm5RaXhKUVVGakxFVkJRVVVzUTBGQk1FTTdVVUZEZEVVc1NVRkJTU3hGUVVGRkxFZEJRVWNzVlVGQlF5eERRVUZETEVWQlFVVXNTVUZCU1N4SlFVRkxMRTlCUVVFc1EwRkJReXhEUVVGRExFMUJRVTBzU1VGQlNTeEpRVUZKTEV0QlFVc3NTVUZCU1N4SlFVRkpMRU5CUVVNc1EwRkJReXhEUVVGRExFVkJRVVVzU1VGQlNTeERRVUZETEVWQlFYWkRMRU5CUVhWRExFTkJRVU03VVVGRE9VUXNTVUZCU1N4RFFVRkRMRWxCUVVrc1EwRkJReXhQUVVGUExFTkJRVU1zVlVGQlFTeERRVUZETEVsQlFVY3NUMEZCUVN4RlFVRkZMRU5CUVVNc1EwRkJReXhGUVVGRkxFTkJRVU1zUTBGQlF5eExRVUZMTEVOQlFVTXNSVUZCWkN4RFFVRmpMRU5CUVVNc1EwRkJRenRSUVVOMFF5eEpRVUZKTEVOQlFVTXNSMEZCUnl4RFFVRkRMRTlCUVU4c1EwRkJReXhWUVVGQkxFTkJRVU1zU1VGQlJ5eFBRVUZCTEVWQlFVVXNRMEZCUXl4RFFVRkRMRVZCUVVVc1EwRkJReXhEUVVGRExFbEJRVWtzUTBGQlF5eEZRVUZpTEVOQlFXRXNRMEZCUXl4RFFVRkRPMGxCUTNoRExFTkJRVU03U1VGRFRDeGxRVUZETzBGQlFVUXNRMEZCUXl4QlFYUkNSQ3hKUVhOQ1F6dEJRWFJDV1N3MFFrRkJVVHRCUVhkQ2NrSTdTVUZOU1N4bFFVRlpMRU5CUVZjN1VVRk1ka0lzVTBGQlNTeEhRVUZsTEVWQlFVVXNRMEZCUXp0UlFVMXNRaXhEUVVGRExFTkJRVU1zVFVGQlRTeEhRVUZITEVOQlFVTXNRMEZCUXp0UlFVTmlMRWxCUVVrc1EwRkJReXhGUVVGRkxFZEJRVWNzU1VGQlNTeGhRVUZoTEVOQlFVTXNRMEZCUXl4RFFVRkRMRXRCUVVzc1EwRkJReXhEUVVGRE8xRkJRM0pETEVsQlFVa3NRMEZCUXl4WFFVRlhMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU03U1VGRGVFSXNRMEZCUXp0SlFVVlBMREpDUVVGWExFZEJRVzVDTEZWQlFXOUNMRU5CUVZjN1VVRkRNMElzUTBGQlF5eERRVUZETEV0QlFVc3NSMEZCUnl4SlFVRkpMRU5CUVVNN1VVRkRaaXhKUVVGSkxFTkJRVU1zU1VGQlNTeERRVUZETEVsQlFVa3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJRenRSUVVOc1FpeEpRVUZKTEVOQlFVTXNSVUZCUlN4RFFVRkRMRmRCUVZjc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dFJRVU4yUWl4SlFVRkpMRU5CUVVNc1NVRkJTU3hIUVVGSExFbEJRVWtzUTBGQlF5eEZRVUZGTEVOQlFVTXNUMEZCVHl4RlFVRkZMRU5CUVVNN1NVRkRiRU1zUTBGQlF6dEpRVWRFTEhORFFVRnpRaXhIUVVGMFFqdFJRVU5KTEVsQlFVa3NRMEZCUXl4RlFVRkZMRU5CUVVNc1JVRkJSU3hIUVVGSExFbEJRVWtzUTBGQlF5eEZRVUZGTEVOQlFVTXNSVUZCUlN4SFFVRkhMRWxCUVVrc1EwRkJReXhGUVVGRkxFTkJRVU1zUlVGQlJTeEhRVUZITEVOQlFVTXNRMEZCUXp0UlFVTjZReXhMUVVGTExFbEJRVWtzUTBGQlF5eEhRVUZITEVOQlFVTXNSVUZCUlN4RFFVRkRMRWRCUVVjc1NVRkJTU3hEUVVGRExFbEJRVWtzUTBGQlF5eE5RVUZOTEVWQlFVVXNRMEZCUXl4SFFVRkhMRU5CUVVNc1JVRkJSU3hGUVVGRkxFTkJRVU03V1VGRE5VTXNTVUZCU1N4RFFVRkRMRVZCUVVVc1EwRkJReXhYUVVGWExFTkJRVU1zU1VGQlNTeERRVUZETEVsQlFVa3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRE8xRkJRM1JETEVsQlFVa3NRMEZCUXl4SlFVRkpMRWRCUVVjc1NVRkJTU3hEUVVGRExFVkJRVVVzUTBGQlF5eFBRVUZQTEVWQlFVVXNRMEZCUXp0SlFVTnNReXhEUVVGRE8wbEJSVThzTUVKQlFWVXNSMEZCYkVJc1ZVRkJiVUlzUTBGQlZ5eEZRVUZGTEVOQlFWY3NSVUZCUlN4VlFVRnBRenRSUVVFNVJTeHBRa0ZqUXp0UlFXSkhMRWxCUVVrc1NVRkJTU3hIUVVGSExFTkJRVU1zUTBGQlF5eEpRVUZKTEVWQlFVVXNRMEZCUXp0UlFVTndRaXhEUVVGRExFTkJRVU1zWlVGQlpTeERRVUZETEVOQlFVTXNSVUZCUlN4VlFVRkRMRU5CUVVNc1JVRkJSU3hKUVVGSk8xbEJRM3BDTEVsQlFVa3NTMEZCU3l4SFFVRkhMRXRCUVVrc1EwRkJReXhWUVVGVkxFTkJRVU1zU1VGQlNTeEZRVUZGTEVOQlFVTXNSVUZCUlN4VlFVRlZMRU5CUVVNc1EwRkJRenRaUVVOcVJDeEpRVUZKTEVsQlFVa3NTMEZCU3l4RFFVRkRMRU5CUVVNc1MwRkJTeXhGUVVGRk8yZENRVU5zUWl4SlFVRkpMRWxCUVVrc1MwRkJTeXhIUVVGSExFTkJRVU1zUTBGQlF5eEpRVUZKTEVOQlFVTXNTMEZCU3l4RFFVRkRPMmRDUVVNM1FpeERRVUZETEVOQlFVTXNSVUZCUlN4SFFVRkhMRXRCUVVzc1EwRkJRenRoUVVOb1FqdHBRa0ZCVFR0blFrRkRTQ3hKUVVGSkxFbEJRVWtzUzBGQlN5eEhRVUZITEVOQlFVTXNRMEZCUXl4TFFVRkxMRU5CUVVNc1MwRkJTeXhEUVVGRE8yZENRVU01UWl4RFFVRkRMRU5CUVVNc1JVRkJSU3hIUVVGSExFTkJRVU1zUzBGQlN5eERRVUZETzJGQlEycENPMWxCUTBRc1ZVRkJWU3hEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETzFGQlEyeENMRU5CUVVNc1EwRkJReXhEUVVGRE8xRkJRMGdzVDBGQlR5eEpRVUZKTEVkQlFVY3NRMEZCUXl4RFFVRkRMRXRCUVVzc1EwRkJRenRKUVVNeFFpeERRVUZETzBsQlJVOHNhME5CUVd0Q0xFZEJRVEZDTEZWQlFUSkNMRU5CUVZjc1JVRkJSU3hKUVVGak8xRkJRWFJFTEdsQ1FVMURPMUZCVEVjc1EwRkJReXhEUVVGRExHVkJRV1VzUTBGQlF5eEpRVUZKTEVWQlFVVXNWVUZCUXl4RFFVRkRMRVZCUVVVc1NVRkJTVHRaUVVNMVFpeEpRVUZKTEVOQlFVTXNUVUZCVFN4SFFVRkhMRU5CUVVNc1EwRkJReXhOUVVGTkxFZEJRVWNzUTBGQlF5eEpRVUZKTEV0QlFVc3NRMEZCUXl4RFFVRkRMRXRCUVVzc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVkQlFVY3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUjBGQlJ5eERRVUZETEVOQlFVTTdXVUZETjBRc1MwRkJTU3hEUVVGRExGZEJRVmNzUTBGQlF5eEpRVUZKTEVOQlFVTXNRMEZCUXp0WlFVTjJRaXhMUVVGSkxFTkJRVU1zYTBKQlFXdENMRU5CUVVNc1NVRkJTU3hGUVVGRkxFTkJRVU1zUTBGQlF5eERRVUZETzFGQlEzSkRMRU5CUVVNc1EwRkJReXhEUVVGRE8wbEJRMUFzUTBGQlF6dEpRVWRFTEhkQ1FVRlJMRWRCUVZJc1ZVRkJVeXhMUVVFMlFpeEZRVUZGTEVkQlFWVXNSVUZCUlN4RFFVRXdRaXhGUVVGRkxFbEJRVzFDTzFGQlFXNUhMR2xDUVV0RE8xRkJURzFFTEd0Q1FVRkJMRVZCUVVFc1NVRkJZeXhKUVVGSkxFTkJRVU1zU1VGQlNTeERRVUZETEVOQlFVTXNRMEZCUXp0UlFVRkZMSEZDUVVGQkxFVkJRVUVzVjBGQmJVSTdVVUZETDBZc1EwRkJReXhEUVVGRExHVkJRV1VzUTBGQlF5eEpRVUZKTEVWQlFVVXNWVUZCUXl4RFFVRkRMRVZCUVVVc1NVRkJTVHRaUVVNMVFpeEhRVUZITEVOQlFVTXNTVUZCU1N4RFFVRkRMRXRCUVVzc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETzFsQlEyNUNMRXRCUVVrc1EwRkJReXhSUVVGUkxFTkJRVU1zUzBGQlN5eEZRVUZGTEVkQlFVY3NSVUZCUlN4SlFVRkpMRVZCUVVVc1EwRkJReXhEUVVGRExFTkJRVU03VVVGRGRrTXNRMEZCUXl4RFFVRkRMRU5CUVVNN1NVRkRVQ3hEUVVGRE8wbEJTMFFzZVVKQlFWTXNSMEZCVkR0UlFVTkpMRWxCUVVrc1EwRkJReXhIUVVGbExFbEJRVWtzUTBGQlF6dFJRVU42UWl4SlFVRkpMRU5CUVVNc1ZVRkJWU3hEUVVGRExFbEJRVWtzUTBGQlF5eEpRVUZKTEVOQlFVTXNRMEZCUXl4RFFVRkRMRVZCUVVVc1NVRkJTU3hGUVVGRkxGVkJRVUVzUTBGQlF6dFpRVU5xUXl4SlFVRkpMRU5CUVVNc1EwRkJReXhEUVVGRExGRkJRVkVzU1VGQlNTeERRVUZETEVOQlFVTXNTMEZCU3l4SlFVRkpMRWxCUVVrc1EwRkJReXhEUVVGRExFVkJRVVVzUjBGQlJ5eERRVUZETEVOQlFVTXNSVUZCUlN4RFFVRkRPMmRDUVVGRkxFTkJRVU1zUjBGQlJ5eERRVUZETEVOQlFVTTdVVUZETVVRc1EwRkJReXhEUVVGRExFTkJRVU03VVVGRFNDeFBRVUZQTEVOQlFVTXNRMEZCUXp0SlFVTmlMRU5CUVVNN1NVRkZUeXhuUTBGQlowSXNSMEZCZUVJc1ZVRkJlVUlzUlVGQldTeEZRVUZGTEVWQlFWazdVVUZETDBNc1NVRkJTU3hEUVVGRExGVkJRVlVzUTBGQlF5eEZRVUZGTEVWQlFVVXNTVUZCU1N4RlFVRkZMR05CUVU4c1EwRkJReXhEUVVGRExFTkJRVU03VVVGRGNFTXNTVUZCU1N4RFFVRkRMRWRCUVVjc1NVRkJTU3hEUVVGRE8xRkJRMklzU1VGQlNTeERRVUZETEZGQlFWRXNRMEZCUXl4RlFVRkZMRVZCUVVVc1NVRkJTU3hGUVVGRkxFVkJRVVVzUlVGQlJTeFZRVUZETEVOQlFVTXNSVUZCUlN4SlFVRkpPMWxCUTJoRExFbEJRVWtzUTBGQlF5eERRVUZETEVOQlFVTXNVVUZCVVN4SlFVRkpMRU5CUVVNc1EwRkJReXhMUVVGTExFdEJRVXNzU1VGQlNTeEpRVUZKTEVOQlFVTXNRMEZCUXl4TFFVRkxMRWxCUVVrc1NVRkJTU3hEUVVGRExFTkJRVU1zUlVGQlJTeEhRVUZITEVOQlFVTXNRMEZCUXl4RlFVRkZMRU5CUVVNN1owSkJRVVVzUTBGQlF5eEhRVUZITEVOQlFVTXNRMEZCUXp0UlFVTTVSU3hEUVVGRExFTkJRVU1zUTBGQlF6dFJRVU5JTEU5QlFVOHNRMEZCUXl4RFFVRkRPMGxCUTJJc1EwRkJRenRKUVVWUExIZENRVUZSTEVkQlFXaENMRlZCUVdsQ0xFTkJRVmNzUlVGQlJTeEpRVUZqTEVWQlFVVXNSVUZCV1N4RlFVRkZMRXRCUVRKRE8xRkJRWFpITEdsQ1FWVkRPMUZCVkVjc1NVRkJTU3hSUVVGUkxFZEJRVWNzUzBGQlN5eERRVUZETzFGQlEzSkNMRU5CUVVNc1EwRkJReXhsUVVGbExFTkJRVU1zU1VGQlNTeEZRVUZGTEZWQlFVTXNRMEZCUXl4RlFVRkZMRWxCUVVrN1dVRkROVUlzU1VGQlNTeERRVUZETEZGQlFWRXNTVUZCU1N4RFFVRkRMRWxCUVVrc1MwRkJTeXhGUVVGRkxFbEJRVWtzUzBGQlNTeERRVUZETEZGQlFWRXNRMEZCUXl4SlFVRkpMRVZCUVVVc1EwRkJReXhGUVVGRkxFVkJRVVVzUlVGQlJTeExRVUZMTEVOQlFVTXNRMEZCUXl4RlFVTnVSVHRuUWtGRFNTeFJRVUZSTEVkQlFVY3NTVUZCU1N4RFFVRkRPMmRDUVVOb1FpeExRVUZMTEVOQlFVTXNRMEZCUXl4RlFVRkZMRWxCUVVrc1EwRkJReXhEUVVGRE8yRkJRMnhDTzFGQlEwd3NRMEZCUXl4RFFVRkRMRU5CUVVNN1VVRkRTQ3hQUVVGUExGRkJRVkVzUTBGQlF6dEpRVU53UWl4RFFVRkRPMGxCU1VRc01rTkJRVEpDTEVkQlFUTkNMRlZCUVRSQ0xFTkJRVmNzUlVGQlJTeERRVUZYTzFGQlEyaEVMRWxCUVVrc1EwRkJReXhMUVVGTExFTkJRVU03V1VGQlJTeFBRVUZQTEVsQlFVa3NRMEZCUXp0UlFVTjZRaXhKUVVGSkxFTkJRVU1zUjBGQlJ5eERRVUZETEVOQlFVTXNTVUZCU1N4RFFVRkRMRTFCUVUwc1EwRkJRenRSUVVOMFFpeFBRVUZOTEVOQlFVTXNSVUZCUlN4RlFVRkZPMWxCUTFBc1NVRkJTU3hEUVVGRExFZEJRVWNzUTBGQlF5eERRVUZETEVsQlFVa3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJRenRaUVVOc1FpeEpRVUZKTEVOQlFVTXNRMEZCUXl4TlFVRk5MRWxCUVVrc1NVRkJTU3hEUVVGRExESkNRVUV5UWl4RFFVRkRMRU5CUVVNc1EwRkJReXhMUVVGTExFVkJRVVVzUTBGQlF5eERRVUZETzJkQ1FVTjRSQ3hQUVVGUExFbEJRVWtzUTBGQlF6dFRRVU51UWp0UlFVTkVMRTlCUVU4c1MwRkJTeXhEUVVGRE8wbEJRMnBDTEVOQlFVTTdTVUZIVFN4WFFVRkxMRWRCUVZvc1ZVRkJZU3hEUVVGaE8xRkJTM1JDTEVOQlFVTXNRMEZCUXl4TlFVRk5MRWRCUVVjc1MwRkJTeXhEUVVGRE8xRkJRMnBDTEU5QlFVOHNRMEZCUXl4TFFVRkxMRU5CUVVNc1owSkJRV2RDTEVOQlFVTXNRMEZCUXl4RFFVRkRMRWxCUVVrc1EwRkJReXhGUVVGRkxFdEJRVXNzUTBGQlF5eG5Ra0ZCWjBJc1EwRkJReXhEUVVGRExFTkJRVU1zUzBGQlN5eERRVUZETEVOQlFVTXNRMEZCUXp0SlFVTTNSU3hEUVVGRE8wbEJSV01zYzBKQlFXZENMRWRCUVM5Q0xGVkJRV2RETEZGQlFXdENPMUZCUXpsRExFbEJRVWtzUTBGQlF5eEhRVUZITEVsQlFVa3NTMEZCU3l4RFFVRkRMRkZCUVZFc1EwRkJReXhEUVVGRE8xRkJRelZDTEVOQlFVTXNRMEZCUXl4clFrRkJhMElzUTBGQlF5eFJRVUZSTEVWQlFVVXNTVUZCU1N4RFFVRkRMRU5CUVVNN1VVRkRja01zVDBGQlR5eERRVUZETEVOQlFVTTdTVUZEWWl4RFFVRkRPMGxCUjBRc05FSkJRVmtzUjBGQldpeFZRVUZoTEVWQlFWa3NSVUZCUlN4RlFVRlpPMUZCUzI1RExFbEJRVWtzUTBGQlF5eEhRVUZITEVsQlFVa3NRMEZCUXl4blFrRkJaMElzUTBGQlF5eEZRVUZGTEVWQlFVVXNSVUZCUlN4RFFVRkRMRU5CUVVNN1VVRkRkRU1zU1VGQlNTeERRVUZETEV0QlFVc3NTVUZCU1N4RlFVRkZPMWxCUTFvc1NVRkJTU3hGUVVGRkxFZEJRVWNzUzBGQlN5eERRVUZETEV0QlFVc3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJRenRaUVVONFFpeFBRVUZQTEVWQlFVVXNWVUZCVlN4RlFVRkZMRU5CUVVNc1JVRkJSU3hGUVVGRkxFVkJRVVVzUlVGQlJTeERRVUZETEVOQlFVTXNRMEZCUXl4RlFVRkZMRVZCUVVVc1JVRkJSU3hGUVVGRkxFTkJRVU1zUTBGQlF5eERRVUZETEVWQlFVVXNRMEZCUXp0VFFVTnNSRHRSUVVWRUxFOUJRVThzU1VGQlNTeERRVUZETzBsQlEyaENMRU5CUVVNN1NVRkZSQ3d5UWtGQlZ5eEhRVUZZTEZWQlFWa3NRMEZCVVN4RlFVRkZMRU5CUVdFc1JVRkJSU3hKUVVGWk8xRkJRemRETEVOQlFVTXNRMEZCUXl4TlFVRk5MRWRCUVVjc1NVRkJTU3hEUVVGRE8xRkJRMmhDTEV0QlFVc3NTVUZCU1N4RFFVRkRMRWRCUVVjc1EwRkJReXhGUVVGRkxFTkJRVU1zUjBGQlJ5eERRVUZETEVOQlFVTXNTVUZCU1N4RFFVRkRMRTFCUVUwc1JVRkJSU3hEUVVGRExFZEJRVWNzUTBGQlF5eEZRVUZGTEVWQlFVVXNRMEZCUXl4RlFVRkZPMWxCUXpORExFbEJRVWtzUTBGQlF5eEhRVUZITEVOQlFVTXNRMEZCUXl4SlFVRkpMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU03V1VGRGJFSXNRMEZCUXl4RFFVRkRMRTFCUVUwc1NVRkJTU3hKUVVGSkxFTkJRVU03V1VGRGFrSXNTVUZCU1N4RFFVRkRMRmRCUVZjc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dFRRVU4yUWp0UlFVTkVMRWxCUVVrc1EwRkJReXhKUVVGSkxFZEJRVWNzU1VGQlNTeERRVUZETEVWQlFVVXNRMEZCUXl4UFFVRlBMRVZCUVVVc1EwRkJRenRKUVVOc1F5eERRVUZETzBsQlJVUXNiMEpCUVVrc1IwRkJTanRSUVVOSkxFbEJRVWtzUjBGQlJ5eEhRVUZITEVOQlFVTXNSVUZCUlN4RFFVRkRMRWRCUVVjc1NVRkJTU3hEUVVGRExFbEJRVWtzUTBGQlF5eE5RVUZOTEVOQlFVTTdVVUZEYkVNc1QwRkJUeXhEUVVGRExFVkJRVVVzUlVGQlJUdFpRVU5TTEVsQlFVa3NRMEZCUXl4SFFVRkhMRWxCUVVrc1EwRkJReXhKUVVGSkxFTkJRVU1zUTBGQlF5eERRVUZETEVWQlEyaENMRU5CUVVNc1IwRkJSeXhEUVVGRExFTkJRVU1zVVVGQlVTeEZRVUZGTEVkQlFVY3NRMEZCUXl4RFFVRkRMR1ZCUVdVc1EwRkJRenRaUVVONlF5eEhRVUZITEVsQlFVa3NRMEZCUXl4SFFVRkhMRU5CUVVNc1IwRkJSeXhEUVVGRExFTkJRVU1zVFVGQlRTeERRVUZETzFOQlF6TkNPMUZCUTBRc1QwRkJUeXhIUVVGSExFTkJRVU03U1VGRFppeERRVUZETzBsQlUwd3NXVUZCUXp0QlFVRkVMRU5CUVVNc1FVRnNTMFFzU1VGclMwTTdRVUZzUzFrc2MwSkJRVXM3UVVGdlMyeENPMGxCUjBrc1owSkJRVzFDTEVWQlFXTTdVVUZCWkN4UFFVRkZMRWRCUVVZc1JVRkJSU3hEUVVGWk8xRkJRemRDTEVsQlFVa3NRMEZCUXl4SFFVRkhMRVZCUVVVc1EwRkJReXhOUVVGTkxFTkJRVU03VVVGRGJFSXNTVUZCU1N4RFFVRkRMRWxCUVVrc1IwRkJSeXhKUVVGSkxFdEJRVXNzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXp0UlFVTjZRaXhQUVVGUExFTkJRVU1zUlVGQlJTeEZRVUZGTzFsQlExSXNTVUZCU1N4RFFVRkRMRWRCUVVjc1NVRkJTU3hMUVVGTExFTkJRVU1zUlVGQlJTeERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRMRU5CUVVNN1dVRkRla0lzU1VGQlNTeERRVUZETEVsQlFVa3NRMEZCUXl4RFFVRkRMRU5CUVVNc1IwRkJSeXhEUVVGRExFTkJRVU03V1VGRGFrSXNRMEZCUXl4RFFVRkRMRkZCUVZFc1IwRkJSeXhEUVVGRExFTkJRVU03VTBGRGJFSTdTVUZEVEN4RFFVRkRPMGxCUlVRc2NVSkJRVWtzUjBGQlNqdFJRVU5KTEVsQlFVa3NSMEZCUnl4SFFVRkhMRU5CUVVNc1JVRkJSU3hEUVVGRExFZEJRVWNzU1VGQlNTeERRVUZETEVsQlFVa3NRMEZCUXl4TlFVRk5MRU5CUVVNN1VVRkRiRU1zVDBGQlR5eERRVUZETEVWQlFVVTdXVUZCUlN4SFFVRkhMRWxCUVVrc1NVRkJTU3hEUVVGRExFbEJRVWtzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4SlFVRkpMRVZCUVVVc1EwRkJRenRSUVVOMlF5eFBRVUZQTEVkQlFVY3NRMEZCUXp0SlFVTm1MRU5CUVVNN1NVRkZSQ3gxUWtGQlRTeEhRVUZPTEZWQlFVOHNRMEZCVVR0UlFVbFlMRU5CUVVNc1EwRkJReXhSUVVGUkxFZEJRVWNzU1VGQlNTeERRVUZETEVsQlFVa3NRMEZCUXl4TlFVRk5MRU5CUVVNN1VVRkRPVUlzU1VGQlNTeERRVUZETEVsQlFVa3NRMEZCUXl4SlFVRkpMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU03U1VGTGRFSXNRMEZCUXp0SlFVVkVMSFZDUVVGTkxFZEJRVTRzVlVGQlR5eERRVUZSTzFGQlMxZ3NTVUZCU1N4SlFVRkpMRWRCUVVjc1NVRkJTU3hEUVVGRExFbEJRVWtzUTBGQlF5eE5RVUZOTEVkQlFVY3NRMEZCUXl4RFFVRkRPMUZCUTJoRExFbEJRVWtzVTBGQlV5eEhRVUZITEVsQlFVa3NRMEZCUXl4SlFVRkpMRU5CUVVNc1NVRkJTU3hEUVVGRExFTkJRVU03VVVGRGFFTXNTVUZCU1N4RFFVRkRMRWxCUVVrc1EwRkJReXhOUVVGTkxFZEJRVWNzU1VGQlNTeERRVUZETzFGQlEzaENMRWxCUVVrc1EwRkJReXhMUVVGTExGTkJRVk1zUlVGQlJUdFpRVU5xUWl4SlFVRkpMRU5CUVVNc1NVRkJTU3hEUVVGRExFTkJRVU1zUTBGQlF5eFJRVUZSTEVOQlFVTXNSMEZCUnl4VFFVRlRMRU5CUVVNN1dVRkRiRU1zVTBGQlV5eERRVUZETEZGQlFWRXNSMEZCUnl4RFFVRkRMRU5CUVVNc1VVRkJVU3hEUVVGRE8xTkJTVzVETzBsQlEwd3NRMEZCUXp0SlFVbEVMSE5DUVVGTExFZEJRVXdzVlVGQlRTeERRVUZoTzFGQlEyWXNTVUZCU1N4RFFVRkRMRWRCUVVjc1EwRkJReXhEUVVGRExFbEJRVWtzUTBGQlF5eExRVUZMTEVWQlFVVXNRMEZCUXl4SFFVRkhMRU5CUVVNc1EwRkJReXhMUVVGTExFTkJRVU1zUzBGQlN5eERRVUZETzFGQlNYaERMRWxCUVVrc1NVRkJTU3hIUVVGSExFTkJRVU1zUTBGQlF5eExRVUZMTEVOQlFVTXNUVUZCVFN4SFFVRkhMRU5CUVVNc1EwRkJReXhKUVVGSkxFTkJRVU1zVFVGQlRTeEhRVUZITEVOQlFVTXNRMEZCUXl4SFFVRkhMRU5CUVVNN1VVRkRiRVFzU1VGQlNTeERRVUZETEVOQlFVTXNTVUZCU1N4RFFVRkRMRTFCUVUwc1IwRkJSeXhEUVVGRExFTkJRVU1zU1VGQlNTeERRVUZETEUxQlFVMHNSVUZCUlR0WlFVTXZRaXhEUVVGRExFTkJRVU1zVjBGQlZ5eERRVUZETEVOQlFVTXNSVUZCUlN4RFFVRkRMRVZCUVVVc1NVRkJTU3hEUVVGRExFTkJRVU03V1VGRE1VSXNTVUZCU1N4RFFVRkRMRTFCUVUwc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dFRRVU5zUWp0aFFVRk5PMWxCUTBnc1EwRkJReXhEUVVGRExGZEJRVmNzUTBGQlF5eERRVUZETEVWQlFVVXNRMEZCUXl4RlFVRkZMRU5CUVVNc1NVRkJTU3hEUVVGRExFTkJRVU03V1VGRE0wSXNTVUZCU1N4RFFVRkRMRTFCUVUwc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dFRRVU5zUWp0SlFVdE1MRU5CUVVNN1NVRkZSQ3gzUWtGQlR5eEhRVUZRTEZWQlFWRXNRMEZCWjBNN1VVRkRjRU1zU1VGQlNTeERRVUZETEVsQlFVa3NRMEZCUXl4UFFVRlBMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU03U1VGRGVrSXNRMEZCUXp0SlFVZEVMSEZEUVVGdlFpeEhRVUZ3UWp0UlFVTkpMRWxCUVVrc1EwRkJReXhKUVVGSkxFTkJRVU1zVDBGQlR5eERRVUZETEZWQlFVRXNRMEZCUXl4SlFVRkhMRTlCUVVFc1EwRkJReXhEUVVGRExITkNRVUZ6UWl4RlFVRkZMRVZCUVRGQ0xFTkJRVEJDTEVOQlFVTXNRMEZCUXp0SlFVTjBSQ3hEUVVGRE8wbEJSMFFzYzBKQlFVc3NSMEZCVEN4VlFVRk5MRkZCUVhOQ08xRkJRVFZDTEdsQ1FXVkRPMUZCWkVjc1NVRkJTU3hEUVVGRExHOUNRVUZ2UWl4RlFVRkZMRU5CUVVNN1VVRkROVUlzU1VGQlNTeERRVUZETEVsQlFVa3NRMEZCUXl4UFFVRlBMRU5CUVVNc1ZVRkJRU3hEUVVGRE8xbEJRMllzU1VGQlNTeERRVUZETEVkQlFVY3NRMEZCUXl4RFFVRkRMRk5CUVZNc1JVRkJSU3hEUVVGRE8xbEJRM1JDTEVsQlFVa3NRMEZCUXl4TFFVRkxMRWxCUVVrc1NVRkJTU3hEUVVGRExFTkJRVU1zUlVGQlJTeEhRVUZITEUxQlFVMHNRMEZCUXl4dlFrRkJiMElzUlVGQlJUdG5Ra0ZEYkVRc1EwRkJReXhIUVVGSExFTkJRVU1zUTBGQlF5eEpRVUZKTEVOQlFVTXNTMEZCU3l4RFFVRkRPMmRDUVVOcVFpeExRVUZMTEVOQlFVTXNTMEZCU3l4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFOUJRVThzUTBGQlF5eFZRVUZCTEVWQlFVVXNTVUZCUlN4UFFVRkJMRXRCUVVrc1EwRkJReXhOUVVGTkxFTkJRVU1zUlVGQlJTeERRVUZETEVWQlFXWXNRMEZCWlN4RFFVRkRMRU5CUVVNN1owSkJRelZETEV0QlFVa3NRMEZCUXl4TlFVRk5MRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU03WjBKQlEyWXNVVUZCVVN4RFFVRkRMRWxCUVVrc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dGhRVXR3UWp0UlFVTk1MRU5CUVVNc1EwRkJReXhEUVVGRE8wbEJRMUFzUTBGQlF6dEpRVzlDVEN4aFFVRkRPMEZCUVVRc1EwRkJReXhCUVd4SVJDeEpRV3RJUXp0QlFXeElXU3gzUWtGQlRUdEJRVzlJYmtJN1NVRlBTU3huUWtGQmJVSXNSVUZCWXl4RlFVRlRMRVZCUVdkQ08xRkJRWFpETEU5QlFVVXNSMEZCUml4RlFVRkZMRU5CUVZrN1VVRkJVeXhQUVVGRkxFZEJRVVlzUlVGQlJTeERRVUZqTzFGQlEzUkVMRWxCUVVrc1EwRkJReXhGUVVGRkxFZEJRVWNzUlVGQlJTeERRVUZETzFGQlEySXNSVUZCUlN4RFFVRkRMRTlCUVU4c1EwRkJReXhWUVVGQkxFTkJRVU03V1VGRFVpeERRVUZETEVOQlFVTXNSMEZCUnl4SFFVRkhMRVZCUVVVc1JVRkJSU3hEUVVGRExFTkJRVU1zU1VGQlNTeEhRVUZITEVWQlFVVXNRMEZCUXp0UlFVazFRaXhEUVVGRExFTkJRVU1zUTBGQlF6dFJRVU5JTEVsQlFVa3NRMEZCUXl4RlFVRkZMRWRCUVVjc1JVRkJSU3hEUVVGRE8xRkJRMklzUlVGQlJTeERRVUZETEU5QlFVOHNRMEZCUXl4VlFVRkJMRU5CUVVNN1dVRkRVaXhEUVVGRExFTkJRVU1zU1VGQlNTeERRVUZETEVsQlFVa3NRMEZCUXl4SlFVRkpMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU03V1VGRGNFSXNRMEZCUXl4RFFVRkRMRXRCUVVzc1EwRkJReXhIUVVGSExFTkJRVU1zU1VGQlNTeERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRPMUZCU1hoQ0xFTkJRVU1zUTBGQlF5eERRVUZETzFGQlEwZ3NTVUZCU1N4RFFVRkRMRkZCUVZFc1IwRkJSeXhGUVVGRkxFTkJRVU1zUjBGQlJ5eERRVUZETEZWQlFVRXNRMEZCUXl4SlFVRkxMRU5CUVVNc1EwRkJReXhOUVVGTkxFZEJRVWNzUzBGQlN5eERRVUZETEVOQlFVTXNUMEZCVHl4RFFVRkRMRU5CUVVNc1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF6dFJRVU0xUkN4SlFVRkpMRU5CUVVNc1JVRkJSU3hIUVVGSExFbEJRVWtzUTBGQlF6dEpRVU51UWl4RFFVRkRPMGxCUlVRc2NVSkJRVWtzUjBGQlNqdFJRVU5KTEU5QlFVOHNTVUZCU1N4RFFVRkRMRVZCUVVVc1EwRkJReXhKUVVGSkxFVkJRVVVzUTBGQlF6dEpRVU14UWl4RFFVRkRPMGxCU1VRc2NVTkJRVzlDTEVkQlFYQkNMRlZCUVhGQ0xFVkJRVms3VVVGRE4wSXNTVUZCU1N4RFFVRkRMRkZCUVZFc1IwRkJSeXhKUVVGSkxFTkJRVU1zUlVGQlJTeERRVUZETEVkQlFVY3NRMEZCUXl4VlFVRkJMRU5CUVVNc1NVRkJTeXhEUVVGRExFTkJRVU1zVFVGQlRTeEhRVUZITEV0QlFVc3NRMEZCUXl4RFFVRkRMRTlCUVU4c1EwRkJReXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETEVOQlFVTTdVVUZEYWtVc1NVRkJTU3hEUVVGRExFVkJRVVVzUjBGQlJ5eEpRVUZKTEUxQlFVMHNRMEZCUXl4SlFVRkpMRU5CUVVNc1JVRkJSU3hEUVVGRExFTkJRVU03VVVGRE9VSXNTVUZCU1N4RFFVRkRMRVZCUVVVc1EwRkJReXhQUVVGUExFTkJRVU1zVlVGQlF5eERRVUZETEVWQlFVVXNRMEZCUXl4SlFVRkxMRTlCUVVFc1EwRkJReXhEUVVGRExFbEJRVWtzUjBGQlJ5eEZRVUZGTEVOQlFVTXNRMEZCUXl4RFFVRkRMRVZCUVdRc1EwRkJZeXhEUVVGRExFTkJRVU03U1VGRE9VTXNRMEZCUXp0SlFVVkVMRzlEUVVGdFFpeEhRVUZ1UWl4VlFVRnZRaXhGUVVGWk8xRkJRelZDTEVsQlFVa3NRMEZCUXl4RlFVRkZMRU5CUVVNc1QwRkJUeXhEUVVGRExGVkJRVU1zUTBGQlF5eEZRVUZGTEVOQlFVTXNTVUZCU3l4UFFVRkJMRU5CUVVNc1EwRkJReXhsUVVGbExFZEJRVWNzUlVGQlJTeERRVUZETEVOQlFVTXNRMEZCUXl4RlFVRjZRaXhEUVVGNVFpeERRVUZETEVOQlFVTTdTVUZEZWtRc1EwRkJRenRKUVRKQ1R5dzJRa0ZCV1N4SFFVRndRanRSUVVOSkxFbEJRVWtzVVVGQlVTeEhRVUZITEUxQlFVMHNRMEZCUXl4VFFVRlRMRVZCUXpOQ0xFTkJRVU1zUjBGQlpTeEpRVUZKTEVWQlEzQkNMRU5CUVVNc1IwRkJSeXhKUVVGSkxFTkJRVU1zVVVGQlVTeEZRVU5xUWl4RFFVRkRMRWRCUVVjc1EwRkJReXhEUVVGRExFMUJRVTBzUlVGRFdpeFhRVUZYTEVkQlFVY3NRMEZCUXl4RFFVRkRPMUZCUTNCQ0xFdEJRVXNzU1VGQlNTeERRVUZETEVkQlFVY3NRMEZCUXl4RlFVRkZMRU5CUVVNc1IwRkJSeXhEUVVGRExFVkJRVVVzUlVGQlJTeERRVUZETEVWQlFVVTdXVUZEZUVJc1NVRkJTU3hEUVVGRExFZEJRVWNzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXl4RFFVRkRPMWxCUTJJc1NVRkJTU3hEUVVGRExFTkJRVU1zWVVGQllUdG5Ra0ZCUlN4VFFVRlRPMWxCUXpsQ0xFbEJRVWtzUzBGQlN5eEhRVUZITEVOQlFVTXNRMEZCUXl4TFFVRkxMRVZCUVVVc1EwRkJRenRaUVVOMFFpeEpRVUZKTEVOQlFVTXNRMEZCUXl4UlFVRlJMRWxCUVVrc1MwRkJTeXhIUVVGSExGRkJRVkVzUlVGQlJUdG5Ra0ZEYUVNc1VVRkJVU3hIUVVGSExFdEJRVXNzUTBGQlF6dG5Ra0ZEYWtJc1EwRkJReXhIUVVGSExFTkJRVU1zUTBGQlF6dG5Ra0ZEVGl4WFFVRlhMRWRCUVVjc1EwRkJReXhEUVVGRE8yZENRVU5vUWl4SlFVRkpMRU5CUVVNc1EwRkJReXhSUVVGUk8yOUNRVUZGTEUxQlFVMDdZVUZEZWtJN1UwRkRTanRSUVVORUxFbEJRVWtzVjBGQlZ5eExRVUZMTEVOQlFVTTdXVUZEYWtJc1EwRkJReXhSUVVGUkxFZEJRVWNzVFVGQlRTeERRVUZETEdWQlFXVXNTVUZCU1N4RFFVRkRMRU5CUVVNc1EwRkJReXhOUVVGTkxFbEJRVWtzUTBGQlF5eERRVUZETEZGQlFWRXNRMEZCUXl4RlFVTnNSVHRaUVVOSkxFTkJRVU1zUTBGQlF5eFhRVUZYTEVOQlFVTXNSMEZCUnl4RFFVRkRMRU5CUVVNc1EwRkJReXhIUVVGSExFTkJRVU1zUTBGQlF5eERRVUZETzFsQlF6RkNMRU5CUVVNc1EwRkJReXhOUVVGTkxFZEJRVWNzUTBGQlF5eEhRVUZITEVOQlFVTXNRMEZCUXp0VFFVTndRanRSUVVORUxFOUJRVThzUTBGQlF5eERRVUZETzBsQlEySXNRMEZCUXp0SlFVbEVMSGRDUVVGUExFZEJRVkE3VVVGRFNTeEpRVUZKTEVsQlFVa3NRMEZCUXl4RlFVRkZMRWxCUVVrc1NVRkJTU3hGUVVGRk8xbEJRMnBDTEVsQlFVa3NRMEZCUXl4RlFVRkZMRWRCUVVjc1NVRkJTU3hOUVVGTkxFTkJRVU1zU1VGQlNTeERRVUZETEVWQlFVVXNRMEZCUXl4RFFVRkRPMU5CUTJwRE8xRkJTVVFzU1VGQlNTeERRVUZETEVWQlFVVXNRMEZCUXl4TFFVRkxMRU5CUVVNc1NVRkJTU3hEUVVGRExGRkJRVkVzUTBGQlF5eERRVUZETzFGQlF6ZENMRWxCUVVrc1EwRkJReXhIUVVGbExFbEJRVWtzUTBGQlF6dFJRVU42UWl4UFFVRlBMRU5CUVVNc1EwRkJReXhIUVVGSExFbEJRVWtzUTBGQlF5eFpRVUZaTEVWQlFVVXNRMEZCUXl4SlFVRkpMRU5CUVVNc1EwRkJReXhEUVVGRExGRkJRVkVzU1VGQlNTeERRVUZETEVOQlFVTXNTMEZCU3l4RlFVRkZMRWRCUVVjc1RVRkJUU3hEUVVGRExHVkJRV1VzU1VGQlNTeERRVUZETEVOQlFVTXNRMEZCUXl4TlFVRk5MRU5CUVVNc1JVRkJSVHRaUVVOcVJ5eEpRVUZKTEVWQlFVVXNSMEZCUnl4RFFVRkRMRU5CUVVNc1NVRkJTU3hEUVVGRExFdEJRVXNzUlVGQlJTeEZRVUZGTEVkQlFVY3NRMEZCUXl4RFFVRkRMRXRCUVVzc1EwRkJReXhMUVVGTExFTkJRVU03V1VGTk1VTXNTVUZCU1N4RlFVRkZMRXRCUVVzc1JVRkJSU3hGUVVGRk8yZENRVU5ZTEVsQlFVa3NRMEZCUXl4RlFVRkZMRU5CUVVNc1MwRkJTeXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETzJGQlEzQkNPMmxDUVVGTk8yZENRVU5JTEVsQlFVa3NSVUZCUlN4RFFVRkRMREpDUVVFeVFpeERRVUZETEVOQlFVTXNRMEZCUXl4TFFVRkxMRVZCUVVVc1EwRkJReXhEUVVGRExFbEJRVWtzUTBGQlF5eEZRVUZGTzI5Q1FVVnFSQ3hEUVVGRExFTkJRVU1zWVVGQllTeEhRVUZITEVsQlFVa3NRMEZCUXp0dlFrRkRka0lzVTBGQlV6dHBRa0ZEV2p0blFrRkZSQ3hKUVVGSkxFdEJRVXNzUjBGQlJ5eEZRVUZGTEVOQlFVTXNXVUZCV1N4RFFVRkRMRU5CUVVNc1EwRkJReXhKUVVGSkxFVkJRVVVzUTBGQlF5eERRVUZETEV0QlFVc3NRMEZCUXl4RFFVRkRPMmRDUVVNM1F5eEpRVUZKTEV0QlFVc3NTMEZCU3l4SlFVRkpMRVZCUVVVN2IwSkJRMmhDTEVsQlFVa3NRMEZCUXl4RlFVRkZMRU5CUVVNc1RVRkJUU3hEUVVGRExFdEJRVXNzUTBGQlF5eEZRVUZGTEVOQlFVTXNRMEZCUXp0dlFrRkRla0lzU1VGQlNTeERRVUZETEVWQlFVVXNRMEZCUXl4TlFVRk5MRU5CUVVNc1MwRkJTeXhEUVVGRExFVkJRVVVzUTBGQlF5eERRVUZETzI5Q1FVTjZRaXhKUVVGSkxFTkJRVU1zUlVGQlJTeERRVUZETEUxQlFVMHNRMEZCUXl4RlFVRkZMRU5CUVVNc1EwRkJRenR2UWtGRGJrSXNTVUZCU1N4RFFVRkRMRkZCUVZFc1EwRkJReXhKUVVGSkxFTkJRVU1zUzBGQlN5eERRVUZETEZWQlFWVXNRMEZCUXl4RFFVRkRPMmxDUVVONFF6dHhRa0ZCVFR0dlFrRkpTQ3hEUVVGRExFTkJRVU1zWVVGQllTeEhRVUZITEVsQlFVa3NRMEZCUXp0dlFrRkRka0lzVTBGQlV6dHBRa0ZEV2p0blFrRkRSQ3hKUVVGSkxFTkJRVU1zUTBGQlF5eExRVUZMTEVWQlFVVXNTVUZCU1N4RFFVRkRMRVZCUVVVN2IwSkJTMmhDTEVsQlFVa3NRMEZCUXl4UlFVRlJMRU5CUVVNc1NVRkJTU3hEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETzJsQ1FVTjZRanR4UWtGQlRUdHZRa0ZKU0N4SlFVRkpMRU5CUVVNc1JVRkJSU3hEUVVGRExFdEJRVXNzUTBGQlF5eERRVUZETEVOQlFVTXNRMEZCUXp0cFFrRkRjRUk3WVVGRFNqdFRRVTFLTzBsQlNVd3NRMEZCUXp0SlFVZEVMSE5DUVVGTExFZEJRVXc3VVVGRFNTeEpRVUZKTEVOQlFVTXNUMEZCVHl4RlFVRkZMRU5CUVVNN1VVRkRaaXhKUVVGSkxGRkJRVkVzUjBGQlJ5eE5RVUZOTEVOQlFVTXNVMEZCVXl4RlFVRkZMRWxCUVVrc1IwRkJSeXhKUVVGSkxFTkJRVU1zUlVGQlJTeERRVUZETEVsQlFVa3NSVUZCUlN4RFFVRkRPMUZCUTNaRUxFOUJRVThzU1VGQlNTeERRVUZETEVkQlFVY3NRMEZCUXl4UlFVRlJMRWRCUVVjc1NVRkJTU3hEUVVGRExFZEJRVWNzVFVGQlRTeEZRVUZGTzFsQlEzWkRMRWxCUVVrc1EwRkJReXhQUVVGUExFVkJRVVVzUTBGQlF6dFpRVU5tTEZGQlFWRXNSMEZCUnl4SlFVRkpMRU5CUVVNN1dVRkRhRUlzU1VGQlNTeEhRVUZITEVsQlFVa3NRMEZCUXl4RlFVRkZMRU5CUVVNc1NVRkJTU3hGUVVGRkxFTkJRVU03VTBGRGVrSTdVVUZEUkN4UFFVRlBMRWxCUVVrc1EwRkJRenRKUVVOb1FpeERRVUZETzBsQmNFdE5MREpDUVVGdlFpeEhRVUZITEVOQlFVTXNTVUZCU1N4RFFVRkRPMGxCUXpkQ0xITkNRVUZsTEVkQlFVY3NRMEZCUXl4TFFVRkxMRU5CUVVNN1NVRnZTM0JETEdGQlFVTTdRMEZCUVN4QlFYcExSQ3hKUVhsTFF6dEJRWHBMV1N4M1FrRkJUVHRCUVdsTWJrSXNVMEZCWjBJc01rSkJRVEpDTEVOQlFVTXNTMEZCWjBRc1JVRkJSU3hWUVVGdFFpeEZRVUZGTEZWQlFXMUNPMGxCUjJ4SkxFbEJRVTBzUlVGQlJTeEhRVUZsTEV0QlFVc3NRMEZCUXl4SFFVRkhMRU5CUVVNc1ZVRkJRU3hEUVVGRExFbEJRVWtzVDBGQlFTeEpRVUZKTEZGQlFWRXNRMEZCUXl4RFFVRkRMRU5CUVVNc1lVRkJZU3hEUVVGRExFVkJRVGRDTEVOQlFUWkNMRU5CUVVNc1EwRkJRenRKUVVOeVJTeEpRVUZOTEVWQlFVVXNSMEZCYVVJc1JVRkJSU3hEUVVGRE8wbEJRelZDTEVsQlFVMHNRMEZCUXl4SFFVRkhMRXRCUVVzc1EwRkJReXhOUVVGTkxFTkJRVU03U1VGRGRrSXNTMEZCU3l4SlFVRkpMRU5CUVVNc1IwRkJSeXhEUVVGRExFVkJRVVVzUTBGQlF5eEhRVUZITEVOQlFVTXNSMEZCUnl4RFFVRkRMRVZCUVVVc1EwRkJReXhGUVVGRkxFVkJRVVU3VVVGRE5VSXNTVUZCVFN4SlFVRkpMRWRCUVVjc1MwRkJTeXhEUVVGRExFTkJRVU1zUTBGQlF5eEZRVUZGTEV0QlFVc3NSMEZCUnl4TFFVRkxMRU5CUVVNc1EwRkJReXhIUVVGSExFTkJRVU1zUTBGQlF5eERRVUZETzFGQlF6VkRMRVZCUVVVc1EwRkJReXhKUVVGSkxFTkJRVU1zU1VGQlNTeFZRVUZWTEVOQlFVTXNSVUZCUlN4RFFVRkRMRU5CUVVNc1EwRkJReXhGUVVGRkxFVkJRVVVzUTBGQlF5eERRVUZETEVkQlFVY3NRMEZCUXl4RFFVRkRMRVZCUVVVc1EwRkJReXhKUVVGSkxFTkJRVU1zU1VGQlNTeEhRVUZITEV0QlFVc3NRMEZCUXl4SlFVRkpMRU5CUVVNc1IwRkJSeXhEUVVGRExFTkJRVU1zUTBGQlF5eERRVUZETzB0QlF6TkZPMGxCUTBRc1NVRkJUU3hSUVVGUkxFZEJRVWNzUlVGQlJTeERRVUZETEVOQlFVTXNRMEZCUXl4RlFVTnNRaXhUUVVGVExFZEJRVWNzUlVGQlJTeERRVUZETEVOQlFVTXNSMEZCUnl4RFFVRkRMRU5CUVVNc1JVRkRja0lzV1VGQldTeEhRVUZITEV0QlFVc3NRMEZCUXl4RFFVRkRMRU5CUVVNc1EwRkJReXhKUVVGSkxFZEJRVWNzUTBGQlF5eEZRVU5vUXl4aFFVRmhMRWRCUVVjc1MwRkJTeXhEUVVGRExFTkJRVU1zUjBGQlJ5eERRVUZETEVOQlFVTXNRMEZCUXl4SlFVRkpMRWRCUVVjc1EwRkJReXhEUVVGRE8wbEJRekZETEVsQlFVa3NUVUZCVFN4SFFVRmhMRWxCUVVrc1JVRkJSU3hOUVVGTkxFZEJRV0VzU1VGQlNTeERRVUZETzBsQlEzSkVMRWxCUVVrc1ZVRkJWU3hGUVVGRk8xRkJRMW9zVFVGQlRTeEhRVUZITEVsQlFVa3NVVUZCVVN4RFFVRkRMRlZCUVZVc1JVRkJSU3hSUVVGUkxFTkJRVU1zVFVGQlRTeEhRVUZITEVsQlFVa3NRMEZCUXl4RFFVRkRPMUZCUXpGRUxFVkJRVVVzUTBGQlF5eEpRVUZKTEVOQlFVTXNUVUZCVFN4RFFVRkRMRU5CUVVNN1VVRkRhRUlzUlVGQlJTeERRVUZETEVsQlFVa3NRMEZCUXl4SlFVRkpMRlZCUVZVc1EwRkJReXhOUVVGTkxFVkJRVVVzVVVGQlVTeEZRVUZGTEZsQlFWa3NRMEZCUXl4RFFVRkRMRU5CUVVNN1MwRkRNMFE3U1VGRFJDeEpRVUZKTEZWQlFWVXNSVUZCUlR0UlFVTmFMRTFCUVUwc1IwRkJSeXhKUVVGSkxGRkJRVkVzUTBGQlF5eFZRVUZWTEVWQlFVVXNVMEZCVXl4RFFVRkRMRTFCUVUwc1IwRkJSeXhKUVVGSkxFTkJRVU1zUTBGQlF6dFJRVU16UkN4RlFVRkZMRU5CUVVNc1NVRkJTU3hEUVVGRExFMUJRVTBzUTBGQlF5eERRVUZETzFGQlEyaENMRVZCUVVVc1EwRkJReXhKUVVGSkxFTkJRVU1zU1VGQlNTeFZRVUZWTEVOQlFVTXNVMEZCVXl4RlFVRkZMRTFCUVUwc1JVRkJSU3hoUVVGaExFTkJRVU1zUTBGQlF5eERRVUZETzB0QlF6ZEVPMGxCUTBRc1NVRkJTU3hOUVVGTkxFZEJRVWNzU1VGQlNTeE5RVUZOTEVOQlFVTXNSVUZCUlN4RlFVRkZMRVZCUVVVc1EwRkJReXhEUVVGRE8wbEJRMmhETEUxQlFVMHNRMEZCUXl4TFFVRkxMRVZCUVVVc1EwRkJRenRKUVVObUxFOUJRVTg3VVVGRFNDeFZRVUZWTEVWQlFVVXNSVUZCUlN4RFFVRkRMRXRCUVVzc1EwRkJReXhEUVVGRExFVkJRVVVzUzBGQlN5eERRVUZETEUxQlFVMHNRMEZCUXl4RFFVRkRMRWRCUVVjc1EwRkJReXhWUVVGQkxFTkJRVU1zU1VGQlNTeFBRVUZCTEVOQlFVTXNRMEZCUXl4UlFVRlJMRVZCUVVVc1JVRkJXaXhEUVVGWkxFTkJRVU03VVVGRE5VUXNWVUZCVlN4RlFVRkZMRTFCUVUwc1EwRkJReXhEUVVGRExFTkJRVU1zVFVGQlRTeERRVUZETEZGQlFWRXNSVUZCUlN4RFFVRkRMRU5CUVVNc1EwRkJReXhSUVVGUkxFTkJRVU1zVVVGQlVTeEZRVUZGTEVkQlFVY3NXVUZCV1R0UlFVTXpSU3hWUVVGVkxFVkJRVVVzVFVGQlRTeERRVUZETEVOQlFVTXNRMEZCUXl4TlFVRk5MRU5CUVVNc1VVRkJVU3hGUVVGRkxFTkJRVU1zUTBGQlF5eERRVUZETEZOQlFWTXNRMEZCUXl4UlFVRlJMRVZCUVVVc1IwRkJSeXhoUVVGaE8wdEJRMmhHTEVOQlFVTTdRVUZEVGl4RFFVRkRPMEZCYUVORUxHdEZRV2REUXlKOSJdfQ==
