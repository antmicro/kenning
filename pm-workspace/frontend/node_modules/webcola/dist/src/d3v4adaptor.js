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
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZDN2NGFkYXB0b3IuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi9XZWJDb2xhL3NyYy9kM3Y0YWRhcHRvci50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7QUFHQSxtQ0FBaUQ7QUFVakQ7SUFBMEMsd0NBQU07SUFpQjVDLDhCQUFvQixTQUFvQjtRQUF4QyxZQUNJLGlCQUFPLFNBeUJWO1FBMUJtQixlQUFTLEdBQVQsU0FBUyxDQUFXO1FBRXBDLEtBQUksQ0FBQyxLQUFLLEdBQUcsU0FBUyxDQUFDLFFBQVEsQ0FBQyxrQkFBUyxDQUFDLGtCQUFTLENBQUMsS0FBSyxDQUFDLEVBQUUsa0JBQVMsQ0FBQyxrQkFBUyxDQUFDLElBQUksQ0FBQyxFQUFFLGtCQUFTLENBQUMsa0JBQVMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDO1FBR2pILElBQUksUUFBUSxHQUFHLEtBQUksQ0FBQztRQUNwQixJQUFJLElBQUksQ0FBQztRQUNULEtBQUksQ0FBQyxJQUFJLEdBQUc7WUFDUixJQUFJLENBQUMsSUFBSSxFQUFFO2dCQUNQLElBQUksSUFBSSxHQUFHLFNBQVMsQ0FBQyxJQUFJLEVBQUU7cUJBQ3RCLE9BQU8sQ0FBQyxlQUFNLENBQUMsVUFBVSxDQUFDO3FCQUMxQixFQUFFLENBQUMsaUJBQWlCLEVBQUUsZUFBTSxDQUFDLFNBQVMsQ0FBQztxQkFDdkMsRUFBRSxDQUFDLGdCQUFnQixFQUFFLFVBQUEsQ0FBQztvQkFDbkIsZUFBTSxDQUFDLElBQUksQ0FBTSxDQUFDLEVBQUUsU0FBUyxDQUFDLEtBQUssQ0FBQyxDQUFDO29CQUNyQyxRQUFRLENBQUMsTUFBTSxFQUFFLENBQUM7Z0JBQ3RCLENBQUMsQ0FBQztxQkFDRCxFQUFFLENBQUMsZUFBZSxFQUFFLGVBQU0sQ0FBQyxPQUFPLENBQUMsQ0FBQzthQUM1QztZQUVELElBQUksQ0FBQyxTQUFTLENBQUMsTUFBTTtnQkFBRSxPQUFPLElBQUksQ0FBQztZQUtuQyxTQUFTLENBQUMsQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDO1FBQzVCLENBQUMsQ0FBQTs7SUFDTCxDQUFDO0lBekNELHNDQUFPLEdBQVAsVUFBUSxDQUFRO1FBQ1osSUFBSSxPQUFPLEdBQUcsRUFBRSxJQUFJLEVBQUUsa0JBQVMsQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLEVBQUUsS0FBSyxFQUFFLENBQUMsQ0FBQyxLQUFLLEVBQUUsTUFBTSxFQUFFLENBQUMsQ0FBQyxNQUFNLEVBQUUsQ0FBQztRQUc1RSxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxPQUFPLENBQUMsSUFBSSxFQUFPLE9BQU8sQ0FBQyxDQUFDO0lBQ2hELENBQUM7SUFHRCxtQ0FBSSxHQUFKO1FBQUEsaUJBRUM7UUFERyxJQUFJLENBQUMsR0FBRyxJQUFJLENBQUMsU0FBUyxDQUFDLEtBQUssQ0FBQyxjQUFNLE9BQUEsaUJBQU0sSUFBSSxZQUFFLElBQUksQ0FBQyxDQUFDLElBQUksRUFBRSxFQUF4QixDQUF3QixDQUFDLENBQUM7SUFDakUsQ0FBQztJQWtDRCxpQ0FBRSxHQUFGLFVBQUcsU0FBNkIsRUFBRSxRQUFvQjtRQUNsRCxJQUFJLE9BQU8sU0FBUyxLQUFLLFFBQVEsRUFBRTtZQUMvQixJQUFJLENBQUMsS0FBSyxDQUFDLEVBQUUsQ0FBQyxTQUFTLEVBQUUsUUFBUSxDQUFDLENBQUM7U0FDdEM7YUFBTTtZQUNILElBQUksQ0FBQyxLQUFLLENBQUMsRUFBRSxDQUFDLGtCQUFTLENBQUMsU0FBUyxDQUFDLEVBQUUsUUFBUSxDQUFDLENBQUM7U0FDakQ7UUFDRCxPQUFPLElBQUksQ0FBQztJQUNoQixDQUFDO0lBQ0wsMkJBQUM7QUFBRCxDQUFDLEFBdERELENBQTBDLGVBQU0sR0FzRC9DO0FBdERZLG9EQUFvQiJ9