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
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYWRhcHRvci5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uL1dlYkNvbGEvc3JjL2FkYXB0b3IudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7O0FBQUEsbUNBQWlEO0FBRTdDO0lBQW1DLGlDQUFNO0lBYXJDLHVCQUFhLE9BQU87UUFBcEIsWUFDSSxpQkFBTyxTQXlCVjtRQXJCRyxJQUFJLElBQUksR0FBRyxLQUFJLENBQUM7UUFDaEIsSUFBSSxDQUFDLEdBQUcsT0FBTyxDQUFDO1FBRWhCLElBQUssQ0FBQyxDQUFDLE9BQU8sRUFBRztZQUNiLEtBQUksQ0FBQyxPQUFPLEdBQUcsQ0FBQyxDQUFDLE9BQU8sQ0FBQztTQUM1QjtRQUVELElBQUssQ0FBQyxDQUFDLElBQUksRUFBRTtZQUNULEtBQUksQ0FBQyxJQUFJLEdBQUcsQ0FBQyxDQUFDLElBQUksQ0FBQztTQUN0QjtRQUVELElBQUssQ0FBQyxDQUFDLElBQUksRUFBRTtZQUNULEtBQUksQ0FBQyxJQUFJLEdBQUcsQ0FBQyxDQUFDLElBQUksQ0FBQztTQUN0QjtRQUVELElBQUssQ0FBQyxDQUFDLEVBQUUsRUFBRTtZQUNQLEtBQUksQ0FBQyxFQUFFLEdBQUcsQ0FBQyxDQUFDLEVBQUUsQ0FBQztTQUNsQjtRQUVELEtBQUksQ0FBQyxTQUFTLEdBQUcsS0FBSSxDQUFDLFNBQVMsR0FBRyxlQUFNLENBQUMsU0FBUyxDQUFDO1FBQ25ELEtBQUksQ0FBQyxPQUFPLEdBQUcsS0FBSSxDQUFDLE9BQU8sR0FBRyxlQUFNLENBQUMsT0FBTyxDQUFDOztJQUNqRCxDQUFDO0lBcENELCtCQUFPLEdBQVAsVUFBUSxDQUFRLElBQUcsQ0FBQztJQUFBLENBQUM7SUFDckIsNEJBQUksR0FBSixjQUFRLENBQUM7SUFBQSxDQUFDO0lBQ1YsNEJBQUksR0FBSixjQUFRLENBQUM7SUFBQSxDQUFDO0lBQ1YsMEJBQUUsR0FBRixVQUFHLFNBQTZCLEVBQUUsUUFBb0IsSUFBVyxPQUFPLElBQUksQ0FBQyxDQUFDLENBQUM7SUFBQSxDQUFDO0lBa0NwRixvQkFBQztBQUFELENBQUMsQUF4Q0QsQ0FBbUMsZUFBTSxHQXdDeEM7QUF4Q1ksc0NBQWE7QUE2QzFCLFNBQWdCLE9BQU8sQ0FBRSxPQUFPO0lBQzVCLE9BQU8sSUFBSSxhQUFhLENBQUUsT0FBTyxDQUFFLENBQUM7QUFDeEMsQ0FBQztBQUZELDBCQUVDIn0=