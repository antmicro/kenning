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
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZDNhZGFwdG9yLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vV2ViQ29sYS9zcmMvZDNhZGFwdG9yLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiI7O0FBQUEsb0NBQXFDO0FBQ3JDLG9DQUFxQztBQUdVLENBQUM7QUE0QmhELFNBQWdCLFNBQVMsQ0FBQyxTQUF3QztJQUM5RCxJQUFJLENBQUMsU0FBUyxJQUFJLE1BQU0sQ0FBQyxTQUFTLENBQUMsRUFBRTtRQUNqQyxPQUFPLElBQUksSUFBSSxDQUFDLG9CQUFvQixFQUFFLENBQUM7S0FDMUM7SUFDRCxPQUFPLElBQUksSUFBSSxDQUFDLG9CQUFvQixDQUFDLFNBQVMsQ0FBQyxDQUFDO0FBQ3BELENBQUM7QUFMRCw4QkFLQztBQUVELFNBQVMsTUFBTSxDQUFDLFNBQXVDO0lBQ25ELElBQU0sS0FBSyxHQUFHLE1BQU0sQ0FBQztJQUNyQixPQUFhLFNBQVUsQ0FBQyxPQUFPLElBQVUsU0FBVSxDQUFDLE9BQU8sQ0FBQyxLQUFLLENBQUMsS0FBSyxDQUFDLEtBQUssSUFBSSxDQUFDO0FBQ3RGLENBQUMifQ==