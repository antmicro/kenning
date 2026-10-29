"use strict";var e=require("../services/utils.js"),t=require("vue"),i=require("../plugin-vue_export-helper.js");const s={name:"FlowerSpinner",props:{animationDuration:{type:Number,default:2500},size:{type:Number,default:70},color:{type:String,default:"#fff"}},data(){return{smallDotName:`flower-spinner-small-dot-${Date.now()}`,bigDotName:`flower-spinner-big-dot-${Date.now()}`}},computed:{dotSize(){return this.size/7},spinnerStyle(){return{width:`${this.size}px`,height:`${this.size}px`}},dotsContainerStyle(){return{width:`${this.dotSize}px`,height:`${this.dotSize}px`}},smallerDotStyle(){return{background:this.color,animationDuration:`${this.animationDuration}ms`,animationName:this.smallDotName}},biggerDotStyle(){return{background:this.color,animationDuration:`${this.animationDuration}ms`,animationName:this.bigDotName}}},watch:{size:{handler:"updateAnimation",immediate:!0},color:{handler:"updateAnimation",immediate:!0}},beforeUnmount(){e.utils.removeKeyframes(this.smallDotName),e.utils.removeKeyframes(this.bigDotName)},methods:{updateAnimation(){e.utils.removeKeyframes(this.smallDotName),e.utils.appendKeyframes(this.smallDotName,this.generateSmallDotKeyframes()),e.utils.removeKeyframes(this.bigDotName),e.utils.appendKeyframes(this.bigDotName,this.generateBigDotKeyframes())},generateSmallDotKeyframes(){return`
        0%, 100% {
          box-shadow: 0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color};
        }
        25%, 75% {
          box-shadow: ${this.dotSize*1.4}px 0 0 ${this.color},
                      -${this.dotSize*1.4}px 0 0 ${this.color},
                      0 ${this.dotSize*1.4}px 0 ${this.color},
                      0 -${this.dotSize*1.4}px 0 ${this.color},
                      ${this.dotSize}px -${this.dotSize}px 0 ${this.color},
                      ${this.dotSize}px ${this.dotSize}px 0 ${this.color},
                      -${this.dotSize}px -${this.dotSize}px 0 ${this.color},
                      -${this.dotSize}px ${this.dotSize}px 0 ${this.color};
        }
        100% {
          box-shadow: 0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color};
        }`},generateBigDotKeyframes(){return`
        0%, 100% {
          box-shadow: 0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color};
        }
        50% {
          transform: rotate(180deg);
        }
        25%, 75% {
          box-shadow: ${this.dotSize*2.6}px 0 0 ${this.color},
                      -${this.dotSize*2.6}px 0 0 ${this.color},
                      0 ${this.dotSize*2.6}px 0 ${this.color},
                      0 -${this.dotSize*2.6}px 0 ${this.color},
                      ${this.dotSize*1.9}px -${this.dotSize*1.9}px 0 ${this.color},
                      ${this.dotSize*1.9}px ${this.dotSize*1.9}px 0 ${this.color},
                      -${this.dotSize*1.9}px -${this.dotSize*1.9}px 0 ${this.color},
                      -${this.dotSize*1.9}px ${this.dotSize*1.9}px 0 ${this.color};
        }
        100% {
          transform: rotate(360deg);
          box-shadow: 0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color};
        }`}}};function r(a,h,n,c,$,o){return t.openBlock(),t.createElementBlock("div",{class:"flower-spinner",style:t.normalizeStyle(o.spinnerStyle)},[t.createElementVNode("div",{class:"dots-container",style:t.normalizeStyle(o.dotsContainerStyle)},[t.createElementVNode("div",{class:"big-dot",style:t.normalizeStyle(o.biggerDotStyle)},[t.createElementVNode("div",{class:"small-dot",style:t.normalizeStyle(o.smallerDotStyle)},null,4)],4)],4)],4)}var l=i._export_sfc(s,[["render",r]]);exports.FlowerSpinner=l;
//# sourceMappingURL=FlowerSpinner.js.map
