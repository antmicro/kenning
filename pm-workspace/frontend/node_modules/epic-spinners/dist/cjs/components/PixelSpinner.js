"use strict";var i=require("../services/utils.js"),e=require("vue"),p=require("../plugin-vue_export-helper.js");const s={name:"PixelSpinner",props:{animationDuration:{type:Number,default:1500},size:{type:Number,default:70},color:{type:String,default:"#fff"}},data(){return{animationName:`pixel-spinner-animation-${Date.now()}`}},computed:{pixelSize(){return this.size/7},spinnerStyle(){return{width:`${this.size}px`,height:`${this.size}px`}},spinnerInnerStyle(){return{animationDuration:`${this.animationDuration}ms`,animationName:this.animationName,width:`${this.pixelSize}px`,height:`${this.pixelSize}px`,backgroundColor:this.color,color:this.color,boxShadow:`
          ${this.pixelSize*1.5}px ${this.pixelSize*1.5}px 0 0,
          ${this.pixelSize*-1.5}px ${this.pixelSize*-1.5}px 0 0,
          ${this.pixelSize*1.5}px ${this.pixelSize*-1.5}px 0 0,
          ${this.pixelSize*-1.5}px ${this.pixelSize*1.5}px 0 0,
          0 ${this.pixelSize*1.5}px 0 0,
          ${this.pixelSize*1.5}px 0 0 0,
          ${this.pixelSize*-1.5}px 0 0 0,
          0 ${this.pixelSize*-1.5}px 0 0
        `}}},watch:{size:{handler:"updateAnimation",immediate:!0}},mounted(){this.updateAnimation()},beforeUnmount(){i.utils.removeKeyframes(this.animationName)},methods:{updateAnimation(){i.utils.removeKeyframes(this.animationName),i.utils.appendKeyframes(this.animationName,this.generateKeyframes())},generateKeyframes(){return`
        50% {
          box-shadow:  ${this.pixelSize*2}px ${this.pixelSize*2}px 0 0,
                       ${this.pixelSize*-2}px ${this.pixelSize*-2}px 0 0,
                       ${this.pixelSize*2}px ${this.pixelSize*-2}px 0 0,
                       ${this.pixelSize*-2}px ${this.pixelSize*2}px 0 0,
                       0 ${this.pixelSize}px 0 0,
                       ${this.pixelSize}px 0 0 0,
                       ${this.pixelSize*-1}px 0 0 0,
                       0 ${this.pixelSize*-1}px 0 0;
        }
        75% {
          box-shadow:  ${this.pixelSize*2}px ${this.pixelSize*2}px 0 0,
                       ${this.pixelSize*-2}px ${this.pixelSize*-2}px 0 0,
                       ${this.pixelSize*2}px ${this.pixelSize*-2}px 0 0,
                       ${this.pixelSize*-2}px ${this.pixelSize*2}px 0 0,
                       0 ${this.pixelSize}px 0 0,
                       ${this.pixelSize}px 0 0 0,
                       ${this.pixelSize*-1}px 0 0 0,
                       0 ${this.pixelSize*-1}px 0 0;
        }
        100% {
          transform: rotate(360deg);
        }`}}};function x(l,r,a,h,o,t){return e.openBlock(),e.createElementBlock("div",{class:"pixel-spinner",style:e.normalizeStyle(t.spinnerStyle)},[e.createElementVNode("div",{class:"pixel-spinner-inner",style:e.normalizeStyle(t.spinnerInnerStyle)},null,4)],4)}var n=p._export_sfc(s,[["render",x]]);exports.PixelSpinner=n;
//# sourceMappingURL=PixelSpinner.js.map
