"use strict";var r=require("../services/utils.js"),e=require("vue"),i=require("../plugin-vue_export-helper.js");const n={name:"SpringSpinner",props:{animationDuration:{type:Number,default:3e3},size:{type:Number,default:70},color:{type:String,default:"#fff"}},data(){return{animationName:`spring-spinner-animation-${Date.now()}`}},computed:{spinnerStyle(){return{height:`${this.size}px`,width:`${this.size}px`}},spinnerPartStyle(){return{height:`${this.size/2}px`,width:`${this.size}px`}},rotatorStyle(){return{height:`${this.size}px`,width:`${this.size}px`,borderRightColor:this.color,borderTopColor:this.color,borderWidth:`${this.size/7}px`,animationDuration:`${this.animationDuration}ms`,animationName:this.animationName}}},watch:{size:{handler:"updateAnimation",immediate:!0},color:{handler:"updateAnimation",immediate:!0}},mounted(){this.updateAnimation()},beforeUnmount(){r.utils.removeKeyframes(this.animationName)},methods:{updateAnimation(){r.utils.removeKeyframes(this.animationName),r.utils.appendKeyframes(this.animationName,this.generateKeyframes())},generateKeyframes(){return`
        0% {
          border-width: ${this.size/7}px;
        }
        25% {
          border-width: ${this.size/23.33}px;
        }
        50% {
          transform: rotate(115deg);
          border-width: ${this.size/7}px;
        }
        75% {
          border-width: ${this.size/23.33}px;
          }
        100% {
          border-width: ${this.size/7}px;
        }`}}};function a(o,l,p,m,d,t){return e.openBlock(),e.createElementBlock("div",{class:"spring-spinner",style:e.normalizeStyle(t.spinnerStyle)},[e.createElementVNode("div",{class:"spring-spinner-part top",style:e.normalizeStyle(t.spinnerPartStyle)},[e.createElementVNode("div",{class:"spring-spinner-rotator",style:e.normalizeStyle(t.rotatorStyle)},null,4)],4),e.createElementVNode("div",{class:"spring-spinner-part bottom",style:e.normalizeStyle(t.spinnerPartStyle)},[e.createElementVNode("div",{class:"spring-spinner-rotator",style:e.normalizeStyle(t.rotatorStyle)},null,4)],4)],4)}var s=i._export_sfc(n,[["render",a]]);exports.SpringSpinner=s;
//# sourceMappingURL=SpringSpinner.js.map
