"use strict";var r={appendKeyframes:(t,e)=>{if(!window?.document)return;const n=document.createElement("style");n.setAttribute("id",t),n.innerHTML=`@keyframes ${t} {${e}}`,document.body.appendChild(n)},removeKeyframes:t=>{if(!window?.document)return;const e=document.getElementById(t);!e||e.parentNode?.removeChild(e)}};exports.utils=r;
//# sourceMappingURL=utils.js.map
