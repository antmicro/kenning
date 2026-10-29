import '../../style/FulfillingBouncingCircleSpinner.css';
import { openBlock, createElementBlock, normalizeStyle, createElementVNode } from 'vue';
import { _export_sfc } from '../plugin-vue_export-helper.js';

var FulfillingBouncingCircleSpinner_vue_vue_type_style_index_0_lang = '';

const _sfc_main = {
  name: 'FulfillingBouncingCircleSpinner',

  props: {
    animationDuration: {
      type: Number,
      default: 4000,
    },
    size: {
      type: Number,
      default: 60,
    },
    color: {
      type: String,
      default: '#fff',
    },
  },

  computed: {
    spinnerStyle() {
      return {
        height: `${this.size}px`,
        width: `${this.size}px`,
        animationDuration: `${this.animationDuration}ms`,
      }
    },

    orbitStyle() {
      return {
        height: `${this.size}px`,
        width: `${this.size}px`,
        borderColor: this.color,
        borderWidth: `${this.size * 0.03}px`,
        animationDuration: `${this.animationDuration}ms`,
      }
    },

    circleStyle() {
      return {
        height: `${this.size}px`,
        width: `${this.size}px`,
        borderColor: this.color,
        color: this.color,
        borderWidth: `${this.size * 0.1}px`,
        animationDuration: `${this.animationDuration}ms`,
      }
    },
  },
};

function _sfc_render(_ctx, _cache, $props, $setup, $data, $options) {
  return (openBlock(), createElementBlock("div", {
    class: "fulfilling-bouncing-circle-spinner",
    style: normalizeStyle($options.spinnerStyle)
  }, [
    createElementVNode("div", {
      class: "circle",
      style: normalizeStyle($options.circleStyle)
    }, null, 4),
    createElementVNode("div", {
      class: "orbit",
      style: normalizeStyle($options.orbitStyle)
    }, null, 4)
  ], 4))
}
var FulfillingBouncingCircleSpinner = /*#__PURE__*/_export_sfc(_sfc_main, [['render',_sfc_render]]);

export { FulfillingBouncingCircleSpinner };
