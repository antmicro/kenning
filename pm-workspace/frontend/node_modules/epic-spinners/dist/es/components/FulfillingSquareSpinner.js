import '../../style/FulfillingSquareSpinner.css';
import { openBlock, createElementBlock, normalizeStyle, createElementVNode } from 'vue';
import { _export_sfc } from '../plugin-vue_export-helper.js';

var FulfillingSquareSpinner_vue_vue_type_style_index_0_lang = '';

const _sfc_main = {
  name: 'FulfillingSquareSpinner',

  props: {
    animationDuration: {
      type: Number,
      default: 4000,
    },
    size: {
      type: Number,
      default: 50,
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
        borderColor: this.color,
        animationDuration: `${this.animationDuration}ms`,
      }
    },

    spinnerInnerStyle() {
      return {
        backgroundColor: this.color,
        animationDuration: `${this.animationDuration}ms`,
      }
    },
  },
};

function _sfc_render(_ctx, _cache, $props, $setup, $data, $options) {
  return (openBlock(), createElementBlock("div", {
    class: "fulfilling-square-spinner",
    style: normalizeStyle($options.spinnerStyle)
  }, [
    createElementVNode("div", {
      class: "spinner-inner",
      style: normalizeStyle($options.spinnerInnerStyle)
    }, null, 4)
  ], 4))
}
var FulfillingSquareSpinner = /*#__PURE__*/_export_sfc(_sfc_main, [['render',_sfc_render]]);

export { FulfillingSquareSpinner };
