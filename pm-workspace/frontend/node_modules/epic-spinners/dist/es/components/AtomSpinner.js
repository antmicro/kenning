import '../../style/AtomSpinner.css';
import { openBlock, createElementBlock, normalizeStyle, createElementVNode } from 'vue';
import { _export_sfc } from '../plugin-vue_export-helper.js';

var AtomSpinner_vue_vue_type_style_index_0_lang = '';

const _sfc_main = {
  name: 'AtomSpinner',

  props: {
    animationDuration: {
      type: Number,
      default: 1000,
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
      }
    },

    circleStyle() {
      return {
        color: this.color,
        fontSize: `${this.size * 0.24}px`,
      }
    },

    lineStyle() {
      return {
        animationDuration: `${this.animationDuration}ms`,
        borderLeftWidth: `${this.size / 25}px`,
        borderTopWidth: `${this.size / 25}px`,
        borderLeftColor: this.color,
      }
    },
  },
};

const _hoisted_1 = { class: "spinner-inner" };

function _sfc_render(_ctx, _cache, $props, $setup, $data, $options) {
  return (openBlock(), createElementBlock("div", {
    class: "atom-spinner",
    style: normalizeStyle($options.spinnerStyle)
  }, [
    createElementVNode("div", _hoisted_1, [
      createElementVNode("div", {
        class: "spinner-line",
        style: normalizeStyle($options.lineStyle)
      }, null, 4),
      createElementVNode("div", {
        class: "spinner-line",
        style: normalizeStyle($options.lineStyle)
      }, null, 4),
      createElementVNode("div", {
        class: "spinner-line",
        style: normalizeStyle($options.lineStyle)
      }, null, 4),
      createElementVNode("div", {
        class: "spinner-circle",
        style: normalizeStyle($options.circleStyle)
      }, "●", 4)
    ])
  ], 4))
}
var AtomSpinner = /*#__PURE__*/_export_sfc(_sfc_main, [['render',_sfc_render]]);

export { AtomSpinner };
