import '../../style/HalfCircleSpinner.css';
import { openBlock, createElementBlock, normalizeStyle, createElementVNode } from 'vue';
import { _export_sfc } from '../plugin-vue_export-helper.js';

var HalfCircleSpinner_vue_vue_type_style_index_0_lang = '';

const _sfc_main = {
  name: 'HalfCircleSpinner',

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
        borderWidth: `${this.size / 10}px`,
        animationDuration: `${this.animationDuration}ms`,
      }
    },

    circle1Style() {
      return Object.assign(
        {
          borderTopColor: this.color,
        },
        this.circleStyle,
      )
    },

    circle2Style() {
      return Object.assign(
        {
          borderBottomColor: this.color,
        },
        this.circleStyle,
      )
    },
  },
};

function _sfc_render(_ctx, _cache, $props, $setup, $data, $options) {
  return (openBlock(), createElementBlock("div", {
    class: "half-circle-spinner",
    style: normalizeStyle($options.spinnerStyle)
  }, [
    createElementVNode("div", {
      class: "circle circle-1",
      style: normalizeStyle($options.circle1Style)
    }, null, 4),
    createElementVNode("div", {
      class: "circle circle-2",
      style: normalizeStyle($options.circle2Style)
    }, null, 4)
  ], 4))
}
var HalfCircleSpinner = /*#__PURE__*/_export_sfc(_sfc_main, [['render',_sfc_render]]);

export { HalfCircleSpinner };
