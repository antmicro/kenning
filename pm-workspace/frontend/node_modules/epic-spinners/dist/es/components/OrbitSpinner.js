import '../../style/OrbitSpinner.css';
import { openBlock, createElementBlock, normalizeStyle, createElementVNode } from 'vue';
import { _export_sfc } from '../plugin-vue_export-helper.js';

var OrbitSpinner_vue_vue_type_style_index_0_lang = '';

const _sfc_main = {
  name: 'OrbitSpinner',

  props: {
    animationDuration: {
      type: Number,
      default: 1000,
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
      }
    },

    orbitStyle() {
      return {
        borderColor: this.color,
        animationDuration: `${this.animationDuration}ms`,
      }
    },
  },
};

function _sfc_render(_ctx, _cache, $props, $setup, $data, $options) {
  return (openBlock(), createElementBlock("div", {
    class: "orbit-spinner",
    style: normalizeStyle($options.spinnerStyle)
  }, [
    createElementVNode("div", {
      class: "orbit one",
      style: normalizeStyle($options.orbitStyle)
    }, null, 4),
    createElementVNode("div", {
      class: "orbit two",
      style: normalizeStyle($options.orbitStyle)
    }, null, 4),
    createElementVNode("div", {
      class: "orbit three",
      style: normalizeStyle($options.orbitStyle)
    }, null, 4)
  ], 4))
}
var OrbitSpinner = /*#__PURE__*/_export_sfc(_sfc_main, [['render',_sfc_render]]);

export { OrbitSpinner };
