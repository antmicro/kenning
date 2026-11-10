import { openBlock, createElementBlock, normalizeStyle, createElementVNode } from 'vue';
import { _export_sfc } from '../plugin-vue_export-helper.mjs';

var TrinityRingsSpinner_vue_vue_type_style_index_0_lang = '';

const _sfc_main = {
  name: 'TrinityRingsSpinner',

  props: {
    animationDuration: {
      type: Number,
      default: 1500,
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

  data() {
    return {
      containerPadding: 3,
    }
  },

  computed: {
    outerRingSize() {
      return this.size - this.containerPadding * 2
    },

    spinnerStyle() {
      return {
        height: `${this.size}px`,
        width: `${this.size}px`,
        padding: `${this.containerPadding}px`,
      }
    },

    ring1Style() {
      return {
        height: `${this.outerRingSize}px`,
        width: `${this.outerRingSize}px`,
        borderColor: this.color,
        animationDuration: `${this.animationDuration}ms`,
      }
    },

    ring2Style() {
      return {
        height: `${this.outerRingSize * 0.65}px`,
        width: `${this.outerRingSize * 0.65}px`,
        borderColor: this.color,
        animationDuration: `${this.animationDuration}ms`,
      }
    },

    ring3Style() {
      return {
        height: `${this.outerRingSize * 0.1}px`,
        width: `${this.outerRingSize * 0.1}px`,
        borderColor: this.color,
        animationDuration: `${this.animationDuration}ms`,
      }
    },
  },
};

function _sfc_render(_ctx, _cache, $props, $setup, $data, $options) {
  return (openBlock(), createElementBlock("div", {
    class: "trinity-rings-spinner",
    style: normalizeStyle($options.spinnerStyle)
  }, [
    createElementVNode("div", {
      class: "circle circle1",
      style: normalizeStyle($options.ring1Style)
    }, null, 4),
    createElementVNode("div", {
      class: "circle circle2",
      style: normalizeStyle($options.ring2Style)
    }, null, 4),
    createElementVNode("div", {
      class: "circle circle3",
      style: normalizeStyle($options.ring3Style)
    }, null, 4)
  ], 4))
}
var TrinityRingsSpinner = /*#__PURE__*/_export_sfc(_sfc_main, [['render',_sfc_render]]);

export { TrinityRingsSpinner };
