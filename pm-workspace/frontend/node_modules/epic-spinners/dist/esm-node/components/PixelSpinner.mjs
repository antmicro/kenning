import { utils } from '../services/utils.mjs';
import { openBlock, createElementBlock, normalizeStyle, createElementVNode } from 'vue';
import { _export_sfc } from '../plugin-vue_export-helper.mjs';

var PixelSpinner_vue_vue_type_style_index_0_lang = '';

const _sfc_main = {
  name: 'PixelSpinner',

  props: {
    animationDuration: {
      type: Number,
      default: 1500,
    },
    size: {
      type: Number,
      default: 70,
    },
    color: {
      type: String,
      default: '#fff',
    },
  },

  data() {
    return {
      animationName: `pixel-spinner-animation-${Date.now()}`,
    }
  },

  computed: {
    pixelSize() {
      return this.size / 7
    },

    spinnerStyle() {
      return {
        width: `${this.size}px`,
        height: `${this.size}px`,
      }
    },

    spinnerInnerStyle() {
      return {
        animationDuration: `${this.animationDuration}ms`,
        animationName: this.animationName,
        width: `${this.pixelSize}px`,
        height: `${this.pixelSize}px`,
        backgroundColor: this.color,
        color: this.color,
        boxShadow: `
          ${this.pixelSize * 1.5}px ${this.pixelSize * 1.5}px 0 0,
          ${this.pixelSize * -1.5}px ${this.pixelSize * -1.5}px 0 0,
          ${this.pixelSize * 1.5}px ${this.pixelSize * -1.5}px 0 0,
          ${this.pixelSize * -1.5}px ${this.pixelSize * 1.5}px 0 0,
          0 ${this.pixelSize * 1.5}px 0 0,
          ${this.pixelSize * 1.5}px 0 0 0,
          ${this.pixelSize * -1.5}px 0 0 0,
          0 ${this.pixelSize * -1.5}px 0 0
        `,
      }
    },
  },

  watch: {
    size: {
      handler: 'updateAnimation',
      immediate: true,
    },
  },

  mounted() {
    this.updateAnimation();
  },

  beforeUnmount() {
    utils.removeKeyframes(this.animationName);
  },

  methods: {
    updateAnimation() {
      utils.removeKeyframes(this.animationName);
      utils.appendKeyframes(this.animationName, this.generateKeyframes());
    },

    generateKeyframes() {
      return `
        50% {
          box-shadow:  ${this.pixelSize * 2}px ${this.pixelSize * 2}px 0 0,
                       ${this.pixelSize * -2}px ${this.pixelSize * -2}px 0 0,
                       ${this.pixelSize * 2}px ${this.pixelSize * -2}px 0 0,
                       ${this.pixelSize * -2}px ${this.pixelSize * 2}px 0 0,
                       0 ${this.pixelSize}px 0 0,
                       ${this.pixelSize}px 0 0 0,
                       ${this.pixelSize * -1}px 0 0 0,
                       0 ${this.pixelSize * -1}px 0 0;
        }
        75% {
          box-shadow:  ${this.pixelSize * 2}px ${this.pixelSize * 2}px 0 0,
                       ${this.pixelSize * -2}px ${this.pixelSize * -2}px 0 0,
                       ${this.pixelSize * 2}px ${this.pixelSize * -2}px 0 0,
                       ${this.pixelSize * -2}px ${this.pixelSize * 2}px 0 0,
                       0 ${this.pixelSize}px 0 0,
                       ${this.pixelSize}px 0 0 0,
                       ${this.pixelSize * -1}px 0 0 0,
                       0 ${this.pixelSize * -1}px 0 0;
        }
        100% {
          transform: rotate(360deg);
        }`
    },
  },
};

function _sfc_render(_ctx, _cache, $props, $setup, $data, $options) {
  return (openBlock(), createElementBlock("div", {
    class: "pixel-spinner",
    style: normalizeStyle($options.spinnerStyle)
  }, [
    createElementVNode("div", {
      class: "pixel-spinner-inner",
      style: normalizeStyle($options.spinnerInnerStyle)
    }, null, 4)
  ], 4))
}
var PixelSpinner = /*#__PURE__*/_export_sfc(_sfc_main, [['render',_sfc_render]]);

export { PixelSpinner };
