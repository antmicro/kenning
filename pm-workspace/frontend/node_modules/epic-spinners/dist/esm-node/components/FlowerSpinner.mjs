import { utils } from '../services/utils.mjs';
import { openBlock, createElementBlock, normalizeStyle, createElementVNode } from 'vue';
import { _export_sfc } from '../plugin-vue_export-helper.mjs';

var FlowerSpinner_vue_vue_type_style_index_0_lang = '';

const _sfc_main = {
  name: 'FlowerSpinner',

  props: {
    animationDuration: {
      type: Number,
      default: 2500,
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
      smallDotName: `flower-spinner-small-dot-${Date.now()}`,
      bigDotName: `flower-spinner-big-dot-${Date.now()}`,
    }
  },

  computed: {
    dotSize() {
      return this.size / 7
    },

    spinnerStyle() {
      return {
        width: `${this.size}px`,
        height: `${this.size}px`,
      }
    },

    dotsContainerStyle() {
      return {
        width: `${this.dotSize}px`,
        height: `${this.dotSize}px`,
      }
    },

    smallerDotStyle() {
      return {
        background: this.color,
        animationDuration: `${this.animationDuration}ms`,
        animationName: this.smallDotName,
      }
    },

    biggerDotStyle() {
      return {
        background: this.color,
        animationDuration: `${this.animationDuration}ms`,
        animationName: this.bigDotName,
      }
    },
  },

  watch: {
    size: {
      handler: 'updateAnimation',
      immediate: true,
    },
    color: {
      handler: 'updateAnimation',
      immediate: true,
    },
  },

  beforeUnmount() {
    utils.removeKeyframes(this.smallDotName);
    utils.removeKeyframes(this.bigDotName);
  },

  methods: {
    updateAnimation() {
      utils.removeKeyframes(this.smallDotName);
      utils.appendKeyframes(this.smallDotName, this.generateSmallDotKeyframes());
      utils.removeKeyframes(this.bigDotName);
      utils.appendKeyframes(this.bigDotName, this.generateBigDotKeyframes());
    },

    generateSmallDotKeyframes() {
      return `
        0%, 100% {
          box-shadow: 0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color};
        }
        25%, 75% {
          box-shadow: ${this.dotSize * 1.4}px 0 0 ${this.color},
                      -${this.dotSize * 1.4}px 0 0 ${this.color},
                      0 ${this.dotSize * 1.4}px 0 ${this.color},
                      0 -${this.dotSize * 1.4}px 0 ${this.color},
                      ${this.dotSize}px -${this.dotSize}px 0 ${this.color},
                      ${this.dotSize}px ${this.dotSize}px 0 ${this.color},
                      -${this.dotSize}px -${this.dotSize}px 0 ${this.color},
                      -${this.dotSize}px ${this.dotSize}px 0 ${this.color};
        }
        100% {
          box-shadow: 0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color};
        }`
    },

    generateBigDotKeyframes() {
      return `
        0%, 100% {
          box-shadow: 0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color};
        }
        50% {
          transform: rotate(180deg);
        }
        25%, 75% {
          box-shadow: ${this.dotSize * 2.6}px 0 0 ${this.color},
                      -${this.dotSize * 2.6}px 0 0 ${this.color},
                      0 ${this.dotSize * 2.6}px 0 ${this.color},
                      0 -${this.dotSize * 2.6}px 0 ${this.color},
                      ${this.dotSize * 1.9}px -${this.dotSize * 1.9}px 0 ${this.color},
                      ${this.dotSize * 1.9}px ${this.dotSize * 1.9}px 0 ${this.color},
                      -${this.dotSize * 1.9}px -${this.dotSize * 1.9}px 0 ${this.color},
                      -${this.dotSize * 1.9}px ${this.dotSize * 1.9}px 0 ${this.color};
        }
        100% {
          transform: rotate(360deg);
          box-shadow: 0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color},
                      0 0 0 ${this.color};
        }`
    },
  },
};

function _sfc_render(_ctx, _cache, $props, $setup, $data, $options) {
  return (openBlock(), createElementBlock("div", {
    class: "flower-spinner",
    style: normalizeStyle($options.spinnerStyle)
  }, [
    createElementVNode("div", {
      class: "dots-container",
      style: normalizeStyle($options.dotsContainerStyle)
    }, [
      createElementVNode("div", {
        class: "big-dot",
        style: normalizeStyle($options.biggerDotStyle)
      }, [
        createElementVNode("div", {
          class: "small-dot",
          style: normalizeStyle($options.smallerDotStyle)
        }, null, 4)
      ], 4)
    ], 4)
  ], 4))
}
var FlowerSpinner = /*#__PURE__*/_export_sfc(_sfc_main, [['render',_sfc_render]]);

export { FlowerSpinner };
