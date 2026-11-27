import '../../style/SpringSpinner.css';
import { utils } from '../services/utils.js';
import { openBlock, createElementBlock, normalizeStyle, createElementVNode } from 'vue';
import { _export_sfc } from '../plugin-vue_export-helper.js';

var SpringSpinner_vue_vue_type_style_index_0_lang = '';

const _sfc_main = {
  name: 'SpringSpinner',

  props: {
    animationDuration: {
      type: Number,
      default: 3000,
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
      animationName: `spring-spinner-animation-${Date.now()}`,
    }
  },

  computed: {
    spinnerStyle() {
      return {
        height: `${this.size}px`,
        width: `${this.size}px`,
      }
    },

    spinnerPartStyle() {
      return {
        height: `${this.size / 2}px`,
        width: `${this.size}px`,
      }
    },

    rotatorStyle() {
      return {
        height: `${this.size}px`,
        width: `${this.size}px`,
        borderRightColor: this.color,
        borderTopColor: this.color,
        borderWidth: `${this.size / 7}px`,
        animationDuration: `${this.animationDuration}ms`,
        animationName: this.animationName,
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
        0% {
          border-width: ${this.size / 7}px;
        }
        25% {
          border-width: ${this.size / 23.33}px;
        }
        50% {
          transform: rotate(115deg);
          border-width: ${this.size / 7}px;
        }
        75% {
          border-width: ${this.size / 23.33}px;
          }
        100% {
          border-width: ${this.size / 7}px;
        }`
    },
  },
};

function _sfc_render(_ctx, _cache, $props, $setup, $data, $options) {
  return (openBlock(), createElementBlock("div", {
    class: "spring-spinner",
    style: normalizeStyle($options.spinnerStyle)
  }, [
    createElementVNode("div", {
      class: "spring-spinner-part top",
      style: normalizeStyle($options.spinnerPartStyle)
    }, [
      createElementVNode("div", {
        class: "spring-spinner-rotator",
        style: normalizeStyle($options.rotatorStyle)
      }, null, 4)
    ], 4),
    createElementVNode("div", {
      class: "spring-spinner-part bottom",
      style: normalizeStyle($options.spinnerPartStyle)
    }, [
      createElementVNode("div", {
        class: "spring-spinner-rotator",
        style: normalizeStyle($options.rotatorStyle)
      }, null, 4)
    ], 4)
  ], 4))
}
var SpringSpinner = /*#__PURE__*/_export_sfc(_sfc_main, [['render',_sfc_render]]);

export { SpringSpinner };
