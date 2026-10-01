import '../../style/IntersectingCirclesSpinner.css';
import { openBlock, createElementBlock, normalizeStyle, createElementVNode, Fragment, renderList } from 'vue';
import { _export_sfc } from '../plugin-vue_export-helper.js';

var IntersectingCirclesSpinner_vue_vue_type_style_index_0_lang = '';

const _sfc_main = {
  name: 'IntersectingCirclesSpinner',

  props: {
    animationDuration: {
      type: Number,
      default: 1200,
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

  computed: {
    circleSize() {
      return this.size / 2
    },

    spinnerStyle() {
      return {
        width: `${this.size}px`,
        height: `${this.size}px`,
      }
    },

    spinnerBlockStyle() {
      return {
        animationDuration: `${this.animationDuration}ms`,
        width: `${this.circleSize}px`,
        height: `${this.circleSize}px`,
      }
    },

    circleStyle() {
      return {
        borderColor: this.color,
      }
    },

    circleStyles() {
      const circlesPositions = [
        { top: 0, left: 0 },
        { left: `${this.circleSize * -0.36}px`, top: `${this.circleSize * 0.2}px` },
        { left: `${this.circleSize * -0.36}px`, top: `${this.circleSize * -0.2}px` },
        { left: 0, top: `${this.circleSize * -0.36}px` },
        { left: `${this.circleSize * 0.36}px`, top: `${this.circleSize * -0.2}px` },
        { left: `${this.circleSize * 0.36}px`, top: `${this.circleSize * 0.2}px` },
        { left: 0, top: `${this.circleSize * 0.36}px` },
      ];

      return circlesPositions.map((cp) => Object.assign(cp, this.circleStyle))
    },
  },
};

function _sfc_render(_ctx, _cache, $props, $setup, $data, $options) {
  return (openBlock(), createElementBlock("div", {
    class: "intersecting-circles-spinner",
    style: normalizeStyle($options.spinnerStyle)
  }, [
    createElementVNode("div", {
      class: "spinnerBlock",
      style: normalizeStyle($options.spinnerBlockStyle)
    }, [
      (openBlock(true), createElementBlock(Fragment, null, renderList($options.circleStyles, (cs, index) => {
        return (openBlock(), createElementBlock("span", {
          key: index,
          class: "circle",
          style: normalizeStyle(cs)
        }, null, 4))
      }), 128))
    ], 4)
  ], 4))
}
var IntersectingCirclesSpinner = /*#__PURE__*/_export_sfc(_sfc_main, [['render',_sfc_render]]);

export { IntersectingCirclesSpinner };
