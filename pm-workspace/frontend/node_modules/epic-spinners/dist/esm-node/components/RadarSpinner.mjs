import { openBlock, createElementBlock, normalizeStyle, Fragment, renderList, createElementVNode } from 'vue';
import { _export_sfc } from '../plugin-vue_export-helper.mjs';

var RadarSpinner_vue_vue_type_style_index_0_lang = '';

const _sfc_main = {
  name: 'RadarSpinner',

  props: {
    animationDuration: {
      type: Number,
      default: 2000,
    },
    size: {
      type: Number,
      default: 110,
    },
    color: {
      type: String,
      default: '#fff',
    },
  },

  data() {
    return {
      circlesNum: 4,
    }
  },

  computed: {
    borderWidth() {
      return (this.size * 5) / 110
    },

    spinnerStyle() {
      return {
        height: `${this.size}px`,
        width: `${this.size}px`,
      }
    },

    circleStyle() {
      return {
        animationDuration: `${this.animationDuration}ms`,
      }
    },

    circleInnerContainerStyle() {
      return {
        borderWidth: `${this.borderWidth}px`,
      }
    },

    circleInnerStyle() {
      return {
        borderLeftColor: this.color,
        borderRightColor: this.color,
        borderWidth: `${this.borderWidth}px`,
      }
    },

    circlesStyles() {
      const circlesStyles = [];
      const delay = this.animationDuration * 0.15;

      for (let i = 0; i < this.circlesNum; i++) {
        circlesStyles.push(
          Object.assign(
            {
              padding: `${this.borderWidth * 2 * i}px`,
              animationDelay: `${i === this.circlesNum - 1 ? 0 : delay}ms`,
            },
            this.circleStyle,
          ),
        );
      }

      return circlesStyles
    },
  },
};

function _sfc_render(_ctx, _cache, $props, $setup, $data, $options) {
  return (openBlock(), createElementBlock("div", {
    class: "radar-spinner",
    style: normalizeStyle($options.spinnerStyle)
  }, [
    (openBlock(true), createElementBlock(Fragment, null, renderList($options.circlesStyles, (cs, index) => {
      return (openBlock(), createElementBlock("div", {
        key: index,
        class: "circle",
        style: normalizeStyle(cs)
      }, [
        createElementVNode("div", {
          class: "circle-inner-container",
          style: normalizeStyle($options.circleInnerContainerStyle)
        }, [
          createElementVNode("div", {
            class: "circle-inner",
            style: normalizeStyle($options.circleInnerStyle)
          }, null, 4)
        ], 4)
      ], 4))
    }), 128))
  ], 4))
}
var RadarSpinner = /*#__PURE__*/_export_sfc(_sfc_main, [['render',_sfc_render]]);

export { RadarSpinner };
