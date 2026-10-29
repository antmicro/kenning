import { openBlock, createElementBlock, normalizeStyle, Fragment, renderList, normalizeClass, createElementVNode } from 'vue';
import { _export_sfc } from '../plugin-vue_export-helper.mjs';

var BreedingRhombusSpinner_vue_vue_type_style_index_0_lang = '';

const _sfc_main = {
  name: 'BreedingRhombusSpinner',

  props: {
    animationDuration: {
      type: Number,
      default: 2000,
    },
    size: {
      type: Number,
      default: 150,
    },
    color: {
      type: String,
      default: '#fff',
    },
  },

  data() {
    return {
      animationBaseName: 'breeding-rhombus-spinner-animation-child',
      rhombusesNum: 8,
    }
  },

  computed: {
    spinnerStyle() {
      return {
        height: `${this.size}px`,
        width: `${this.size}px`,
      }
    },

    rhombusStyle() {
      return {
        height: `${this.size / 7.5}px`,
        width: `${this.size / 7.5}px`,
        animationDuration: `${this.animationDuration}ms`,
        top: `${this.size / 2.3077}px`,
        left: `${this.size / 2.3077}px`,
        backgroundColor: this.color,
      }
    },

    rhombusesStyles() {
      const rhombusesStyles = [];
      const delayModifier = this.animationDuration * 0.05;

      for (let i = 1; i <= this.rhombusesNum; i++) {
        rhombusesStyles.push(
          Object.assign(
            {
              animationDelay: `${delayModifier * (i + 1)}ms`,
            },
            this.rhombusStyle,
          ),
        );
      }

      return rhombusesStyles
    },

    bigRhombusStyle() {
      return {
        height: `${this.size / 3}px`,
        width: `${this.size / 3}px`,
        animationDuration: `${this.animationDuration}`,
        top: `${this.size / 3}px`,
        left: `${this.size / 3}px`,
        backgroundColor: this.color,
      }
    },
  },
};

function _sfc_render(_ctx, _cache, $props, $setup, $data, $options) {
  return (openBlock(), createElementBlock("div", {
    class: "breeding-rhombus-spinner",
    style: normalizeStyle($options.spinnerStyle)
  }, [
    (openBlock(true), createElementBlock(Fragment, null, renderList($options.rhombusesStyles, (rs, index) => {
      return (openBlock(), createElementBlock("div", {
        key: index,
        class: normalizeClass(["rhombus", `child-${index + 1}`]),
        style: normalizeStyle(rs)
      }, null, 6))
    }), 128)),
    createElementVNode("div", {
      class: "rhombus big",
      style: normalizeStyle($options.bigRhombusStyle)
    }, null, 4)
  ], 4))
}
var BreedingRhombusSpinner = /*#__PURE__*/_export_sfc(_sfc_main, [['render',_sfc_render]]);

export { BreedingRhombusSpinner };
