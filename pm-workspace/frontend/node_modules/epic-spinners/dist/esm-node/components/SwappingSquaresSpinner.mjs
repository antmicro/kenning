import { openBlock, createElementBlock, normalizeStyle, Fragment, renderList, normalizeClass } from 'vue';
import { _export_sfc } from '../plugin-vue_export-helper.mjs';

var SwappingSquaresSpinner_vue_vue_type_style_index_0_lang = '';

const _sfc_main = {
  name: 'SwappingSquaresSpinner',

  props: {
    animationDuration: {
      type: Number,
      default: 1000,
    },
    size: {
      type: Number,
      default: 65,
    },
    color: {
      type: String,
      default: '#fff',
    },
  },

  data() {
    return {
      animationBaseName: 'swapping-squares-animation-child',
      squaresNum: 4,
    }
  },

  computed: {
    spinnerStyle() {
      return {
        height: `${this.size}px`,
        width: `${this.size}px`,
      }
    },

    squareStyle() {
      return {
        height: `${(this.size * 0.25) / 1.3}px`,
        width: `${(this.size * 0.25) / 1.3}px`,
        animationDuration: `${this.animationDuration}ms`,
        borderWidth: `${(this.size * 0.04) / 1.3}px`,
        borderColor: this.color,
      }
    },

    squaresStyles() {
      const squaresStyles = [];
      const delay = this.animationDuration * 0.5;

      for (let i = 1; i <= this.squaresNum; i++) {
        squaresStyles.push(
          Object.assign(
            {
              animationDelay: `${i % 2 === 0 ? delay : 0}ms`,
            },
            this.squareStyle,
          ),
        );
      }

      return squaresStyles
    },
  },
};

function _sfc_render(_ctx, _cache, $props, $setup, $data, $options) {
  return (openBlock(), createElementBlock("div", {
    class: "swapping-squares-spinner",
    style: normalizeStyle($options.spinnerStyle)
  }, [
    (openBlock(true), createElementBlock(Fragment, null, renderList($options.squaresStyles, (ss, index) => {
      return (openBlock(), createElementBlock("div", {
        key: index,
        class: normalizeClass(["square", `square-${index + 1}`]),
        style: normalizeStyle(ss)
      }, null, 6))
    }), 128))
  ], 4))
}
var SwappingSquaresSpinner = /*#__PURE__*/_export_sfc(_sfc_main, [['render',_sfc_render]]);

export { SwappingSquaresSpinner };
