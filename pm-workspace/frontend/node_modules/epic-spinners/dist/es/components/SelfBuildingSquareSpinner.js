import '../../style/SelfBuildingSquareSpinner.css';
import { openBlock, createElementBlock, normalizeStyle, Fragment, renderList, normalizeClass } from 'vue';
import { _export_sfc } from '../plugin-vue_export-helper.js';

var SelfBuildingSquareSpinner_vue_vue_type_style_index_0_lang = '';

const _sfc_main = {
  name: 'SelfBuildingSquareSpinner',

  props: {
    animationDuration: {
      type: Number,
      default: 6000,
    },
    size: {
      type: Number,
      default: 40,
    },
    color: {
      type: String,
      default: '#fff',
    },
  },

  data() {
    return {
      squaresNum: 9,
    }
  },

  computed: {
    squareSize() {
      return this.size / 4
    },

    initialTopPosition() {
      return (-this.squareSize * 2) / 3
    },

    spinnerStyle() {
      return {
        top: `${-this.initialTopPosition}px`,
        height: `${this.size}px`,
        width: `${this.size}px`,
      }
    },

    squareStyle() {
      return {
        height: `${this.squareSize}px`,
        width: `${this.squareSize}px`,
        top: `${this.initialTopPosition}px`,
        marginRight: `${this.squareSize / 3}px`,
        marginTop: `${this.squareSize / 3}px`,
        animationDuration: `${this.animationDuration}ms`,
        background: this.color,
      }
    },

    squaresStyles() {
      const squaresStyles = [];
      const delaysMultipliers = [6, 7, 8, 3, 4, 5, 0, 1, 2];
      const delayModifier = this.animationDuration * 0.05;

      for (let i = 0; i < this.squaresNum; i++) {
        squaresStyles.push(
          Object.assign(
            {
              animationDelay: `${delayModifier * delaysMultipliers[i]}ms`,
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
    class: "self-building-square-spinner",
    style: normalizeStyle($options.spinnerStyle)
  }, [
    (openBlock(true), createElementBlock(Fragment, null, renderList($options.squaresStyles, (ss, index) => {
      return (openBlock(), createElementBlock("div", {
        key: index,
        class: normalizeClass(["square", { clear: index && index % 3 === 0 }]),
        style: normalizeStyle(ss)
      }, null, 6))
    }), 128))
  ], 4))
}
var SelfBuildingSquareSpinner = /*#__PURE__*/_export_sfc(_sfc_main, [['render',_sfc_render]]);

export { SelfBuildingSquareSpinner };
