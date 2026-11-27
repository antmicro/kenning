import { openBlock, createElementBlock, normalizeStyle, Fragment, renderList } from 'vue';
import { _export_sfc } from '../plugin-vue_export-helper.mjs';

var SemipolarSpinner_vue_vue_type_style_index_0_lang = '';

const _sfc_main = {
  name: 'SemipolarSpinner',

  props: {
    animationDuration: {
      type: Number,
      default: 2000,
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
      ringsNum: 5,
    }
  },

  computed: {
    spinnerStyle() {
      return {
        height: `${this.size}px`,
        width: `${this.size}px`,
      }
    },
    ringStyle() {
      return {
        animationDuration: `${this.animationDuration}ms`,
        borderTopColor: this.color,
        borderLeftColor: this.color,
      }
    },
    ringsStyles() {
      const ringsStyles = [];
      const delayModifier = 0.1;
      const ringWidth = this.size * 0.05;
      const positionIncrement = ringWidth * 2;
      const sizeDecrement = this.size * 0.2;

      for (let i = 0; i < this.ringsNum; i++) {
        const computedSize = `${this.size - sizeDecrement * i}px`;
        const computedPosition = `${positionIncrement * i}px`;
        const style = Object.assign(
          {
            animationDelay: `${this.animationDuration * delayModifier * (this.ringsNum - i - 1)}ms`,
            height: computedSize,
            width: computedSize,
            left: computedPosition,
            top: computedPosition,
            borderWidth: `${ringWidth}px`,
          },
          this.ringStyle,
        );
        ringsStyles.push(style);
      }

      return ringsStyles
    },
  },
};

function _sfc_render(_ctx, _cache, $props, $setup, $data, $options) {
  return (openBlock(), createElementBlock("div", {
    class: "semipolar-spinner",
    style: normalizeStyle($options.spinnerStyle)
  }, [
    (openBlock(true), createElementBlock(Fragment, null, renderList($options.ringsStyles, (rs, index) => {
      return (openBlock(), createElementBlock("div", {
        key: index,
        class: "ring",
        style: normalizeStyle(rs)
      }, null, 4))
    }), 128))
  ], 4))
}
var SemipolarSpinner = /*#__PURE__*/_export_sfc(_sfc_main, [['render',_sfc_render]]);

export { SemipolarSpinner };
