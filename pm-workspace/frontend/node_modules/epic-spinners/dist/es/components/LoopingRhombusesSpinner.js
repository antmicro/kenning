import '../../style/LoopingRhombusesSpinner.css';
import { openBlock, createElementBlock, normalizeStyle, Fragment, renderList } from 'vue';
import { _export_sfc } from '../plugin-vue_export-helper.js';

var LoopingRhombusesSpinner_vue_vue_type_style_index_0_lang = '';

const _sfc_main = {
  name: 'LoopingRhombusesSpinner',

  props: {
    animationDuration: {
      type: Number,
      default: 2500,
    },
    rhombusSize: {
      type: Number,
      default: 15,
    },
    color: {
      type: String,
      default: '#fff',
    },
  },

  data() {
    return {
      rhombusesNum: 3,
    }
  },

  computed: {
    spinnerStyle() {
      return {
        height: `${this.rhombusSize}px`,
        width: `${this.rhombusSize * 4}px`,
      }
    },

    rhombusStyle() {
      return {
        height: `${this.rhombusSize}px`,
        width: `${this.rhombusSize}px`,
        backgroundColor: this.color,
        animationDuration: `${this.animationDuration}ms`,
        left: `${this.rhombusSize * 4}px`,
      }
    },

    rhombusesStyles() {
      const rhombusesStyles = [];
      const delay = -this.animationDuration / 1.5;

      for (let i = 1; i <= this.rhombusesNum; i++) {
        const style = Object.assign(
          {
            animationDelay: `${i * delay}ms`,
          },
          this.rhombusStyle,
        );

        rhombusesStyles.push(style);
      }

      return rhombusesStyles
    },
  },
};

function _sfc_render(_ctx, _cache, $props, $setup, $data, $options) {
  return (openBlock(), createElementBlock("div", {
    class: "looping-rhombuses-spinner",
    style: normalizeStyle($options.spinnerStyle)
  }, [
    (openBlock(true), createElementBlock(Fragment, null, renderList($options.rhombusesStyles, (rs, index) => {
      return (openBlock(), createElementBlock("div", {
        key: index,
        class: "rhombus",
        style: normalizeStyle(rs)
      }, null, 4))
    }), 128))
  ], 4))
}
var LoopingRhombusesSpinner = /*#__PURE__*/_export_sfc(_sfc_main, [['render',_sfc_render]]);

export { LoopingRhombusesSpinner };
