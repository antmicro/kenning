import '../../style/CirclesToRhombusesSpinner.css';
import { openBlock, createElementBlock, normalizeStyle, Fragment, renderList } from 'vue';
import { _export_sfc } from '../plugin-vue_export-helper.js';

var CirclesToRhombusesSpinner_vue_vue_type_style_index_0_lang = '';

const _sfc_main = {
  name: 'CirclesToRhombusesSpinner',

  props: {
    animationDuration: {
      type: Number,
      default: 1200,
    },
    circleSize: {
      type: Number,
      default: 15,
    },
    color: {
      type: String,
      default: '#fff',
    },
    circlesNum: {
      type: Number,
      default: 3,
    },
  },

  computed: {
    circleMarginLeft() {
      return this.circleSize * 1.125
    },

    spinnerStyle() {
      return {
        height: `${this.circleSize}px`,
        width: `${(this.circleSize + this.circleMarginLeft) * this.circlesNum}px`,
      }
    },

    circleStyle() {
      return {
        borderColor: this.color,
        animationDuration: `${this.animationDuration}ms`,
        height: `${this.circleSize}px`,
        width: `${this.circleSize}px`,
        marginLeft: `${this.circleMarginLeft}px`,
      }
    },

    circlesStyles() {
      const circlesStyles = [];
      const delay = 150;

      for (let i = 1; i <= this.circlesNum; i++) {
        const style = Object.assign(
          {
            animationDelay: `${i * delay}ms`,
          },
          this.circleStyle,
        );

        if (i === 1) {
          style.marginLeft = 0;
        }

        circlesStyles.push(style);
      }

      return circlesStyles
    },
  },
};

function _sfc_render(_ctx, _cache, $props, $setup, $data, $options) {
  return (openBlock(), createElementBlock("div", {
    class: "circles-to-rhombuses-spinner",
    style: normalizeStyle($options.spinnerStyle)
  }, [
    (openBlock(true), createElementBlock(Fragment, null, renderList($options.circlesStyles, (cs, index) => {
      return (openBlock(), createElementBlock("div", {
        key: index,
        class: "circle",
        style: normalizeStyle(cs)
      }, null, 4))
    }), 128))
  ], 4))
}
var CirclesToRhombusesSpinner = /*#__PURE__*/_export_sfc(_sfc_main, [['render',_sfc_render]]);

export { CirclesToRhombusesSpinner };
