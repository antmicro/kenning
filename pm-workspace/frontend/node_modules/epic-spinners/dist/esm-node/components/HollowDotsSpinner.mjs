import { openBlock, createElementBlock, normalizeStyle, Fragment, renderList } from 'vue';
import { _export_sfc } from '../plugin-vue_export-helper.mjs';

var HollowDotsSpinner_vue_vue_type_style_index_0_lang = '';

const _sfc_main = {
  name: 'HollowDotsSpinner',

  props: {
    animationDuration: {
      type: Number,
      default: 1000,
    },
    dotSize: {
      type: Number,
      default: 15,
    },
    dotsNum: {
      type: Number,
      default: 3,
    },
    color: {
      type: String,
      default: '#fff',
    },
  },

  computed: {
    horizontalMargin() {
      return this.dotSize / 2
    },

    spinnerStyle() {
      return {
        height: `${this.dotSize}px`,
        width: `${(this.dotSize + this.horizontalMargin * 2) * this.dotsNum}px`,
      }
    },

    dotStyle() {
      return {
        animationDuration: `${this.animationDuration}ms`,
        width: `${this.dotSize}px`,
        height: `${this.dotSize}px`,
        margin: `0 ${this.horizontalMargin}px`,
        borderWidth: `${this.dotSize / 5}px`,
        borderColor: this.color,
      }
    },

    dotsStyles() {
      const dotsStyles = [];
      const delayModifier = 0.3;
      const basicDelay = this.animationDuration;

      for (let i = 1; i <= this.dotsNum; i++) {
        dotsStyles.push({
          animationDelay: `${basicDelay * i * delayModifier}ms`,
          ...this.dotStyle,
        });
      }

      return dotsStyles
    },
  },
};

function _sfc_render(_ctx, _cache, $props, $setup, $data, $options) {
  return (openBlock(), createElementBlock("div", {
    class: "hollow-dots-spinner",
    style: normalizeStyle($options.spinnerStyle)
  }, [
    (openBlock(true), createElementBlock(Fragment, null, renderList($options.dotsStyles, (ds, index) => {
      return (openBlock(), createElementBlock("div", {
        key: index,
        class: "dot",
        style: normalizeStyle(ds)
      }, null, 4))
    }), 128))
  ], 4))
}
var HollowDotsSpinner = /*#__PURE__*/_export_sfc(_sfc_main, [['render',_sfc_render]]);

export { HollowDotsSpinner };
