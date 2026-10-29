import '../../style/FingerprintSpinner.css';
import { openBlock, createElementBlock, normalizeStyle, Fragment, renderList } from 'vue';
import { _export_sfc } from '../plugin-vue_export-helper.js';

var FingerprintSpinner_vue_vue_type_style_index_0_lang = '';

const _sfc_main = {
  name: 'FingerprintSpinner',

  props: {
    animationDuration: {
      type: Number,
      default: 1500,
    },
    size: {
      type: Number,
      default: 60,
    },
    color: {
      type: String,
      default: '#fff',
    },
  },

  data() {
    return {
      ringsNum: 9,
      containerPadding: 2,
    }
  },

  computed: {
    outerRingSize() {
      return this.size - this.containerPadding * 2
    },

    spinnerStyle() {
      return {
        height: `${this.size}px`,
        width: `${this.size}px`,
        padding: `${this.containerPadding}px`,
      }
    },

    ringStyle() {
      return {
        borderTopColor: this.color,
        animationDuration: `${this.animationDuration}ms`,
      }
    },

    ringsStyles() {
      const ringsStyles = [];
      const ringBase = this.outerRingSize / this.ringsNum;
      const ringInc = ringBase;

      for (let i = 1; i <= this.ringsNum; i++) {
        const style = Object.assign(
          {
            animationDelay: `${i * 50}ms`,
            height: `${ringBase + (i - 1) * ringInc}px`,
            width: `${ringBase + (i - 1) * ringInc}px`,
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
    class: "fingerprint-spinner",
    style: normalizeStyle($options.spinnerStyle)
  }, [
    (openBlock(true), createElementBlock(Fragment, null, renderList($options.ringsStyles, (rs, index) => {
      return (openBlock(), createElementBlock("div", {
        key: index,
        class: "spinner-ring",
        style: normalizeStyle(rs)
      }, null, 4))
    }), 128))
  ], 4))
}
var FingerprintSpinner = /*#__PURE__*/_export_sfc(_sfc_main, [['render',_sfc_render]]);

export { FingerprintSpinner };
