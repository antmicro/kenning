# vue3-click-outside

Vue 3 directive to react on clicks outside an element without stopping the event propagation.
Great for closing dialogues and menus among other things.

## Install
```bash
$ npm install --save click-outside-vue3
```

```bash
$ yarn add click-outside-vue3
```

## Use

```js
import { createApp } from "vue"
import App from "./App.vue"
import vClickOutside from "click-outside-vue3"

const app = createApp(App)
app.use(vClickOutside)
```

```js
<script>
  export default {
    data () {
      vcoConfig: {
        handler: this.handler,
        middleware: this.middleware,
        events: ['dblclick', 'click'],
        // Note: The default value is true, but in case you want to activate / deactivate
        //       this directive dynamically use this attribute.
        isActive: true,
        // Note: The default value is true. See "Detecting Iframe Clicks" section
        //       to understand why this behaviour is behind a flag.
        detectIFrame: true,
        // Note: The default value is false. Sets the capture option for EventTarget addEventListener method.
        //       Could be useful if some event's handler calls stopPropagation method preventing event bubbling.
        capture: false
      }
    },
    methods: {
      onClickOutside (event) {
        console.log('Clicked outside. Event: ', event)
      },

      handler (event) {
        console.log('Clicked outside (Using config), middleware returned true :)')
      },
      // Note: The middleware will be executed if the event was fired outside the element.
      //       It should have only sync functionality and it should return a boolean to
      //       define if the handler should be fire or not
      middleware (event) {
        return event.target.className !== 'modal'
      }
    }
  };
</script>

<template>
  <div v-click-outside="onClickOutside"></div>
  <div v-click-outside="vcoConfig"></div>
</template>
```
Or use it as a directive

```js
import vClickOutside from 'click-outside-vue3'

<script>
  export default {
    directives: {
      clickOutside: vClickOutside.directive
    },
    methods: {
      onClickOutside (event) {
        console.log('Clicked outside. Event: ', event)
      }
    }
  };
</script>

<template>
  <div v-click-outside="onClickOutside"></div>
</template>
```

## Detecting Iframe Clicks

To our knowledge, there isn't an idiomatic way to detect a click on a `<iframe>` (`HTMLIFrameElement`).
Clicks on iframes moves `focus` to its contents’ `window` but don't `bubble` up to main `window`, therefore not triggering our `document.documentElement` listeners. On the other hand, the abovementioned `focus` event does trigger a `window.blur` event on main `window` that we use in conjunction with `document.activeElement` to detect if it came from an `<iframe>`, and execute the provided `handler`.

**As with any workaround, this also has its caveats:**

- Click outside will be triggered once on iframe. Subsequent clicks on iframe will not execute the handler **until focus has been moved back to main window** — as in by clicking anywhere outside the iframe. This is the "expected" behaviour since, as mentioned before, by clicking the iframe focus will move to iframe contents — a different window, so subsequent clicks are inside its frame. There might be way to workaround this such as calling window.focus() at the end of the provided handler but that will break normal tab/focus flow;
- Moving focus to `iframe` via `keyboard` navigation also triggers `window.blur` consequently the handler - no workaround found ATM;

Because of these reasons, the detection mechansim is behind the `detectIframe` flag that you can optionally set to `false` if you find it conflicting with your use-case.
Any improvements or suggestions to this are welcomed.


## License

[MIT License](https://github.com/andymark-by/vue3-click-outside/blob/master/LICENSE)
