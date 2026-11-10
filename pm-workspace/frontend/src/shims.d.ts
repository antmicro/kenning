// eslint-disable-next-line @typescript-eslint/no-unused-vars
import { ComponentCustomProperties } from 'vue';

declare module '@vue/runtime-core' {
  // eslint-disable-next-line no-shadow
  interface ComponentCustomProperties {
    $isMobile: boolean;
  }
}
