<!--
Copyright (c) 2022-2024 Antmicro <www.antmicro.com>

SPDX-License-Identifier: Apache-2.0
-->

<!--
List of notitications that creates a `Notification` component for each message registered.
-->

<template>
    <div class="notifications" tabindex="-1">
        <div class="info">
            <span>Notifications ({{ notificationStore.notifications.length }})</span>
            <button @click="this.removeAll" tabindex="-1">
                <Cross />
                Clear all
            </button>
        </div>
        <div class="panel">
            <ul>
                <!-- Iterating in reverse so that the newest messages are rendered first -->
                <li v-for="index in notificationsLength" :key="index">
                    <Notification
                        :type="notifications[notificationsLength - index].type"
                        :message="`${notifications[notificationsLength - index].message}`"
                        :timestamp="`${notifications[notificationsLength - index].timestamp}`"
                        :index="notificationsLength - index"
                        border="border"
                    />
                </li>
            </ul>
        </div>
    </div>
</template>

<script>
import Cross from '../icons/Cross.vue';
import Notification from './Notification.vue';
import { notificationStore } from '../core/stores';

export default {
    components: {
        Cross,
        Notification,
    },
    computed: {
        notifications() {
            return notificationStore.notifications;
        },
        notificationsLength() {
            return notificationStore.notifications.length;
        },
    },
    data() {
        return {
            notificationStore,
        };
    },
    methods: {
        removeAll() {
            notificationStore.remove();
        },
    },
};
</script>

<style lang="scss" scoped>
.notifications {
    $notification-maxwidth: calc(100vw - 2 * $spacing-xxl);
    /* Set height to 100 view port minus height of navigation bar and paddings
    and minus the terminal panel */
    height: calc(100% - 120px - $terminal-container-height);
    width: 435px;
    max-width: $notification-maxwidth;
    background-color: $gray-600;
    opacity: 0.9;
    position: absolute;
    /* Height of navigation bar*/
    top: $navbar-height;
    /* As default notification panel is hidden (minus value)
      panel width (435px) + 2 * padding (30px) = 495px
    */
    right: -495px;
    z-index: 4;
    padding: $spacing-xxl;
    overflow-y: auto;

    // Prevent selection of text
    -webkit-user-select: none; /* Safari */
    -ms-user-select: none; /* IE 10 and IE 11 */
    user-select: none; /* Standard syntax */

    & > .info {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: $spacing-l;

        & > span {
            color: $white;
            font-size: $fs-large;
        }

        & > button {
            color: $gray-200;
            font-size: $fs-medium;
        }
    }

    & > .panel {
        display: grid;
        grid-row-gap: $spacing-l;
        ul {
            max-width: $notification-maxwidth;
        }
        li {
            -webkit-user-select: text; /* Safari */
            -ms-user-select: text; /* IE 10 and IE 11 */
            user-select: text; /* Standard syntax */
        }
    }
}
</style>
