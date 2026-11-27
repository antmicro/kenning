import { JSONRPCServerAndClient } from 'json-rpc-2.0';
import NotificationHandler from '../notifications.js';
import ExternalApp from './externalApp/base';
import { ClientParams } from './utils';

export default class ConnectionManager {
    private isRunning = false;

    public connected: Map<ExternalApp, boolean> = new Map();

    // eslint-disable-next-line no-useless-constructor
    public constructor(
        public connectionHook: (externalApp: ExternalApp) => void,
        public disconnectionHook: (externalApp: ExternalApp) => void,
    ) {}

    public add(externalApp: ExternalApp) { this.connected.set(externalApp, false); }

    public remove(externalApp: ExternalApp) { return this.connected.delete(externalApp); }

    public async poll(interval = 1500) {
        this.isRunning = true;

        const updating = new Set<ExternalApp>();
        while (this.isRunning) {
            Array.from(this.connected.keys())
                .filter((externalApp) => !updating.has(externalApp))
                .forEach((externalApp) => {
                    updating.add(externalApp);
                    this.updateConnection(externalApp).then(() => updating.delete(externalApp));
                });

            // eslint-disable-next-line no-await-in-loop
            await new Promise((resolve) => { setTimeout(resolve, interval); });
        }
    }

    private async updateConnection(externalApp: ExternalApp) {
        try {
            const wasConnected = this.connected.get(externalApp);
            let isConnected = await externalApp.isConnected();

            // If the application was connected and the connection was lost, a warning is displayed.
            if (!isConnected && wasConnected) {
                NotificationHandler.terminalLog('warning', 'External application was disconnected', undefined);
                this.disconnectionHook(externalApp);
            }

            if (isConnected && !wasConnected) {
                if (await externalApp.onConnect()) {
                    this.connectionHook(externalApp);
                } else {
                    isConnected = false;
                }
            }

            this.connected.set(externalApp, isConnected);
        } catch (error) {
            this.connected.set(externalApp, false);
        }
    }

    public pollStop() { this.isRunning = false; }
}
