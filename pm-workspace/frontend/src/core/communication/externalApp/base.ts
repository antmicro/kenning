/* eslint-disable semi */

import { JSONRPCRequest } from 'json-rpc-2.0';

export type EndpointType = 'external-api' | 'backend-api';

export default interface ExternalApp {
    isConnected(): Promise<boolean>;

    /**
     * Connection initialization hook.
     *
     * @returns True if the connection initialized successfully.
     */
    onConnect(): Promise<boolean>;

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    request(data: JSONRPCRequest, endpoint: EndpointType): void;
}
