import { JSONRPCMethod, JSONRPCServer, JSONRPCServerMiddleware, SimpleJSONRPCMethod } from "./server";
import { JSONRPCClient, JSONRPCRequester } from "./client";
import { ErrorListener, JSONRPCParams, JSONRPCRequest, JSONRPCResponse } from "./models";
export interface JSONRPCServerAndClientOptions {
    errorListener?: ErrorListener;
}
export declare class JSONRPCServerAndClient<ServerParams = void, ClientParams = void> {
    server: JSONRPCServer<ServerParams>;
    client: JSONRPCClient<ClientParams>;
    private readonly errorListener;
    constructor(server: JSONRPCServer<ServerParams>, client: JSONRPCClient<ClientParams>, options?: JSONRPCServerAndClientOptions);
    applyServerMiddleware(...middlewares: JSONRPCServerMiddleware<ServerParams>[]): void;
    hasMethod(name: string): boolean;
    addMethod(name: string, method: SimpleJSONRPCMethod<ServerParams>): void;
    addMethodAdvanced(name: string, method: JSONRPCMethod<ServerParams>): void;
    timeout(delay: number): JSONRPCRequester<ClientParams>;
    request(method: string, params: JSONRPCParams, clientParams: ClientParams): PromiseLike<any>;
    requestAdvanced(jsonRPCRequest: JSONRPCRequest, clientParams: ClientParams): PromiseLike<JSONRPCResponse>;
    requestAdvanced(jsonRPCRequest: JSONRPCRequest[], clientParams: ClientParams): PromiseLike<JSONRPCResponse[]>;
    notify(method: string, params: JSONRPCParams, clientParams: ClientParams): void;
    rejectAllPendingRequests(message: string): void;
    receiveAndSend(payload: any, serverParams: ServerParams, clientParams: ClientParams): Promise<void>;
}
