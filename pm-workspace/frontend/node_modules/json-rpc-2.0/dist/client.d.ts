import { JSONRPCErrorResponse, JSONRPCID, JSONRPCParams, JSONRPCRequest, JSONRPCResponse } from "./models";
export type SendRequest<ClientParams> = (payload: any, clientParams: ClientParams) => PromiseLike<void> | void;
export type CreateID = () => JSONRPCID;
export interface JSONRPCRequester<ClientParams> {
    request(method: string, params?: JSONRPCParams, clientParams?: ClientParams): PromiseLike<any>;
    requestAdvanced(request: JSONRPCRequest, clientParams?: ClientParams): PromiseLike<JSONRPCResponse>;
    requestAdvanced(request: JSONRPCRequest[], clientParams?: ClientParams): PromiseLike<JSONRPCResponse[]>;
}
export declare class JSONRPCClient<ClientParams = void> implements JSONRPCRequester<ClientParams> {
    private _send;
    private createID?;
    private idToResolveMap;
    private id;
    constructor(_send: SendRequest<ClientParams>, createID?: CreateID | undefined);
    private _createID;
    timeout(delay: number, overrideCreateJSONRPCErrorResponse?: (id: JSONRPCID) => JSONRPCErrorResponse): JSONRPCRequester<ClientParams>;
    request(method: string, params: JSONRPCParams, clientParams: ClientParams): PromiseLike<any>;
    private requestWithID;
    requestAdvanced(request: JSONRPCRequest, clientParams: ClientParams): PromiseLike<JSONRPCResponse>;
    requestAdvanced(request: JSONRPCRequest[], clientParams: ClientParams): PromiseLike<JSONRPCResponse[]>;
    notify(method: string, params: JSONRPCParams, clientParams: ClientParams): void;
    send(payload: any, clientParams: ClientParams): Promise<void>;
    rejectAllPendingRequests(message: string): void;
    receive(responses: JSONRPCResponse | JSONRPCResponse[]): void;
}
