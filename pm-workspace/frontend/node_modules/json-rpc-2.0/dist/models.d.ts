export type JSONRPC = "2.0";
export declare const JSONRPC: JSONRPC;
export type JSONRPCID = string | number | null;
export type JSONRPCParams = any;
export declare const isJSONRPCID: (id: any) => id is JSONRPCID;
export interface JSONRPCRequest {
    jsonrpc: JSONRPC;
    method: string;
    params?: JSONRPCParams;
    id?: JSONRPCID;
}
export type JSONRPCResponse = JSONRPCSuccessResponse | JSONRPCErrorResponse;
export interface JSONRPCSuccessResponse {
    jsonrpc: JSONRPC;
    id: JSONRPCID;
    result: any;
    error?: undefined;
}
export interface JSONRPCErrorResponse {
    jsonrpc: JSONRPC;
    id: JSONRPCID;
    result?: undefined;
    error: JSONRPCError;
}
export declare const isJSONRPCRequest: (payload: any) => payload is JSONRPCRequest;
export declare const isJSONRPCRequests: (payload: any) => payload is JSONRPCRequest[];
export declare const isJSONRPCResponse: (payload: any) => payload is JSONRPCResponse;
export declare const isJSONRPCResponses: (payload: any) => payload is JSONRPCResponse[];
export interface JSONRPCError {
    code: number;
    message: string;
    data?: any;
}
export declare class JSONRPCErrorException extends Error implements JSONRPCError {
    code: number;
    data?: any;
    constructor(message: string, code: number, data?: any);
    toObject(): JSONRPCError;
}
export declare enum JSONRPCErrorCode {
    ParseError = -32700,
    InvalidRequest = -32600,
    MethodNotFound = -32601,
    InvalidParams = -32602,
    InternalError = -32603
}
export declare const createJSONRPCErrorResponse: (id: JSONRPCID, code: number, message: string, data?: any) => JSONRPCErrorResponse;
export declare const createJSONRPCSuccessResponse: (id: JSONRPCID, result?: any) => JSONRPCSuccessResponse;
export declare const createJSONRPCRequest: (id: JSONRPCID, method: string, params?: JSONRPCParams) => JSONRPCRequest;
export declare const createJSONRPCNotification: (method: string, params?: JSONRPCParams) => JSONRPCRequest;
export type ErrorListener = (message: string, data: unknown) => void;
