import {
    JSONRPC,
    JSONRPCError, JSONRPCID, JSONRPCRequest, JSONRPCResponse, JSONRPCServerAndClient,
    JSONRPCSuccessResponse,
} from 'json-rpc-2.0';
import { io, Socket } from 'socket.io-client';
import { v4 as uuidv4 } from 'uuid';
import { JSONRPCCustomErrorCode } from '../../utils';
import ExternalApp, { EndpointType } from './base';
import { ClientParams } from '../utils';
import NotificationHandler from '../../notifications';
import MessageTracker from './message_tracker';

import { CHUNKED_MESSAGE_TIMEOUT, MAX_CHUNK_SIZE } from '../config';

interface JSONRPCMaybeChunkedResponse extends JSONRPCSuccessResponse{
    chunk_id: string;
    chunk?: string;
    end?: boolean;
}

interface JSONRPCMaybeChunkedRequest extends JSONRPCRequest{
    chunk_id: string;
    chunk?: string;
    end?: boolean;
}

function* splitDataIntoChunks(data: any, chunkSize: number, id: string|undefined = undefined) {
    const chunkId = id ?? uuidv4();

    const msg = JSON.stringify(data);
    let i = 0;
    for (; i < msg.length - chunkSize; i += chunkSize) {
        const chunk = msg.substring(i, i + chunkSize);
        yield {
            chunk_id: chunkId,
            chunk,
        };
    }

    const lastChunk = msg.substring(i, i + chunkSize);

    yield {
        chunk_id: chunkId,
        chunk: lastChunk,
        end: true,
    };
}

export default class ExternalBackendApp implements ExternalApp {
    public static commonHeaders = {
        'Access-Control-Allow-Origin': 'http://localhost',
        'Access-Control-Allow-Headers': 'Origin, X-Requested-With, Content-Type, Accept',
    };

    private socket: Socket;

    private chunks: Map<string, MessageTracker>;

    // eslint-disable-next-line no-useless-constructor
    public constructor(
        public url: string,
        private jsonRPC: JSONRPCServerAndClient<void, ClientParams>,
        private maxMessageLength = MAX_CHUNK_SIZE,
    ) {
        this.socket = io(url, { extraHeaders: ExternalBackendApp.commonHeaders });

        this.socket.on('connect', () => NotificationHandler.terminalLog('info', 'Initialized connection with communication server', null));
        this.socket.on('disconnect', () => {
            NotificationHandler.terminalLog('warning', 'Connection with communication server disrupted', null);
            this.jsonRPC.rejectAllPendingRequests('WebSocket disconnected');
        });

        const shouldBeChunked = (data:any) => {
            const stringify = JSON.stringify(data);

            return new Blob([stringify]).size > maxMessageLength;
        };

        this.chunks = new Map<string, MessageTracker>();

        this.socket.on('api', async (request: JSONRPCMaybeChunkedRequest) => {
            const msg = this.processMessage(request) as JSONRPCMaybeChunkedRequest;
            if (msg === undefined) {
                return;
            }
            const response = await this.jsonRPC.server.receive(msg);

            if (response) {
                if (shouldBeChunked(response)) {
                    try {
                        const id = uuidv4();

                        await Promise.all(splitDataIntoChunks(response, maxMessageLength, id)
                            .map(async (chunk) => {
                                const ack = await this.socket.emitWithAck('external-api', chunk);
                                if (ack !== undefined && !ack) {
                                    NotificationHandler.terminalLog('error', 'Response to external app was not sent', null);
                                }
                            }));
                    } catch (error) {
                        NotificationHandler.terminalLog('error', `Response to ${msg.method} request cannot be sent`, error);
                    }
                } else {
                    const ack = await this.socket.emitWithAck('external-api', response);
                    if (ack !== undefined && !ack) {
                        NotificationHandler.terminalLog('error', 'Response to external app was not sent', null);
                    }
                }
            }
        });
        this.socket.on('api-response', (response: JSONRPCMaybeChunkedResponse) => {
            const msg = this.processMessage(response);
            if (msg !== undefined) {
                this.jsonRPC.client.receive(msg as JSONRPCMaybeChunkedResponse);
            }
        });
    }

    public cleanOldMessages() {
        const currentTime = Date.now() / 1000.0;
        const toRemove:string[] = [];

        this.chunks.forEach((value:MessageTracker, key:string) => {
            if ((currentTime - value.time) > CHUNKED_MESSAGE_TIMEOUT) {
                toRemove.push(key);
            }
        });

        toRemove.forEach((key:string) => {
            this.chunks.delete(key);
        });
    }

    public processMessage(msg: JSONRPCMaybeChunkedResponse|JSONRPCMaybeChunkedRequest) {
        this.cleanOldMessages();

        const isChunk = msg?.chunk_id !== undefined;
        if (isChunk) {
            const chunkId = msg.chunk_id;

            const currentChunks = this.chunks.get(chunkId) ?? new MessageTracker();

            const data = msg?.chunk;
            if (data !== undefined) {
                // update time
                currentChunks.time = Date.now() / 1000.0;
                currentChunks.data.push(data);
            }
            this.chunks.set(chunkId, currentChunks);

            // Is it final chunk?
            if (msg?.end) {
                const RawData = currentChunks.data.join('');
                try {
                    const parsed = JSON.parse(RawData) as JSONRPCResponse;
                    this.chunks.delete(chunkId);
                    return parsed;
                } catch (error) {
                    NotificationHandler.terminalLog('error', `Couldn't parse chunked response`, error);
                }
            }
        } else {
            return msg;
        }

        return undefined;
    }

    public async isConnected() {
        try {
            const { status: { connected } } = await this.jsonRPC.request('status_get', undefined, { externalApp: this });
            return connected;
        } catch {
            return false;
        }
    }

    public async onConnect() {
        try {
            await this.jsonRPC.request('external_app_connect', undefined, { externalApp: this });
            return true;
        } catch (unknownError) {
            const error = unknownError as JSONRPCError;
            const errorCode = error.code ?? JSONRPCCustomErrorCode.EXCEPTION_RAISED;
            const messageType = (errorCode !== JSONRPCCustomErrorCode.NEWER_SESSION_AVAILABLE) ? 'warning' : 'info';
            NotificationHandler.terminalLog(messageType, error.message, undefined);
            return false;
        }
    }

    public request(data: JSONRPCRequest, endpoint: EndpointType) {
        if (this.socket.disconnected) throw new Error('WebSocket is disconnected. Make sure the communication server is available.');

        const stringify = JSON.stringify(data);

        // Emit request in chunks
        if (new Blob([stringify]).size > this.maxMessageLength) {
            const chunkId = uuidv4();

            splitDataIntoChunks(data, this.maxMessageLength, chunkId).forEach((chunk) => {
                this.socket.emit(endpoint, chunk);
            });
        } else {
            this.socket.emit(endpoint, data);
        }
    }
}
