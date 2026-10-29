import { terminalStore } from '../stores';
import ExternalApp from './externalApp/base';

export type SpecType = {
    params: object,
    returns: object | null, // null when method is a notification
};

export type Endpoints = { [key: string]: SpecType };

export type ClientParams = { externalApp: ExternalApp | null };

export type TerminalView = {
    names: string[] | undefined,
    clearButton: boolean | undefined,
}

export type TerminalManager = {
    hide(): void;
    show(name: string | undefined): void;
    view(params: TerminalView): void;
}

export function checkTerminalExistence(name: string, exists = true) {
    if (terminalStore.exists(name) !== exists) {
        const message = `Terminal instance of name '${name} ${exists ? 'does not exist' : 'already exists'}`;
        throw new Error(message);
    }
}
