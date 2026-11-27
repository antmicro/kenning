declare const _default: import("vue").DefineComponent<{
    animationDuration: {
        type: NumberConstructor;
        default: number;
    };
    size: {
        type: NumberConstructor;
        default: number;
    };
    color: {
        type: StringConstructor;
        default: string;
    };
}, any, {
    squaresNum: number;
}, {
    squareSize(): number;
    initialTopPosition(): number;
    spinnerStyle(): {
        top: string;
        height: string;
        width: string;
    };
    squareStyle(): {
        height: string;
        width: string;
        top: string;
        marginRight: string;
        marginTop: string;
        animationDuration: string;
        background: string;
    };
    squaresStyles(): ({
        animationDelay: string;
    } & {
        height: string;
        width: string;
        top: string;
        marginRight: string;
        marginTop: string;
        animationDuration: string;
        background: string;
    })[];
}, {}, import("vue").ComponentOptionsMixin, import("vue").ComponentOptionsMixin, {}, string, import("vue").VNodeProps & import("vue").AllowedComponentProps & import("vue").ComponentCustomProps, Readonly<import("vue").ExtractPropTypes<{
    animationDuration: {
        type: NumberConstructor;
        default: number;
    };
    size: {
        type: NumberConstructor;
        default: number;
    };
    color: {
        type: StringConstructor;
        default: string;
    };
}>>, {
    animationDuration: number;
    color: string;
    size: number;
}>;
export default _default;
