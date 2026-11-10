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
    spinnerStyle(): {
        height: string;
        width: string;
        animationDuration: string;
    };
    squareStyle(): {
        height: string;
        width: string;
        animationDuration: string;
        borderWidth: string;
        borderColor: string;
    };
    squaresStyles(): {
        height: string;
        width: string;
        animationDuration: string;
        borderWidth: string;
        borderColor: string;
    }[];
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
