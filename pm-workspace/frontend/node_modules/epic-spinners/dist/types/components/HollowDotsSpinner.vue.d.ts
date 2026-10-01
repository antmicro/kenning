declare const _default: import("vue").DefineComponent<{
    animationDuration: {
        type: NumberConstructor;
        default: number;
    };
    dotSize: {
        type: NumberConstructor;
        default: number;
    };
    dotsNum: {
        type: NumberConstructor;
        default: number;
    };
    color: {
        type: StringConstructor;
        default: string;
    };
}, any, any, {
    horizontalMargin(): number;
    spinnerStyle(): {
        height: string;
        width: string;
    };
    dotStyle(): {
        animationDuration: string;
        width: string;
        height: string;
        margin: string;
        borderWidth: string;
        borderColor: any;
    };
    dotsStyles(): any[];
}, {}, import("vue").ComponentOptionsMixin, import("vue").ComponentOptionsMixin, {}, string, import("vue").VNodeProps & import("vue").AllowedComponentProps & import("vue").ComponentCustomProps, Readonly<import("vue").ExtractPropTypes<{
    animationDuration: {
        type: NumberConstructor;
        default: number;
    };
    dotSize: {
        type: NumberConstructor;
        default: number;
    };
    dotsNum: {
        type: NumberConstructor;
        default: number;
    };
    color: {
        type: StringConstructor;
        default: string;
    };
}>>, {
    animationDuration: number;
    dotSize: number;
    dotsNum: number;
    color: string;
}>;
export default _default;
