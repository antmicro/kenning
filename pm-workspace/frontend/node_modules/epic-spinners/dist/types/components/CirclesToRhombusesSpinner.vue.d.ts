declare const _default: import("vue").DefineComponent<{
    animationDuration: {
        type: NumberConstructor;
        default: number;
    };
    circleSize: {
        type: NumberConstructor;
        default: number;
    };
    color: {
        type: StringConstructor;
        default: string;
    };
    circlesNum: {
        type: NumberConstructor;
        default: number;
    };
}, any, any, {
    circleMarginLeft(): number;
    spinnerStyle(): {
        height: string;
        width: string;
    };
    circleStyle(): {
        borderColor: any;
        animationDuration: string;
        height: string;
        width: string;
        marginLeft: string;
    };
    circlesStyles(): any[];
}, {}, import("vue").ComponentOptionsMixin, import("vue").ComponentOptionsMixin, {}, string, import("vue").VNodeProps & import("vue").AllowedComponentProps & import("vue").ComponentCustomProps, Readonly<import("vue").ExtractPropTypes<{
    animationDuration: {
        type: NumberConstructor;
        default: number;
    };
    circleSize: {
        type: NumberConstructor;
        default: number;
    };
    color: {
        type: StringConstructor;
        default: string;
    };
    circlesNum: {
        type: NumberConstructor;
        default: number;
    };
}>>, {
    animationDuration: number;
    color: string;
    circleSize: number;
    circlesNum: number;
}>;
export default _default;
