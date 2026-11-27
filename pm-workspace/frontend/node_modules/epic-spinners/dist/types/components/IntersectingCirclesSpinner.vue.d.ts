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
}, any, any, {
    circleSize(): number;
    spinnerStyle(): {
        width: string;
        height: string;
    };
    spinnerBlockStyle(): {
        animationDuration: string;
        width: string;
        height: string;
    };
    circleStyle(): {
        borderColor: any;
    };
    circleStyles(): any[];
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
