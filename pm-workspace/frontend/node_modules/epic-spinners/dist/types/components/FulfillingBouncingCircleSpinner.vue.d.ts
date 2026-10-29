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
    spinnerStyle(): {
        height: string;
        width: string;
        animationDuration: string;
    };
    orbitStyle(): {
        height: string;
        width: string;
        borderColor: any;
        borderWidth: string;
        animationDuration: string;
    };
    circleStyle(): {
        height: string;
        width: string;
        borderColor: any;
        color: any;
        borderWidth: string;
        animationDuration: string;
    };
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
