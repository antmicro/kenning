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
    containerPadding: number;
}, {
    outerRingSize(): number;
    spinnerStyle(): {
        height: string;
        width: string;
        padding: string;
    };
    ring1Style(): {
        height: string;
        width: string;
        borderColor: string;
        animationDuration: string;
    };
    ring2Style(): {
        height: string;
        width: string;
        borderColor: string;
        animationDuration: string;
    };
    ring3Style(): {
        height: string;
        width: string;
        borderColor: string;
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
