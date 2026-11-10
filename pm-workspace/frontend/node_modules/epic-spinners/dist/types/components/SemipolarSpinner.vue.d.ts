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
    ringsNum: number;
}, {
    spinnerStyle(): {
        height: string;
        width: string;
    };
    ringStyle(): {
        animationDuration: string;
        borderTopColor: string;
        borderLeftColor: string;
    };
    ringsStyles(): ({
        animationDelay: string;
        height: string;
        width: string;
        left: string;
        top: string;
        borderWidth: string;
    } & {
        animationDuration: string;
        borderTopColor: string;
        borderLeftColor: string;
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
