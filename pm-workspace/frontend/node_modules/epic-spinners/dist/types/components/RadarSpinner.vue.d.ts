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
    circlesNum: number;
}, {
    borderWidth(): number;
    spinnerStyle(): {
        height: string;
        width: string;
    };
    circleStyle(): {
        animationDuration: string;
    };
    circleInnerContainerStyle(): {
        borderWidth: string;
    };
    circleInnerStyle(): {
        borderLeftColor: string;
        borderRightColor: string;
        borderWidth: string;
    };
    circlesStyles(): ({
        padding: string;
        animationDelay: string;
    } & {
        animationDuration: string;
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
