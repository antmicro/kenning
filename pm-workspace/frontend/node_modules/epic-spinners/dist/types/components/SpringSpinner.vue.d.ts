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
    animationName: string;
}, {
    spinnerStyle(): {
        height: string;
        width: string;
    };
    spinnerPartStyle(): {
        height: string;
        width: string;
    };
    rotatorStyle(): {
        height: string;
        width: string;
        borderRightColor: string;
        borderTopColor: string;
        borderWidth: string;
        animationDuration: string;
        animationName: string;
    };
}, {
    updateAnimation(): void;
    generateKeyframes(): string;
}, import("vue").ComponentOptionsMixin, import("vue").ComponentOptionsMixin, {}, string, import("vue").VNodeProps & import("vue").AllowedComponentProps & import("vue").ComponentCustomProps, Readonly<import("vue").ExtractPropTypes<{
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
