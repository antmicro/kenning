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
    animationBaseName: string;
    rhombusesNum: number;
}, {
    spinnerStyle(): {
        height: string;
        width: string;
    };
    rhombusStyle(): {
        height: string;
        width: string;
        animationDuration: string;
        top: string;
        left: string;
        backgroundColor: string;
    };
    rhombusesStyles(): ({
        animationDelay: string;
    } & {
        height: string;
        width: string;
        animationDuration: string;
        top: string;
        left: string;
        backgroundColor: string;
    })[];
    bigRhombusStyle(): {
        height: string;
        width: string;
        animationDuration: string;
        top: string;
        left: string;
        backgroundColor: string;
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
