declare const _default: import("vue").DefineComponent<{
    animationDuration: {
        type: NumberConstructor;
        default: number;
    };
    rhombusSize: {
        type: NumberConstructor;
        default: number;
    };
    color: {
        type: StringConstructor;
        default: string;
    };
}, any, {
    rhombusesNum: number;
}, {
    spinnerStyle(): {
        height: string;
        width: string;
    };
    rhombusStyle(): {
        height: string;
        width: string;
        backgroundColor: string;
        animationDuration: string;
        left: string;
    };
    rhombusesStyles(): ({
        animationDelay: string;
    } & {
        height: string;
        width: string;
        backgroundColor: string;
        animationDuration: string;
        left: string;
    })[];
}, {}, import("vue").ComponentOptionsMixin, import("vue").ComponentOptionsMixin, {}, string, import("vue").VNodeProps & import("vue").AllowedComponentProps & import("vue").ComponentCustomProps, Readonly<import("vue").ExtractPropTypes<{
    animationDuration: {
        type: NumberConstructor;
        default: number;
    };
    rhombusSize: {
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
    rhombusSize: number;
}>;
export default _default;
