export declare class Locks {
    locks: {
        [key: number]: number[];
    };
    add(id: number, x: number[]): void;
    clear(): void;
    isEmpty(): boolean;
    apply(f: (id: number, x: number[]) => void): void;
}
export declare class Descent {
    D: number[][];
    G: number[][];
    threshold: number;
    H: number[][][];
    g: number[][];
    x: number[][];
    k: number;
    n: number;
    locks: Locks;
    private static zeroDistance;
    private minD;
    private Hd;
    private a;
    private b;
    private c;
    private d;
    private e;
    private ia;
    private ib;
    private xtmp;
    numGridSnapNodes: number;
    snapGridSize: number;
    snapStrength: number;
    scaleSnapByMaxH: boolean;
    private random;
    project: {
        (x0: number[], y0: number[], r: number[]): void;
    }[];
    constructor(x: number[][], D: number[][], G?: number[][]);
    static createSquareMatrix(n: number, f: (i: number, j: number) => number): number[][];
    private offsetDir;
    computeDerivatives(x: number[][]): void;
    private static dotProd;
    private static rightMultiply;
    computeStepSize(d: number[][]): number;
    reduceStress(): number;
    private static copy;
    private stepAndProject;
    private static mApply;
    private matrixApply;
    private computeNextPosition;
    run(iterations: number): number;
    rungeKutta(): number;
    private static mid;
    takeDescentStep(x: number[], d: number[], stepSize: number): void;
    computeStress(): number;
}
export declare class PseudoRandom {
    seed: number;
    private a;
    private c;
    private m;
    private range;
    constructor(seed?: number);
    getNext(): number;
    getNextBetween(min: number, max: number): number;
}
