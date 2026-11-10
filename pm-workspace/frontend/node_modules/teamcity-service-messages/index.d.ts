declare class Message {
    constructor(type: string, args?: unknown)
    toString(): string
}

type Return<S extends boolean> = S extends true ? tsm.API<true> : string
declare namespace tsm {
    export interface Args {
        timestamp?: string
        flowId?: string
    }

    export interface BlockClosedArgs extends Args {
        name: string
        description?: string
    }

    export interface BlockClosedArgs extends Args {
        name: string
    }

    export interface BlockOpenedArgs extends Args {
        name: string
        description?: string
    }

    export interface BuildProblemArgs extends Args {
        description: string
        identity?: string
    }

    export interface BuildStatisticValueArgs extends Args {
        key: string
        value: number
    }

    export interface BuildStatusArgs extends Args {
        status?: 'SUCCESS'
        text: string
    }

    export interface CompilationArgs extends Args {
        compiler: string
    }

    export interface ImportDataArgs extends Args {
        type:
            | 'junit'
            | 'surefire'
            | 'nunit'
            | 'mstest'
            | 'vstest'
            | 'gtest'
            | 'intellij-inspections'
            | 'checkstyle'
            | 'jslint'
            | 'FxCop1'
            | 'pmd'
            | 'pmdCpd'
            | 'DotNetDupFinder'
        path: string
    }

    export interface ImportFindBugsDataArgs extends Args {
        type: 'findBugs'
        path: string
        findBugsHome: string
    }

    export interface ImportDotNetCoverageDataArgs extends Args {
        type: 'dotNetCoverage'
        path: string
        tool: 'dotcover' | 'partcover' | 'ncover' | 'ncover3'
    }

    export interface InspectionTypeArgs extends Args {
        id: string
        name: string
        category: string
        description: string
    }

    export interface InspectionArgs extends Args {
        typeId: string
        message?: string
        file: string
        line?: string
        SEVERITY?: 'INFO' | 'ERROR' | 'WARNING' | 'WEAK WARNING'

        [key: string]: unknown
    }

    export interface MessageArgs extends Args {
        text: string
        status?: 'NORMAL' | 'WARNING' | 'FAILURE'
    }

    export interface ErrorMessageArgs extends Args {
        text: string
        status: 'ERROR'
        errorDetails?: string
    }

    export interface SetParameterArgs extends Args {
        name: string
        value: string | number | boolean
    }

    export interface TestFailedArgs extends Args {
        name: string
        message?: string
        details?: string
    }

    export interface TestComparisonFailureArgs extends TestFailedArgs {
        type: 'comparisonFailure'
        expected: unknown
        actual: unknown
    }

    export interface TestFinishedArgs extends Args {
        name: string
        duration?: number
    }

    export interface TestIgnoredArgs extends Args {
        name: string
        message?: string
    }

    export interface TestMetadataArgs extends Args {
        name?: string
        testName?: string
    }

    export interface NumberTestMetadataArgs extends TestMetadataArgs {
        type: 'number'
        value: number
    }

    export interface StringTestMetadataArgs extends TestMetadataArgs {
        type?: 'text' | 'link' | 'artifact' | 'image' | 'video'
        value: string
    }

    export interface TestStartedArgs extends Args {
        name: string
        captureStandardOutput?: boolean
    }

    export interface TestStdOutArgs extends Args {
        name: string
        out: string
    }

    export interface TestSuiteArgs extends Args {
        name: string
    }

    export interface API<S extends boolean> {
        Message: typeof Message
        stdout: S
        autoFlowId: boolean

        blockClosed(args: BlockClosedArgs): Return<S>

        blockOpened(args: BlockOpenedArgs): Return<S>

        buildNumber(number: string | number): Return<S>

        buildProblem(args: BuildProblemArgs): Return<S>

        buildStatisticValue(args: BuildStatisticValueArgs): Return<S>

        buildStatus(args: BuildStatusArgs): Return<S>

        compilationFinished(args: CompilationArgs): Return<S>

        compilationStarted(args: CompilationArgs): Return<S>

        disableServiceMessages(): Return<S>

        enableServiceMessages(): Return<S>

        importData(
            args:
                | ImportDataArgs
                | ImportFindBugsDataArgs
                | ImportDotNetCoverageDataArgs,
        ): Return<S>

        inspectionType(args: InspectionTypeArgs): Return<S>

        inspection(args: InspectionArgs): Return<S>

        message(args: MessageArgs | ErrorMessageArgs): Return<S>

        progressFinish(message: string): Return<S>

        progressMessage(message: string): Return<S>

        progressStart(message: string): Return<S>

        publishArtifacts(path: string): Return<S>

        setParameter(args: SetParameterArgs): Return<S>

        testFailed(args: TestFailedArgs | TestComparisonFailureArgs): Return<S>

        testFinished(args: TestFinishedArgs): Return<S>

        testIgnored(args: TestIgnoredArgs): Return<S>

        testMetadata(
            args: NumberTestMetadataArgs | StringTestMetadataArgs,
        ): Return<S>

        testStarted(args: TestStartedArgs): Return<S>

        testStdErr(args: TestStdOutArgs): Return<S>

        testStdOut(args: TestStdOutArgs): Return<S>

        testSuiteFinished(args: TestSuiteArgs): Return<S>

        testSuiteStarted(args: TestSuiteArgs): Return<S>
    }
}

declare const tsm: tsm.API<true>
export = tsm
