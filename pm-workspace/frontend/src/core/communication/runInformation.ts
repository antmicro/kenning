/*
 * Copyright (c) 2022-2024 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* eslint-disable max-classes-per-file */

/**
 * Class with information about ran procedure.
 */
class RunInfo {
    procedureName;

    hook: (() => undefined) | undefined = undefined;

    /** @private */
    private pr_inProgress = false;

    /**
     * @param procedureName Name of the procedure
     */
    constructor(procedureName: string) {
        this.procedureName = procedureName;
    }

    /**
     * Finds progress bar affiliated with run.
     * If cannot be found, throws Error.
     */
    get progressBar() {
        const progressBar = document.querySelector<HTMLDivElement>(
            `#navbar-button-${this.procedureName} > .progress-bar`,
        );
        return progressBar;
    }

    /**
     * Specifies if run is in progress.
     */
    get inProgress() {
        return this.pr_inProgress;
    }

    set inProgress(value) {
        const { progressBar } = this;
        if (progressBar) {
            if (!value) progressBar.classList.remove('animate');
            progressBar.style.width = '0%';
        }
        this.pr_inProgress = value;
        if (this.hook !== undefined) {
            this.hook();
        }
    }
}

/**
 * @template K key type
 * @template V value type
 * @extends {Map<K,V>}
 *
 * Map automatically creating values if key is not found.
 */
class DefaultMap<K, V> extends Map<K, V> {
    defaultFactory;

    constructor(defaultFactory: (key: K) => V) {
        super();
        this.defaultFactory = defaultFactory;
    }

    get(key: K): V {
        if (!super.has(key)) {
            super.set(key, this.defaultFactory(key));
        }
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        return super.get(key)!;
    }
}

/**
 * @extends {DefaultMap<string, RunInfo>}
 *
 * DefaultMap with custom hook run when RunInfo inProgress changes value.
 */
class RunInfoMap extends DefaultMap<string, RunInfo> {
    hook: (() => undefined) | undefined = undefined;

    get(key: string): RunInfo {
        if (!super.has(key)) {
            const newValue = this.defaultFactory(key);
            newValue.hook = this.hook;
            super.set(key, newValue);
        }
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        return super.get(key)!;
    }

    setHook(hook: () => undefined) {
        this.hook = hook;
        // eslint-disable-next-line no-param-reassign
        this.forEach((v) => { v.hook = hook; });
    }
}

export default new RunInfoMap((key) => new RunInfo(key));
