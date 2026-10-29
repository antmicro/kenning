/*
 * Copyright (c) 2022-2024 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* eslint-disable import/prefer-default-export */

// Mouse position
let y = 0;
let h = 0;

// Height setting callback
let setTerminalHeight;

const mouseMoveHandler = (e) => {
    const { clientY } = e;
    const navBarHeight = 61;

    // Calculate mouse move
    const dy = y - clientY;

    // Prevent terminal overflow under nav bar
    if (clientY > navBarHeight) {
        setTerminalHeight(h + dy);
    }
};

const mouseUpHandler = () => {
    const terminalWrapper = document.querySelector('.terminal-wrapper');

    document.removeEventListener('mousemove', mouseMoveHandler);
    document.removeEventListener('mouseup', mouseUpHandler);
    terminalWrapper.style['pointer-events'] = '';
};

export const mouseDownHandler = (setHeight) => (e) => {
    const terminalWrapper = document.querySelector('.terminal-wrapper');
    setTerminalHeight = setHeight;

    y = e.clientY;

    // Calculate height of terminal panel
    const styles = window.getComputedStyle(terminalWrapper);
    h = parseInt(styles.height, 10);

    document.addEventListener('mousemove', mouseMoveHandler);
    document.addEventListener('mouseup', mouseUpHandler);
    terminalWrapper.style['pointer-events'] = 'none';
};
