import { test, expect } from '@playwright/test';
import { getUrl, loadVideoNodeId, closeTerminal, getContextMenu, getNode, getElementStyleAttribute } from './config.js';

test('test zoom', async ({ page }) => {
    await page.goto(getUrl());

    const zoomButton = page.locator('.zoom-center');
    const editor = await page.locator('.baklava-editor');

    const loadVideo = await getNode(page,'LoadVideo');
    const saveVideo = await getNode(page,'SaveVideo');

    await zoomButton.click();

    const scaleBefore = await getElementStyleAttribute(editor,'--scale');

    const settings = page.locator('.settings-panel');
    expect(settings).toBeVisible();
    await settings.hover({ force: true });
    // get layers section
    const layers = page.getByText("Hide Layers:").locator('..');

    await layers.getByText("IOs").dispatchEvent("click");
    await page.locator('.baklava-node-palette').hover();

    await loadVideo.waitFor({state:'hidden'});
    await saveVideo.waitFor({state:'hidden'});

    await zoomButton.click({force: true});
    const scaleAfter = await getElementStyleAttribute(editor,'--scale');

    expect(scaleBefore).not.toBe(scaleAfter);
});