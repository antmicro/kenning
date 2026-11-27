import { test, expect } from '@playwright/test';
import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import { getUrl, loadVideoNodeId, closeTerminal, getContextMenu, openFileChooser, getPathToJsonFile, loadSpecification } from './config.js';

async function enableReadOnly(spec)
{
    const filePath = getPathToJsonFile(spec);
    const outFilePath = `/tmp/readonly${crypto.randomBytes(4).readUInt32LE(0)}.json`

    const data = await fs.readFile(filePath);

    const specification = JSON.parse(data);

    if(!Object.hasOwn(specification,"metadata"))
    {
        specification["metadata"] = {};
    }

    specification["metadata"]["readonly"] = true;

    await fs.writeFile(outFilePath,JSON.stringify(specification));

    return outFilePath;
}


test('empty settings panel', async ({ page }) => {
    await page.goto(getUrl());

    const fileName = await enableReadOnly('sample-inout-specification.json');

    const fileChooser = await openFileChooser(page, 'specification');
    await fileChooser.setFiles(fileName);

    const settingsButton = page.getByText('Show settings').locator('..').locator('..');

    expect(settingsButton).not.toBeVisible();
});

test('settings panel with only layers', async ({ page }) => {
    await page.goto(getUrl());

    const fileName = await enableReadOnly('sample-specification.json');

    const fileChooser = await openFileChooser(page, 'specification');
    await fileChooser.setFiles(fileName);

    const settingsButton = page.getByText('Show settings').locator('..').locator('..');

    expect(settingsButton).toBeVisible();

    await settingsButton.click();

    const settingPanel = page.locator('.settings-panel .panel > div');

    expect(await settingPanel.count()).toBe(1);
});