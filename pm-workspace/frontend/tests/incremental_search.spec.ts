import { test, expect } from '@playwright/test';
import { getUrl, loadSpecification, loadDataflow } from './config.js';

test("test incremental search", async ({ page }) => {
    await page.goto(getUrl());
    await loadSpecification(page, 'sample-inheritance-specification.json');
    await loadDataflow(page, 'sample-inheritance-dataflow.json');

    const totalNodes = await page.locator('.baklava-node').count();

    await page.locator('.search-section').click();
    await page.getByPlaceholder('Search for nodes').fill('Type');

    let greyedOutNodes = await page.locator('.baklava-node.--greyed-out').count();
    let highlightedNodes = await page.locator('.baklava-node:not(.--greyed-out)').count();
    expect(greyedOutNodes).toBe(0);
    expect(highlightedNodes).toBe(totalNodes);

    await page.locator('.search-section').click();
    await page.getByPlaceholder('Search for nodes').fill('Type C');

    greyedOutNodes = await page.locator('.baklava-node.--greyed-out').count();
    highlightedNodes = await page.locator('.baklava-node:not(.--greyed-out)').count();

    expect(greyedOutNodes).toBe(3);
    expect(highlightedNodes).toBe(2);
    expect(greyedOutNodes + highlightedNodes).toBe(totalNodes);
});

