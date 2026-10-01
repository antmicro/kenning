import { test, expect } from '@playwright/test';
import { getUrl, loadVideoNodeId, closeTerminal, getContextMenu } from './config.js';

test('has title', async ({ page }) => {
    await page.goto(getUrl());
    expect(await page.title()).toBe('Pipeline Manager');
});

test('remove node', async ({ page }) => {
    // Load a website and wait until nodes are loaded.
    await page.goto(getUrl());
    await page.waitForSelector(`#${loadVideoNodeId}`);
    await closeTerminal(page);

    // Find the node and invoke a context menu with a right click.
    const loadVideoNode = page.locator(`#${loadVideoNodeId}`);
    expect(loadVideoNode).toBeVisible();
    const nodeTitle = loadVideoNode.locator('.__title');
    await nodeTitle.click({ button: 'right' });

    // Delete the node.
    const deleteButton = getContextMenu(page).getByText('Delete', { exact: true });
    await deleteButton.click({ force: true });

    // Verify that the node has disappeared.
    await expect(page.locator(`#${loadVideoNodeId}`)).not.toBeVisible();
});

test('show menu by right-clicking on node', async ({ page }) => {
    // Load a website and wait until nodes are loaded.
    await page.goto(getUrl());
    const loadVideoNodeId = 'f50b4f2a-a2e2-4409-a5c9-891a8de44a5b';
    await page.waitForSelector(`#${loadVideoNodeId}`);
    const node = page.locator(`#${loadVideoNodeId}`);

    // Right-click on the node's title.
    const nodeTitle = node.locator('.__title');
    await nodeTitle.click({ button: 'right' });
    const menu = getContextMenu(page);
    await menu.waitFor()
    expect(await menu.filter({ hasText: 'Details' }).count()).toBeGreaterThan(0);

    // Close the context menu.
    await page.mouse.click(200, 0);

    // Right-click on the node's content.
    const nodeBody = node.locator('.__content');
    await nodeBody.click({ button: 'right' , position: { x: 0, y: 0 }});
    expect(await menu.filter({ hasText: 'Details' }).count()).toBeGreaterThan(0);
});

test('show menu by right-clicking on interface', async ({ page }) => {
    // Load a website.
    await page.goto(getUrl());
    const loadVideoNodeId = 'f50b4f2a-a2e2-4409-a5c9-891a8de44a5b';
    await page.waitForSelector(`#${loadVideoNodeId}`);
    const node = page.locator(`#${loadVideoNodeId}`);

    // Right-click on a node.
    const nodeInterface = node.getByText('frames').locator('..');
    await nodeInterface.click({ button: 'right' });

    // Verify presence and content of a context menu.
    const menu = getContextMenu(page);
    await menu.waitFor();
    expect(await menu.filter({ hasText: 'Space Up' }).count()).toBeGreaterThan(0);
});
