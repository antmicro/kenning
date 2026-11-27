import { test, expect } from '@playwright/test';
import { expectNoConsoleErrors, getNode, getUrl, loadVideoNodeId, closeTerminal, getContextMenu, loadSpecification, loadDataflow, openNodePalette } from './config.js';

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

test('show context menu with URLs button and click it', async ({ page }) => {
    // Do not allow errors in console, apart from Failed to load resource
    expectNoConsoleErrors(page, ["Failed to load resource: net::ERR_FILE_NOT_FOUND"]);
    // Load a website and wait until nodes are loaded.
    await page.goto(getUrl());
    await loadSpecification(page, 'sample-specification-urls.json');
    await loadDataflow(page, 'sample-dataflow.json');
    const saveVideoNodeId = 'fc7d1706-6240-41e2-a8da-91c8577e09f9';
    await page.waitForSelector(`#${saveVideoNodeId}`);
    const node = page.locator(`#${saveVideoNodeId}`);

    // Right-click on the node's title.
    const nodeTitle = node.locator('.__title');
    await nodeTitle.click({ button: 'right' });
    const menu = getContextMenu(page);
    await menu.waitFor();
    // Check if button with custom link is present
    expect(await menu.filter({ hasText: 'Open Hardware Portal' }).count()).toBeGreaterThan(0);
    // Click custom link and make sure no error is raised
    await menu.getByText('Renodepedia').click();
});
test('test custom context menu visibility, with disconnected backend', async ({ page }) => {
    // Load a website.
    await page.goto(getUrl());
    const loadVideoNodeId = 'f50b4f2a-a2e2-4409-a5c9-891a8de44a5b';
    await page.waitForSelector(`#${loadVideoNodeId}`);

    const filter2D = getNode(page, "Filter2D");
    await filter2D.click({ button: "right" });

    const nodeInfo = page.getByText("Node info");
    // When backend is disconnected it should not be visible
    expect(nodeInfo).not.toBeVisible();
});
test('check metadata override', async ({ page }) => {
    // Load a website.
    await page.goto(getUrl());
    // Load specification first and check if New Graph Node is present
    await loadSpecification(page, 'sample-with-shape-specification.json');
    await openNodePalette(page);
    const newNodeType = page.getByText("New Node Type");

    expect(newNodeType).not.toBeVisible();

    await loadDataflow(page, 'sample-with-shape-dataflow.json');
    await openNodePalette(page);

    expect(newNodeType).toBeVisible();
});
