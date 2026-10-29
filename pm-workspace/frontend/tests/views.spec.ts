import { test, expect, Locator } from '@playwright/test';
import { getUrl, loadVideoNodeId, closeTerminal, getContextMenu,
    getNode, getElementStyleAttribute, loadSpecification, loadDataflow, openSettingsPanel } from './config.js';


async function getNodePositions(node: Locator)
{
    const left = await getElementStyleAttribute(node,'left');
    const top = await getElementStyleAttribute(node,'top');

    return {
        x: left,
        y: top
    }
}

test('test node position across views', async ({ page }) => {
    await page.goto(getUrl());
    await loadSpecification(page, 'sample-views-specification.json');
    await loadDataflow(page, 'sample-views-dataflow.json');

    const zoomButton = page.locator('.zoom-center');

    // get node position
    const node = await getNode(page,"Source").first();
    const nodeStartPosition = await node.boundingBox();

    // move node
    await node.click();
    await page.mouse.down();
    await page.mouse.move(nodeStartPosition.x + 100,nodeStartPosition.y);
    await page.mouse.up();

    await zoomButton.dispatchEvent('click');
    // get current node position
    const nodeBoxDefaultView = await getNodePositions(node);

    // Switch views.
    await openSettingsPanel(page);

    const selectedView = page.getByTitle('Selected view');
    await selectedView.click();
    const dropdown = selectedView.locator('.__dropdown');

    await dropdown.getByText('tree').click({ force: true });
    await zoomButton.dispatchEvent('click');

    const nodeTreePosition = await getNodePositions(node);

    expect((nodeBoxDefaultView.x !== nodeTreePosition.x) &&
        (nodeBoxDefaultView.y !== nodeTreePosition.y)).toBeTruthy();

    // Switch views to default
    await selectedView.dispatchEvent('click');
    await dropdown.getByText('default').click({ force: true });
    await zoomButton.dispatchEvent('click');

    const nodeCurrentPosition = await getNodePositions(node);

    expect((nodeBoxDefaultView.x === nodeCurrentPosition.x) &&
        (nodeBoxDefaultView.y === nodeCurrentPosition.y)).toBeTruthy();
});