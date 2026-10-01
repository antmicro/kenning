import YAML from 'yaml';
import {
    test, expect, Page,
    Locator,
} from '@playwright/test';
import {
    getUrl, loadSpecification, loadDataflow, getContextMenu, deleteNode,
} from './config.js';

function getNode(page: Page, name: string): Locator {
    return page.locator(`.baklava-node[data-node-type="${name}"]`);
}
function getGroup(page: Page, name: string): Locator {
    return page.locator('.grouping-container').getByText(name).locator('..');
}

async function getRect(nodes: Locator[]) {
    let left = Infinity;
    let right = -Infinity;
    let top = Infinity;
    let bottom = -Infinity;
    // eslint-disable-next-line no-restricted-syntax
    for (const node of nodes) {
        // eslint-disable-next-line no-await-in-loop
        const rect = await node.boundingBox();
        if (!rect) {
            // eslint-disable-next-line no-continue
            continue;
        }

        left = Math.min(left, rect.x);
        right = Math.max(right, rect.x + rect.width);
        top = Math.min(top, rect.y);
        bottom = Math.max(bottom, rect.y + rect.height);
    }
    return {
        x: left,
        y: top,
        width: right - left,
        height: bottom - top,
    };
}
// returns max difference in dimensions between two rects
function compareRects(rect1: any, rect2: any) {
    let diff = 0;
    diff = Math.max(diff, Math.abs(rect1.x - rect2.x));
    diff = Math.max(diff, Math.abs(rect1.y - rect2.y));
    diff = Math.max(diff, Math.abs(rect1.x + rect1.width - (rect2.x + rect2.width)));
    diff = Math.max(diff, Math.abs(rect1.y + rect1.height - (rect2.y + rect2.height)));
    return diff;
}
async function removeFromGroup(node: Locator, page: Page) {
    await node.locator('.__title').click({ button: 'right' });
    await getContextMenu(page).getByText('Remove from group').click();
}
async function createGroup(nodes: Locator[], page: Page) {
    // eslint-disable-next-line no-restricted-syntax
    for (let i = 0; i < nodes.length; i += 1) {
        // eslint-disable-next-line no-await-in-loop
        await nodes[i].locator('.__title').click({ modifiers: ['ControlOrMeta'] });
    }
    await nodes[0].locator('.__title').click({ button: 'right' });
    // in some cases does not fit on the screen, the test would require dragging screen
    // or changing screen resolution. This is why instead dispatchEvent was used.
    await getContextMenu(page).getByText('Group Nodes').dispatchEvent('click');
    await page.locator('.baklava-button').getByText('Create group').click();
}

test('check for group presence', async ({ page }) => {
    await page.goto(getUrl());

    await loadSpecification(page, 'sample-rectangle-grouping-specification.json');
    await loadDataflow(page, 'sample-rectangle-grouping-dataflow.json');

    const groupCount = await page.locator('.grouping-container > div').count();
    expect(groupCount).toBe(6);
});
test('check for group dimensions', async ({ page }) => {
    await page.goto(getUrl());

    await loadSpecification(page, 'sample-rectangle-grouping-specification.json');
    await loadDataflow(page, 'sample-rectangle-grouping-dataflow.json');

    const modelOptGroup = getGroup(page, 'Model Optimizer');
    const nodeOpt = getNode(page, 'Optimizer');
    const nodeComp = getNode(page, 'Compiler');

    const nodeRect = await getRect([nodeOpt, nodeComp]);
    const groupRect = await modelOptGroup.boundingBox();
    expect(compareRects(nodeRect, groupRect)).toBeLessThanOrEqual(10);
});
test('group creation', async ({ page }) => {
    await page.goto(getUrl());

    await loadSpecification(page, 'sample-specification.json');
    await loadDataflow(page, 'sample-dataflow.json');

    const nodeA = getNode(page, 'Threshold');
    const nodeB = getNode(page, 'StructuringElement');
    await createGroup([nodeB, nodeA], page);
    const group = page.locator('.rectangle-grouping');
    expect(group).toBeVisible();

    await nodeB.locator('.__title').click(); // move focus

    const nodeRect = await getRect([nodeA, nodeB]);
    const groupRect = await group.boundingBox();
    expect(compareRects(nodeRect, groupRect)).toBeLessThanOrEqual(10);
    await page.keyboard.press('Control+KeyZ');

    await group.waitFor({ state: 'hidden' });
    expect(group).not.toBeVisible();
    await page.keyboard.press('Control+KeyY');

    expect(compareRects(nodeRect, groupRect)).toBeLessThanOrEqual(10);
});
test('remove from group', async ({ page }) => {
    await page.goto(getUrl());

    await loadSpecification(page, 'sample-rectangle-grouping-specification.json');
    await loadDataflow(page, 'sample-rectangle-grouping-dataflow.json');

    const group = getGroup(page, 'Dataset Wrapper');
    const nodeA = getNode(page, 'Dataset');
    const nodeB = getNode(page, 'I/O processing - Dataset');
    const nodeC = getNode(page, 'Model evaluation');

    const nodeRect = await getRect([nodeA, nodeB, nodeC]);
    const groupRect = await group.boundingBox();
    expect(compareRects(nodeRect, groupRect)).toBeLessThanOrEqual(10);

    await removeFromGroup(nodeC, page);
    await expect(async () => {
        const newGroupRect = await group.boundingBox();
        expect(newGroupRect).not.toEqual(groupRect);
    }).toPass({ timeout: 5000 });
    await page.keyboard.press('Control+KeyZ');
    await expect(async () => {
        const revertedGroupRect = await group.boundingBox();
        expect(compareRects(nodeRect, revertedGroupRect)).toBeLessThanOrEqual(10);
    }).toPass({ timeout: 5000 });
    await page.keyboard.press('Control+KeyY');
    await expect(async () => {
        const redoneGroupRect = await group.boundingBox();
        expect(redoneGroupRect).not.toEqual(groupRect);
    }).toPass({ timeout: 5000 });

    const nodeRectPost = await getRect([nodeA, nodeB]);
    const groupRectPost = await group.boundingBox();
    expect(compareRects(nodeRectPost, groupRectPost)).toBeLessThanOrEqual(10);
});
test('delete group', async ({ page }) => {
    await page.goto(getUrl());

    await loadSpecification(page, 'sample-rectangle-grouping-specification.json');
    await loadDataflow(page, 'sample-rectangle-grouping-dataflow.json');

    const group = getGroup(page, 'Dataset Wrapper');
    const nodeA = getNode(page, 'Dataset');
    const nodeB = getNode(page, 'I/O processing - Dataset');
    const nodeC = getNode(page, 'Model evaluation');

    await removeFromGroup(nodeA, page);
    await removeFromGroup(nodeC, page);

    await expect(group).not.toBeVisible();

    await createGroup([nodeA, nodeB, nodeC], page);

    const newGroup = getGroup(page, 'New Group');
    await expect(newGroup).toBeVisible();

    await nodeC.click();
    await deleteNode(nodeC, page);
    await deleteNode(nodeB, page);

    await expect(newGroup).not.toBeVisible();
});
