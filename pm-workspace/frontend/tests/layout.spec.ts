import { test, expect } from '@playwright/test';

import {
    getUrl,
    loadSpecification,
    loadDataflow,
    openFileChooser,
    expectNoErrors,
    saveFileAs,
    openSettingsPanel,
    getNode,
    assertInputCount,
    assertOutputCount
} from './config.js';

test('export and import layout',async ({page},testInfo) => {
    await page.goto(getUrl());
    await loadSpecification(page, 'sample-views-specification.json');
    await loadDataflow(page, 'sample-views-dataflow.json');
    // save current layout
    const filePath = await saveFileAs(page,testInfo,'layout','layout');
    // load current layout
    const fileChooserPromise = page.waitForEvent('filechooser');
    await page.mouse.move(25, 25);
    await page.getByText("Load layout").click();
    const fileChooser = await fileChooserPromise;
    await fileChooser.setFiles(filePath);

    await expectNoErrors(page);
});

async function dumpAllNodesStates(page)
{
    const nodesStates = [];
    const nodes = await page.locator('.baklava-node').all();

    for (const n of nodes)
    {
        const bbox = await n.boundingBox();
        const id = await n.getAttribute('id');

        const outputsStates = [];
        const inputsStates = [];

        // dump all node inputs and outputs
        const outputs = await n.locator('.__outputs').locator('.baklava-node-interface').all();
        for (const out of outputs)
        {
            const id = await out.getAttribute('id');
            outputsStates.push(id);
        }

        const inputs = await n.locator('.__inputs').locator('.baklava-node-interface').all();
        for (const ins of inputs)
        {
            const id = await ins.getAttribute('id');
            inputsStates.push(id);
        }

        nodesStates.push({
            id,
            bbox,
            inputsStates,
            outputsStates
        })
    }

    return nodesStates;
}

test('import layout into other view',async ({page},testInfo) => {
    await page.goto(getUrl());
    await loadSpecification(page, 'sample-views-specification.json');
    await loadDataflow(page, 'sample-views-dataflow.json');

    const zoomButton = page.locator('.zoom-center');

    // Select a tree view and save it's layout
    await openSettingsPanel(page);

    const selectedView = page.getByTitle('Selected view');
    await selectedView.click();
    const dropdown = selectedView.locator('.__dropdown');

    await dropdown.getByText('tree').click({ force: true });
    await page.mouse.click(0,0);
    await zoomButton.click();

    // store current nodes states
    const nodesStatesBefore = await dumpAllNodesStates(page);

    // save current layout
    // const targetScreenshot = await page.screenshot({path: "target.png",fullPage: true});
    const filePath = await saveFileAs(page,testInfo,'layout','layout_1');
    // Select a horse view and save it's layout
    await openSettingsPanel(page);

    await selectedView.click();
    await dropdown.getByText('horse').click({ force: true });
    // load current layout
    const fileChooserPromise = page.waitForEvent('filechooser');
    await page.mouse.move(25, 25);
    await page.getByText("Load layout").click();
    const fileChooser = await fileChooserPromise;
    await fileChooser.setFiles(filePath);
    await page.mouse.move(0, 0);

    await expectNoErrors(page);
    await zoomButton.click();
    // store current nodes states
    const nodesStatesAfter = await dumpAllNodesStates(page);

    await expect(nodesStatesBefore).toStrictEqual(nodesStatesAfter);
    // compare screenshots
});

test('preserve changes across views',async ({page},testInfo) => {
    await page.goto(getUrl());
    await loadSpecification(page, 'sample-views-specification.json');
    await loadDataflow(page, 'sample-views-dataflow.json');

    await openSettingsPanel(page);

    const selectedView = await page.getByTitle('Selected view');
    await selectedView.click();
    const dropdown = await selectedView.locator('.__dropdown');

    await dropdown.getByText('tree').click({ force: true });

    const node = await getNode(page,'Source').first();

    const outputInf = node.locator('.__interfaces .__outputs .__port').first();
    await outputInf.dblclick();

    await assertInputCount(node,1);
    await assertOutputCount(node,0);

    // save dataflow with views
    const dataflowWithViews = await saveFileAs(page, testInfo, 'dataflow');
    // load dataflow with virws
    const fileChooser = await openFileChooser(page, 'dataflow');
    await fileChooser.setFiles(dataflowWithViews);

    await openSettingsPanel(page);

    await selectedView.click();
    await dropdown.getByText('tree').click({ force: true });

    await assertInputCount(node,1);
    await assertOutputCount(node,0);
});