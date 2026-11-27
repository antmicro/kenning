import {
    test
    ,
} from '@playwright/test';
import {
    getUrl, assertPropertyCount,
    getNode,
    loadSpecification, loadDataflow, getContextMenu,
    openSettingsPanel,
} from './config.js';

test('show hidden properties test', async ({page}) => {
    await page.goto(getUrl());

    await loadSpecification(page, 'sample-interface-groups-specification.json');
    await loadDataflow(page, 'sample-interface-groups-dataflow.json');

    const nodeLocators = getNode(page, "TestNode");
    let node = nodeLocators.first();
    await assertPropertyCount(node, 1);

    const props = node.locator('.__properties > div');
    await props.click({button:"right", force: true})
    const hideButton = getContextMenu(page).getByText('Hide', {exact: true})
    await hideButton.click({force: true})

    await assertPropertyCount(node, 0);

    await openSettingsPanel(page);
    const checkbox = page.getByText('Show hidden properties', {exact: true});
    await checkbox.dispatchEvent("click");

    await assertPropertyCount(node, 1);
})
