import YAML from 'yaml';
import {
    test, expect, Page,
    Locator,
} from '@playwright/test';
import {
    closeYAMLEditor, setYAMLEditorContent, getYAMLEditorContent, createNewNodeType,
    waitForSubgraph, reopenNode, getUrl, assertPropertyCount, assertOutputCount,
    assertInputCount, addInterface, getNode, addNode, dragAndDrop, enableEditingNodes,
    loadSpecification, loadDataflow, openNodePalette, getContextMenu, AddConnection,
} from './config.js';

async function expectNoErrors(page: Page) {
    const loading = page.locator('.loading-screen');
    await loading.waitFor({ state: 'hidden' });
    await expect(loading).not.toBeVisible();
    const notifications = page.locator(
        '.notifications > .panel > ul > *:not(:has(.info))',
    );
    const count = await notifications.count();
    expect(count).toBe(0);
}

async function checkIfYAMLPersists(page: any) {
    // Open a pop-up for the first node.
    const node = page
        .locator(`.baklava-node[data-node-type="Test node #1"] .__title`)
        .first();
    await node.dblclick();

    // Type a new text to the YAML editor.
    const newValue = 'ABC';
    const textarea = page.locator('textarea');
    await textarea.fill(newValue);

    // Switch to a different node.
    await page
        .locator(`.baklava-node[data-node-type="Test node #2"] .__title`)
        .first()
        .dblclick();

    // Get back and verify if the new value persists.
    await node.dblclick();
    await expect(textarea).toHaveValue(newValue);
}

async function addAndOpenNode(page: Page, group: string, nodeName: string, x = 750, y = 180) {
    await addNode(page, group, nodeName, x, y);
    const node = page.locator(`[data-node-type="${nodeName}"]`).last();
    await node.locator('.__title').dblclick();
    const content = await getYAMLEditorContent(page);
    return { node, content };
}

test('create new node type', async ({ page }) => {
    await page.goto(getUrl());

    await loadSpecification(page, 'sample-subgraph-specification.json');
    await loadDataflow(page, 'sample-subgraph-dataflow.json');

    await enableEditingNodes(page);
    await checkIfYAMLPersists(page);
});

test('add extends to a new node type', async ({ page }) => {
    await page.goto(getUrl());

    // Insatiate a new node.
    const nodeName = 'Custom Node';
    await openNodePalette(page);
    await createNewNodeType(page);
    await addNode(page, 'Default category', nodeName, 500, 80);

    // Double click the node to open the YAML editor.
    const customNode = page.locator('[data-node-type="Custom Node"]').first();
    await customNode.dblclick({ force: true });

    // Retrieve the initial content of the YAML editor.
    const content = await getYAMLEditorContent(page);
    // Add extends attribute to a newly created node
    content.extends = ["Filter2D"];
    await setYAMLEditorContent(page, content);

    const node = getNode(page,nodeName).nth(0);

    // check interfaces
    await assertOutputCount(node,1);
    await assertInputCount(node,2);

    // check for properties
    await assertPropertyCount(node,3)
});

test('remove extends from a node type',async ({page}) => {
    await page.goto(getUrl());

    // Insatiate a new node.
    const nodeName = 'Filter2D';

    const node = getNode(page,nodeName).nth(0);
    // check interfaces
    await assertOutputCount(node,1);
    await assertInputCount(node,2);

    // Double click the node to open the YAML editor.
    const customNode = page.locator('[data-node-type="Filter2D"]').first();
    await customNode.locator('.__title').dblclick({ force: true });

    // Retrieve the initial content of the YAML editor.
    const content = await getYAMLEditorContent(page);
    // Add extends attribute to a newly created node
    content.extends = [];
    await setYAMLEditorContent(page, content);

    // check interfaces
    await assertOutputCount(node,0);
    await assertInputCount(node,1);
});

test('adding interface from UI reflected in YAML editor', async ({ page }) => {
    await page.goto(getUrl());
    await loadSpecification(page, 'sample-subgraph-specification.json');
    await loadDataflow(page, 'sample-subgraph-dataflow.json');
    await enableEditingNodes(page);

    // Insatiate a new node.
    const nodeName = 'Custom Node';
    await openNodePalette(page);
    await createNewNodeType(page);
    await addNode(page, 'Default category', nodeName, 500, 80);

    // Double click the node to open the YAML editor.
    const customNode = page.locator('[data-node-type="Custom Node"]').first();
    await customNode.dblclick({ force: true });

    // Retrieve the initial content of the YAML editor.
    const parsedContent = await getYAMLEditorContent(page);
    await closeYAMLEditor(page);

    // Retrieve the modified content of the YAML editor.
    const node = getNode(page,nodeName).first();
    await addInterface(page, node);
    await assertInputCount(node, 1);
    const modifiedParsedContent = await getYAMLEditorContent(page);

    // Count the number of elements in the `interfaces` attribute.
    const initialInterfacesCount = parsedContent.interfaces.length;
    const modifiedInterfacesCount = modifiedParsedContent.interfaces.length;

    expect(initialInterfacesCount).toBe(0);
    expect(modifiedInterfacesCount).toBe(1);
});
test('adding interface to YAML', async ({ page }) => {
    await page.goto(getUrl());
    await loadSpecification(page, 'sample-include-specification.json');
    const nodeName = 'LoadVideo';
    const { content } = await addAndOpenNode(page, 'Filesystem', nodeName);
    const node = getNode(page,nodeName);
    await assertInputCount(node.first(), 0);
    content.interfaces.push({
        name: 'new_interface',
        type: 'unique',
        direction: 'input',
    });
    await setYAMLEditorContent(page, content);
    const first = node.nth(0);
    const second = node.nth(1);
    await assertInputCount(first, 1);
    await assertInputCount(second, 1);
    // check if newly added nodes have this change
    await addNode(page, 'Filesystem', nodeName, 750, 160, false);
    const third = node.nth(2);
    await assertInputCount(third, 1);
});
test('adding property to YAML', async ({ page }) => {
    await page.goto(getUrl());
    await loadSpecification(page, 'sample-include-specification.json');
    const nodeName = 'LoadVideo';
    const { content } = await addAndOpenNode(page, 'Filesystem', nodeName);

    const node = getNode(page,nodeName);
    const first = node.nth(0);
    const second = node.nth(1);
    await assertPropertyCount(first, 1);
    content.properties = [{
        name: 'new_property',
        type: 'integer',
        default: 0,
    }];
    await setYAMLEditorContent(page, content);
    await assertPropertyCount(first, 2);
    await assertPropertyCount(second, 2);
    // check if newly added nodes have this change
    await addNode(page, 'Filesystem', nodeName, 750, 160, false);
    const third = node.nth(2);
    await assertPropertyCount(third, 2);
});
test('removing interface from YAML', async ({ page }) => {
    await page.goto(getUrl());
    await loadSpecification(page, 'sample-include-specification.json');
    const nodeName = 'LoadVideo';
    const { content } = await addAndOpenNode(page, 'Filesystem', nodeName);
    const node = getNode(page,nodeName);
    const first = node.nth(0);
    const second = node.nth(1);
    await assertOutputCount(first, 1);
    await assertOutputCount(second, 1);
    content.interfaces = [];
    await setYAMLEditorContent(page, content);
    await assertOutputCount(first, 0);
    await assertOutputCount(second, 0);
    // check if newly added nodes have this change
    await addNode(page, 'Filesystem', nodeName, 750, 160, false);
    const third = node.nth(2);
    await assertOutputCount(third, 0);
});
test('editing property in YAML', async ({ page }) => {
    await page.goto(getUrl());
    await loadSpecification(page, 'sample-include-specification.json');
    const nodeName = 'GaussianKernel';
    const { node, content } = await addAndOpenNode(page, 'Generators', nodeName);

    content.properties.find((p) => p.name === 'size').name = 'width';
    Object.assign(
        content.properties.find((p) => p.name === 'sigma'),
        { type: 'bool', default: true },
    );
    await setYAMLEditorContent(page, content);
    await assertPropertyCount(node, 3);
    // check if newly added nodes have this change
    await addNode(page, 'Generators', nodeName, 750, 160, false);
    const second_node = getNode(page,nodeName).nth(1);
    await assertPropertyCount(second_node, 3);
    const changedProp = node.locator('.__properties > div').getByText('width');
    expect(changedProp).toBeVisible();
    const checkbox = node.locator('.__properties > div').locator('.baklava-checkbox');
    expect(checkbox).toBeVisible();
});
test('editing interface in YAML', async ({ page }) => {
    await page.goto(getUrl());
    await loadSpecification(page, 'sample-include-specification.json');
    const nodeName = 'LoadVideo';
    const { node, content } = await addAndOpenNode(page, 'Filesystem', nodeName);
    await assertInputCount(node, 0);
    await assertOutputCount(node, 1);

    content.interfaces.find((p) => p.name === 'frames').direction = 'input';
    await reopenNode(page,node);
    await setYAMLEditorContent(page, content);
    await assertInputCount(node, 1);
    await assertOutputCount(node, 0);

    content.interfaces.find((p) => p.name === 'frames').name = 'unique_input';
    await reopenNode(page,node);
    await setYAMLEditorContent(page, content);
    await assertInputCount(node, 1);
    await assertOutputCount(node, 0);
    expect(node.getByText('unique_input')).toBeVisible();

    // check if newly added nodes have this change
    await addNode(page, 'Filesystem', nodeName, 750, 160, false);
    const third = getNode(page,nodeName).nth(2);
    await assertInputCount(third, 1);
    await assertOutputCount(third, 0);
});
test('interface maxConnectionCount YAML', async ({ page }) => {
    await page.goto(getUrl());
    await loadSpecification(page, 'sample-include-specification.json');

    const connections = page.locator('.custom-connections-container > g');
    await expect(connections, {
        message: 'The initial conditions of presence of six connections are not met.',
    }).toHaveCount(6);

    const firstName = 'SaveFrame';
    await addNode(page, 'Filesystem', firstName, 0 , 0);
    const targetNode = getNode(page,firstName).last();
    // get SaveFrame input interface
    const colorInterface = targetNode.locator('.__interfaces .__inputs .__port').first();
    const secNode = 'GaussianKernel';
    const sourceNode = getNode(page,secNode);
    const kernelInterface = sourceNode.locator('.__interfaces .__outputs .__port').first();

    const content = await reopenNode(page,sourceNode);
    content.interfaces[0].maxConnectionsCount = 0;
    await setYAMLEditorContent(page, content);

    await AddConnection(page,kernelInterface,colorInterface);

    await expect(connections, {
        message: 'New connections has not been added.',
    }).toHaveCount(7);

    await reopenNode(page,sourceNode);
    content.interfaces[0].maxConnectionsCount = 1;
    await setYAMLEditorContent(page, content);

    await expect(connections, {
        message: 'Connections has no been overrode.',
    }).toHaveCount(6);
});
test('subgraph cascade interface YAML', async ({ page }) => {
    await page.goto(getUrl());
    await loadSpecification(page, 'sample-subgraph-specification.json');
    await loadDataflow(page, 'sample-subgraph-dataflow.json');
    await enableEditingNodes(page);
    const node = page.locator(`[data-node-type="Test subgraph #1"]`).first();
    const intf = node.getByText('Subgraph Output 2').locator('../..');
    expect(intf).toHaveClass('baklava-node-interface --output --connected');

    // enter subgraph
    await node.locator('.__title').click({ button: 'right' });
    const contextMenuOption = getContextMenu(page).getByText('Go to graph');
    await contextMenuOption.click();
    await waitForSubgraph(page, 'Test subgraph #1');

    // edit interface that is exposed in the subgraph
    const subnode = page.locator(`[data-node-type="Test node #1"]`).nth(1);
    subnode.locator('.__title').dblclick();
    const content = await getYAMLEditorContent(page);
    content.interfaces.find((i) => i.name === 'Output').type = 'non-valid';
    await setYAMLEditorContent(page, content);

    // leave
    const leaveButton = page.getByText('Return from subgraph editor').locator('../..');
    await leaveButton.click();

    expect(intf).not.toHaveClass('baklava-node-interface --output --connected');
});
test('subgraph cascade property YAML', async ({ page }) => {
    await page.goto(getUrl());
    await loadSpecification(page, 'sample-subgraph-specification.json');
    await loadDataflow(page, 'sample-subgraph-dataflow.json');
    await enableEditingNodes(page);
    const node = page.locator(`[data-node-type="Test subgraph #2"]`).first();
    const prop = node.locator('.__properties').locator('.baklava-node-interface');
    // first property type
    expect(prop.locator('.baklava-select')).toBeVisible();

    // enter subgraph
    await node.locator('.__title').click({ button: 'right' });
    const contextMenuOption = getContextMenu(page).getByText('Go to graph');
    await contextMenuOption.click();
    await waitForSubgraph(page, 'Test subgraph #2');

    // change prop
    const subnode = page.locator(`[data-node-type="Test node #2"]`);
    subnode.locator('.__title').dblclick();
    const content = await getYAMLEditorContent(page);
    const cprop = content.properties[0];
    delete cprop.values;
    cprop.type = 'integer';
    cprop.default = 2137;
    await setYAMLEditorContent(page, content);
    // leave
    const leaveButton = page.getByText('Return from subgraph editor').locator('../..');
    await leaveButton.click();
    // first property type
    expect(prop.locator('.baklava-select')).not.toBeVisible();
    expect(prop.locator('.baklava-num-input')).toBeVisible();
});
test('rename node type', async ({ page }) => {
    await page.goto(getUrl());

    await loadSpecification(page, 'sample-include-specification.json');

    await addNode(page, 'Filesystem', 'LoadVideo', 750, 80);
    const node = page.getByText('LoadVideo').locator('..').last();
    await node.dblclick();

    const textarea = page.locator('textarea');
    const nodeSpecification = await textarea.inputValue();
    const newSpecification = nodeSpecification.replace('LoadVideo', 'New node name');

    await textarea.fill(newSpecification);
    await page.getByText('Apply', { exact: true }).click();

    // assert that both nodes are renamed
    const editedNode = page.locator('[data-node-type="New node name"]');
    expect(await editedNode.count()).toBe(2);

    // assert that the node is renamed in node palette
    const nodePalette = page.locator('.baklava-node-palette');
    const category = nodePalette.getByText('Filesystem');

    await openNodePalette(page);
    await category.scrollIntoViewIfNeeded();
    await expect(category).toBeVisible();

    const newNodeEntry = nodePalette.getByText('New node name');
    expect(newNodeEntry).toBeVisible();
    const oldNodeEntry = nodePalette.getByText('LoadVideo', { exact: true });
    expect(oldNodeEntry).not.toBeAttached();

    // assert that both nodes have inherited properties
    expect(editedNode.first().getByText('filename')).toBeVisible();
    expect(editedNode.first().getByText('frames')).toBeVisible();
    expect(editedNode.nth(1).getByText('filename')).toBeVisible();
    expect(editedNode.nth(1).getByText('frames')).toBeVisible();

    // assert that a new node has inherited properties
    await dragAndDrop(page, newNodeEntry, 300, 300);
    expect(editedNode.nth(2).getByText('filename')).toBeVisible();
    expect(editedNode.nth(2).getByText('frames')).toBeVisible();
});
test('editing bus type interface', async ({ page }) => {
    await page.goto(getUrl());
    await loadSpecification(page, 'sample-bus-specification.json');
    await loadDataflow(page, 'sample-bus-dataflow.json');
    await enableEditingNodes(page);
    const node = getNode(page, 'Motherboard');
    const bus = node.locator('.__big-bus');
    const stubs = bus.locator('>div');
    expect(bus).toBeVisible();
    expect(stubs).toHaveCount(3);
    node.locator('.__title').dblclick();
    const content = await getYAMLEditorContent(page);
    const bigBus = content.interfaces.find((i) => i.name === 'control-bus');
    expect(bigBus).not.toBeUndefined();
    bigBus.bus.size = 200;
    await setYAMLEditorContent(page, content);
    expect(stubs).toHaveCount(3);
});
test('check yaml desync', async ({ page }) => {
    await page.goto(getUrl());
    await loadSpecification(page, 'sample-bus-specification.json');
    await loadDataflow(page, 'sample-bus-dataflow.json');
    await enableEditingNodes(page);
    const node = getNode(page, 'Memory');
    const title = node.locator('.__title');
    await title.dblclick();
    await title.click({ button: 'right' });
    const contextMenuOption = getContextMenu(page).getByText('Add property');
    await contextMenuOption.click();
    await page.locator('.create-menu').getByText('Add property').click();
    const sidebar = page.locator('.baklava-sidebar');
    await sidebar.locator('.__close').click();
    // wait for animation to finish
    await expectNoErrors(page);
});
