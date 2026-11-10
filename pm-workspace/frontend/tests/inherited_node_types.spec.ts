import YAML from 'yaml';
import {
    test, expect, Page, Locator,
} from '@playwright/test';
import {
    setYAMLEditorContent, getYAMLEditorContent, getUrl, addProperty, checkForSubgraph, addSubgraph,
    addNode, enableEditingNodes, loadSpecification, loadDataflow, getNode,
} from './config.js';

test('checking inherited properties', async ({ page }) => {
    await page.goto(getUrl());
    await loadSpecification(page, 'sample-inheritance-specification.json');
    await loadDataflow(page, 'sample-inheritance-dataflow.json');

    await enableEditingNodes(page);

    const node = page.locator('[data-node-type="Type B"]');
    await node.locator('.__title').dblclick();

    const parsedContent = await getYAMLEditorContent(page);
    expect(parsedContent.properties.length).toBe(1);

    await addProperty(page, node);

    const modifiedParsedContent = await getYAMLEditorContent(page);
    expect(modifiedParsedContent.properties.length).toBe(2);
});
test('check added inherited property in spec', async ({ page }) => {
    await page.goto(getUrl());
    await loadSpecification(page, 'sample-inheritance-specification.json');

    await enableEditingNodes(page);

    await addNode(page, 'Classes', 'Type A', 400, 200);
    await addProperty(page, page.locator('[data-node-type="Type A"]'));
    await addNode(page, 'Classes', 'Type B', 700, 200, false);
    const nodeB = page.locator('[data-node-type="Type B"]');
    await nodeB.locator('.__title').dblclick();
    const parsedContent = await getYAMLEditorContent(page);
    expect(parsedContent.properties.length).toBe(1);
    const nodeBproperties = nodeB.locator('.__content > .__properties > div');
    expect(await nodeBproperties.count()).toBe(3);
});
test('checking renamed inherited property', async ({ page }) => {
    await page.goto(getUrl());
    await loadSpecification(page, 'sample-inheritance-specification.json');
    await loadDataflow(page, 'sample-inheritance-dataflow.json');

    await enableEditingNodes(page);

    const nodeA = page.locator('[data-node-type="Type A"]');
    await nodeA.locator('.__title').dblclick();

    const parsedContent = await getYAMLEditorContent(page);
    expect(parsedContent.properties.length).toBe(1);
    parsedContent.properties.find((prop) => prop.name === 'prop-a').name = 'prop-new';
    await setYAMLEditorContent(page, parsedContent);

    const nodeBproperties = page.locator('[data-node-type="Type B"]')
        .locator('.__content > .__properties > div');
    expect(await nodeBproperties.count()).toBe(2);
});

test('check override blocking', async ({ page }) => {
    await page.goto(getUrl());
    await loadSpecification(page, 'sample-inheritance-specification.json');
    await loadDataflow(page, 'sample-inheritance-dataflow.json');
    await enableEditingNodes(page);
    const textarea = page.locator('textarea');
    const applybutton = page.getByText('Apply', { exact: true });
    const applicable = () => applybutton.click({ trial: true, timeout: 1000 })
        .then(() => true).catch(() => false);

    const nodeB = getNode(page, 'Type B').locator('.__title').first();
    await nodeB.dblclick();
    const parsedContent = await getYAMLEditorContent(page);
    const illegalOverrideInterface = {
        ...parsedContent,
        ...Object.fromEntries([
            ['name', 'output-a'],
            ['type', 'Interface'],
            ['direction', 'inout'],
        ]),
    };
    await textarea.fill(YAML.stringify(illegalOverrideInterface));
    expect(await applicable()).not.toBeTruthy();
    const illegalOverrideProperty = {
        ...parsedContent,
        ...Object.fromEntries([
            ['name', 'prop-a'],
            ['type', 'text'],
            ['default', '""'],
        ]),
    };
    await textarea.fill(YAML.stringify(illegalOverrideProperty));
    expect(await applicable()).not.toBeTruthy();
});
test('override interface', async ({ page }) => {
    await page.goto(getUrl());
    await loadSpecification(page, 'sample-inheritance-specification.json');
    await loadDataflow(page, 'sample-inheritance-dataflow.json');

    await enableEditingNodes(page);

    const nodeB = getNode(page, 'Type B').locator('.__title').first();
    await nodeB.dblclick();

    const parsedContent = await getYAMLEditorContent(page);
    expect(parsedContent.interfaces.length).toBe(1);
    parsedContent.interfaces.push(Object.fromEntries([
        ['name', 'output-a'],
        ['type', 'Interface'],
        ['direction', 'inout'],
        ['override', true],
        ['side', 'right'],
    ]));
    await setYAMLEditorContent(page, parsedContent);
    const rightOutputs = getNode(page, 'Type B')
        .locator('.__content > .__interfaces > .__outputs > div');
    expect(await rightOutputs.count()).toBe(2);

    await nodeB.dblclick();
    const editedContent = await getYAMLEditorContent(page);
    expect(editedContent.interfaces.length).toBe(2);
    editedContent.interfaces.pop();
    await setYAMLEditorContent(page, editedContent);
    expect(await rightOutputs.count()).toBe(1);
});
test('override property', async ({ page }) => {
    await page.goto(getUrl());
    await loadSpecification(page, 'sample-inheritance-specification.json');
    await loadDataflow(page, 'sample-inheritance-dataflow.json');

    await enableEditingNodes(page);

    // inherits from B but overrides prop-a to be a checkbox
    const nodeB = getNode(page, 'Type B');
    const nodeBpropA = nodeB.getByTitle('prop-a');
    expect(nodeBpropA).toHaveClass('baklava-input');
    await nodeB.locator('.__title').dblclick();

    const parsedContent = await getYAMLEditorContent(page);
    expect(parsedContent.properties.length).toBe(1);
    parsedContent.properties.push(Object.fromEntries([
        ['name', 'prop-a'],
        ['type', 'hex'],
        ['override', true],
        ['default', '0xffff'],
    ]));
    await setYAMLEditorContent(page, parsedContent);
    const properties = getNode(page, 'Type B')
        .locator('.__content > .__properties > div');
    expect(await properties.count()).toBe(2);

    // check if prop type changed
    expect(nodeBpropA).toHaveClass('baklava-input hex-input');

    await nodeB.locator('.__title').dblclick();
    const editedContent = await getYAMLEditorContent(page);
    editedContent.properties.pop();
    await setYAMLEditorContent(page, editedContent);
    expect(await properties.count()).toBe(2);
    // check if rolls back to previous state
    expect(nodeBpropA).toHaveClass('baklava-input');
});
test('override child propagation', async ({ page }) => {
    await page.goto(getUrl());
    await loadSpecification(page, 'sample-inheritance-specification.json');
    await loadDataflow(page, 'sample-inheritance-dataflow.json');
    await enableEditingNodes(page);
    // inherits from B directly
    await addNode(page, 'Class', 'Type D', 500, 1000);
    const nodeD = getNode(page, 'Type D');
    const nodeA = getNode(page, 'Type A');
    const nodeB = getNode(page, 'Type B');
    await nodeD.locator('.__title').dblclick();
    const contentD = await getYAMLEditorContent(page);
    contentD.properties = [];
    await setYAMLEditorContent(page, contentD);
    // expect to fall back to type A property
    const nodeDpropA = nodeD.getByTitle('prop-a', { exact: true });
    expect(nodeDpropA).toHaveClass('baklava-input');

    await nodeA.locator('.__title').dblclick();
    const contentA = await getYAMLEditorContent(page);
    const propA = contentA.properties.find((p) => p.name === 'prop-a');
    propA.type = 'hex';
    propA.default = '0xffff';
    await setYAMLEditorContent(page, contentA);
    // expect to inherit the new type of interface
    expect(nodeDpropA).toHaveClass('baklava-input hex-input');

    await nodeB.locator('.__title').dblclick();
    const contentB = await getYAMLEditorContent(page);
    contentB.properties.push({
        name: 'prop-a',
        type: 'button-api',
        default: null,
        override: true,
    });
    await setYAMLEditorContent(page, contentB);
    expect(nodeDpropA).toHaveClass('baklava-button --block');
});
test('add subgraph to child node', async ({ page }) => {
    await page.goto(getUrl());
    await loadSpecification(page, 'sample-inheritance-specification.json');
    await loadDataflow(page, 'sample-inheritance-dataflow.json');

    await enableEditingNodes(page);

    const node = page.locator('[data-node-type="Type E"]');
    await addSubgraph(node, page);
    await checkForSubgraph(node, page);

    const outputs = await node.locator('.__content > .__interfaces > .__outputs > div').count();
    const inputs = await node.locator('.__content > .__interfaces > .__inputs > div').count();
    expect(inputs + outputs).toBe(4);
});
