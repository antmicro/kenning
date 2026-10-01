import {
    test, expect, Page, TestInfo,
} from '@playwright/test';

import os from 'os';
import fs from 'fs/promises';

import {
    createNewNodeType, addInterface, getUrl, assertInputCount, getNode, getPathToJsonFile, addNode,
    openFileChooser, dragAndDrop, openNodePalette, setYAMLEditorContent, getContextMenu,
    loadSpecification, loadDataflow, enableEditingNodes, addProperty,addOutputInterface,
    assertOutputCount, addInputInterface, assertInoutCount
} from './config.js';

const temporaryDir = `${os.tmpdir()}/`;

async function loadIncludeSpecification(page: Page, testInfo: TestInfo) {
    const fileChooser = await openFileChooser(page, 'specification');
    const specificationName = 'sample-include-specification.json';
    const specification = await fs.readFile(
        getPathToJsonFile(specificationName),
        { encoding: 'utf-8' },
    );
    const newSpecification = specification.replaceAll(
        'https://raw.githubusercontent.com/antmicro/kenning-pipeline-manager/main/examples/',
        `http://localhost:7001/`,
    );
    const newSpecificationPath =
        `${temporaryDir}sample-include-specification-worker-${testInfo.workerIndex}.json`;
    await fs.writeFile(newSpecificationPath, newSpecification);
    await fileChooser.setFiles(newSpecificationPath);
}

async function renameNodeType(page: Page, oldName: string, newName: string) {
    const node = page.getByText(oldName).last();
    await node.click({ button: 'right', force: true });
    await getContextMenu(page).getByText('Configure').click();
    await page.locator('.create-menu').getByTitle('Node name').first().fill(newName);
    await page.getByRole('button', { name: 'Configure' }).dblclick();
}

async function addParentAndChildNode(page: Page, coord: number, openCategory = true) {
    await addNode(page, 'Processing', 'Binary images', 750, 80, openCategory);

    await openNodePalette(page);
    const nodePalette = page.locator('.baklava-node-palette');

    const categoryNodeEntry = nodePalette.getByText('Binary images');
    const orient = openCategory ? 'right' : 'left';
    const categoryNodeButton = categoryNodeEntry.locator('../..').locator(`svg.arrow.${orient}.small`);
    await categoryNodeButton.scrollIntoViewIfNeeded();
    await expect(categoryNodeButton).toBeVisible();
    if (openCategory) await categoryNodeButton.click();

    const childNodeEntry = nodePalette.getByText('Logical AND');
    await expect(childNodeEntry).toBeVisible();
    await dragAndDrop(page, childNodeEntry, 500, coord);
}

async function saveSpecificationAs(page: Page, filenameWithoutExtension: string): Promise<string> {
    const logo = page.locator('.logo');
    await logo.hover();
    const saveAsMenuOption = page.getByRole('button', { name: 'Save specification as...' });
    await saveAsMenuOption.click();

    await page.getByPlaceholder('File name').first().fill(filenameWithoutExtension);
    const saveAsButton = page.getByRole('button', { name: 'Save' });

    const downloadPromise = page.waitForEvent('download');
    await saveAsButton.click();
    const download = await downloadPromise;

    const downloadedFilePath = temporaryDir + download.suggestedFilename();
    await download.saveAs(downloadedFilePath);

    return downloadedFilePath;
}

async function verifyNodePresence(page: Page, specificationPath: string, nodeName: string) {
    const specFile = await fs.readFile(specificationPath, 'utf-8');
    const specification = JSON.parse(specFile);

    expect(
        specification.nodes.filter(
            (node: any) =>
                node.name === nodeName &&
                Array.isArray(node.interfaces) &&
                node.interfaces.length === 0 &&
                Array.isArray(node.properties) &&
                node.properties.length === 0,
        ).length === 1,
    ).toBeTruthy();
}

test('enable editing', async ({ page }, testInfo) => {
    await page.goto(getUrl());
    await loadIncludeSpecification(page, testInfo);
    await openNodePalette(page);
});

test('check loopback connections', async ({ page }) => {
    await page.goto(getUrl());
    await loadSpecification(page, 'sample-loopback-specification.json');
    await loadDataflow(page, 'sample-loopback-dataflow.json');

    async function checkPathIsLoopback(path)
    {
        const d = await path.locator('.baklava-connection').first().getAttribute('d');
        const points = d.split(/\s/).filter((c)=> c !== '');
        return points.at(1) === points.at(-2) && points.at(2) === points.at(-1)
    }

    // Get connections in graph
    const connections = await page.locator('.custom-connections-container .baklava-connection').locator('..');
    expect(await connections.count()).toBe(6);
    const firstLoopback = await connections.nth(4);
    const secondLoopback = await connections.nth(5);

    // check loopback
    expect(await checkPathIsLoopback(firstLoopback)).toBeTruthy();
    expect(await checkPathIsLoopback(secondLoopback)).toBeTruthy();
});

test('create new node type', async ({ page }, testInfo) => {
    await page.goto(getUrl());
    await loadIncludeSpecification(page, testInfo);

    await openNodePalette(page);
    await createNewNodeType(page);
    await addNode(page, 'Default category', 'Custom Node', 750, 80);
});

test('check node resize with long title', async ({ page }) => {
    await page.goto(getUrl());

    await openNodePalette(page);
    await addNode(page, 'Filesystem', 'LoadVideo', 750, 80);
    const node = getNode(page, 'LoadVideo').last();

    await node.locator('.__title').click({ button: 'right'});
    await page.getByText('Rename').click();
    await node.locator('.__title .baklava-input').first().fill('Load Video AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA');
    await page.keyboard.press('Enter');

    const titleBox = await node.locator('.__title .__title-label').boundingBox();
    const nodeBox = await node.boundingBox();
    expect(nodeBox.width >= titleBox.width).toBeTruthy();
});

test('add interface to custom node in specification with "include" keyword', async ({ page }, testInfo) => {
    await page.goto(getUrl());
    await loadIncludeSpecification(page, testInfo);
    await openNodePalette(page);

    const nodeName = 'Custom Node';
    await createNewNodeType(page);
    await addNode(page, 'Default category', nodeName, 750, 80);
    const node = getNode(page,nodeName).last();
    await addInterface(page, node);
    await assertInputCount(node, 1);
});

test('register custom node in specification with "include" keyword', async ({ page }, testInfo) => {
    await page.goto(getUrl());
    await loadIncludeSpecification(page, testInfo);
    await openNodePalette(page);

    const nodeName = 'Custom Node';
    await createNewNodeType(page);
    await addNode(page, 'Default category', nodeName, 750, 80);
    const specificationPath = await saveSpecificationAs(page, 'new_specification');
    console.log(specificationPath);
    await verifyNodePresence(page, specificationPath, nodeName);
});

test('rename extending node', async ({ page }, testInfo) => {
    await page.goto(getUrl());
    await loadIncludeSpecification(page, testInfo);

    await addNode(page, 'Filesystem', 'LoadVideo', 750, 80);
    await openNodePalette(page);
    await renameNodeType(page, 'LoadVideo', 'New node name');

    // assert that both nodes are renamed
    const editedNode = page.locator('[data-node-type="New node name"]');
    expect(await editedNode.count()).toBe(2);

    // assert that the node is renamed in node palette
    const nodePalette = page.locator('.baklava-node-palette');

    await openNodePalette(page);

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

test('rename category node', async ({ page }, testInfo) => {
    await page.goto(getUrl());
    await loadIncludeSpecification(page, testInfo);

    await addParentAndChildNode(page, 200);

    await renameNodeType(page, 'Binary images', 'New node name');
    const and_node = getNode(page,'Logical AND');
    await assertInputCount(and_node, 2);

    // check category in custom sidebar
    const node = page.getByText('Logical AND').locator('..').last();
    await node.click({ button: 'right' });
    await getContextMenu(page).getByText('Details', { exact: true }).click();

    const parents = page.getByText('Generalize');
    const siblings = page.getByText('Choose other type');
    await expect(parents).toBeVisible();
    await expect(siblings).toBeVisible();
});

test('add interface to category node', async ({ page }, testInfo) => {
    await page.goto(getUrl());
    await loadIncludeSpecification(page, testInfo);

    await addParentAndChildNode(page, 200);

    const node = getNode(page,'Binary images');
    await addInterface(page, node);
    const and_node = getNode(page,'Logical AND');
    await assertInputCount(and_node, 3);

    await addParentAndChildNode(page, 200, false);
    await assertInputCount(node, 4);
    await assertInputCount(and_node, 6);
});

test('add inout interface to custom shape node', async ({ page }, testInfo) => {
    await page.goto(getUrl());
    await loadSpecification(page, 'sample-with-shape-specification.json');
    await loadDataflow(page, 'sample-with-shape-dataflow.json');

    const node = getNode(page,'Voltage');
    await assertInoutCount(node, 2);
    await addInterface(page, node);
    await page.mouse.click(370,360);
    await assertInoutCount(node, 3);
});

test('add input interface to custom shape node', async ({ page }, testInfo) => {
    await page.goto(getUrl());
    await loadSpecification(page, 'sample-with-shape-specification.json');
    await loadDataflow(page, 'sample-with-shape-dataflow.json');

    const node = getNode(page,'Voltage');
    await assertInputCount(node, 0);
    await addInputInterface(page, node);
    await page.mouse.click(370,360);
    await assertInputCount(node, 1);
});

test('add output interface to custom shape node', async ({ page }, testInfo) => {
    await page.goto(getUrl());
    await loadSpecification(page, 'sample-with-shape-specification.json');
    await loadDataflow(page, 'sample-with-shape-dataflow.json');

    const node = getNode(page,'Voltage');
    await assertOutputCount(node, 0);
    await addOutputInterface(page, node);
    await page.mouse.click(370,360);
    await assertOutputCount(node, 1);
});

test('hiding property', async ({ page }, testInfo) => {
    await page.goto(getUrl());
    await loadIncludeSpecification(page, testInfo);

    await addNode(page, 'Generators', 'GaussianKernel', 750, 80, true);
    const node = page.locator('[data-node-type="GaussianKernel"]').first();
    const nodePropertiesBefore = node.locator('.__content > .__properties > div');

    expect(await nodePropertiesBefore.count()).toBe(3);
    const sigmaProp = page.getByText('sigma').first();
    await sigmaProp.click({ button: 'right' });
    await getContextMenu(page).getByText('Hide').click();
    expect(await nodePropertiesBefore.count()).toBe(2);
    const nodeTitle = node.locator('.__title');
    await nodeTitle.dblclick();
    await page.locator('.baklava-sidebar').locator('.__property-button').click();
    expect(await nodePropertiesBefore.count()).toBe(3);
});

test('editing properties', async ({ page }, testInfo) => {
    await page.goto(getUrl());
    // eslint-disable-next-line max-len
    const TO = 500;
    const props: any = [
        ['constant', null, 'this is a test'],
        ['text',
            async (node) => {
                const inp = node.getByTitle('text');
                await inp.fill('edited cont', { timeout: TO });
                await expect(inp).toHaveValue('edited cont');
            }, 'this is a test'],
        ['multiline',
            async (node) => {
                const p = node.locator('.__markdown-content');
                await p.dblclick();
                const inp = node.getByTitle('multiline', { timeout: TO });
                await inp.fill('edited cont', { timeout: TO });
                await expect(inp).toHaveValue('edited cont');
            }, 'this is a test'],
        ['number',
            async (node) => {
                const inp = node.getByTitle('number').locator('../..');
                const val = await inp.locator('.__value').innerHTML();
                await inp.locator('.--inc', { timeout: TO }).click({ timeout: TO });
                const newVal = await inp.locator('.__value').innerHTML();
                await expect(newVal).not.toBe(val);
            }, 1234],
        ['integer',
            async (node) => {
                const inp = node.getByTitle('integer').locator('../..');
                const val = await inp.locator('.__value').innerHTML();
                await inp.locator('.--inc', { timeout: TO }).click({ timeout: TO });
                const newVal = await inp.locator('.__value').innerHTML();
                await expect(newVal).not.toBe(val);
            }, 1234],
        ['hex',
            async (node) => {
                const inp = node.getByTitle('hex');
                await inp.fill('0xffffff', { timeout: TO });
                await expect(inp).toHaveValue('0xffffff');
            }, '0x859'],
        ['select',
            async (node) => {
                const inp = node.getByTitle('select');
                await inp.click();
                await inp.getByText('volition').click({ timeout: TO });
                await expect(inp.locator('.__text')).toContainText('volition', { timeout: TO });
            }, 'drama', {
                values: [
                    'drama',
                    'conceptualization',
                    'visual calculus',
                    'volition',
                    'authority',
                ],
            }],
        ['bool',
            async (node) => {
                const inp = node.getByTitle('bool');
                await inp.click();
                await expect(inp).toHaveClass('baklava-checkbox --checked', { timeout: TO });
                await inp.click();
                await expect(inp).not.toHaveClass('baklava-checkbox --checked', { timeout: TO });
            }, false],
        ['slider',
            async (node) => {
                const inp = node.getByTitle('slider');
                const val = await inp.locator('.__value').innerHTML();
                const box = await inp.boundingBox();
                const startX = box.x + box.width / 2;
                const startY = box.y + box.height / 2;
                await page.mouse.move(startX, startY);
                await page.mouse.down();
                await page.mouse.move(startX + 10, startY);
                await page.mouse.up();
                const newVal = await inp.locator('.__value').innerHTML();
                await expect(newVal).not.toBe(val);
            }, 68, {
                min: 67,
                max: 69,
            }],
        ['list',
            async (node) => {
                const inp = node.getByTitle('list');
                await inp.fill('d e f g', { timeout: TO });
                await expect(inp).toHaveValue('d e f g');
            }, ['a', 'b', 'c'], {
                dtype: 'string',
            }],
        ['button-url', null, 'https://en.wikipedia.org/wiki/Insanity'],
        ['button-api', null, null],
    ];
    await openNodePalette(page);
    await createNewNodeType(page);
    const node = await getNode(page, 'Custom Node');
    node.dblclick();
    const content = (rdonly: boolean) => ({
        name: 'Custom Node',
        properties:
        props.map((parr) => ({
            readonly: rdonly,
            name: parr[0],
            type: parr[0],
            default: parr[2],
            ...(parr[3] ?? {}),
        })),
    });
    await setYAMLEditorContent(page, content(false));
    for (let i = 0; i < props.length; i += 1) {
        if (props[i][1] !== null) {
            const check = props[i][1] ?? (() => true);
            // eslint-disable-next-line no-await-in-loop
            await check(node);
        }
    }
    const nodeTitle = node.locator('.__title').first();
    nodeTitle.dblclick();
    await setYAMLEditorContent(page, { name: 'Custom Node' });
    nodeTitle.dblclick();
    await setYAMLEditorContent(page, content(true));
    for (let i = 0; i < props.length; i += 1) {
        if (props[i][1] !== null) {
            const check = props[i][1] ?? (() => { throw new Error(); });
            // eslint-disable-next-line no-await-in-loop
            const result = await check(node).then(() => false).catch(() => true);
            // inverting actionability tests to return inverse results
            expect(result, { message: '"'.concat(String(props[i][0]).concat('" interface is interactable when readonly')) }).toBeTruthy();
        }
    }
    addInterface(page, node);
    // takes some time for the interface to show up, otherwise fails
    await page.waitForTimeout(250);
    await assertInputCount(node, 1);
});
test('editing bus node', async ({ page }) => {
    await page.goto(getUrl());
    await loadSpecification(page, 'sample-bus-specification.json');
    await loadDataflow(page, 'sample-bus-dataflow.json');
    // await loadDataflow(page, 'sample-bus-dataflow.json');
    await enableEditingNodes(page);
    const busNode = await getNode(page, 'Motherboard');
    const ports = busNode.locator('.__port');
    const props = busNode.locator('.__properties > div');
    const buses = busNode.locator('.__port-bus');

    expect(await buses.count()).toBe(3);
    const prevPorts = await ports.count();
    const prevProps = await props.count();
    await addProperty(page, busNode);
    expect(await buses.count()).toBe(3);
    expect(await props.count()).toBe(prevProps + 1);
    // Has to be hard coded, no reactivity change it just is not responsive so
    // soon after previous edit.
    await page.waitForTimeout(100);
    await addInterface(page, busNode);
    expect(await buses.count()).toBe(3);
    expect(await ports.count()).toBe(prevPorts + 1);
});

test('generalize and specialize node', async ({page}) => {
    await page.goto(getUrl());
    const node = getNode(page,"Filter2D");
    await node.locator('.__title').dblclick();

    // Generalize selected node
    const replacePanel = page.locator('.baklava-sidebar .__content .__replace');
    await replacePanel.getByText("Color images").click();
    // It should be replaced by Color images so it should not be visible
    await expect(node).not.toBeVisible();
    // Check for Color images
    const generalizedNode = getNode(page,"Color images");
    await expect(generalizedNode).toBeVisible();

    // Specialize color images node
    await generalizedNode.locator('.__title').dblclick();
    await replacePanel.getByText("Filter2D").click();

    // It should be replaced by Filter2D so it should not be visible
    await expect(generalizedNode).not.toBeVisible();
    // Check for Filter2D
    await expect(node).toBeVisible();
});
