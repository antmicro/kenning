/* eslint-disable no-multi-spaces */
import fs from 'fs/promises';
import {
    test, expect, Page,
    TestInfo,
} from '@playwright/test';
import { readFileSync, writeFileSync } from 'node:fs';
import {
    getUrl,
    loadSpecification,
    loadDataflow,
    openFileChooser,
    getPathToJsonFile,
    enableNavigationBar,
    getNode,
    openNodePalette,
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

async function deepCleanEditor(page: Page) {
    const settings = page.locator('.settings-panel');
    expect(settings).toBeVisible();
    await settings.hover({ force: true });

    const button = page.getByText('Clean editor');
    expect(button).toBeVisible();
    await button.dispatchEvent('click');
}

async function saveFileAs(
    page: Page,
    testInfo: TestInfo,
    purpose: 'specification' | 'dataflow',
    filenameWithoutExtension: string,
    selectGraphs: string[] = [],
): Promise<string> {
    const text = (purpose === 'specification' ? 'Save specification as...' : 'Save graph as file as...');
    const logo = page.locator('.logo');
    await logo.hover();

    await page.getByRole('button', { name: text }).click();

    const graphList = await page.locator('.graph_list');

    // eslint-disable-next-line no-restricted-syntax
    for (const graph of selectGraphs) {
        // eslint-disable-next-line no-await-in-loop
        await graphList.getByText(graph).first().click();
    }

    const uniqueName = `${filenameWithoutExtension}-${testInfo.workerIndex}-${Date.now()}`;

    await page.getByPlaceholder('File name').first().fill(uniqueName);

    const downloadPromise = page.waitForEvent('download');
    await page.getByRole('button', { name: 'Save' }).click();
    const download = await downloadPromise;

    const filePath = testInfo.outputPath(download.suggestedFilename());
    await download.saveAs(filePath);

    return filePath;
}

async function saveSpecificationAs(page: Page, testInfo: TestInfo, filename: string) {
    return saveFileAs(page, testInfo, 'specification', filename);
}

async function saveDataflowAs(
    page: Page,
    testInfo: TestInfo,
    filename: string,
    graphs: string[] = [],
) {
    return saveFileAs(page, testInfo, 'dataflow', filename, graphs);
}

async function loadIncludeSpecification(testInfo: TestInfo) {
    const specificationName = 'sample-include-specification.json';
    const specification = await fs.readFile(
        getPathToJsonFile(specificationName),
        { encoding: 'utf-8' },
    );
    const newSpecification = specification.replaceAll(
        'https://raw.githubusercontent.com/antmicro/kenning-pipeline-manager/main/examples/',
        'http://localhost:7001/',
    );
    const newSpecificationPath = testInfo.outputPath(specificationName);
    await fs.writeFile(newSpecificationPath, newSpecification);
    return newSpecificationPath;
}

async function loadSpecificationFromFile(page: Page, specificationFile: string) {
    const fileChooser = await openFileChooser(page, 'specification');
    await fileChooser.setFiles(specificationFile);
}

async function loadDatflowFromFile(page: Page, dataflowFile: string) {
    const fileChooser = await openFileChooser(page, 'dataflow');
    await fileChooser.setFiles(dataflowFile);
}

const examples = [
    { specification: 'sample-specification.json',                    dataflow: 'sample-dataflow.json'                     },
    { specification: 'sample-include-specification.json',            dataflow: 'sample-include-dataflow.json'             },
    { specification: 'sample-include-subgraph-specification.json',   dataflow: 'sample-include-subgraph-dataflow.json',   graphCount: 2 },
    { specification: 'sample-inheritance-specification.json',        dataflow: 'sample-inheritance-dataflow.json'         },
    { specification: 'sample-inout-specification.json',              dataflow: 'sample-inout-dataflow.json'               },
    { specification: 'sample-interface-groups-specification.json',   dataflow: 'sample-interface-groups-dataflow.json'    },
    { specification: 'sample-loopback-specification.json',           dataflow: 'sample-loopback-dataflow.json'            },
    { specification: 'sample-multiple-io-specification.json',        dataflow: 'sample-multiple-io-dataflow.json'         },
    { specification: 'sample-subgraph-specification.json',           dataflow: 'sample-subgraph-dataflow.json',           graphCount: 6 },
    { specification: 'sample-related-graph-specification.json',      dataflow: undefined,                                 graphCount: 3 },
    { specification: 'sample-dynamic-interfaces-specification.json', dataflow: 'sample-dynamic-interfaces-dataflow.json'  },
    { specification: 'sample-with-shape-specification.json',         dataflow: 'sample-with-shape-dataflow.json'          },
    { specification: 'sample-rectangle-grouping-specification.json', dataflow: 'sample-rectangle-grouping-dataflow.json'  },
    { specification: 'sample-styling-specification.json',            dataflow: 'sample-styling-dataflow.json'             },
    {
        specification: 'sample-huge-specification.json',
        dataflow: 'sample-huge-dataflow.json',
        timeout: 180_000,
    },
];

async function dragAndDropFile(page: Page, selector: string, fileName: string, testInfo: TestInfo) {
    const filePath = getPathToJsonFile(fileName);
    const buffer = readFileSync(filePath).toString('base64');

    const dataTransfer = await page.evaluateHandle(
        async ({ bufferData, localFileName, localFileType }) => {
            const dt = new DataTransfer();
            const blobData = await fetch(bufferData).then((res) => res.blob());
            const file = new File([blobData],
                localFileName,
                { type: localFileType },
            );
            dt.items.add(file);
            return dt;
        },
        {
            bufferData: `data:application/octet-stream;base64,${buffer}`,
            localFileName: testInfo.outputPath(fileName.substring(0, fileName.indexOf('.'))),
            localFileType: 'json',
        },
    );

    await page.waitForSelector(selector);
    await page.dispatchEvent(selector, 'drop', { dataTransfer });
    await page.locator('.loading-screen').waitFor({ state: 'hidden' });

    const bufferStr = readFileSync(filePath, 'utf8');
    const data = JSON.parse(bufferStr);
    if (data.entryGraph !== undefined) {
        return false;
    }
    return true;
}

examples.forEach(({ dataflow, specification, timeout, graphCount }) => {
    test(`spec loading ${specification}`, async ({ page }, testInfo) => {
        if (timeout) {
            test.setTimeout(timeout);
        }
        await page.goto(getUrl());
        if (specification === 'sample-include-specification.json') {
            const newSpecificationPath = await loadIncludeSpecification(testInfo);
            const fileChooser = await openFileChooser(page, 'specification');
            await fileChooser.setFiles(newSpecificationPath);
        } else {
            await loadSpecification(page, specification);
        }
        await expectNoErrors(page);
        const filepath = await saveSpecificationAs(page, testInfo, 'temp');
        await expectNoErrors(page);
        await loadSpecificationFromFile(page, filepath);
        await expectNoErrors(page);
    });

    if (!dataflow) return;

    test(`context loading ${dataflow}, ${specification}`, async ({ page }) => {
        if (timeout) {
            test.setTimeout(timeout);
        }
        await page.goto(getUrl());
        await loadSpecification(page, specification);
        await expectNoErrors(page);
        await loadDataflow(page, dataflow);
        await expectNoErrors(page);
        if (graphCount) {
            await enableNavigationBar(page);
            await page.locator('.tab').getByText('Graphs').click();
            await expect(page.locator('.entries').locator('.__entry-content')).toHaveCount(graphCount);
            await page.waitForTimeout(2000);
        }
    });
    test(`welcome loading ${dataflow}, ${specification}`, async ({ page }, testInfo) => {
        if (timeout) {
            test.setTimeout(timeout);
        }
        await page.goto(getUrl());
        await deepCleanEditor(page);
        const continueProcessing = await dragAndDropFile(page, '.welcome-container', specification, testInfo);
        await expectNoErrors(page);
        if (!continueProcessing) {
            return;
        }
        await dragAndDropFile(page, '.baklava-editor', dataflow, testInfo);
        await expectNoErrors(page);
    });

    test(`canvas loading ${dataflow}, ${specification}`, async ({ page }, testInfo) => {
        if (timeout) {
            test.setTimeout(timeout);
        }
        await page.goto(getUrl());
        await deepCleanEditor(page);
        await dragAndDropFile(page, '.baklava-editor', specification, testInfo);
        await expectNoErrors(page);
        await dragAndDropFile(page, '.baklava-editor', dataflow, testInfo);
        await expectNoErrors(page);
    });

    test(`context load+save ${dataflow}, ${specification}`, async ({ page }, testInfo) => {
        if (timeout) {
            test.setTimeout(timeout * 2);
        }
        await page.goto(getUrl());
        await loadSpecification(page, specification);
        await expectNoErrors(page);
        await loadDataflow(page, dataflow);
        await expectNoErrors(page);
        const filepath = await saveDataflowAs(page, testInfo, 'temp');
        await expectNoErrors(page);
        await loadDatflowFromFile(page, filepath);
        await expectNoErrors(page);
    });
});

test('save and load graph partially', async ({ page }, testInfo) => {
    await page.goto(getUrl());
    await loadSpecification(page, 'sample-subgraph-specification.json');
    await expectNoErrors(page);
    await loadDataflow(page, 'sample-subgraph-dataflow.json');
    const savedSub = 'Test subgraph #1';
    const filepath = await saveDataflowAs(page, testInfo, 'temp', [
        'Example of a graph with graph nodes',
        savedSub,
    ]);
    await expectNoErrors(page);
    await loadDatflowFromFile(page, filepath);
    await expectNoErrors(page);
    await enableNavigationBar(page);
    const paletteTitle = await page.locator('.palette-title');
    await paletteTitle.getByText('Graphs', { exact: true }).first().click();
    const entries = await page.locator('.entries');
    expect(entries.getByText(savedSub)).toBeVisible();
    await page.waitForTimeout(3000);
    expect(await entries.locator('.__entry').count()).toBe(1);
});

const exampleToFail = {
    specification: 'sample-specification.json',
};

async function expectErrors(page: Page) {
    const loading = page.locator('.loading-screen');
    await loading.waitFor({ state: 'hidden' });
    await expect(loading).not.toBeVisible();
    const notifications = page.locator(
        '.notifications > .panel > ul > *:not(:has(.info))',
    );
    const count = await notifications.count();
    expect(count).toBeGreaterThan(0);
}

test(`spec loading ${exampleToFail.specification} (check for fail)`, async ({ page }, testInfo) => {
    const json = JSON.parse(readFileSync(getPathToJsonFile(exampleToFail.specification), 'utf-8'));

    json.nodes[1].extends[0] = 'NonexistentType';

    const outputPath = testInfo.outputPath('failing-specification.json');
    writeFileSync(outputPath, JSON.stringify(json, null, 2), 'utf-8');

    await page.goto(getUrl());

    const fileChooserSpec = await openFileChooser(page, 'specification');
    await fileChooserSpec.setFiles(outputPath);

    await expectErrors(page);
});

const exampleNested = {
    specification: 'sample-include-subgraph-specification.json',
    dataflow: 'sample-include-subgraph-dataflow.json',
};
test(`entryGraph ${exampleNested.specification} in middle of structure`, async ({ page }, testInfo) => {
    const json = JSON.parse(readFileSync(getPathToJsonFile(exampleNested.dataflow), 'utf-8'));
    json.entryGraph = '1185eb1c-6dad-4c9d-98ff-dc31da9c8624';
    const outputPath = testInfo.outputPath('nested-subgraph-entry-specification.json');
    writeFileSync(outputPath, JSON.stringify(json, null, 2), 'utf-8');

    await page.goto(getUrl());
    await loadSpecification(page, exampleNested.specification);
    await expectNoErrors(page);
    const fileChooserSpec = await openFileChooser(page, 'dataflow');
    await fileChooserSpec.setFiles(outputPath);
    expect(getNode(page, 'MultipleIOGraphNode')).not.toBeVisible();
    await page.locator('span').getByText('Graphs').click();
    const entryCount = await page.locator('.entries').locator('.__entry').count();
    expect(entryCount).toBe(2);
});
