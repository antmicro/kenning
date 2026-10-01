// Configuration file with the default configuration for playwright tests.
// @ts-check
import { fileURLToPath } from 'url';
import path, { dirname, join } from 'path';
import { expect } from 'playwright/test';

import os from 'os';
import fs from 'fs';
import YAML from 'yaml';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Configuration for path and directory of static HTML-based version of the frontend.
const config = {
    directory: './dist',
    file: 'index.html',
    directoryWithJsonFile: path.join(__dirname, '../../../examples'),
};

export const loadVideoNodeId = 'f50b4f2a-a2e2-4409-a5c9-891a8de44a5b';

export function getContextMenu(page) {
    return page.locator('.baklava-editor > .baklava-context-menu');
}
/**
 * Get an URL of the main Kenning Pipeline Manager page based on the configuration.
 * @returns {string} URL of the main page.
 */
export function getUrl() {
    return `file://${join(__dirname, `../${config.directory}/${config.file}`)}`;
}

/**
 * Get the URL to either a dataflow or specification JSON file from `examples` directory.
 * @param {string} filename
 * @returns {string} Path to the JSON file.
 */
export function getPathToJsonFile(filename) {
    return path.join(config.directoryWithJsonFile, filename);
}

/**
 * Get CSS property value from element.
 * @param {import('@playwright/test').Locator} element - DOM element
 * @param {string} name - Name of the target CSS property
 * @returns {Promise<string>}
 */
export async function getElementStyleAttribute(element,name)
{
    return await element.evaluate((el,_name)=> window.getComputedStyle(el).getPropertyValue(_name),name);
}

/**
 * Open the file chooser dialog for loading either a specification or dataflow file.
 * @param {import('@playwright/test').Page} page
 * @param {'specification' | 'dataflow'} purpose
 * @returns {Promise<import('@playwright/test').FileChooser>}
 */
export async function openFileChooser(page, purpose) {
    const fileChooserPromise = page.waitForEvent('filechooser');
    await page.mouse.move(25, 25);
    const text = purpose === 'specification' ? 'Load specification' : 'Load graph file';
    await page.getByText(text).click();
    return await fileChooserPromise;
}

/**
 * Add connection between source and target interface.
 * @param {import('@playwright/test').Page} page - The Playwright Page object to interact with.
 * @param {import('@playwright/test').Locator} sourceInterface - Source interface of connection
 * @param {import('@playwright/test').Locator} targetInterface - Target interface of connection
 * @returns {Promise<void>}
 */
export async function AddConnection(page, sourceInterface, targetInterface)
{
    const [sourcePosition, targetPosition] = await Promise.all(
        [sourceInterface, targetInterface]
            .map(async (locator) => {
                const bbox = await locator.boundingBox();

                if(bbox === null)
                {
                    return [0,0]
                }

                return [bbox.x + bbox.width / 2, bbox.y + bbox.height / 2];
            }));

    await page.mouse.move(sourcePosition[0], sourcePosition[1]);
    await page.mouse.down();
    await page.mouse.move(targetPosition[0], targetPosition[1], { steps: 2 });
    await page.mouse.up();
}

/**
 * Enable the navigation bar by simulating a mouse movement and clicking
 * on the element with the text "Show node browser".
 *
 * @async
 * @param {import('@playwright/test').Page} page - The Playwright Page object to interact with.
 * @returns {Promise<void>} Resolves when the navigation bar has been enabled.
 */
export async function enableNavigationBar(page) {
    await page.mouse.move(500, 0);
    const opened = page.locator('.hoverbox').filter({ hasText: /^Hide node browser$/ });
    if (await opened.isVisible()) {
        return;
    }
    await page
        .locator('.hoverbox')
        .filter({ hasText: /^Show node browser$/ })
        .first()
        .click();
}

/**
 * Disable the navigation bar by simulating a mouse movement and clicking
 * on the element with the text "Hide node browser".
 *
 * @async
 * @param {import('@playwright/test').Page} page - The Playwright Page object to interact with.
 * @returns {Promise<void>} Resolves when the navigation bar has been disabled.
 */
export async function disableNavigationBar(page) {
    await page.mouse.move(500, 0);
    const opened = page.locator('.hoverbox').filter({ hasText: /^Show node browser$/ });
    if (await opened.isVisible()) {
        return;
    }
    await page
        .locator('.hoverbox')
        .filter({ hasText: /^Hide node browser$/ })
        .first()
        .click();
}

/**
 * Loads specification file by using the file chooser.
 *
 * @async
 * @param {string} specificationFile - the name of specification file to load
 * @param {import('@playwright/test').Page} page - The Playwright Page object to interact with.
 * @returns {Promise<void>} Resolves when the specification has been loaded
 */
export async function loadSpecification(page, specificationFile) {
    const fileChooser = await openFileChooser(page, 'specification');
    await fileChooser.setFiles(getPathToJsonFile(specificationFile));
}

/**
 * Loads dataflow file by using the file chooser.
 *
 * @async
 * @param {string} dataflowFile - the name of dataflow file to load
 * @param {import('@playwright/test').Page} page - The Playwright Page object to interact with.
 * @returns {Promise<void>} Resolves when the dataflow has been loaded
 */
export async function loadDataflow(page, dataflowFile) {
    const fileChooser = await openFileChooser(page, 'dataflow');
    await fileChooser.setFiles(getPathToJsonFile(dataflowFile));
}

/**
 * Subgraph section
 */

/**
 * Leave current subgraph.
 *
 * @async
 * @param {import('@playwright/test').Page} page - The Playwright Page object to interact with.
 * @returns {Promise<void>} Resolves when playwright left current subgraph.
 */
export async function leaveSubgraph(page) {
    const leaveButton = page.getByText('Return from subgraph editor').locator('../..');
    await leaveButton.click();
}

/**
 * Get node title if not, fallback to node itself.
 *
 * @async
 * @param {import('@playwright/test').Locator} node - The Playwright Locator representing a node.
 * @returns {Promise<import('@playwright/test').Locator>} Locator of node title or node itself.
 */
async function getNodeTitle(node)
{
    const title = node.locator('.__title');

    if(await title.isVisible())
    {
        return title;
    }

    return node;
}

/**
 * Enter subgraph attached to a selected node.
 *
 * @async
 * @param {import('@playwright/test').Locator} node - The Playwright Locator representing a node.
 * @returns {Promise<void>} Resolves when playwright enter subgraph of selected node.
 */
export async function enterSubgraph(node, page) {
    const contextMenuOption = getContextMenu(page).getByText('Go to graph');
    if (!await contextMenuOption.isVisible()) {
        const title = await getNodeTitle(node);
        await title.click({ button: 'right' });
    }
    await contextMenuOption.click();
}

/**
 * Wait for subgraph to be loaded.
 *
 * @async
 * @param {import('@playwright/test').Page} page - The Playwright Page object to interact with.
 * @param {string} graphName - A name of the associated subraph.
 * @returns {Promise<void>} Resolves when subgraph is loaded.
 */
export async function waitForSubgraph(page, graphName) {
    const editorTitle = await page.locator('.editorTitle');
    await expect(editorTitle.getByText(graphName)).toBeVisible();
}

/**
 * Check whether a selected node has subgraph attached to it.
 *
 * @async
 * @param {import('@playwright/test').Locator} node - The Playwright Locator representing a node.
 * @returns {Promise<void>} Resolves when node has been found and has subgraph attached to it.
 */
export async function checkForSubgraph(node, page) {
    const contextMenuOption = getContextMenu(page).getByText('Go to graph');
    if (!await contextMenuOption.isVisible()) {
        const title = await getNodeTitle(node);
        await title.click({ button: 'right' });
    }
    expect(await contextMenuOption).toHaveCount(1,{ timeout: 10_000 });
    await (await getNodeTitle(node)).click({ button: 'right'});
    expect(await contextMenuOption).toBeVisible();
    await (await getNodeTitle(node)).click({ button: 'right' });
}

/**
 * Add subgraph to selected node.
 *
 * @async
 * @param {import('@playwright/test').Locator} node - The Playwright Locator representing a node.
 * @returns {Promise<void>} Resolves when subgraph has been added to selected node.
 */
export async function addSubgraph(node, page) {
    const title = await getNodeTitle(node);
    await title.click({ button: 'right' });
    const contextMenuOption = getContextMenu(page).getByText('Add subgraph');
    await contextMenuOption.click({force: true});
    expect(await title.locator('.__subgraph-icon')).toHaveCount(1);
}

/**
 * Wait until subgraph is added to node.
 *
 * @async
 * @param {import('@playwright/test').Locator} node - The Playwright Locator representing a node.
 * @returns {Promise<void>} Resolves when subgraph has been added to a node.
 */
export async function waitForNodeSubgraph(node) {
    const nodeTitle = await getNodeTitle(node);
    await expect(nodeTitle.locator('.__subgraph-icon')).toHaveCount(1);
}
/**
 * YAML section
 */

/**
 * Get node specification.
 *
 * @async
 * @param {import('@playwright/test').Page} page - The Playwright Page object to interact with.
 * @param {import('@playwright/test').Locator} node - The Playwright Locator representing a node.
 * @returns {Promise<object>} A parssed YAML settings.
 */
export async function getYAML(page, node) {
    await (await getNodeTitle(node)).dblclick();

    const textarea = page.locator('textarea');
    const content = YAML.parse(await textarea.evaluate((el) => el.value));
    return content;
}

/**
 * Set node specification.
 *
 * @async
 * @param {import('@playwright/test').Page} page - The Playwright Page object to interact with.
 * @param {object} content - A new node specification
 * @param {import('@playwright/test').Locator} node - The Playwright Locator representing a node.
 * @returns {Promise<void>} Resolves when a new specification has been set for a node.
 */
export async function setYAML(page, content, node) {
    await (await getNodeTitle(node)).dblclick();

    const textarea = page.locator('textarea');
    await textarea.fill(YAML.stringify(content));
}

/**
 * Set node specification.
 *
 * @async
 * @param {import('@playwright/test').Page} page - The Playwright Page object to interact with.
 * @param {object} content - A new node specification
 * @param {import('@playwright/test').Locator} node - The Playwright Locator representing a node.
 * @returns {Promise<void>} Resolves when a new specification has been set for a node.
 */
export async function closeYAMLEditor(page) {
    const sidebar = page.locator('.baklava-sidebar');
    await expect(sidebar).toBeVisible();
    await sidebar.locator('.__close').click();
}

export async function getYAMLEditorContent(page) {
    const textarea = page.locator('textarea');
    const value = await textarea.evaluate((el) => el.value);
    return YAML.parse(value);
}

export async function setYAMLEditorContent(page, content) {
    const textarea = page.locator('textarea');
    await textarea.fill(YAML.stringify(content));
    await page.getByText('Apply', { exact: true }).click();
}


/**
 * Interface and property section
 */

/**
 * Add a new interface to a selected node.
 *
 * @async
 * @param {import('@playwright/test').Page} page - The Playwright Page object to interact with.
 * @param {import('@playwright/test').Locator} node - The Playwright Locator representing a node.
 * @returns {Promise<void>} Resolves when a interface is added to a node.
 */
export async function addInterface(page, node) {
    await (await getNodeTitle(node)).click({ button: 'right', force: true });
    await getContextMenu(page).getByText('Add interface').click();
    await page.getByRole('button', { name: 'Add interface' }).click({ force: true });
}

/**
 * Add a new output interface to a selected node.
 *
 * @async
 * @param {import('@playwright/test').Page} page - The Playwright Page object to interact with.
 * @param {import('@playwright/test').Locator} node - The Playwright Locator representing a node.
 * @returns {Promise<void>} Resolves when a interface is added to a node.
 */
export async function addOutputInterface(page, node) {
    await (await getNodeTitle(node)).click({ button: 'right', force: true });
    await getContextMenu(page).getByText('Add interface').click();
    const select = page.getByTitle('Interface direction');
    await select.click();
    await select.locator('.__dropdown').getByText('output').click();
    await page.getByRole('button', { name: 'Add interface' }).click({ force: true });
}

/**
 * Add a new input interface to a selected node.
 *
 * @async
 * @param {import('@playwright/test').Page} page - The Playwright Page object to interact with.
 * @param {import('@playwright/test').Locator} node - The Playwright Locator representing a node.
 * @returns {Promise<void>} Resolves when a interface is added to a node.
 */
export async function addInputInterface(page, node) {
    await (await getNodeTitle(node)).click({ button: 'right', force: true });
    await getContextMenu(page).getByText('Add interface').click();
    const select = page.getByTitle('Interface direction');
    await select.click();
    await select.locator('.__dropdown').getByText('input').click();
    await page.getByRole('button', { name: 'Add interface' }).click({ force: true });
}


/**
 * Add a new property to a selected node.
 *
 * @async
 * @param {import('@playwright/test').Page} page - The Playwright Page object to interact with.
 * @param {import('@playwright/test').Locator} node - The Playwright Locator representing a node.
 * @returns {Promise<void>} Resolves when a property is added to a node.
 */
export async function addProperty(page, node) {
    await (await getNodeTitle(node)).click({ button: 'right', force: true });
    await getContextMenu(page).getByText('Add property').click();
    await page.getByRole('button', { name: 'Add property' }).click();
}

/**
 * Remove a property with propName of selected node.
 *
 * @async
 * @param {import('@playwright/test').Page} page - The Playwright Page object to interact with.
 * @param {import('@playwright/test').Locator} node - The Playwright Locator representing a node.
 * @param {string} propName - A property name.
 * @returns {Promise<void>} Resolves when a property has been removed.
 */
export async function deleteProperty(page, node, propName) {
    await (await getNodeTitle(node)).click({ button: 'right', force: true });
    await getContextMenu(page).getByText('Delete property').click();
    await page.locator('.create-menu').last().getByText(propName).click();
    await page.getByRole('button', { name: 'Remove properties' }).click();
}

/**
 * Check inputs count of a node.
 *
 * @async
 * @param {import('@playwright/test').Locator} node - The Playwright Locator representing a node.
 * @param {number} count - An expected number of inputs.
 * @returns {Promise<void>} Resolves when inputs count match expected count.
 */
export async function assertInputCount(node, count) {
    const inputs = await node
        .locator('.__interfaces .__inputs > div')
        .count();
    const positionedInputs = await node.
        locator('.baklava-node-interface-positioned > .--input')
        .count();
    expect(inputs+positionedInputs).toBe(count);
}

/**
 * Check inout count of a node.
 *
 * @async
 * @param {import('@playwright/test').Locator} node - The Playwright Locator representing a node.
 * @param {number} count - An expected number of inputs.
 * @returns {Promise<void>} Resolves when inputs count match expected count.
 */
export async function assertInoutCount(node, count) {
    const inputs = await node
        .locator('.__interfaces .__inputs > div')
        .count();
    const positionedInputs = await node.
        locator('.baklava-node-interface-positioned > .--inout')
        .count();
    expect(inputs+positionedInputs).toBe(count);
}

/**
 * Check outputs count of a node.
 *
 * @async
 * @param {import('@playwright/test').Locator} node - The Playwright Locator representing a node.
 * @param {number} count - An expected number of inputs.
 * @returns {Promise<void>} Resolves when outputs count match expected count.
 */
export async function assertOutputCount(node, count) {
    const inputs = await node
        .locator('.__interfaces .__outputs > div')
        .count();
    const positionedOutputs = await node.
        locator('.baklava-node-interface-positioned > .--output')
        .count();
    expect(inputs+positionedOutputs).toBe(count);
}

/**
 * Check property count of a node.
 *
 * @async
 * @param {import('@playwright/test').Locator} node - The Playwright Locator representing a node.
 * @param {number} count - An expected number of inputs.
 * @returns {Promise<void>} Resolves when property count match expected count.
 */
export async function assertPropertyCount(node, count) {
    const props = await node
        .locator('.__properties > div')
        .count();
    expect(props).toBe(count);
}

/**
 * Node section
 */

/**
 * Opens node palette
 *
 * @async
 * @param {import('@playwright/test').Page} page - The Playwright Page object to interact with.
 */
export async function openNodePalette(page) {
    await enableNavigationBar(page);
    const nodePalette = page.locator('.baklava-node-palette');
    const addNodeButton = nodePalette.getByText('New Node Type').first();
    expect(addNodeButton).toBeVisible();
    // because sometimes the page shifts which makes the mouse hover over the logo,
    // breaking the test (hiding some nodes).
    await page.mouse.move(300, 300);
}

/**
 * Add a node to the canvas by dragging it from the navigation bar.
 * @param {import('@playwright/test').Page} page - The Playwright page object.
 * @param {string} category - The category name in the navigation bar.
 * @param {string} nodeName - The name of the node to add.
 * @param {number} x - The x-coordinate to drop the node.
 * @param {number} y - The y-coordinate to drop the node.
 * @param {boolean} [openCategory] - Whether to click on category entry.
 * @returns {Promise<void>} Promise that resolves when the drag-and-drop operation is complete.
 */
export async function addNode(page, category, nodeName, x, y, openCategory = true) {
    const categoryBar = page.locator('.__title-label').getByText(category, { exact: true });
    const node = page.locator('.__title-label').getByText(nodeName, { exact: true }).first();

    // Open a proper category.
    await enableNavigationBar(page);
    await categoryBar.scrollIntoViewIfNeeded();
    await expect(categoryBar).toBeVisible();
    if (openCategory) await categoryBar.click();

    // Drag and drop to the [x, y] position.
    await dragAndDrop(page, node, x, y);

    try {
        await page.locator('.zoom-center').click({ timeout: 1000 });
    } catch {
        // not clickable, could be hidden by node config panel
    }
}

/**
 * Check for a node presents, show error message otherwise.
 *
 * @async
 * @param {boolean} exists - Whether a node should be visible.
 * @param {import('@playwright/test').Locator} node - The Playwright Locator representing a node.
 * @param {string} errorMessage - An error message.
 * @returns {Promise<void>} Resolves when node is present, show error message otherwise.
 */
export async function expectNode(exists, node, errorMessage) {
    if (exists) {
        expect(node, { message: errorMessage }).toBeVisible({ timeout: 5000 });
    } else {
        await node.waitFor({ state: 'hidden' });
        expect(node, { message: errorMessage }).not.toBeVisible({ timeout: 5000 });
    }
}

/**
 * Open a node and get its specification.
 *
 * @async
 * @param {import('@playwright/test').Page} page - The Playwright page object.
 * @param {import('@playwright/test').Locator} node - The Playwright Locator representing a node.
 * @returns {Promise<object>} A node specification.
 */
export async function reopenNode(page, node) {
    await (await getNodeTitle(node)).dblclick();
    return getYAMLEditorContent(page);
}

/**
 * Check node count.
 *
 * @async
 * @param {import('@playwright/test').Page} page - The Playwright page object.
 * @param {number} count - An expected number of nodes.
 * @returns {Promise<void>} Resolves when property count match expected count.
 */
export async function verifyNodeCount(page, count) {
    const nodes = page.locator('.node-container > div');
    await expect(nodes).toHaveCount(count); //toBe(expectedCount,{timeout: 3000});
}

/**
 * Get a nodes with a specific name.
 *
 * @param {import('@playwright/test').Page} page - The Playwright page object.
 * @param {string} name - node name to look for.
 * @returns {import('@playwright/test').Locator} A locator pointing to a nodes with specific names.
 */
export function getNode(page, name) {
    return page.locator(`.baklava-node[data-node-type="${name}"]`);
}

/**
 * Get a node with a specific id.
 *
 * @param {import('@playwright/test').Page} page - The Playwright page object.
 * @param {string} nodeId - id of desired node.
 * @returns {import('@playwright/test').Locator} A locator pointing to a node.
 */
export function getNodeByID(page, nodeId) {
    return page.locator(`#${nodeId}`);
}

/**
 * Delete specific node.
 *
 * @async
 * @param {import('@playwright/test').Locator} node - The Playwright Locator representing a node.
 * @returns {Promise<void>} Resolves when node is deleted.
 */
export async function deleteNode(node, page) {
    // Invoke a context menu with a right click.
    await (await getNodeTitle(node)).click({ button: 'right', force: true });

    // Delete the node.
    const deleteButton = getContextMenu(page).getByText('Delete', { exact: true });
    await deleteButton.click({ force: true });
}

/**
 * Opens settings panel
 * @param {import('@playwright/test').Page} page - The Playwright page object.
 */
export async function openSettingsPanel(page) {
    const settings = page.getByText('Show settings').locator('..').locator('..');
    await settings.hover({ force: true });
    expect(settings).toBeVisible();
    await settings.click();
}

/**
 * Create a new node.
 *
 * @param {import('@playwright/test').Page} page - The Playwright page object.
 * @returns {Promise<void>} Resolves when node is created.
 */
export async function createNewNodeType(page) {
    // Open node configuration menu
    const nodePalette = page.locator('.baklava-node-palette');
    const addNodeButton = nodePalette.getByText('New Node Type').first();
    await dragAndDrop(page, addNodeButton, 750, 80);

    // Create node
    const nodeMenu = page.locator('#container').locator('.create-menu');
    const createButton = nodeMenu.getByText('Create');
    await createButton.click();
}
/**
 * Enables the option to edit node types
 * @param {import('@playwright/test').Page} page - The Playwright page object.
 */
export async function enableEditingNodes(page) {
    // Assert that node types cannot be added.
    const logo = page.locator('.logo');
    await logo.hover();
    const addNodeButton = logo.locator('#create-new-node-type-button');
    expect(addNodeButton).toBeHidden();

    const checkbox = page.getByText('Modify node types');
    await checkbox.dispatchEvent("click");
}

/**
 * Enables the option to auto remove subgraphs
 * @param {import('@playwright/test').Page} page - The Playwright page object.
 */
export async function enableSubgraphRemoval(page) {
    // Assert that node types cannot be added.
    const logo = page.locator('.logo');
    await logo.hover();
    const addNodeButton = logo.locator('#create-new-node-type-button');
    expect(addNodeButton).toBeHidden();

    // Enable modifying node types.
    const settings = page.locator('.settings-panel');
    expect(settings).toBeVisible();
    await settings.hover({ force: true });

    const checkbox = page.getByText('Remove empty subgraphs');
    expect(checkbox).toBeVisible();
    await checkbox.dispatchEvent("click");
}

/**
 * Drag-and-drop operation on the specified locator element.
 *
 * @async
 * @param {import('playwright').Page} page - Playwright Page to perform actions on.
 * @param {import('playwright').Locator} locator - Locator of the element to drag.
 * @param {number} x - X-coordinate to move the mouse to (relative to the viewport).
 * @param {number} y - Y-coordinate to move the mouse to (relative to the viewport).
 * @returns {Promise<void>} Promise that resolves when the drag-and-drop operation is complete.
 */
export async function dragAndDrop(page, locator, x, y) {
    await locator.hover();
    await page.mouse.down();
    await page.mouse.move(x, y);
    await page.mouse.up();
}

/**
 * Closes the terminal so it doesn't cover page content.
 *
 * @async
 * @param {import('playwright').Page} page - Playwright Page to perform actions on.
 * @returns {Promise<void>} Promise that resolves when the terminal has been closed.
 */
export async function closeTerminal(page) {
    if (await page.isVisible('#hterm-terminal')) {
        await page.locator('.terminal-wrapper').locator('.container').getByRole('button').last().click();
    }
    await expect(page.locator('#hterm-terminal')).not.toBeVisible();
}
