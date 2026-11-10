import { test, expect } from '@playwright/test';
import { getUrl, loadSpecification, loadDataflow } from './config.js';

test.describe("test popup closing", () => {
    test.beforeEach(async ({ page }) => {
        await page.goto(getUrl());
        await loadSpecification(page, 'sample-inheritance-specification.json');
        await loadDataflow(page, 'sample-inheritance-dataflow.json');

        await page.mouse.move(25, 25);
        await page.getByText('Save specification as...').click();

        await expect(page.getByText("Save configuration")).toBeVisible();
    });

    test.afterEach(async ({ page }) => {
        await expect(page.getByText("Save configuration")).toBeHidden();
    });

    // Tests if popup is not closed without an action
    test.fail("Control trial", async (_) => {});

    test("test popup closing with escape", async ({ page }) => {
        await page.keyboard.press("Escape");
    });

    test("test popup closing with button", async ({ page }) => {
        await page.locator(".__close:visible").click();
    });

    test("test popup closing with click outside", async ({ page }) => {
        await page.mouse.click(25, 25);
    });
});
