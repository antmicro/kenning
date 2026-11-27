import {
  test, expect, Page,
} from '@playwright/test';
import {
  getUrl,
  loadSpecification,
  loadDataflow,
} from './config.js';
import assert from 'node:assert';

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


test("Autolayout test", async ({ page }) => {
  await page.goto(getUrl());

  await loadSpecification(page, "sample-network-specification.json");
  await expectNoErrors(page);
  await loadDataflow(page, "sample-network-dataflow.json");
  await expectNoErrors(page);

  const elements = await page.locator('.baklava-node').all()

  var previous_bounding_box = null;
  var theSame = null;

  // If the autolayout is not triggered all the nodes are stacked on top of each other.
  for (const li of elements) {
    const bb = await li.boundingBox()
    if (bb === null) {
      continue;
    }

    if (previous_bounding_box === null) {
      previous_bounding_box = bb;
      theSame = true;
    }
    else if (previous_bounding_box.x !== bb.x ||
      previous_bounding_box.y !== bb.y) {
      theSame = false;
      break;
    }
  }
  assert(!theSame)

})
