import { test, expect, Locator, Page } from '@playwright/test';
import { getUrl, loadVideoNodeId, closeTerminal, getContextMenu,
    getNode, getElementStyleAttribute, loadSpecification, loadDataflow, openSettingsPanel } from './config.js';


async function dumpConnectionsStartAndEnd(connections: Locator)
{
    const fromStartToEnd = [];
    const connectionsCount = await connections.count();

    for(let i = 0; i < connectionsCount; i++ )
    {
        const conn = await connections.nth(i);
        const paths = await conn.locator('path');

        const startPathD = await paths.first().getAttribute('d');
        const startPath = startPathD.split(/\s/).filter((c)=> c !== '');
        const endPathD = await paths.last().getAttribute('d');
        const endPath = endPathD.split(/\s/).filter((c)=> c !== '');

        const x1 = Number.parseFloat(startPath.at(1));
        const y1 = Number.parseFloat(startPath.at(2));

        const x2 = Number.parseFloat(endPath.at(-2));
        const y2 = Number.parseFloat(endPath.at(-1));

        fromStartToEnd.push({
            start: {
                x: x1,
                y: y1
            },
            end: {
                x: x2,
                y: y2
            }
        });
    }

    return fromStartToEnd;
}

function* chunked(arr: [], chunkSize: number)
{
    let i = 0;
    for (; i < arr.length; i += chunkSize) {
        const chunk = arr.slice(i, i + chunkSize);
        yield chunk;
    }
}

async function dumpConnectionSegments(connection: Locator)
{
    const segments = [];

    const paths = await connection.locator('.connection-wrapper');
    const pathsCount = await paths.count();

    for(let o = 1; o < pathsCount; o++ )
    {
        const pathElem = paths.nth(o);
        const pathD = await pathElem.getAttribute('d');
        const pathPoints = pathD.split(/\s/).filter((c)=> c !== '');

        const parssed = pathPoints.map(Number.parseFloat).filter((p) => !Number.isNaN(p));
        const chunks = [...chunked(parssed, 2)];
        const segm = [...chunked(chunks,2)];

        segments.push(...segm);
    }

    return segments;
}

async function getNodeBox(node: Locator)
{
    const left = await getElementStyleAttribute(node,'left');
    const top = await getElementStyleAttribute(node,'top');
    const width = await getElementStyleAttribute(node,'width');
    const height = await getElementStyleAttribute(node,'height');

    return {
        x: Number.parseFloat(left.replace('px','')),
        y: Number.parseFloat(top.replace('px','')),
        width: Number.parseFloat(width.replace('px','')),
        height: Number.parseFloat(height.replace('px',''))
    }
}

async function enableAStarConnections(page: Page)
{
    await openSettingsPanel(page);
    const connectionStyle = await page.getByTitle('Connection style');
    await connectionStyle.click();
    const dropdown = connectionStyle.locator('.__dropdown');

    await dropdown.getByText('A*-based').click({ force: true });
}

test('check connections exits', async ({ page }) => {
    await page.goto(getUrl());

    // get connections count
    const connections = await page.locator('.custom-connections-container > g');
    const connectionsCount = await connections.count();

    expect(connectionsCount).toBe(6);

    // switch to A Star connection count
    await enableAStarConnections(page);

    expect(await connections.count()).toBe(6);
});

test('check connections end', async ({ page }) => {
    await page.goto(getUrl());

    const connections = await page.locator('.custom-connections-container > g');
    // collect all starts and ends positions for all connections

    const orthogonalPaths = await dumpConnectionsStartAndEnd(connections);

    // switch to A Star connection count
    await enableAStarConnections(page);

    const aStarPaths = await dumpConnectionsStartAndEnd(connections);

    // compare paths
    const results = orthogonalPaths.every((path, index) =>
        path.start.x == aStarPaths[index].start.x &&
        path.start.y == aStarPaths[index].start.y &&
        path.end.x == aStarPaths[index].end.x &&
        path.end.y == aStarPaths[index].end.y
    );

    expect(results).toBeTruthy();
});

test('check for non horizontal or non vertical segments', async ({ page }) => {
    await page.goto(getUrl());

    // switch to A Star connection count
    await enableAStarConnections(page);

    const connections = await page.locator('.custom-connections-container > g');
    const connectionCount = await connections.count();

    for(let i = 0; i < connectionCount; i++ )
    {
        const segments = await dumpConnectionSegments(await connections.nth(i));
        segments.forEach((segm) => {
            const start = segm[0];
            const end = segm[1];

            const dx = start[0] - end[0];
            const dy = start[1] - end[1];

            expect(dx === 0 || dy === 0).toBeTruthy();
        });
    }
});

test('check for segments going in the opposite directions', async ({ page }) => {
    await page.goto(getUrl());

    // switch to A Star connection count
    await enableAStarConnections(page);

    const connections = await page.locator('.custom-connections-container > g');
    const connectionCount = await connections.count();

    function calcPointsDiff(point1, point2)
    {
        return [
            Math.abs(point1[0] - point2[0]),
            Math.abs(point1[1] - point2[1])
        ]
    }

    for(let i = 0; i < connectionCount; i++ )
    {
        const segments = await dumpConnectionSegments(await connections.nth(i));
        const segmentsCount = segments.length;
        for(let o = 0; o < segmentsCount-1; o ++)
        {
            const firstSegment = segments.at(o);
            const secondSegment = segments.at(o+1);

            const firstDiff = calcPointsDiff(firstSegment[0], firstSegment[1]);
            const secondDiff = calcPointsDiff(secondSegment[0], secondSegment[1]);

            if(firstDiff[0] === 0 && secondDiff[0] === 0)
            {
                const higherIndex = firstSegment[1][1] > firstSegment[0][1] ? 1 : 0;
                const p1 = firstSegment[1-higherIndex];
                const p2 = firstSegment[higherIndex];
                const p3 = secondSegment[1];
                expect(p3[1] <= p1[1] || p3[1] >= p2[1]).toBeTruthy();
            } else if(firstDiff[1] === 0 && secondDiff[1] === 0)
            {
                const higherIndex = firstSegment[1][0] > firstSegment[0][0] ? 1 : 0;
                const p1 = firstSegment[1-higherIndex];
                const p2 = firstSegment[higherIndex];
                const p3 = secondSegment[1];
                expect(p3[0] <= p1[0] || p3[0] >= p2[0]).toBeTruthy();
            }
        }
    }
});

test('check for collisions', async ({ page }) => {
    await page.goto(getUrl());

    // switch to A Star connection count
    await enableAStarConnections(page);

    const nodes = await page.locator('.baklava-node');
    const bboxes = [];

    function getMaxPointIndex(p1,p2)
    {
        if(p1[0] > p2[0] || p1[1] > p2[1])
        {
            return 0;
        }

        return 1;
    }

    function checkCollisions(bbox, line)
    {
        const maxIndex = getMaxPointIndex(line[0],line[1]);
        const p1 = line[1 - maxIndex];
        const p2 = line[maxIndex];

        const ret = (
            bbox.x < p2[0] &&
            bbox.x + bbox.width > p1[0] &&
            bbox.y < p2[1] &&
            bbox.y + bbox.height > p1[1]
        );

        return ret;
    }

    for(let i = 0; i < await nodes.count(); i++)
    {
        const bbox = await getNodeBox(nodes.nth(i));
        bboxes.push(bbox);
    }
    const connections = await page.locator('.custom-connections-container > g');
    const connectionCount = await connections.count();

    for(let i = 0; i < connectionCount; i++ )
    {
        const segments = await dumpConnectionSegments(await connections.nth(i));
        const segmentsCount = segments.length;
        // we do not check first and last segments in connection
        for(let o = 1; o < segmentsCount-1; o ++ )
        {
            const segm = segments.at(o);
            bboxes.forEach((bbox) => {
                expect(checkCollisions(
                    bbox, segm
                )).toBeFalsy();
            });
        }
    }
});