/*
 * Copyright (c) 2022-2024 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Functionality used as a node script to validate specification and dataflows
 * without having to run the application in a browser.
 *
 * Returns:
 * - 0 if validation was successful
 * - 1 if specification was invalid
 * - 2 if dataflow was invalid
 * - 3 if input was invalid
 */

/* eslint-disable */
import fs from 'fs';
import http from 'node:http';
import path from 'node:path';
import { parseArgs } from "node:util";
import EditorManager from './src/core/EditorManager.js';

const {
  values,
  positionals,
} = parseArgs({
    allowPositionals: true,
    options: {
        help: {
            type: "boolean",
            short: "h",
        },
        resolvedSpecification: {
            type: "string",
        },
        hostJsonUriSubs: {
            type: "string",
        }
    },
});

console.warn = function() {}; // Suppressing baklavajs logging

function printHelp() {
    console.log(
        'Pass a specification path and optional arguments\n\n' +
        'node validator.js specification_path.json [dataflow_path.json] [--resolvedSpecification resolved_spec_path.json]'
    )
}

if (values.h) {
    printHelp();
    process.exit(0);
}

if (positionals.length === 0) {
    printHelp();
    process.exit(3);
}


let server;
if (values.hostJsonUriSubs) {
    let hostJsonUriSubs = JSON.parse(values.hostJsonUriSubs);

    const hostJsonUriSubsPort = process.env.PM_JSONSUBS_PORT ?? 7001
    const address = `http://localhost:${hostJsonUriSubsPort}`;
    const jsonsubs = Object.fromEntries(Object.keys(hostJsonUriSubs).map((schema) => [schema, `${address}/${schema}/{}`]));
    process.env.VUE_APP_JSON_URL_SUBSTITUTES = JSON.stringify(jsonsubs);

    // Helper functions
    const trimLeadingSlash = (str) => str.replace(/^\//, '');
    function splitAtFirstSlash(str) {
        const idx = str.indexOf('/');
        if (idx === -1) return [str];
        return [str.slice(0, idx), str.slice(idx + 1)];
    }
    const send404 = (res) => {
        res.writeHead(404, {'Content-Type': 'text/plain'});
        res.write('Error 404: Resource not found.');
        res.end();
    }

    // Create simple http server
    http.createServer((req, res) => {
        if (req.method !== 'GET') return;

        // Overridden schema and a relative URL from schema://
        const [schema, target] = splitAtFirstSlash(trimLeadingSlash(req.url));
        if (!hostJsonUriSubs[schema]) {
            send404(res)
            return;
        }

        // Served directory,
        const { directory, route } = hostJsonUriSubs[schema];
        const relFilePath = target.replace(new RegExp(`^${trimLeadingSlash(route)}`), '');
        const filePath = path.resolve(directory + '/' + relFilePath);

        // Serve JSON if exists
        fs.lstat(filePath, (err, stats) => {
            if(err || !stats.isFile()){
                send404(res);
                return;
            }

            fs.createReadStream(filePath)
                .pipe(res.writeHead(200, {'Content-Type': 'application/json'}));
        });
    }).listen(hostJsonUriSubsPort);
}

const exit = (code) => {
    server?.close();
    process.exit(code);
}

fs.readFile(positionals[0], async function (err, spec) {
    if (err) {
        console.log(`\x1b[31mError reading specification file: ${err}\x1b[0m`);
        exit(1);
    }
    let validationError = EditorManager.validateSpecification(spec.toString());
    if (Array.isArray(validationError) && validationError.length) {
        console.log('Specification errors:')
        validationError.forEach((error) => console.log(`* \t\x1b[31m${error}\x1b[0m`))
        console.log('\x1b[31mSpecification invalid.\x1b[0m');
        exit(1);
    }

    const instance = EditorManager.getEditorManagerInstance();
    let { errors, warnings } = await instance.updateEditorSpecification(spec.toString());
    if (Array.isArray(warnings) && warnings.length) {
        console.log('Specification warnings:')
        warnings.forEach((warning) => console.log(`* \t\x1b[33m${warning}\x1b[0m`))
    }
    if (Array.isArray(errors) && errors.length) {
        console.log('Specification errors:')
        errors.forEach((error) => console.log(`* \t\x1b[31m${error}\x1b[0m`))
        console.log('\x1b[31mSpecification invalid.\x1b[0m');
        exit(1);
    }
    console.log('\x1b[32mSpecification valid.\x1b[0m');

    if (values.resolvedSpecification) {
        // Write resolved specification in a human-readable format
        fs.writeFileSync(values.resolvedSpecification, JSON.stringify(instance.specification.currentSpecification, null, 4));
    }

    if (positionals.length >= 2) {
        console.log('-----')
        console.log(`Validating ${positionals.length - 1} dataflows.`)
        let failing = 0;
        for (let i = 1; i < positionals.length; i++) {
            let filename = positionals[i];
            let dataflow;
            try {
                dataflow = fs.readFileSync(filename);
            } catch (error) {
                console.log(`\x1b[31mError reading dataflow file: ${error} [file: ${filename}]\x1b[0m`);
                failing++;
                continue;
            }
            ({ errors, warnings } = await instance.loadDataflow(dataflow.toString()));
            if (Array.isArray(warnings) && warnings.length) {
                console.log('Dataflow warnings:')
                warnings.forEach((warning) => console.log(`* \t\x1b[33m${warning}\x1b[0m`))
            }
            if (Array.isArray(errors) && errors.length) {
                console.log(`\x1b[31mDataflow invalid: ${filename}\x1b[0m`);
                console.log('\x1b[31mDataflow errors:\x1b[0m')
                errors.forEach((error) => console.log(`* \t\x1b[31m${error}\x1b[0m`))
                failing++;
                continue;
            }
            console.log(`\x1b[32mDataflow valid: ${filename}\x1b[0m`);
        }
        if (failing > 0) {
            console.log(`\x1b[31mInvalid dataflows: ${failing}\x1b[0m`);
            exit(2);
        }
    }
    exit(0);
});
