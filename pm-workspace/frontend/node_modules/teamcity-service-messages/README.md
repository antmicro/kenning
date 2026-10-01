# TeamCity Service Messages

[![Build Status](https://travis-ci.org/pifantastic/teamcity-service-messages.png)](https://travis-ci.org/pifantastic/teamcity-service-messages)

From the [TeamCity documentation][tcd]:

> If TeamCity doesn't support your testing framework or build runner out of the box, you
> can still avail yourself of many TeamCity benefits by customizing your build scripts to
> interact with the TeamCity server. This makes a wide range of features available to any
> team regardless of their testing frameworks and runners. Some of these features include
> displaying real-time test results and customized statistics, changing the build status,
> and publishing artifacts before the build is finished.

## Installation

```shell
npm install teamcity-service-messages --save
```

## Usage

### Chainable API

```javascript
var tsm = require("teamcity-service-messages");

// It doesn't really make sense that it's chainable. WHATEVER.
tsm.testStarted({ name: "my test" }).testFinished({ name: "my test" });

// You'll more likely use it like this:
tsm.message({ text: "Finished step 1" });

// Do some stuff.
tsm.message({ text: "Finished step 2" });
```

##### Output:

```
##teamcity[testStarted name='my test' flowId='65345909446478' timestamp='2013-12-19T19:54:24.449Z']
##teamcity[testFinished name='my test' flowId='65345909446478' timestamp='2013-12-19T19:54:24.449Z']
```

#### Methods

[Full Documentation][tcd]

- `blockOpened`/`blockClosed`
- `buildNumber`
- `buildProblem`
- `buildStatisticValue`
- `buildStatus`
- `compilationStarted`/`compilationFinished`
- `enableServiceMessages`/`disableServiceMessages`
- `importData`
- `inspection`
- `inspectionType`
- `message`
- `progressMessage`
- `progressStart`/`progressFinish`
- `publishArtifacts`
- `setParameter`
- `testFinished`
- `testIgnored`
- `testMetadata`
- `testStarted`/`testFailed`
- `testStdErr`
- `testStdOut`
- `testSuiteStarted`/`testSuiteFinished`

If you want the messages returned instead of printed to stdout, you can set `tsm.stdout = false`.
The API will no longer be chainable, but instead, each method will return its generated message.

```javascript
var tsm = require("teamcity-service-messages");

tsm.stdout = false;

typeof tsm.message("test"); // string
```

### Low-level API

```javascript
var Message = require("teamcity-service-messages").Message;

var message = new Message("testStarted", {
  name: "my test",
});

console.log(message.toString());
```

##### Output:

```
##teamcity[testStarted name='my test' flowId='65345909446478' timestamp='2013-12-19T19:54:24.449Z']
```

## Changelog

- `0.1.12`
  - [#22](https://github.com/pifantastic/teamcity-service-messages/pull/22): Add TypeScript type definitions
- `0.1.11`
  - [#20](https://github.com/pifantastic/teamcity-service-messages/pull/20): Fix reporting 0 as empty string
- `0.1.10`
  - [#17](https://github.com/pifantastic/teamcity-service-messages/pull/17): Add testMetadata method
- `0.1.9`
  - [#16](https://github.com/pifantastic/teamcity-service-messages/pull/16): Export inspectionType and inspection functions
- `0.1.8`
  - [#15](https://github.com/pifantastic/teamcity-service-messages/pull/15): Escape only characters specified in teamcity docs
- `0.1.7`
  - [#13](https://github.com/pifantastic/teamcity-service-messages/pull/13): `escape` now returns '' for `undefined`
- `0.1.6`
  - [#12](https://github.com/pifantastic/teamcity-service-messages/pull/12): Added global option to disable automatic flow ids
- `0.1.5`
  - [#10](https://github.com/pifantastic/teamcity-service-messages/pull/10): Fixed timestamp format: removed 'Z' at the end
- `0.1.4`
  - [#9](https://github.com/pifantastic/teamcity-service-messages/pull/9): Correct unicode padding
- `0.1.3`
  - [#7](https://github.com/pifantastic/teamcity-service-messages/issues/7): Single attribute messages
  - [#5](https://github.com/pifantastic/teamcity-service-messages/pull/5): Added `buildProblem` method
  - [#6](https://github.com/pifantastic/teamcity-service-messages/pull/6): Minimized npm package via `.npmignore`
  - [#4](https://github.com/pifantastic/teamcity-service-messages/pull/4): Updating documentation links to TC8, fixing codestyle
- `0.1.2`
  - [#3](https://github.com/pifantastic/teamcity-service-messages/pull/3): Allow timestamp to be overridden.
- `0.1.1`
  - [#2](https://github.com/pifantastic/teamcity-service-messages/pull/2): Moved `nodeunit` to `devDependencies`
  - [#1](https://github.com/pifantastic/teamcity-service-messages/pull/1): Mention repository in package.json
- `0.1.0`
  - Chainable API
  - `Message` class moved to `Message` namespace.
- `0.0.1`
  - Initial release

## License

Copyright (c) 2013 Aaron Forsander

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

[tcd]: https://www.jetbrains.com/help/teamcity/build-script-interaction-with-teamcity.html
