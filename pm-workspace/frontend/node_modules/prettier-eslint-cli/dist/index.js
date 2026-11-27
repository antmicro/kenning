#!/usr/bin/env node
// eslint-disable-next-line import/no-unassigned-import
"use strict";

require("./add-exception-handler");

var _loglevelColoredLevelPrefix = _interopRequireDefault(require("loglevel-colored-level-prefix"));

var _parser = _interopRequireDefault(require("./parser"));

var _formatFiles = _interopRequireDefault(require("./format-files"));

var _argv = _interopRequireDefault(require("./argv"));

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

// want to do this first
const logger = (0, _loglevelColoredLevelPrefix.default)({
  prefix: 'prettier-eslint-cli'
});
const args = process.argv.slice(2);
logger.trace('Parsing args: ', args);

let argv = _parser.default.parse(args);

argv = (0, _argv.default)(argv);
(0, _formatFiles.default)(argv);