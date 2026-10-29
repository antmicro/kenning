"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _commonTags = require("common-tags");

var _loglevelColoredLevelPrefix = _interopRequireDefault(require("loglevel-colored-level-prefix"));

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

const logger = (0, _loglevelColoredLevelPrefix.default)({
  prefix: 'prettier-eslint-cli'
});
var _default = onUncaughtException;
exports.default = _default;

function onUncaughtException(err) {
  const level = logger.getLevel();
  const isTrace = level === 0;
  const traceResolution = (0, _commonTags.oneLine)`
    Run the script again with the LOG_LEVEL
    environment variable set to "trace"
  `;
  const resolutionSteps = [`${isTrace ? '✅ ' : '1.'} ${traceResolution}`, (0, _commonTags.oneLine)`
      2. Search existing issues on GitHub:
      ${(0, _commonTags.oneLineTrim)`
        https://github.com/prettier/prettier-eslint-cli/issues
        ?utf8=%E2%9C%93&q=${encodeURIComponent(err.message)}
      `}
    `, (0, _commonTags.oneLine)`
      3. Make a minimal reproduction in a totally separate repository.
      You can fork this one:
      https://github.com/kentcdodds/prettier-eslint-cli-repro
    `, (0, _commonTags.oneLine)`
      4. Post an issue with a link to your reproduction to the issues
      on GitHub: https://github.com/prettier/prettier-eslint-cli/issues/new
    `].join('\n  ');
  logger.error((0, _commonTags.oneLine)`
      There has been an unknown error when running the prettier-eslint CLI.
      If it's unclear to you what went wrong, then try this:
    `, `\n  ${resolutionSteps}`);
  throw err;
}