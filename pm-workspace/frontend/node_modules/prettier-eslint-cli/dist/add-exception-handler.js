"use strict";

var _uncaughtExceptionHandler = _interopRequireDefault(require("./uncaught-exception-handler"));

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/* istanbul ignore-next */
process.on('uncaughtException', _uncaughtExceptionHandler.default);