"use strict";
var import_bootstrap = require("../../bootstrap");
var import_program = require("./program");
(0, import_program.program)().catch((e) => {
  console.error(e.message);
  (0, import_bootstrap.processExit)(1);
});
