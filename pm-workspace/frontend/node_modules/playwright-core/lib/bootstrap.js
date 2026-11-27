"use strict";
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);
var bootstrap_exports = {};
__export(bootstrap_exports, {
  processExit: () => processExit
});
module.exports = __toCommonJS(bootstrap_exports);
const minimumMajorNodeVersion = 20;
const currentNodeVersion = process.versions.node;
const major = +currentNodeVersion.split(".")[0];
if (major < minimumMajorNodeVersion) {
  console.error(
    "You are running Node.js " + currentNodeVersion + `.
Playwright requires Node.js ${minimumMajorNodeVersion} or higher. 
Please update your version of Node.js.`
  );
  process.exit(1);
}
let stopProfiling = (callback) => callback();
if (process.env.PW_PROFILE_DIR) {
  const fs = require("fs");
  const path = require("path");
  const { Session } = require("inspector");
  const profileDir = path.resolve(process.env.PW_PROFILE_DIR);
  const script = process.argv[1] ? path.basename(process.argv[1], path.extname(process.argv[1])) : "node";
  const command = /^[a-z][\w-]*$/i.test(process.argv[2] ?? "") ? process.argv[2] : void 0;
  const profileName = [script, command, process.pid].filter(Boolean).join("-");
  const session = new Session();
  session.connect();
  session.post("Profiler.enable");
  session.post("Profiler.start");
  stopProfiling = (callback) => {
    stopProfiling = (callback2) => callback2();
    session.post("Profiler.stop", (error, result) => {
      try {
        if (!error) {
          fs.mkdirSync(profileDir, { recursive: true });
          fs.writeFileSync(path.join(profileDir, profileName + ".cpuprofile"), JSON.stringify(result.profile));
        }
      } finally {
        callback();
      }
    });
  };
  process.on("beforeExit", () => stopProfiling(() => {
  }));
}
function processExit(code) {
  stopProfiling(() => {
    process.exit(code);
  });
  return void 0;
}
if (process.env.PW_INSTRUMENT_MODULES) {
  const Module = require("module");
  const originalLoad = Module._load;
  const root = { name: "<root>", selfMs: 0, totalMs: 0, childrenMs: 0, children: [] };
  let current = root;
  const stack = [];
  Module._load = function(request, _parent, _isMain) {
    const node = { name: request, selfMs: 0, totalMs: 0, childrenMs: 0, children: [] };
    current.children.push(node);
    stack.push(current);
    current = node;
    const start = performance.now();
    let result;
    try {
      result = originalLoad.apply(this, arguments);
    } catch (e) {
      current = stack.pop();
      current.children.pop();
      throw e;
    }
    const duration = performance.now() - start;
    node.totalMs = duration;
    node.selfMs = Math.max(0, duration - node.childrenMs);
    current = stack.pop();
    current.childrenMs += duration;
    return result;
  };
  process.on("exit", () => {
    function printTree(node, prefix, isLast, lines2, depth) {
      if (node.totalMs < 1 && depth > 0)
        return;
      const connector = depth === 0 ? "" : isLast ? "\u2514\u2500\u2500 " : "\u251C\u2500\u2500 ";
      const time = `${node.totalMs.toFixed(1).padStart(8)}ms`;
      const self = node.children.length ? ` (self: ${node.selfMs.toFixed(1)}ms)` : "";
      lines2.push(`${time}  ${prefix}${connector}${node.name}${self}`);
      const childPrefix = prefix + (depth === 0 ? "" : isLast ? "    " : "\u2502   ");
      const sorted2 = node.children.slice().sort((a, b) => b.totalMs - a.totalMs);
      for (let i = 0; i < sorted2.length; i++)
        printTree(sorted2[i], childPrefix, i === sorted2.length - 1, lines2, depth + 1);
    }
    let totalModules = 0;
    function count(n) {
      totalModules++;
      n.children.forEach(count);
    }
    root.children.forEach(count);
    const lines = [];
    const sorted = root.children.slice().sort((a, b) => b.totalMs - a.totalMs);
    for (let i = 0; i < sorted.length; i++)
      printTree(sorted[i], "", i === sorted.length - 1, lines, 0);
    const totalMs = root.children.reduce((s, c) => s + c.totalMs, 0);
    process.stderr.write(`
--- Module load tree: ${totalModules} modules, ${totalMs.toFixed(0)}ms total ---
` + lines.join("\n") + "\n");
    const flat = /* @__PURE__ */ new Map();
    function gather(n) {
      const existing = flat.get(n.name);
      if (existing) {
        existing.selfMs += n.selfMs;
        existing.totalMs += n.totalMs;
        existing.count++;
      } else {
        flat.set(n.name, { selfMs: n.selfMs, totalMs: n.totalMs, count: 1 });
      }
      n.children.forEach(gather);
    }
    root.children.forEach(gather);
    const top50 = [...flat.entries()].sort((a, b) => b[1].selfMs - a[1].selfMs).slice(0, 50);
    const flatLines = top50.map(
      ([mod, { selfMs, totalMs: totalMs2, count: count2 }]) => `${selfMs.toFixed(1).padStart(8)}ms self ${totalMs2.toFixed(1).padStart(8)}ms total  (x${String(count2).padStart(3)})  ${mod}`
    );
    process.stderr.write(`
--- Top 50 modules by self time ---
` + flatLines.join("\n") + "\n");
  });
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  processExit
});
