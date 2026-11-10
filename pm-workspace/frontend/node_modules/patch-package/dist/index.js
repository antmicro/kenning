"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const chalk_1 = __importDefault(require("chalk"));
const process_1 = __importDefault(require("process"));
const minimist_1 = __importDefault(require("minimist"));
const applyPatches_1 = require("./applyPatches");
const getAppRootPath_1 = require("./getAppRootPath");
const makePatch_1 = require("./makePatch");
const makeRegExp_1 = require("./makeRegExp");
const detectPackageManager_1 = require("./detectPackageManager");
const path_1 = require("./path");
const path_2 = require("path");
const slash = require("slash");
const ci_info_1 = require("ci-info");
const appPath = getAppRootPath_1.getAppRootPath();
const argv = minimist_1.default(process_1.default.argv.slice(2), {
    boolean: [
        "use-yarn",
        "case-sensitive-path-filtering",
        "reverse",
        "help",
        "version",
        "error-on-fail",
        "error-on-warn",
        "create-issue",
    ],
    string: ["patch-dir"],
});
const packageNames = argv._;
console.log(chalk_1.default.bold("patch-package"), 
// tslint:disable-next-line:no-var-requires
require(path_1.join(__dirname, "../package.json")).version);
if (argv.version || argv.v) {
    // noop
}
else if (argv.help || argv.h) {
    printHelp();
}
else {
    const patchDir = slash(path_2.normalize((argv["patch-dir"] || "patches") + path_2.sep));
    if (patchDir.startsWith("/")) {
        throw new Error("--patch-dir must be a relative path");
    }
    if (packageNames.length) {
        const includePaths = makeRegExp_1.makeRegExp(argv.include, "include", /.*/, argv["case-sensitive-path-filtering"]);
        const excludePaths = makeRegExp_1.makeRegExp(argv.exclude, "exclude", /^package\.json$/, argv["case-sensitive-path-filtering"]);
        const packageManager = detectPackageManager_1.detectPackageManager(appPath, argv["use-yarn"] ? "yarn" : null);
        const createIssue = argv["create-issue"];
        packageNames.forEach((packagePathSpecifier) => {
            makePatch_1.makePatch({
                packagePathSpecifier,
                appPath,
                packageManager,
                includePaths,
                excludePaths,
                patchDir,
                createIssue,
            });
        });
    }
    else {
        console.log("Applying patches...");
        const reverse = !!argv["reverse"];
        // don't want to exit(1) on postinstall locally.
        // see https://github.com/ds300/patch-package/issues/86
        const shouldExitWithError = !!argv["error-on-fail"] ||
            (process_1.default.env.NODE_ENV === "production" && ci_info_1.isCI) ||
            (ci_info_1.isCI && !process_1.default.env.PATCH_PACKAGE_INTEGRATION_TEST) ||
            process_1.default.env.NODE_ENV === "test";
        const shouldExitWithWarning = !!argv["error-on-warn"];
        applyPatches_1.applyPatchesForApp({
            appPath,
            reverse,
            patchDir,
            shouldExitWithError,
            shouldExitWithWarning,
        });
    }
}
function printHelp() {
    console.log(`
Usage:

  1. Patching packages
  ====================

    ${chalk_1.default.bold("patch-package")}

  Without arguments, the ${chalk_1.default.bold("patch-package")} command will attempt to find and apply
  patch files to your project. It looks for files named like

     ./patches/<package-name>+<version>.patch

  Options:

    ${chalk_1.default.bold("--patch-dir <dirname>")}

      Specify the name for the directory in which the patch files are located.
      
    ${chalk_1.default.bold("--error-on-fail")}
    
      Forces patch-package to exit with code 1 after failing.
    
      When running locally patch-package always exits with 0 by default.
      This happens even after failing to apply patches because otherwise 
      yarn.lock and package.json might get out of sync with node_modules,
      which can be very confusing.
      
      --error-on-fail is ${chalk_1.default.bold("switched on")} by default on CI.
      
      See https://github.com/ds300/patch-package/issues/86 for background.
      
    ${chalk_1.default.bold("--error-on-warn")}
    
      Forces patch-package to exit with code 1 after warning.
      
      See https://github.com/ds300/patch-package/issues/314 for background.

    ${chalk_1.default.bold("--reverse")}
        
      Un-applies all patches.

      Note that this will fail if the patched files have changed since being
      patched. In that case, you'll probably need to re-install 'node_modules'.

      This option was added to help people using CircleCI avoid an issue around caching
      and patch file updates (https://github.com/ds300/patch-package/issues/37),
      but might be useful in other contexts too.
      

  2. Creating patch files
  =======================

    ${chalk_1.default.bold("patch-package")} <package-name>${chalk_1.default.italic("[ <package-name>]")}

  When given package names as arguments, patch-package will create patch files
  based on any changes you've made to the versions installed by yarn/npm.

  Options:
  
    ${chalk_1.default.bold("--create-issue")}
    
       For packages whose source is hosted on GitHub this option opens a web
       browser with a draft issue based on your diff.

    ${chalk_1.default.bold("--use-yarn")}

        By default, patch-package checks whether you use npm or yarn based on
        which lockfile you have. If you have both, it uses npm by default.
        Set this option to override that default and always use yarn.

    ${chalk_1.default.bold("--exclude <regexp>")}

        Ignore paths matching the regexp when creating patch files.
        Paths are relative to the root dir of the package to be patched.

        Default: 'package\\.json$'

    ${chalk_1.default.bold("--include <regexp>")}

        Only consider paths matching the regexp when creating patch files.
        Paths are relative to the root dir of the package to be patched.

        Default '.*'

    ${chalk_1.default.bold("--case-sensitive-path-filtering")}

        Make regexps used in --include or --exclude filters case-sensitive.
    
    ${chalk_1.default.bold("--patch-dir")}

        Specify the name for the directory in which to put the patch files.
`);
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiaW5kZXguanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi9zcmMvaW5kZXgudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6Ijs7Ozs7QUFBQSxrREFBeUI7QUFDekIsc0RBQTZCO0FBQzdCLHdEQUErQjtBQUUvQixpREFBbUQ7QUFDbkQscURBQWlEO0FBQ2pELDJDQUF1QztBQUN2Qyw2Q0FBeUM7QUFDekMsaUVBQTZEO0FBQzdELGlDQUE2QjtBQUM3QiwrQkFBcUM7QUFDckMsK0JBQStCO0FBQy9CLHFDQUE4QjtBQUU5QixNQUFNLE9BQU8sR0FBRywrQkFBYyxFQUFFLENBQUE7QUFDaEMsTUFBTSxJQUFJLEdBQUcsa0JBQVEsQ0FBQyxpQkFBTyxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQUU7SUFDM0MsT0FBTyxFQUFFO1FBQ1AsVUFBVTtRQUNWLCtCQUErQjtRQUMvQixTQUFTO1FBQ1QsTUFBTTtRQUNOLFNBQVM7UUFDVCxlQUFlO1FBQ2YsZUFBZTtRQUNmLGNBQWM7S0FDZjtJQUNELE1BQU0sRUFBRSxDQUFDLFdBQVcsQ0FBQztDQUN0QixDQUFDLENBQUE7QUFDRixNQUFNLFlBQVksR0FBRyxJQUFJLENBQUMsQ0FBQyxDQUFBO0FBRTNCLE9BQU8sQ0FBQyxHQUFHLENBQ1QsZUFBSyxDQUFDLElBQUksQ0FBQyxlQUFlLENBQUM7QUFDM0IsMkNBQTJDO0FBQzNDLE9BQU8sQ0FBQyxXQUFJLENBQUMsU0FBUyxFQUFFLGlCQUFpQixDQUFDLENBQUMsQ0FBQyxPQUFPLENBQ3BELENBQUE7QUFFRCxJQUFJLElBQUksQ0FBQyxPQUFPLElBQUksSUFBSSxDQUFDLENBQUMsRUFBRTtJQUMxQixPQUFPO0NBQ1I7S0FBTSxJQUFJLElBQUksQ0FBQyxJQUFJLElBQUksSUFBSSxDQUFDLENBQUMsRUFBRTtJQUM5QixTQUFTLEVBQUUsQ0FBQTtDQUNaO0tBQU07SUFDTCxNQUFNLFFBQVEsR0FBRyxLQUFLLENBQUMsZ0JBQVMsQ0FBQyxDQUFD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Ozs7Ozs7TUFPaEMsZUFBSyxDQUFDLElBQUksQ0FBQyxvQkFBb0IsQ0FBQzs7Ozs7OztNQU9oQyxlQUFLLENBQUMsSUFBSSxDQUFDLGlDQUFpQyxDQUFDOzs7O01BSTdDLGVBQUssQ0FBQyxJQUFJLENBQUMsYUFBYSxDQUFDOzs7Q0FHOUIsQ0FBQyxDQUFBO0FBQ0YsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCBjaGFsayBmcm9tIFwiY2hhbGtcIlxuaW1wb3J0IHByb2Nlc3MgZnJvbSBcInByb2Nlc3NcIlxuaW1wb3J0IG1pbmltaXN0IGZyb20gXCJtaW5pbWlzdFwiXG5cbmltcG9ydCB7IGFwcGx5UGF0Y2hlc0ZvckFwcCB9IGZyb20gXCIuL2FwcGx5UGF0Y2hlc1wiXG5pbXBvcnQgeyBnZXRBcHBSb290UGF0aCB9IGZyb20gXCIuL2dldEFwcFJvb3RQYXRoXCJcbmltcG9ydCB7IG1ha2VQYXRjaCB9IGZyb20gXCIuL21ha2VQYXRjaFwiXG5pbXBvcnQgeyBtYWtlUmVnRXhwIH0gZnJvbSBcIi4vbWFrZVJlZ0V4cFwiXG5pbXBvcnQgeyBkZXRlY3RQYWNrYWdlTWFuYWdlciB9IGZyb20gXCIuL2RldGVjdFBhY2thZ2VNYW5hZ2VyXCJcbmltcG9ydCB7IGpvaW4gfSBmcm9tIFwiLi9wYXRoXCJcbmltcG9ydCB7IG5vcm1hbGl6ZSwgc2VwIH0gZnJvbSBcInBhdGhcIlxuaW1wb3J0IHNsYXNoID0gcmVxdWlyZShcInNsYXNoXCIpXG5pbXBvcnQgeyBpc0NJIH0gZnJvbSBcImNpLWluZm9cIlxuXG5jb25zdCBhcHBQYXRoID0gZ2V0QXBwUm9vdFBhdGgoKVxuY29uc3QgYXJndiA9IG1pbmltaXN0KHByb2Nlc3MuYXJndi5zbGljZSgyKSwge1xuICBib29sZWFuOiBbXG4gICAgXCJ1c2UteWFyblwiLFxuICAgIFwiY2FzZS1zZW5zaXRpdmUtcGF0aC1maWx0ZXJpbmdcIixcbiAgICBcInJldmVyc2VcIixcbiAgICBcImhlbHBcIixcbiAgICBcInZlcnNpb25cIixcbiAgICBcImVycm9yLW9uLWZhaWxcIixcbiAgICBcImVycm9yLW9uLXdhcm5cIixcbiAgICBcImNyZWF0ZS1pc3N1ZVwiLFxuICBdLFxuICBzdHJpbmc6IFtcInBhdGNoLWRpclwiXSxcbn0pXG5jb25zdCBwYWNrYWdlTmFtZXMgPSBhcmd2Ll9cblxuY29uc29sZS5sb2coXG4gIGNoYWxrLmJvbGQoXCJwYXRjaC1wYWNrYWdlXCIpLFxuICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tdmFyLXJlcXVpcmVzXG4gIHJlcXVpcmUoam9pbihfX2Rpcm5hbWUsIFwiLi4vcGFja2FnZS5qc29uXCIpKS52ZXJzaW9uLFxuKVxuXG5pZiAoYXJndi52ZXJzaW9uIHx8IGFyZ3Yudikge1xuICAvLyBub29wXG59IGVsc2UgaWYgKGFyZ3YuaGVscCB8fCBhcmd2LmgpIHtcbiAgcHJpbnRIZWxwKClcbn0gZWxzZSB7XG4gIGNvbnN0IHBhdGNoRGlyID0gc2xhc2gobm9ybWFsaXplKChhcmd2W1wicGF0Y2gtZGlyXCJdIHx8IFwicGF0Y2hlc1wiKSArIHNlcCkpXG4gIGlmIChwYXRjaERpci5zdGFydHNXaXRoKFwiL1wiKSkge1xuICAgIHRocm93IG5ldyBFcnJvcihcIi0tcGF0Y2gtZGlyIG11c3QgYmUgYSByZWxhdGl2ZSBwYXRoXCIpXG4gIH1cbiAgaWYgKHBhY2thZ2VOYW1lcy5sZW5ndGgpIHtcbiAgICBjb25zdCBpbmNsdWRlUGF0aHMgPSBtYWtlUmVnRXhwKFxuICAgICAgYXJndi5pbmNsdWRlLFxuICAgICAgXCJpbmNsdWRlXCIsXG4gICAgICAvLiovLFxuICAgICAgYXJndltcImNhc2Utc2Vuc2l0aXZlLXBhdGgtZmlsdGVyaW5nXCJdLFxuICAgIClcbiAgICBjb25zdCBleGNsdWRlUGF0aHMgPSBtYWtlUmVnRXhwKFxuICAgICAgYXJndi5leGNsdWRlLFxuICAgICAgXCJleGNsdWRlXCIsXG4gICAgICAvXnBhY2thZ2VcXC5qc29uJC8sXG4gICAgICBhcmd2W1wiY2FzZS1zZW5zaXRpdmUtcGF0aC1maWx0ZXJpbmdcIl0sXG4gICAgKVxuICAgIGNvbnN0IHBhY2thZ2VNYW5hZ2VyID0gZGV0ZWN0UGFja2FnZU1hbmFnZXIoXG4gICAgICBhcHBQYXRoLFxuICAgICAgYXJndltcInVzZS15YXJuXCJdID8gXCJ5YXJuXCIgOiBudWxsLFxuICAgIClcbiAgICBjb25zdCBjcmVhdGVJc3N1ZSA9IGFyZ3ZbXCJjcmVhdGUtaXNzdWVcIl1cbiAgICBwYWNrYWdlTmFtZXMuZm9yRWFjaCgocGFja2FnZVBhdGhTcGVjaWZpZXI6IHN0cmluZykgPT4ge1xuICAgICAgbWFrZVBhdGNoKHtcbiAgICAgICAgcGFja2FnZVBhdGhTcGVjaWZpZXIsXG4gICAgICAgIGFwcFBhdGgsXG4gICAgICAgIHBhY2thZ2VNYW5hZ2VyLFxuICAgICAgICBpbmNsdWRlUGF0aHMsXG4gICAgICAgIGV4Y2x1ZGVQYXRocyxcbiAgICAgICAgcGF0Y2hEaXIsXG4gICAgICAgIGNyZWF0ZUlzc3VlLFxuICAgICAgfSlcbiAgICB9KVxuICB9IGVsc2Uge1xuICAgIGNvbnNvbGUubG9nKFwiQXBwbHlpbmcgcGF0Y2hlcy4uLlwiKVxuICAgIGNvbnN0IHJldmVyc2UgPSAhIWFyZ3ZbXCJyZXZlcnNlXCJdXG4gICAgLy8gZG9uJ3Qgd2FudCB0byBleGl0KDEpIG9uIHBvc3RpbnN0YWxsIGxvY2FsbHkuXG4gICAgLy8gc2VlIGh0dHBzOi8vZ2l0aHViLmNvbS9kczMwMC9wYXRjaC1wYWNrYWdlL2lzc3Vlcy84NlxuICAgIGNvbnN0IHNob3VsZEV4aXRXaXRoRXJyb3IgPVxuICAgICAgISFhcmd2W1wiZXJyb3Itb24tZmFpbFwiXSB8fFxuICAgICAgKHByb2Nlc3MuZW52Lk5PREVfRU5WID09PSBcInByb2R1Y3Rpb25cIiAmJiBpc0NJKSB8fFxuICAgICAgKGlzQ0kgJiYgIXByb2Nlc3MuZW52LlBBVENIX1BBQ0tBR0VfSU5URUdSQVRJT05fVEVTVCkgfHxcbiAgICAgIHByb2Nlc3MuZW52Lk5PREVfRU5WID09PSBcInRlc3RcIlxuXG4gICAgY29uc3Qgc2hvdWxkRXhpdFdpdGhXYXJuaW5nID0gISFhcmd2W1wiZXJyb3Itb24td2FyblwiXVxuXG4gICAgYXBwbHlQYXRjaGVzRm9yQXBwKHtcbiAgICAgIGFwcFBhdGgsXG4gICAgICByZXZlcnNlLFxuICAgICAgcGF0Y2hEaXIsXG4gICAgICBzaG91bGRFeGl0V2l0aEVycm9yLFxuICAgICAgc2hvdWxkRXhpdFdpdGhXYXJuaW5nLFxuICAgIH0pXG4gIH1cbn1cblxuZnVuY3Rpb24gcHJpbnRIZWxwKCkge1xuICBjb25zb2xlLmxvZyhgXG5Vc2FnZTpcblxuICAxLiBQYXRjaGluZyBwYWNrYWdlc1xuICA9PT09PT09PT09PT09PT09PT09PVxuXG4gICAgJHtjaGFsay5ib2xkKFwicGF0Y2gtcGFja2FnZVwiKX1cblxuICBXaXRob3V0IGFyZ3VtZW50cywgdGhlICR7Y2hhbGsuYm9sZChcbiAgICBcInBhdGNoLXBhY2thZ2VcIixcbiAgKX0gY29tbWFuZCB3aWxsIGF0dGVtcHQgdG8gZmluZCBhbmQgYXBwbHlcbiAgcGF0Y2ggZmlsZXMgdG8geW91ciBwcm9qZWN0LiBJdCBsb29rcyBmb3IgZmlsZXMgbmFtZWQgbGlrZVxuXG4gICAgIC4vcGF0Y2hlcy88cGFja2FnZS1uYW1lPis8dmVyc2lvbj4ucGF0Y2hcblxuICBPcHRpb25zOlxuXG4gICAgJHtjaGFsay5ib2xkKFwiLS1wYXRjaC1kaXIgPGRpcm5hbWU+XCIpfVxuXG4gICAgICBTcGVjaWZ5IHRoZSBuYW1lIGZvciB0aGUgZGlyZWN0b3J5IGluIHdoaWNoIHRoZSBwYXRjaCBmaWxlcyBhcmUgbG9jYXRlZC5cbiAgICAgIFxuICAgICR7Y2hhbGsuYm9sZChcIi0tZXJyb3Itb24tZmFpbFwiKX1cbiAgICBcbiAgICAgIEZvcmNlcyBwYXRjaC1wYWNrYWdlIHRvIGV4aXQgd2l0aCBjb2RlIDEgYWZ0ZXIgZmFpbGluZy5cbiAgICBcbiAgICAgIFdoZW4gcnVubmluZyBsb2NhbGx5IHBhdGNoLXBhY2thZ2UgYWx3YXlzIGV4aXRzIHdpdGggMCBieSBkZWZhdWx0LlxuICAgICAgVGhpcyBoYXBwZW5zIGV2ZW4gYWZ0ZXIgZmFpbGluZyB0byBhcHBseSBwYXRjaGVzIGJlY2F1c2Ugb3RoZXJ3aXNlIFxuICAgICAgeWFybi5sb2NrIGFuZCBwYWNrYWdlLmpzb24gbWlnaHQgZ2V0IG91dCBvZiBzeW5jIHdpdGggbm9kZV9tb2R1bGVzLFxuICAgICAgd2hpY2ggY2FuIGJlIHZlcnkgY29uZnVzaW5nLlxuICAgICAgXG4gICAgICAtLWVycm9yLW9uLWZhaWwgaXMgJHtjaGFsay5ib2xkKFwic3dpdGNoZWQgb25cIil9IGJ5IGRlZmF1bHQgb24gQ0kuXG4gICAgICBcbiAgICAgIFNlZSBodHRwczovL2dpdGh1Yi5jb20vZHMzMDAvcGF0Y2gtcGFja2FnZS9pc3N1ZXMvODYgZm9yIGJhY2tncm91bmQuXG4gICAgICBcbiAgICAke2NoYWxrLmJvbGQoXCItLWVycm9yLW9uLXdhcm5cIil9XG4gICAgXG4gICAgICBGb3JjZXMgcGF0Y2gtcGFja2FnZSB0byBleGl0IHdpdGggY29kZSAxIGFmdGVyIHdhcm5pbmcuXG4gICAgICBcbiAgICAgIFNlZSBodHRwczovL2dpdGh1Yi5jb20vZHMzMDAvcGF0Y2gtcGFja2FnZS9pc3N1ZXMvMzE0IGZvciBiYWNrZ3JvdW5kLlxuXG4gICAgJHtjaGFsay5ib2xkKFwiLS1yZXZlcnNlXCIpfVxuICAgICAgICBcbiAgICAgIFVuLWFwcGxpZXMgYWxsIHBhdGNoZXMuXG5cbiAgICAgIE5vdGUgdGhhdCB0aGlzIHdpbGwgZmFpbCBpZiB0aGUgcGF0Y2hlZCBmaWxlcyBoYXZlIGNoYW5nZWQgc2luY2UgYmVpbmdcbiAgICAgIHBhdGNoZWQuIEluIHRoYXQgY2FzZSwgeW91J2xsIHByb2JhYmx5IG5lZWQgdG8gcmUtaW5zdGFsbCAnbm9kZV9tb2R1bGVzJy5cblxuICAgICAgVGhpcyBvcHRpb24gd2FzIGFkZGVkIHRvIGhlbHAgcGVvcGxlIHVzaW5nIENpcmNsZUNJIGF2b2lkIGFuIGlzc3VlIGFyb3VuZCBjYWNoaW5nXG4gICAgICBhbmQgcGF0Y2ggZmlsZSB1cGRhdGVzIChodHRwczovL2dpdGh1Yi5jb20vZHMzMDAvcGF0Y2gtcGFja2FnZS9pc3N1ZXMvMzcpLFxuICAgICAgYnV0IG1pZ2h0IGJlIHVzZWZ1bCBpbiBvdGhlciBjb250ZXh0cyB0b28uXG4gICAgICBcblxuICAyLiBDcmVhdGluZyBwYXRjaCBmaWxlc1xuICA9PT09PT09PT09PT09PT09PT09PT09PVxuXG4gICAgJHtjaGFsay5ib2xkKFwicGF0Y2gtcGFja2FnZVwiKX0gPHBhY2thZ2UtbmFtZT4ke2NoYWxrLml0YWxpYyhcbiAgICBcIlsgPHBhY2thZ2UtbmFtZT5dXCIsXG4gICl9XG5cbiAgV2hlbiBnaXZlbiBwYWNrYWdlIG5hbWVzIGFzIGFyZ3VtZW50cywgcGF0Y2gtcGFja2FnZSB3aWxsIGNyZWF0ZSBwYXRjaCBmaWxlc1xuICBiYXNlZCBvbiBhbnkgY2hhbmdlcyB5b3UndmUgbWFkZSB0byB0aGUgdmVyc2lvbnMgaW5zdGFsbGVkIGJ5IHlhcm4vbnBtLlxuXG4gIE9wdGlvbnM6XG4gIFxuICAgICR7Y2hhbGsuYm9sZChcIi0tY3JlYXRlLWlzc3VlXCIpfVxuICAgIFxuICAgICAgIEZvciBwYWNrYWdlcyB3aG9zZSBzb3VyY2UgaXMgaG9zdGVkIG9uIEdpdEh1YiB0aGlzIG9wdGlvbiBvcGVucyBhIHdlYlxuICAgICAgIGJyb3dzZXIgd2l0aCBhIGRyYWZ0IGlzc3VlIGJhc2VkIG9uIHlvdXIgZGlmZi5cblxuICAgICR7Y2hhbGsuYm9sZChcIi0tdXNlLXlhcm5cIil9XG5cbiAgICAgICAgQnkgZGVmYXVsdCwgcGF0Y2gtcGFja2FnZSBjaGVja3Mgd2hldGhlciB5b3UgdXNlIG5wbSBvciB5YXJuIGJhc2VkIG9uXG4gICAgICAgIHdoaWNoIGxvY2tmaWxlIHlvdSBoYXZlLiBJZiB5b3UgaGF2ZSBib3RoLCBpdCB1c2VzIG5wbSBieSBkZWZhdWx0LlxuICAgICAgICBTZXQgdGhpcyBvcHRpb24gdG8gb3ZlcnJpZGUgdGhhdCBkZWZhdWx0IGFuZCBhbHdheXMgdXNlIHlhcm4uXG5cbiAgICAke2NoYWxrLmJvbGQoXCItLWV4Y2x1ZGUgPHJlZ2V4cD5cIil9XG5cbiAgICAgICAgSWdub3JlIHBhdGhzIG1hdGNoaW5nIHRoZSByZWdleHAgd2hlbiBjcmVhdGluZyBwYXRjaCBmaWxlcy5cbiAgICAgICAgUGF0aHMgYXJlIHJlbGF0aXZlIHRvIHRoZSByb290IGRpciBvZiB0aGUgcGFja2FnZSB0byBiZSBwYXRjaGVkLlxuXG4gICAgICAgIERlZmF1bHQ6ICdwYWNrYWdlXFxcXC5qc29uJCdcblxuICAgICR7Y2hhbGsuYm9sZChcIi0taW5jbHVkZSA8cmVnZXhwPlwiKX1cblxuICAgICAgICBPbmx5IGNvbnNpZGVyIHBhdGhzIG1hdGNoaW5nIHRoZSByZWdleHAgd2hlbiBjcmVhdGluZyBwYXRjaCBmaWxlcy5cbiAgICAgICAgUGF0aHMgYXJlIHJlbGF0aXZlIHRvIHRoZSByb290IGRpciBvZiB0aGUgcGFja2FnZSB0byBiZSBwYXRjaGVkLlxuXG4gICAgICAgIERlZmF1bHQgJy4qJ1xuXG4gICAgJHtjaGFsay5ib2xkKFwiLS1jYXNlLXNlbnNpdGl2ZS1wYXRoLWZpbHRlcmluZ1wiKX1cblxuICAgICAgICBNYWtlIHJlZ2V4cHMgdXNlZCBpbiAtLWluY2x1ZGUgb3IgLS1leGNsdWRlIGZpbHRlcnMgY2FzZS1zZW5zaXRpdmUuXG4gICAgXG4gICAgJHtjaGFsay5ib2xkKFwiLS1wYXRjaC1kaXJcIil9XG5cbiAgICAgICAgU3BlY2lmeSB0aGUgbmFtZSBmb3IgdGhlIGRpcmVjdG9yeSBpbiB3aGljaCB0byBwdXQgdGhlIHBhdGNoIGZpbGVzLlxuYClcbn1cbiJdfQ==