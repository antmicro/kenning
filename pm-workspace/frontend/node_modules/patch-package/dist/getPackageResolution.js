"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.getPackageResolution = void 0;
const path_1 = require("./path");
const PackageDetails_1 = require("./PackageDetails");
const detectPackageManager_1 = require("./detectPackageManager");
const fs_extra_1 = require("fs-extra");
const lockfile_1 = require("@yarnpkg/lockfile");
const yaml_1 = __importDefault(require("yaml"));
const find_yarn_workspace_root_1 = __importDefault(require("find-yarn-workspace-root"));
const getPackageVersion_1 = require("./getPackageVersion");
const coerceSemVer_1 = require("./coerceSemVer");
function getPackageResolution({ packageDetails, packageManager, appPath, }) {
    if (packageManager === "yarn") {
        let lockFilePath = "yarn.lock";
        if (!fs_extra_1.existsSync(lockFilePath)) {
            const workspaceRoot = find_yarn_workspace_root_1.default();
            if (!workspaceRoot) {
                throw new Error("Can't find yarn.lock file");
            }
            lockFilePath = path_1.join(workspaceRoot, "yarn.lock");
        }
        if (!fs_extra_1.existsSync(lockFilePath)) {
            throw new Error("Can't find yarn.lock file");
        }
        const lockFileString = fs_extra_1.readFileSync(lockFilePath).toString();
        let appLockFile;
        if (lockFileString.includes("yarn lockfile v1")) {
            const parsedYarnLockFile = lockfile_1.parse(lockFileString);
            if (parsedYarnLockFile.type !== "success") {
                throw new Error("Could not parse yarn v1 lock file");
            }
            else {
                appLockFile = parsedYarnLockFile.object;
            }
        }
        else {
            try {
                appLockFile = yaml_1.default.parse(lockFileString);
            }
            catch (e) {
                console.error(e);
                throw new Error("Could not parse yarn v2 lock file");
            }
        }
        const installedVersion = getPackageVersion_1.getPackageVersion(path_1.join(path_1.resolve(appPath, packageDetails.path), "package.json"));
        const entries = Object.entries(appLockFile).filter(([k, v]) => k.startsWith(packageDetails.name + "@") &&
            // @ts-ignore
            coerceSemVer_1.coerceSemVer(v.version) === coerceSemVer_1.coerceSemVer(installedVersion));
        const resolutions = entries.map(([_, v]) => {
            // @ts-ignore
            return v.resolved;
        });
        if (resolutions.length === 0) {
            throw new Error(`\`${packageDetails.pathSpecifier}\`'s installed version is ${installedVersion} but a lockfile entry for it couldn't be found. Your lockfile is likely to be corrupt or you forgot to reinstall your packages.`);
        }
        if (new Set(resolutions).size !== 1) {
            console.warn(`Ambigious lockfile entries for ${packageDetails.pathSpecifier}. Using version ${installedVersion}`);
            return installedVersion;
        }
        if (resolutions[0]) {
            return resolutions[0];
        }
        const resolution = entries[0][0].slice(packageDetails.name.length + 1);
        // resolve relative file path
        if (resolution.startsWith("file:.")) {
            return `file:${path_1.resolve(appPath, resolution.slice("file:".length))}`;
        }
        if (resolution.startsWith("npm:")) {
            return resolution.replace("npm:", "");
        }
        return resolution;
    }
    else {
        const lockfile = require(path_1.join(appPath, packageManager === "npm-shrinkwrap"
            ? "npm-shrinkwrap.json"
            : "package-lock.json"));
        const lockFileStack = [lockfile];
        for (const name of packageDetails.packageNames.slice(0, -1)) {
            const child = lockFileStack[0].dependencies;
            if (child && name in child) {
                lockFileStack.push(child[name]);
            }
        }
        lockFileStack.reverse();
        const relevantStackEntry = lockFileStack.find((entry) => {
            if (entry.dependencies) {
                return entry.dependencies && packageDetails.name in entry.dependencies;
            }
            else if (entry.packages) {
                return entry.packages && packageDetails.path in entry.packages;
            }
            throw new Error("Cannot find dependencies or packages in lockfile");
        });
        const pkg = relevantStackEntry.dependencies
            ? relevantStackEntry.dependencies[packageDetails.name]
            : relevantStackEntry.packages[packageDetails.path];
        return pkg.resolved || pkg.version || pkg.from;
    }
}
exports.getPackageResolution = getPackageResolution;
if (require.main === module) {
    const packageDetails = PackageDetails_1.getPatchDetailsFromCliString(process.argv[2]);
    if (!packageDetails) {
        console.error(`Can't find package ${process.argv[2]}`);
        process.exit(1);
    }
    console.log(getPackageResolution({
        appPath: process.cwd(),
        packageDetails,
        packageManager: detectPackageManager_1.detectPackageManager(process.cwd(), null),
    }));
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZ2V0UGFja2FnZVJlc29sdXRpb24uanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi9zcmMvZ2V0UGFja2FnZVJlc29sdXRpb24udHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6Ijs7Ozs7O0FBQUEsaUNBQXNDO0FBQ3RDLHFEQUErRTtBQUMvRSxpRUFBNkU7QUFDN0UsdUNBQW1EO0FBQ25ELGdEQUE4RDtBQUM5RCxnREFBdUI7QUFDdkIsd0ZBQXdEO0FBQ3hELDJEQUF1RDtBQUN2RCxpREFBNkM7QUFFN0MsU0FBZ0Isb0JBQW9CLENBQUMsRUFDbkMsY0FBYyxFQUNkLGNBQWMsRUFDZCxPQUFPLEdBS1I7SUFDQyxJQUFJLGNBQWMsS0FBSyxNQUFNLEVBQUU7UUFDN0IsSUFBSSxZQUFZLEdBQUcsV0FBVyxDQUFBO1FBQzlCLElBQUksQ0FBQyxxQkFBVSxDQUFDLFlBQVksQ0FBQyxFQUFFO1lBQzdCLE1BQU0sYUFBYSxHQUFHLGtDQUFpQixFQUFFLENBQUE7WUFDekMsSUFBSSxDQUFDLGFBQWEsRUFBRTtnQkFDbEIsTUFBTSxJQUFJLEtBQUssQ0FBQywyQkFBMkIsQ0FBQyxDQUFBO2FBQzdDO1lBQ0QsWUFBWSxHQUFHLFdBQUksQ0FBQyxhQUFhLEVBQUUsV0FBVyxDQUFDLENBQUE7U0FDaEQ7UUFDRCxJQUFJLENBQUMscUJBQVUsQ0FBQyxZQUFZLENBQUMsRUFBRTtZQUM3QixNQUFNLElBQUksS0FBSyxDQUFDLDJCQUEyQixDQUFDLENBQUE7U0FDN0M7UUFDRCxNQUFNLGNBQWMsR0FBRyx1QkFBWSxDQUFDLFlBQVksQ0FBQyxDQUFDLFFBQVEsRUFBRSxDQUFBO1FBQzVELElBQUksV0FBVyxDQUFBO1FBQ2YsSUFBSSxjQUFjLENBQUMsUUFBUSxDQUFDLGtCQUFrQixDQUFDLEVBQUU7WUFDL0MsTUFBTSxrQkFBa0IsR0FBRyxnQkFBaUIsQ0FBQyxjQUFjLENBQUMsQ0FBQTtZQUM1RCxJQUFJLGtCQUFrQixDQUFDLElBQUksS0FBSyxTQUFTLEVBQUU7Z0JBQ3pDLE1BQU0sSUFBSSxLQUFLLENBQUMsbUNBQW1DLENBQUMsQ0FBQTthQUNyRDtpQkFBTTtnQkFDTCxXQUFXLEdBQUcsa0JBQWtCLENBQUMsTUFBTSxDQUFBO2FBQ3hDO1NBQ0Y7YUFBTTtZQUNMLElBQUk7Z0JBQ0YsV0FBVyxHQUFHLGNBQUksQ0FBQyxLQUFLLENBQUMsY0FBYyxDQUFDLENBQUE7YUFDekM7WUFBQyxPQUFPLENBQUMsRUFBRTtnQkFDVixPQUFPLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFBO2dCQUNoQixNQUFNLElBQUksS0FBSyxDQUFDLG1DQUFtQyxDQUFDLENBQUE7YUFDckQ7U0FDRjtRQUVELE1BQU0sZ0JBQWdCLEdBQUcscUNBQWlCLENBQ3hDLFdBQUksQ0FBQyxjQUFPLENBQUMsT0FBTyxFQUFFLGNBQWMsQ0FBQyxJQUFJLENBQUMsRUFBRSxjQUFjLENBQUMsQ0FDNUQsQ0FBQTtRQUVELE1BQU0sT0FBTyxHQUFHLE1BQU0sQ0FBQyxPQUFPLENBQUMsV0FBVyxDQUFDLENBQUMsTUFBTSxDQUNoRCxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLEVBQUUsQ0FDVCxDQUFDLENBQUMsVUFBVSxDQUFDLGNBQWMsQ0FBQyxJQUFJLEdBQUcsR0FBRyxDQUFDO1lBQ3ZDLGFBQWE7WUFDYiwyQkFBWSxDQUFDLENBQUMsQ0FBQyxPQUFPLENBQUMsS0FBSywyQkFBWSxDQUFDLGdCQUFnQixDQUFDLENBQzdELENBQUE7UUFFRCxNQUFNLFdBQVcsR0FBRyxPQUFPLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsRUFBRTtZQUN6QyxhQUFhO1lBQ2IsT0FBTyxDQUFDLENBQUMsUUFBUSxDQUFBO1FBQ25CLENBQUMsQ0FBQyxDQUFBO1FBRUYsSUFBSSxXQUFXLENBQUMsTUFBTSxLQUFLLENBQUMsRUFBRTtZQUM1QixNQUFNLElBQUksS0FBSyxDQUNiLEtBQUssY0FBYyxDQUFDLGFBQWEsNkJBQTZCLGdCQUFnQixpSUFBaUksQ0FDaE4sQ0FBQTtTQUNGO1FBRUQsSUFBSSxJQUFJLEdBQUcsQ0FBQyxXQUFXLENBQUMsQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUFFO1lBQ25DLE9BQU8sQ0FBQyxJQUFJLENBQ1Ysa0NBQWtDLGNBQWMsQ0FBQyxhQUFhLG1CQUFtQixnQkFBZ0IsRUFBRSxDQUNwRyxDQUFBO1lBQ0QsT0FBTyxnQkFBZ0IsQ0FBQTtTQUN4QjtRQUVELElBQUksV0FBVyxDQUFDLENBQUMsQ0FBQyxFQUFFO1lBQ2xCLE9BQU8sV0FBVyxDQUFDLENBQUMsQ0FBQyxDQUFBO1NBQ3RCO1FBRUQsTUFBTSxVQUFVLEdBQUcsT0FBTyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLEtBQUssQ0FBQyxjQUFjLENBQUMsSUFBSSxDQUFDLE1BQU0sR0FBRyxDQUFDLENBQUMsQ0FBQTtRQUV0RSw2QkFBNkI7UUFDN0IsSUFBSSxVQUFVLENBQUMsVUFBVSxDQUFDLFFBQVEsQ0FBQyxFQUFFO1lBQ25DLE9BQU8sUUFBUSxjQUFPLENBQUMsT0FBTyxFQUFFLFVBQVUsQ0FBQyxLQUFLLENBQUMsT0FBTyxDQUFDLE1BQU0sQ0FBQyxDQUFDLEVBQUUsQ0FBQTtTQUNwRTtRQUVELElBQUksVUFBVSxDQUFDLFVBQVUsQ0FBQyxNQUFNLENBQUMsRUFBRTtZQUNqQyxPQUFPLFVBQVUsQ0FBQyxPQUFPLENBQUMsTUFBTSxFQUFFLEVBQUUsQ0FBQyxDQUFBO1NBQ3RDO1FBRUQsT0FBTyxVQUFVLENBQUE7S0FDbEI7U0FBTTtRQUNMLE1BQU0sUUFBUSxHQUFHLE9BQU8sQ0FBQyxXQUFJLENBQzNCLE9BQU8sRUFDUCxjQUFjLEtBQUssZ0JBQWdCO1lBQ2pDLENBQUMsQ0FBQyxxQkFBcUI7WUFDdkIsQ0FBQyxDQUFDLG1CQUFtQixDQUN4QixDQUFDLENBQUE7UUFDRixNQUFNLGFBQWEsR0FBRyxDQUFDLFFBQVEsQ0FBQyxDQUFBO1FBQ2hDLEtBQUssTUFBTSxJQUFJLElBQUksY0FBYyxDQUFDLFlBQVksQ0FBQyxLQUFLLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLEVBQUU7WUFDM0QsTUFBTSxLQUFLLEdBQUcsYUFBYSxDQUFDLENBQUMsQ0FBQyxDQUFDLFlBQVksQ0FBQTtZQUMzQyxJQUFJLEtBQUssSUFBSSxJQUFJLElBQUksS0FBSyxFQUFFO2dCQUMxQixhQUFhLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFBO2FBQ2hDO1NBQ0Y7UUFDRCxhQUFhLENBQUMsT0FBTyxFQUFFLENBQUE7UUFDdkIsTUFBTSxrQkFBa0IsR0FBRyxhQUFhLENBQUMsSUFBSSxDQUFDLENBQUMsS0FBSyxFQUFFLEVBQUU7WUFDdEQsSUFBSSxLQUFLLENBQUMsWUFBWSxFQUFFO2dCQUN0QixPQUFPLEtBQUssQ0FBQyxZQUFZLElBQUksY0FBYyxDQUFDLElBQUksSUFBSSxLQUFLLENBQUMsWUFBWSxDQUFBO2FBQ3ZFO2lCQUFNLElBQUksS0FBSyxDQUFDLFFBQVEsRUFBRTtnQkFDekIsT0FBTyxLQUFLLENBQUMsUUFBUSxJQUFJLGNBQWMsQ0FBQyxJQUFJLElBQUksS0FBSyxDQUFDLFFBQVEsQ0FBQTthQUMvRDtZQUNELE1BQU0sSUFBSSxLQUFLLENBQUMsa0RBQWtELENBQUMsQ0FBQTtRQUNyRSxDQUFDLENBQUMsQ0FBQTtRQUNGLE1BQU0sR0FBRyxHQUFHLGtCQUFrQixDQUFDLFlBQVk7WUFDekMsQ0FBQyxDQUFDLGtCQUFrQixDQUFDLFlBQVksQ0FBQyxjQUFjLENBQUMsSUFBSSxDQUFDO1lBQ3RELENBQUMsQ0FBQyxrQkFBa0IsQ0FBQyxRQUFRLENBQUMsY0FBYyxDQUFDLElBQUksQ0FBQyxDQUFBO1FBQ3BELE9BQU8sR0FBRyxDQUFDLFFBQVEsSUFBSSxHQUFHLENBQUMsT0FBTyxJQUFJLEdBQUcsQ0FBQyxJQUFJLENBQUE7S0FDL0M7QUFDSCxDQUFDO0FBaEhELG9EQWdIQztBQUVELElBQUksT0FBTyxDQUFDLElBQUksS0FBSyxNQUFNLEVBQUU7SUFDM0IsTUFBTSxjQUFjLEdBQUcsNkNBQTRCLENBQUMsT0FBTyxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFBO0lBQ3BFLElBQUksQ0FBQyxjQUFjLEVBQUU7UUFDbkIsT0FBTyxDQUFDLEtBQUssQ0FBQyxzQkFBc0IsT0FBTyxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUE7UUFDdEQsT0FBTyxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQTtLQUNoQjtJQUNELE9BQU8sQ0FBQyxHQUFHLENBQ1Qsb0JBQW9CLENBQUM7UUFDbkIsT0FBTyxFQUFFLE9BQU8sQ0FBQyxHQUFHLEVBQUU7UUFDdEIsY0FBYztRQUNkLGNBQWMsRUFBRSwyQ0FBb0IsQ0FBQyxPQUFPLENBQUMsR0FBRyxFQUFFLEVBQUUsSUFBSSxDQUFDO0tBQzFELENBQUMsQ0FDSCxDQUFBO0NBQ0YiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgeyBqb2luLCByZXNvbHZlIH0gZnJvbSBcIi4vcGF0aFwiXG5pbXBvcnQgeyBQYWNrYWdlRGV0YWlscywgZ2V0UGF0Y2hEZXRhaWxzRnJvbUNsaVN0cmluZyB9IGZyb20gXCIuL1BhY2thZ2VEZXRhaWxzXCJcbmltcG9ydCB7IFBhY2thZ2VNYW5hZ2VyLCBkZXRlY3RQYWNrYWdlTWFuYWdlciB9IGZyb20gXCIuL2RldGVjdFBhY2thZ2VNYW5hZ2VyXCJcbmltcG9ydCB7IHJlYWRGaWxlU3luYywgZXhpc3RzU3luYyB9IGZyb20gXCJmcy1leHRyYVwiXG5pbXBvcnQgeyBwYXJzZSBhcyBwYXJzZVlhcm5Mb2NrRmlsZSB9IGZyb20gXCJAeWFybnBrZy9sb2NrZmlsZVwiXG5pbXBvcnQgeWFtbCBmcm9tIFwieWFtbFwiXG5pbXBvcnQgZmluZFdvcmtzcGFjZVJvb3QgZnJvbSBcImZpbmQteWFybi13b3Jrc3BhY2Utcm9vdFwiXG5pbXBvcnQgeyBnZXRQYWNrYWdlVmVyc2lvbiB9IGZyb20gXCIuL2dldFBhY2thZ2VWZXJzaW9uXCJcbmltcG9ydCB7IGNvZXJjZVNlbVZlciB9IGZyb20gXCIuL2NvZXJjZVNlbVZlclwiXG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRQYWNrYWdlUmVzb2x1dGlvbih7XG4gIHBhY2thZ2VEZXRhaWxzLFxuICBwYWNrYWdlTWFuYWdlcixcbiAgYXBwUGF0aCxcbn06IHtcbiAgcGFja2FnZURldGFpbHM6IFBhY2thZ2VEZXRhaWxzXG4gIHBhY2thZ2VNYW5hZ2VyOiBQYWNrYWdlTWFuYWdlclxuICBhcHBQYXRoOiBzdHJpbmdcbn0pIHtcbiAgaWYgKHBhY2thZ2VNYW5hZ2VyID09PSBcInlhcm5cIikge1xuICAgIGxldCBsb2NrRmlsZVBhdGggPSBcInlhcm4ubG9ja1wiXG4gICAgaWYgKCFleGlzdHNTeW5jKGxvY2tGaWxlUGF0aCkpIHtcbiAgICAgIGNvbnN0IHdvcmtzcGFjZVJvb3QgPSBmaW5kV29ya3NwYWNlUm9vdCgpXG4gICAgICBpZiAoIXdvcmtzcGFjZVJvb3QpIHtcbiAgICAgICAgdGhyb3cgbmV3IEVycm9yKFwiQ2FuJ3QgZmluZCB5YXJuLmxvY2sgZmlsZVwiKVxuICAgICAgfVxuICAgICAgbG9ja0ZpbGVQYXRoID0gam9pbih3b3Jrc3BhY2VSb290LCBcInlhcm4ubG9ja1wiKVxuICAgIH1cbiAgICBpZiAoIWV4aXN0c1N5bmMobG9ja0ZpbGVQYXRoKSkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKFwiQ2FuJ3QgZmluZCB5YXJuLmxvY2sgZmlsZVwiKVxuICAgIH1cbiAgICBjb25zdCBsb2NrRmlsZVN0cmluZyA9IHJlYWRGaWxlU3luYyhsb2NrRmlsZVBhdGgpLnRvU3RyaW5nKClcbiAgICBsZXQgYXBwTG9ja0ZpbGVcbiAgICBpZiAobG9ja0ZpbGVTdHJpbmcuaW5jbHVkZXMoXCJ5YXJuIGxvY2tmaWxlIHYxXCIpKSB7XG4gICAgICBjb25zdCBwYXJzZWRZYXJuTG9ja0ZpbGUgPSBwYXJzZVlhcm5Mb2NrRmlsZShsb2NrRmlsZVN0cmluZylcbiAgICAgIGlmIChwYXJzZWRZYXJuTG9ja0ZpbGUudHlwZSAhPT0gXCJzdWNjZXNzXCIpIHtcbiAgICAgICAgdGhyb3cgbmV3IEVycm9yKFwiQ291bGQgbm90IHBhcnNlIHlhcm4gdjEgbG9jayBmaWxlXCIpXG4gICAgICB9IGVsc2Uge1xuICAgICAgICBhcHBMb2NrRmlsZSA9IHBhcnNlZFlhcm5Mb2NrRmlsZS5vYmplY3RcbiAgICAgIH1cbiAgICB9IGVsc2Uge1xuICAgICAgdHJ5IHtcbiAgICAgICAgYXBwTG9ja0ZpbGUgPSB5YW1sLnBhcnNlKGxvY2tGaWxlU3RyaW5nKVxuICAgICAgfSBjYXRjaCAoZSkge1xuICAgICAgICBjb25zb2xlLmVycm9yKGUpXG4gICAgICAgIHRocm93IG5ldyBFcnJvcihcIkNvdWxkIG5vdCBwYXJzZSB5YXJuIHYyIGxvY2sgZmlsZVwiKVxuICAgICAgfVxuICAgIH1cblxuICAgIGNvbnN0IGluc3RhbGxlZFZlcnNpb24gPSBnZXRQYWNrYWdlVmVyc2lvbihcbiAgICAgIGpvaW4ocmVzb2x2ZShhcHBQYXRoLCBwYWNrYWdlRGV0YWlscy5wYXRoKSwgXCJwYWNrYWdlLmpzb25cIiksXG4gICAgKVxuXG4gICAgY29uc3QgZW50cmllcyA9IE9iamVjdC5lbnRyaWVzKGFwcExvY2tGaWxlKS5maWx0ZXIoXG4gICAgICAoW2ssIHZdKSA9PlxuICAgICAgICBrLnN0YXJ0c1dpdGgocGFja2FnZURldGFpbHMubmFtZSArIFwiQFwiKSAmJlxuICAgICAgICAvLyBAdHMtaWdub3JlXG4gICAgICAgIGNvZXJjZVNlbVZlcih2LnZlcnNpb24pID09PSBjb2VyY2VTZW1WZXIoaW5zdGFsbGVkVmVyc2lvbiksXG4gICAgKVxuXG4gICAgY29uc3QgcmVzb2x1dGlvbnMgPSBlbnRyaWVzLm1hcCgoW18sIHZdKSA9PiB7XG4gICAgICAvLyBAdHMtaWdub3JlXG4gICAgICByZXR1cm4gdi5yZXNvbHZlZFxuICAgIH0pXG5cbiAgICBpZiAocmVzb2x1dGlvbnMubGVuZ3RoID09PSAwKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgIGBcXGAke3BhY2thZ2VEZXRhaWxzLnBhdGhTcGVjaWZpZXJ9XFxgJ3MgaW5zdGFsbGVkIHZlcnNpb24gaXMgJHtpbnN0YWxsZWRWZXJzaW9ufSBidXQgYSBsb2NrZmlsZSBlbnRyeSBmb3IgaXQgY291bGRuJ3QgYmUgZm91bmQuIFlvdXIgbG9ja2ZpbGUgaXMgbGlrZWx5IHRvIGJlIGNvcnJ1cHQgb3IgeW91IGZvcmdvdCB0byByZWluc3RhbGwgeW91ciBwYWNrYWdlcy5gLFxuICAgICAgKVxuICAgIH1cblxuICAgIGlmIChuZXcgU2V0KHJlc29sdXRpb25zKS5zaXplICE9PSAxKSB7XG4gICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgIGBBbWJpZ2lvdXMgbG9ja2ZpbGUgZW50cmllcyBmb3IgJHtwYWNrYWdlRGV0YWlscy5wYXRoU3BlY2lmaWVyfS4gVXNpbmcgdmVyc2lvbiAke2luc3RhbGxlZFZlcnNpb259YCxcbiAgICAgIClcbiAgICAgIHJldHVybiBpbnN0YWxsZWRWZXJzaW9uXG4gICAgfVxuXG4gICAgaWYgKHJlc29sdXRpb25zWzBdKSB7XG4gICAgICByZXR1cm4gcmVzb2x1dGlvbnNbMF1cbiAgICB9XG5cbiAgICBjb25zdCByZXNvbHV0aW9uID0gZW50cmllc1swXVswXS5zbGljZShwYWNrYWdlRGV0YWlscy5uYW1lLmxlbmd0aCArIDEpXG5cbiAgICAvLyByZXNvbHZlIHJlbGF0aXZlIGZpbGUgcGF0aFxuICAgIGlmIChyZXNvbHV0aW9uLnN0YXJ0c1dpdGgoXCJmaWxlOi5cIikpIHtcbiAgICAgIHJldHVybiBgZmlsZToke3Jlc29sdmUoYXBwUGF0aCwgcmVzb2x1dGlvbi5zbGljZShcImZpbGU6XCIubGVuZ3RoKSl9YFxuICAgIH1cblxuICAgIGlmIChyZXNvbHV0aW9uLnN0YXJ0c1dpdGgoXCJucG06XCIpKSB7XG4gICAgICByZXR1cm4gcmVzb2x1dGlvbi5yZXBsYWNlKFwibnBtOlwiLCBcIlwiKVxuICAgIH1cblxuICAgIHJldHVybiByZXNvbHV0aW9uXG4gIH0gZWxzZSB7XG4gICAgY29uc3QgbG9ja2ZpbGUgPSByZXF1aXJlKGpvaW4oXG4gICAgICBhcHBQYXRoLFxuICAgICAgcGFja2FnZU1hbmFnZXIgPT09IFwibnBtLXNocmlua3dyYXBcIlxuICAgICAgICA/IFwibnBtLXNocmlua3dyYXAuanNvblwiXG4gICAgICAgIDogXCJwYWNrYWdlLWxvY2suanNvblwiLFxuICAgICkpXG4gICAgY29uc3QgbG9ja0ZpbGVTdGFjayA9IFtsb2NrZmlsZV1cbiAgICBmb3IgKGNvbnN0IG5hbWUgb2YgcGFja2FnZURldGFpbHMucGFja2FnZU5hbWVzLnNsaWNlKDAsIC0xKSkge1xuICAgICAgY29uc3QgY2hpbGQgPSBsb2NrRmlsZVN0YWNrWzBdLmRlcGVuZGVuY2llc1xuICAgICAgaWYgKGNoaWxkICYmIG5hbWUgaW4gY2hpbGQpIHtcbiAgICAgICAgbG9ja0ZpbGVTdGFjay5wdXNoKGNoaWxkW25hbWVdKVxuICAgICAgfVxuICAgIH1cbiAgICBsb2NrRmlsZVN0YWNrLnJldmVyc2UoKVxuICAgIGNvbnN0IHJlbGV2YW50U3RhY2tFbnRyeSA9IGxvY2tGaWxlU3RhY2suZmluZCgoZW50cnkpID0+IHtcbiAgICAgIGlmIChlbnRyeS5kZXBlbmRlbmNpZXMpIHtcbiAgICAgICAgcmV0dXJuIGVudHJ5LmRlcGVuZGVuY2llcyAmJiBwYWNrYWdlRGV0YWlscy5uYW1lIGluIGVudHJ5LmRlcGVuZGVuY2llc1xuICAgICAgfSBlbHNlIGlmIChlbnRyeS5wYWNrYWdlcykge1xuICAgICAgICByZXR1cm4gZW50cnkucGFja2FnZXMgJiYgcGFja2FnZURldGFpbHMucGF0aCBpbiBlbnRyeS5wYWNrYWdlc1xuICAgICAgfVxuICAgICAgdGhyb3cgbmV3IEVycm9yKFwiQ2Fubm90IGZpbmQgZGVwZW5kZW5jaWVzIG9yIHBhY2thZ2VzIGluIGxvY2tmaWxlXCIpXG4gICAgfSlcbiAgICBjb25zdCBwa2cgPSByZWxldmFudFN0YWNrRW50cnkuZGVwZW5kZW5jaWVzXG4gICAgICA/IHJlbGV2YW50U3RhY2tFbnRyeS5kZXBlbmRlbmNpZXNbcGFja2FnZURldGFpbHMubmFtZV1cbiAgICAgIDogcmVsZXZhbnRTdGFja0VudHJ5LnBhY2thZ2VzW3BhY2thZ2VEZXRhaWxzLnBhdGhdXG4gICAgcmV0dXJuIHBrZy5yZXNvbHZlZCB8fCBwa2cudmVyc2lvbiB8fCBwa2cuZnJvbVxuICB9XG59XG5cbmlmIChyZXF1aXJlLm1haW4gPT09IG1vZHVsZSkge1xuICBjb25zdCBwYWNrYWdlRGV0YWlscyA9IGdldFBhdGNoRGV0YWlsc0Zyb21DbGlTdHJpbmcocHJvY2Vzcy5hcmd2WzJdKVxuICBpZiAoIXBhY2thZ2VEZXRhaWxzKSB7XG4gICAgY29uc29sZS5lcnJvcihgQ2FuJ3QgZmluZCBwYWNrYWdlICR7cHJvY2Vzcy5hcmd2WzJdfWApXG4gICAgcHJvY2Vzcy5leGl0KDEpXG4gIH1cbiAgY29uc29sZS5sb2coXG4gICAgZ2V0UGFja2FnZVJlc29sdXRpb24oe1xuICAgICAgYXBwUGF0aDogcHJvY2Vzcy5jd2QoKSxcbiAgICAgIHBhY2thZ2VEZXRhaWxzLFxuICAgICAgcGFja2FnZU1hbmFnZXI6IGRldGVjdFBhY2thZ2VNYW5hZ2VyKHByb2Nlc3MuY3dkKCksIG51bGwpLFxuICAgIH0pLFxuICApXG59XG4iXX0=