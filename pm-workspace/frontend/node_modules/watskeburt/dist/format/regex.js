import { extname } from "node:path";
const DEFAULT_CHANGE_TYPES = new Set([
	"modified",
	"added",
	"renamed",
	"copied",
	"untracked",
]);
export default function formatAsRegex(
	pChanges,
	pExtensions,
	pChangeTypes = DEFAULT_CHANGE_TYPES,
) {
	const lChanges = pChanges
		.filter(
			(pChange) =>
				pChangeTypes.has(pChange.type) &&
				(pExtensions.has(".*") || pExtensions.has(extname(pChange.name))),
		)
		.map(({ name }) => name.replace(/\\/g, "\\\\").replace(/\./g, "[.]"))
		.join("|");
	return `^(${lChanges})$`;
}
