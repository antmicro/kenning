import formatAsRegex from "./regex.js";
import formatAsJSON from "./json.js";
const OUTPUT_TYPE_TO_FUNCTION = new Map([
	["regex", formatAsRegex],
	["json", formatAsJSON],
]);
export function format(pChanges, pOutputType, pExtensions) {
	const lExtensions = new Set(
		(pExtensions ?? "*")
			.split(",")
			.map((pExtension) => `.${pExtension.trim()}`),
	);
	return OUTPUT_TYPE_TO_FUNCTION.get(pOutputType)(pChanges, lExtensions);
}
