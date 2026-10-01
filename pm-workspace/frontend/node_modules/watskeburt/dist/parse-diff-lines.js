import { mapChangeType } from "./map-change-type.js";
const DIFF_NAME_STATUS_LINE_PATTERN =
	/^(?<type>[ACDMRTUXB])(?<similarity>\d{3})?[ \t]+(?<name>[^ \t]+)[ \t]*(?<newName>[^ \t]+)?$/;
export function parseDiffLines(pString) {
	return pString
		.split(/\r?\n/)
		.filter(Boolean)
		.map(parseDiffLine)
		.filter(({ name, type }) => Boolean(name) && Boolean(type));
}
export function parseDiffLine(pString) {
	const lMatchResult = DIFF_NAME_STATUS_LINE_PATTERN.exec(pString);
	const lReturnValue = {};
	if (lMatchResult?.groups) {
		lReturnValue.type = mapChangeType(lMatchResult.groups.type);
		if (lMatchResult.groups.newName) {
			lReturnValue.name = lMatchResult.groups.newName;
			lReturnValue.oldName = lMatchResult.groups.name;
		} else {
			lReturnValue.name = lMatchResult.groups.name;
		}
	}
	return lReturnValue;
}
