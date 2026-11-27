import { mapChangeType } from "./map-change-type.js";
const DIFF_SHORT_STATUS_LINE_PATTERN =
	/^(?<stagedType>[ ACDMRTUXB?!])(?<unStagedType>[ ACDMRTUXB?!])[ \t]+(?<name>[^ \t]+)(( -> )(?<newName>[^ \t]+))?$/;
export function parseStatusLines(pString) {
	return pString
		.split(/\r?\n/)
		.filter(Boolean)
		.map(parseStatusLine)
		.filter(({ name, type }) => Boolean(name) && Boolean(type));
}
export function parseStatusLine(pString) {
	const lMatchResult = DIFF_SHORT_STATUS_LINE_PATTERN.exec(pString);
	const lReturnValue = {};
	if (lMatchResult?.groups) {
		const lStagedType = mapChangeType(lMatchResult.groups.stagedType);
		const lUnStagedType = mapChangeType(lMatchResult.groups.unStagedType);
		lReturnValue.type =
			lStagedType === "unmodified" ? lUnStagedType : lStagedType;
		if (lMatchResult.groups.newName) {
			lReturnValue.name = lMatchResult.groups.newName;
			lReturnValue.oldName = lMatchResult.groups.name;
		} else {
			lReturnValue.name = lMatchResult.groups.name;
		}
	}
	return lReturnValue;
}
