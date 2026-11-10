const CHANGE_TYPE_MAP = new Map([
	["A", "added"],
	["C", "copied"],
	["D", "deleted"],
	["M", "modified"],
	["R", "renamed"],
	["T", "type changed"],
	["U", "unmerged"],
	["B", "pairing broken"],
	[" ", "unmodified"],
	["?", "untracked"],
	["!", "ignored"],
]);
export function mapChangeType(pChar) {
	return CHANGE_TYPE_MAP.get(pChar) ?? "unknown";
}
