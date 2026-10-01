const INDENT = 2;
export default function formatAsJSON(pChanges) {
	return JSON.stringify(pChanges, null, INDENT);
}
