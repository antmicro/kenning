var util = require('util');

var index = require('../index');

/**
 * Pad input string from left with paddingValue up to paddingValue.length.
 * @see http://stackoverflow.com/a/14760377/737477
 *
 * @param {String} str
 * @param {String} paddingValue
 * @return {String}
 */
var padLeft = function (str, paddingValue) {
	return String(paddingValue + str).slice(-paddingValue.length);
};

/**
 * Constructs a message formatted for consumption by TeamCity.
 *
 * @param {String} type
 * @param {Object} args
 */
var Message = function (type, args) {
	this.type = type;
	this.single = false;

	// Message is a 'multiple attribute message'.
	if (typeof args === 'object' || typeof args === 'undefined') {
		this.args = args || {};
	}
	// Message is a 'single attribute message'.
	else {
		this.single = true;
		this.args = args;
	}

	if (!this.single) {
		if (this.args.flowId) {
			this.arg('flowId', this.args.flowId);
		}
		else if (index.autoFlowId) {
			this.arg('flowId', Message.flowId);
		}
	}
};

Message.flowId = Math.floor(Math.random() * (1e10 - 1e6 + 1)) + 1e6;

/**
 * Add a keyword argument to the message.
 *
 * @param  {String} key
 * @param  {String} value
 * @return {Message}
 */
Message.prototype.arg = function (key, value) {
	if (this.single) {
		throw new Error('Cannot add arguments to a single attribute message.');
	}

	this.args[key] = value;
	return this;
};

/**
 * Escape string for TeamCity output.
 * @see https://confluence.jetbrains.com/display/TCD65/Build+Script+Interaction+with+TeamCity#BuildScriptInteractionwithTeamCity-servMsgsServiceMessages
 *
 * @param  {String} str
 * @return {String}
 */
Message.prototype.escape = function (str) {
	if (str == null) {
		return '';
	}

	return str
	    .toString()
	    .replace(/\|/g, '||')
	    .replace(/\n/g, '|n')
	    .replace(/\r/g, '|r')
	    .replace(/\[/g, '|[')
	    .replace(/\]/g, '|]')
	    .replace(/\u0085/g, '|x') // next line
	    .replace(/\u2028/g, '|l') // line separator
	    .replace(/\u2029/g, '|p') // paragraph separator
	    .replace(/'/g, '|\'');
};

/**
 * Format keyword arguments for use in a message.
 *
 * @return {String}
 */
Message.prototype.formatArgs = function () {
	return Object.keys(this.args).map(function (key) {
		return util.format('%s=\'%s\'', key, this.escape(this.args[key]));
	}, this).join(' ');
};

/**
 * Format the message as a string.
 *
 * @return {String}
 */
Message.prototype.toString = function () {
	if (this.single) {
		return util.format('##teamcity[%s \'%s\']', this.type, this.escape(this.args));
	}

	if (!this.args.timestamp) {
		// TeamCity not fully support ISO 8601 (see TW-36173) so we need to cut off 'Z' at the end.
		this.arg('timestamp', new Date().toISOString().slice(0, -1));
	}

	return util.format('##teamcity[%s %s]', this.type, this.formatArgs());
};

module.exports = Message;
