var fs = require('fs');

var File = function(path) {
	this.path = path;
	this.resolvedPath = null;
	this.content = null;
	this.dependencies = [];
};

File.prototype.loadContent = function(callback) {
	var self = this;
	if (this.resolvedPath == null) {
		throw Error("Must set resolvedPath before loading content");
	}
    fs.readFile(this.resolvedPath, 'utf8', function(err, data) {
    	if (err) return callback(err);
        self.content = data;
        self.parseDependencies();
        callback(null);
    });
};

File.prototype.parseDependencies = function() {
    var foundNonRequireLine = false,
        i, line, lines;
    if (this.content == null) {
    	throw Error("Must load content before parsing dependencies");
    }
    //TODO: don't assume unix style line separator
    lines = this.content.split("\n");
    this.content = null;
    while (lines.length > 0 && this.isRequireLine(lines[0])) {
    	this.dependencies.push(this.extractDependency(lines.shift()));
    }
    //TODO: don't assume unix style line separator
    this.content = lines.join("\n");
};

File.prototype.isRequireLine = function(line) {
    return /^\/\/require/.test(line);
};

File.prototype.extractDependency = function(line) {
    return line.match(/^\/\/require (.*)$/)[1];
};

module.exports.File = File;
