var File = require('./file').File;

var Request = function(filePaths, minify) {
	this.files = [];
	this.addFiles(filePaths);
	this.minify = Boolean(minify);
	this.fileDataList = [];
	this.fileDataMap = {};
};

Request.prototype.addFilesIfUnique = function(filePaths) {
	for (var i = 0; i < filePaths.length; i++) {
		if (!this.containsFile(filePaths[i])) {
			this.addFile(filePaths[i]);
		}
	}
};

Request.prototype.addFiles = function(filePaths) {
    filePaths = filePaths || [];
	for (var i = 0; i < filePaths.length; i++) {
		this.addFile(filePaths[i]);
	}
};

Request.prototype.addFile = function(filePath) {
	this.files.push(new File(filePath));
};

Request.prototype.containsFile = function(filePath) {
	for (var i = 0; i < this.files.length; i++) {
		if (this.files[i].path === filePath) {
			return true;
		}
	}
	return false;
};

Request.prototype.sortFilesByDependencies = function() {
	var files = this.files;
	this.files = [];
	while (files.length > 0) {
        this.files.push(removeNextFileData(files));
    }
    
    function removeNextFileData(files) {
		var i, file;
		for (i = 0; i < files.length; i++) {
		    if (files[i].dependencies.length == 0) {
		        file = files.splice(i, 1)[0];
		        removeFromAllDependencies(files, file);
		        return file;
		    }
		}
		throw new Error('Unable to resolve dependencies. You probably have a ' +
			'circular dependency or a dependency on a file that is unavailable.');
	}
	
	function removeFromAllDependencies(files, file) {
		var i, c, dependencies;
		for (i = 0; i < files.length; i++) {
		    dependencies = files[i].dependencies;
		    for (c = 0; c < dependencies.length; c++) {
		        if (dependencies[c] == file.path) {
		            dependencies.splice(c, 1);
		        }
		    }
		}
	}
};

module.exports.Request = Request;
