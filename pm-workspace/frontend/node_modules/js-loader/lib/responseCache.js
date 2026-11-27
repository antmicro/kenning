var fs = require('fs');

var ResponseCache = function() {
	this.cache = {};
    this.cacheKeysByFilePath = {};
};

ResponseCache.prototype.buildCacheKey = function(request) {
	var i, filePaths = [];
	for (i = 0; i < request.files.length; i++) {
		filePaths.push(request.files[i].path);
	}
    return filePaths.sort().join(',') + (request.minify ? '|min' : '');
};

ResponseCache.prototype.get = function(request) {
    var cacheKey = this.buildCacheKey(request);
    return this.cache[cacheKey];
};

ResponseCache.prototype.put = function(request, response) {
    var self = this,
        cacheKey = this.buildCacheKey(request);
    this.cache[cacheKey] = response;
    this.watchFiles(request.files, cacheKey);
};

ResponseCache.prototype.watchFiles = function(files, cacheKey) {
    var self = this;
    for (i = 0; i < files.length; i++) {
        if (!this.cacheKeysByFilePath[files[i].resolvedPath]) {
            this.cacheKeysByFilePath[files[i].resolvedPath] = [];
            (function(filePath) {
                fs.watchFile(filePath, function(curr, prev) {
                    self.expireByFilePath(filePath);
                });
            })(files[i].resolvedPath);
        }
        this.cacheKeysByFilePath[files[i].resolvedPath].push(cacheKey);
    }
};

ResponseCache.prototype.expireByFilePath = function(filePath) {
    var cacheKey = this.cacheKeysByFilePath[filePath];
    delete this.cache[cacheKey];
};

module.exports.ResponseCache = ResponseCache;
