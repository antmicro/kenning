var JSLoader = require('./jsloader').JSLoader,
    url = require('url');

var handleRequest = function(req, res, jsloader) {
    var files, content, query;
    query = url.parse(req.url, true).query;

    if (typeof(query) === 'undefined' || typeof(query.sources) === 'undefined' || /\.\./.test(query.sources)) {
        res.writeHead(500, {'Content-Type': 'text/plain'});
        res.end("ERROR: Missing or invalid sources parameter.\n");
        return;
    }

    files = query.sources.split(',');
    minify = false;
    if (query.minify) minify = true;
    content = jsloader.getContent(files, function(err, content) {
        if (err) {
            res.writeHead(500, {'Content-Type': 'text/plain'});
            res.end("ERROR: " + err + "\n");
        } else {
            res.writeHead(200, {'Content-Type': 'application/x-javascript'});
            res.end(content);
        }
    }, minify);
};

var connectMiddleware = function(urlPath, srcDirs) {
    var jsloader = new JSLoader(srcDirs);
    return function(req, res, next) {
        var pathname = url.parse(req.url).pathname;
        if (pathname === urlPath) {
            handleRequest(req, res, jsloader);
        } else {
            req.jsloader = jsloader;
            next();
        }
    };
};

module.exports = connectMiddleware;
