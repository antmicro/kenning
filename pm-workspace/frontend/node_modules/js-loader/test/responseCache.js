var vows = require('vows'),
    assert = require('assert'),
    fs = require('fs'),
    path = require('path'),
    temp = require('temp'),
    ResponseCache = require('../lib/responseCache').ResponseCache,
    Response = require('../lib/response').Response,
    Request = require('../lib/request').Request,
    File = require('../lib/file').File;

vows.describe('ResponseCache').addBatch({
    'when we create a new ResponseCache': {
        topic: function() { return new ResponseCache() },

        'we get an instance of ResponseCache': function(cache) {
            assert.ok(cache instanceof ResponseCache);
        }
    },
    
    'when we put a reponse': {
    	topic: function() {
            var self = this;
            temp.mkdir('jsloader-test', function(err, dirPath) {
                var filePath = path.join(dirPath, "a.js");
                fs.writeFile(filePath, "abcd", function(err) {
                    var cache = new ResponseCache(),
                        request = new Request(),
                        response = new Response();
                        file = new File('a.js');
                    file.resolvedPath = filePath;
                    request.files = [file];
                    response.content = 'abcd';
                    cache.put(request, response);
                    self.callback(null, cache.get(request));
                });
            });
    	},
    	
    	'we can get it back out with get': function(response) {
    		assert.equal(response.content, 'abcd');
    	}
    },

    'when we put a reponse & change the file': {
    	topic: function() {
            var self = this;
            temp.mkdir('jsloader-test', function(err, dirPath) {
                var filePath = path.join(dirPath, "a.js");
                fs.writeFile(filePath, "abcd", function(err) {
                    var cache = new ResponseCache(),
                        request = new Request(),
                        response = new Response();
                        file = new File('a.js');
                    file.resolvedPath = filePath;
                    request.files = [file];
                    response.content = 'abcd';
                    cache.put(request, response);
                    fs.watchFile(filePath, function(oldStats, newStats) {
                        self.callback(null, cache.get(request));
                    });
                    fs.writeFile(filePath, "abcde", function(err) {});
                });
            });
    	},
    	
    	'the cache is expired': function(response) {
    		assert.isUndefined(response);
    	}
    }
}).export(module);
