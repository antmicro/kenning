var vows = require('vows'),
    assert = require('assert'),
    Request = require('../lib/request').Request;

vows.describe('Request').addBatch({
    'when we create a new Request': {
        topic: function() { return new Request(['chat.js']) },

        'we get an instance of Request': function(topic) {
            assert.ok(topic instanceof Request);
        },
        
        'minify defaults to false': function(topic) {
            assert.isFalse(topic.minify);
        },
        
        'files are set': function(topic) {
        	assert.equal(topic.files.length, 1);
        	assert.instanceOf(topic.files[0], require('../lib/file').File);
        	assert.equal(topic.files[0].path, 'chat.js');
        },
        
        'defaults are set': function(topic) {
        	assert.deepEqual(topic.fileDataList, []);
        	assert.deepEqual(topic.fileDataMap, {});
        }
    },
    
    'when we create a new Request with minify set to true': {
        topic: function() { return new Request(['chat.js'], true) },

        'minify is true': function(topic) {
            assert.isTrue(topic.minify);
        }
    },
    
    'when we try to add files using addFilesIfUnique': {
        topic: function() {
        	var request = new Request(['chat.js'], true);
        	request.addFilesIfUnique(['chat.js', 'other.js']);
        	return request;
        },

        'only the uniuqe ones are added': function(request) {
			assert.equal(request.files.length, 2);
			assert.equal(request.files[0].path, 'chat.js');
			assert.equal(request.files[1].path, 'other.js');
        }
    },
    
    'when we sort by dependencies': {
    	topic: function() {
    		var request = new Request(['a.js', 'b.js']);
    		request.files[0].dependencies = ['b.js'];
    		request.sortFilesByDependencies();
    		return request;
    	},
    	
    	'there file count is correct': function(request) {
    		assert.equal(request.files.length, 2);
    	},
    	
    	'the dependant comes last': function(request) {
    		assert.equal(request.files[0].path, 'b.js');
    	}
    }
}).export(module);
