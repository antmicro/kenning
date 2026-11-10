#Node JS Loader (client-side)

This is a web-service which will serve your JS files for you. It does a few spiffy things:

 * Concatenate a set of JS files on-the-fly so you only download one file
 * Automatically resolve dependencies

This allows you to accomplish the following objectives:

 * Keep JS tidy in as many separate files as you want without peformance concerns
 * Get the benefit of concatenated JS without having to constantly update the concatenated files
 * Get the benefit of dependency resolution, like you would with any other language (if a.js depends on b.js, you only need to ask for a.js)

This app is written in node.js, but is really not *for* node.js. You can use it for any web project. It is for clinet-side javascript, not server-side.

#Installation

npm install js-loader@latest

#Synopsis

##Using as a standalone server

Usage: jsloader ADDRESS PORT JS_SOURCE_DIR [JS_SOURCE_DIR...]

Run the server

    jsloader 127.0.0.1 1234 /path/to/js

Request your files

    <script type="text/javascript" src="http://js.mysite.com/?sources=a.js,b.js,c.js"></script>

And if the first line of c.js looks like this:

    //require d.js

Then d.js will be included in the output.

You can also request minfied output

    <script type="text/javascript" src="http://js.mysite.com/?sources=a.js,b.js,c.js&minify=1"></script>

And you can have multiple source directories

    jsloader 127.0.0.1 1234 /path/to/js1 /path/to/js2

##Advanced integration with Node.js projects

If your project is written in Node.js and you are using connect, then you have two more options. First,
set up the connect middleware.

    app.configure(function() {
        ...
        require('js-loader').JSLoader.connect('/js', ['/path/to/js'])
        ...
    });

The first parameter is the pathname that the jsloader should handle. The second is an array of javascript
source directories. With this in place, you can use your existing connect server to serve your js.

    <script type="text/javascript" src="http://www.mysite.com/js?sources=a.js,b.js,c.js"></script>

The connect middleware will also make the jsloader instance available on all requests so you can request
JavaScript content on-the-fly and embed it directly in the page. To do this, you'll want to generate
the JavaScript in your controller and pass it to your view.

    var minify = true;
    req.jsloader.getContent(['cool.js'], function(err, jsCode) {
        res.render('myview.ejs', {
            locals: { jsCode: jsCode }   
        }); 
    }, minify);

Then, in your view, display the content. Be sure not to escape it.

    <script type="text/javascript"><%- jsCode %></script>

#Tech Details about the bundled server

 * Supports Conditional Gets.
 * Supports GZip compression.
 * Requests are cached. Cache is expired as needed when underlying files are modified.
