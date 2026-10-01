jqjs is a JavaScript implementation of the [jq] query language. It
implements the core language features in pure JavaScript.

The main entry point to jqjs is the compile function, which turns a jq
program string into a generator function:

    import jq from './jq.js'
    let filter = jq.compile(".x[].y")
    for (let v of filter({x:[{y:2}, {y:4}]}) { ... }

The module also has a prettyPrint function for rendering an object to
text.

As a shorthand, the default export is itself callable in two ways:

    let filter = jq('.x[].y')
    for (let x of filter(obj)) { ... }

    for (let x of jq('.x[].y', obj)) { ... }

With a single argument, this is equivalent to jq.compile,
and with two arguments it is equivalent to jq.compile(arg1)(arg2).

Features
--------

jqjs supports *most* of the core jq language features, but lacks
some functions and some of the advanced functionality. It also uses
JavaScript strings as backing, so does not have jq proper's Unicode
support.

- [x] Identity: `.`
- [x] Object Identifier-Index: `.foo`, `.foo.bar`
- [x] Generic Object Index: `.[<string>]`
- [x] Array Index: `.[2]`
- [x] Array/String Slice: `.[10:15]`
- [x] Array/Object Value Iterator: `.[]`
- [x] Comma: `,`
- [x] Pipe: `|`
- [x] Parentheses: `(...)`
- [x] Array Construction: `[...]`
- [x] Object Construction: `{ ... }` including shorthand `{ user, title }`
      and { $var }, and computed keys `{ (.x): true }`
- [x] Recursive Descent: `..`
- [x] Addition, subtraction, multiplication, division, and modulo,
  including the overloaded type operations.
- [x] Named functions
    - Built-in functions
        - [x] tostring, tonumber, toboolean, tojson, fromjson
        - [x] length, keys, has, in, type, del
        - [x] empty, select, arrays, objects, booleans, numbers, strings, nulls
        - [x] values, iterables, scalars, normals, finites
        - [x] map, map_values, sort, sort_by, group_by, unique, unique_by
        - [x] explode, implode, split, join, add, add/1, abs
        - [x] to_entries, from_entries, with_entries, walk/1
        - [x] range/1, range/2, range/3, min, max, min_by/2, max_by/2
        - [x] any/0, any/1, any/2, all/0, all/1, all/2, not, isempty
        - [x] contains, inside, indices/1, index/1, rindex/1
        - [x] path, getpath/1, setpath/2, delpaths/1, pick/1, paths, paths/1
        - [x] trim/0, ltrim/0, rtrim/0, trimstr/1, ltrimstr/1, rtrimstr/1
        - [x] startswith/1, endswith/1, scan/1, scan/2
        - [x] first, last, nth/1, first/1, last/1, nth/2, limit/2, skip/2
        - [x] flatten, flatten/1, transpose, combinations, combinations/1
        - [x] sub/1, sub/2, gsub/1, gsub/2, test/1, test/2, split/2
        - [x] capture/1, capture/2, match/1, match/2, splits/1, splits/2
        - [x] ascii_upcase/1, ascii_downcase/1
        - [x] recurse/0, recurse/1, recurse/2, while/2, until/2, builtins
        - [x] isfinite, isinfinite, isnormal, isnan, infinite, nan
        - [ ] repeat/1, debug/1, error/1, the others...
    - [x] User-defined functions
      - [x] def foo: . + 1; def foo(f;g): f|g; def addvalue($f): map(. + $f);
    - [x] Mathematical functions (found in JavaScript Math object)
      - [x] fabs, acos, acosh, asin, asinh, atan, atanh, cbrt, ceil, cos
      - [x] cosh, exp, expm1, floor, log, log10, log1p, log2, round, sin
      - [x] sinh, sqrt, tan, tanh, trunc
      - [x] atan2/2, fmax/2, fmin/2, hypot/2, pow/2
      - [ ] erf, erfc, exp10, exp2, fabs, gamma, j0, j1, lgamma, logb
      - [ ] nearbyint, rint, significand, tgamma, y0, y1
      - [ ] copysign/2, drem/2, fdim/2, fmod/2, frexp/2, jn/2, ldexp/2, modf/2
      - [ ] nextafter/2, nexttoward/2, remainder/2, scalb/2, scalbln/2, yn/2
      - [ ] fma/3
    - [ ] Date functions
      - [x] now, todate, fromdate, todateiso8601, fromdateiso8601
    - [x] SQL-style operators
      - [x] INDEX/1, INDEX/2, JOIN/4, JOIN/3, JOIN/2, IN/1, IN/2
- [x] String interpolation: `\(foo)`
- [x] Format strings and escaping: `@text`, `@json`, `@html`, `@uri`, `@csv`,
  `@tsv`, `@sh`, `@base64`, `@base64d`
    - [x] Format interpolations: `@uri "https://google.com/search?q=\(.x)"`
- [x] Equality checks: `==`, `!=`
- [x] Comparisons: `<`, `>`, `<=`, `>=`
    - [x] Correct sorting order for unequal types (`null < 7`, `[] < {}`)
- [x] Conditionals: `if A then B elif C then D else E end`
- [x] Boolean operators `and`, `or`
- [x] Alternative operator: `//`
- [ ] Try-catch: `try EXP catch EXP`
  - [x] Error Suppression operator `?`
- [x] Regular expressions (uses JavaScript RegExp, so behaviour is incomplete)
- [x] Variable/Symbolic Binding Operator `... as $identifier | ...`
- [x] Reduce: `reduce .[] as $item (0; . + $item)`
- [x] foreach: `foreach .[] as $item (...;...;...)`
- [ ] Destructuring: `... as { x: $x, y: [$fst, $snd] }`
- [ ] I/O (unlikely to make sense here)
- [ ] Streams and stream functions
- [x] Update-assignment: `.posts[].comments |= . + ["Another"]`
- [x] Arithmetic update-assignment: `+=`, `-=`, `*=`, `/=`, `%=`, `//=`
- [x] Plain assignment: `(.a,.b) = range(2)`
- [ ] Modules with `import` and `include`

Installing and using
--------------------

The jq.js module can be imported and used directly:

    import jq from "./jqjs.js";

but this library can also be installed through npm:

    npm install @michaelhomer/jqjs

then

    import jq from "@michaelhomer/jqjs";
    // or
    const jq = require("@michaelhomer/jqjs");

Or

    npm install mwh/jqjs

then

    import jq from "jqjs/jq.js";
    // or
    const jq = require("jqjs/jq.js");

After that

    let func = jq.compile(".x[].y")

will create a `func` function that can be given any JavaScript
object to process, and will return an iterator producing each output
of the jq program:

    for (let v of filter({x:[{y:2}, {y:4}]}) { ... }

will run the loop body with `v` holding 2, then 4, then stop.

`func` also exposes the jq syntax tree (as `func.filter`) and
a function returning a complete trace of the output of every component
of the jq program (`func.trace({x:[{y:2}, {y:4}]})`) as nested arrays
and objects.

The default export can also be used as a tag function:

    jq`.x[].y`

With no interpolations, this is almost the same as jq.compile, but uses
raw strings so no escape sequences are necessary. Interpolated
expressions are used *as JSON values*, so:

    const obj = {x:1, y: [2, 3]}
    let func = jq`. + ${obj} | paths`

is equivalent to

    let func = jq.compile('. + {"x": 1, "y": [2, 3]} | paths')

As a result, they cannot be used to splice in e.g. a function name,
other syntactic element, or the middle of a string, only a whole value.

Performance
-----------

Not great.

The intention is to be semantically correct first and to have clear code
second. Performance improvements sit after that, if at all. Executing a
program may reëvaluate parts of it or traverse the object multiple times
where that makes things simpler, and internally evaluation happens by
tree-walking the input syntax.

Demonstration
-------------

[demo.html] is a [live demo](https://mwh.github.io/jqjs/demo.html) of how
to use jqjs that lets you enter a jq program and an input JSON value and
see the output JSON values it produces.

A [demonstration of the tracing functionality][trace-demo] from the paper
"[Branching Compositional Data Transformations in jq, Visually][paper]"
is also available.

[jq]: https://jqlang.org/
[demo.html]: demo.html
[trace-demo]: https://homepages.ecs.vuw.ac.nz/~mwh/demos/paint2023/
[paper]: https://doi.org/10.1145/3623504.3623567
