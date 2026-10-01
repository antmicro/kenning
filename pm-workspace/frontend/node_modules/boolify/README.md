# boolify

Convert true/false strings to booleans

[![Build Status](https://travis-ci.org/timhudson/boolify.png?branch=master)](https://travis-ci.org/timhudson/boolify)

## Example

``` js
var boolify = require('boolify')

boolify('true')
// true

boolify('false')
// false

boolify('A string')
// "A string"

var obj = boolify({
  t: 'true',
  f: 'false',
  s: 'Another string',
  n: 5
})
// {
//   t: true,
//   f: false,
//   s: 'Another string',
//   n: 5
// }

boolify('TRUE')
// true

boolify('tRuE')
// true
```

## Usage

`boolify` is case-insensitive and will convert any string of `"true"` or `"false"` in to the
appropriate boolean.

This is helpful for handling checkboxes in express' `req.body`

``` js
app.put('/users/:username', function(req, res) {
  console.log(req.body.someCheckbox) // 'true'
  req.body = boolify(req.body)
  console.log(req.body.someCheckbox) // true
})
```

### `boolify(object | string)`

If provided an object, `boolify` will recursively convert all values.

``` js
var result = boolify({one: {fish: {two: {fish: 'true'}}}})
console.log(result.one.fish.two.fish)
// true
```

## Install

With [npm](https://npmjs.org) do:

```
npm install boolify
```

## See also

 * [yn](https://github.com/sindresorhus/yn) supports additional values for booleans, both uppercase and lowercase, and returns null when a value is recognized is neither `true` or `false`

## License

MIT
