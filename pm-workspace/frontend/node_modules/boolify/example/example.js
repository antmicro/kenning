var boolify = require('../')

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
