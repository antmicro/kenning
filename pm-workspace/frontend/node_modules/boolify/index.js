module.exports = function boolify (obj) {
  if (typeof obj === 'string') return convert(obj)

  if (toString.call(obj) === '[object Object]') {
    var keys = Object.keys(obj)

    for (var i = 0, l = keys.length; i < l; i++) {
      obj[keys[i]] = boolify(obj[keys[i]])
    }
  }

  return obj
}

function convert (value) {
  var v = value.toLowerCase()
  return v === 'true' ? true :
         v === 'false' ? false : value
}
