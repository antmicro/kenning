/*! @license Acorn JSX Walk v2.0.0

	Copyright 2019 Contributors.

	Permission is hereby granted, free of charge, to any person obtaining a copy
	of this software and associated documentation files (the "Software"), to deal
	in the Software without restriction, including without limitation the rights
	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
	copies of the Software, and to permit persons to whom the Software is
	furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in all
	copies or substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
	SOFTWARE.

*/
/**
 * @module acornJsxWalk
 */

/**
 * Type check for functions.
 *
 * @param  {any} value
 * @return {boolean}
 */
function isFunc(value) {
  return (
    typeof value === 'function' ||
    !!(value && value.constructor && value.call && value.apply)
  )
}

/**
 * Check base object for required visitor functions.
 *
 * @param  {Object} base - Visitor base object.
 * @return {boolean}
 */
function isValidBase(base) {
  if ( base === void 0 ) base = {};

  return (
    isFunc(base.ExpressionStatement) &&
    isFunc(base.SpreadElement) &&
    isFunc(base.Identifier)
  )
}

/**
 * Attempt to extend the base object with
 * JSX visitor function definitions.
 *
 * @param  {Object} base - Visitor base object.
 * @return {void}
 */
function extend(base) {
  if ( base === void 0 ) base = {};

  if (!isValidBase(base)) {
    throw Error('Invalid visitor base object.')
  }

  // prettier-ignore
  {
    base.JSXExpressionContainer = base.ExpressionStatement;
    base.JSXSpreadChild         = base.ExpressionStatement;
    base.JSXClosingFragment     = base.Identifier;
    base.JSXEmptyExpression     = base.Identifier;
    base.JSXIdentifier          = base.Identifier;
    base.JSXOpeningFragment     = base.Identifier;
    base.JSXText                = base.Identifier;
    base.JSXSpreadAttribute     = base.SpreadElement;
  }

  base.JSXAttribute = function (node, state, callback) {
    callback(node.name, state);
    if (node.value) { callback(node.value, state); }
  };

  base.JSXMemberExpression = function (node, state, callback) {
    callback(node.object, state);
    callback(node.property, state);
  };

  base.JSXNamespacedName = function (node, state, callback) {
    callback(node.namespace, state);
    callback(node.name, state);
  };

  base.JSXOpeningElement = function (node, state, callback) {
    callback(node.name, state);
    for (var i = 0; i < node.attributes.length; ++i) {
      callback(node.attributes[i], state);
    }
  };

  base.JSXClosingElement = function (node, state, callback) {
    callback(node.name, state);
  };

  base.JSXElement = function (node, state, callback) {
    callback(node.openingElement, state);
    for (var i = 0; i < node.children.length; ++i) {
      callback(node.children[i], state);
    }
    if (node.closingElement) { callback(node.closingElement, state); }
  };

  base.JSXFragment = function (node, state, callback) {
    callback(node.openingFragment, state);
    for (var i = 0; i < node.children.length; ++i) {
      callback(node.children[i], state);
    }
    callback(node.closingFragment, state);
  };
}

export { extend };
