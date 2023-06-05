module.exports = {
  'root': true,
  'env': {
    'browser': true,
    'node': true,
    'es6': true
  },
  'extends': 'eslint:recommended',
  'globals': {
    Float64Array: true
  },
  'rules': {
    'space-before-function-paren': ['error', {
      'anonymous': 'never',
      'named': 'never',
      'asyncArrow': 'always'
    }],
    'indent': [
      'error',
      2,
      {
        SwitchCase: 1,
        VariableDeclarator: 2,
        ArrayExpression: 'first'
      }
    ],
    'linebreak-style': [
      'error',
      'unix'
    ],
    'max-len': [
      'error',
      200
    ],
    'no-console': 'off',
    'no-constant-condition': [
      'error',
      { checkLoops: false }
    ],
    'no-empty': [
      'error',
      { allowEmptyCatch: true }
    ],
    'no-extra-bind': 'error',
    'no-redeclare': 'off',
    'no-unused-vars': 'off',
    'no-warning-comments': 'error',
    'quotes': [
      'error',
      'single',
      { avoidEscape: true }
    ],
    'no-undef': 'off',
    'strict': 'off'
  }
};