const {test} = require('node:test');
const {Script} = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');

test('complete Dashboard script parses with server-rendered values', () => {
  const template = fs.readFileSync(path.join(__dirname, '../server_4090/templates/index.html'), 'utf8');
  const scripts = [...template.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)];
  for (const [, source] of scripts) {
    new Script(source.replace(/\{\{[\s\S]*?\}\}/g, 'null'));
  }
});
