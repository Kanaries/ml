const { spawnSync } = require('node:child_process');
const path = require('node:path');

const root = path.resolve(__dirname, '..');
const target = path.join(__dirname, 'gen_all.py');

const candidates = [];
if (process.env.PYTHON) {
    candidates.push([process.env.PYTHON]);
}
candidates.push(['python3']);
candidates.push(['python']);
candidates.push(['py', '-3']);

for (const candidate of candidates) {
    const [cmd, ...baseArgs] = candidate;
    const result = spawnSync(cmd, [...baseArgs, target], {
        cwd: root,
        stdio: 'inherit',
    });

    if (result.error && result.error.code === 'ENOENT') {
        continue;
    }

    if (result.status === 0) {
        process.exit(0);
    }

    process.exit(result.status ?? 1);
}

console.error('No Python interpreter found. Tried PYTHON, python3, python, and py -3.');
process.exit(1);
