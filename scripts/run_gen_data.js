const { spawnSync } = require('node:child_process');
const path = require('node:path');

const root = path.resolve(__dirname, '..');
const target = path.join(__dirname, 'gen_all.py');

const candidates = [];
if (process.env.PYTHON) {
    candidates.push([process.env.PYTHON]);
}
candidates.push(['python3']);
candidates.push(['/usr/local/bin/python3']);
candidates.push(['/Library/Frameworks/Python.framework/Versions/3.10/bin/python3']);
candidates.push(['python']);
candidates.push(['py', '-3']);

function supportsSklearn(candidate) {
    const [cmd, ...baseArgs] = candidate;
    const result = spawnSync(cmd, [...baseArgs, '-c', 'import sklearn, numpy'], {
        cwd: root,
        stdio: 'ignore',
    });

    if (result.error && result.error.code === 'ENOENT') {
        return false;
    }
    return result.status === 0;
}

for (const candidate of candidates) {
    const [cmd, ...baseArgs] = candidate;
    if (!supportsSklearn(candidate)) {
        continue;
    }
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

console.error('No Python interpreter with sklearn found. Tried PYTHON, python3, /usr/local/bin/python3, /Library/Frameworks/Python.framework/Versions/3.10/bin/python3, python, and py -3.');
process.exit(1);
