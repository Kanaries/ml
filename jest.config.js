module.exports = {
    preset: 'ts-jest',
    globals: {
        'ts-jest': {
            diagnostics: false,
            tsConfig: './config/tsconfig.cjs.json',
        },
    },
    browser: false,
    testPathIgnorePatterns: ['/node_modules/', 'test/', 'build/', 'workers/'],
};
