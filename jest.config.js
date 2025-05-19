module.exports = {
    preset: 'ts-jest',
    testEnvironment: 'node',
    globals: {
        'ts-jest': {
            diagnostics: false,
            tsconfig: './config/tsconfig.cjs.json',
        },
    },
    testMatch: ['**/__test__/**/*.test.ts'],
    testPathIgnorePatterns: ['/node_modules/', 'test/', 'build/', 'workers/'],
};
