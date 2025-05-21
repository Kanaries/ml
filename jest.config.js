/** @type {import('ts-jest').JestConfigWithTsJest} */
module.exports = {
    preset: 'ts-jest',
    transform: {
        '^.+\\.tsx?$': ['ts-jest', {
            diagnostics: false,
            tsconfig: './config/tsconfig.cjs.json',
        }]
    },
    testPathIgnorePatterns: ['/node_modules/', 'test/', 'build/', 'workers/'],
    testEnvironment: 'node',
    setupFiles: ['<rootDir>/jest.setup.js']
};
