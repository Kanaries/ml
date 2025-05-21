// This file configures the test environment for Jest
process.env.NODE_ENV = 'test';

// Fix for __dirname in ESM
if (typeof global.__dirname === 'undefined') {
    const { dirname } = require('path');
    const { fileURLToPath } = require('url');
    
    global.__dirname = dirname;
    global.__filename = fileURLToPath;
} 