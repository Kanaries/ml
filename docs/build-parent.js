#!/usr/bin/env node

const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

const parentDir = path.join(__dirname, '..');

console.log('Building parent project...');
console.log('Parent directory:', parentDir);

try {
  // Change to parent directory
  process.chdir(parentDir);
  
  // Check if package.json exists
  if (!fs.existsSync('package.json')) {
    throw new Error('package.json not found in parent directory');
  }
  
  console.log('Installing dependencies...');
  execSync('yarn install --frozen-lockfile --production=false', { stdio: 'inherit' });
  
  console.log('Building parent project...');
  execSync('yarn build', { stdio: 'inherit' });
  
  console.log('Parent project built successfully!');
} catch (error) {
  console.error('Failed to build parent project:', error.message);
  process.exit(1);
} 