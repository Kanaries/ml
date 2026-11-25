import fs from 'node:fs';

const chromeCandidates = [
    process.env.CHROME_BIN,
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/usr/bin/google-chrome',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
].filter(Boolean);
export const chrome = chromeCandidates.find(candidate => fs.existsSync(candidate));
if (!chrome) throw new Error('Chrome/Chromium not found; set CHROME_BIN to run browser checks');
