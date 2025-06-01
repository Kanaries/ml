export function asyncMode<P extends any[], R>(fn: (...args: P) => R) {
    return (...args: P): Promise<R> => {
        if (typeof Worker !== 'undefined' && typeof window !== 'undefined') {
            return runBrowser(fn, args);
        }
        if (typeof process !== 'undefined' && process.versions && process.versions.node) {
            return runNode(fn, args);
        }
        return Promise.resolve(fn(...args));
    };
}

function runBrowser(fn: Function, args: any[]): Promise<any> {
    return new Promise((resolve, reject) => {
        const workerCode = `self.onmessage = function(e) {\n  const { fnStr, args } = e.data;\n  const f = eval('(' + fnStr + ')');\n  Promise.resolve(f.apply(null, args)).then(r => {\n    self.postMessage({ result: r });\n  }).catch(err => {\n    self.postMessage({ error: err.message });\n  });\n}`;
        const blob = new Blob([workerCode], { type: 'application/javascript' });
        const worker = new Worker(URL.createObjectURL(blob));
        worker.onmessage = (e) => {
            if (e.data && 'error' in e.data) {
                reject(new Error(e.data.error));
            } else {
                resolve(e.data.result);
            }
            worker.terminate();
        };
        worker.onerror = (err) => {
            reject(err);
            worker.terminate();
        };
        worker.postMessage({ fnStr: fn.toString(), args });
    });
}

function runNode(fn: Function, args: any[]): Promise<any> {
    return new Promise((resolve, reject) => {
        const { Worker } = require('worker_threads');
        const worker = new Worker(
            `const { parentPort } = require('worker_threads');\nparentPort.on('message', (data) => {\n  const fn = eval('(' + data.fnStr + ')');\n  Promise.resolve(fn.apply(null, data.args)).then(r => {\n    parentPort.postMessage({ result: r });\n  }).catch(err => {\n    parentPort.postMessage({ error: err.message });\n  });\n});`,
            { eval: true }
        );
        worker.on('message', (msg: any) => {
            if (msg && 'error' in msg) {
                reject(new Error(msg.error));
            } else {
                resolve(msg.result);
            }
            worker.terminate();
        });
        worker.on('error', (err: any) => {
            reject(err);
            worker.terminate();
        });
        worker.postMessage({ fnStr: fn.toString(), args });
    });
}
