import { SVC } from './svc';
import { Params, registerEstimator } from '../base/estimator';
import { KernelMatrix, KernelType, SMOSolution, solveNuSVC } from './smo';

export interface NuSVCProps {
    /** upper bound on the fraction of margin errors, lower bound on the fraction of support vectors */
    nu?: number;
    kernel?: KernelType;
    gamma?: number | 'scale' | 'auto';
    degree?: number;
    coef0?: number;
    tol?: number;
    /** hard limit on SMO pair updates; -1 (default) = run until convergence */
    maxIter?: number;
}

/**
 * nu-Support Vector Classification (libsvm Solver_NU semantics): `nu`
 * replaces `C` and directly trades off support-vector fraction against
 * margin errors. Multiclass problems are one-vs-one; each pairwise
 * subproblem must satisfy nu <= 2 * min(n+, n-) / (n+ + n-).
 */
export class NuSVC extends SVC {
    private nu: number;

    constructor(props: NuSVCProps = {}) {
        const { nu = 0.5, ...rest } = props;
        super(rest);
        if (!Number.isFinite(nu) || nu <= 0 || nu > 1) {
            throw new Error('nu <= 0 or nu > 1');
        }
        this.nu = nu;
    }

    public getParams(): Params {
        return {
            nu: this.nu,
            kernel: this.kernel,
            gamma: this.gamma,
            degree: this.degree,
            coef0: this.coef0,
            tol: this.tol,
            maxIter: this.maxIter,
        };
    }

    protected solveBinary(X: number[][], y: number[]): SMOSolution {
        let nPos = 0;
        for (const v of y) {
            if (v === 1) {
                nPos++;
            }
        }
        const nNeg = y.length - nPos;
        if ((this.nu * y.length) / 2 > Math.min(nPos, nNeg)) {
            throw new Error('specified nu is infeasible');
        }
        const K = new KernelMatrix(X, this.kernelConfig());
        return solveNuSVC(K, y, this.nu, this.tol, this.maxIter);
    }
}
registerEstimator('NuSVC', NuSVC);
