import { SVR } from './svr';
import { Params, registerEstimator } from '../base/estimator';
import { KernelMatrix, KernelType, SVRSolution, solveNuSVR } from './smo';

export interface NuSVRProps {
    kernel?: KernelType;
    degree?: number;
    gamma?: number | 'scale' | 'auto';
    coef0?: number;
    tol?: number;
    C?: number;
    /**
     * upper bound on the fraction of training errors (points outside the
     * tube), lower bound on the fraction of support vectors
     */
    nu?: number;
    /** hard limit on SMO pair updates; -1 (default) = run until convergence */
    maxIter?: number;
}

/**
 * nu-Support Vector Regression (libsvm solve_nu_svr semantics): `nu`
 * replaces `epsilon` and directly trades off support-vector fraction against
 * points outside the tube; the tube half-width epsilon becomes a variable of
 * the optimization (exposed after fit as `fittedEpsilon`).
 */
export class NuSVR extends SVR {
    private nu: number;
    /** tube half-width found by the optimizer (libsvm's -r); 0 before fit */
    protected fittedEpsilon: number;

    constructor(props: NuSVRProps = {}) {
        const { nu = 0.5, ...rest } = props;
        super(rest);
        if (!Number.isFinite(nu) || nu <= 0 || nu > 1) {
            throw new Error('nu <= 0 or nu > 1');
        }
        this.nu = nu;
        this.fittedEpsilon = 0;
    }

    public getParams(): Params {
        return {
            kernel: this.kernel,
            degree: this.degree,
            gamma: this.gamma,
            coef0: this.coef0,
            tol: this.tol,
            C: this.C,
            nu: this.nu,
            maxIter: this.maxIter,
        };
    }

    protected solveDual(K: KernelMatrix, y: number[]): SVRSolution {
        const sol = solveNuSVR(K, y, this.C, this.nu, this.tol, this.maxIter);
        this.fittedEpsilon = sol.epsilon;
        return sol;
    }

    /** epsilon-tube half-width chosen by the nu-SVR optimization */
    public getFittedEpsilon(): number {
        this.checkFitted();
        return this.fittedEpsilon;
    }
}
registerEstimator('NuSVR', NuSVR);
