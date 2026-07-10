import { SVC, SVCProps } from './svc';

export interface NuSVCProps extends SVCProps {
    nu?: number;
}

let warned = false;

/**
 * @deprecated NuSVC does NOT implement ν-SVM semantics. `nu` is only used to
 * derive a default cost (`C = 1/nu`) and has no effect when `C` is passed;
 * it does not bound the support-vector fraction or margin errors, and results
 * will not match scikit-learn's NuSVC. Scheduled for removal or full
 * reimplementation — use SVC or LinearSVC instead.
 */
export class NuSVC extends SVC {
    private nu: number;
    constructor(props: NuSVCProps = {}) {
        const { nu = 0.5, ...rest } = props;
        const C = rest.C !== undefined ? rest.C : 1 / nu;
        super({ ...rest, C });
        this.nu = nu;
        if (!warned) {
            warned = true;
            // eslint-disable-next-line no-console
            console.warn(
                '[@kanaries/ml] NuSVC is deprecated: it does not implement ν-SVM semantics ' +
                    '(`nu` only sets a default C = 1/nu). Use SVC or LinearSVC instead.'
            );
        }
    }
}
