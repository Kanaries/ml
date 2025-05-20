import { SVC, SVCProps } from './svc';

export interface NuSVCProps extends SVCProps {
    nu?: number;
}

export class NuSVC extends SVC {
    private nu: number;
    constructor(props: NuSVCProps = {}) {
        const { nu = 0.5, ...rest } = props;
        const C = rest.C !== undefined ? rest.C : 1 / nu;
        super({ ...rest, C });
        this.nu = nu;
    }
}
