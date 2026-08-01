export interface FrozenScenario {
    readonly id: string;
    readonly title: string;
    readonly frozenAt: '2026-07-31';
    readonly dataset: {
        readonly name: string;
        readonly source: string;
        readonly protocol: string;
    };
    readonly workflow: readonly string[];
    readonly algorithms: {
        readonly include: readonly string[];
        readonly exclude: readonly string[];
    };
    readonly parity: {
        readonly state: 'pending' | 'green';
        readonly blockedBy: readonly string[];
        readonly reason: string;
    };
}

export function freezeScenario<T extends FrozenScenario>(scenario: T): Readonly<T> {
    Object.freeze(scenario.dataset);
    Object.freeze(scenario.workflow);
    Object.freeze(scenario.algorithms.include);
    Object.freeze(scenario.algorithms.exclude);
    Object.freeze(scenario.algorithms);
    Object.freeze(scenario.parity.blockedBy);
    Object.freeze(scenario.parity);
    return Object.freeze(scenario);
}
