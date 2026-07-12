import {
    BaseEstimator,
    Params,
    encodeValue,
    decodeValue,
    loadModel,
    registerEstimator,
    registerSerializableClass,
} from '../estimator';

describe('tagged JSON codec', () => {
    it('round-trips plain data through JSON.stringify', () => {
        const value = {
            a: [1, 2.5, -3],
            b: 'text',
            c: true,
            d: null,
            nested: { x: [[1, 2], [3, 4]] },
        };
        const revived = decodeValue(JSON.parse(JSON.stringify(encodeValue(value))));
        expect(revived).toEqual(value);
    });

    it('round-trips special numbers', () => {
        const value = { inf: Infinity, ninf: -Infinity, nan: NaN, zero: -0, arr: [Infinity, NaN] };
        const revived = decodeValue(JSON.parse(JSON.stringify(encodeValue(value)))) as typeof value;
        expect(revived.inf).toBe(Infinity);
        expect(revived.ninf).toBe(-Infinity);
        expect(Number.isNaN(revived.nan)).toBe(true);
        expect(revived.arr[0]).toBe(Infinity);
        expect(Number.isNaN(revived.arr[1])).toBe(true);
    });

    it('round-trips undefined, Map, Set and typed arrays', () => {
        const value = {
            u: undefined,
            m: new Map<unknown, unknown>([[1, 'one'], ['k', [1, 2]]]),
            s: new Set([1, 2, 3]),
            f: Float64Array.from([1.5, 2.5]),
        };
        const revived = decodeValue(JSON.parse(JSON.stringify(encodeValue(value)))) as typeof value;
        expect(revived.u).toBeUndefined();
        expect(revived.m).toEqual(value.m);
        expect(revived.s).toEqual(value.s);
        expect(revived.f).toEqual(value.f);
        expect(revived.f).toBeInstanceOf(Float64Array);
    });

    it('escapes plain objects whose keys collide with tags', () => {
        const value = { $num: 'not-a-tag', other: 1 };
        const revived = decodeValue(JSON.parse(JSON.stringify(encodeValue(value))));
        expect(revived).toEqual(value);
    });

    it('round-trips registered class instances via prototype revival', () => {
        class Node {
            constructor(public split: number, public left: Node | null = null) {}
            depth(): number { return 1 + (this.left ? this.left.depth() : 0); }
        }
        registerSerializableClass('test.Node', Node);
        const tree = new Node(3, new Node(1));
        const revived = decodeValue(JSON.parse(JSON.stringify(encodeValue(tree)))) as Node;
        expect(revived).toBeInstanceOf(Node);
        expect(revived.depth()).toBe(2);
        expect(revived.split).toBe(3);
        expect(revived.left!.split).toBe(1);
    });

    it('throws a clear error on functions and unregistered classes', () => {
        expect(() => encodeValue({ cb: () => 1 })).toThrow(/Function-valued/);
        class Unregistered { x = 1; }
        expect(() => encodeValue(new Unregistered())).toThrow(/registerSerializableClass/);
    });
});

interface DummyProps { alpha?: number; beta?: string }

class DummyEstimator extends BaseEstimator {
    private alpha: number;
    private beta: string;
    private coef: number[];
    private fitted: boolean;
    constructor(props: DummyProps = {}) {
        super();
        const { alpha = 1, beta = 'auto' } = props;
        if (alpha <= 0) throw new Error('alpha must be > 0');
        this.alpha = alpha;
        this.beta = beta;
        this.coef = [];
        this.fitted = false;
    }
    public getParams(): Params {
        return { alpha: this.alpha, beta: this.beta };
    }
    public fit(x: number[]): void {
        this.coef = x.map((v) => v * this.alpha);
        this.fitted = true;
    }
    public getCoef(): number[] { return this.coef; }
    public isFitted(): boolean { return this.fitted; }
}
registerEstimator('test.DummyEstimator', DummyEstimator);

describe('BaseEstimator contract', () => {
    it('setParams merges, validates keys and reruns constructor validation', () => {
        const est = new DummyEstimator({ alpha: 2 });
        est.setParams({ beta: 'manual' });
        expect(est.getParams()).toEqual({ alpha: 2, beta: 'manual' });
        expect(() => est.setParams({ nope: 1 })).toThrow(/Invalid parameter "nope"/);
        expect(() => est.setParams({ alpha: -1 })).toThrow(/alpha must be > 0/);
    });

    it('setParams resets fitted state', () => {
        const est = new DummyEstimator();
        est.fit([1, 2]);
        expect(est.isFitted()).toBe(true);
        est.setParams({ alpha: 3 });
        expect(est.isFitted()).toBe(false);
    });

    it('clone copies params but not fitted state', () => {
        const est = new DummyEstimator({ alpha: 4, beta: 'b' });
        est.fit([1]);
        const copy = est.clone();
        expect(copy).toBeInstanceOf(DummyEstimator);
        expect(copy.getParams()).toEqual(est.getParams());
        expect(copy.isFitted()).toBe(false);
    });

    it('serializes fitted models through JSON text and revives them', () => {
        const est = new DummyEstimator({ alpha: 2 });
        est.fit([1, 2, 3]);
        const text = JSON.stringify(est);
        const revived = loadModel(text) as DummyEstimator;
        expect(revived).toBeInstanceOf(DummyEstimator);
        expect(revived.isFitted()).toBe(true);
        expect(revived.getCoef()).toEqual([2, 4, 6]);
        expect(revived.getParams()).toEqual({ alpha: 2, beta: 'auto' });
    });

    it('fromJSON enforces the class', () => {
        const est = new DummyEstimator();
        est.fit([1]);
        const revived = DummyEstimator.fromJSON(est.toJSON());
        expect(revived.getCoef()).toEqual([1]);
        class Other extends DummyEstimator {}
        registerEstimator('test.Other', Other);
        expect(() => Other.fromJSON(new DummyEstimator().toJSON())).toThrow(/not a Other/);
    });

    it('loadModel rejects foreign payloads and unknown estimators', () => {
        expect(() => loadModel('{"format":"nope"}')).toThrow(/payload/);
        const json = new DummyEstimator().toJSON();
        expect(() => loadModel({ ...json, estimator: 'missing.Estimator' })).toThrow(/Unknown estimator/);
        expect(() => loadModel({ ...json, formatVersion: 99 })).toThrow(/format version/);
    });
});
