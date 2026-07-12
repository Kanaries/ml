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

    it('round-trips NaN/Infinity inside typed arrays', () => {
        const value = Float64Array.of(NaN, Infinity, -Infinity, 1.5);
        const revived = decodeValue(JSON.parse(JSON.stringify(encodeValue(value)))) as Float64Array;
        expect(revived).toBeInstanceOf(Float64Array);
        expect(Number.isNaN(revived[0])).toBe(true);
        expect(revived[1]).toBe(Infinity);
        expect(revived[2]).toBe(-Infinity);
        expect(revived[3]).toBe(1.5);
    });

    it('preserves negative zero through JSON text', () => {
        const revived = decodeValue(JSON.parse(JSON.stringify(encodeValue({ z: -0 })))) as { z: number };
        expect(Object.is(revived.z, -0)).toBe(true);
    });

    it('rejects prototype-polluting keys in crafted payloads', () => {
        // JSON.parse creates __proto__ as an own key; decoding must refuse it
        const polluted = JSON.parse('{"a": 1, "__proto__": {"hacked": true}}');
        expect(() => decodeValue(polluted)).toThrow(/unsafe property key/);
        const viaObjTag = { $obj: [['__proto__', { hacked: true }]] };
        expect(() => decodeValue(viaObjTag)).toThrow(/unsafe property key/);
        expect(({} as Record<string, unknown>).hacked).toBeUndefined();
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

    it('rejects prototype-polluting keys in model state', () => {
        const est = new DummyEstimator();
        est.fit([1]);
        const json = JSON.parse(JSON.stringify(est));
        const evil = JSON.parse('{"__proto__": {"hacked": true}}');
        expect(() => loadModel({ ...json, state: evil })).toThrow(/unsafe property key/);
        expect((new DummyEstimator() as unknown as Record<string, unknown>).hacked).toBeUndefined();
    });

    it('rejects inherited object keys in setParams', () => {
        const est = new DummyEstimator();
        expect(() => est.setParams({ constructor: 1 } as never)).toThrow(/Invalid parameter/);
        expect(() => est.setParams({ toString: 1 } as never)).toThrow(/Invalid parameter/);
    });

    it('nested estimators serialize as full envelopes and revive functional', () => {
        class Meta extends BaseEstimator {
            public inner: DummyEstimator;
            public bag: Map<string, DummyEstimator>;
            constructor(props: { inner?: DummyEstimator } = {}) {
                super();
                this.inner = props.inner ?? new DummyEstimator();
                this.bag = new Map([['a', new DummyEstimator({ alpha: 5 })]]);
            }
            public getParams(): Params {
                return { inner: this.inner };
            }
        }
        registerEstimator('test.Meta', Meta);
        const meta = new Meta({ inner: new DummyEstimator({ alpha: 3 }) });
        meta.inner.fit([2]);
        const revived = loadModel(JSON.stringify(meta)) as Meta;
        expect(revived.inner).toBeInstanceOf(DummyEstimator);
        expect(revived.inner.getCoef()).toEqual([6]);
        // revived nested estimators must be able to refit (ctor ran)
        revived.inner.fit([10]);
        expect(revived.inner.getCoef()).toEqual([30]);
        expect(revived.bag.get('a')).toBeInstanceOf(DummyEstimator);
        // clone deep-clones estimators inside Maps too
        const copy = meta.clone();
        expect(copy.inner).not.toBe(meta.inner);
        expect(copy.inner.getParams()).toEqual(meta.inner.getParams());
    });

    it('loadModel rejects foreign payloads and unknown estimators', () => {
        expect(() => loadModel('{"format":"nope"}')).toThrow(/payload/);
        const json = new DummyEstimator().toJSON();
        expect(() => loadModel({ ...json, estimator: 'missing.Estimator' })).toThrow(/Unknown estimator/);
        expect(() => loadModel({ ...json, formatVersion: 99 })).toThrow(/format version/);
    });
});
