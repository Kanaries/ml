/**
 * Core estimator contract shared by every model in the library.
 *
 * Every estimator must:
 *  - accept a single props object in its constructor (positional overloads may
 *    be kept for backwards compatibility, but the props form is canonical);
 *  - implement `getParams()` returning exactly the props-object shape;
 *  - be registered via `registerEstimator()` so it can be revived by
 *    `loadModel()`.
 *
 * `setParams`, `clone`, `toJSON` and `loadModel` are then provided generically
 * by `BaseEstimator`.
 *
 * Serialization format: a versioned, JSON-safe envelope. Values that plain
 * JSON cannot express are wrapped in single-key tag objects ("$num", "$map",
 * "$set", "$typed", "$cls", "$undef", "$obj"). Class instances held in fitted
 * state (tree nodes, KD-trees, nested estimators, ...) must be registered with
 * `registerSerializableClass()` (estimators are registered automatically).
 */

export type Params = Record<string, unknown>;

export const MODEL_FORMAT = '@kanaries/ml-model';
export const MODEL_FORMAT_VERSION = 1;

export interface SerializedModel {
    format: typeof MODEL_FORMAT;
    formatVersion: number;
    estimator: string;
    params: unknown;
    state: unknown;
}

type AnyCtor = new (...args: never[]) => object;

const estimatorRegistry = new Map<string, AnyCtor>();
const serializableClasses = new Map<string, AnyCtor>();
const serializableNames = new Map<AnyCtor, string>();

export function registerSerializableClass(name: string, ctor: AnyCtor): void {
    const existing = serializableClasses.get(name);
    if (existing && existing !== ctor) {
        throw new Error(`A different class is already registered as "${name}"`);
    }
    serializableClasses.set(name, ctor);
    serializableNames.set(ctor, name);
}

export function registerEstimator(name: string, ctor: AnyCtor): void {
    const existing = estimatorRegistry.get(name);
    if (existing && existing !== ctor) {
        throw new Error(`A different estimator is already registered as "${name}"`);
    }
    estimatorRegistry.set(name, ctor);
    registerSerializableClass(name, ctor);
}

export function getRegisteredEstimators(): ReadonlyMap<string, AnyCtor> {
    return estimatorRegistry;
}

// ---------------------------------------------------------------------------
// Tagged-JSON codec
// ---------------------------------------------------------------------------

/**
 * Keys that could mutate an object's prototype chain when assigned from a
 * crafted payload. Rejected everywhere the codec writes decoded keys.
 */
function assertSafeKey(key: string): void {
    if (key === '__proto__' || key === 'constructor' || key === 'prototype') {
        throw new Error(`Refusing to decode unsafe property key "${key}"`);
    }
}

const TYPED_ARRAY_CTORS: Record<string, (values: number[]) => ArrayBufferView> = {
    Float64Array: (v) => Float64Array.from(v),
    Float32Array: (v) => Float32Array.from(v),
    Int32Array: (v) => Int32Array.from(v),
    Int16Array: (v) => Int16Array.from(v),
    Int8Array: (v) => Int8Array.from(v),
    Uint32Array: (v) => Uint32Array.from(v),
    Uint16Array: (v) => Uint16Array.from(v),
    Uint8Array: (v) => Uint8Array.from(v),
    Uint8ClampedArray: (v) => Uint8ClampedArray.from(v),
};

function encodeOwnProps(value: object): Record<string, unknown> {
    const out: Record<string, unknown> = {};
    for (const key of Object.keys(value)) {
        out[key] = encodeValue((value as Record<string, unknown>)[key]);
    }
    return out;
}

export function encodeValue(value: unknown): unknown {
    if (value === undefined) return { $undef: true };
    if (value === null) return null;
    const t = typeof value;
    if (t === 'number') {
        const n = value as number;
        if (Number.isNaN(n)) return { $num: 'nan' };
        if (n === Infinity) return { $num: 'inf' };
        if (n === -Infinity) return { $num: '-inf' };
        if (Object.is(n, -0)) return { $num: '-0' };
        return n;
    }
    if (t === 'boolean' || t === 'string') return value;
    if (t === 'function' || t === 'symbol' || t === 'bigint') {
        throw new Error(`Cannot serialize a value of type "${t}". ` +
            'Function-valued params (e.g. custom callbacks) are not serializable; use a named built-in instead.');
    }
    if (Array.isArray(value)) return value.map(encodeValue);
    if (ArrayBuffer.isView(value) && !(value instanceof DataView)) {
        const name = value.constructor.name;
        if (!TYPED_ARRAY_CTORS[name]) throw new Error(`Unsupported typed array ${name}`);
        // elements go through the codec so NaN/±Infinity survive JSON text
        return { $typed: name, values: Array.from(value as unknown as ArrayLike<number>, encodeValue) };
    }
    if (value instanceof Map) {
        return { $map: Array.from(value.entries(), ([k, v]) => [encodeValue(k), encodeValue(v)]) };
    }
    if (value instanceof Set) {
        return { $set: Array.from(value, encodeValue) };
    }
    if (value instanceof BaseEstimator) {
        // full envelope: revival runs the constructor, so hidden
        // (non-enumerable) fields installed there are restored
        return { $est: value.toJSON() };
    }
    const proto = Object.getPrototypeOf(value);
    if (proto === Object.prototype || proto === null) {
        const obj = value as Record<string, unknown>;
        const keys = Object.keys(obj);
        if (keys.some((k) => k.startsWith('$'))) {
            return { $obj: keys.map((k) => [k, encodeValue(obj[k])]) };
        }
        const out: Record<string, unknown> = {};
        for (const k of keys) out[k] = encodeValue(obj[k]);
        return out;
    }
    const clsName = serializableNames.get(proto.constructor);
    if (clsName !== undefined) {
        return { $cls: clsName, state: encodeOwnProps(value as object) };
    }
    throw new Error(`Cannot serialize an instance of "${proto.constructor?.name ?? 'unknown'}". ` +
        'Register the class with registerSerializableClass() to make it serializable.');
}

function isTag(value: Record<string, unknown>, key: string): boolean {
    return key in value;
}

export function decodeValue(value: unknown): unknown {
    if (value === null || typeof value !== 'object') return value;
    if (Array.isArray(value)) return value.map(decodeValue);
    const obj = value as Record<string, unknown>;
    if (isTag(obj, '$undef')) return undefined;
    if (isTag(obj, '$num')) {
        switch (obj.$num) {
            case 'nan': return NaN;
            case 'inf': return Infinity;
            case '-inf': return -Infinity;
            case '-0': return -0;
            default: throw new Error(`Unknown numeric tag "${obj.$num}"`);
        }
    }
    if (isTag(obj, '$typed')) {
        const make = TYPED_ARRAY_CTORS[obj.$typed as string];
        if (!make) throw new Error(`Unknown typed array "${obj.$typed}"`);
        return make((obj.values as unknown[]).map(decodeValue) as number[]);
    }
    if (isTag(obj, '$est')) {
        return loadModel(obj.$est as SerializedModel);
    }
    if (isTag(obj, '$map')) {
        return new Map((obj.$map as [unknown, unknown][]).map(([k, v]) => [decodeValue(k), decodeValue(v)]));
    }
    if (isTag(obj, '$set')) {
        return new Set((obj.$set as unknown[]).map(decodeValue));
    }
    if (isTag(obj, '$obj')) {
        const out: Record<string, unknown> = {};
        for (const [k, v] of obj.$obj as [string, unknown][]) {
            assertSafeKey(k);
            out[k] = decodeValue(v);
        }
        return out;
    }
    if (isTag(obj, '$cls')) {
        const ctor = serializableClasses.get(obj.$cls as string);
        if (!ctor) {
            throw new Error(`Cannot deserialize unregistered class "${obj.$cls}". ` +
                'Make sure the module defining it has been imported.');
        }
        const inst = Object.create(ctor.prototype) as Record<string, unknown>;
        const state = obj.state as Record<string, unknown>;
        for (const k of Object.keys(state)) {
            assertSafeKey(k);
            inst[k] = decodeValue(state[k]);
        }
        return inst;
    }
    const out: Record<string, unknown> = {};
    for (const k of Object.keys(obj)) {
        assertSafeKey(k);
        out[k] = decodeValue(obj[k]);
    }
    return out;
}

// ---------------------------------------------------------------------------
// BaseEstimator
// ---------------------------------------------------------------------------

export abstract class BaseEstimator {
    /**
     * Return the constructor parameters of this estimator, using exactly the
     * key names accepted by the props-object constructor. Must reflect current
     * values (after any setParams), not the values passed at construction.
     */
    public abstract getParams(): Params;

    /**
     * Merge `params` into the estimator's parameters and reset it to the
     * unfitted state (the estimator is rebuilt through its constructor so all
     * validation logic reruns). Unknown keys throw.
     */
    public setParams(params: Params): this {
        const known = this.getParams();
        for (const key of Object.keys(params)) {
            if (!Object.prototype.hasOwnProperty.call(known, key)) {
                throw new Error(`Invalid parameter "${key}" for estimator ${this.constructor.name}. ` +
                    `Valid parameters are: ${Object.keys(known).join(', ')}.`);
            }
        }
        const Ctor = this.constructor as new (props: Params) => this;
        const fresh = new Ctor({ ...known, ...params });
        // rebuild via property descriptors so non-enumerable hidden fields
        // (e.g. RNG closures) are replaced too, not silently kept
        for (const key of Object.getOwnPropertyNames(this)) {
            delete (this as Record<string, unknown>)[key];
        }
        for (const key of Object.getOwnPropertyNames(fresh)) {
            Object.defineProperty(this, key, Object.getOwnPropertyDescriptor(fresh, key)!);
        }
        return this;
    }

    /**
     * A new unfitted estimator with identical parameters. Nested estimators
     * inside params (meta-estimators: pipelines, search, ensembles) are
     * themselves cloned, so the copy shares no mutable estimator state.
     */
    public clone(): this {
        const Ctor = this.constructor as new (props: Params) => this;
        return new Ctor(cloneNestedEstimators(this.getParams()) as Params);
    }

    /**
     * Serialize the estimator (parameters + fitted state) to a JSON-safe
     * object. `JSON.stringify(model)` therefore produces a portable model
     * file; revive it with `loadModel()`.
     */
    public toJSON(): SerializedModel {
        const name = serializableNames.get(this.constructor as AnyCtor);
        if (name === undefined || !estimatorRegistry.has(name)) {
            throw new Error(`Estimator ${this.constructor.name} is not registered; ` +
                'call registerEstimator() next to its class definition.');
        }
        return {
            format: MODEL_FORMAT,
            formatVersion: MODEL_FORMAT_VERSION,
            estimator: name,
            params: encodeValue(this.getParams()),
            state: encodeOwnProps(this),
        };
    }

    /** Typed convenience wrapper around `loadModel` for a known class. */
    public static fromJSON<T extends BaseEstimator>(this: new (...args: never[]) => T, json: SerializedModel | string): T {
        const model = loadModel(json);
        if (!(model instanceof this)) {
            throw new Error(`Serialized model is a ${model.constructor.name}, not a ${this.name}`);
        }
        return model;
    }
}

/** Recursively clone BaseEstimator values inside a params structure. */
function cloneNestedEstimators(value: unknown): unknown {
    if (value instanceof BaseEstimator) return value.clone();
    if (Array.isArray(value)) return value.map(cloneNestedEstimators);
    if (value instanceof Map) {
        return new Map(Array.from(value.entries(), ([k, v]) => [cloneNestedEstimators(k), cloneNestedEstimators(v)]));
    }
    if (value instanceof Set) {
        return new Set(Array.from(value, cloneNestedEstimators));
    }
    if (value !== null && typeof value === 'object') {
        const proto = Object.getPrototypeOf(value);
        if (proto === Object.prototype || proto === null) {
            const out: Record<string, unknown> = {};
            for (const k of Object.keys(value)) out[k] = cloneNestedEstimators((value as Record<string, unknown>)[k]);
            return out;
        }
    }
    return value;
}

/** Revive any serialized estimator produced by `estimator.toJSON()`. */
export function loadModel(json: SerializedModel | string): BaseEstimator {
    const model: SerializedModel = typeof json === 'string' ? JSON.parse(json) : json;
    if (model === null || typeof model !== 'object' || model.format !== MODEL_FORMAT) {
        throw new Error(`Not a ${MODEL_FORMAT} payload`);
    }
    if (model.formatVersion !== MODEL_FORMAT_VERSION) {
        throw new Error(`Unsupported model format version ${model.formatVersion} (supported: ${MODEL_FORMAT_VERSION})`);
    }
    const Ctor = estimatorRegistry.get(model.estimator) as (new (props: unknown) => BaseEstimator) | undefined;
    if (!Ctor) {
        throw new Error(`Unknown estimator "${model.estimator}". ` +
            'Make sure the module defining it has been imported before calling loadModel().');
    }
    const inst = new Ctor(decodeValue(model.params));
    for (const key of Object.keys(inst)) {
        delete (inst as unknown as Record<string, unknown>)[key];
    }
    const state = model.state as Record<string, unknown>;
    for (const key of Object.keys(state)) {
        assertSafeKey(key);
        (inst as unknown as Record<string, unknown>)[key] = decodeValue(state[key]);
    }
    return inst;
}
