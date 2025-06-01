import * as Stat from './stat';
import * as Sampling from './sampling';
import { asyncMode } from './asyncMode';

type Constructor<T = {}> = new (...args: any[]) => T;
export interface ClassType<InstanceType extends {} = {}> extends Function {
    new(...args: any[]): InstanceType
    prototype: InstanceType
}

export function mixins<T extends Constructor<InstanceType<T>>>(extend: T) {
    return function mixin<B extends Constructor<InstanceType<B>>>(target: B) {
        return Object.assign<Constructor<InstanceType<B>>, T>(target.prototype, extend);
    }
}

export function assert(condition: any, message: string) {
    if (!Boolean(condition)) {
        throw new Error(message);
    }
}

export {
    Stat,
    Sampling,
    asyncMode
}