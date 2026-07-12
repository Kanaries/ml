/**
 * Copy of `props` without the keys whose value is undefined, so spreading
 * the result cannot override downstream defaults with undefined (e.g.
 * `{ criterion: 'gini', ...treeProps }`). Needed because getParams() always
 * returns the full key set (with undefined for unset optional params) and
 * setParams()/clone() feed that back through the constructor.
 */
export function definedProps<T extends object>(props: T): T {
    const out = {} as T;
    for (const key of Object.keys(props) as (keyof T)[]) {
        if (props[key] !== undefined) out[key] = props[key];
    }
    return out;
}
