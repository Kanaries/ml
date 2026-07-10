export type SubsetSizeOption = number | 'sqrt' | 'log2' | 'all';

let warnedIntegerOne = false;

/**
 * Resolve a max_features / max_samples style option to an absolute count.
 *
 * JavaScript cannot distinguish the integer 1 from the float 1.0, so sklearn's
 * "int means count, float means fraction" convention is not portable. The
 * library-wide convention is:
 *   - integer n >= 1      -> absolute count (capped at total)
 *   - 0 < x < 1           -> fraction, max(1, floor(x * total))
 *   - 'sqrt' / 'log2'     -> max(1, floor(sqrt/log2(total)))
 *   - 'all' / undefined   -> total
 * Everything else (0, negatives, NaN, non-integer > 1) throws.
 *
 * NOTE: `1` therefore means ONE feature/sample, not 100% — use 'all' for the
 * full set.
 */
export function resolveSubsetSize(
    option: SubsetSizeOption | undefined,
    total: number,
    paramName: string = 'max_features'
): number {
    if (option === undefined || option === 'all') {
        return total;
    }
    if (option === 'sqrt') {
        return Math.max(1, Math.floor(Math.sqrt(total)));
    }
    if (option === 'log2') {
        return Math.max(1, Math.floor(Math.log2(total)));
    }
    if (typeof option !== 'number' || !Number.isFinite(option) || option <= 0) {
        throw new Error(`${paramName} must be a positive number, 'sqrt', 'log2' or 'all'`);
    }
    if (option < 1) {
        return Math.max(1, Math.floor(option * total));
    }
    if (!Number.isInteger(option)) {
        throw new Error(`${paramName} must be an integer when >= 1 (got ${option}); use a fraction in (0, 1) or 'all'`);
    }
    if (option === 1 && !warnedIntegerOne) {
        warnedIntegerOne = true;
        // eslint-disable-next-line no-console
        console.warn(
            `[@kanaries/ml] ${paramName}: 1 now means ONE feature/sample (absolute count), ` +
                "not 100%. Pass 'all' (or omit the option) for the full set."
        );
    }
    return Math.min(total, option);
}
