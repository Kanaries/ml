import { QuadraticDiscriminantAnalysis } from '../qda';
import { LinearDiscriminantAnalysis } from '../lda';

function lcg(seed: number): () => number {
    let state = seed >>> 0;
    return () => {
        state = (state * 1664525 + 1013904223) >>> 0;
        return state / 4294967296;
    };
}

function gauss(rand: () => number): number {
    let u = 0;
    while (u === 0) u = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

function accuracy(pred: number[], truth: number[]): number {
    let ok = 0;
    for (let i = 0; i < truth.length; i++) if (pred[i] === truth[i]) ok++;
    return ok / truth.length;
}

/**
 * Concentric classes with the same mean but very different covariances: a
 * tight Gaussian blob at the origin inside a wide ring. No linear boundary
 * separates them, but the per-class covariances do.
 */
function concentricDataset(): { X: number[][]; y: number[] } {
    const rand = lcg(9);
    const X: number[][] = [];
    const y: number[] = [];
    for (let i = 0; i < 40; i++) {
        X.push([gauss(rand) * 0.3, gauss(rand) * 0.3]);
        y.push(0);
    }
    for (let i = 0; i < 40; i++) {
        const angle = (2 * Math.PI * i) / 40;
        const r = 4 + gauss(rand) * 0.3;
        X.push([r * Math.cos(angle), r * Math.sin(angle)]);
        y.push(1);
    }
    return { X, y };
}

/**
 * Hand-verifiable 2-class data with diagonal, unequal covariances.
 * Class 0: mean (0, 0), unbiased cov diag(2/3, 8/3).
 * Class 1: mean (6, 0), unbiased cov diag(2/3, 2/3).
 */
const HAND_X = [
    [-1, 0], [1, 0], [0, -2], [0, 2],
    [5, 0], [7, 0], [6, -1], [6, 1],
];
const HAND_Y = [0, 0, 0, 0, 1, 1, 1, 1];

describe('QuadraticDiscriminantAnalysis: hand-verified decision function', () => {
    it('matches the closed-form log-posterior parts for diagonal covariances', () => {
        const qda = new QuadraticDiscriminantAnalysis();
        qda.fit(HAND_X, HAND_Y);

        const point = [2, 1];
        // class 0: cov diag(2/3, 8/3), Xm = (2, 1)
        const maha0 = (2 * 2) / (2 / 3) + (1 * 1) / (8 / 3);
        const logdet0 = Math.log(2 / 3) + Math.log(8 / 3);
        const expected0 = -0.5 * (maha0 + logdet0) + Math.log(0.5);
        // class 1: cov diag(2/3, 2/3), Xm = (-4, 1)
        const maha1 = (4 * 4) / (2 / 3) + (1 * 1) / (2 / 3);
        const logdet1 = 2 * Math.log(2 / 3);
        const expected1 = -0.5 * (maha1 + logdet1) + Math.log(0.5);

        const dec = qda.decisionFunction([point]) as number[];
        expect(dec[0]).toBeCloseTo(expected1 - expected0, 8);
        expect(qda.predict([point])).toEqual([0]);
        expect(qda.predict(HAND_X)).toEqual(HAND_Y);
    });

    it('reconstructed per-class covariances match the sample covariances', () => {
        const qda = new QuadraticDiscriminantAnalysis();
        qda.fit(HAND_X, HAND_Y);
        const cov = qda.getCovariance();
        expect(cov[0][0][0]).toBeCloseTo(2 / 3, 8);
        expect(cov[0][1][1]).toBeCloseTo(8 / 3, 8);
        expect(cov[0][0][1]).toBeCloseTo(0, 8);
        expect(cov[1][0][0]).toBeCloseTo(2 / 3, 8);
        expect(cov[1][1][1]).toBeCloseTo(2 / 3, 8);
    });
});

describe('QuadraticDiscriminantAnalysis vs LDA on unequal covariances', () => {
    const { X, y } = concentricDataset();

    it('QDA separates concentric classes that defeat LDA', () => {
        const qda = new QuadraticDiscriminantAnalysis();
        qda.fit(X, y);
        const qdaAcc = accuracy(qda.predict(X), y);

        const lda = new LinearDiscriminantAnalysis();
        lda.fit(X, y);
        const ldaAcc = accuracy(lda.predict(X), y);

        expect(qdaAcc).toBeGreaterThan(0.95);
        expect(ldaAcc).toBeLessThan(0.75);
        expect(qdaAcc).toBeGreaterThan(ldaAcc);
    });

    it('predictProba rows sum to 1 and argmax matches predict', () => {
        const qda = new QuadraticDiscriminantAnalysis();
        qda.fit(X, y);
        const proba = qda.predictProba(X);
        const pred = qda.predict(X);
        const classes = qda.getClasses();
        proba.forEach((row, i) => {
            expect(row.length).toBe(2);
            expect(row[0] + row[1]).toBeCloseTo(1, 10);
            const best = row[1] > row[0] ? 1 : 0;
            expect(classes[best]).toBe(pred[i]);
        });
    });

    it('regParam > 0 fits and still separates the classes', () => {
        const plain = new QuadraticDiscriminantAnalysis();
        const regularized = new QuadraticDiscriminantAnalysis({ regParam: 0.5 });
        plain.fit(X, y);
        regularized.fit(X, y);
        expect(accuracy(regularized.predict(X), y)).toBeGreaterThan(0.9);
        // regularization actually changes the model
        const d0 = (plain.decisionFunction([[1, 1]]) as number[])[0];
        const d1 = (regularized.decisionFunction([[1, 1]]) as number[])[0];
        expect(d0).not.toBeCloseTo(d1, 6);
        // covariance eigenvalues are shrunk toward 1
        const cov = regularized.getCovariance();
        cov.forEach((c) => c.forEach((row) => row.forEach((v) => expect(Number.isFinite(v)).toBe(true))));
    });

    it('rank-deficient input (duplicate feature column) does not produce NaN', () => {
        const Xdup = X.map((row) => [row[0], row[1], row[0]]);
        const qda = new QuadraticDiscriminantAnalysis();
        qda.fit(Xdup, y);
        const dec = qda.decisionFunction(Xdup) as number[];
        dec.forEach((v) => expect(Number.isFinite(v)).toBe(true));
        qda.predictProba(Xdup).forEach((row) => row.forEach((v) => expect(Number.isFinite(v)).toBe(true)));
        expect(accuracy(qda.predict(Xdup), y)).toBeGreaterThan(0.95);
    });
});

describe('QuadraticDiscriminantAnalysis: validation', () => {
    it('a class with fewer than 2 samples throws', () => {
        const qda = new QuadraticDiscriminantAnalysis();
        expect(() => qda.fit([[0, 0], [1, 1], [5, 5]], [0, 0, 1])).toThrow(/only 1 sample/);
    });

    it('regParam outside [0, 1] throws', () => {
        const qda = new QuadraticDiscriminantAnalysis({ regParam: 2 });
        expect(() => qda.fit(HAND_X, HAND_Y)).toThrow(/regParam/);
    });

    it('predict before fit throws', () => {
        const qda = new QuadraticDiscriminantAnalysis();
        expect(() => qda.predict([[0, 0]])).toThrow(/fitted/);
    });

    it('explicit priors are normalized and used', () => {
        const qda = new QuadraticDiscriminantAnalysis({ priors: [3, 1] });
        qda.fit(HAND_X, HAND_Y);
        expect(qda.getPriors()).toEqual([0.75, 0.25]);
    });
});
