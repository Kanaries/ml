import fs from 'fs';
import path from 'path';
import { getRegisteredEstimators, loadModel } from '../../base';
import { NearestNeighbors } from '../nearestNeighbors';

const fixture = JSON.parse(fs.readFileSync(path.join(__dirname, '../../../test_data/wave_c.json'), 'utf8')).advanced.nearest_neighbors;

describe('NearestNeighbors', () => {
    test('follows registration, params, clone, and reset conformance', () => {
        const model = new NearestNeighbors({ nNeighbors: 2, algorithm: 'brute' });
        expect(getRegisteredEstimators().get('NearestNeighbors')).toBe(NearestNeighbors);
        expect(model.clone().getParams()).toEqual(model.getParams());
        expect(() => model.setParams({ missing: true })).toThrow(/Invalid parameter/);
        model.fit(fixture.X); model.setParams({ nNeighbors: 1 });
        expect(() => model.kneighbors(fixture.query)).toThrow(/not fitted/);
        model.fit(fixture.X); expect((model.kneighbors(fixture.query, 1, false) as number[][])).toHaveLength(fixture.query.length);
    });
    test.each(['brute', 'kdTree', 'ballTree'] as const)('%s matches sklearn query fixture', algorithm => {
        const model = new NearestNeighbors({ nNeighbors: 2, algorithm });
        model.fit(fixture.X);
        expect(model.kneighbors(fixture.query)).toEqual({ distances: fixture.distances, indices: fixture.indices });
        expect(model.kneighbors()).toEqual({ distances: fixture.own_distances, indices: fixture.own_indices });
        expect(model.radiusNeighbors(fixture.query, 1.5)).toEqual({ distances: fixture.radius_distances, indices: fixture.radius_indices });
        expect(model.radiusNeighbors(undefined, 0)).toEqual({ distances: fixture.self_radius_distances, indices: fixture.self_radius_indices });
    });
    test('rejects non-metric Minkowski powers required by tree pruning', () => { expect(() => new NearestNeighbors({ p: .5 })).toThrow(/at least 1/); });
    test('requires distances when sorting radius results', () => { const model = new NearestNeighbors(); model.fit(fixture.X); expect(() => model.radiusNeighbors(fixture.query, 2, false, true)).toThrow(/returnDistance/); });
    test('sorts radius results by distance then index when requested', () => {
        const model = new NearestNeighbors({ algorithm: 'brute' }); model.fit([[2], [0], [1]]);
        expect(model.radiusNeighbors([[.9]], 2, true, true)).toEqual({ distances: [[.09999999999999998, .9, 1.1]], indices: [[2, 1, 0]] });
    });
    test('clone and serialization retain fitted searchable data', () => {
        const model = new NearestNeighbors({ nNeighbors: 1, algorithm: 'brute' }); model.fit([[0, 0], [2, 2]]);
        expect(model.clone().getParams()).toEqual(model.getParams());
        const revived = loadModel(JSON.stringify(model)) as NearestNeighbors;
        const result = revived.kneighbors([[1.9, 2.1]]) as { distances: number[][]; indices: number[][] };
        expect(result.indices).toEqual([[1]]); expect(result.distances[0][0]).toBeCloseTo(Math.sqrt(.02), 12);
    });
});
