import { HierarchicalNSW } from 'hnswlib-node';
import fs from 'fs';
import path from 'path';
import { logger } from '../logger';
import { FEATURE_DIM as DIM, DIM_WEIGHTS } from './featureEncoder';
const HNSW_DIR = path.join(process.env['INSTANA_USER_DATA'], 'hnsw');
function euclideanSq(a, b) {
    let sum = 0;
    for (let i = 0; i < DIM; i++) {
        const diff = (a[i] - b[i]) * DIM_WEIGHTS[i];
        sum += diff * diff;
    }
    return sum;
}
function addVec(a, b) {
    const out = new Float32Array(DIM);
    for (let i = 0; i < DIM; i++)
        out[i] = a[i] + b[i];
    return out;
}
function scaleVec(v, s) {
    const out = new Float32Array(DIM);
    for (let i = 0; i < DIM; i++)
        out[i] = v[i] * s;
    return out;
}
export async function clusterVectors(vectors, accountId) {
    const n = vectors.length;
    const k = Math.max(2, Math.min(8, Math.round(n / 5)));
    // k-means++ initialization
    const centroids = [new Float32Array(vectors[Math.floor(Math.random() * n)])];
    while (centroids.length < k) {
        const weights = vectors.map((v) => {
            let minD2 = Infinity;
            for (const cen of centroids) {
                const d2 = euclideanSq(v, cen);
                if (d2 < minD2)
                    minD2 = d2;
            }
            return minD2;
        });
        const total = weights.reduce((s, w) => s + w, 0);
        let r = Math.random() * total;
        let chosen = n - 1;
        for (let i = 0; i < n; i++) {
            r -= weights[i];
            if (r <= 0) {
                chosen = i;
                break;
            }
        }
        centroids.push(new Float32Array(vectors[chosen]));
    }
    // 20 iterations of k-means
    const assignments = new Array(n).fill(0);
    for (let iter = 0; iter < 20; iter++) {
        for (let i = 0; i < n; i++) {
            let best = 0;
            let bestD = Infinity;
            for (let c = 0; c < k; c++) {
                const d = euclideanSq(vectors[i], centroids[c]);
                if (d < bestD) {
                    bestD = d;
                    best = c;
                }
            }
            assignments[i] = best;
        }
        for (let c = 0; c < k; c++) {
            const members = [];
            for (let i = 0; i < n; i++) {
                if (assignments[i] === c)
                    members.push(vectors[i]);
            }
            if (members.length === 0)
                continue;
            centroids[c] = scaleVec(members.reduce(addVec, new Float32Array(DIM)), 1 / members.length);
        }
    }
    // Build and persist HNSW index — best-effort, never blocks the clustering result
    fs.mkdirSync(HNSW_DIR, { recursive: true });
    try {
        const index = new HierarchicalNSW('cosine', DIM);
        index.initIndex(n);
        for (let i = 0; i < n; i++) {
            index.addPoint(Array.from(vectors[i]), i);
        }
        const timeout = new Promise((_, reject) => setTimeout(() => reject(new Error('writeIndex timed out after 10s')), 10_000));
        await Promise.race([index.writeIndex(path.join(HNSW_DIR, `${accountId}.bin`)), timeout]);
    }
    catch (err) {
        logger.warn(`HNSW index write skipped: ${err instanceof Error ? err.message : String(err)}`);
    }
    return { assignments, centroids };
}
