import { igClient } from './client';
import { db } from '../db/index';
import { media, mediaInsights } from '../db/schema';
import { eq } from 'drizzle-orm';
import { logger } from '../logger';
function extractValue(insight) {
    return insight.total_value?.value ?? insight.values?.[0]?.value ?? insight.value ?? 0;
}
async function fetchInsightsForMedia(mediaId, mediaType) {
    const viewMetric = mediaType === 'REEL' ? 'plays' : 'views';
    try {
        const { data } = await igClient.get(`/${mediaId}/insights`, {
            params: { metric: `${viewMetric},likes,comments,saved,shares`, period: 'lifetime' },
        });
        const vals = {};
        for (const insight of data.data) {
            vals[insight.name] = extractValue(insight);
        }
        const plays = vals[viewMetric] ?? 0;
        const likes = vals['likes'] ?? 0;
        const comments = vals['comments'] ?? 0;
        const saves = vals['saved'] ?? 0;
        const shares = vals['shares'] ?? 0;
        const totalInteractions = likes + comments + saves + shares;
        const engagementRate = plays > 0 ? (totalInteractions / plays) * 100 : 0;
        db.insert(mediaInsights).values({
            mediaId,
            plays,
            views: null,
            likes,
            comments,
            reposts: saves,
            sends: shares,
            engagementRate,
        }).onConflictDoUpdate({
            target: mediaInsights.mediaId,
            set: { plays, views: null, likes, comments, reposts: saves, sends: shares, engagementRate },
        }).run();
    }
    catch (err) {
        const body = err.response?.data;
        logger.warn(`Skipping insights for ${mediaId} — ${String(err)}${body ? ` | API response: ${JSON.stringify(body)}` : ''}`);
    }
}
const INSIGHTS_BATCH_SIZE = 5;
export async function fetchAllInsights(accountId) {
    const items = db.select({ id: media.id, mediaType: media.mediaType }).from(media).where(eq(media.accountId, accountId)).all();
    for (let i = 0; i < items.length; i += INSIGHTS_BATCH_SIZE) {
        await Promise.all(items.slice(i, i + INSIGHTS_BATCH_SIZE).map((item) => fetchInsightsForMedia(item.id, item.mediaType)));
    }
    logger.info(`Insights fetched for ${items.length} reels`);
}
