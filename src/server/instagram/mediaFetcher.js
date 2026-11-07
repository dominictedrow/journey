import { igClient } from './client';
import { db } from '../db/index';
import { media } from '../db/schema';
import { logger } from '../logger';
export async function fetchAllMedia(accountId) {
    let cursor;
    let total = 0;
    do {
        const params = {
            fields: 'id,media_type,media_url,thumbnail_url,caption,timestamp',
            limit: '50',
        };
        if (cursor)
            params['after'] = cursor;
        const { data: page } = await igClient.get(`/${accountId}/media`, { params });
        if (!page.data.length)
            break;
        for (const item of page.data) {
            if (item.media_type !== 'REEL' && item.media_type !== 'VIDEO')
                continue;
            db.insert(media).values({
                id: item.id,
                accountId,
                mediaType: item.media_type,
                mediaUrl: item.media_url ?? null,
                thumbnailUrl: item.thumbnail_url ?? null,
                caption: item.caption ?? null,
                timestamp: item.timestamp ? new Date(item.timestamp) : null,
            }).onConflictDoUpdate({
                target: media.id,
                set: {
                    mediaType: item.media_type,
                    mediaUrl: item.media_url ?? null,
                    thumbnailUrl: item.thumbnail_url ?? null,
                    caption: item.caption ?? null,
                },
            }).run();
            total++;
        }
        cursor = page.paging?.cursors?.after;
        logger.info(`Reels fetch: ${total} reels so far`);
    } while (cursor);
    return total;
}
