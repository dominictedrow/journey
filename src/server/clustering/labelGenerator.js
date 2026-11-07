import OpenAI from 'openai';
import { tokenStore } from '../auth/tokenStore';
import { settings } from '../config/settings';
import { logger } from '../logger';
function describeCluster(c, i) {
    return `Cluster ${i}: footage=${c.footageType}, category=${c.contentCategory}, structure=${c.narrativeStructure}, subject=${c.subjectType}, presenter=${c.presenterStyle}, face_to_camera=${c.faceToCamera}, background=${c.backgroundType}, filming=${c.filmingStyle}, lighting=${c.lighting}, setting=${c.setting}, color=${c.colorTone}, aesthetic=${c.aesthetic}, audio=${c.audioType}, voiceover=${c.voiceoverStyle}, hook=${c.hookStrength} (${c.hookType}), pacing=${c.pacing}, quality=${c.productionQuality}, cuts/sec=${c.cutsPerSecond}`;
}
export async function generateFormatLabels(clusters) {
    const fallback = clusters.map((c) => c.footageType);
    const keychainKey = await tokenStore.getGrokKey();
    const apiKey = keychainKey ?? process.env['GROK_API_KEY'] ?? null;
    if (!apiKey)
        return fallback;
    const client = new OpenAI({ apiKey, baseURL: 'https://api.x.ai/v1' });
    const prompt = `You are naming Instagram Reel content formats. These ${clusters.length} clusters were grouped automatically by visual and structural similarity. Give each a short, descriptive name (3–6 words) that captures the specific combination of qualities that caused those videos to cluster together — not just the footage type alone.

${clusters.map(describeCluster).join('\n')}

Return ONLY valid JSON: {"labels": ["name for cluster 0", "name for cluster 1", ...]}`;
    try {
        const response = await client.chat.completions.create({
            model: settings.grok.textModel,
            temperature: 0,
            max_tokens: 256,
            messages: [
                { role: 'system', content: 'Return ONLY valid JSON. No markdown, no explanation.' },
                { role: 'user', content: prompt },
            ],
        });
        const raw = response.choices[0]?.message?.content ?? '{}';
        const parsed = JSON.parse(raw);
        if (!Array.isArray(parsed.labels) || parsed.labels.length !== clusters.length) {
            logger.warn('Grok label generation returned unexpected shape, using fallback');
            return fallback;
        }
        return parsed.labels;
    }
    catch (err) {
        logger.warn(`Grok label generation failed, using fallback: ${err instanceof Error ? err.message : String(err)}`);
        return fallback;
    }
}
