import OpenAI from 'openai';
import sharp from 'sharp';
import { settings } from '../config/settings';
import { tokenStore } from '../auth/tokenStore';
import { logger } from '../logger';
// ── GROK API PLACEHOLDER ──
// Used when no Grok key is configured. The pipeline still runs end-to-end,
// feature vectors are encoded, and clustering works with neutral values.
// To enable real analysis: paste your xAI key via Profile → Grok Key (or GROK_API_KEY in .env).
const STUB_ANALYSIS = {
    footage_type: 'talking_head',
    content_category: 'other',
    narrative_structure: 'single_take',
    subject: { type: 'single_person', face_to_camera: true, presenter_style: 'direct_monologue' },
    visual_style: { color_tone: 'neutral', aesthetic: 'raw' },
    scene: { background: 'plain', filming_style: 'static', lighting: 'artificial', setting: 'indoor' },
    text_overlays: {
        present: false,
        has_subtitles: false,
        overlay_count: 0,
        primary_text: '',
        hook_text: '',
        position: 'none',
        style: 'minimal',
        animated: false,
    },
    audio_type: 'voiceover',
    voiceover_style: 'none',
    hook_strength: 'medium',
    hook_type: 'bold_statement',
    pacing: 'medium',
    production_quality: 'medium',
    structural_fingerprint: [],
    confidence: 0.0,
};
let cachedApiKey = undefined;
let grokClient = null;
async function getClient() {
    if (cachedApiKey === undefined) {
        const keychainKey = await tokenStore.getGrokKey();
        cachedApiKey = keychainKey ?? process.env['GROK_API_KEY'] ?? null;
    }
    if (!cachedApiKey)
        return null;
    if (!grokClient) {
        grokClient = new OpenAI({ apiKey: cachedApiKey, baseURL: 'https://api.x.ai/v1' });
    }
    return grokClient;
}
export function resetGrokClient() {
    cachedApiKey = undefined;
    grokClient = null;
}
async function toBase64(imagePath) {
    const buf = await sharp(imagePath).resize(512, 512, { fit: 'inside' }).jpeg().toBuffer();
    return buf.toString('base64');
}
function framesForDuration(seconds) {
    if (seconds >= 30)
        return settings.frames.sentToGrok.long;
    if (seconds >= 15)
        return settings.frames.sentToGrok.medium;
    return settings.frames.sentToGrok.short;
}
function strideSelect(paths, count) {
    if (paths.length <= count)
        return paths;
    const step = paths.length / count;
    return Array.from({ length: count }, (_, i) => paths[Math.floor(i * step)]);
}
export async function analyzeFrames(framePaths, durationSeconds, detectedText, hasAudio, cutsPerSecond, knownFingerprints = []) {
    const client = await getClient();
    if (!client)
        return STUB_ANALYSIS;
    const frameCount = framesForDuration(durationSeconds);
    const selected = strideSelect(framePaths, frameCount);
    const base64Frames = await Promise.all(selected.map(toBase64));
    const imageContent = base64Frames.map((b64) => ({
        type: 'image_url',
        image_url: { url: `data:image/jpeg;base64,${b64}` },
    }));
    const intervalSecs = durationSeconds / selected.length;
    const frameLabels = selected
        .map((_, i) => `Frame ${i + 1} (~${((i + 0.5) * intervalSecs).toFixed(0)}s)`)
        .join(', ');
    const knownTemplatesBlock = knownFingerprints.length > 0
        ? `\nKnown production templates already identified in this account:\n${knownFingerprints
            .map((fp, i) => `- Template ${i + 1}: [${fp.map((s) => `"${s}"`).join(', ')}]`)
            .join('\n')}\n\nIf this video matches one of the templates above — same camera position, same background environment, same text overlay pattern — your structural_fingerprint MUST reproduce those exact phrases. Only coin new phrases if no existing template matches.\n`
        : '';
    const prompt = `You are classifying an Instagram Reel to identify the recurring production template it belongs to.
Your primary goal is to recognise format consistency — two videos shot with the same setup, same
background, and same text overlay style are the SAME FORMAT even if their topics differ completely.

Structural and visual production choices take priority over semantic content. A talking-head tutorial
and a talking-head motivational post filmed in the same location with the same framing and text layout
are one format, not two. Do not allow differences in topic, spoken content, or hook wording to
override structural similarity.
${knownTemplatesBlock}
Frames provided: ${selected.length} equally-spaced stills across a ${durationSeconds.toFixed(0)}s video (${frameLabels}).
OCR detected text: ${detectedText || 'none'}
Audio track present: ${hasAudio ? 'yes' : 'unknown'}
Measured edit-cut rate: ${cutsPerSecond.toFixed(2)} cuts/sec

Examine every frame for production template signals: camera framing and angle, subject position in
frame, background environment (be specific — not just "plain" but e.g. "white wall, no props"),
lighting rig inferred from catchlights or shadows, text overlay layout (position, font weight, number
of visible layers), color grade consistency across frames.

If OCR text is present, use its position, casing, and typographic style to anchor structural_fingerprint
classification — a recurring text overlay pattern (same position, same font weight, same caption
structure) is a strong format identity signal.

For pacing: cross-check your visual assessment against the measured cut rate above — they must agree.
A measured rate above 1.5 cuts/sec is fast or very_fast; below 0.3 cuts/sec is slow or static.

For content_category: classify by delivery mechanism, not topic. If the delivery mechanism is
ambiguous between two similar categories (e.g. "educational" vs "motivational"), prefer the broader
or more common one — a consistent label across similar videos matters more than the precise semantic
distinction.

For narrative_structure: classify the structural shape of the video, not its content. If the
structure fits both "hook_reveal" and "single_take", prefer "hook_reveal" — it is more specific.

Return ONLY valid JSON — no markdown, no explanation:
{
  "footage_type": "talking_head"|"b_roll"|"animation"|"screen_recording"|"mixed",
  "content_category": "tutorial"|"vlog"|"entertainment"|"product_showcase"|"lifestyle"|"motivational"|"educational"|"review"|"comedy"|"other",
  "narrative_structure": "hook_reveal"|"problem_solution"|"story"|"listicle"|"before_after"|"single_take"|"montage"|"other",
  "subject": {
    "type": "single_person"|"group"|"product"|"scenery"|"abstract"|"text_only",
    "face_to_camera": boolean,
    "presenter_style": "direct_monologue"|"interview"|"voiceover_only"|"silent"|"reaction"|"none"
  },
  "visual_style": {
    "color_tone": "warm"|"cool"|"neutral"|"high_contrast"|"desaturated"|"vibrant",
    "aesthetic": "cinematic"|"raw"|"polished"|"minimal"|"chaotic"|"corporate"
  },
  "scene": {
    "background": "plain"|"natural"|"urban"|"studio"|"digital"|"other",
    "filming_style": "handheld"|"tripod"|"drone"|"static"|"screen_capture",
    "lighting": "natural"|"studio"|"artificial"|"low_light"|"mixed",
    "setting": "indoor"|"outdoor"|"mixed"
  },
  "text_overlays": {
    "present": boolean,
    "has_subtitles": boolean,
    "overlay_count": integer,
    "primary_text": "exact wording of the largest or most prominent text overlay, or empty string",
    "hook_text": "text visible in Frame 1 or Frame 2, or empty string",
    "position": "top"|"center"|"bottom"|"full_screen"|"scattered"|"none",
    "style": "bold_caption"|"subtitle"|"handwritten"|"minimal"|"graphic_heavy",
    "animated": boolean
  },
  "audio_type": "music"|"voiceover"|"music_and_voiceover"|"silent"|"trending_audio",
  "voiceover_style": "educational"|"conversational"|"dramatic"|"hype"|"none",
  "hook_strength": "strong"|"medium"|"weak",
  "hook_type": "question"|"bold_statement"|"action"|"visual_surprise"|"text_only"|"product_reveal"|"story_open",
  "pacing": "very_fast"|"fast"|"medium"|"slow"|"static",
  "production_quality": "high"|"medium"|"low",
  "structural_fingerprint": ["3–5 observable production choices that identify this as a recurring template on sight: camera framing (e.g. 'tight face crop, center frame'), background detail (e.g. 'white wall, no props'), text overlay pattern (e.g. 'bold white caption, bottom third, 2 layers'), lighting signature (e.g. 'ring light catchlights'), audio signature (e.g. 'voiceover over music bed'). Observable production choices only — no content topics, no descriptions of what is said or shown."],
  "confidence": 0.0–1.0
}`;
    try {
        const response = await client.chat.completions.create({
            model: settings.grok.visionModel,
            temperature: settings.grok.temperature,
            max_tokens: settings.grok.maxTokens,
            messages: [
                {
                    role: 'system',
                    content: 'You are a video format classifier. Return ONLY valid JSON matching the schema. No markdown, no explanation.',
                },
                {
                    role: 'user',
                    content: [...imageContent, { type: 'text', text: prompt }],
                },
            ],
        });
        const raw = response.choices[0]?.message?.content ?? '';
        return JSON.parse(raw);
    }
    catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        logger.warn(`Grok API error, falling back to stub: ${msg}`);
        return STUB_ANALYSIS;
    }
}
