export const FEATURE_DIM = 111;
const FOOTAGE_TYPES = ['talking_head', 'b_roll', 'animation', 'screen_recording', 'mixed'];
const CONTENT_CATEGORIES = [
    'tutorial', 'vlog', 'entertainment', 'product_showcase', 'lifestyle',
    'motivational', 'educational', 'review', 'comedy', 'other',
];
const NARRATIVE_STRUCTURES = [
    'hook_reveal', 'problem_solution', 'story', 'listicle',
    'before_after', 'single_take', 'montage', 'other',
];
const SUBJECT_TYPES = ['single_person', 'group', 'product', 'scenery', 'abstract', 'text_only'];
const PRESENTER_STYLES = ['direct_monologue', 'interview', 'voiceover_only', 'silent', 'reaction'];
const BACKGROUND_TYPES = ['plain', 'natural', 'urban', 'studio', 'digital', 'other'];
const FILMING_STYLES = ['handheld', 'tripod', 'drone', 'static', 'screen_capture'];
const LIGHTING_TYPES = ['natural', 'studio', 'artificial', 'low_light', 'mixed'];
const SETTING_TYPES = ['indoor', 'outdoor', 'mixed'];
const COLOR_TONES = ['warm', 'cool', 'neutral', 'high_contrast', 'desaturated', 'vibrant'];
const AESTHETICS = ['cinematic', 'raw', 'polished', 'minimal', 'chaotic', 'corporate'];
const AUDIO_TYPES = ['music', 'voiceover', 'music_and_voiceover', 'silent', 'trending_audio'];
const VOICEOVER_STYLES = ['educational', 'conversational', 'dramatic', 'hype', 'none'];
const HOOK_STRENGTHS = ['strong', 'medium', 'weak'];
const HOOK_TYPES = [
    'question', 'bold_statement', 'action', 'visual_surprise',
    'text_only', 'product_reveal', 'story_open',
];
const PACING_TYPES = ['very_fast', 'fast', 'medium', 'slow', 'static'];
const PRODUCTION_QUALITIES = ['high', 'medium', 'low'];
const TEXT_STYLES = ['bold_caption', 'subtitle', 'handwritten', 'minimal', 'graphic_heavy'];
const TEXT_POSITIONS = ['top', 'center', 'bottom', 'full_screen', 'scattered', 'none'];
function oneHot(value, classes) {
    return classes.map((c) => (c === value ? 1.0 : 0.0));
}
function argmax(arr, start, end) {
    let best = start;
    for (let i = start + 1; i <= end; i++) {
        if ((arr[i] ?? 0) > (arr[best] ?? 0))
            best = i;
    }
    return best;
}
// Vector layout (111 dims):
// 0–4:     footage_type (5)           ← HIGH RELIABILITY 2×
// 5–14:    content_category (10)
// 15–22:   narrative_structure (8)
// 23–28:   subject.type (6)
// 29:      subject.face_to_camera (1) ← HIGH RELIABILITY 2×
// 30–34:   subject.presenter_style (5) — 'none' → all zeros
// 35–40:   scene.background (6)       ← HIGH RELIABILITY 2×
// 41–45:   scene.filming_style (5)    ← HIGH RELIABILITY 2×
// 46–50:   scene.lighting (5)
// 51–53:   scene.setting (3)
// 54–59:   visual_style.color_tone (6) ← LOW RELIABILITY 0.5×
// 60–65:   visual_style.aesthetic (6)  ← LOW RELIABILITY 0.5×
// 66–70:   audio_type (5) — trending_audio is a distinct 5th class
// 71–75:   voiceover_style (5)
// 76–78:   hook_strength (3)
// 79–85:   hook_type (7)
// 86–90:   pacing (5)
// 91–93:   production_quality (3)
// 94:      text_overlays.present (1)  ← HIGH RELIABILITY 2×
// 95:      text_overlays.has_subtitles (1)
// 96:      text_overlays.animated (1)
// 97–101:  text_overlays.style (5)
// 102–107: text_overlays.position (6)
// 108:     text_overlays.overlay_count (1 scalar) — 0=none, 0.33=1–2, 0.67=3–5, 1.0=6+
// 109:     cuts_per_second (normalized) — Math.min(cps / 5.0, 1.0)
// 110:     confidence
// Dimension weights for euclidean distance.
// Applied by callers before computing distance — not during centroid updates.
export const DIM_WEIGHTS = (() => {
    const w = new Float32Array(FEATURE_DIM).fill(1.0);
    for (let d = 0; d <= 4; d++)
        w[d] = 2.0; // footage_type
    w[29] = 2.0; // face_to_camera
    for (let d = 35; d <= 45; d++)
        w[d] = 2.0; // background + filming_style
    for (let d = 54; d <= 65; d++)
        w[d] = 0.5; // color_tone + aesthetic
    w[94] = 2.0; // text_overlays.present
    return w;
})();
export function encodeAnalysis(analysis, cutsPerSecond) {
    const vec = new Float32Array(FEATURE_DIM);
    let offset = 0;
    const write = (bits) => {
        for (const b of bits)
            vec[offset++] = b;
    };
    write(oneHot(analysis.footage_type, FOOTAGE_TYPES));
    write(oneHot(analysis.content_category, CONTENT_CATEGORIES));
    write(oneHot(analysis.narrative_structure, NARRATIVE_STRUCTURES));
    write(oneHot(analysis.subject.type, SUBJECT_TYPES));
    vec[offset++] = analysis.subject.face_to_camera ? 1.0 : 0.0;
    write(oneHot(analysis.subject.presenter_style, PRESENTER_STYLES));
    write(oneHot(analysis.scene.background, BACKGROUND_TYPES));
    write(oneHot(analysis.scene.filming_style, FILMING_STYLES));
    write(oneHot(analysis.scene.lighting, LIGHTING_TYPES));
    write(oneHot(analysis.scene.setting, SETTING_TYPES));
    write(oneHot(analysis.visual_style.color_tone, COLOR_TONES));
    write(oneHot(analysis.visual_style.aesthetic, AESTHETICS));
    write(oneHot(analysis.audio_type, AUDIO_TYPES));
    write(oneHot(analysis.voiceover_style, VOICEOVER_STYLES));
    write(oneHot(analysis.hook_strength, HOOK_STRENGTHS));
    write(oneHot(analysis.hook_type, HOOK_TYPES));
    write(oneHot(analysis.pacing, PACING_TYPES));
    write(oneHot(analysis.production_quality, PRODUCTION_QUALITIES));
    vec[offset++] = analysis.text_overlays.present ? 1.0 : 0.0;
    vec[offset++] = analysis.text_overlays.has_subtitles ? 1.0 : 0.0;
    vec[offset++] = analysis.text_overlays.animated ? 1.0 : 0.0;
    write(oneHot(analysis.text_overlays.style, TEXT_STYLES));
    write(oneHot(analysis.text_overlays.position, TEXT_POSITIONS));
    const oc = analysis.text_overlays.overlay_count;
    vec[offset++] = oc === 0 ? 0.0 : oc <= 2 ? 0.33 : oc <= 5 ? 0.67 : 1.0;
    vec[offset++] = Math.min(cutsPerSecond / 5.0, 1.0);
    vec[offset++] = analysis.confidence;
    return vec;
}
export function decodeCentroid(centroid) {
    return {
        footageType: (FOOTAGE_TYPES[argmax(centroid, 0, 4) - 0] ?? 'mixed').replace(/_/g, ' '),
        contentCategory: (CONTENT_CATEGORIES[argmax(centroid, 5, 14) - 5] ?? 'other').replace(/_/g, ' '),
        narrativeStructure: (NARRATIVE_STRUCTURES[argmax(centroid, 15, 22) - 15] ?? 'other').replace(/_/g, ' '),
        subjectType: (SUBJECT_TYPES[argmax(centroid, 23, 28) - 23] ?? 'single_person').replace(/_/g, ' '),
        faceToCamera: (centroid[29] ?? 0) >= 0.5,
        presenterStyle: (PRESENTER_STYLES[argmax(centroid, 30, 34) - 30] ?? 'none').replace(/_/g, ' '),
        backgroundType: (BACKGROUND_TYPES[argmax(centroid, 35, 40) - 35] ?? 'other').replace(/_/g, ' '),
        filmingStyle: (FILMING_STYLES[argmax(centroid, 41, 45) - 41] ?? 'static').replace(/_/g, ' '),
        lighting: (LIGHTING_TYPES[argmax(centroid, 46, 50) - 46] ?? 'mixed').replace(/_/g, ' '),
        setting: SETTING_TYPES[argmax(centroid, 51, 53) - 51] ?? 'mixed',
        colorTone: (COLOR_TONES[argmax(centroid, 54, 59) - 54] ?? 'neutral').replace(/_/g, ' '),
        aesthetic: (AESTHETICS[argmax(centroid, 60, 65) - 60] ?? 'raw').replace(/_/g, ' '),
        audioType: (AUDIO_TYPES[argmax(centroid, 66, 70) - 66] ?? 'music').replace(/_/g, ' '),
        voiceoverStyle: (VOICEOVER_STYLES[argmax(centroid, 71, 75) - 71] ?? 'none').replace(/_/g, ' '),
        hookStrength: HOOK_STRENGTHS[argmax(centroid, 76, 78) - 76] ?? 'medium',
        hookType: (HOOK_TYPES[argmax(centroid, 79, 85) - 79] ?? 'bold_statement').replace(/_/g, ' '),
        pacing: (PACING_TYPES[argmax(centroid, 86, 90) - 86] ?? 'medium').replace(/_/g, ' '),
        productionQuality: PRODUCTION_QUALITIES[argmax(centroid, 91, 93) - 91] ?? 'medium',
        cutsPerSecond: Math.round((centroid[109] ?? 0) * 5 * 10) / 10,
    };
}
