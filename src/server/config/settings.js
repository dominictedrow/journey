export const settings = {
    frames: {
        totalExtracted: 20,
        sentToGrok: {
            long: 16,
            medium: 10,
            short: 6,
        },
    },
    cron: {
        defaultSchedule: '0 9 * * *',
    },
    grok: {
        visionModel: 'grok-4.3',
        textModel: 'grok-3-mini',
        temperature: 0,
        maxTokens: 1500,
    },
};
