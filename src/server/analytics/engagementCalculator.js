export function computeEngagementRate(likes, comments, reposts, sends, plays) {
    if (plays === 0)
        return 0;
    return ((likes + comments + reposts + sends) / plays) * 100;
}
