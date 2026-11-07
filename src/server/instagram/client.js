import axios from 'axios';
import { tokenStore } from '../auth/tokenStore';
import { refreshTokenIfNeeded } from '../auth/oauth';
import { logger } from '../logger';
const IG_BASE = 'https://graph.instagram.com/v21.0';
function createIGClient() {
    const client = axios.create({ baseURL: IG_BASE });
    client.interceptors.request.use(async (config) => {
        const token = await tokenStore.getToken();
        if (token)
            config.params = { ...config.params, access_token: token };
        return config;
    });
    client.interceptors.response.use((res) => res, async (error) => {
        if (!axios.isAxiosError(error))
            throw error;
        const status = error.response?.status;
        if (status === 401 && error.config) {
            await refreshTokenIfNeeded();
            return client(error.config);
        }
        if (status === 429 && error.config) {
            const retryAfter = parseInt(error.response?.headers['retry-after'] ?? '60') * 1000;
            logger.warn(`IG rate limited — waiting ${retryAfter}ms`);
            await new Promise((r) => setTimeout(r, retryAfter));
            return client(error.config);
        }
        throw error;
    });
    return client;
}
export const igClient = createIGClient();
