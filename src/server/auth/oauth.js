import http from 'http';
import axios from 'axios';
import { tokenStore } from './tokenStore';
import { db } from '../db/index';
import { accounts, profiles } from '../db/schema';
import { logger } from '../logger';
const REDIRECT_PORT = 8788;
const REDIRECT_URI = `http://localhost:${REDIRECT_PORT}/callback`;
const IG_OAUTH_BASE = 'https://api.instagram.com';
const IG_GRAPH_BASE = 'https://graph.instagram.com/v21.0';
function appId() {
    return process.env['META_APP_ID'] ?? process.env['IG_APP_ID'] ?? '';
}
function appSecret() {
    return process.env['META_APP_SECRET'] ?? process.env['IG_APP_SECRET'] ?? '';
}
function buildAuthUrl() {
    const params = new URLSearchParams({
        client_id: appId(),
        redirect_uri: REDIRECT_URI,
        scope: 'instagram_business_basic,instagram_business_manage_insights',
        response_type: 'code',
    });
    return `https://www.instagram.com/oauth/authorize?${params}`;
}
function waitForCallback() {
    return new Promise((resolve, reject) => {
        const server = http.createServer((req, res) => {
            const url = new URL(req.url, `http://localhost:${REDIRECT_PORT}`);
            const code = url.searchParams.get('code');
            const error = url.searchParams.get('error_reason') ?? url.searchParams.get('error');
            res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
            const html = '<html><body style="font-family:sans-serif;text-align:center;padding:80px;background:#000;color:#fff">' +
                '<h2>Connected to Instana ✓</h2><p>You can close this tab.</p></body></html>';
            res.end(html, () => server.close());
            if (error)
                reject(new Error(String(error)));
            else if (code)
                resolve(code);
            else
                reject(new Error('No code in callback'));
        });
        server.listen(REDIRECT_PORT, () => {
            logger.info('OAuth callback server listening on port ' + REDIRECT_PORT);
        });
        server.on('error', reject);
        setTimeout(() => {
            server.close();
            reject(new Error('OAuth timeout after 5 minutes'));
        }, 5 * 60 * 1000);
    });
}
async function exchangeCode(code) {
    const formData = new URLSearchParams({
        client_id: appId(),
        client_secret: appSecret(),
        grant_type: 'authorization_code',
        redirect_uri: REDIRECT_URI,
        code,
    });
    const { data } = await axios.post(`${IG_OAUTH_BASE}/oauth/access_token`, formData.toString(), { headers: { 'Content-Type': 'application/x-www-form-urlencoded' } });
    return data.access_token;
}
async function exchangeForLongLived(shortToken) {
    const { data } = await axios.get(`${IG_GRAPH_BASE}/access_token`, {
        params: {
            grant_type: 'ig_exchange_token',
            client_id: appId(),
            client_secret: appSecret(),
            access_token: shortToken,
        },
    });
    return { token: data.access_token, expiresIn: data.expires_in };
}
async function getIGUserInfo(longToken) {
    const { data } = await axios.get(`${IG_GRAPH_BASE}/me`, { params: { fields: 'id,username', access_token: longToken } });
    return data;
}
export async function refreshTokenIfNeeded() {
    const token = await tokenStore.getToken();
    const expiry = await tokenStore.getTokenExpiry();
    if (!token || !expiry)
        return;
    const daysLeft = (expiry.getTime() - Date.now()) / (1000 * 60 * 60 * 24);
    if (daysLeft > 7)
        return;
    const { data } = await axios.get(`${IG_GRAPH_BASE}/refresh_access_token`, {
        params: {
            grant_type: 'ig_refresh_token',
            access_token: token,
        },
    });
    const userId = (await tokenStore.getUserId()) ?? '';
    await tokenStore.saveToken(data.access_token, new Date(Date.now() + data.expires_in * 1000), userId);
    logger.info('Instagram access token refreshed');
}
export async function connectWithToken(longToken) {
    const [profile] = db.select().from(profiles).limit(1).all();
    if (!profile)
        throw new Error('Create a profile before connecting Instagram');
    const igUser = await getIGUserInfo(longToken);
    const expiresAt = new Date(Date.now() + 60 * 24 * 60 * 60 * 1000);
    await tokenStore.saveToken(longToken, expiresAt, igUser.id);
    db.insert(accounts).values({
        id: igUser.id,
        profileId: profile.id,
        username: igUser.username,
        followersCount: igUser.followers_count ?? null,
    }).onConflictDoUpdate({
        target: accounts.id,
        set: {
            username: igUser.username,
            followersCount: igUser.followers_count ?? null,
        },
    }).run();
    logger.info(`Connected Instagram account via token: ${igUser.username}`);
    return { userId: igUser.id, username: igUser.username };
}
export async function connectInstagram() {
    const [profile] = db.select().from(profiles).limit(1).all();
    if (!profile)
        throw new Error('Create a profile before connecting Instagram');
    const authUrl = buildAuthUrl();
    const callbackPromise = waitForCallback();
    const { shell } = await import('electron');
    await shell.openExternal(authUrl);
    const code = await callbackPromise;
    const shortToken = await exchangeCode(code);
    const { token: longToken, expiresIn } = await exchangeForLongLived(shortToken);
    const igUser = await getIGUserInfo(longToken);
    await tokenStore.saveToken(longToken, new Date(Date.now() + expiresIn * 1000), igUser.id);
    db.insert(accounts).values({
        id: igUser.id,
        profileId: profile.id,
        username: igUser.username,
        followersCount: igUser.followers_count ?? null,
    }).onConflictDoUpdate({
        target: accounts.id,
        set: {
            username: igUser.username,
            followersCount: igUser.followers_count ?? null,
        },
    }).run();
    logger.info(`Connected Instagram account: ${igUser.username}`);
    return { userId: igUser.id, username: igUser.username };
}
export async function disconnectInstagram() {
    await tokenStore.deleteToken();
    logger.info('Instagram account disconnected');
}
