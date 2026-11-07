import { sqliteTable, text, integer, real, blob } from 'drizzle-orm/sqlite-core'

export const profiles = sqliteTable('profiles', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  username: text('username').notNull(),
  pfpUrl: text('pfp_url'),
  createdAt: integer('created_at', { mode: 'timestamp' }).notNull().$defaultFn(() => new Date()),
})

export const sections = sqliteTable('sections', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  name: text('name').notNull(),
  color: text('color').notNull(),
  igProfileUrl: text('ig_profile_url').notNull(),
  accountId: text('account_id'),
  createdAt: integer('created_at', { mode: 'timestamp' }).notNull().$defaultFn(() => new Date()),
})

export const accounts = sqliteTable('accounts', {
  id: text('id').primaryKey(),
  profileId: integer('profile_id').notNull().references(() => profiles.id),
  username: text('username').notNull(),
  sectionId: integer('section_id').references(() => sections.id),
  followersCount: integer('followers_count'),
  lastSyncAt: integer('last_sync_at', { mode: 'timestamp' }),
})

export const media = sqliteTable('media', {
  id: text('id').primaryKey(),
  accountId: text('account_id').notNull().references(() => accounts.id),
  mediaType: text('media_type').notNull(),
  mediaUrl: text('media_url'),
  thumbnailUrl: text('thumbnail_url'),
  caption: text('caption'),
  timestamp: integer('timestamp', { mode: 'timestamp' }),
})

export const mediaInsights = sqliteTable('media_insights', {
  mediaId: text('media_id').primaryKey().references(() => media.id),
  views: integer('views'),
  plays: integer('plays'),
  likes: integer('likes'),
  comments: integer('comments'),
  reposts: integer('reposts'),
  sends: integer('sends'),
  engagementRate: real('engagement_rate'),
})

export const videoAnalyses = sqliteTable('video_analyses', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  mediaId: text('media_id').notNull().references(() => media.id),
  detectedText: text('detected_text'),
  cutsPerSecond: real('cuts_per_second'),
  footageType: text('footage_type'),
  backgroundType: text('background_type'),
  sceneJson: text('scene_json'),
  textStyleJson: text('text_style_json'),
  audioType: text('audio_type'),
  hookStrength: text('hook_strength'),
  productionQuality: text('production_quality'),
  grokRawJson: text('grok_raw_json'),
  status: text('status').notNull().default('pending'),
})

export const featureVectors = sqliteTable('feature_vectors', {
  mediaId: text('media_id').primaryKey().references(() => media.id),
  vector: blob('vector').notNull(),
  dimension: integer('dimension').notNull().default(111),
})

export const formats = sqliteTable('formats', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  accountId: text('account_id').notNull().references(() => accounts.id),
  label: text('label').notNull(),
  centroidJson: text('centroid_json').notNull(),
  videoCount: integer('video_count').notNull().default(0),
  avgEngagement: real('avg_engagement'),
  topMediaId: text('top_media_id').references(() => media.id),
  bottomMediaId: text('bottom_media_id').references(() => media.id),
  createdAt: integer('created_at', { mode: 'timestamp' }).notNull().$defaultFn(() => new Date()),
})

export const formatMemberships = sqliteTable('format_memberships', {
  mediaId: text('media_id').notNull().references(() => media.id),
  formatId: integer('format_id').notNull().references(() => formats.id),
  distance: real('distance'),
})

export const insights = sqliteTable('insights', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  accountId: text('account_id').notNull().references(() => accounts.id),
  bestFormatId: integer('best_format_id').references(() => formats.id),
  worstFormatId: integer('worst_format_id').references(() => formats.id),
  reportJson: text('report_json').notNull(),
  generatedAt: integer('generated_at', { mode: 'timestamp' }).notNull().$defaultFn(() => new Date()),
})

export const syncJobs = sqliteTable('sync_jobs', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  accountId: text('account_id').notNull().references(() => accounts.id),
  triggeredBy: text('triggered_by').notNull(),
  status: text('status').notNull().default('pending'),
  mediaFetched: integer('media_fetched').default(0),
  analyzed: integer('analyzed').default(0),
  startedAt: integer('started_at', { mode: 'timestamp' }),
  finishedAt: integer('finished_at', { mode: 'timestamp' }),
})

export const scanSnapshots = sqliteTable('scan_snapshots', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  accountId: text('account_id').notNull().references(() => accounts.id),
  scannedAt: integer('scanned_at', { mode: 'timestamp' }).notNull().$defaultFn(() => new Date()),
  avgEngagementRate: real('avg_engagement_rate').notNull(),
  formatsCount: integer('formats_count').notNull(),
})
