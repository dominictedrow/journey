CREATE TABLE `accounts` (
	`id` text PRIMARY KEY NOT NULL,
	`profile_id` integer NOT NULL,
	`username` text NOT NULL,
	`section_id` integer,
	`followers_count` integer,
	`last_sync_at` integer,
	FOREIGN KEY (`profile_id`) REFERENCES `profiles`(`id`) ON UPDATE no action ON DELETE no action,
	FOREIGN KEY (`section_id`) REFERENCES `sections`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
CREATE TABLE `feature_vectors` (
	`media_id` text PRIMARY KEY NOT NULL,
	`vector` blob NOT NULL,
	`dimension` integer DEFAULT 32 NOT NULL,
	FOREIGN KEY (`media_id`) REFERENCES `media`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
CREATE TABLE `format_memberships` (
	`media_id` text NOT NULL,
	`format_id` integer NOT NULL,
	`distance` real,
	FOREIGN KEY (`media_id`) REFERENCES `media`(`id`) ON UPDATE no action ON DELETE no action,
	FOREIGN KEY (`format_id`) REFERENCES `formats`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
CREATE TABLE `formats` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`account_id` text NOT NULL,
	`label` text NOT NULL,
	`centroid_json` text NOT NULL,
	`video_count` integer DEFAULT 0 NOT NULL,
	`avg_engagement` real,
	`top_media_id` text,
	`bottom_media_id` text,
	`created_at` integer NOT NULL,
	FOREIGN KEY (`account_id`) REFERENCES `accounts`(`id`) ON UPDATE no action ON DELETE no action,
	FOREIGN KEY (`top_media_id`) REFERENCES `media`(`id`) ON UPDATE no action ON DELETE no action,
	FOREIGN KEY (`bottom_media_id`) REFERENCES `media`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
CREATE TABLE `insights` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`account_id` text NOT NULL,
	`best_format_id` integer,
	`worst_format_id` integer,
	`report_json` text NOT NULL,
	`generated_at` integer NOT NULL,
	FOREIGN KEY (`account_id`) REFERENCES `accounts`(`id`) ON UPDATE no action ON DELETE no action,
	FOREIGN KEY (`best_format_id`) REFERENCES `formats`(`id`) ON UPDATE no action ON DELETE no action,
	FOREIGN KEY (`worst_format_id`) REFERENCES `formats`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
CREATE TABLE `media` (
	`id` text PRIMARY KEY NOT NULL,
	`account_id` text NOT NULL,
	`media_type` text NOT NULL,
	`media_url` text,
	`thumbnail_url` text,
	`caption` text,
	`timestamp` integer,
	FOREIGN KEY (`account_id`) REFERENCES `accounts`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
CREATE TABLE `media_insights` (
	`media_id` text PRIMARY KEY NOT NULL,
	`views` integer,
	`plays` integer,
	`likes` integer,
	`comments` integer,
	`shares` integer,
	`saves` integer,
	`engagement_rate` real,
	FOREIGN KEY (`media_id`) REFERENCES `media`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
CREATE TABLE `profiles` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`username` text NOT NULL,
	`pfp_url` text,
	`created_at` integer NOT NULL
);
--> statement-breakpoint
CREATE TABLE `sections` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`name` text NOT NULL,
	`color` text NOT NULL,
	`ig_profile_url` text NOT NULL,
	`account_id` text,
	`created_at` integer NOT NULL
);
--> statement-breakpoint
CREATE TABLE `sync_jobs` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`account_id` text NOT NULL,
	`triggered_by` text NOT NULL,
	`status` text DEFAULT 'pending' NOT NULL,
	`media_fetched` integer DEFAULT 0,
	`analyzed` integer DEFAULT 0,
	`started_at` integer,
	`finished_at` integer,
	FOREIGN KEY (`account_id`) REFERENCES `accounts`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
CREATE TABLE `video_analyses` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`media_id` text NOT NULL,
	`detected_text` text,
	`cuts_per_second` real,
	`footage_type` text,
	`background_type` text,
	`text_style_json` text,
	`audio_type` text,
	`hook_strength` text,
	`production_quality` text,
	`grok_raw_json` text,
	`status` text DEFAULT 'pending' NOT NULL,
	FOREIGN KEY (`media_id`) REFERENCES `media`(`id`) ON UPDATE no action ON DELETE no action
);
