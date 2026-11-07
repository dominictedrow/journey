CREATE TABLE `scan_snapshots` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`account_id` text NOT NULL,
	`scanned_at` integer NOT NULL,
	`avg_engagement_rate` real NOT NULL,
	`formats_count` integer NOT NULL,
	FOREIGN KEY (`account_id`) REFERENCES `accounts`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
PRAGMA foreign_keys=OFF;--> statement-breakpoint
CREATE TABLE `__new_feature_vectors` (
	`media_id` text PRIMARY KEY NOT NULL,
	`vector` blob NOT NULL,
	`dimension` integer DEFAULT 45 NOT NULL,
	FOREIGN KEY (`media_id`) REFERENCES `media`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
INSERT INTO `__new_feature_vectors`("media_id", "vector", "dimension") SELECT "media_id", "vector", "dimension" FROM `feature_vectors`;--> statement-breakpoint
DROP TABLE `feature_vectors`;--> statement-breakpoint
ALTER TABLE `__new_feature_vectors` RENAME TO `feature_vectors`;--> statement-breakpoint
PRAGMA foreign_keys=ON;--> statement-breakpoint
ALTER TABLE `video_analyses` ADD `scene_json` text;