import Database from 'better-sqlite3'
import { drizzle } from 'drizzle-orm/better-sqlite3'
import { migrate } from 'drizzle-orm/better-sqlite3/migrator'
import * as schema from './schema'
import fs from 'fs'
import path from 'path'

const userData = process.env['INSTANA_USER_DATA']!
const dbPath = path.join(userData, 'instana.db')
if (!fs.existsSync(userData)) fs.mkdirSync(userData, { recursive: true })

const sqlite = new Database(dbPath)
sqlite.pragma('journal_mode = WAL')
sqlite.pragma('foreign_keys = ON')

export const db = drizzle(sqlite, { schema })

const migrationsFolder = path.join(process.env['INSTANA_APP_PATH']!, 'drizzle')
if (fs.existsSync(migrationsFolder)) {
  migrate(db, { migrationsFolder })
}
