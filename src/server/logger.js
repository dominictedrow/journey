import winston from 'winston';
import path from 'path';
import fs from 'fs';
const logsDir = path.join(process.env['INSTANA_USER_DATA'], 'logs');
if (!fs.existsSync(logsDir))
    fs.mkdirSync(logsDir, { recursive: true });
export const logger = winston.createLogger({
    level: 'info',
    format: winston.format.combine(winston.format.timestamp(), winston.format.json()),
    transports: [
        new winston.transports.Console({
            format: winston.format.combine(winston.format.colorize(), winston.format.simple()),
        }),
        new winston.transports.File({
            filename: path.join(logsDir, 'instana.log'),
        }),
    ],
});
