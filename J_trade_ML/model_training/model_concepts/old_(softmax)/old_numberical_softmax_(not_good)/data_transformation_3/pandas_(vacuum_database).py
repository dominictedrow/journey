import pandas as pd
import sqlite3
from tqdm import tqdm

sqlite_path = 'C:\\Users\\crgon\\OneDrive\\Desktop\\thinkbe_app\\ai_side_projects\\trading\\stock_data.db'

conn = sqlite3.connect(sqlite_path)
c = conn.cursor()
c.execute("VACUUM")
conn.close()
