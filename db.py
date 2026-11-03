"""Per-user SQLite access and PBKDF2 password hashing. No plaintext passwords."""
import sqlite3,hashlib,hmac,secrets,uuid,shutil,csv
from pathlib import Path
from datetime import datetime,timezone
from PIL import Image,ImageOps
from config import ROOT,CLASSES
ITERATIONS=600000
class Database:
    def __init__(self,path=None):
        self.path=Path(path or ROOT/'app_data'/'insects.sqlite3');self.path.parent.mkdir(parents=True,exist_ok=True)
        self.records=self.path.parent/'records';self.records.mkdir(exist_ok=True)
        with self.connect() as c:
            c.executescript('''CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY,username TEXT UNIQUE NOT NULL,salt TEXT NOT NULL,password_hash TEXT NOT NULL,created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS image_records(id INTEGER PRIMARY KEY,user_id INTEGER NOT NULL REFERENCES users(id),image_path TEXT NOT NULL,predicted_label TEXT NOT NULL,confidence REAL NOT NULL,true_label TEXT,created_at TEXT NOT NULL);''')
    def connect(self):
        c=sqlite3.connect(self.path);c.row_factory=sqlite3.Row;c.execute('PRAGMA foreign_keys=ON');return c
    def register(self,username,password):
        username=username.strip()
        if not 3<=len(username)<=40:raise ValueError('Username must contain 3-40 characters.')
        if len(password)<8:raise ValueError('Password must contain at least 8 characters.')
        salt=secrets.token_hex(16);hashed=hashlib.pbkdf2_hmac('sha256',password.encode(),bytes.fromhex(salt),ITERATIONS).hex()
        try:
            with self.connect() as c:c.execute('INSERT INTO users(username,salt,password_hash,created_at) VALUES (?,?,?,?)',(username,salt,hashed,datetime.now(timezone.utc).isoformat()))
        except sqlite3.IntegrityError:raise ValueError('Username already exists.') from None
    def login(self,username,password):
        with self.connect() as c:row=c.execute('SELECT * FROM users WHERE username=?',(username.strip(),)).fetchone()
        if row:
            hashed=hashlib.pbkdf2_hmac('sha256',password.encode(),bytes.fromhex(row['salt']),ITERATIONS).hex()
            if hmac.compare_digest(hashed,row['password_hash']):return {'id':row['id'],'username':row['username']}
        raise ValueError('Wrong username or password.')
    def save(self,user_id,path,label,confidence,true_label=None):
        if label not in CLASSES or (true_label and true_label not in CLASSES):raise ValueError('Invalid class label.')
        if not 0<=float(confidence)<=1:raise ValueError('Confidence must be between 0 and 1.')
        dest=self.records/(uuid.uuid4().hex+'.png')
        with Image.open(path) as im:ImageOps.exif_transpose(im).convert('RGB').save(dest)
        try:
            with self.connect() as c:
                cursor=c.execute('INSERT INTO image_records(user_id,image_path,predicted_label,confidence,true_label,created_at) VALUES (?,?,?,?,?,?)',(user_id,str(dest),label,float(confidence),true_label or None,datetime.now(timezone.utc).isoformat()))
                return cursor.lastrowid
        except Exception:dest.unlink(missing_ok=True);raise
    def list_records(self,user_id):
        with self.connect() as c:return [dict(r) for r in c.execute('SELECT * FROM image_records WHERE user_id=? ORDER BY id DESC',(user_id,))]
    def delete(self,user_id,record_id):
        with self.connect() as c:
            row=c.execute('SELECT image_path FROM image_records WHERE id=? AND user_id=?',(record_id,user_id)).fetchone()
            if not row:raise ValueError('Record not found for this user.')
            c.execute('DELETE FROM image_records WHERE id=? AND user_id=?',(record_id,user_id))
        path=Path(row['image_path'])
        if path.resolve().parent==self.records.resolve():path.unlink(missing_ok=True)
    def export(self,user_id,path):
        rows=self.list_records(user_id)
        with open(path,'w',newline='',encoding='utf-8') as f:
            fields=['id','predicted_label','confidence','true_label','created_at']
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
