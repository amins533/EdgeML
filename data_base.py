from pathlib import Path
import sqlite3
import threading
import queue
from typing import List

from data_class import GameData, PlayData, PlayerData, TrackingPoint, RouteData

BASE_DIR = Path(__file__).resolve().parent
DATABASE_DIR = BASE_DIR / "data-base"
DATABASE_NAME = DATABASE_DIR / "football_analysis.db"

class DatabasePipeline:
    def __init__(self, max_queue_size=128):
        self.stopped = False
        self.Q = queue.Queue(maxsize=max_queue_size)

        DATABASE_DIR.mkdir(parents=True, exist_ok=True)
        self.create_database()

    def create_database(self):
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS games (
            game_id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            video TEXT
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS plays (
            play_id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id INTEGER,
            timestamp TEXT,
            formation TEXT,
            FOREIGN KEY(game_id) REFERENCES games(game_id)
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS players (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tracking_id INTEGER,
            jersey_number INTEGER,
            player_role TEXT,
            team_side TEXT
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS player_tracking (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tracking_id INTEGER,
            play_id INTEGER,
            frame_id INTEGER,
            x_coord REAL,
            y_coord REAl,
            FOREIGN KEY(play_id) REFERENCES plays(play_id),
            FOREIGN KEY(tracking_id) REFERENCES players(tracking_id)
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS routes (
            route_id INTEGER PRIMARY KEY AUTOINCREMENT,
            tracking_id INTEGER,
            play_id INTEGER,
            frame_id INTEGER,
            route_type TEXT,
            FOREIGN KEY(play_id) REFERENCES plays(play_id),
            FOREIGN KEY(tracking_id) REFERENCES players(tracking_id)
        )
        """)

        conn.commit()
        conn.close()

        print("Database and tables created!")

    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        print("DB thread started")
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()

        while True:
            if self.stopped and self.Q.empty():
                break

            try:
                action, data = self.Q.get(timeout=0.1)


                if action == "INSERT_GAME":
                    cursor.execute("""
                        INSERT INTO games (date, video) VALUES (?,?)
                        """, (data.date, data.video))

                elif action == "INSERT_PLAY":
                    cursor.execute("""
                        INSERT INTO plays (game_id, timestamp, formation) VALUES (?,?,?)
                        """, (data.game_id, data.timestamp,data.formation))

                elif action == "INSERT_PLAYER":
                    cursor.execute("""
                        INSERT INTO players (tracking_id, jersey_number, player_role, team_side) VALUES (?,?,?,?)
                        """, (data.tracking_id, data.jersey_number, data.player_role, data.team_side))

                elif action == "INSERT_ROUTE":
                    cursor.execute("""
                        INSERT INTO routes (play_id, tracking_id, route_type) VALUES (?,?,?)
                        """, (data.play_id, data.tracking_id,data.route_type))

                elif action == "BATCH_TRACKING":
                    tracking_tuples = [pt.to_tuple() for pt in data]

                    cursor.executemany("""
                        INSERT INTO player_tracking (play_id, tracking_id, frame_id, x_coord, y_coord) VALUES(?,?,?,?,?)
                        """, tracking_tuples)
          

                conn.commit()

            except queue.Empty:
                continue

        conn.close()

    def save_game(self, game: GameData):
        if not self.Q.full(): self.Q.put(("INSERT_GAME", game))
    
    def save_play(self, play:PlayData):
        if not self.Q.full():self.Q.put(("INSERT_PLAY", play))
    
    def save_player(self, player: PlayerData):
        if not self.Q.full(): self.Q.put(("INSERT_PLAYER", player))

    def save_route(self, route: RouteData):
        if not self.Q.full(): self.Q.put(("INSERT_ROUTE", route))
    def save_tracking_batch(self, tracking_points: List[TrackingPoint]):
        if not self.Q.full() and tracking_points:
            self.Q.put(("BATCH_TRACKING", tracking_points))

    def stop(self):
        self.stopped = True