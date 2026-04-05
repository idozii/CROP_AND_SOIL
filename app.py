import os
import sqlite3
from functools import wraps
from pathlib import Path
from typing import Any, Optional

from flask import Flask, flash, g, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except Exception:  # pragma: no cover - optional dependency during local setup
    psycopg2 = None
    RealDictCursor = None

from notebook_bridge import NotebookLoadError, load_prediction_functions


BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "instance" / "crop_and_soil.db"
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-change-me-in-production")


class DatabaseAdapter:
    """Unifies query APIs across SQLite and PostgreSQL connections."""

    def __init__(self, connection: Any, backend: str) -> None:
        self.connection = connection
        self.backend = backend

    def _format_query(self, query: str) -> str:
        if self.backend == "postgres":
            return query.replace("?", "%s")
        return query

    def execute(self, query: str, params: tuple = ()):
        formatted = self._format_query(query)
        if self.backend == "postgres":
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(formatted, params)
                return cursor

        cursor = self.connection.cursor()
        cursor.execute(formatted, params)
        return cursor

    def executescript(self, script: str) -> None:
        if self.backend == "postgres":
            with self.connection.cursor() as cursor:
                cursor.execute(script)
            return

        self.connection.executescript(script)

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


def init_prediction_engine() -> None:
    try:
        engine = load_prediction_functions(BASE_DIR / "main.ipynb")
    except NotebookLoadError as exc:
        raise RuntimeError(f"Failed to load prediction functions from notebook: {exc}") from exc

    app.config["PREDICT_CROP"] = engine.predict_crop
    app.config["PREDICT_FERTILIZER"] = engine.predict_fertilizer


def get_db() -> DatabaseAdapter:
    if "db" not in g:
        if DATABASE_URL:
            if psycopg2 is None:
                raise RuntimeError("DATABASE_URL is set but psycopg2 is not installed")
            connection = psycopg2.connect(DATABASE_URL)
            g.db = DatabaseAdapter(connection, "postgres")
        else:
            DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(str(DATABASE_PATH))
            connection.row_factory = sqlite3.Row
            g.db = DatabaseAdapter(connection, "sqlite")
    return g.db


@app.teardown_appcontext
def close_db(_error: Optional[Exception] = None) -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db() -> None:
    db = get_db()
    if db.backend == "postgres":
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                full_name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL,
                soil_type TEXT NOT NULL,
                temperature DOUBLE PRECISION NOT NULL,
                humidity DOUBLE PRECISION NOT NULL,
                moisture DOUBLE PRECISION NOT NULL,
                nitrogen DOUBLE PRECISION NOT NULL,
                potassium DOUBLE PRECISION NOT NULL,
                phosphorus DOUBLE PRECISION NOT NULL,
                crop_prediction TEXT NOT NULL,
                crop_confidence DOUBLE PRECISION NOT NULL,
                fertilizer_prediction TEXT NOT NULL,
                fertilizer_confidence DOUBLE PRECISION NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            );
            """
        )
    else:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                soil_type TEXT NOT NULL,
                temperature REAL NOT NULL,
                humidity REAL NOT NULL,
                moisture REAL NOT NULL,
                nitrogen REAL NOT NULL,
                potassium REAL NOT NULL,
                phosphorus REAL NOT NULL,
                crop_prediction TEXT NOT NULL,
                crop_confidence REAL NOT NULL,
                fertilizer_prediction TEXT NOT NULL,
                fertilizer_confidence REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            );
            """
        )
    db.commit()


def login_required(view_func):
    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    return wrapped_view


@app.before_request
def load_logged_in_user() -> None:
    user_id = session.get("user_id")
    if not user_id:
        g.user = None
        return

    db = get_db()
    user_cursor = db.execute(
        "SELECT id, full_name, email, created_at FROM users WHERE id = ?",
        (user_id,),
    )
    g.user = user_cursor.fetchone()


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        full_name = (request.form.get("full_name") or "").strip()
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""

        if len(full_name) < 2:
            flash("Full name must be at least 2 characters.", "danger")
            return render_template("signup.html")
        if "@" not in email or len(email) < 5:
            flash("Please provide a valid email address.", "danger")
            return render_template("signup.html")
        if len(password) < 8:
            flash("Password must be at least 8 characters.", "danger")
            return render_template("signup.html")

        db = get_db()
        exists_cursor = db.execute("SELECT id FROM users WHERE email = ?", (email,))
        exists = exists_cursor.fetchone()
        if exists:
            flash("An account with this email already exists.", "danger")
            return render_template("signup.html")

        db.execute(
            "INSERT INTO users (full_name, email, password_hash) VALUES (?, ?, ?)",
            (full_name, email, generate_password_hash(password)),
        )
        db.commit()
        flash("Account created. Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""

        db = get_db()
        user_cursor = db.execute(
            "SELECT id, full_name, email, password_hash FROM users WHERE email = ?",
            (email,),
        )
        user = user_cursor.fetchone()

        if user is None or not check_password_hash(user["password_hash"], password):
            flash("Invalid email or password.", "danger")
            return render_template("login.html")

        session.clear()
        session["user_id"] = user["id"]
        flash(f"Welcome back, {user['full_name']}.", "success")
        return redirect(url_for("home"))

    return render_template("login.html")


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


@app.route("/")
@login_required
def home():
    db = get_db()
    recent_cursor = db.execute(
        """
        SELECT crop_prediction, fertilizer_prediction, soil_type, created_at
        FROM predictions
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT 5
        """,
        (session["user_id"],),
    )
    recent_predictions = recent_cursor.fetchall()
    return render_template("home.html", recent_predictions=recent_predictions)


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    try:
        data = {
            "soil_type": (request.form.get("soil_type") or "").strip(),
            "temperature": float(request.form.get("temperature") or 0),
            "humidity": float(request.form.get("humidity") or 0),
            "moisture": float(request.form.get("moisture") or 0),
            "nitrogen": float(request.form.get("nitrogen") or 0),
            "potassium": float(request.form.get("potassium") or 0),
            "phosphorus": float(request.form.get("phosphorus") or 0),
        }

        crop_result = app.config["PREDICT_CROP"](
            data["soil_type"],
            data["temperature"],
            data["humidity"],
            data["moisture"],
            data["nitrogen"],
            data["potassium"],
            data["phosphorus"],
        )
        fertilizer_result = app.config["PREDICT_FERTILIZER"](
            data["soil_type"],
            data["temperature"],
            data["humidity"],
            data["moisture"],
            data["nitrogen"],
            data["potassium"],
            data["phosphorus"],
        )

        db = get_db()
        db.execute(
            """
            INSERT INTO predictions (
                user_id, soil_type, temperature, humidity, moisture,
                nitrogen, potassium, phosphorus,
                crop_prediction, crop_confidence,
                fertilizer_prediction, fertilizer_confidence
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session["user_id"],
                data["soil_type"],
                data["temperature"],
                data["humidity"],
                data["moisture"],
                data["nitrogen"],
                data["potassium"],
                data["phosphorus"],
                crop_result.get("prediction", "Unknown"),
                float(crop_result.get("confidence", 0)),
                fertilizer_result.get("prediction", "Unknown"),
                float(fertilizer_result.get("confidence", 0)),
            ),
        )
        db.commit()

        return render_template(
            "results.html",
            crop=crop_result,
            fertilizer=fertilizer_result,
            input_data=data,
        )
    except Exception as e:
        return render_template("error.html", error=str(e))


with app.app_context():
    init_db()
    init_prediction_engine()

app_handler = app

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
