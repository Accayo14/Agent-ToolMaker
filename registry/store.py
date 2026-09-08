import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path(__file__).parent / "tools.db"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tools (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                repo_url    TEXT NOT NULL,
                code        TEXT,
                test_code   TEXT,
                spec_json   TEXT,
                iterations  INTEGER,
                status      TEXT,
                last_error  TEXT,
                created_at  TEXT
            )
        """)


def upsert_tool(job_id: str, name: str, repo_url: str, spec: dict, status: str,
                code: str = "", test_code: str = "", iterations: int = 0, last_error: str = ""):
    with _conn() as conn:
        conn.execute("""
            INSERT INTO tools (id, name, repo_url, code, test_code, spec_json, iterations, status, last_error, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                code=excluded.code, test_code=excluded.test_code,
                iterations=excluded.iterations, status=excluded.status,
                last_error=excluded.last_error
        """, (job_id, name, repo_url, code, test_code, json.dumps(spec),
              iterations, status, last_error, datetime.now(timezone.utc).isoformat()))


def get_tool(job_id: str) -> dict | None:
    with _conn() as conn:
        row = conn.execute("SELECT * FROM tools WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None


def list_tools(status: str = "success") -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM tools WHERE status = ? ORDER BY created_at DESC", (status,)
        ).fetchall()
        return [dict(r) for r in rows]


def export_markdown() -> Path:
    """
    Regenerate tools_registry.md — human-readable summary of every run.
    Opens cleanly in VSCode. Called automatically after every save.
    """
    registry_path = Path(__file__).parent.parent / "tools_registry.md"

    with _conn() as conn:
        all_rows = conn.execute(
            "SELECT * FROM tools ORDER BY created_at DESC"
        ).fetchall()

    tools = [dict(r) for r in all_rows]
    successful = [t for t in tools if t["status"] == "success"]
    failed     = [t for t in tools if t["status"] == "failed"]

    lines = [
        "# Tools Registry",
        "",
        f"> Auto-generated — do not edit by hand. Last updated: "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Summary",
        "",
        f"| | Count |",
        f"|---|---|",
        f"| Total runs | {len(tools)} |",
        f"| Successful | {len(successful)} |",
        f"| Failed | {len(failed)} |",
        "",
    ]

    if successful:
        lines += ["## Successful Tools", ""]
        for t in successful:
            status_icon = "✅"
            lines += [
                f"### {status_icon} `{t['name']}`",
                "",
                f"| Field | Value |",
                f"|---|---|",
                f"| Repo | [{t['repo_url']}]({t['repo_url']}) |",
                f"| Iterations | {t['iterations']} |",
                f"| Created | {t['created_at'][:19].replace('T', ' ')} UTC |",
                f"| ID | `{t['id'][:8]}...` |",
                "",
            ]
            if t.get("code"):
                lines += [
                    "**Wrapper code:**",
                    "```python",
                    t["code"],
                    "```",
                    "",
                ]
            if t.get("test_code"):
                lines += [
                    "**Tests:**",
                    "```python",
                    t["test_code"],
                    "```",
                    "",
                ]
            lines.append("---")
            lines.append("")

    if failed:
        lines += ["## Failed Runs", ""]
        for t in failed:
            lines += [
                f"### ❌ `{t['name']}`",
                "",
                f"| Field | Value |",
                f"|---|---|",
                f"| Repo | {t['repo_url']} |",
                f"| Iterations attempted | {t['iterations']} |",
                f"| Created | {t['created_at'][:19].replace('T', ' ')} UTC |",
                "",
            ]
            if t.get("last_error"):
                lines += [
                    "**Last error:**",
                    "```",
                    t["last_error"][:500],
                    "```",
                    "",
                ]
            lines.append("---")
            lines.append("")

    registry_path.write_text("\n".join(lines))
    return registry_path
