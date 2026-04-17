"""
cli.py

CPT command-line interface — rich terminal UI for the Crypto Price Terminal.
Works standalone (hits local DB directly) or against a running server via HTTP.

Usage:
    python cli.py status
    python cli.py predict SOL
    python cli.py predict DOGE
    python cli.py history SOL 10
    python cli.py notify enable discord
    python cli.py notify disable whatsapp
    python cli.py run [--reload]
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

app = typer.Typer(
    name="cpt",
    help="CPT — Crypto Price Terminal CLI",
    add_completion=False,
    pretty_exceptions_enable=False,
)
console = Console()

_SERVER = "http://localhost:8000"
_SOL_COLOR = "cyan"
_DOGE_COLOR = "yellow"
_UP_COLOR = "bright_green"
_DOWN_COLOR = "bright_red"
_FLAT_COLOR = "bright_black"


# ─── Helpers ──────────────────────────────────────────────────────


def _coin_color(coin: str) -> str:
    return _SOL_COLOR if coin.upper() == "SOL" else _DOGE_COLOR


def _dir_style(direction: str) -> tuple[str, str]:
    d = (direction or "neutral").lower()
    if d in ("bullish", "up"):
        return "↑", _UP_COLOR
    if d in ("bearish", "down"):
        return "↓", _DOWN_COLOR
    return "—", _FLAT_COLOR


def _pct_str(v: Optional[float]) -> str:
    if v is None:
        return "[bright_black]---[/]"
    color = _UP_COLOR if v > 0.1 else (_DOWN_COLOR if v < -0.1 else _FLAT_COLOR)
    sign = "+" if v > 0 else ""
    return f"[{color}]{sign}{v:.2f}%[/]"


def _fmt_price(v: Optional[float], coin: str) -> str:
    if v is None:
        return "---"
    return f"${v:.6f}" if coin.upper() == "DOGE" else f"${v:,.4f}"


def _conf_bar(v: float) -> str:
    pct = int(v * 100)
    filled = int(20 * v)
    bar = "█" * filled + "░" * (20 - filled)
    color = _UP_COLOR if v >= 0.7 else ("yellow" if v >= 0.5 else _FLAT_COLOR)
    return f"[{color}]{bar}[/] [{color}]{pct}%[/]"


def _fetch(path: str) -> Optional[dict | list]:
    try:
        import httpx

        r = httpx.get(f"{_SERVER}{path}", timeout=5.0)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        console.print(f"[red]Server unreachable:[/] {exc}")
        return None


def _latest_from_db(coin: str) -> Optional[dict]:
    try:
        from storage.database import SessionLocal
        from storage import prediction_repository

        db = SessionLocal()
        pred = prediction_repository.get_latest(db, coin.upper())
        db.close()
        if pred is None:
            return None
        return {
            "coin": pred.coin,
            "timestamp": str(pred.timestamp),
            "direction": pred.direction,
            "magnitude_pct": pred.magnitude_pct,
            "confidence": pred.confidence,
            "target_24h": pred.target_24h,
            "target_72h": pred.target_72h,
            "target_7d": pred.target_7d,
        }
    except Exception as exc:
        console.print(f"[red]DB read failed:[/] {exc}")
        return None


def _history_from_db(coin: str, limit: int) -> list[dict]:
    try:
        from storage.database import SessionLocal
        from storage import prediction_repository

        db = SessionLocal()
        rows = prediction_repository.get_history(db, coin.upper(), limit=limit)
        db.close()
        return [
            {
                "coin": r.coin,
                "timestamp": str(r.timestamp),
                "direction": r.direction,
                "magnitude_pct": r.magnitude_pct,
                "confidence": r.confidence,
                "target_24h": r.target_24h,
                "target_72h": r.target_72h,
                "target_7d": r.target_7d,
            }
            for r in rows
        ]
    except Exception as exc:
        console.print(f"[red]DB read failed:[/] {exc}")
        return []


# ─── Commands ─────────────────────────────────────────────────────


@app.command()
def status() -> None:
    """Show server health and engine status."""
    data = _fetch("/api/status")

    if data is None:
        console.print()
        console.print(
            Panel(
                "[red]Server is not running.[/]\n\n" "Start with:  [cyan]python cli.py run[/]",
                title="[bold red]SERVER OFFLINE[/]",
                border_style="red",
            )
        )
        console.print()
        return

    uptime = int(data.get("uptime_seconds", 0))
    h, rem = divmod(uptime, 3600)
    m, s = divmod(rem, 60)

    console.print()
    console.print(
        Panel(
            f"[bright_black]Status :[/]  [bright_green]ONLINE[/]\n"
            f"[bright_black]Uptime :[/]  [cyan]{h:02d}:{m:02d}:{s:02d}[/]\n"
            f"[bright_black]Started:[/]  [bright_black]{data.get('started_at', '—')}[/]\n"
            f"[bright_black]Dash   :[/]  [cyan underline]http://localhost:8000[/]",
            title="[bold cyan]◈  CPT SERVER[/]",
            border_style="cyan",
        )
    )

    engines = data.get("engines", {})
    tbl = Table(
        box=box.SIMPLE_HEAVY,
        header_style="bold bright_black",
        border_style="bright_black",
        title="[bold]Engine Status[/]",
        expand=True,
    )
    tbl.add_column("ENGINE", style="bold", min_width=12)
    tbl.add_column("STATUS", min_width=10)
    tbl.add_column("LAST FETCH", min_width=22)
    tbl.add_column("DETAIL", style="bright_black")

    style_map = {"ok": _UP_COLOR, "error": _DOWN_COLOR, "idle": _FLAT_COLOR}
    dot_map = {"ok": "●", "error": "✗", "idle": "○"}

    for name, info in engines.items():
        st = info.get("status", "idle")
        color = style_map.get(st, "white")
        tbl.add_row(
            name.upper(),
            f"[{color}]{dot_map.get(st, '?')} {st.upper()}[/]",
            info.get("last_fetch") or "—",
            info.get("detail", ""),
        )

    console.print(tbl)
    console.print()


@app.command()
def predict(coin: str = typer.Argument(..., help="SOL or DOGE")) -> None:
    """Show the latest cached prediction for a coin."""
    coin = coin.upper()
    if coin not in ("SOL", "DOGE"):
        console.print("[red]Coin must be SOL or DOGE[/]")
        raise typer.Exit(1)

    data = _fetch(f"/api/predictions/{coin}") or _latest_from_db(coin)
    color = _coin_color(coin)

    if data is None:
        console.print(f"\n[{color}]{coin}[/] — No predictions in database yet.\n")
        return

    arrow, dir_color = _dir_style(data.get("direction", "neutral"))
    is_demo = data.get("_demo", False)

    body = (
        f"  [bright_black]Timestamp :[/]  {data.get('timestamp', '—')}\n"
        f"  [bright_black]Direction :[/]  [{dir_color}]{arrow} {(data.get('direction') or '—').upper()}[/]\n"
        f"  [bright_black]Magnitude :[/]  {_pct_str(data.get('magnitude_pct'))}\n\n"
        f"  [bright_black]Target 24h:[/]  {_fmt_price(data.get('target_24h'), coin)}\n"
        f"  [bright_black]Target 72h:[/]  {_fmt_price(data.get('target_72h'), coin)}\n"
        f"  [bright_black]Target  7d:[/]  {_fmt_price(data.get('target_7d'), coin)}\n\n"
        f"  [bright_black]Confidence:[/]  {_conf_bar(data.get('confidence', 0))}"
    )
    if is_demo:
        body += "\n\n  [yellow]⚠  DEMO DATA — no real predictions yet[/]"

    console.print()
    console.print(
        Panel(
            body, title=f"[bold {color}]◈  {coin} PREDICTION[/]", border_style=color, padding=(1, 2)
        )
    )
    console.print()


@app.command()
def history(
    coin: str = typer.Argument(..., help="SOL or DOGE"),
    limit: int = typer.Argument(10, help="Number of rows to show"),
) -> None:
    """Show recent prediction history for a coin."""
    coin = coin.upper()
    color = _coin_color(coin)

    raw = _fetch(f"/api/predictions/{coin}/history?limit={limit}")
    if raw is None:
        raw = _history_from_db(coin, limit)

    rows = [r for r in (raw or []) if not r.get("_demo")]
    if not rows:
        console.print(f"\n[{color}]{coin}[/] — No history yet.\n")
        return

    tbl = Table(
        box=box.SIMPLE_HEAVY,
        header_style="bold bright_black",
        border_style="bright_black",
        title=f"[bold {color}]{coin} PREDICTION HISTORY[/]",
        expand=True,
    )
    tbl.add_column("TIME", min_width=20)
    tbl.add_column("DIRECTION", min_width=14)
    tbl.add_column("CONF", min_width=6)
    tbl.add_column("MAGNITUDE", min_width=10)
    tbl.add_column("24H TARGET", min_width=12)

    for row in rows:
        arrow, dc = _dir_style(row.get("direction", "neutral"))
        tbl.add_row(
            str(row.get("timestamp", "—"))[:19].replace("T", " "),
            f"[{dc}]{arrow} {(row.get('direction') or '—').upper()}[/]",
            f"{int(row.get('confidence', 0) * 100)}%",
            _pct_str(row.get("magnitude_pct")),
            _fmt_price(row.get("target_24h"), coin),
        )

    console.print()
    console.print(tbl)
    console.print()


@app.command()
def notify(
    action: str = typer.Argument(..., help="enable or disable"),
    platform: str = typer.Argument(..., help="discord, whatsapp, or zalo"),
) -> None:
    """Toggle a notification channel on or off."""
    action = action.lower()
    platform = platform.lower()

    if action not in ("enable", "disable"):
        console.print("[red]Action must be 'enable' or 'disable'[/]")
        raise typer.Exit(1)
    if platform not in ("discord", "whatsapp", "zalo"):
        console.print("[red]Platform must be discord, whatsapp, or zalo[/]")
        raise typer.Exit(1)

    color = _UP_COLOR if action == "enable" else _DOWN_COLOR
    symbol = "✓" if action == "enable" else "✗"
    console.print(
        f"\n  [{color}]{symbol}[/]  [bold]{platform.upper()}[/] notifications "
        f"[{color}]{action.upper()}D[/]\n"
        f"     [bright_black](Notification pipeline available in a future phase)[/]\n"
    )


@app.command()
def run(
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8000, help="Bind port"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on file change"),
) -> None:
    """Start the CPT FastAPI server and open the dashboard."""
    try:
        import uvicorn
    except ImportError:
        console.print("[red]uvicorn not installed.[/]")
        raise typer.Exit(1)

    reload_str = "[bright_green]on[/]" if reload else "[bright_black]off[/]"
    console.print()
    console.print(
        Panel(
            f"[bright_black]Dashboard:[/]  [cyan underline]http://{host}:{port}[/]\n"
            f"[bright_black]API docs :[/]  [cyan underline]http://{host}:{port}/docs[/]\n"
            f"[bright_black]Reload   :[/]  {reload_str}",
            title="[bold cyan]◈  STARTING CPT SERVER[/]",
            border_style="cyan",
        )
    )
    console.print()
    uvicorn.run("server.app:app", host=host, port=port, reload=reload, log_level="info")


# ─── Entry point ──────────────────────────────────────────────────

if __name__ == "__main__":
    app()
