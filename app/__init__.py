from flask import Flask
from datetime import datetime, timezone

from config.settings import resource_path


def create_app():
    app = Flask(
        __name__,
        template_folder=resource_path("app/templates"),
        static_folder=resource_path("app/static"),
    )
    app.config['TEMPLATES_AUTO_RELOAD'] = True

    from app.routes.home import bp as home_bp
    from app.routes.framework import bp as framework_bp
    from app.routes.assets import bp as assets_bp
    from app.routes.routines import bp as routines_bp
    from app.routes.journal import bp as journal_bp
    from app.routes.portfolio import bp as portfolio_bp
    from app.routes.api import bp as api_bp
    from app.routes.pwa import bp as pwa_bp

    app.register_blueprint(home_bp)
    app.register_blueprint(framework_bp)
    app.register_blueprint(assets_bp)
    app.register_blueprint(routines_bp)
    app.register_blueprint(journal_bp)
    app.register_blueprint(portfolio_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(pwa_bp)

    @app.template_filter("time_ago")
    def time_ago(dt):
        if dt is None:
            return "unknown"
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        diff = now - dt
        seconds = int(diff.total_seconds())
        if seconds < 60:
            return f"{seconds}s ago"
        if seconds < 3600:
            return f"{seconds // 60}m ago"
        if seconds < 86400:
            return f"{seconds // 3600}h ago"
        return f"{seconds // 86400}d ago"

    @app.template_filter("fmt_num")
    def fmt_num(v, decimals=2):
        if v is None:
            return "—"
        v = float(v)
        if abs(v) >= 1_000_000:
            return f"{v / 1_000_000:,.1f}M"
        if abs(v) >= 1_000:
            return f"{v:,.0f}"
        if v != 0 and abs(v) < 0.001:
            return f"{v:.5f}"
        return f"{v:,.{decimals}f}"

    @app.template_filter("sector_color")
    def sector_color(pct):
        if pct is None:
            return "hsl(0, 0%, 15%)"
        pct = float(pct)
        clamped = max(-5.0, min(5.0, pct))
        hue = 120 if clamped >= 0 else 0
        lightness = 15 + abs(clamped) * 4
        return f"hsl({hue}, 70%, {lightness:.0f}%)"

    return app
