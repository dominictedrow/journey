from flask import Blueprint, current_app, render_template, send_from_directory

bp = Blueprint("pwa", __name__)


@bp.route("/sw.js")
def service_worker():
    resp = send_from_directory(current_app.static_folder + "/js", "sw.js")
    resp.headers["Content-Type"] = "application/javascript"
    resp.headers["Service-Worker-Allowed"] = "/"
    resp.headers["Cache-Control"] = "no-cache"
    return resp


@bp.route("/offline")
def offline():
    return render_template("offline.html")
