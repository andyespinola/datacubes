from __future__ import annotations

from flask import Flask, Response, abort, jsonify, request, send_from_directory

from .constants import STATIC_DIR
from .store import GalaxyStore


def create_app(store: GalaxyStore | None = None) -> Flask:
    app = Flask(__name__, static_folder=None)
    store = store or GalaxyStore.from_default_files()

    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/<path:filename>")
    def static_files(filename: str):
        path = (STATIC_DIR / filename).resolve()
        try:
            path.relative_to(STATIC_DIR.resolve())
        except ValueError:
            abort(404)
        if not path.exists():
            abort(404)
        return send_from_directory(STATIC_DIR, filename)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/api/config")
    def config():
        return jsonify(store.get_config())

    @app.get("/api/map")
    def map_data():
        component = request.args.get("component", "stars")
        quantity = request.args.get("quantity", "mass")
        view = request.args.get("view", "faceon")
        radius_kpc = float(request.args.get("radius_kpc", 35.0))
        bins = int(request.args.get("bins", 220))
        try:
            payload = store.get_map(component=component, quantity=quantity, view=view, radius_kpc=radius_kpc, bins=bins)
        except KeyError as exc:
            abort(400, str(exc))
        return jsonify(payload)

    @app.get("/api/profile")
    def profile_data():
        component = request.args.get("component", "stars")
        quantity = request.args.get("quantity", "mass")
        radius_kpc = float(request.args.get("radius_kpc", 35.0))
        bins = int(request.args.get("bins", 48))
        try:
            payload = store.get_profile(component=component, quantity=quantity, radius_kpc=radius_kpc, bins=bins)
        except KeyError as exc:
            abort(400, str(exc))
        return jsonify(payload)

    @app.get("/api/particles3d")
    def particles_3d():
        radius_kpc = float(request.args.get("radius_kpc", 35.0))
        max_stars = int(request.args.get("max_stars", 0))
        max_gas = int(request.args.get("max_gas", 0))
        star_quantity = request.args.get("star_quantity", "mass")
        gas_quantity = request.args.get("gas_quantity", "mass")
        try:
            payload = store.get_particle_cloud(
                radius_kpc=radius_kpc,
                max_stars=max_stars,
                max_gas=max_gas,
                star_quantity=star_quantity,
                gas_quantity=gas_quantity,
            )
        except KeyError as exc:
            abort(400, str(exc))
        return jsonify(payload)

    @app.get("/api/particles3d/buffer")
    def particles_3d_buffer():
        component = request.args.get("component", "stars")
        quantity = request.args.get("quantity", "mass")
        radius_kpc = float(request.args.get("radius_kpc", 35.0))
        max_points = int(request.args.get("max_points", 0))
        try:
            selection = store.get_particle_selection(
                component=component,
                radius_kpc=radius_kpc,
                max_points=max_points,
                quantity=quantity,
            )
        except KeyError as exc:
            abort(400, str(exc))
        response = Response(selection.packed.astype("float32", copy=False).tobytes(), mimetype="application/octet-stream")
        response.headers["X-Point-Count"] = str(selection.sampled_count)
        response.headers["X-Selected-Count"] = str(selection.selected_count)
        response.headers["X-Float-Stride"] = "4"
        response.headers["X-Component-Label"] = selection.label
        response.headers["X-Quantity"] = selection.quantity
        response.headers["X-Quantity-Label"] = selection.quantity_label
        return response

    return app
