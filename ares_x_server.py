"""Local Flask API and static web app host for the ARES-X project."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import os
from pathlib import Path
import threading
import uuid
from typing import Any

from flask import Flask, jsonify, request, send_file, Response

from ares_x_engine import MissionEngine, PLANET_PROFILES


ROOT = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 1_000_000

_allowed_origins = {
    "https://ares-x-ac.netlify.app",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
}
_allowed_origins.update(
    origin.strip().rstrip("/")
    for origin in os.environ.get("ARES_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
)

_engine: MissionEngine | None = None
_engine_lock = threading.RLock()
_experiment_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ares-x-experiment")
_experiments: dict[str, Future[dict[str, Any]]] = {}


def _get_engine() -> MissionEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = MissionEngine(seed=42)
        return _engine


def _json_error(message: str, status: int = 400) -> tuple[Response, int]:
    return jsonify({"error": message}), status


def _body() -> dict[str, Any]:
    value = request.get_json(silent=False)
    if not isinstance(value, dict):
        raise ValueError("Se esperaba un objeto JSON.")
    return value


@app.before_request
def validate_mutation_origin() -> Response | None:
    if request.path.startswith("/api/") and request.method in {"POST", "PATCH", "DELETE"}:
        if request.headers.get("Origin", "").rstrip("/") not in _allowed_origins:
            return _json_error("Origen no autorizado para modificar la simulación.", 403)
    return None


@app.after_request
def add_security_headers(response: Response) -> Response:
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("X-Frame-Options", "DENY")
    if request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/")
def home() -> Response:
    return send_file(ROOT / "ares_x_3d.html")


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
        '<rect width="64" height="64" rx="14" fill="#c1e879"/>'
        '<path d="M32 10 54 54H43l-4-9H25l-4 9H10L32 10Zm0 17-4 9h8l-4-9Z" fill="#111819"/>'
        "</svg>",
        mimetype="image/svg+xml",
    )


@app.get("/ares_x_3d.css")
def styles() -> Response:
    return send_file(ROOT / "ares_x_3d.css", mimetype="text/css")


@app.get("/ares_x_3d.js")
def client_script() -> Response:
    return send_file(ROOT / "ares_x_3d.js", mimetype="text/javascript")


@app.get("/api/health")
def health() -> Response:
    return jsonify({
        "status": "ok",
        "engine": "python",
        "model": "scikit-learn MLP",
        "planets": list(PLANET_PROFILES),
    })


@app.get("/api/state")
def get_state() -> Response:
    with _engine_lock:
        return jsonify(_get_engine().state())


@app.post("/api/reset")
def reset() -> Response | tuple[Response, int]:
    global _engine
    try:
        body = _body()
        seed = int(body.get("seed", 42))
        if not 0 <= seed <= 2**31 - 1:
            raise ValueError("La semilla debe estar entre 0 y 2^31−1.")
        with _engine_lock:
            _engine = MissionEngine(seed=seed)
            return jsonify(_engine.state())
    except (TypeError, ValueError) as error:
        return _json_error(str(error))


@app.post("/api/config")
def configure() -> Response | tuple[Response, int]:
    try:
        body = _body()
        with _engine_lock:
            engine = _get_engine()
            if "planet" in body:
                engine.set_planet(str(body["planet"]))
            if "battery" in body and engine.steps == 0:
                engine.configure(battery=body["battery"])
            allowed = {
                key: body[key]
                for key in (
                    "risk_weight", "gas_concentration", "strategy", "science_priority"
                )
                if key in body
            }
            if allowed:
                engine.configure(**allowed)
            return jsonify(engine.state())
    except (TypeError, ValueError) as error:
        return _json_error(str(error))


@app.post("/api/step")
def step() -> Response:
    with _engine_lock:
        return jsonify(_get_engine().step())


@app.post("/api/objects")
def add_object() -> Response | tuple[Response, int]:
    try:
        body = _body()
        with _engine_lock:
            return jsonify(_get_engine().add_object(
                kind=str(body["kind"]),
                row=int(body["row"]),
                col=int(body["col"]),
                object_id=str(body["id"]) if body.get("id") else None,
            ))
    except KeyError as error:
        return _json_error(f"Falta el campo obligatorio: {error.args[0]}")
    except (TypeError, ValueError) as error:
        return _json_error(str(error))


@app.patch("/api/objects/<object_id>")
def move_object(object_id: str) -> Response | tuple[Response, int]:
    try:
        body = _body()
        with _engine_lock:
            return jsonify(_get_engine().move_object(
                object_id, int(body["row"]), int(body["col"])
            ))
    except KeyError as error:
        return _json_error(f"Falta el campo obligatorio: {error.args[0]}")
    except (TypeError, ValueError) as error:
        return _json_error(str(error))


@app.delete("/api/objects/<object_id>")
def delete_object(object_id: str) -> Response | tuple[Response, int]:
    try:
        with _engine_lock:
            return jsonify(_get_engine().remove_object(object_id))
    except ValueError as error:
        return _json_error(str(error), 404)


@app.post("/api/map/clear")
def clear_map() -> Response:
    with _engine_lock:
        return jsonify(_get_engine().clear_objects())


@app.post("/api/markers/<marker>")
def move_marker(marker: str) -> Response | tuple[Response, int]:
    try:
        body = _body()
        row, col = int(body["row"]), int(body["col"])
        with _engine_lock:
            engine = _get_engine()
            if marker == "start":
                return jsonify(engine.move_start(row, col))
            if marker == "goal":
                return jsonify(engine.move_goal(row, col))
            return _json_error("El marcador debe ser 'start' o 'goal'.")
    except KeyError as error:
        return _json_error(f"Falta el campo obligatorio: {error.args[0]}")
    except (TypeError, ValueError) as error:
        return _json_error(str(error))


@app.post("/api/events")
def event() -> Response | tuple[Response, int]:
    try:
        body = _body()
        with _engine_lock:
            return jsonify(_get_engine().inject_event(str(body["event"])))
    except KeyError as error:
        return _json_error(f"Falta el campo obligatorio: {error.args[0]}")
    except (TypeError, ValueError) as error:
        return _json_error(str(error))


@app.post("/api/faults")
def fault() -> Response | tuple[Response, int]:
    try:
        body = _body()
        with _engine_lock:
            return jsonify(_get_engine().set_fault(
                str(body["fault"]), bool(body.get("enabled", True))
            ))
    except KeyError as error:
        return _json_error(f"Falta el campo obligatorio: {error.args[0]}")
    except (TypeError, ValueError) as error:
        return _json_error(str(error))


@app.get("/api/export/trace.csv")
def export_csv() -> Response:
    with _engine_lock:
        csv_content = _get_engine().export_csv()
    return Response(
        csv_content,
        mimetype="text/csv",
        headers={"Content-Disposition": 'attachment; filename="ares_x_traza.csv"'},
    )


@app.get("/api/export/mission.json")
def export_json() -> Response:
    with _engine_lock:
        json_content = _get_engine().export_json()
    return Response(
        json_content,
        mimetype="application/json; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="ares_x_mision.json"'},
    )


def _run_experiment(experiment_id: str, seeds: int) -> dict[str, Any]:
    with _engine_lock:
        source = _get_engine()
        engine = MissionEngine(source.seed, train_model=False)
        engine.model, engine.perception_metrics = source.model, source.perception_metrics
        engine.objects = [dict(item) for item in source.objects]
        engine.start, engine.goal = source.start, source.goal
        engine.planet = source.planet
        engine.science_sites = set(source.science_sites)
        engine.risk_weight = source.risk_weight
        engine.gas_concentration = source.gas_concentration
        engine.science_priority = source.science_priority
        engine.battery_initial = source.battery_initial
    return engine.run_ablation_experiment(seeds)


@app.post("/api/experiments")
def start_experiment() -> Response:
    try:
        body = _body()
        seeds = int(body.get("seeds", 20))
        if not 20 <= seeds <= 30:
            raise ValueError("La comparación debe usar entre 20 y 30 semillas.")
        experiment_id = uuid.uuid4().hex
        _experiments[experiment_id] = _experiment_executor.submit(
            _run_experiment, experiment_id, seeds
        )
        return jsonify({"id": experiment_id, "status": "running"}), 202
    except (TypeError, ValueError) as error:
        return _json_error(str(error))


@app.get("/api/experiments/<experiment_id>")
def experiment_status(experiment_id: str) -> Response | tuple[Response, int]:
    future = _experiments.get(experiment_id)
    if future is None:
        return _json_error("No se encontró el experimento.", 404)
    if not future.done():
        return jsonify({"id": experiment_id, "status": "running"})
    try:
        return jsonify({
            "id": experiment_id,
            "status": "complete",
            "result": future.result(),
        })
    except Exception as error:  # noqa: BLE001 - async API must surface worker failures.
        return _json_error(f"El experimento falló: {error}", 500)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False, threaded=True)
