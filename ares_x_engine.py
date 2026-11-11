"""Simulation engine for the ARES-X educational rover demonstrator.

All physical parameters in this module are adjustable simulation assumptions,
not spacecraft engineering limits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import csv
import heapq
import io
import json
import math
import random
import re
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


MAP_SIZE = 20
CELL_METERS = 10
CLASSES = ("libre", "obstaculo", "barranco")
MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))
PLANET_PROFILES = {
    "mars": {"name": "Marte", "mode": "rover", "temperature_c": -63.0},
    "mercury": {"name": "Mercurio", "mode": "rover", "temperature_c": 167.0},
    "earth": {"name": "Tierra", "mode": "rover", "temperature_c": 15.0},
    "venus": {"name": "Venus", "mode": "rover_hipotetico", "temperature_c": 464.0},
    "jupiter": {"name": "Júpiter", "mode": "sonda_virtual", "temperature_c": -110.0},
    "saturn": {"name": "Saturno", "mode": "sonda_virtual", "temperature_c": -140.0},
    "uranus": {"name": "Urano", "mode": "sonda_virtual", "temperature_c": -195.0},
    "neptune": {"name": "Neptuno", "mode": "sonda_virtual", "temperature_c": -200.0},
}
FEATURE_NAMES = (
    "distancia_obstaculo_m",
    "distancia_crater_m",
    "pendiente_grados",
    "vibracion_normalizada",
    "textura_roca",
    "temperatura_normalizada",
    "radiacion_normalizada",
    "gas_normalizado",
)


@dataclass(frozen=True)
class FuzzyRule:
    identifier: str
    text: str
    speed: float
    caution: float


FUZZY_RULES = (
    FuzzyRule("R1", "Si el riesgo próximo es crítico, detener.", 0.0, 1.0),
    FuzzyRule("R2", "Si la batería es crítica, avanzar muy lento.", 0.16, 0.95),
    FuzzyRule("R3", "Si el riesgo es alto y la batería baja, avanzar lento.", 0.24, 0.86),
    FuzzyRule("R4", "Si la pendiente es fuerte, avanzar lento.", 0.28, 0.78),
    FuzzyRule("R5", "Si hay mucho gas, reducir la velocidad.", 0.38, 0.68),
    FuzzyRule("R6", "Si el riesgo es medio y la batería media, cautela moderada.", 0.48, 0.58),
    FuzzyRule("R7", "Si el riesgo es bajo y la batería alta, velocidad moderada.", 0.7, 0.28),
    FuzzyRule("R8", "Si el terreno está despejado y la batería alta, velocidad alta.", 0.92, 0.12),
    FuzzyRule("R9", "Si la confianza neuronal es baja, conducir con cautela.", 0.32, 0.8),
    FuzzyRule("R10", "Si la batería y el riesgo son moderados, conservar energía.", 0.55, 0.48),
)


@dataclass
class RouteCandidate:
    name: str
    cells: list[tuple[int, int]]
    distance_m: float
    risk: float
    energy: float
    science: float
    fitness: float
    feasible: bool
    genes: list[tuple[int, int]] | None = None

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["cells"] = [{"row": row, "col": col} for row, col in self.cells]
        return value


def _triangle(value: float, left: float, peak: float, right: float) -> float:
    if value == peak:
        return 1.0
    if value <= left or value >= right:
        return 0.0
    if value < peak:
        return (value - left) / (peak - left)
    return (right - value) / (right - peak)


def _trap(value: float, left: float, inner_left: float, inner_right: float, right: float) -> float:
    if inner_left <= value <= inner_right:
        return 1.0
    if value <= left or value >= right:
        return 0.0
    if value < inner_left:
        return (value - left) / (inner_left - left)
    return (right - value) / (right - inner_right)


def fuzzy_membership(value: float, kind: str) -> float:
    if kind == "low":
        return _trap(value, 0, 0, 0.2, 0.5)
    if kind == "medium":
        return _triangle(value, 0.2, 0.5, 0.8)
    if kind == "high":
        return _trap(value, 0.55, 0.8, 1, 1)
    if kind == "critical":
        return _trap(value, 0, 0, 0.12, 0.3)
    if kind == "near":
        return _trap(value, 0, 0, 2, 6)
    if kind == "mid":
        return _triangle(value, 3, 10, 17)
    if kind == "far":
        return _trap(value, 14, 18, 20, 20)
    return 0.0


def fuzzy_control(
    *,
    battery: float,
    hazard: float,
    slope: float,
    gas: float,
    neural_confidence: float,
) -> dict[str, Any]:
    """Mamdani min-max inference with numeric centroid defuzzification."""
    battery_n = min(1.0, max(0.0, battery / 100.0))
    hazard_n = min(1.0, max(0.0, hazard))
    slope_n = min(1.0, max(0.0, slope / 45.0))
    gas_n = min(1.0, max(0.0, gas))
    memberships = {
        "bateria_critica": fuzzy_membership(battery_n, "critical"),
        "bateria_baja": fuzzy_membership(battery_n, "low"),
        "bateria_media": fuzzy_membership(battery_n, "medium"),
        "bateria_alta": fuzzy_membership(battery_n, "high"),
        "riesgo_cerca": fuzzy_membership(hazard_n, "high"),
        "riesgo_medio": fuzzy_membership(hazard_n, "medium"),
        "riesgo_bajo": fuzzy_membership(hazard_n, "low"),
        "pendiente_fuerte": fuzzy_membership(slope_n, "high"),
        "gas_alto": fuzzy_membership(gas_n, "high"),
        "confianza_baja": fuzzy_membership(neural_confidence, "low"),
    }

    activations = {
        "R1": min(memberships["riesgo_cerca"], fuzzy_membership(hazard_n, "high")),
        "R2": memberships["bateria_critica"],
        "R3": min(memberships["riesgo_cerca"], memberships["bateria_baja"]),
        "R4": memberships["pendiente_fuerte"],
        "R5": memberships["gas_alto"],
        "R6": min(memberships["riesgo_medio"], memberships["bateria_media"]),
        "R7": min(memberships["riesgo_bajo"], memberships["bateria_alta"]),
        "R8": min(memberships["riesgo_bajo"], memberships["bateria_alta"]),
        "R9": memberships["confianza_baja"],
        "R10": min(memberships["bateria_media"], memberships["riesgo_medio"]),
    }

    # Sampled output universe; min clips each rule, max aggregates rule consequents.
    output_axis = np.linspace(0.0, 1.0, 501)
    aggregate = np.zeros_like(output_axis)
    for rule in FUZZY_RULES:
        activation = activations[rule.identifier]
        consequent = np.maximum(0.0, 1.0 - np.abs(output_axis - rule.speed) / 0.2)
        aggregate = np.maximum(aggregate, np.minimum(activation, consequent))
    area = float(np.trapezoid(aggregate, output_axis))
    speed = (
        float(np.trapezoid(aggregate * output_axis, output_axis) / area)
        if area > 0
        else 0.0
    )
    caution = max(
        (activations[rule.identifier] * rule.caution for rule in FUZZY_RULES),
        default=1.0,
    )
    fired = [
        {
            "id": rule.identifier,
            "text": rule.text,
            "activation": round(activations[rule.identifier], 3),
            "speed": rule.speed,
            "caution": rule.caution,
        }
        for rule in FUZZY_RULES
        if activations[rule.identifier] > 0.01
    ]
    return {
        "speed": round(speed, 3),
        "caution": round(caution, 3),
        "memberships": {key: round(value, 3) for key, value in memberships.items()},
        "rules": fired,
        "aggregation_area": round(area, 4),
        "defuzzification": "centroide Mamdani (min–max)",
        "fallback": None if area > 0 else "Sin reglas activas: velocidad segura 0.",
    }


def _scenario(seed: int, size: int = MAP_SIZE) -> list[dict[str, int | str]]:
    rng = random.Random(seed)
    objects: list[dict[str, int | str]] = []
    kinds = (
        ("rock", 22),
        ("crater", 11),
        ("rough", 18),
        ("gas", 12),
        ("cliff", 6),
    )
    cells = [(row, col) for row in range(size) for col in range(size)]
    rng.shuffle(cells)
    cell_index = 0
    for kind, count in kinds:
        for _ in range(count):
            row, col = cells[cell_index]
            cell_index += 1
            objects.append({
                "kind": kind,
                "row": row,
                "col": col,
            })
    return objects


def _distance_to_kind(
    row: int, col: int, objects: list[dict[str, int | str]], kinds: set[str]
) -> float:
    distances = [
        abs(row - int(item["row"])) + abs(col - int(item["col"]))
        for item in objects
        if item["kind"] in kinds
    ]
    return min(distances, default=20) * CELL_METERS


def _sensor_features(
    row: int,
    col: int,
    objects: list[dict[str, int | str]],
    rng: np.random.Generator,
    *,
    noisy: bool,
    ambient_temperature: float,
) -> list[float]:
    nearby = [
        item for item in objects
        if abs(row - int(item["row"])) + abs(col - int(item["col"])) <= 3
    ]
    on_cell = [
        item for item in objects
        if int(item["row"]) == row and int(item["col"]) == col
    ]
    obstacle_distance = _distance_to_kind(row, col, nearby, {"rock", "cliff"})
    crater_distance = _distance_to_kind(row, col, nearby, {"crater"})
    slope = min(42.0, abs(math.sin(row * 0.37 + col * 0.19)) * 31 + 1)
    vibration = 0.13 + (0.48 if any(item["kind"] == "rough" for item in on_cell) else 0.04)
    texture = 0.86 if any(item["kind"] in {"rock", "rough"} for item in on_cell) else 0.2
    gas = 0.78 if any(item["kind"] == "gas" for item in on_cell) else 0.05
    temp = ambient_temperature
    if noisy:
        obstacle_distance += float(rng.normal(0.0, 2.4))
        crater_distance += float(rng.normal(0.0, 2.8))
        slope += float(rng.normal(0.0, 3.0))
        vibration += float(rng.normal(0.0, 0.11))
        texture += float(rng.normal(0.0, 0.14))
        gas += float(rng.normal(0.0, 0.12))
        temp += float(rng.normal(0.0, 8.0))
    return [
        min(20.0, max(0.0, obstacle_distance)),
        min(20.0, max(0.0, crater_distance)),
        min(45.0, max(0.0, slope)),
        min(1.0, max(0.0, vibration)),
        min(1.0, max(0.0, texture)),
        min(1.0, max(0.0, (temp + 250) / 750)),
        min(1.0, max(0.0, 0.18 + (row % 5) * 0.035)),
        min(1.0, max(0.0, gas)),
    ]


@lru_cache(maxsize=8)
def _train_perception_model(training_seed: int) -> tuple[Any, dict[str, Any]]:
    rng = np.random.default_rng(training_seed)
    feature_rows: list[list[float]] = []
    labels: list[int] = []
    split_data: dict[str, tuple[list[list[float]], list[int]]] = {}
    split_map_counts: dict[str, int] = {}
    split_scenarios: dict[str, list[int]] = {}
    split_sizes = {"train": 10, "validation": 2, "test": 2}
    map_offset = 0
    for split_name, map_count in split_sizes.items():
        rows: list[list[float]] = []
        targets: list[int] = []
        scenario_ids: list[int] = []
        for map_index in range(map_count):
            scenario_seed = training_seed * 1000 + map_offset
            scenario_ids.append(scenario_seed)
            objects = _scenario(scenario_seed)
            for row in range(MAP_SIZE):
                for col in range(MAP_SIZE):
                    cell_items = [
                        item for item in objects
                        if int(item["row"]) == row and int(item["col"]) == col
                    ]
                    kind = str(cell_items[0]["kind"]) if cell_items else "free"
                    # Several independent noisy reads per cell; maps remain disjoint by split.
                    for _ in range(3):
                        rows.append(_sensor_features(
                            row, col, objects, rng, noisy=True, ambient_temperature=-60.0
                        ))
                        targets.append(
                            1 if kind in {"rock", "cliff"} else
                            2 if kind == "crater" else 0
                        )
            map_offset += 1
        split_data[split_name] = (rows, targets)
        split_map_counts[split_name] = map_count
        split_scenarios[split_name] = scenario_ids

    train_x, train_y = split_data["train"]
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            max_iter=220,
            early_stopping=False,
            random_state=training_seed,
            tol=5e-4,
        ),
    )
    model.fit(np.asarray(train_x), np.asarray(train_y))

    metrics: dict[str, Any] = {
        "architecture": "MLP 8-32-16-3 · ReLU · softmax",
        "features": list(FEATURE_NAMES),
        "classes": list(CLASSES),
        "seed": training_seed,
        "split_maps": split_map_counts,
        "split_scenarios": split_scenarios,
        "training_maps_do_not_overlap_validation_or_test": (
            set(split_scenarios["train"]).isdisjoint(split_scenarios["validation"])
            and set(split_scenarios["train"]).isdisjoint(split_scenarios["test"])
            and set(split_scenarios["validation"]).isdisjoint(split_scenarios["test"])
        ),
        "loss_curve": [round(float(value), 4) for value in model[-1].loss_curve_],
        "training_iterations": int(model[-1].n_iter_),
        "converged": model[-1].n_iter_ < model[-1].max_iter,
        "metrics": {},
        "confusion_matrix": {},
    }
    for split_name in ("train", "validation", "test"):
        split_x, split_y = split_data[split_name]
        predictions = model.predict(split_x)
        metrics["metrics"][split_name] = {
            "accuracy": round(float(accuracy_score(split_y, predictions)), 4),
            "macro_f1": round(float(f1_score(
                split_y, predictions, labels=[0, 1, 2], average="macro", zero_division=0
            )), 4),
            "hazard_recall": round(float(recall_score(
                split_y, predictions, labels=[2], average="macro", zero_division=0
            )), 4),
        }
        if split_name == "test":
            matrix = confusion_matrix(split_y, predictions, labels=[0, 1, 2])
            metrics["confusion_matrix"] = {
                "labels": list(CLASSES),
                "values": matrix.tolist(),
            }
    return model, metrics


class MissionEngine:
    """Stateful, reproducible simulator with RNA, fuzzy and genetic planning."""

    def __init__(self, seed: int = 42, train_model: bool = True) -> None:
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.np_rng = np.random.default_rng(self.seed)
        self.planet = "mars"
        self.objects: list[dict[str, Any]] = [
            {"id": f"rock-wall-{col}", "kind": "rock", "row": 10, "col": col}
            for col in range(MAP_SIZE) if col not in (4, 15)
        ] + [
            {"id": "crater-a", "kind": "crater", "row": 8, "col": 4},
            {"id": "crater-b", "kind": "crater", "row": 14, "col": 6},
            {"id": "gas-a", "kind": "gas", "row": 7, "col": 11},
            {"id": "rough-a", "kind": "rough", "row": 9, "col": 3},
            {"id": "rough-b", "kind": "rough", "row": 11, "col": 8},
        ]
        self.science_sites: set[tuple[int, int]] = {(4, 16), (15, 5), (5, 5)}
        self.start = (18, 1)
        self.goal = (1, 18)
        self.rover = self.start
        self.battery_initial = 100.0
        self.battery = 100.0
        self.risk_weight = 0.6
        self.gas_concentration = 0.7
        self.science_priority = 0.7
        self.strategy = "safe"
        self.steps = 0
        self.distance_travelled = 0.0
        self.science_collected = 0
        self.events: list[dict[str, Any]] = []
        self.trace: list[dict[str, Any]] = []
        self.genetic_history: list[dict[str, Any]] = []
        self.genetic_population_size = 32
        self.genetic_generations = 30
        self.last_decision: dict[str, Any] | None = None
        self.route_candidates: list[RouteCandidate] = []
        self.route: list[tuple[int, int]] = []
        self.ablation = {"rna": True, "fuzzy": True, "genetic": True}
        self.faults: set[str] = set()
        self.failure: str | None = None
        self.complete = False
        self._id_counter = len(self.objects)
        self._beliefs: dict[tuple[int, int], dict[str, Any]] = {}
        self._confidence_threshold = 0.66
        if train_model:
            self.model, self.perception_metrics = _train_perception_model(self.seed)
        else:
            self.model, self.perception_metrics = None, {}
        self._scan_nearby()
        self.replan(reason="Inicio de misión")

    def set_planet(self, planet: str) -> dict[str, Any]:
        if planet not in PLANET_PROFILES:
            raise ValueError(f"Planeta no reconocido: {planet}")
        self.planet = planet
        if PLANET_PROFILES[planet]["mode"] == "sonda_virtual":
            for item in self.objects:
                if item["kind"] in {"rock", "crater", "rough", "cliff", "meteorite"}:
                    item.setdefault("_surface_kind", item["kind"])
                    item["kind"] = "gas"
        else:
            for item in self.objects:
                if "_surface_kind" in item:
                    item["kind"] = item.pop("_surface_kind")
        self._scan_nearby()
        self.replan(reason=f"Perfil de {PLANET_PROFILES[planet]['name']}")
        return self.state()

    def configure(self, **values: Any) -> dict[str, Any]:
        if "risk_weight" in values:
            self.risk_weight = float(np.clip(float(values["risk_weight"]), 0, 1.5))
        if "gas_concentration" in values:
            self.gas_concentration = float(np.clip(float(values["gas_concentration"]), 0.1, 1))
        if "science_priority" in values:
            self.science_priority = float(np.clip(float(values["science_priority"]), 0, 1))
        if "battery" in values:
            self.battery_initial = float(np.clip(float(values["battery"]), 0, 100))
            self.battery = self.battery_initial
        if "strategy" in values:
            if values["strategy"] not in {"safe", "distance"}:
                raise ValueError("La estrategia debe ser 'safe' o 'distance'.")
            self.strategy = values["strategy"]
        self.replan(reason="Parámetros de misión actualizados")
        return self.state()

    def _true_objects(self) -> list[dict[str, int | str]]:
        return [
            {"kind": str(item["kind"]), "row": int(item["row"]), "col": int(item["col"])}
            for item in self.objects
        ]

    def _scan_nearby(self, radius: int = 4) -> None:
        if self.failure:
            return
        objects = self._true_objects()
        profile = PLANET_PROFILES[self.planet]
        observations: list[tuple[int, int, list[float]]] = []
        for row in range(MAP_SIZE):
            for col in range(MAP_SIZE):
                if abs(row - self.rover[0]) + abs(col - self.rover[1]) > radius:
                    continue
                features = _sensor_features(
                    row, col, objects, self.np_rng, noisy=True,
                    ambient_temperature=float(profile["temperature_c"]),
                )
                observations.append((row, col, features))
        if not observations:
            return

        matrix = np.asarray([item[2] for item in observations])
        if self.model is not None and self.ablation["rna"] and not any(
            fault in self.faults for fault in ("all", "camera", "lidar")
        ):
            probabilities = self.model.predict_proba(matrix)
        else:
            probabilities = np.tile([0.6, 0.2, 0.2], (len(observations), 1))

        for (row, col, features), probability in zip(observations, probabilities):
            predicted_index = int(np.argmax(probability))
            confidence = float(probability[predicted_index])
            sensed_obstacle = features[0] <= 0.5
            if "lidar" in self.faults or "all" in self.faults:
                sensed_obstacle = False
            kind = CLASSES[predicted_index]
            if features[7] >= 0.62:
                terrain_estimate = "gas"
            elif features[1] <= 0.5:
                terrain_estimate = "barranco"
            elif kind == "obstaculo" and confidence >= self._confidence_threshold:
                terrain_estimate = "obstaculo"
            elif kind == "barranco" and features[1] <= 0.5:
                terrain_estimate = "barranco"
            elif features[4] >= 0.62 and features[3] >= 0.35:
                terrain_estimate = "rocoso"
            else:
                terrain_estimate = "libre"
            blocked = sensed_obstacle or features[1] <= 0.5 or (
                kind == "obstaculo" and confidence >= self._confidence_threshold
            )
            gas = features[7] if "gas" not in self.faults else 0.0
            self._beliefs[(row, col)] = {
                "features": [round(float(value), 3) for value in features],
                "probabilities": {
                    CLASSES[index]: round(float(value), 4)
                    for index, value in enumerate(probability)
                },
                "prediction": kind,
                "confidence": round(confidence, 4),
                "blocked": blocked,
                "gas": round(float(gas), 3),
                "slope": round(float(features[2]), 2),
                "observed_step": self.steps,
                "terrain": terrain_estimate,
            }

    def _terrain_kind(self, row: int, col: int) -> str:
        for item in self.objects:
            if item["row"] == row and item["col"] == col:
                return str(item["kind"])
        return "libre"

    def add_object(self, kind: str, row: int, col: int, object_id: str | None = None) -> dict[str, Any]:
        allowed = {"rock", "crater", "gas", "rough", "cliff", "science", "meteorite"}
        if kind not in allowed:
            raise ValueError(f"Tipo de elemento no válido: {kind}")
        if not (0 <= int(row) < MAP_SIZE and 0 <= int(col) < MAP_SIZE):
            raise ValueError("La celda debe estar dentro del mapa.")
        if (int(row), int(col)) in {self.rover, self.goal, self.start}:
            raise ValueError("No se puede colocar un elemento sobre el rover o un punto de misión.")
        if kind != "gas" and any(
            item["row"] == int(row) and item["col"] == int(col) for item in self.objects
        ):
            raise ValueError("La celda ya contiene un elemento.")
        if object_id is None:
            self._id_counter += 1
            object_id = f"object-{self._id_counter}"
        elif not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", str(object_id)):
            raise ValueError("El identificador debe tener 1–64 caracteres alfanuméricos, guiones o guiones bajos.")
        self.objects = [item for item in self.objects if item["id"] != object_id]
        self.objects.append({
            "id": str(object_id), "kind": kind, "row": int(row), "col": int(col),
        })
        if kind == "science":
            self.science_sites.add((int(row), int(col)))
        self._scan_nearby(radius=6)
        self.events.append({
            "step": self.steps, "type": "map_edit", "message": f"{kind} agregado en F{row + 1} C{col + 1}",
        })
        self.replan(reason=f"Nuevo {kind} en el mapa")
        return self.state()

    def move_object(self, object_id: str, row: int, col: int) -> dict[str, Any]:
        entity = next((item for item in self.objects if item["id"] == object_id), None)
        if entity is None:
            raise ValueError("No se encontró el elemento.")
        if (int(row), int(col)) in {self.rover, self.goal, self.start}:
            raise ValueError("No se puede solapar el elemento con el rover o los puntos de misión.")
        if any(
            item["id"] != object_id and item["row"] == int(row) and item["col"] == int(col)
            for item in self.objects
        ):
            raise ValueError("La celda de destino ya contiene otro elemento.")
        previous_cell = (int(entity["row"]), int(entity["col"]))
        entity["row"], entity["col"] = int(row), int(col)
        if entity["kind"] == "science":
            self.science_sites.discard(previous_cell)
            self.science_sites.add((int(row), int(col)))
        self._beliefs.pop((int(row), int(col)), None)
        self._scan_nearby(radius=6)
        self.events.append({
            "step": self.steps, "type": "map_edit", "message": f"{entity['kind']} reubicado a F{row + 1} C{col + 1}",
        })
        self.replan(reason="Elemento movido")
        return self.state()

    def remove_object(self, object_id: str) -> dict[str, Any]:
        before = len(self.objects)
        removed = [item for item in self.objects if item["id"] == object_id]
        self.objects = [item for item in self.objects if item["id"] != object_id]
        if len(self.objects) == before:
            raise ValueError("No se encontró el elemento.")
        for item in removed:
            if item["kind"] == "science":
                self.science_sites.discard((item["row"], item["col"]))
        self._scan_nearby(radius=6)
        self.events.append({"step": self.steps, "type": "map_edit", "message": f"{object_id} retirado"})
        self.replan(reason="Elemento retirado")
        return self.state()

    def clear_objects(self) -> dict[str, Any]:
        self.objects.clear()
        self.science_sites.clear()
        self._beliefs.clear()
        self._scan_nearby(radius=6)
        self.events.append({
            "step": self.steps, "type": "map_edit",
            "message": "Se retiraron todos los elementos del terreno.",
        })
        self.replan(reason="Mapa despejado por el operador")
        return self.state()

    def move_start(self, row: int, col: int) -> dict[str, Any]:
        return self._move_marker("start", row, col)

    def move_goal(self, row: int, col: int) -> dict[str, Any]:
        return self._move_marker("goal", row, col)

    def _move_marker(self, marker: str, row: int, col: int) -> dict[str, Any]:
        point = (int(row), int(col))
        if not (0 <= point[0] < MAP_SIZE and 0 <= point[1] < MAP_SIZE):
            raise ValueError("El punto debe estar dentro del mapa.")
        if self._true_blocked(*point):
            raise ValueError("El punto debe ubicarse en una celda libre.")
        if marker == "start":
            self.start = point
            self.rover = point
            self.steps = 0
            self.battery = self.battery_initial
            self.trace.clear()
            self.failure = None
            self.complete = False
        else:
            self.goal = point
        self._scan_nearby(radius=6)
        self.replan(reason=f"{marker} reubicado")
        return self.state()

    def set_fault(self, fault: str, enabled: bool = True) -> dict[str, Any]:
        supported = {"camera", "lidar", "gas", "all", "link"}
        if fault not in supported:
            raise ValueError("Tipo de fallo no reconocido.")
        if enabled:
            self.faults.add(fault)
        else:
            self.faults.discard(fault)
        if fault in {"camera", "lidar", "all"}:
            self._scan_nearby()
        self.events.append({
            "step": self.steps, "type": "sensor_fault",
            "message": f"Fallo de sensor {fault}: {'activado' if enabled else 'desactivado'}",
        })
        self.replan(reason=f"Fallo de sensor: {fault}")
        return self.state()

    def inject_event(self, event: str) -> dict[str, Any]:
        if event == "storm":
            self.gas_concentration = min(1.0, self.gas_concentration + 0.25)
            self.faults.add("camera")
            message = "Tormenta de polvo: visibilidad y lectura óptica reducidas."
        elif event == "quake":
            for _ in range(4):
                row, col = self.rng.randrange(MAP_SIZE), self.rng.randrange(MAP_SIZE)
                if (row, col) not in {self.rover, self.goal, self.start}:
                    self.objects.append({
                        "id": f"quake-{self.steps}-{row}-{col}",
                        "kind": "rough", "row": row, "col": col,
                    })
            message = "Sismo: terreno alterado en cuatro celdas simuladas."
        elif event == "meteor":
            choices = [
                (r, c) for r in range(MAP_SIZE) for c in range(MAP_SIZE)
                if abs(r - self.rover[0]) + abs(c - self.rover[1]) > 3
                and (r, c) not in {self.goal, self.start}
                and not self._true_blocked(r, c)
            ]
            row, col = self.rng.choice(choices)
            self.objects.append({
                "id": f"meteor-{self.steps}-{row}-{col}",
                "kind": "crater", "row": row, "col": col,
            })
            message = f"Impacto simulado en F{row + 1} C{col + 1}; zona bloqueada."
        elif event == "night":
            message = "Noche simulada: energía solar reducida; adaptar la reserva."
            self.battery = max(0, self.battery - 2)
        else:
            raise ValueError(f"Evento no reconocido: {event}")
        self._scan_nearby(radius=6)
        self.events.append({"step": self.steps, "type": event, "message": message})
        self.replan(reason=message)
        return self.state()

    def _true_blocked(self, row: int, col: int) -> bool:
        return any(
            item["row"] == row and item["col"] == col
            and item["kind"] in {"rock", "crater", "cliff", "meteorite"}
            for item in self.objects
        )

    def _belief_at(self, cell: tuple[int, int]) -> dict[str, Any] | None:
        return self._beliefs.get(cell)

    def _perceived_risk(self, cell: tuple[int, int]) -> float:
        belief = self._belief_at(cell)
        if belief is None:
            return 0.7
        probabilities = belief["probabilities"]
        terrain = belief["terrain"]
        if self.ablation["rna"]:
            explicit = {
                "barranco": 0.72, "obstaculo": 0.88, "rocoso": 0.34,
                "gas": self.gas_concentration * 0.72,
            }.get(terrain, 0.0)
        else:
            explicit = (
                self.gas_concentration * 0.72
                if belief["features"][7] >= 0.62
                else 0.34
                if belief["features"][4] >= 0.62 and belief["features"][3] >= 0.35
                else 0.0
            )
        neural_risk = (
            0.48 * probabilities.get("barranco", 0.0)
            + 0.78 * probabilities.get("obstaculo", 0.0)
        ) if self.ablation["rna"] else 0.0
        return min(1.0, max(explicit, neural_risk))

    def _belief_blocked(self, cell: tuple[int, int]) -> bool:
        belief = self._belief_at(cell)
        if belief is None:
            return False
        if belief["blocked"]:
            return True
        if not self.ablation["rna"]:
            return bool(belief["features"][0] <= 0.5)
        probabilities = belief["probabilities"]
        return (
            belief["confidence"] >= self._confidence_threshold
            and probabilities.get("obstaculo", 0.0) >= 0.72
        )

    def _slope(self, cell: tuple[int, int]) -> float:
        belief = self._belief_at(cell)
        return float(belief["slope"]) if belief else 18.0

    def _fitness(self, cells: list[tuple[int, int]], *, risk_weight: float | None = None) -> dict[str, float]:
        distance = max(0, len(cells) - 1) * CELL_METERS
        risk = sum(self._perceived_risk(cell) for cell in cells[1:])
        rough_count = sum(
            1 for cell in cells[1:]
            if self._belief_at(cell) and self._belief_at(cell)["terrain"] == "rough"
        )
        energy = distance * (0.045 + rough_count * 0.002)
        science = sum(1 for cell in cells if cell in self.science_sites) * 12.0
        weight = self.risk_weight if risk_weight is None else risk_weight
        fitness = (
            0.55 * distance / (MAP_SIZE * CELL_METERS)
            + weight * risk
            + 0.3 * energy / 100.0
            - science * self.science_priority
        )
        return {
            "distance_m": float(distance),
            "risk": float(risk),
            "energy": float(energy),
            "science": float(science),
            "fitness": float(fitness),
        }

    def _shortest_segment(
        self, start: tuple[int, int], goal: tuple[int, int], risk_weight: float,
        *, blocked: set[tuple[int, int]] | None = None,
    ) -> list[tuple[int, int]] | None:
        blocked_cells = blocked if blocked is not None else {
            cell for cell in self._beliefs if self._belief_blocked(cell)
        }
        if start in blocked_cells or goal in blocked_cells:
            return None
        frontier: list[tuple[float, int, int]] = [(0.0, start[0], start[1])]
        costs = {start: 0.0}
        previous: dict[tuple[int, int], tuple[int, int]] = {}
        while frontier:
            cost, row, col = heapq.heappop(frontier)
            current = (row, col)
            if cost > costs[current]:
                continue
            if current == goal:
                break
            for dr, dc in MOVES:
                neighbor = (row + dr, col + dc)
                if not (0 <= neighbor[0] < MAP_SIZE and 0 <= neighbor[1] < MAP_SIZE):
                    continue
                if neighbor in blocked_cells:
                    continue
                next_cost = cost + 1.0 + risk_weight * self._perceived_risk(neighbor)
                if next_cost < costs.get(neighbor, math.inf):
                    costs[neighbor] = next_cost
                    previous[neighbor] = current
                    heapq.heappush(frontier, (next_cost, *neighbor))
        if goal not in costs:
            return None
        path = [goal]
        while path[-1] != start:
            path.append(previous[path[-1]])
        return list(reversed(path))

    def _decode_waypoints(self, waypoints: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
        checkpoints = [self.rover, *waypoints, self.goal]
        path: list[tuple[int, int]] = []
        for index in range(len(checkpoints) - 1):
            segment = self._shortest_segment(
                checkpoints[index], checkpoints[index + 1], self.risk_weight
            )
            if segment is None:
                return None
            path.extend(segment if not path else segment[1:])
        return path

    def _genetic_plan(self, population_size: int = 32, generations: int = 30) -> tuple[list[RouteCandidate], list[dict[str, Any]]]:
        shortest = self._shortest_segment(self.rover, self.goal, 0.0)
        safe = self._shortest_segment(self.rover, self.goal, self.risk_weight)
        if shortest is None or safe is None:
            return [], []

        if len(shortest) <= 3:
            return [], []

        waypoint_count = min(3, max(1, len(shortest) // 8))
        open_cells = [
            (row, col)
            for row in range(MAP_SIZE) for col in range(MAP_SIZE)
            if (row, col) not in {self.rover, self.goal} and not self._belief_blocked((row, col))
        ]
        cache: dict[tuple[tuple[int, int], ...], tuple[list[tuple[int, int]] | None, float]] = {}

        def genome_from_path(path: list[tuple[int, int]], jitter: int = 0) -> tuple[tuple[int, int], ...]:
            genome: list[tuple[int, int]] = []
            for index in range(1, waypoint_count + 1):
                point_index = round(index * (len(path) - 1) / (waypoint_count + 1))
                base = path[min(len(path) - 1, max(0, point_index))]
                candidates = [
                    cell for cell in open_cells
                    if abs(cell[0] - base[0]) + abs(cell[1] - base[1]) <= max(1, jitter)
                ]
                genome.append(self.rng.choice(candidates) if candidates else base)
            return tuple(genome)

        def evaluate(genome: tuple[tuple[int, int], ...]) -> tuple[list[tuple[int, int]] | None, float]:
            if genome not in cache:
                route = self._decode_waypoints(list(genome))
                metrics = self._fitness(route) if route else {"fitness": 1e6}
                cache[genome] = (route, metrics["fitness"])
            return cache[genome]

        population = [
            genome_from_path(shortest),
            genome_from_path(safe),
        ]
        for site in sorted(self.science_sites):
            to_site = self._shortest_segment(self.rover, site, self.risk_weight)
            from_site = self._shortest_segment(site, self.goal, self.risk_weight)
            if not to_site or not from_site:
                continue
            science_route = to_site + from_site[1:]
            genome = list(genome_from_path(science_route))
            genome[0] = site
            population.append(tuple(genome))
            if len(population) >= population_size:
                break
        while len(population) < population_size:
            seed_path = safe if len(population) % 2 else shortest
            population.append(genome_from_path(seed_path, jitter=2 + len(population) % 5))

        history: list[dict[str, Any]] = []
        for generation in range(generations):
            ranked = sorted(
                ((evaluate(genome)[1], genome) for genome in population),
                key=lambda entry: entry[0],
            )
            current_best = ranked[0][0]
            history.append({
                "generation": generation + 1,
                "best_fitness": round(float(current_best), 4),
                "mean_fitness": round(float(np.mean([score for score, _ in ranked])), 4),
                "population": len(population),
                "operators": "torneo · cruce de puntos de paso · mutación · elitismo",
            })
            elite_count = max(2, population_size // 8)
            next_population = [genome for _, genome in ranked[:elite_count]]

            def tournament() -> tuple[tuple[int, int], ...]:
                contestants = self.rng.sample(ranked, k=min(3, len(ranked)))
                return min(contestants, key=lambda entry: entry[0])[1]

            while len(next_population) < population_size:
                parent_a, parent_b = tournament(), tournament()
                cut = self.rng.randrange(1, waypoint_count + 1)
                child = list(parent_a[:cut] + parent_b[cut:])
                if self.rng.random() < 0.35:
                    gene = self.rng.randrange(waypoint_count)
                    nearby = [
                        cell for cell in open_cells
                        if abs(cell[0] - child[gene][0]) + abs(cell[1] - child[gene][1]) <= 4
                    ]
                    if nearby:
                        child[gene] = self.rng.choice(nearby)
                next_population.append(tuple(child))
            population = next_population
        final = sorted(
            ((evaluate(genome)[1], genome, evaluate(genome)[0]) for genome in population),
            key=lambda entry: entry[0],
        )
        candidates: list[RouteCandidate] = []
        for _, genome, path in final:
            if not path or any(candidate.cells == path for candidate in candidates):
                continue
            metrics = self._fitness(path)
            candidates.append(RouteCandidate(
                name=f"AG · candidato {len(candidates) + 1}",
                cells=path,
                distance_m=metrics["distance_m"],
                risk=metrics["risk"],
                energy=metrics["energy"],
                science=metrics["science"],
                fitness=metrics["fitness"],
                feasible=True,
                genes=list(genome),
            ))
            if len(candidates) == 3:
                break
        return candidates, history

    def replan(self, reason: str = "Replanificación") -> None:
        distance_path = self._shortest_segment(self.rover, self.goal, 0.0)
        safe_path = self._shortest_segment(self.rover, self.goal, self.risk_weight)
        candidates: list[RouteCandidate] = []
        for name, path in (("Menor distancia", distance_path), ("Menor riesgo", safe_path)):
            if path:
                metrics = self._fitness(path)
                candidates.append(RouteCandidate(
                    name=name,
                    cells=path,
                    distance_m=metrics["distance_m"],
                    risk=metrics["risk"],
                    energy=metrics["energy"],
                    science=metrics["science"],
                    fitness=metrics["fitness"],
                    feasible=True,
                ))
        self.genetic_history = []
        if self.ablation["genetic"] and self.model is not None:
            genetic, self.genetic_history = self._genetic_plan(
                population_size=self.genetic_population_size,
                generations=self.genetic_generations,
            )
            existing = {tuple(candidate.cells) for candidate in candidates}
            candidates.extend(
                candidate for candidate in genetic if tuple(candidate.cells) not in existing
            )
        if self.strategy == "distance" and distance_path:
            self.route = distance_path
        elif self.ablation["genetic"] and len(candidates) > 2:
            self.route = min(candidates, key=lambda candidate: candidate.fitness).cells
        elif safe_path:
            self.route = safe_path
        else:
            self.route = []
        self.route_candidates = candidates[:5]
        self.events.append({
            "step": self.steps, "type": "replan", "message": reason,
            "candidates": len(self.route_candidates),
        })

    def _sensor_observation(self) -> dict[str, Any]:
        cell = self._belief_at(self.rover)
        if cell is None:
            self._scan_nearby()
            cell = self._belief_at(self.rover)
        if cell is None:
            return {
                "features": [0.0] * len(FEATURE_NAMES),
                "probabilities": {"libre": 0.33, "obstaculo": 0.34, "barranco": 0.33},
                "prediction": "desconocido",
                "confidence": 0.0,
            }
        return cell

    def step(self, *, ablation: dict[str, bool] | None = None) -> dict[str, Any]:
        if self.complete or self.failure:
            return self.state()
        original_ablation = dict(self.ablation)
        if ablation:
            self.ablation.update(ablation)
        try:
            self._scan_nearby()
            if not self.route or self.route[0] != self.rover:
                self.replan(reason="La ruta dejó de coincidir con la posición actual")
            if not self.route or len(self.route) < 2:
                self.failure = "No existe una ruta estimada factible."
                return self.state()
            observation = self._sensor_observation()
            current_cell, next_cell = self.rover, self.route[1]
            next_belief = self._belief_at(next_cell)
            if next_belief is None:
                self._scan_nearby(radius=6)
                next_belief = self._belief_at(next_cell)
            if next_belief and self._belief_blocked(next_cell):
                self.replan(reason="RNA o LIDAR detectó bloqueo en la siguiente celda")
                if not self.route or len(self.route) < 2:
                    self.failure = "Supervisor: celda de avance bloqueada y no hay desvío."
                    return self.state()
                next_cell = self.route[1]
                next_belief = self._belief_at(next_cell)
            hard_veto = self._true_blocked(*next_cell) and next_belief is not None
            if hard_veto:
                self._beliefs[next_cell] = {
                    **(next_belief or {}),
                    "blocked": True,
                    "terrain": self._terrain_kind(*next_cell),
                }
                self.events.append({
                    "step": self.steps, "type": "safety_veto",
                    "message": f"Supervisor: veto de seguridad antes de F{next_cell[0] + 1} C{next_cell[1] + 1}",
                })
                self.replan(reason="Supervisor vetó una celda bloqueada")
                if not self.route or len(self.route) < 2:
                    self.failure = "El supervisor detuvo el rover: siguiente celda bloqueada y sin desvío."
                    return self.state()
                next_cell = self.route[1]
                next_belief = self._belief_at(next_cell)

            probabilities = observation["probabilities"]
            neural_confidence = float(observation["confidence"])
            neural_hazard = (
                float(probabilities.get("barranco", 0.0))
                + float(probabilities.get("obstaculo", 0.0))
            ) if self.ablation["rna"] else 0.15
            local_risk = self._perceived_risk(next_cell)
            fuzzy = fuzzy_control(
                battery=self.battery if self.ablation["fuzzy"] else 100.0,
                hazard=max(local_risk, neural_hazard * 0.72),
                slope=self._slope(next_cell),
                gas=float(next_belief["gas"]) if next_belief else 0.0,
                neural_confidence=neural_confidence if self.ablation["rna"] else 1.0,
            ) if self.ablation["fuzzy"] else {
                "speed": 0.8, "caution": 0.0, "memberships": {},
                "rules": [], "aggregation_area": 0.0,
                "defuzzification": "ablación: control difuso desactivado",
                "fallback": None,
            }
            if "all" in self.faults or (
                ("lidar" in self.faults) and ("camera" in self.faults)
            ):
                self.failure = "Modo seguro: sensores críticos de navegación sin lecturas."
                self.events.append({
                    "step": self.steps, "type": "safe_mode",
                    "message": "Todos los sensores críticos fallaron; movimiento bloqueado.",
                })
                return self.state()
            if self.battery <= 0:
                self.failure = "Batería agotada; movimiento detenido."
                return self.state()

            energy_cost = 0.45 + 0.55 * fuzzy["speed"] + 0.25 * local_risk
            if self.battery < energy_cost:
                self.failure = "Reserva de batería insuficiente para el siguiente paso."
                return self.state()
            if fuzzy["speed"] < 0.06 and local_risk > 0.8:
                self.events.append({
                    "step": self.steps, "type": "safe_stop",
                    "message": "Control difuso ordenó detenerse ante riesgo elevado.",
                })
                self.failure = "Modo seguro: el controlador difuso ordenó detenerse."
                return self.state()

            previous_route = self.route
            self.rover = next_cell
            self.steps += 1
            self.distance_travelled += CELL_METERS
            self.battery = max(0.0, self.battery - energy_cost)
            self._scan_nearby()
            remaining_route = previous_route[1:]
            blocked_ahead = any(
                self._belief_blocked(cell)
                for cell in remaining_route[1:5]
            )
            severe_risk_ahead = any(
                self._perceived_risk(cell) >= 0.9
                for cell in remaining_route[1:4]
            )
            if blocked_ahead or severe_risk_ahead:
                self.replan(reason="Peligro detectado en ruta; replanteo genético")
            else:
                self.route = remaining_route
            if self.rover in self.science_sites:
                self.science_collected += 1
                self.science_sites.discard(self.rover)
                self.events.append({
                    "step": self.steps, "type": "science",
                    "message": "Muestra científica simulada recolectada.",
                })
            self.complete = self.rover == self.goal
            reason = (
                "RNA estima peligro; cautela elevada."
                if self.ablation["rna"] and neural_hazard > 0.55
                else f"Fitness mínimo entre {len(self.route_candidates)} candidatos."
            )
            action = {
                "type": "AVANZAR",
                "reason": reason,
                "distance_m": CELL_METERS,
                "energy_cost": round(energy_cost, 3),
                "risk": round(local_risk, 3),
                "speed": fuzzy["speed"],
            }
            decision = {
                "step": self.steps,
                "planet": self.planet,
                "position": {"row": self.rover[0], "col": self.rover[1]},
                "sensors": {
                    "features": dict(zip(FEATURE_NAMES, observation["features"])),
                    "failure_flags": sorted(self.faults),
                },
                "rna": {
                    "prediction": observation["prediction"],
                    "probabilities": observation["probabilities"],
                    "confidence": neural_confidence,
                    "enabled": self.ablation["rna"],
                },
                "fuzzy": fuzzy,
                "genetic": {
                    "route_candidates": [candidate.as_dict() for candidate in self.route_candidates],
                    "best_fitness_by_generation": self.genetic_history,
                    "enabled": self.ablation["genetic"],
                },
                "supervisor": {
                    "action": action,
                    "hard_constraints_checked": True,
                    "veto": None,
                    "reason": reason,
                },
                "state": {
                    "battery_pct": round(self.battery, 2),
                    "distance_travelled_m": self.distance_travelled,
                    "science_collected": self.science_collected,
                    "mission_complete": self.complete,
                },
            }
            self.last_decision = decision
            self.trace.append(decision)
            self.events.append({
                "step": self.steps,
                "type": "decision",
                "message": (
                    f"{action['type']} · RNA {observation['prediction']} "
                    f"({neural_confidence:.0%}) · velocidad difusa {fuzzy['speed']:.2f}"
                ),
            })
            if self.complete:
                self.events.append({
                    "step": self.steps, "type": "mission_complete",
                    "message": "Rover llegó al objetivo de misión.",
                })
            return self.state()
        finally:
            self.ablation = original_ablation

    def run_mission(
        self,
        *,
        use_rna: bool = True,
        use_fuzzy: bool = True,
        use_genetic: bool = True,
        max_steps: int = 250,
    ) -> dict[str, Any]:
        scenario = MissionEngine(self.seed, train_model=False)
        scenario.model, scenario.perception_metrics = self.model, self.perception_metrics
        scenario.objects = json.loads(json.dumps(self.objects))
        scenario.start, scenario.goal = self.start, self.goal
        scenario.planet = self.planet
        scenario.science_sites = set(self.science_sites)
        scenario.risk_weight = self.risk_weight
        scenario.gas_concentration = self.gas_concentration
        scenario.science_priority = self.science_priority
        scenario.strategy = self.strategy
        scenario.battery_initial = self.battery_initial
        scenario.battery = self.battery_initial
        scenario.rover = self.start
        scenario.genetic_population_size = self.genetic_population_size
        scenario.genetic_generations = self.genetic_generations
        scenario.ablation = {
            "rna": use_rna, "fuzzy": use_fuzzy, "genetic": use_genetic,
        }
        scenario._scan_nearby()
        scenario.replan()
        while not scenario.complete and not scenario.failure and scenario.steps < max_steps:
            scenario.step()
        return {
            "seed": self.seed,
            "success": scenario.complete,
            "failure": scenario.failure,
            "steps": scenario.steps,
            "distance_m": round(scenario.distance_travelled, 2),
            "battery_remaining_pct": round(scenario.battery, 2),
            "science_collected": scenario.science_collected,
            "safety_violations": sum(
                1 for event in scenario.events if event["type"] == "safety_veto"
            ),
            "replans": sum(1 for event in scenario.events if event["type"] == "replan"),
        }

    def run_ablation_experiment(self, seeds: int = 20) -> dict[str, Any]:
        seed_count = max(20, min(30, int(seeds)))
        policies = {
            "hibrido": (True, True, True),
            "sin_rna": (False, True, True),
            "sin_difuso": (True, False, True),
            "sin_ag": (True, True, False),
        }
        results: dict[str, Any] = {}
        for policy, (use_rna, use_fuzzy, use_genetic) in policies.items():
            rows = []
            for offset in range(seed_count):
                experiment = MissionEngine(self.seed + offset, train_model=False)
                experiment.model, experiment.perception_metrics = self.model, self.perception_metrics
                experiment.objects = json.loads(json.dumps(self.objects))
                experiment.start, experiment.goal = self.start, self.goal
                experiment.planet = self.planet
                experiment.science_sites = set(self.science_sites)
                experiment.risk_weight = self.risk_weight
                experiment.gas_concentration = self.gas_concentration
                experiment.science_priority = self.science_priority
                experiment.battery_initial = self.battery_initial
                experiment.battery = self.battery_initial
                experiment.genetic_population_size = 8
                experiment.genetic_generations = 5
                rows.append(experiment.run_mission(
                    use_rna=use_rna, use_fuzzy=use_fuzzy, use_genetic=use_genetic
                ))
            results[policy] = {
                "runs": rows,
                "success_rate_pct": round(
                    100 * sum(bool(row["success"]) for row in rows) / len(rows), 1
                ),
                "mean_distance_m": round(float(np.mean([row["distance_m"] for row in rows])), 2),
                "median_distance_m": round(float(np.median([row["distance_m"] for row in rows])), 2),
                "std_distance_m": round(float(np.std([row["distance_m"] for row in rows])), 2),
                "mean_battery_remaining_pct": round(
                    float(np.mean([row["battery_remaining_pct"] for row in rows])), 2
                ),
                "mean_science_collected": round(
                    float(np.mean([row["science_collected"] for row in rows])), 2
                ),
                "mean_safety_violations": round(
                    float(np.mean([row["safety_violations"] for row in rows])), 2
                ),
                "mean_replans": round(
                    float(np.mean([row["replans"] for row in rows])), 2
                ),
                "seed_range": [self.seed, self.seed + seed_count - 1],
            }
        return {
            "seed_count": seed_count,
            "policies": results,
            "genetic_search": {"population": 8, "generations": 5},
            "note": (
                "Experimento simulado; misma configuración y semillas para cada política. "
                "Para agilizar la comparación se usa el mismo presupuesto AG compacto "
                "(población 8, 5 generaciones) en las políticas que lo incluyen."
            ),
        }

    def export_json(self) -> str:
        return json.dumps({
            "configuration": {
                "seed": self.seed, "planet": self.planet, "start": self.start,
                "goal": self.goal, "battery_initial_pct": self.battery_initial,
                "risk_weight": self.risk_weight, "gas_concentration": self.gas_concentration,
                "strategy": self.strategy,
                "science_priority": self.science_priority,
                "objects": self.objects,
            },
            "perception_metrics": self.perception_metrics,
            "trace": self.trace,
            "events": self.events,
        }, ensure_ascii=False, indent=2)

    def export_csv(self) -> str:
        buffer = io.StringIO()
        fieldnames = [
            "step", "planet", "row", "col", *FEATURE_NAMES,
            "prediction", "p_libre", "p_obstaculo", "p_barranco", "confidence",
            "fuzzy_speed", "fuzzy_caution", "fuzzy_memberships", "fuzzy_rules",
            "route_candidates", "fitness_by_generation", "action", "risk",
            "battery_pct", "energy_cost", "distance_travelled_m",
            "science_collected", "reason",
        ]
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        for record in self.trace:
            position = record["position"]
            probabilities = record["rna"]["probabilities"]
            action = record["supervisor"]["action"]
            sensors = record["sensors"]["features"]
            writer.writerow({
                "step": record["step"],
                "planet": record["planet"],
                "row": position["row"],
                "col": position["col"],
                **sensors,
                "prediction": record["rna"]["prediction"],
                "confidence": record["rna"]["confidence"],
                "p_libre": probabilities.get("libre"),
                "p_obstaculo": probabilities.get("obstaculo"),
                "p_barranco": probabilities.get("barranco"),
                "fuzzy_speed": record["fuzzy"]["speed"],
                "fuzzy_caution": record["fuzzy"]["caution"],
                "fuzzy_memberships": json.dumps(record["fuzzy"]["memberships"], ensure_ascii=False),
                "fuzzy_rules": json.dumps(record["fuzzy"]["rules"], ensure_ascii=False),
                "route_candidates": json.dumps(record["genetic"]["route_candidates"], ensure_ascii=False),
                "fitness_by_generation": json.dumps(
                    record["genetic"]["best_fitness_by_generation"], ensure_ascii=False
                ),
                "action": action["type"],
                "risk": action["risk"],
                "battery_pct": record["state"]["battery_pct"],
                "energy_cost": action["energy_cost"],
                "distance_travelled_m": record["state"]["distance_travelled_m"],
                "science_collected": record["state"]["science_collected"],
                "reason": record["supervisor"]["reason"],
            })
        return buffer.getvalue()

    def state(self) -> dict[str, Any]:
        profile = PLANET_PROFILES[self.planet]
        belief_map = []
        for (row, col), belief in self._beliefs.items():
            belief_map.append({
                "row": row, "col": col,
                "kind": belief["terrain"],
                "prediction": belief["prediction"],
                "confidence": belief["confidence"],
                "blocked": belief["blocked"],
                "risk": round(self._perceived_risk((row, col)), 3),
            })
        candidates = [candidate.as_dict() for candidate in self.route_candidates]
        return {
            "seed": self.seed,
            "planet": self.planet,
            "planet_profile": profile,
            "start": {"row": self.start[0], "col": self.start[1]},
            "goal": {"row": self.goal[0], "col": self.goal[1]},
            "rover": {"row": self.rover[0], "col": self.rover[1]},
            "battery": round(self.battery, 2),
            "battery_initial": round(self.battery_initial, 2),
            "risk_weight": self.risk_weight,
            "gas_concentration": self.gas_concentration,
            "science_priority": self.science_priority,
            "strategy": self.strategy,
            "steps": self.steps,
            "distance_travelled_m": round(self.distance_travelled, 2),
            "science_collected": self.science_collected,
            "mission_complete": self.complete,
            "failure": self.failure,
            "objects": self.objects,
            "science_sites": [{"row": r, "col": c} for r, c in sorted(self.science_sites)],
            "beliefs": belief_map,
            "routes": {
                "selected": self.route,
                "candidates": candidates,
                "history": self.genetic_history,
            },
            "last_decision": self.last_decision,
            "perception_metrics": {
                **self.perception_metrics,
                "last_prediction": self.last_decision["rna"] if self.last_decision else None,
            },
            "events": self.events[-40:],
            "ablation": self.ablation,
            "faults": sorted(self.faults),
        }
