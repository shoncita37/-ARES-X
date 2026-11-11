"""Colab notebook dashboard; execute after the ARES-X engine code cell."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import FileLink, clear_output, display


mission = MissionEngine(seed=42)
map_output = widgets.Output()
status_output = widgets.Output()
rna_output = widgets.Output()
fuzzy_output = widgets.Output()
genetic_output = widgets.Output()
log_output = widgets.Output()
experiment_output = widgets.Output()


def draw_mission() -> None:
    state = mission.state()
    terrain = np.zeros((MAP_SIZE, MAP_SIZE), dtype=int)
    palette = {
        "rock": 1, "cliff": 1, "crater": 2, "rough": 3,
        "gas": 4, "science": 5, "meteorite": 2,
    }
    for item in state["objects"]:
        terrain[item["row"], item["col"]] = palette.get(item["kind"], 0)
    for site in state["science_sites"]:
        terrain[site["row"], site["col"]] = 5

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(
        terrain,
        origin="upper",
        cmap=plt.matplotlib.colors.ListedColormap(
            ["#c9b991", "#626b73", "#9d5949", "#d0a25a", "#78acc4", "#91c5df"]
        ),
        vmin=0,
        vmax=5,
    )
    for candidate in state["routes"]["candidates"]:
        color = "#277db0" if candidate["name"] == "Menor distancia" else "#315d35"
        points = candidate["cells"]
        ax.plot(
            [cell[1] for cell in points],
            [cell[0] for cell in points],
            color=color,
            linewidth=1.6,
            alpha=0.72,
        )
    for belief in state["beliefs"]:
        if belief["risk"] > 0.45:
            ax.scatter(
                belief["col"], belief["row"], s=130,
                facecolors="none", edgecolors="#dd705b", linewidths=1.0,
            )
    ax.scatter(state["start"]["col"], state["start"]["row"], marker="s", s=95, c="#3679d3", label="Inicio")
    ax.scatter(state["goal"]["col"], state["goal"]["row"], marker="*", s=190, c="#edb34e", label="Destino")
    ax.scatter(state["rover"]["col"], state["rover"]["row"], marker="o", s=110, c="#e85f49", edgecolors="white", label="Rover")
    ax.set_xticks(range(MAP_SIZE))
    ax.set_yticks(range(MAP_SIZE))
    ax.set_xlabel("Columna · 10 m por celda")
    ax.set_ylabel("Fila")
    ax.set_title(f"ARES-X · {state['planet_profile']['name']} · paso {state['steps']}")
    ax.grid(color="white", alpha=0.35, linewidth=0.6)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    with map_output:
        clear_output(wait=True)
        display(fig)
        plt.close(fig)


def render_diagnostics() -> None:
    state = mission.state()
    decision = state["last_decision"]
    with status_output:
        clear_output(wait=True)
        routes = state["routes"]["candidates"]
        display(widgets.HTML(
            f"<b>Estado:</b> {state['failure'] or ('Misión completada' if state['mission_complete'] else 'Explorando')} · "
            f"<b>Posición:</b> {state['rover']} · <b>Batería:</b> {state['battery']:.1f}% · "
            f"<b>Distancia:</b> {state['distance_travelled_m']:.0f} m · "
            f"<b>Muestras:</b> {state['science_collected']} · "
            f"<b>Rutas candidatas:</b> {len(routes)}"
        ))
    if decision:
        with rna_output:
            clear_output(wait=True)
            display(widgets.HTML(
                f"<b>MLP 8–32–16–3</b> · predicción: {decision['rna']['prediction']} · "
                f"confianza: {decision['rna']['confidence']:.1%} · "
                f"entrenamiento: {'convergió' if state['perception_metrics']['converged'] else 'llegó al límite'} "
                f"({state['perception_metrics']['training_iterations']} iteraciones) · "
                f"prueba F1 macro: {state['perception_metrics']['metrics']['test']['macro_f1']:.3f} · "
                f"recall barranco: {state['perception_metrics']['metrics']['test']['hazard_recall']:.3f}"
            ))
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
            names = list(decision["rna"]["probabilities"])
            axes[0].bar(names, [decision["rna"]["probabilities"][name] for name in names], color=["#7aaed1", "#d59664", "#b96b5d"])
            axes[0].set_ylim(0, 1)
            axes[0].set_title("Probabilidad RNA")
            axes[1].plot(state["perception_metrics"]["loss_curve"], color="#739a54")
            axes[1].set_title("Pérdida de entrenamiento")
            axes[1].set_xlabel("Iteración")
            fig.tight_layout()
            display(fig)
            plt.close(fig)
    with fuzzy_output:
        clear_output(wait=True)
        if decision:
            fuzzy = decision["fuzzy"]
            display(widgets.HTML(
                f"<b>Mamdani min–max · centroide:</b> velocidad {fuzzy['speed']:.3f} · "
                f"cautela {fuzzy['caution']:.3f} · reglas activas {len(fuzzy['rules'])}"
            ))
            for rule in fuzzy["rules"]:
                display(widgets.HTML(
                    f"<code>{rule['id']} · activación {rule['activation']:.2f}</code> — {rule['text']}"
                ))
            display(widgets.HTML(f"<b>Membresías:</b> {json.dumps(fuzzy['memberships'], ensure_ascii=False)}"))
    with genetic_output:
        clear_output(wait=True)
        display(widgets.HTML(
            f"<b>AG:</b> población 32 · generaciones {len(state['routes']['history'])} · "
            f"selección por torneo · cruce · mutación · elitismo"
        ))
        for candidate in state["routes"]["candidates"]:
            display(widgets.HTML(
                f"<b>{candidate['name']}</b> · fitness {candidate['fitness']:.3f} · "
                f"distancia {candidate['distance_m']:.0f} m · riesgo {candidate['risk']:.2f} · "
                f"energía {candidate['energy']:.1f}% · ciencia {candidate['science']:.0f}"
            ))
        if state["routes"]["history"]:
            fig, ax = plt.subplots(figsize=(8, 2.4))
            ax.plot(
                [item["generation"] for item in state["routes"]["history"]],
                [item["best_fitness"] for item in state["routes"]["history"]],
                color="#6e9950",
            )
            ax.set_xlabel("Generación")
            ax.set_ylabel("Mejor fitness")
            ax.grid(alpha=0.2)
            fig.tight_layout()
            display(fig)
            plt.close(fig)
    with log_output:
        clear_output(wait=True)
        for event in reversed(state["events"][-14:]):
            display(widgets.HTML(
                f"<code>PASO {event['step']} · {event['type']}</code> — {event['message']}"
            ))


def redraw() -> None:
    draw_mission()
    render_diagnostics()


planet_control = widgets.Dropdown(
    options=[
        ("Marte · Rover", "mars"),
        ("Mercurio · Rover", "mercury"),
        ("Tierra · Rover", "earth"),
        ("Venus · Rover hipotético", "venus"),
        ("Júpiter · Sonda virtual", "jupiter"),
        ("Saturno · Sonda virtual", "saturn"),
        ("Urano · Sonda virtual", "uranus"),
        ("Neptuno · Sonda virtual", "neptune"),
    ],
    description="Planeta:",
)
tool_control = widgets.Dropdown(
    options=[
        ("Piedra", "rock"), ("Cráter", "crater"), ("Barranco", "cliff"),
        ("Gas", "gas"), ("Terreno rocoso", "rough"), ("Lugar científico", "science"),
    ],
    description="Elemento:",
)
row_control = widgets.BoundedIntText(value=9, min=1, max=MAP_SIZE, description="Fila:")
col_control = widgets.BoundedIntText(value=9, min=1, max=MAP_SIZE, description="Columna:")
risk_control = widgets.FloatSlider(
    value=mission.risk_weight, min=0, max=1.5, step=0.05, description="Peso riesgo:",
    readout_format=".2f", continuous_update=False,
)
science_control = widgets.FloatSlider(
    value=mission.science_priority, min=0, max=1, step=0.05, description="Prioridad científica:",
    readout_format=".2f", continuous_update=False,
)
battery_control = widgets.IntSlider(
    value=100, min=0, max=100, step=1, description="Batería inicial:",
    continuous_update=False,
)
strategy_control = widgets.ToggleButtons(
    options=[("Menor riesgo", "safe"), ("Menor distancia", "distance")],
    description="Estrategia:",
)
apply_object_button = widgets.Button(description="Añadir al mapa", button_style="info")
step_button = widgets.Button(description="Un paso", button_style="primary")
reset_button = widgets.Button(description="Reiniciar")
auto_play = widgets.Play(
    value=0, min=0, max=100000, step=1, interval=600, repeat=True,
    description="Play automático",
)
event_buttons = [
    widgets.Button(description="Tormenta"),
    widgets.Button(description="Sismo"),
    widgets.Button(description="Impacto"),
    widgets.Button(description="Noche"),
]
fault_control = widgets.Dropdown(
    options=[
        ("Cámara", "camera"), ("LIDAR", "lidar"),
        ("Sensor de gas", "gas"), ("Todos", "all"),
    ],
    description="Fallo:",
)
fault_button = widgets.Button(description="Inyectar fallo", button_style="warning")
restore_button = widgets.Button(description="Restaurar sensores")
export_button = widgets.Button(description="Exportar CSV y JSON")
experiment_button = widgets.Button(
    description="Comparar 20 semillas y ablaciones",
    button_style="success",
)


def _apply_config(_change=None) -> None:
    mission.configure(
        risk_weight=risk_control.value,
        science_priority=science_control.value,
        strategy=strategy_control.value,
    )
    if mission.steps == 0:
        mission.configure(battery=battery_control.value)
    redraw()


def _apply_planet(change) -> None:
    if change["name"] == "value":
        mission.set_planet(change["new"])
        redraw()


def _add_object(_button) -> None:
    try:
        mission.add_object(
            tool_control.value, row_control.value - 1, col_control.value - 1
        )
        redraw()
    except ValueError as error:
        with status_output:
            clear_output(wait=True)
            display(widgets.HTML(f"<b>Validación:</b> {error}"))


def _step(_button=None) -> None:
    mission.step()
    redraw()


def _auto_step(change) -> None:
    if change["name"] == "value":
        _step()


def _reset(_button=None) -> None:
    global mission
    mission = MissionEngine(seed=mission.seed)
    planet_control.value = "mars"
    redraw()


def _inject(index: int) -> None:
    event_names = ("storm", "quake", "meteor", "night")
    mission.inject_event(event_names[index])
    redraw()


def _inject_fault(_button=None) -> None:
    mission.set_fault(fault_control.value)
    redraw()


def _restore(_button=None) -> None:
    for fault in list(mission.faults):
        mission.set_fault(fault, enabled=False)
    redraw()


def _export(_button=None) -> None:
    Path("ares_x_traza.csv").write_text(mission.export_csv(), encoding="utf-8")
    Path("ares_x_mision.json").write_text(mission.export_json(), encoding="utf-8")
    with log_output:
        display(FileLink("ares_x_traza.csv"), FileLink("ares_x_mision.json"))


def _experiment(_button=None) -> None:
    experiment_button.disabled = True
    experiment_button.description = "Ejecutando 20 semillas…"
    with experiment_output:
        clear_output(wait=True)
        display(widgets.HTML("<b>Experimento en curso.</b> El cálculo se ejecuta en CPU."))
    try:
        results = mission.run_ablation_experiment(seeds=20)
        with experiment_output:
            clear_output(wait=True)
            rows = [
                f"<tr><td>{name}</td><td>{value['success_rate_pct']}%</td>"
                f"<td>{value['mean_distance_m']} ± {value['std_distance_m']} m</td>"
                f"<td>{value['mean_battery_remaining_pct']}%</td>"
                f"<td>{value['mean_safety_violations']}</td></tr>"
                for name, value in results["policies"].items()
            ]
            display(widgets.HTML(
                "<table><thead><tr><th>Política</th><th>Éxito</th>"
                "<th>Distancia media ± DE</th><th>Batería media</th>"
                "<th>Vetos de seguridad medios</th></tr></thead>"
                f"<tbody>{''.join(rows)}</tbody></table>"
                f"<p>{results['note']} · {results['seed_count']} semillas.</p>"
            ))
    finally:
        experiment_button.disabled = False
        experiment_button.description = "Comparar 20 semillas y ablaciones"


planet_control.observe(_apply_planet, names="value")
risk_control.observe(_apply_config, names="value")
science_control.observe(_apply_config, names="value")
battery_control.observe(_apply_config, names="value")
strategy_control.observe(_apply_config, names="value")
apply_object_button.on_click(_add_object)
step_button.on_click(_step)
reset_button.on_click(_reset)
auto_play.observe(_auto_step, names="value")
for index, button in enumerate(event_buttons):
    button.on_click(lambda _button, selected=index: _inject(selected))
fault_button.on_click(_inject_fault)
restore_button.on_click(_restore)
export_button.on_click(_export)
experiment_button.on_click(_experiment)

mission_tab = widgets.VBox([
    widgets.HBox([planet_control, strategy_control]),
    widgets.HBox([tool_control, row_control, col_control, apply_object_button]),
    widgets.HBox([battery_control, risk_control, science_control]),
    widgets.HBox([auto_play, step_button, reset_button]),
    widgets.HBox(event_buttons),
    widgets.HBox([fault_control, fault_button, restore_button, export_button]),
    status_output,
    map_output,
])
tabs = widgets.Tab(children=[
    mission_tab, rna_output, fuzzy_output, genetic_output,
    widgets.VBox([log_output, experiment_button, experiment_output]),
])
for index, title in enumerate(("Misión", "Laboratorio RNA", "Lógica difusa", "Algoritmo genético", "Bitácora y experimentos")):
    tabs.set_title(index, title)
redraw()
display(tabs)
