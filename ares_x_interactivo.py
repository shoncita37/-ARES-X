"""ARES-X: editor interactivo de mapa y planificador de rutas para Google Colab."""

from dataclasses import dataclass
import heapq
import math

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import ipywidgets as widgets
from IPython.display import clear_output, display


MAP_SIZE = 20
CELL_METERS = 10
MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass
class Route:
    cells: list[tuple[int, int]]
    distance_m: float
    risk: float
    score: float


class AresXDashboard:
    def __init__(self) -> None:
        self.rocks: set[tuple[int, int]] = {
            (10, column) for column in range(MAP_SIZE) if column not in (4, 15)
        } | {(7, 7), (7, 8), (12, 12)}
        self.craters: set[tuple[int, int]] = {(8, 4), (14, 6)}
        self.rough: set[tuple[int, int]] = {(9, 3), (9, 5), (11, 8)}
        self.start = (18, 1)
        self.goal = (1, 18)
        self.rover = self.start
        self.steps_taken = 0
        self.distance_route: Route | None = None
        self.safe_route: Route | None = None
        self.active_route: Route | None = None
        self.message = "Mapa listo. Elige «Avance automático» para iniciar el rover."

        self.tool = widgets.Dropdown(
            options=[
                ("Piedra (bloquea el paso)", "rock"),
                ("Cráter (zona peligrosa)", "crater"),
                ("Terreno rocoso (transitable)", "rough"),
                ("Borrar elemento", "erase"),
                ("Reposicionar rover e inicio", "start"),
                ("Mover destino", "goal"),
            ],
            description="Herramienta:",
            layout=widgets.Layout(width="310px"),
        )
        self.row = widgets.BoundedIntText(
            value=10, min=1, max=MAP_SIZE, description="Fila:",
            layout=widgets.Layout(width="150px"),
        )
        self.column = widgets.BoundedIntText(
            value=10, min=1, max=MAP_SIZE, description="Columna:",
            layout=widgets.Layout(width="170px"),
        )
        self.object_button = widgets.Button(
            description="Aplicar en celda",
            button_style="info",
            icon="map-marker",
        )
        self.objective = widgets.ToggleButtons(
            options=[
                ("Menor distancia", "distance"),
                ("Más segura", "safe"),
            ],
            description="Priorizar:",
        )
        self.risk_weight = widgets.FloatSlider(
            value=0.35,
            min=0.0,
            max=1.0,
            step=0.05,
            description="Penalización de riesgo:",
            readout_format=".2f",
            continuous_update=False,
            layout=widgets.Layout(width="430px"),
        )
        self.plan_button = widgets.Button(
            description="Calcular y comparar rutas",
            button_style="success",
            icon="random",
        )
        self.step_button = widgets.Button(
            description="Avanzar un paso",
            button_style="primary",
            icon="arrow-right",
        )
        self.auto_play = widgets.Play(
            value=0,
            min=0,
            max=100000,
            step=1,
            interval=500,
            repeat=True,
            description="Avance automático",
        )
        self.speed = widgets.IntSlider(
            value=500,
            min=100,
            max=1500,
            step=100,
            description="Pausa (ms):",
            continuous_update=False,
            layout=widgets.Layout(width="320px"),
        )
        self.reset_button = widgets.Button(
            description="Reiniciar rover",
            icon="refresh",
        )
        self.clear_button = widgets.Button(
            description="Limpiar obstáculos",
            button_style="warning",
            icon="trash",
        )
        self.output = widgets.Output()

        self.object_button.on_click(self._apply_tool)
        self.plan_button.on_click(self._plan)
        self.step_button.on_click(self._step)
        self.reset_button.on_click(self._reset_rover)
        self.clear_button.on_click(self._clear_obstacles)
        self.auto_play.observe(self._auto_step, names="value")
        self.speed.observe(self._change_speed, names="value")
        self.objective.observe(self._change_strategy, names="value")
        self.risk_weight.observe(self._change_strategy, names="value")

        controls = widgets.VBox(
            [
                widgets.HTML(
                    "<h2>ARES-X · Editor de terreno y rutas</h2>"
                    "<p>Una celda representa 10 m. Las piedras y los cráteres "
                    "bloquean el paso; el terreno rocoso añade riesgo. El rover "
                    "recalcula su ruta desde su posición antes de cada movimiento.</p>"
                ),
                widgets.HBox([self.tool, self.row, self.column, self.object_button]),
                widgets.HBox([self.objective, self.risk_weight, self.plan_button]),
                widgets.HBox([self.auto_play, self.speed]),
                widgets.HBox([self.step_button, self.reset_button, self.clear_button]),
            ]
        )
        self.ui = widgets.VBox([controls, self.output])
        self._refresh_routes()
        self._render()

    def _inside(self, cell: tuple[int, int]) -> bool:
        return 0 <= cell[0] < MAP_SIZE and 0 <= cell[1] < MAP_SIZE

    def _risk_at(self, cell: tuple[int, int]) -> float:
        risk = 2.0 if cell in self.rough else 0.0
        for crater in self.craters:
            distance = abs(cell[0] - crater[0]) + abs(cell[1] - crater[1])
            if distance == 1:
                risk += 8.0
            elif distance == 2:
                risk += 3.0
        return risk

    def _find_route(self, risk_weight: float) -> Route | None:
        if self.rover in self.rocks or self.rover in self.craters:
            return None
        if self.goal in self.rocks or self.goal in self.craters:
            return None

        frontier: list[tuple[float, int, int]] = [(0.0, *self.rover)]
        costs = {self.rover: 0.0}
        previous: dict[tuple[int, int], tuple[int, int]] = {}

        while frontier:
            cost, row, column = heapq.heappop(frontier)
            current = (row, column)
            if cost > costs[current]:
                continue
            if current == self.goal:
                break

            for dr, dc in MOVES:
                neighbor = (row + dr, column + dc)
                if not self._inside(neighbor):
                    continue
                if neighbor in self.rocks or neighbor in self.craters:
                    continue
                next_cost = cost + 1.0 + risk_weight * self._risk_at(neighbor)
                if next_cost < costs.get(neighbor, math.inf):
                    costs[neighbor] = next_cost
                    previous[neighbor] = current
                    heapq.heappush(frontier, (next_cost, *neighbor))

        if self.goal not in costs:
            return None

        cells = [self.goal]
        while cells[-1] != self.rover:
            cells.append(previous[cells[-1]])
        cells.reverse()
        risk = sum(self._risk_at(cell) for cell in cells[1:])
        return Route(
            cells=cells,
            distance_m=(len(cells) - 1) * CELL_METERS,
            risk=risk,
            score=costs[self.goal],
        )

    def _refresh_routes(self) -> None:
        self.distance_route = self._find_route(risk_weight=0.0)
        self.safe_route = self._find_route(risk_weight=self.risk_weight.value)
        self.active_route = (
            self.safe_route if self.objective.value == "safe" else self.distance_route
        )

    def _apply_tool(self, _button: widgets.Button) -> None:
        cell = (self.row.value - 1, self.column.value - 1)
        tool = self.tool.value

        if tool in ("start", "goal") and (cell in self.rocks or cell in self.craters):
            self.message = "El inicio y el destino no pueden estar sobre un obstáculo."
            self._render()
            return
        if tool in ("rock", "crater", "rough") and cell in (
            self.start,
            self.goal,
            self.rover,
        ):
            self.message = "No se puede poner un obstáculo bajo el rover o un punto de misión."
            self._render()
            return

        if tool == "start":
            self.start = cell
            self.rover = cell
            self.steps_taken = 0
            self.message = f"Inicio del rover movido a fila {cell[0] + 1}, columna {cell[1] + 1}."
        elif tool == "goal":
            self.goal = cell
            self.message = f"Destino movido a fila {cell[0] + 1}, columna {cell[1] + 1}."
        else:
            self.rocks.discard(cell)
            self.craters.discard(cell)
            self.rough.discard(cell)
            if tool == "rock":
                self.rocks.add(cell)
                self.message = f"Piedra añadida en fila {cell[0] + 1}, columna {cell[1] + 1}."
            elif tool == "crater":
                self.craters.add(cell)
                self.message = f"Cráter añadido en fila {cell[0] + 1}, columna {cell[1] + 1}."
            elif tool == "rough":
                self.rough.add(cell)
                self.message = f"Terreno rocoso añadido en fila {cell[0] + 1}, columna {cell[1] + 1}."
            else:
                self.message = f"Celda fila {cell[0] + 1}, columna {cell[1] + 1} despejada."

        self._refresh_routes()
        self._render()

    def _plan(self, _button: widgets.Button) -> None:
        self._refresh_routes()
        if self.active_route is None:
            self.message = "No existe una ruta desde la posición actual. Revisa el mapa."
        else:
            selected = (
                "más segura" if self.objective.value == "safe" else "más corta"
            )
            self.message = f"Rutas recalculadas desde el rover. Prioridad: {selected}."
        self._render()

    def _step(self, _button: widgets.Button) -> None:
        if self.rover == self.goal:
            self.message = "Misión completada: el rover llegó al destino."
            self.auto_play.playing = False
            self._render()
            return

        self._refresh_routes()
        if self.active_route is None:
            self.message = "No hay ruta desde la posición actual; rover detenido."
            self.auto_play.playing = False
        else:
            self.rover = self.active_route.cells[1]
            self.steps_taken += 1
            self._refresh_routes()
            self.message = (
                f"Paso {self.steps_taken}: rover en "
                f"({self.rover[0] + 1}, {self.rover[1] + 1}). "
                f"Ruta recalculada desde cada posición."
            )
            if self.rover == self.goal:
                self.message += " Misión completada."
                self.auto_play.playing = False
        self._render()

    def _auto_step(self, _change: dict) -> None:
        if self.auto_play.playing:
            self._step(self.step_button)

    def _change_strategy(self, _change: dict) -> None:
        self._refresh_routes()
        self.message = "Prioridad actualizada; rutas recalculadas desde la posición del rover."
        self._render()

    def _change_speed(self, change: dict) -> None:
        self.auto_play.interval = change["new"]

    def _reset_rover(self, _button: widgets.Button) -> None:
        self.rover = self.start
        self.steps_taken = 0
        self._refresh_routes()
        self.message = "Rover devuelto al inicio."
        self._render()

    def _clear_obstacles(self, _button: widgets.Button) -> None:
        self.rocks.clear()
        self.craters.clear()
        self.rough.clear()
        self._refresh_routes()
        self.message = "Se quitaron todas las piedras, cráteres y zonas rocosas."
        self._render()

    def _draw_map(self) -> None:
        terrain = np.zeros((MAP_SIZE, MAP_SIZE), dtype=int)
        for cell in self.rough:
            terrain[cell] = 1
        for cell in self.rocks:
            terrain[cell] = 2
        for cell in self.craters:
            terrain[cell] = 3

        fig, ax = plt.subplots(figsize=(8, 7))
        cmap = ListedColormap(["#e9dfc7", "#d9a441", "#626b73", "#a76d55"])
        ax.imshow(terrain, cmap=cmap, vmin=0, vmax=3, origin="upper")

        for route, color, label in (
            (self.distance_route, "#2574a9", "Menor distancia"),
            (self.safe_route, "#2d8a57", "Más segura"),
        ):
            if route:
                rows = [cell[0] for cell in route.cells]
                columns = [cell[1] for cell in route.cells]
                ax.plot(columns, rows, color=color, linewidth=2, alpha=0.8, label=label)

        for crater in self.craters:
            circle = plt.Circle((crater[1], crater[0]), 1.35, color="#e15b64", alpha=0.17)
            ax.add_patch(circle)
            ax.text(crater[1], crater[0], "C", ha="center", va="center", weight="bold")
        for rock in self.rocks:
            ax.text(rock[1], rock[0], "P", ha="center", va="center", color="white", weight="bold")

        ax.scatter(self.start[1], self.start[0], marker="s", s=100, c="#2864dc", label="Inicio")
        ax.scatter(self.goal[1], self.goal[0], marker="*", s=180, c="#ef9b20", label="Destino")
        ax.scatter(self.rover[1], self.rover[0], marker="o", s=110, c="#e53935", edgecolors="white", label="Rover")
        ax.set_xticks(range(MAP_SIZE))
        ax.set_yticks(range(MAP_SIZE))
        ax.set_xticklabels(range(1, MAP_SIZE + 1))
        ax.set_yticklabels(range(1, MAP_SIZE + 1))
        ax.set_xticks(np.arange(-0.5, MAP_SIZE, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, MAP_SIZE, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.7)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_xlabel("Columna (10 m por celda)")
        ax.set_ylabel("Fila")
        ax.set_title("Mapa de navegación ARES-X")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    def _route_summary(self) -> str:
        if self.distance_route is None and self.safe_route is None:
            return "Pulsa «Calcular y comparar rutas» para medir los caminos disponibles."

        def line(name: str, route: Route | None) -> str:
            if route is None:
                return f"<li><b>{name}:</b> no hay ruta.</li>"
            return (
                f"<li><b>{name}:</b> {route.distance_m:.0f} m · "
                f"{len(route.cells) - 1} pasos · riesgo acumulado {route.risk:.1f}</li>"
            )

        chosen = "más segura" if self.objective.value == "safe" else "menor distancia"
        return (
            "<b>Comparación de rutas:</b><ul>"
            + line("Menor distancia", self.distance_route)
            + line("Más segura", self.safe_route)
            + f"</ul>Objetivo seleccionado: <b>{chosen}</b>. "
            f"El peso de riesgo configurado es {self.risk_weight.value:.2f}; "
            "se penalizan celdas cercanas a cráteres y terreno rocoso."
        )

    def _render(self) -> None:
        with self.output:
            clear_output(wait=True)
            display(widgets.HTML(f"<b>Estado:</b> {self.message}"))
            self._draw_map()
            display(widgets.HTML(self._route_summary()))


dashboard = AresXDashboard()
display(dashboard.ui)
