"""Modelo del tablero: regiones, colores, fronteras y estado de juego."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class Mark(IntEnum):
    """Lo que el jugador ha puesto en una celda."""

    EMPTY = 0
    CROSS = 1
    QUEEN = 2


@dataclass
class Board:
    """Un tablero de Queens ya digitalizado.

    `region` guarda el id de region de cada celda y es lo unico que necesita el
    solver. `colors` y `cells` vienen de la deteccion y solo los usa la interfaz:
    los colores para pintar y los rectangulos para que el inspector del panel de
    vision pueda traducir un punto de la imagen a una celda.
    """

    n: int
    region: np.ndarray                      # (n, n) int, id de region por celda
    colors: list[tuple[int, int, int]]      # BGR por region
    cells: np.ndarray | None = None         # (n, n, 4) rect x,y,w,h en la imagen
    marks: np.ndarray = field(init=False)   # (n, n) Mark

    def __post_init__(self) -> None:
        self.marks = np.full((self.n, self.n), Mark.EMPTY, dtype=np.uint8)

    def is_border(self, row: int, col: int, drow: int, dcol: int) -> bool:
        """¿Hay frontera de region entre esta celda y la vecina indicada?

        Fuera del tablero cuenta como frontera: el borde exterior se dibuja
        grueso igual que las fronteras interiores.
        """
        r, c = row + drow, col + dcol
        if not (0 <= r < self.n and 0 <= c < self.n):
            return True
        return bool(self.region[row, col] != self.region[r, c])

    def region_sizes(self) -> np.ndarray:
        """Numero de celdas de cada region, indexado por id."""
        return np.bincount(self.region.ravel(), minlength=len(self.colors))

    def regions_are_connected(self) -> bool:
        """¿Toda region forma una sola pieza conexa (vecindad de 4)?

        Una region partida en dos trozos delata que el agrupamiento de colores
        ha unido regiones distintas que casualmente se parecen.
        """
        for rid in range(len(self.colors)):
            cells = np.argwhere(self.region == rid)
            if len(cells) == 0:
                return False
            pending = [tuple(cells[0])]
            seen = {tuple(cells[0])}
            while pending:
                r, c = pending.pop()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nb = (r + dr, c + dc)
                    if (0 <= nb[0] < self.n and 0 <= nb[1] < self.n
                            and nb not in seen and self.region[nb] == rid):
                        seen.add(nb)
                        pending.append(nb)
            if len(seen) != len(cells):
                return False
        return True
