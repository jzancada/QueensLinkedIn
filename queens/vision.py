"""Deteccion del tablero de Queens en una captura, con OpenCV.

El pipeline registra cada etapa intermedia (imagen + notas) mientras trabaja.
Ese registro no es depuracion opcional: es lo que alimenta el panel de vision,
donde se ve como se genera la entrada digital a partir del PNG.

Cuando algo falla, `Detection.error` explica que fallo y `stages` conserva todo
lo avanzado hasta ese punto. Nunca se devuelve un tablero a medias: un tablero
corrupto en silencio seria el peor resultado posible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .board import Board

# Umbral de gris por debajo del cual un pixel se considera "linea" del tablero.
# El marco es casi negro y la pagina esta por encima de 200, asi que hay margen.
DARK_THRESHOLD = 100

# Un contorno candidato debe tener al menos estos agujeros para ser el tablero.
# Es el filtro decisivo: en las capturas de doc/ el tablero tiene 81 y el
# siguiente candidato (el chrome del navegador) se queda en 12.
MIN_HOLES = 20

# Distancia maxima en el espacio Lab de 8 bits de OpenCV para considerar que dos
# celdas son de la misma region. En las capturas de doc/ el muestreo da colores
# identicos dentro de una region, y el par de regiones mas parecido (morado y
# rosa) esta a 23.4: hay que quedarse por debajo de esa distancia, con margen
# suficiente para absorber la variacion que introduce un reescalado.
COLOR_TOLERANCE = 15


@dataclass
class Stage:
    """Una etapa intermedia del pipeline, lista para mostrarse."""

    name: str
    image: np.ndarray          # BGR
    notes: list[str] = field(default_factory=list)


@dataclass
class Detection:
    board: Board | None
    stages: list[Stage]
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.board is not None


class _Recorder:
    """Acumula las etapas. Desactivado no copia nada, para no gastar memoria."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.stages: list[Stage] = []

    def add(self, name: str, image: np.ndarray, notes: list[str] | None = None) -> None:
        if not self.enabled:
            return
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        self.stages.append(Stage(name, image.copy(), list(notes or ())))


def _label(img: np.ndarray, text: str, org: tuple[int, int],
           color: tuple[int, int, int]) -> None:
    """Texto con reborde blanco, legible sobre cualquier fondo."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def _cluster_1d(values: np.ndarray, gap: float) -> list[float]:
    """Agrupa valores proximos y devuelve el centro de cada grupo, ordenado."""
    centers: list[float] = []
    group: list[float] = []
    for v in np.sort(values):
        if group and v - group[-1] > gap:
            centers.append(float(np.mean(group)))
            group = []
        group.append(float(v))
    if group:
        centers.append(float(np.mean(group)))
    return centers


def _dominant_color(roi: np.ndarray) -> tuple[int, int, int]:
    """Color mas frecuente del ROI, cuantizado a 4 bits por canal.

    No se usa la media: si la celda ya tiene una reina o una cruz dibujada, la
    media queda sesgada por el glifo. La moda es inmune a cualquier dibujo que
    ocupe menos de la mitad del area.
    """
    q = (roi.reshape(-1, 3) >> 4).astype(np.int32)
    key = (q[:, 0] << 8) | (q[:, 1] << 4) | q[:, 2]
    dominant = int(np.bincount(key).argmax())
    mask = key == dominant
    return tuple(int(v) for v in roi.reshape(-1, 3)[mask].mean(axis=0))


def _group_by_color(colors: list[tuple[int, int, int]],
                    tolerance: float) -> tuple[list[int], list[tuple[int, int, int]]]:
    """Agrupa colores por cercania en Lab. Devuelve (id por celda, color por id).

    Nearest-centroid incremental en vez de k-means: los colores son planos y no
    se sabe de antemano cuantas regiones hay (comprobar que salen N es
    justamente una de las validaciones).
    """
    swatches = np.array(colors, dtype=np.uint8).reshape(-1, 1, 3)
    lab = cv2.cvtColor(swatches, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)

    ids: list[int] = []
    centroids: list[np.ndarray] = []
    members: list[list[int]] = []
    for i, point in enumerate(lab):
        if centroids:
            distances = [np.linalg.norm(point - c) for c in centroids]
            best = int(np.argmin(distances))
            if distances[best] <= tolerance:
                ids.append(best)
                members[best].append(i)
                centroids[best] = lab[members[best]].mean(axis=0)
                continue
        ids.append(len(centroids))
        centroids.append(point.copy())
        members.append([i])

    palette = [tuple(int(v) for v in np.array(colors)[m].mean(axis=0)) for m in members]
    return ids, palette


def _find_board(dark: np.ndarray, img: np.ndarray, rec: _Recorder):
    """Localiza el contorno del marco exterior. Devuelve (indice, contornos, jerarquia).

    Se apoya en RETR_CCOMP: como las lineas interiores tocan el marco, el tablero
    entero es un unico contorno de nivel 0 y sus celdas son sus agujeros. Contar
    agujeros es lo que lo distingue de cualquier otro cuadrado oscuro.
    """
    contours, hierarchy = cv2.findContours(dark, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    hierarchy = hierarchy[0]

    holes_of = {}
    for i, h in enumerate(hierarchy):
        if h[3] != -1:
            holes_of[h[3]] = holes_of.get(h[3], 0) + 1

    min_area = 0.02 * img.shape[0] * img.shape[1]
    overlay = img.copy()
    survivors, notes = [], []

    for i, h in enumerate(hierarchy):
        if h[3] != -1:                       # solo contornos de nivel 0
            continue
        contour = contours[i]
        area = cv2.contourArea(contour)
        if area < 500:                       # ruido, ni se comenta
            continue
        x, y, w, hh = cv2.boundingRect(contour)
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        aspect = w / hh
        solidity = area / (w * hh)
        holes = holes_of.get(i, 0)

        reason = None
        if area < min_area:
            reason = f"area {area:.0f} < 2% de la imagen"
        elif len(approx) != 4:
            reason = f"{len(approx)} vertices, no 4"
        elif not cv2.isContourConvex(approx):
            reason = "no convexo"
        elif not 0.9 <= aspect <= 1.1:
            reason = f"aspect ratio {aspect:.2f} fuera de [0.9, 1.1]"
        elif solidity <= 0.9:
            reason = f"solidez {solidity:.2f} <= 0.90"
        elif holes < MIN_HOLES:
            reason = f"solo {holes} agujeros, se piden {MIN_HOLES}"

        color = (0, 0, 255) if reason else (0, 170, 0)
        cv2.drawContours(overlay, [approx], -1, color, 2)
        _label(overlay, reason or f"CANDIDATO ok, {holes} agujeros", (x + 4, max(14, y - 6)), color)
        notes.append(f"({x},{y}) {w}x{hh}  AR={aspect:.2f} solidez={solidity:.2f} "
                     f"agujeros={holes}  ->  {reason or 'ACEPTADO'}")

        if reason is None:
            survivors.append((area, i))

    rec.add("4. Contornos candidatos", overlay,
            ["Cada contorno de nivel 0 con su veredicto:", *notes])

    if not survivors:
        return None, contours, hierarchy
    return max(survivors)[1], contours, hierarchy


def detect(img: np.ndarray, debug: bool = True) -> Detection:
    """Detecta el tablero de Queens en una imagen BGR."""
    rec = _Recorder(debug)

    def fail(msg: str) -> Detection:
        return Detection(None, rec.stages, msg)

    if img is None or img.ndim != 3:
        return fail("La imagen no es BGR valida.")

    rec.add("1. Original", img, [f"{img.shape[1]} x {img.shape[0]} px"])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(gray, DARK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    rec.add("2. Mascara de oscuros", dark,
            [f"Pixeles con gris < {DARK_THRESHOLD}: {int(dark.sum() // 255)}",
             "El marco es casi negro; la pagina esta por encima de 200."])

    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    rec.add("3. Tras MORPH_CLOSE", dark,
            ["Cierra los huecos que deja el antialias de las lineas,",
             "para que el marco sea un contorno continuo."])

    board_idx, contours, hierarchy = _find_board(dark, img, rec)
    if board_idx is None:
        return fail("Ningun contorno parece un tablero (ver etapa 4).")

    bx, by, bw, bh = cv2.boundingRect(contours[board_idx])
    frame = img.copy()
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 170, 0), 2)
    rec.add("5. Cuadrado exterior", frame,
            [f"Marco en ({bx},{by}), {bw}x{bh} px"])

    rec.add("6. Recorte del tablero", img[by:by + bh, bx:bx + bw],
            ["El tablero aislado del resto de la captura."])

    # --- celdas: son los agujeros del contorno del marco --------------------
    holes = [contours[i] for i, h in enumerate(hierarchy) if h[3] == board_idx]
    if not holes:
        return fail("El marco no contiene celdas.")

    rects = np.array([cv2.boundingRect(c) for c in holes])
    areas = rects[:, 2] * rects[:, 3]
    rects = rects[areas > 0.3 * np.median(areas)]       # descarta astillas del antialias

    n = int(round(len(rects) ** 0.5))
    if n * n != len(rects) or n < 4:
        return fail(f"{len(rects)} celdas detectadas: no es un cuadrado NxN plausible.")

    cx = rects[:, 0] + rects[:, 2] / 2
    cy = rects[:, 1] + rects[:, 3] / 2
    gap = (cx.max() - cx.min()) / n * 0.5
    col_centers = _cluster_1d(cx, gap)
    row_centers = _cluster_1d(cy, gap)
    if len(col_centers) != n or len(row_centers) != n:
        return fail(f"La rejilla no es {n}x{n}: "
                    f"{len(col_centers)} columnas y {len(row_centers)} filas.")

    cells = np.zeros((n, n, 4), dtype=np.int32)
    for (x, y, w, h) in rects:
        col = int(np.argmin([abs(x + w / 2 - c) for c in col_centers]))
        row = int(np.argmin([abs(y + h / 2 - c) for c in row_centers]))
        cells[row, col] = (x, y, w, h)
    if (cells[:, :, 2] == 0).any():
        return fail("Hay posiciones de la rejilla sin celda asignada.")

    grid = img.copy()
    for row in range(n):
        for col in range(n):
            x, y, w, h = cells[row, col]
            cv2.rectangle(grid, (x, y), (x + w, y + h), (255, 0, 255), 1)
    rec.add("7. Rejilla de celdas", grid,
            [f"N = {n} (de {len(rects)} agujeros del marco)",
             f"Lado medio de celda: {cells[:, :, 2].mean():.1f} px",
             "Las celdas salen de los agujeros del contorno, no de una",
             "division aritmetica: cada una es la que OpenCV encontro."])

    # --- color de cada celda ------------------------------------------------
    sampling = img.copy()
    colors: list[tuple[int, int, int]] = []
    for row in range(n):
        for col in range(n):
            x, y, w, h = cells[row, col]
            mx, my = int(w * 0.2), int(h * 0.2)
            roi = img[y + my:y + h - my, x + mx:x + w - mx]
            color = _dominant_color(roi)
            colors.append(color)
            cv2.rectangle(sampling, (x + mx, y + my), (x + w - mx, y + h - my), (0, 0, 0), 1)
            cv2.rectangle(sampling, (x + mx + 1, y + my + 1),
                          (x + mx + 10, y + my + 10), color, -1)
    rec.add("8. Muestreo de color", sampling,
            ["ROI del 60% central de cada celda, para no coger el borde.",
             "Se toma la moda cuantizada, no la media: asi una reina o una",
             "cruz ya dibujada no altera el color de la celda."])

    # --- regiones -----------------------------------------------------------
    ids, palette = _group_by_color(colors, COLOR_TOLERANCE)
    region = np.array(ids, dtype=np.int32).reshape(n, n)

    regions_img = img.copy()
    for row in range(n):
        for col in range(n):
            x, y, w, h = cells[row, col]
            _label(regions_img, str(region[row, col]),
                   (x + w // 2 - 4, y + h // 2 + 5), (0, 0, 0))
    rec.add("9. Regiones", regions_img,
            [f"{len(palette)} regiones agrupadas en Lab (tolerancia {COLOR_TOLERANCE}).",
             *[f"  region {i}: BGR {c}" for i, c in enumerate(palette)]])

    board = Board(n=n, region=region, colors=palette, cells=cells)

    if len(palette) != n:
        return fail(f"Se esperaban {n} regiones y salieron {len(palette)}.")
    if not board.regions_are_connected():
        return fail("Alguna region no es conexa: el agrupamiento de colores ha fallado.")

    borders = img.copy()
    for row in range(n):
        for col in range(n):
            x, y, w, h = cells[row, col]
            for drow, dcol, p0, p1 in (
                (-1, 0, (x, y), (x + w, y)),
                (1, 0, (x, y + h), (x + w, y + h)),
                (0, -1, (x, y), (x, y + h)),
                (0, 1, (x + w, y), (x + w, y + h)),
            ):
                if board.is_border(row, col, drow, dcol):
                    cv2.line(borders, p0, p1, (0, 0, 0), 3)
                else:
                    cv2.line(borders, p0, p1, (160, 160, 160), 1)
    rec.add("10. Fronteras", borders,
            ["Trazo grueso donde cambia de region, fino donde no.",
             "Es la entrada final del modelo, y lo que hace que el tablero",
             "se lea como el de LinkedIn y no como una rejilla uniforme."])

    return Detection(board, rec.stages)


def detect_file(path: str, debug: bool = True) -> Detection:
    """Detecta el tablero de un fichero de imagen."""
    img = cv2.imread(str(path))
    if img is None:
        return Detection(None, [], f"No se pudo leer la imagen: {path}")
    return detect(img, debug)
