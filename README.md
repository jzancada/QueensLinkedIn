# Queens de LinkedIn — visión, solver y paso a paso

Lee un tablero del juego **Queens** de LinkedIn desde un PNG o una captura de
pantalla, lo detecta con **OpenCV**, lo resuelve por **backtracking** y deja
jugarlo.

El objetivo no es solo resolverlo: es **verlo resolver**. Nada debe ser una caja
negra, así que el proyecto tiene dos paneles observables — uno para ver cómo se
genera la entrada digital a partir de la imagen, y otro para ver el backtracking
avanzar y retroceder paso a paso sobre el tablero.

## Estado

| Parte | Estado |
|---|---|
| Detección con OpenCV (`queens/vision.py`) | funcionando sobre las tres capturas de `doc/` |
| Modelo de tablero (`queens/board.py`) | hecho |
| Solver por backtracking | pendiente |
| Interfaz PySide6 (3 pestañas) | pendiente |

## Cómo funciona la detección

El tablero es un marco negro grueso sobre fondo claro, y las líneas interiores
tocan el marco: con `RETR_CCOMP` el tablero entero sale como **un único contorno**
y **sus celdas son sus agujeros**. Contar agujeros es lo que lo distingue de
cualquier otro cuadrado oscuro de la captura — y hace falta, porque el contorno
más grande de la imagen **no** es el tablero, sino el chrome del navegador, que
pasa los filtros de forma (aspect ratio 1.08, solidez 1.00, 4 vértices) y solo cae
por tener 12 agujeros en vez de 81.

El color de cada celda se toma como **moda cuantizada** del 60 % central, no como
media: así una reina o una cruz ya dibujada no altera el color. Las regiones se
agrupan por cercanía en Lab, y el resultado se valida antes de devolverse — deben
salir exactamente N regiones y cada una debe ser conexa.

## Instalación

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Capturas de prueba

`doc/Example1..3.png` son el mismo tablero 9×9 recortado de tres formas
distintas. La prueba de la detección es que las tres produzcan **la misma**
matriz de regiones. Están saneadas: sin avatar ni badge de notificaciones.

## Reglas del juego

1. Exactamente una reina por fila.
2. Exactamente una por columna.
3. Exactamente una por región de color.
4. Dos reinas no pueden tocarse, **ni siquiera en diagonal** — pero sí pueden
   compartir diagonal a distancia ≥ 2. Esta es la diferencia con el problema
   clásico de las N reinas.
