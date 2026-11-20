"""
Medidas de postura y distancia.

- Ángulos con los centros de las pegatinas E (oreja), C (cuello C7) y H (cadera):
      theta_CV = arctan(|yE - yC| / |xE - xC|)     (ángulo craneovertebral)
      theta_T  = arctan(|xC - xH| / |yC - yH|)     (inclinación del tronco)
- Borde de la pantalla: Canny + HoughLinesP (Tema 4, Sesión 4 B) dentro del ROI fijo del
  ordenador. Se calcula una vez al seleccionar, porque el portátil no se mueve.
- Escala: pegatina de color de tamaño conocido en el portátil. Su ancho en píxeles se
  mide con una umbralización de la saturación + componentes conexas.
      s = W_ref / w_ref,   d = s * d_px,   d_px = |x_E - x_pantalla|
"""

import math
import cv2
import numpy as np

W_REF_CM = 4.0            # ancho real de la pegatina de escala (MEDIDLO con regla)
S_MIN = 60                # saturación mínima para considerar que un píxel es de la pegatina
CANNY_LOW, CANNY_HIGH = 50, 150
HOUGH_THRESHOLD = 40      # votos mínimos en el acumulador
MIN_LINE_FRAC = 0.4       # longitud mínima del segmento, como fracción de la altura del ROI
MAX_GAP = 10
MAX_TILT_DEG = 35         # la pantalla se busca entre las líneas casi verticales (+-35 grados)


# ---------------------------------------------------------------- ángulos
def posture_angles(E, C, H):
    """E, C, H = (x, y) en píxeles. Devuelve (theta_CV, theta_T) en grados."""
    xE, yE = E
    xC, yC = C
    xH, yH = H
    theta_cv = math.degrees(math.atan2(abs(yE - yC), abs(xE - xC)))
    theta_t = math.degrees(math.atan2(abs(xC - xH), abs(yC - yH)))
    return theta_cv, theta_t


# ---------------------------------------------------------------- borde de la pantalla
def detect_screen_edge(frame_bgr, pc_roi, person_x):
    """
    Busca con Canny + HoughLinesP la línea casi vertical del ROI del ordenador más cercana a
    la persona (el borde de la pantalla que mira hacia ella).
    Devuelve el segmento (x1, y1, x2, y2) en coordenadas de la imagen completa, o None.
    También devuelve la imagen de bordes (para el informe).
    """
    x, y, w, h = pc_roi
    roi = frame_bgr[y:y + h, x:x + w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, HOUGH_THRESHOLD,
                            minLineLength=int(MIN_LINE_FRAC * h), maxLineGap=MAX_GAP)
    if lines is None:
        return None, edges

    best, best_dist = None, None
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        tilt = math.degrees(math.atan2(abs(x2 - x1), abs(y2 - y1)))   # 0 = vertical
        if tilt > MAX_TILT_DEG:
            continue
        seg = (x1 + x, y1 + y, x2 + x, y2 + y)                          # a coordenadas de imagen
        dist = abs((seg[0] + seg[2]) / 2 - person_x)
        if best is None or dist < best_dist:
            best, best_dist = seg, dist
    return best, edges


def screen_x_at(segment, y_query):
    """x del borde de la pantalla a la altura y_query (recta que pasa por el segmento)."""
    x1, y1, x2, y2 = segment
    if y2 == y1:
        return (x1 + x2) / 2
    return x1 + (y_query - y1) * (x2 - x1) / (y2 - y1)


# ---------------------------------------------------------------- escala
def measure_scale(frame_bgr, ref_roi, w_ref_cm=W_REF_CM):
    """
    Mide el ancho en píxeles de la pegatina de escala dentro de ref_roi y devuelve
    s = W_ref / w_ref (cm por píxel), o None si no se encuentra.
    """
    x, y, w, h = ref_roi
    hsv = cv2.cvtColor(frame_bgr[y:y + h, x:x + w], cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 1] >= S_MIN).astype(np.uint8) * 255                # umbral de saturación
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None, mask
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))                  # la componente más grande
    w_ref_px = stats[i, cv2.CC_STAT_WIDTH]
    if w_ref_px < 5:
        return None, mask
    return w_ref_cm / w_ref_px, mask


def distance_cm(E, screen_segment, scale):
    """d = s * |x_E - x_pantalla|, con x_pantalla a la altura de E."""
    if screen_segment is None or scale is None:
        return None
    x_screen = screen_x_at(screen_segment, E[1])
    return scale * abs(E[0] - x_screen)
