"""
Tracker del proyecto: Monitor de postura y distancia.

Sigue las 3 pegatinas de la persona (E, C y H), seleccionadas a mano, con el
método del Lab 5 D: histograma del tono (canal H de HSV), retroproyección y
meanShift. Cada región tiene su propio filtro de Kalman de velocidad constante y una zona de búsqueda: una medida de meanShift solo se acepta si cae
dentro de la elipse de confianza de la predicción (distancia de Mahalanobis).

El ordenador no se mueve, así que solo se selecciona una vez su ROI y se queda fijo
(no se sigue con meanShift: un portátil gris o negro no tiene tono que seguir).

Uso:
    python tracker.py
Teclas: q / ESC = salir, r = volver a seleccionar las regiones.
"""

import os
import time
import cv2
import numpy as np

# ---------------------------------------------------------------- parámetros
CAMERA_INDEX = 0
WIDTH, HEIGHT = 1280, 720          # resolución de trabajo del proyecto
CALIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibracion.npz")  # junto a este archivo
CALIB_WIDTH = 1920                 # ancho de las fotos de calibración

REGIONS = ["E (oreja)", "C (cuello C7)", "H (cadera)"]        # pegatinas que se siguen
COLORS = [(0, 255, 0), (0, 255, 255), (0, 64, 255)]           # color del rectángulo dibujado (BGR)
PC_NAME = "Ordenador"                                         # ROI fijo, sin tracker
PC_COLOR = (255, 0, 255)

S_MIN = 60          # saturación mínima: los píxeles grises/blancos no cuentan (¡las pegatinas blancas no sirven!)
V_MIN = 40          # brillo mínimo: los píxeles muy oscuros tampoco
HIST_BINS = 30      # número de bins del histograma de tono (0-180)
BP_ON = 50          # un píxel de la retroproyección cuenta como "del color de la pegatina" si supera este valor
MIN_FILL = 0.35     # la medida se acepta si la ventana tiene al menos el 35 % del color que tenía al seleccionarla
SEARCH_MIN = 1.0    # margen de la zona de búsqueda alrededor de la predicción, en tamaños de pegatina (mín.)
SEARCH_MAX = 3.0    # (máx.): la zona crece con la incertidumbre de Kalman, pero nunca más que esto
GATE = 9.21         # cuantil 0.99 de una chi-cuadrado con 2 g.l.: si se supera, es un cambio brusco de postura


# ---------------------------------------------------------------- calibración
def load_undistort_maps(path=CALIB_FILE, width=WIDTH, height=HEIGHT, calib_width=CALIB_WIDTH):
    """Carga K y dist, escala K a la resolución de trabajo y calcula los mapas de corrección."""
    data = np.load(path)
    dist = data["dist"]
    s = width / calib_width
    K = data["K"].copy()
    K[0, 0] *= s; K[1, 1] *= s   # fx, fy
    K[0, 2] *= s; K[1, 2] *= s   # cx, cy
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K, (width, height), cv2.CV_16SC2)
    return map1, map2


# ---------------------------------------------------------------- tracker de una región
class RegionTracker:
    """meanShift sobre la retroproyección del tono + filtro de Kalman de velocidad constante."""

    def __init__(self, name, color, frame_hsv, roi):
        self.name = name
        self.color = color
        x, y, w, h = roi
        self.w, self.h = w, h
        self.window = (x, y, w, h)
        self.found = True
        self.lost = 0                                   # fotogramas seguidos sin encontrarla

        # Histograma del tono de la selección (como en el Lab 5 D). La máscara quita los
        # píxeles poco saturados o muy oscuros, cuyo tono no es fiable.
        crop = frame_hsv[y:y + h, x:x + w]
        mask = ((crop[:, :, 1] >= S_MIN) & (crop[:, :, 2] >= V_MIN)).astype(np.uint8) * 255
        self.hist = cv2.calcHist([crop], [0], mask, [HIST_BINS], [0, 180])
        cv2.normalize(self.hist, self.hist, 0, 255, cv2.NORM_MINMAX)

        # Proporción de píxeles "del color" en la selección: referencia para validar medidas
        bp0 = self._backproject(frame_hsv)[y:y + h, x:x + w]
        self.fill0 = max(float((bp0 > BP_ON).mean()), 1e-3)
        if (mask > 0).mean() < 0.05:
            print(f"Aviso: la región {name} casi no tiene color (¿pegatina blanca, negra o gris?). "
                  "No se podrá seguir; usad una pegatina de color vivo.")

        # Filtro de Kalman: estado (x, y, vx, vy), medida (x, y)
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)       # A (velocidad constante)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], np.float32)      # C (se mide la posición)
        self.kf.processNoiseCov = np.diag([1.0, 1.0, 10.0, 10.0]).astype(np.float32)  # Q
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 4.0       # R
        cx, cy = x + w / 2, y + h / 2
        self.kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 10.0
        self.center = (cx, cy)
        self.last_good = (cx, cy)                       # última posición en la que se vio de verdad

    def _backproject(self, frame_hsv):
        bp = cv2.calcBackProject([frame_hsv], [0], self.hist, [0, 180], 1)
        sat_ok = (frame_hsv[:, :, 1] >= S_MIN) & (frame_hsv[:, :, 2] >= V_MIN)
        bp[~sat_ok] = 0
        return bp

    def _fill(self, bp, cx, cy):
        """Proporción de píxeles del color en una ventana del tamaño de la pegatina centrada en (cx, cy)."""
        H_img, W_img = bp.shape
        x = int(np.clip(cx - self.w / 2, 0, W_img - self.w))
        y = int(np.clip(cy - self.h / 2, 0, H_img - self.h))
        win = bp[y:y + self.h, x:x + self.w]
        return float((win > BP_ON).mean()) if win.size else 0.0

    def update(self, frame_hsv):
        """Predice, busca con meanShift en la zona de búsqueda y valida la medida."""
        H_img, W_img = frame_hsv.shape[:2]
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

        # 1) Predicción de Kalman
        pred = self.kf.predict()
        px, py = float(pred[0, 0]), float(pred[1, 0])

        # 2) Zona de búsqueda: crece con la incertidumbre (+-3 sigma), pero acotada entre
        #    SEARCH_MIN y SEARCH_MAX tamaños de pegatina para que no se vaya a otra parte de la imagen
        Cm = self.kf.measurementMatrix
        S = Cm @ self.kf.errorCovPre @ Cm.T + self.kf.measurementNoiseCov
        rx = float(np.clip(3 * np.sqrt(S[0, 0]), SEARCH_MIN * self.w, SEARCH_MAX * self.w))
        ry = float(np.clip(3 * np.sqrt(S[1, 1]), SEARCH_MIN * self.h, SEARCH_MAX * self.h))
        sw, sh = int(min(self.w + 2 * rx, W_img)), int(min(self.h + 2 * ry, H_img))
        x0 = int(np.clip(px - sw / 2, 0, W_img - sw))
        y0 = int(np.clip(py - sh / 2, 0, H_img - sh))

        # 3) Medida en dos pasos: meanShift con la ventana grande (encuentra la pegatina en la
        #    zona) y luego con una ventana del tamaño de la pegatina (centra la medida en ella)
        bp = self._backproject(frame_hsv)
        _, (x, y, w, h) = cv2.meanShift(bp, (x0, y0, sw, sh), term_crit)
        cx, cy = x + w / 2, y + h / 2
        x1 = int(np.clip(cx - self.w / 2, 0, W_img - self.w))
        y1 = int(np.clip(cy - self.h / 2, 0, H_img - self.h))
        _, (x, y, w, h) = cv2.meanShift(bp, (x1, y1, self.w, self.h), term_crit)
        mx, my = x + w / 2, y + h / 2

        # 4) Validación por apariencia: ¿hay en la ventana tanto color como al seleccionarla?
        ok = self._fill(bp, mx, my) >= MIN_FILL * self.fill0

        if ok:
            d = np.array([[mx - px], [my - py]], np.float32)
            D2 = (d.T @ np.linalg.inv(S) @ d).item()      # distancia de Mahalanobis
            if D2 <= GATE:
                self.kf.correct(np.array([[mx], [my]], np.float32))
            else:
                # Movimiento brusco (p.ej. encorvarse de golpe): el modelo de velocidad constante
                # no lo esperaba, pero la pegatina está ahí -> se reinicia el filtro en la medida
                self.kf.statePost = np.array([[mx], [my], [0], [0]], np.float32)
                self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 10.0
            self.found, self.lost = True, 0
            self.center = (float(self.kf.statePost[0, 0]), float(self.kf.statePost[1, 0]))
            self.last_good = self.center
        else:
            # No se ve (tapada, girada...): se queda quieta en la última posición buena, sin
            # velocidad, y la zona de búsqueda se agranda poco a poco hasta SEARCH_MAX
            self.found = False
            self.lost += 1
            gx, gy = self.last_good
            self.kf.statePost = np.array([[gx], [gy], [0], [0]], np.float32)
            self.kf.errorCovPost = self.kf.errorCovPre.copy()
            self.center = self.last_good

        self.window = (int(self.center[0] - self.w / 2), int(self.center[1] - self.h / 2), self.w, self.h)
        return self.center, self.found

    def draw(self, frame):
        x, y, w, h = self.window
        thickness = 2 if self.found else 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), self.color, thickness)
        cv2.circle(frame, (int(self.center[0]), int(self.center[1])), 4, self.color, -1)
        label = self.name if self.found else self.name + " (buscando)"
        cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color, 2)


# ---------------------------------------------------------------- selección de regiones
def select_regions(frame, names):
    """Pide con selectROI las regiones indicadas, una tras otra, con un texto que indica cuál toca."""
    rois = []
    for name in names:
        view = frame.copy()
        cv2.putText(view, f"Selecciona: {name}  (ENTER = aceptar, c = repetir)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        roi = cv2.selectROI("Seleccion", view, showCrosshair=True, fromCenter=False)
        while roi[2] == 0 or roi[3] == 0:          # selección vacía: repetir
            roi = cv2.selectROI("Seleccion", view, showCrosshair=True, fromCenter=False)
        rois.append(tuple(int(v) for v in roi))
    cv2.destroyWindow("Seleccion")
    return rois


def create_trackers(frame):
    """Selecciona las 3 pegatinas (con tracker) y el ROI fijo del ordenador."""
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    rois = select_regions(frame, REGIONS + [PC_NAME])
    trackers = [RegionTracker(n, c, frame_hsv, r) for n, c, r in zip(REGIONS, COLORS, rois[:3])]
    pc_roi = rois[3]
    return trackers, pc_roi


def draw_pc_roi(frame, pc_roi):
    x, y, w, h = pc_roi
    cv2.rectangle(frame, (x, y), (x + w, y + h), PC_COLOR, 2)
    cv2.putText(frame, PC_NAME, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, PC_COLOR, 2)


# ---------------------------------------------------------------- programa de prueba
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    if not cap.isOpened():
        print(f"No se puede abrir la cámara (índice {CAMERA_INDEX}).")
        return

    map1, map2 = load_undistort_maps()

    ret, frame = cap.read()
    if not ret:
        print("No se recibe imagen de la cámara.")
        return
    frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
    trackers, pc_roi = create_trackers(frame)

    t_prev = time.time()
    fps = 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for t in trackers:
            t.update(frame_hsv)
            t.draw(frame)
        draw_pc_roi(frame, pc_roi)

        # FPS (obligatorio que aparezca en el vídeo final)
        t_now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Tracker", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        if key == ord('r'):
            trackers, pc_roi = create_trackers(frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
