"""
Avisos por voz y lógica de cuándo avisar.

Se avisa de una regla solo si:
  - la condición mala se mantiene durante t > T_PERSIST segundos, y
  - han pasado al menos T_COOLDOWN segundos desde el último aviso de esa regla.
Cada regla tiene histéresis: entra en "mal" al cruzar un umbral y solo vuelve a "bien" al
cruzar otro un poco más exigente, para que no parpadee cuando el valor está en el límite.

    aviso <=> (theta_CV < theta_CV_min  o  theta_T > theta_T_max  o  d fuera de [d_min, d_max])
              durante t > T_PERSIST

La voz (pyttsx3) va en un hilo aparte para que el vídeo no se congele mientras habla.
"""

import queue
import threading
import time

T_PERSIST = 5.0      # s que tiene que durar la mala postura antes de avisar
T_COOLDOWN = 15.0    # s mínimos entre dos avisos de la misma regla


# ---------------------------------------------------------------- voz en un hilo aparte
class Voice:
    def __init__(self):
        self.q = queue.Queue()
        self.busy = False
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            import pyttsx3
            engine = pyttsx3.init()          # el motor se crea en el mismo hilo que lo usa
            for v in engine.getProperty("voices"):   # voz en español si la hay
                if "spanish" in v.name.lower() or "es" in str(v.languages).lower() or "es-" in v.id.lower():
                    engine.setProperty("voice", v.id)
                    break
            engine.setProperty("rate", 170)
        except Exception as e:                # sin pyttsx3: los avisos solo se imprimen
            print("Aviso: voz no disponible (", e, ")")
            engine = None
        while True:
            text = self.q.get()
            self.busy = True
            if engine is not None:
                engine.say(text)
                engine.runAndWait()
            else:
                print("[VOZ]", text)
            self.busy = False

    def say(self, text):
        if not self.busy and self.q.empty():  # no se acumulan avisos mientras habla
            self.q.put(text)


# ---------------------------------------------------------------- reglas con histéresis
class Rule:
    """
    is_bad(v) y is_good(v) definen la histéresis: la regla pasa a "mal" cuando is_bad(v)
    y solo vuelve a "bien" cuando is_good(v).
    """

    def __init__(self, name, message, is_bad, is_good):
        self.name, self.message = name, message
        self.is_bad, self.is_good = is_bad, is_good
        self.bad = False
        self.bad_since = None
        self.last_alert = -1e9

    def update(self, value, now):
        """Devuelve el mensaje si toca avisar, si no None. value=None -> sin medida."""
        if value is None:
            self.bad_since = None if not self.bad else self.bad_since
            return None
        if not self.bad and self.is_bad(value):
            self.bad, self.bad_since = True, now
        elif self.bad and self.is_good(value):
            self.bad, self.bad_since = False, None
        if (self.bad and self.bad_since is not None
                and now - self.bad_since >= T_PERSIST
                and now - self.last_alert >= T_COOLDOWN):
            self.last_alert = now
            return self.message
        return None


def default_rules(cv_min=50.0, t_max=20.0, d_min=50.0, d_max=70.0, margin=3.0):
    """Umbrales iniciales; se ajustan probando posturas buenas y malas."""
    return {
        "cuello": Rule("cuello", "Posiciona tu cuello recto",
                       lambda v: v < cv_min, lambda v: v > cv_min + margin),
        "espalda": Rule("espalda", "Endereza la espalda",
                        lambda v: v > t_max, lambda v: v < t_max - margin),
        "cerca": Rule("cerca", "Aléjate de la pantalla",
                      lambda v: v < d_min, lambda v: v > d_min + 2),
        "lejos": Rule("lejos", "Acércate a la pantalla",
                      lambda v: v > d_max, lambda v: v < d_max - 2),
    }


class AlertManager:
    def __init__(self, voice=None, rules=None):
        self.voice = voice
        self.rules = rules or default_rules()
        self.last_message, self.last_message_t = "", -1e9
        # estadísticas de la sesión (ampliación: informe final)
        self.t_good = 0.0
        self.t_total = 0.0
        self.history = []        # (t, theta_CV, theta_T, d)
        self._t_prev = None

    def update(self, theta_cv, theta_t, d, now=None):
        now = time.time() if now is None else now
        values = {"cuello": theta_cv, "espalda": theta_t, "cerca": d, "lejos": d}
        for key, rule in self.rules.items():
            msg = rule.update(values[key], now)
            if msg:
                self.last_message, self.last_message_t = msg, now
                if self.voice:
                    self.voice.say(msg)
        # tiempo con buena postura (solo cuando hay medidas)
        if self._t_prev is not None and theta_cv is not None and theta_t is not None:
            dt = now - self._t_prev
            self.t_total += dt
            if not self.bad_posture():
                self.t_good += dt
        self._t_prev = now
        self.history.append((now, theta_cv, theta_t, d))

    def bad_posture(self):
        return self.rules["cuello"].bad or self.rules["espalda"].bad

    def bad_distance(self):
        return self.rules["cerca"].bad or self.rules["lejos"].bad

    def good_ratio(self):
        """eta = t_postura_correcta / t_total * 100 %"""
        return 100.0 * self.t_good / self.t_total if self.t_total > 0 else None
