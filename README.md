# Monitor de postura y distancia

> ⚠️ **README provisional.** El proyecto está en desarrollo; la estructura, los parámetros y los umbrales pueden cambiar.

Proyecto final de **Visión por Ordenador I** (3º Ingeniería Matemática, Comillas ICAI, curso 2026/27).

**Autores:** Manuel O'Shanahan Delgado-Taramona y Lucía Raventós Gonzalvo.

---

## Resumen

Sistema de visión por ordenador **clásica** (sin deep learning) que vigila, con una webcam colocada de lado, la **postura** (cuello y espalda) y la **distancia a la pantalla** de una persona sentada frente al ordenador. Si la mala postura se mantiene, da avisos por voz para corregirla: *«endereza la espalda»*, *«posiciona tu cuello recto»*, *«aléjate»* o *«acércate»*.

El acceso está protegido por un **código de cuatro dígitos en Morse** que el usuario enseña a la cámara con tarjetas impresas.

### Motivación

Pasar muchas horas con la cabeza adelantada, la espalda encorvada o demasiado cerca de la pantalla provoca dolor cervical y lumbar y fatiga visual. Como suele ocurrir sin darse cuenta, un aviso en el momento funciona mejor que una recomendación general. El sistema está pensado para estudiantes y trabajadores de oficina, y en especial para personas con dolor de espalda crónico, en rehabilitación o con baja visión.

---

## Funcionamiento

La calibración de la cámara se hace **offline**, una vez, antes de ejecutar el sistema. Después, el programa pasa por tres fases:

1. **🔒 Bloqueo.** El usuario enseña cuatro tarjetas Morse, una por dígito. Si la secuencia es correcta, el sistema se desbloquea; si no, anuncia «código erróneo» y olvida la secuencia.
2. **🎯 Selección.** Con un rectángulo se marcan, una tras otra, las tres pegatinas de la persona (**E**, **C** y **H**) y el ordenador. Desde ese momento solo se siguen esas cuatro regiones.
3. **👁️ Vigilancia.** El sistema sigue a la persona y al ordenador en tiempo real, mide los ángulos de cuello y espalda y la distancia a la pantalla, y avisa por voz cuando hace falta.

En pantalla se muestran el vídeo, las *bounding boxes*, los puntos detectados, los ángulos, la distancia medida y los FPS.

<!-- TODO: añadir aquí el diagrama de bloques (assets/diagrama_bloques.png) -->

---

## Módulos

### 1. Calibración (offline)
Calibración con tablero de ajedrez (`cv2.findChessboardCorners`, `cv2.cornerSubPix`, `cv2.calibrateCamera`), siguiendo el modelo *pinhole* y la Práctica 2. Se obtienen la matriz intrínseca $K$, los coeficientes de distorsión $\mathbf{d} = (k_1, k_2, p_1, p_2, k_3)$ y el error RMS de reproyección. En ejecución, cada fotograma se corrige con `cv2.initUndistortRectifyMap` y `cv2.remap`.

| Parámetro | Valor |
|---|---|
| $f_x,\ f_y$ | _pendiente_ |
| $c_x,\ c_y$ | _pendiente_ |
| $\mathbf{d}$ | _pendiente_ |
| RMS | _pendiente_ |

### 2. Sistema de seguridad: código Morse
Cada dígito son 5 símbolos y va en una tarjeta blanca: un **punto** es un círculo negro y una **raya** es un rectángulo alargado.

| Dígito | Morse | Dígito | Morse |
|:-:|:-:|:-:|:-:|
| 1 | `·−−−−` | 6 | `−····` |
| 2 | `··−−−` | 7 | `−−···` |
| 3 | `···−−` | 8 | `−−−··` |
| 4 | `····−` | 9 | `−−−−·` |
| 5 | `·····` | 0 | `−−−−−` |

- **Detección:** escala de grises → suavizado gaussiano → umbral fijo → apertura morfológica → componentes conexas con área mínima → *bounding box* de cada símbolo.
- **Clasificación:** por la relación de aspecto $r = w/h$ (punto si $r \approx 1$, raya si $r \gg 1$). Los símbolos se leen de izquierda a derecha según $c_x = x + w/2$.
- **Decodificador:** una tarjeta solo se acepta si se lee igual durante varios fotogramas seguidos. Entre dos tarjetas la cámara tiene que quedar vacía un instante, para poder repetir dígitos (p. ej. `1123`). Se memorizan hasta 4 dígitos y se comparan con la contraseña. La secuencia se borra si el código es erróneo, con una tecla o tras un periodo largo sin actividad.

### 3. Selección y tracker
- Selección de las cuatro regiones con `cv2.selectROI`, con un texto en pantalla que indica qué seleccionar en cada paso.
- Cada región se sigue con **meanShift** sobre la retroproyección del histograma de tono (canal H de HSV), como en el Lab 5.
- Cada región tiene su propio **filtro de Kalman** con modelo de velocidad constante, con estado $\mathbf{x} = (x, y, v_x, v_y)^\top$. El filtro suaviza la posición y la predice durante oclusiones breves.
- Solo se aceptan detecciones dentro de una zona de búsqueda alrededor de la predicción, para no confundirse con el fondo ni con otras personas.

### 4. Análisis de la postura
Se usan tres pegatinas de colores muy saturados, de unos 3–4 cm: **E** en la oreja, **C** en la base del cuello (vértebra C7) y **H** en la cadera.

$$\theta_{CV} = \arctan\left(\frac{|y_E - y_C|}{|x_E - x_C|}\right), \qquad \theta_T = \arctan\left(\frac{|x_C - x_H|}{|y_C - y_H|}\right)$$

Un ángulo craneovertebral $\theta_{CV}$ por debajo de unos 50° indica cabeza adelantada. Una inclinación del tronco $\theta_T$ alta indica espalda encorvada. Los umbrales exactos se ajustarán experimentalmente.

### 5. Distancia al ordenador
El borde de la pantalla se localiza con **Canny + transformada de Hough** dentro de la región del ordenador. Para pasar de píxeles a centímetros se usa como escala una pegatina de tamaño conocido pegada en el ordenador:

$$d_{px} = |x_{ojo} - x_{pantalla}|, \qquad s = \frac{W_{ref}}{w_{ref}}, \qquad d = s \cdot d_{px}$$

### 6. Avisos por voz
Síntesis de voz sin conexión con [`pyttsx3`](https://pypi.org/project/pyttsx3/). Solo se avisa si el problema dura más de un tiempo mínimo; además hay un intervalo mínimo entre avisos e histéresis en los umbrales:

$$\text{aviso} \iff \left(\theta_{CV} < \theta_{CV}^{min} \ \lor\ \theta_T > \theta_T^{max} \ \lor\ d \notin [d_{min}, d_{max}]\right) \text{ durante } t > T_{persist}$$

---

## Ampliaciones previstas

- [ ] Avisos con instrucciones concretas de corrección, no solo una alarma.
- [ ] Filtro de Kalman para suavizar el seguimiento y tolerar oclusiones breves.
- [ ] Informe de la sesión: porcentaje de tiempo con buena postura y gráfica de los ángulos.
- [ ] Recordatorio de pausas tras un tiempo prolongado sentado (p. ej. 45 min).
- [ ] Bloqueo rápido con una tecla para volver a pedir el código.
- [ ] Ahorro energético: la cámara se enciende solo una fracción del tiempo (*duty cycle* ≈ 10 %).

---

## Materiales

| Elemento | Uso |
|---|---|
| Webcam | Entrada de vídeo, colocada de lado respecto al usuario |
| Tablero de ajedrez impreso | Calibración offline |
| Tarjetas Morse impresas (10) | Código de acceso |
| Pegatinas de colores | Oreja, cuello y cadera, y referencia de escala en el ordenador |
| Altavoz del ordenador | Avisos por voz |
| Persona auxiliar | Selecciona las pegatinas y el ordenador mientras el usuario está sentado |

---

## Instalación

```bash
git clone https://github.com/manueloshanahan/computer_vision_final_project.git
cd computer_vision_final_project
pip install opencv-python numpy pyttsx3
```

<!-- TODO: añadir requirements.txt / environment.yml con las versiones exactas -->

## Uso

Para comprobar que la cámara funciona:

```bash
python src/test.py
```

> Si la webcam integrada no es la de índice 1, cambia `camera_index` en `src/test.py` (normalmente `0`).

<!-- TODO: instrucciones para calibrar (script offline) y ejecutar el sistema completo -->

## Estructura del repositorio

```text
.
├── Lab_Project.pdf   # Enunciado
├── src/
│   └── test.py       # Prueba de la cámara
├── assets/           # Imágenes del README
├── template/         # Plantilla LaTeX para el informe
└── README.md
```

<!-- TODO: actualizar cuando estén los módulos (calibración, morse, tracker, postura, distancia, voz) -->

---

## Riesgos considerados

| Riesgo | Plan |
|---|---|
| Fondo con otras personas u objetos | Selección manual de regiones y búsqueda solo alrededor de la predicción de Kalman |
| Cambios de iluminación | El histograma de tono se aprende de la propia selección; si la luz cambia mucho, se vuelven a seleccionar las pegatinas |
| Pegatinas tapadas por ropa o pelo | Colocarlas en zonas visibles |
| Lecturas erróneas de las tarjetas | Exigir la misma lectura en varios fotogramas seguidos |

---

## Demo

_Próximamente: enlace al vídeo de demostración (con los FPS incrustados en los fotogramas)._
