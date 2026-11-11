# ARES-X · Simulador híbrido de exploración

ARES-X es un demostrador académico de navegación autónoma. Une una interfaz de
edición y visualización 3D con un motor de misión en Python: una MLP para
percepción, un controlador difuso Mamdani, planificación genética y un
supervisor que verifica las restricciones de seguridad antes de cada avance.
Todos los planetas, sensores, distancias y eventos son simulados; no son
parámetros de ingeniería ni resultados de una misión real.

## Requisitos

- Python 3.10 o posterior.
- Los paquetes listados en `requirements.txt`.
- Conexión a Internet para cargar Three.js, OrbitControls y las fuentes de la
  interfaz 3D. El motor y la API se ejecutan localmente.

## Iniciar la aplicación 3D

En PowerShell, desde la carpeta del proyecto:

```powershell
py -m pip install -r requirements.txt
py ares_x_server.py
```

Abre <http://127.0.0.1:8000/>. La primera carga entrena la MLP y puede tardar
unos segundos. La interfaz consulta el motor Python por la API local; no se
debe abrir el HTML como archivo ni usar `http.server`.

En el mapa puedes rotar y acercar la cámara, añadir o arrastrar piedras,
cráteres, gas, rocas, barrancos y muestras, y mover el inicio y la estrella de
destino. **Iniciar misión** activa la navegación autónoma; también puedes
avanzar paso a paso. Los cambios de mapa, los eventos y los peligros detectados
provocan un replanteo. La interfaz muestra telemetría, rutas candidatas,
lecturas de sensores, métricas de train/validación/prueba, pérdida de la RNA,
reglas difusas, evolución del AG, bitácora y resultados experimentales.

El mapa incluye relieve erosionado, textura superficial procedural y modelos
visuales más detallados para piedras, cráteres, nubes de gas y barrancos. Esos
detalles de elevación y los 720 fragmentos de roca distribuidos por la escena
son ilustrativos. El rover visual tiene seis ruedas, suspensión articulada y
ruedas que giran durante el avance; su movimiento no es una simulación física.
La ruta calcula el riesgo a partir de los elementos y sensores de la
simulación, no de la altura geométrica del terreno.

En Júpiter, Saturno, Urano y Neptuno se usa una cuadrícula virtual de sonda,
no una superficie sólida. El fondo representa el Sol y varios planetas; la
Luna aparece orbitando la Tierra al seleccionarla, en el lado opuesto al Sol,
con una textura de cráteres. La distancia visual del Sol aumenta según el
planeta seleccionado, pero las distancias se comprimen y los tamaños se
ajustan para que sean visibles junto al mapa: no están a escala astronómica.
Los cuerpos del fondo son decorativos y no participan en la navegación.

## Despliegue público

El frontend está preparado para Netlify: publicar la raíz del repositorio
sirve `ares_x_3d.html` en `/` y reenvía `/api/*` a la API de Render. El archivo
`_redirects` usa `https://ares-x-api.onrender.com`; si Render asigna otra URL,
actualiza ese destino antes de publicar el sitio.
El archivo `_headers` añade controles básicos de seguridad para las respuestas
estáticas.

Para publicar el backend, crea un Blueprint en Render desde este repositorio y
usa `render.yaml`. El servicio `ares-x-api` instala `requirements.txt` y ejecuta
Flask con Gunicorn. Luego conecta el mismo repositorio a Netlify con directorio
base `.` y directorio de publicación `.`; no hace falta comando de build.
Render en el plan gratuito puede suspender el servicio cuando está inactivo,
por lo que la primera solicitud posterior puede tardar en responder.

Las mutaciones de la API comprueban `Origin` para reducir ataques CSRF desde
otros sitios; agrega otros dominios de frontend en Render mediante
`ARES_ALLOWED_ORIGINS`, separados por comas. Esto no es autenticación: la API
permanece pública y la simulación usa estado compartido entre visitantes. No
envíes datos privados ni uses este backend como servicio de producción.

## Notebook autónomo de Google Colab

Sube `ARES_X_Colab_Hibrido.ipynb` a Google Colab y ejecuta sus celdas en orden.
El notebook contiene el mismo motor y el panel interactivo; no depende de
archivos locales del proyecto. Incluye Play/Pausa, edición del mapa,
configuración de la misión, diagnóstico de algoritmos, eventos, fallos,
exportaciones CSV/JSON y comparación reproducible de políticas.

`ARES_X_Interactivo.ipynb` y `ares_x_interactivo.py` son prototipos anteriores;
para la versión híbrida usa el notebook nuevo.

## Modelo y evaluación

- La MLP usa ocho rasgos de sensores y arquitectura 8–32–16–3. Los mapas se
  separan entre entrenamiento, validación y prueba para evitar compartir
  escenarios. Se informan exactitud, F1 macro, recall de barranco, matriz de
  confusión, curva de pérdida y estado de convergencia.
- El controlador difuso implementa reglas interpretables Mamdani min–max y
  defuzzificación por centroide.
- El algoritmo genético evoluciona rutas candidatas con selección por torneo,
  cruce, mutación y elitismo. Se muestran fitness, distancia, riesgo, energía
  y prioridad científica.
- El supervisor bloquea celdas peligrosas conocidas y detiene la misión ante
  fallos críticos o condiciones simuladas de seguridad.
- La comparación usa las mismas semillas y condiciones para el sistema
  híbrido y las ablaciones sin RNA, sin lógica difusa y sin AG. El botón ejecuta
  20 semillas por política; las variantes con AG comparten un presupuesto
  experimental compacto de 8 individuos × 5 generaciones para mantener
  razonable el tiempo de ejecución. Los resultados describen esta simulación,
  e incluyen éxito, distancia media y desviación estándar, batería, muestras y
  vetos medios del supervisor; no son una validación de hardware.

## Pruebas

```powershell
py -m unittest discover -v -p "test_*.py"
```

## Archivos principales

- `ares_x_engine.py`: simulación, percepción, control, planificación y métricas.
- `ares_x_server.py`: API Flask y servidor local de la aplicación.
- `ares_x_3d.html`, `ares_x_3d.css`, `ares_x_3d.js`: interfaz Three.js.
- `ares_x_colab_dashboard.py`: panel ipywidgets incorporado en el notebook.
- `test_ares_x_engine.py`: pruebas del motor y sus medidas de seguridad.
