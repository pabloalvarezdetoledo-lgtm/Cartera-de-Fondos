# Selección cuantitativa de fondos y cartera HRP

Pipeline en Python que analiza muchos fondos a la vez y construye una cartera inicial
**sin juicio cualitativo**: solo datos de rentabilidad y riesgo. Pensado para el
Concurso de Simulación de Gestión de Carteras de CFA Society Spain (IronIA), que se
evalúa por ratio de Sharpe.

## Instalación

```bash
pip install -r requirements.txt
```

Python 3.11 o superior (usa `tomllib`).

## Uso rápido

```bash
python main.py --fuente demo     # 1) prueba con datos sintéticos (sin internet)
python main.py                   # 2) datos reales con el universo de universo.csv
```

Resultados en `salida/`:

| Archivo | Contenido |
|---|---|
| `cartera_inicial.xlsx` | Hojas: Cartera, Resumen, Exposición, Universo (todos los fondos con su estado y motivo), Correlaciones, Backtest, Configuración, Datos descarga y Gráficos |
| `cartera_para_ironia.csv` | ISIN, nombre y peso redondeado (suma exactamente 100) |
| `*.png` | Pesos y contribución al riesgo, dendrograma, riesgo-rentabilidad, backtest |
| `registro_pruebas.csv` | Una fila por ejecución del backtest: fecha, hash del config, parámetros clave y Sharpe por método |
| `informe/Cartera_Fondos_CFA_AAAA-MM-DD.tex` (+ `.pdf`) | Informe narrativo completo (metodología, composición, riesgo, backtest, justificación por fondo), generado por `informe_pdf.py` a partir de los datos de la propia ejecución |

### Informe PDF

`informe_pdf.py` rellena la plantilla Jinja2 `plantillas/informe.tex.j2` con los
mismos objetos que ya usan `informe.py` (gráficos, Excel) y `documentos.py`
(`cartera`, `tabla`, `corr`, `resumen_bt`...): a diferencia de un `.tex` escrito a
mano, ningún número ni nombre de fondo está fijo en la plantilla, así que el informe
nunca queda desincronizado de la cartera real. Si `pdflatex` está instalado
(cualquier distribución TeX, p. ej. MiKTeX o TeX Live), compila el PDF automáticamente
(dos pasadas, para el índice); si no, deja el `.tex` en `salida/informe/` y avisa por
consola sin interrumpir el resto del pipeline. Cada ejecución guarda además una copia
fechada en `salida/informe/historico/`. Las posiciones cuyo nombre o subcategoría
contiene alguna palabra de `informe.palabras_riesgo_cola` (por defecto bonos
catástrofe, seguros, arbitraje de fusiones, mercados frontera, primas de estilo...)
se listan en una sección aparte ("Riesgos de cola no capturados por la ventana").

## Qué hace, paso a paso

1. **Universo** (`universo_ironia.csv`): los 3.885 fondos disponibles en IronIA a 30/09/2026,
   una clase por fondo (preferencia: EUR, acumulación, menores gastos corrientes), con
   gestora, categoría, subcategoría, gastos e inversión mínima. `universo.csv` es un ejemplo
   pequeño para pruebas rápidas (`python main.py --universo universo.csv`). Si Yahoo no encuentra un ISIN, añade su ticker en la
   columna `ticker` (los fondos europeos suelen ser `0P0000XXXX.F`).
2. **Datos**: descarga precios de rentabilidad total, los convierte a EUR y los guarda
   en `cache/` (no vuelve a descargar durante `dias_validez_cache` días).
3. **Rendimientos semanales**: las cifras diarias de fondos de distintas zonas horarias
   distorsionan las correlaciones; las semanales no.
4. **Métricas** (ventana de 3 años): rentabilidad, volatilidad, Sharpe, Sortino, caída
   máxima, asimetría, curtosis y:
   - **Ajuste por autocorrelación** (Lo, 2002): si un fondo tiene NAV "suavizado"
     (típico de activos poco líquidos), su volatilidad real es mayor que la observada.
   - **Probabilistic Sharpe Ratio** (Bailey y López de Prado, 2012): probabilidad de que
     el Sharpe verdadero sea positivo, dada la longitud de la muestra, asimetría y curtosis.
     Es el criterio de ranking por defecto: penaliza historiales cortos y colas feas.
     Con `ranking_multiventana = true` se usa, en vez del PSR a 3 años frente a 0, la media
     del PSR en `ventanas_psr` (1, 2 y 3 años por defecto) frente a un Sharpe anual de
     referencia `psr_referencia` (0,5 por defecto): frente a 0 casi todos los fondos buenos
     salen con PSR ≈ 100 % y no discrimina.
   - **Tipo libre de riesgo**: con `rf_fuente = "estr"` (por defecto) se descarga el €STR
     diario del BCE y se compone a la frecuencia del pipeline, en vez de usar un tipo fijo;
     si falla la descarga se usa `tipo_libre_riesgo` y se avisa.
5. **Análisis de estilo** (Sharpe, 1992): cada fondo se explica como combinación de
   índices (RV desarrollada, RV emergente, RF gobierno, RF corporativa, oro, liquidez).
   Con eso se clasifica en RV / RF / mixto / otros sin depender de datos de composición.
6. **Filtros**: historial mínimo, volatilidad mínima (monetarios fuera, ver abajo),
   volatilidad máxima, caída máxima, opcionalmente PSR mínimo y, con
   `excluir_anomalos = true` (por defecto), series con volatilidad anual > 200 % o alguna
   rentabilidad semanal con \|r\| > 50 % (datos de precio erróneos).
7. **Duplicados**: clustering con enlace completo; los fondos con correlación ≥ 0,95 entre
   todos ellos se tratan como uno solo y se queda el de mejor PSR.
8. **Reducción a N fondos**: clustering jerárquico (Ward) en N grupos; se elige el mejor
   fondo de cada grupo. Opcionalmente se asegura un fondo de cada tipo de activo. En el
   backtest, con `usar_histeresis = true`, un fondo ya en cartera conserva su cluster salvo
   que el mejor candidato le supere en `margen_histeresis` (0,05 por defecto) de puntuación.
9. **Cartera**: HRP (López de Prado, 2016) sobre una covarianza con shrinkage de
   Ledoit-Wolf, con peso máximo por fondo y un suelo que puede ser fijo (`peso_minimo`) o,
   con `suelo = "marginal"`, sin suelo fijo: se elimina iterativamente el fondo con menor
   contribución marginal al Sharpe de la cartera (Sharpe_i − ρ(i, cartera)·Sharpe_cartera
   negativo), sin bajar de `minimo_fondos`. Con `limitar_credito = true` se aplica además un
   techo `peso_maximo_credito` al bloque de renta fija de crédito (por subcategoría o por
   estilo RF Corporativa EUR), redistribuyendo el exceso proporcionalmente; y con
   `usar_corr_estres = true` la correlación usada es el máximo, elemento a elemento, entre
   la correlación normal y la calculada en un periodo de estrés (`estres_inicio`/`estres_fin`,
   2022 por defecto).
10. **Backtest walk-forward**: en cada trimestre se repite TODO (selección + pesos) solo
    con datos pasados y se compara HRP con inversa de la volatilidad, mínima varianza,
    máximo Sharpe y 1/N. Añade el error estándar del Sharpe (Lo, 2002) con intervalo de
    confianza al 95 % y un test de diferencia de Sharpe de cada método frente a HRP
    (Jobson-Korkie con la corrección de Memmel, 2003); cada ejecución añade una fila a
    `salida/registro_pruebas.csv` (fecha, hash del config, parámetros clave y Sharpe por
    método).

## Decisiones que conviene conocer

Todas están en `config.toml`. Las importantes:

- **`rf_fuente`**: `"estr"` (por defecto) descarga el €STR diario del BCE; `"fijo"` usa
  `tipo_libre_riesgo` (ajústalo al que use el concurso si lo dejas fijo).
- **Monetarios excluidos**: un fondo que rinde lo mismo que el tipo libre de riesgo no
  cambia el Sharpe de la cartera (solo la "diluye"), y su volatilidad casi nula haría que
  HRP le diera casi todo el peso.
- **Peso máximo (20 %) y suelo (`suelo`)**: HRP reparte por riesgo, así que los fondos de
  renta variable reciben pesos muy pequeños. Con `suelo = "fijo"`, `peso_minimo` garantiza
  que todo fondo seleccionado entre de verdad; con `suelo = "marginal"` (por defecto) no hay
  suelo fijo: se eliminan los fondos cuya contribución marginal al Sharpe es negativa. El
  techo evita concentrarse en un solo fondo de baja volatilidad. Si quieres más renta
  variable, prueba `metodo = "inversa_vol"`.
- **`limitar_credito` / `usar_corr_estres`**: controles de riesgo adicionales sobre HRP —
  un techo al peso conjunto del bloque de crédito, y una correlación que no infravalora el
  riesgo de cola usando el máximo entre la correlación normal y la de un periodo de estrés.
- **HRP no maximiza el Sharpe**: reparte riesgo sin usar rentabilidades esperadas, que son
  el input más ruidoso. Por eso el backtest lo compara con un máximo Sharpe "encogido".

## Tiempo de la primera ejecución

Con los 3.885 fondos, la primera descarga desde Yahoo tarda del orden de 1–2 horas (una
búsqueda y un histórico por fondo, con una pausa para no saturar). Queda en `cache/` y las
ejecuciones siguientes son de minutos. Para probar antes, filtra el CSV (por ejemplo, por
`categoria`) y pásalo con `--universo`.

## Limitaciones

- **Sesgo de supervivencia**: el universo son fondos que existen hoy; el backtest es
  optimista en términos absolutos. Sirve para comparar métodos entre sí.
- **Rotación**: la selección cambia bastante entre trimestres (~40 %). En el concurso no
  parece haber costes, pero en real sí los habría.
- **Fuentes no oficiales**: `yfinance` y `mstarpy` leen datos públicos de Yahoo y
  Morningstar y pueden dejar de funcionar si esas webs cambian. `mstarpy` además abre un
  Chrome invisible para obtener cookies (necesita Google Chrome instalado).
- Rentabilidades pasadas: el ranking usa historia; la persistencia del rendimiento de los
  fondos es débil. El filtro sirve para evitar lo claramente malo, no para predecir.

## Fuentes de datos

| `fuente` | Ventajas | Inconvenientes |
|---|---|---|
| `yahoo` | Sin navegador, rápido, sencillo | Algunos fondos europeos no están o tienen datos incompletos |
| `morningstar` | Mejor cobertura de fondos UCITS y composición | Necesita Chrome; más lento; API no oficial |
| `csv` | Control total (`datos/precios.csv`, columnas = ISIN, precios en EUR) | Tienes que conseguir tú los precios |
| `demo` | Prueba sin conexión | Datos inventados |

## Estructura

```
main.py          orquestación
datos.py         universo, descarga, caché, divisas
metricas.py      métricas, PSR, ajuste de autocorrelación, análisis de estilo
seleccion.py     filtros, duplicados, clusters
optimizacion.py  Ledoit-Wolf, HRP y demás métodos, límites de peso
backtest.py      walk-forward
informe.py       Excel y gráficos
informe_pdf.py   informe narrativo PDF (plantilla plantillas/informe.tex.j2 + pdflatex)
documentos.py    estado_cartera.md, metodologia.md, diario_decisiones.md
demo.py          datos sintéticos
config.toml      todos los parámetros
```
