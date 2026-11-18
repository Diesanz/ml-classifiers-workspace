# Clasificación de la duración de viajes en taxi de Nueva York con Spark

**Grupo 3** · Técnicas escalables de análisis de datos (Clasificadores) · Universidad de Valladolid, curso 2026-27

Proyecto de clasificación multiclase sobre el conjunto *NYC Taxi Trip Duration*, desarrollado con Spark 4.1.3 (Scala). A partir de los datos de cada viaje (hora y lugar de recogida, destino y pasajeros) se predice si será corto, medio o largo.

## El problema

Cada viaje se clasifica según su duración en tres clases:

| Clase | Duración | Viajes (conjunto completo) |
|---|---|---|
| *short* | hasta 10 min | 44,76 % |
| *medium* | de 10 a 25 min | 42,87 % |
| *long* | más de 25 min | 12,37 % |

Los datos son el fichero `train.csv` de la competición de Kaggle *New York City Taxi Trip Duration*: 1 458 644 viajes de taxis amarillos entre enero y junio de 2016, con 11 atributos. La competición planteaba predecir la duración exacta; en este proyecto solo se predice la clase.

## Desarrollo

El proyecto se ha desarrollado en varias etapas, y cada una parte de la anterior:

1. **Exploración y preparación de los datos.** Análisis de cada atributo, de la clase, de los problemas del conjunto (coordenadas imposibles, duraciones de casi un día, viajes de 0 km…) y de las relaciones entre atributos. Después se preparan los datos:
   - **Partición:** 80 % entrenamiento y 20 % prueba (`randomSplit`, semilla 42). El conjunto de prueba solo se usa para evaluar los modelos finales.
   - **Limpieza:** solo se eliminan los registros imposibles: coordenadas fuera de un rectángulo amplio alrededor de Nueva York y viajes de más de 23 h. Quedan 1 165 381 viajes de entrenamiento y 291 371 de prueba (tasa de no clasificados: 0,143 %).
   - **Atributos (8):** número de pasajeros, las cuatro coordenadas, la distancia Manhattan entre recogida y destino (`distance_km`), y la hora y el día de la semana de la recogida.
2. **Primeros modelos.** Cada integrante del grupo construyó y analizó por su cuenta un árbol de decisión, lo que sirvió para conocer el efecto de sus parámetros.
3. **Creación, selección y evaluación de los modelos finales.** Se revisan la limpieza y la transformación, y se construyen dos modelos: un árbol de decisión y un *Random Forest*. Sus parámetros se eligen con validación cruzada de 3 pliegues (`CrossValidator` y `ParamGridBuilder`), con la tasa de acierto como métrica y una búsqueda en dos fases: una rejilla gruesa y después una fina alrededor de los mejores valores. Por último, los dos modelos se evalúan sobre el conjunto de prueba.

## Resultados

| | Árbol de decisión | *Random Forest* |
|---|---|---|
| Parámetros finales | `maxDepth` 25, `minInstancesPerNode` 50 | 20 árboles, `maxDepth` 30, `minInstancesPerNode` 10 |
| Acierto en validación cruzada | 0,7993 | 0,8150 |
| Error en prueba (IC 95 %) | 0,1983 [0,1968; 0,1997] | 0,1835 [0,1821; 0,1849] |
| Área ROC / PR (*long* frente al resto) | 0,947 / 0,803 | 0,955 / 0,832 |
| Nodos del modelo | 7 601 | 1 041 664 |

El *Random Forest* clasifica bien unos 4 300 viajes más que el árbol, y los intervalos de confianza no se solapan. A cambio, es unas 137 veces más grande y su búsqueda tarda unas cuatro veces más. En los dos modelos, el atributo más importante es la distancia, y el punto débil es la clase *long*: alrededor de un tercio de sus viajes se clasifican como *medium*.

## Contenido

| Fichero | Descripción |
|---|---|
| Scripts/`preparacionDatos.scala` | Carga, partición, limpieza y transformación de los datos. Es el único script donde se indican la ruta y el nombre del fichero de datos (`PATH` y `FILE`); los demás lo cargan. |
| Scripts/`crearArbolDecision.scala` | Crea y selecciona el árbol de decisión y lo guarda en `./modeloDT`. |
| Scripts/`crearRandomForest.scala` | Crea y selecciona el *Random Forest* y lo guarda en `./modeloRF`. |
| Scripts/`evaluacionComun.scala` | Carga la preparación de datos y define las funciones que calculan las métricas. No está pensado para lanzarse por separado: lo cargan los dos scripts de evaluación. |
| Scripts/`evaluarArbolDecision.scala` | Lee `./modeloDT`, ya entrenado, y lo evalúa sobre el conjunto de prueba. |
| Scripts/`evaluarRandomForest.scala` | Lee `./modeloRF`, ya entrenado, y lo evalúa sobre el conjunto de prueba. |
| Scripts/`modeloDT/`, Scripts/`modeloRF/` | Los dos modelos finales. |
| Scripts/`exploracion-datos.ipynb` | Exploración del conjunto de datos en Python, actualizada tras las correcciones. No modifica los datos. |
| Scripts/`descripcion-atributos.ipynb` | Tablas y gráficas descriptivas de cada atributo. |
| `E3_MemoriaG3Final.pdf` | Memoria de la creación, selección y evaluación de los modelos finales. |
| `E1_MemoriaG3Revisada.pdf` | Memoria de la exploración y preparación de los datos, con las correcciones. |

Relación entre los scripts:

```
preparacionDatos.scala
├── crearArbolDecision.scala   → guarda ./modeloDT
├── crearRandomForest.scala    → guarda ./modeloRF
└── evaluacionComun.scala
    ├── evaluarArbolDecision.scala   (lee ./modeloDT)
    └── evaluarRandomForest.scala    (lee ./modeloRF)
```

## Ejecución

1. Indicar en `preparacionDatos.scala` la ruta (`PATH`) y el nombre (`FILE`) del fichero de datos.
2. Lanzar `spark-shell` desde la carpeta `Scripts`, para que encuentre los demás scripts y las carpetas de los modelos, y con más memoria para el driver, porque el *Random Forest* no cabe en el 1 GiB que Spark usa por defecto:

   ```
   cd Scripts
   spark-shell --driver-memory 8g
   ```

3. Para evaluar los modelos no hace falta volver a crearlos:

   ```
   :load evaluarArbolDecision.scala
   :load evaluarRandomForest.scala
   ```

   Cada script muestra la tasa de error con su desviación estándar y su intervalo de confianza al 95 %, la matriz de confusión, la tasa de ciertos positivos, la tasa de falsos positivos y la F-measure de cada clase, las áreas ROC y PR de *long* frente al resto y el error en entrenamiento.

4. Para volver a crearlos:

   ```
   :load crearArbolDecision.scala
   :load crearRandomForest.scala
   ```

   Tardan bastante: unas 2 h el árbol de decisión y unas 8,5 h el *Random Forest* (en una máquina de 4 núcleos y 15 GiB). Cada script sobrescribe el modelo guardado; con las mismas semillas se obtiene el mismo modelo.

## Autores

Gonzalo Agúndez Sáez · Paula Negro Asegurado · Diego Sanz Santos