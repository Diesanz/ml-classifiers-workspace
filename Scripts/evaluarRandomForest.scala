// Evaluación del modelo final de Random Forest (RF)
//
// Carga evaluacionComun.scala (que a su vez ejecuta la preparación de datos),
// lee el modelo previamente generado desde la carpeta ./modeloRF (sin volver a crearlo)
// y evalúa las predicciones sobre el conjunto de prueba (testFinal) usando la función evaluar() de evaluacionComun.
// Al final calcula también el error en entrenamiento, para compararlo con el de prueba.

// Lanzar spark-shell con más memoria para el driver (en modo local hace también de executor):
//   spark-shell --driver-memory 8g
// El bosque tiene más de un millón de nodos y no cabe en el 1 GiB que Spark usa por defecto

:load evaluacionComun.scala

import org.apache.spark.ml.classification.RandomForestClassificationModel

// Carpeta de la que se lee el modelo final (en el enunciado se pide ./modeloXX)
val RUTA_MODELO = "./modeloRF"

// Lee el modelo previamente generado y guardado
println(s"\nCargando el modelo guardado desde ${RUTA_MODELO}")
val bestModelRF = RandomForestClassificationModel.load(RUTA_MODELO)

// Imprime características básicas del bosque para verificar que es el correcto
println(s"Número de árboles: ${bestModelRF.getNumTrees}")
println(s"Profundidad máxima (parámetro): ${bestModelRF.getMaxDepth}")
println(s"Mínimo de instancias por nodo: ${bestModelRF.getMinInstancesPerNode}")
println(s"Nodos totales del bosque: ${bestModelRF.totalNumNodes}")

// Genera las predicciones sobre el conjunto de prueba (testFinal)
val prediccionesRF = bestModelRF.transform(testFinal)

// Llama a la función común de evaluación
evaluar(prediccionesRF, "Random Forest (RF)")

// Error sobre el entrenamiento, para compararlo con el de prueba
errorEntrenamiento(bestModelRF)