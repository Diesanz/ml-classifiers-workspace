// Evaluación del modelo final de Árbol de Decisión (DT)
//
// Carga evaluacionComun.scala (que a su vez ejecuta la preparación de datos),
// lee el modelo previamente generado desde la carpeta ./modeloDT (sin volver a crearlo)
// y evalúa las predicciones sobre el conjunto de prueba (testFinal) usando la función evaluar() de evaluacionComun.

// Al final calcula también el error en entrenamiento, para compararlo con el de prueba.

// Las ejecuciones finales se hicieron con spark-shell --driver-memory 8g.
// En el DT no es imprescindible: el árbol cabe en el 1 GiB que Spark usa por defecto

:load evaluacionComun.scala

import org.apache.spark.ml.classification.DecisionTreeClassificationModel

// Carpeta de la que se lee el modelo final (en el enunciado se pide ./modeloXX)
val RUTA_MODELO = "./modeloDT"

// Lee el modelo previamente generado y guardado
println(s"\nCargando el modelo guardado desde ${RUTA_MODELO}")
val bestModelDT = DecisionTreeClassificationModel.load(RUTA_MODELO)

// Imprime características básicas del árbol para verificar que es el correcto
println(s"Profundidad del árbol: ${bestModelDT.depth}")
println(s"Número de nodos: ${bestModelDT.numNodes}")
println(s"Mínimo de instancias por nodo: ${bestModelDT.getMinInstancesPerNode}")
println(s"Profundidad máxima (parámetro): ${bestModelDT.getMaxDepth}")

// Genera las predicciones sobre el conjunto de prueba (testFinal)
val prediccionesDT = bestModelDT.transform(testFinal)

// Llama a la función común de evaluación
evaluar(prediccionesDT, "Árbol de Decisión (DT)")

// Error sobre el entrenamiento, para compararlo con el de prueba
errorEntrenamiento(bestModelDT)