// Evaluación del modelo final de Árbol de Decisión (DT)
//
// Carga evaluacionComun.scala (que a su vez ejecuta la preparación de datos),
// lee el modelo previamente generado desde la carpeta ./modeloDT (sin volver a crearlo)
// y evalúa las predicciones sobre el conjunto de prueba (testFinal) usando la función evaluar() de evaluacionComun.

:load evaluacionComun.scala

import org.apache.spark.ml.classification.DecisionTreeClassificationModel

// Lee el modelo previamente generado y guardado
println("\nCargando el modelo guardado desde ./modeloDT")
val bestModelDT = DecisionTreeClassificationModel.load("./modeloDT")

// Imprime características básicas del árbol para verificar que es el correcto
println(s"Profundidad del árbol: ${bestModelDT.depth}")
println(s"Número de nodos: ${bestModelDT.numNodes}")
println(s"Mínimo de instancias por nodo: ${bestModelDT.getMinInstancesPerNode}")

// Genera las predicciones sobre el conjunto de prueba (testFinal)
val prediccionesDT = bestModelDT.transform(testFinal)

// Llama a la función común de evaluación
evaluar(prediccionesDT, "Árbol de Decisión (DT)")