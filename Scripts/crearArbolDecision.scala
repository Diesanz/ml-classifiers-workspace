// Validación cruzada del árbol de decisión y guarda ./modeloDT

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.ml.classification.{DecisionTreeClassifier, DecisionTreeClassificationModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

:load preparacionDatos.scala

// Se cachea porque la validación cruzada lo recorre muchas veces
trainFinal.cache()
println(s"Viajes de entrenamiento: ${trainFinal.count()}")

// Métrica de selección: tasa de acierto (= 1 - error, la métrica principal del enunciado)
val metrics = new MulticlassClassificationEvaluator()
metrics.setMetricName("accuracy")

val DTtaxi = new DecisionTreeClassifier()

// Búsqueda gruesa: profundidad y tamaño mínimo de las hojas (el resto, por defecto)
val paramGrid = new ParamGridBuilder().
  addGrid(DTtaxi.maxDepth, Array(5, 10, 15, 20)).
  addGrid(DTtaxi.minInstancesPerNode, Array(1, 100, 1000)).
  build()

// Revisar la rejilla antes de entrenar
println(s"Combinaciones de la rejilla: ${paramGrid.length}")
paramGrid.foreach(pm => println("  " + pm.toSeq.map(pp => pp.param.name + " = " + pp.value).mkString(", ")))

val cv = new CrossValidator().
  setEstimator(DTtaxi).
  setEvaluator(metrics).
  setEstimatorParamMaps(paramGrid).
  setNumFolds(3).      // 3 pliegues: ~388 000 viajes de validación en cada uno
  setParallelism(2).   // entrena 2 árboles a la vez, revisar nucleo sy memoria
  setSeed(42L)         // mismos pliegues en cada ejecución

val inicio = System.nanoTime()
val cvModel = cv.fit(trainFinal)
println(f"Tiempo de la validación cruzada: ${(System.nanoTime() - inicio) / 1e9}%.1f s")

// Resultado de cada combinación (media de los 3 pliegues)
val configuraciones = cvModel.getEstimatorParamMaps
val metricas = cvModel.avgMetrics

println("\nResultados de la validación cruzada:")
configuraciones.zip(metricas).foreach { case (pm, m) =>
  val config = pm.toSeq.map(pp => pp.param.name + " = " + pp.value).mkString(", ")
  println(f"$config%-45s acierto = $m%1.4f  error = ${1 - m}%1.4f")
}

val (mejorConfig, mejorMetrica) = configuraciones.zip(metricas).maxBy(_._2)
println("\nMejor configuración: " + mejorConfig.toSeq.map(pp => pp.param.name + " = " + pp.value).mkString(", "))
println(f"Tasa de acierto media: $mejorMetrica%1.4f  (error = ${1 - mejorMetrica}%1.4f)")

// El mejor modelo ya está reentrenado con todo el entrenamiento
val bestModel = cvModel.bestModel.asInstanceOf[DecisionTreeClassificationModel]
println(s"Profundidad: ${bestModel.depth}  Nodos: ${bestModel.numNodes}")

bestModel.write.overwrite().save("./modeloDT")
println("Modelo guardado en ./modeloDT")