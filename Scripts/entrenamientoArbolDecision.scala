import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics

:load preparacionDatos.scala
// Paso 8: conversión a un problema de clasificación binaria
// El preprocesado del primer entregable no se modifica: trainFinal y testFinal
// tienen la etiqueta de 3 clases (0 = short, 1 = medium, 2 = long).
// Siguiendo la simplificación que permite el enunciado, se agrupa la clase
// mayoritaria (short) frente al resto (medium + long):
//   0 = short (duración <= 10 min)
//   1 = más de 10 min (medium + long): clase positiva
// La nueva columna label se crea con when, por lo que ya no lleva los metadatos
// de 3 clases del StringIndexer: el árbol verá 2 clases
println("\nPaso 8: conversión a clasificación binaria (0 = short <= 10 min, 1 = más de 10 min)")

val trainBin = trainFinal.withColumn("label", when(col("label") === 0.0, 0.0).otherwise(1.0))
val testBin  = testFinal.withColumn("label", when(col("label") === 0.0, 0.0).otherwise(1.0))

// Se cachea trainBin porque se usa varias veces: para partirlo ahora y
// para entrenar el modelo final al terminar la exploración
trainBin.cache()
val nTrainBin = trainBin.count()   // esperado: 1155830

println(s"trainBin: $nTrainBin filas")
trainBin.groupBy("label").count().
  withColumn("percentage", round(col("count") / nTrainBin * 100, 2)).
  orderBy("label").show()


// Paso 9: segunda partición de trainBin en trainSub (80 %) y validación (20 %)
//  - trainSub: para entrenar los árboles candidatos
//  - validación: para estimar su error y elegir maxDepth y maxBins
//  - el test no se usa en ningún momento de la exploración
//  - semilla fija (distinta a la del split train/test) para que sea reproducible
println("\nPaso 9: partición de trainBin en trainSub / validación (80/20)")

val Array(trainSub, validation) = trainBin.randomSplit(Array(0.8, 0.2), seed = 4676L)

// Se cachean porque cada árbol de la exploración los recorre varias veces
trainSub.cache()
validation.cache()

val nTrainSub   = trainSub.count()
val nValidation = validation.count()
println(s"trainSub: $nTrainSub  validación: $nValidation  (suma: ${nTrainSub + nValidation}, trainBin: $nTrainBin)")

println("Reparto de clases en trainSub:")
trainSub.groupBy("label").count().
  withColumn("percentage", round(col("count") / nTrainSub * 100, 2)).
  orderBy("label").show()

println("Reparto de clases en validación:")
validation.groupBy("label").count().
  withColumn("percentage", round(col("count") / nValidation * 100, 2)).
  orderBy("label").show()


// Paso 10: liberar memoria antes de la exploración
// A partir de aquí solo se necesitan trainBin, testBin, trainSub y validación.
// Los árboles profundos necesitan mucha memoria, así que se liberan los DataFrames
// cacheados durante el preprocesado. testBin se cachea (y se materializa con count)
// antes de liberar testClean, del que depende
println("\nPaso 10: liberando memoria de los DataFrames del preprocesado")

testBin.cache()
val nTestBin = testBin.count()   // esperado: 288869

rawTaxiDF.unpersist()
taxiDF.unpersist()
classDF.unpersist()
train.unpersist()
test.unpersist()
trainClean.unpersist()
testClean.unpersist()


// Utilidades para la exploración

// Devuelve todas las combinaciones (maxDepth, maxBins) de las dos listas
def mezclaParametros(depths: Seq[Int], bins: Seq[Int]): Seq[(Int, Int)] = {
  var combinaciones = Seq[(Int, Int)]()          // lista vacía de parejas
  for (d <- depths) {
    for (b <- bins) {
      combinaciones = combinaciones :+ ((d, b))   // añade la pareja al final
    }
  }
  combinaciones
}

// Resultado de cada árbol de la exploración
case class Resultado(maxDepth: Int, maxBins: Int, nodos: Int,
                     errorTrain: Double, errorVal: Double, segundos: Double)

// Evaluador: tasa de acierto (accuracy). El error de cada árbol es 1 - accuracy
val metrics = new MulticlassClassificationEvaluator()
metrics.setMetricName("accuracy")

// Entrena un árbol por cada combinación con train y lo evalúa sobre train
// (resubstitución, solo para comparar) y sobre eval (para elegir los parámetros).
// Imprime cada resultado y los devuelve todos
def explorar(combinaciones: Seq[(Int, Int)], train: DataFrame, eval: DataFrame): Seq[Resultado] = {
  var resultados = Seq[Resultado]()

  for ((maxDepth, maxBins) <- combinaciones) {

    // Crear el árbol: solo se modifican maxDepth y maxBins, el resto por defecto
    val DTtaxi = new DecisionTreeClassifier()
    DTtaxi.setMaxDepth(maxDepth)
    DTtaxi.setMaxBins(maxBins)

    // Entrenar (midiendo el tiempo)
    val inicio = System.nanoTime()
    val DTtaxiModel = DTtaxi.fit(train)
    val segundos = (System.nanoTime() - inicio) / 1e9

    // Número de nodos del árbol
    val nodos = DTtaxiModel.numNodes

    // Error en el conjunto de entrenamiento (resubstitución: no sirve para elegir)
    val predictionsAndLabelsDF_train = DTtaxiModel.transform(train).select("prediction", "label")
    val error_train = 1 - metrics.evaluate(predictionsAndLabelsDF_train)

    // Error en el conjunto de validación (el que se usa para elegir los parámetros)
    val predictionsAndLabelsDF_val = DTtaxiModel.transform(eval).select("prediction", "label")
    val error_val = 1 - metrics.evaluate(predictionsAndLabelsDF_val)

    // Imprimir el resultado de esta combinación
    println(f"maxDepth = $maxDepth%2d  maxBins = $maxBins%3d  nodos = $nodos%6d  " +
            f"error trainSub = $error_train%.4f  error validación = $error_val%.4f  ($segundos%.1f s)")

    // Guardar para la tabla resumen
    resultados = resultados :+ Resultado(maxDepth, maxBins, nodos, error_train, error_val, segundos)
  }

  resultados
}

// Muestra una tabla con los resultados ordenados por error en validación (el mejor primero)
def mostrarResumen(resultados: Seq[Resultado], titulo: String): Unit = {
  println(s"\n$titulo (ordenado por error en validación):")
  println("maxDepth  maxBins    nodos  err.trainSub  err.validación   tiempo")
  resultados.sortBy(_.errorVal).foreach { r =>
    println(f"${r.maxDepth}%8d  ${r.maxBins}%7d  ${r.nodos}%7d  ${r.errorTrain}%12.4f  ${r.errorVal}%14.4f  ${r.segundos}%6.1f s")
  }
}


// Paso 11: fase 1 de la exploración (rejilla gruesa)
//  - maxDepth: 5 = valor por defecto (referencia); 10 y 15 = zona intermedia
//  - maxBins: 16 (< 24: agrupa horas y corta la distancia en pocos umbrales);
//    32 = valor por defecto; 64 = cortes más finos en distancia y coordenadas.
//    Mínimo válido: 3 (flag_idx tiene 3 categorías en los metadatos por handleInvalid = keep)
//  - Se añade (20, 16): árbol profundo para comprobar si aparece sobreajuste.
//    Con maxDepth = 20 solo se usa maxBins = 16 porque, en pruebas previas, (20, 64)
//    agotó la memoria de la máquina virtual (OutOfMemoryError) y (20, 32) tardó más
//    de 7 minutos
//  - El resto de parámetros se deja con su valor por defecto
//  - En total: 3 x 3 + 1 = 10 combinaciones

// ejecutarFase1 debe ser true en la entrega. Solo se pone a false durante el
// desarrollo, para no repetir la fase 1 (unos 10 minutos) cada vez
val ejecutarFase1 = false

val posibles_maxDepth = Seq(5, 10, 15)
val posibles_Bins     = Seq(16, 32, 64)

// Rejilla 3 x 3 más el árbol profundo (20, 16)
val parametros_fase1 = mezclaParametros(posibles_maxDepth, posibles_Bins) :+ ((20, 16))

println(s"\nPaso 11: fase 1 de la exploración (${parametros_fase1.size} combinaciones; entrena con trainSub, evalúa con validación)")

var resultadosFase1 = Seq[Resultado]()
if (ejecutarFase1) {
  resultadosFase1 = explorar(parametros_fase1, trainSub, validation)
  mostrarResumen(resultadosFase1, "Resumen de la fase 1")
} else {
  println("Fase 1 omitida (ejecutarFase1 = false)")
}


// Pasos 12 a 16 (fase 2, elección de parámetros, modelo final, métricas y guardado):
// se añaden después de analizar los resultados de la fase 1

val profundidades_fase2 = Seq(12, 13, 14, 16, 17, 18)
val bins_fase2          = Seq(16, 32)

val parametros_fase2 = mezclaParametros(profundidades_fase2, bins_fase2)

val resultadosFase2 = explorar(parametros_fase2, trainSub, validation)
mostrarResumen(resultadosFase2, "Resumen de la fase 2")

if (ejecutarFase1) {
  mostrarResumen(resultadosFase1 ++ resultadosFase2, "Resumen de las dos fases")
}

// Paso 13: elección de los parámetros finales
//  - Margen de ruido: con 230 890 ejemplos de validación y el mejor error (0,1513),
//    sigma = sqrt(0,1513 * 0,8487 / 230 890) ~ 0,00075 -> +-0,0015 al 95 %
//  - Dentro de ese margen del mejor (error <= 0,1528) empatan las profundidades 13 a 16
//    con cualquier maxBins: las diferencias no son significativas
//  - Entre las combinaciones empatadas se elige la más sencilla: (13, 32)
//      · error en validación 0,1524 (a 0,0011 del mejor, dentro del ruido)
//      · 7 499 nodos: un 44 % menos que (14, 32) y un 65 % menos que (15, 16)
//      · el menor sobreajuste de los empatados (diferencia validación - trainSub = 0,012)
//      · maxBins = 32 es el valor por defecto: como maxBins no influye, no se modifica
val mejorMaxDepth = 13
val mejorMaxBins  = 32

println(s"\nPaso 13: parámetros elegidos: maxDepth = $mejorMaxDepth, maxBins = $mejorMaxBins")


// Paso 14: modelo final con TODO el conjunto de entrenamiento (trainBin)
// Solo se modifican maxDepth y maxBins; el resto de parámetros, por defecto
println("\nPaso 14: modelo final (entrena con trainBin, evalúa con test)")

val DTtaxiFinal = new DecisionTreeClassifier()
DTtaxiFinal.setMaxDepth(mejorMaxDepth)
DTtaxiFinal.setMaxBins(mejorMaxBins)

val inicioFinal = System.nanoTime()
val DTtaxiModelFinal = DTtaxiFinal.fit(trainBin)
val segundosFinal = (System.nanoTime() - inicioFinal) / 1e9

val nodosFinal = DTtaxiModelFinal.numNodes
println(s"Número de clases del modelo: ${DTtaxiModelFinal.numClasses}")   // esperado: 2

// Error en trainBin (resubstitución, solo informativo)
val predictionsTrainFinal = DTtaxiModelFinal.transform(trainBin).select("prediction", "label")
val errorTrainFinal = 1 - metrics.evaluate(predictionsTrainFinal)

// Error en test: es la ÚNICA vez que se usa el conjunto de prueba.
// Se guarda también probability para calcular el AUC y la curva ROC
val predictionsTest = DTtaxiModelFinal.transform(testBin).
  select("prediction", "label", "probability").cache()
val errorTest = 1 - metrics.evaluate(predictionsTest)

println(f"maxDepth = $mejorMaxDepth%2d  maxBins = $mejorMaxBins%3d  nodos = $nodosFinal%6d  " +
        f"error trainBin = $errorTrainFinal%.4f  error test = $errorTest%.4f  ($segundosFinal%.1f s)")


// Paso 15: métricas sobre test con ML y MLlib
println("\nPaso 15: métricas sobre test")

// ML (MulticlassClassificationEvaluator): tasa de acierto, error y F1
val evaluadorF1 = new MulticlassClassificationEvaluator()
evaluadorF1.setMetricName("f1")

println(f"Tasa de acierto = ${1 - errorTest}%.4f")
println(f"Tasa de error   = $errorTest%.4f")
println(f"F1 (ponderada)  = ${evaluadorF1.evaluate(predictionsTest)}%.4f")

// ML (BinaryClassificationEvaluator): áreas bajo las curvas ROC y PR
// (no dependen del umbral: miden lo bien que el árbol ordena los viajes).
// Se usa la columna probability: en un árbol, rawPrediction contiene los recuentos
// de cada clase en la hoja, no probabilidades, y daría un AUC incorrecto
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

val evaluadorROC = new BinaryClassificationEvaluator()
evaluadorROC.setMetricName("areaUnderROC")
evaluadorROC.setRawPredictionCol("probability")

val evaluadorPR = new BinaryClassificationEvaluator()
evaluadorPR.setMetricName("areaUnderPR")
evaluadorPR.setRawPredictionCol("probability")

println(f"AUC ROC         = ${evaluadorROC.evaluate(predictionsTest)}%.4f")
println(f"AUC PR          = ${evaluadorPR.evaluate(predictionsTest)}%.4f")

// MLlib (MulticlassMetrics): matriz de confusión y métricas por clase
// MulticlassMetrics trabaja con un RDD de parejas (prediction, label)
val predictionsAndLabelsRDD = predictionsTest.select("prediction", "label").rdd.
  map(r => (r.getDouble(0), r.getDouble(1)))
val metricsMLlib = new MulticlassMetrics(predictionsAndLabelsRDD)

// Filas = clase real, columnas = clase predicha, en el orden 0 = short, 1 = más de 10 min
//   | TN  FP |
//   | FN  TP |
println("\nMatriz de confusión (filas = clase real, columnas = clase predicha; 0 = short, 1 = más de 10 min):")
println(metricsMLlib.confusionMatrix)

println("\nMétricas por clase:")
println(f"short          (0): precision = ${metricsMLlib.precision(0.0)}%.4f  recall = ${metricsMLlib.recall(0.0)}%.4f  F1 = ${metricsMLlib.fMeasure(0.0)}%.4f")
println(f"más de 10 min  (1): precision = ${metricsMLlib.precision(1.0)}%.4f  recall = ${metricsMLlib.recall(1.0)}%.4f  F1 = ${metricsMLlib.fMeasure(1.0)}%.4f")

// Punto del espacio ROC con el umbral por defecto (0,5), tomando como positiva la clase 1
println(f"\nTasa de ciertos positivos tp = ${metricsMLlib.truePositiveRate(1.0)}%.4f  " +
        f"Tasa de falsos positivos fp = ${metricsMLlib.falsePositiveRate(1.0)}%.4f")

// MLlib (BinaryClassificationMetrics): puntos de la curva ROC
// El score de cada viaje es la probabilidad de la clase 1 (proporción de viajes de más
// de 10 min en la hoja del árbol). numBins = 20 reduce el número de puntos de la curva
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.ml.linalg.Vector

val scoresAndLabelsRDD = predictionsTest.select("probability", "label").rdd.
  map(r => (r.getAs[Vector](0)(1), r.getDouble(1)))
val metricsBinarias = new BinaryClassificationMetrics(scoresAndLabelsRDD, 20)

println("\nPuntos de la curva ROC (fp, tp):")
metricsBinarias.roc().collect().foreach { case (fp, tp) => println(f"  ($fp%.4f; $tp%.4f)") }

// Referencias para interpretar el error
//  - Clasificador trivial que siempre predice la clase mayoritaria (1 = más de 10 min)
//  - Requisito del primer entregable: error >= 10 veces la tasa de no clasificados
val nShortTest = predictionsTest.filter(col("label") === 0.0).count()
val errorMayoritaria = nShortTest.toDouble / nTestBin
val tasaNoClasificados = unclassified.toDouble / nTest

println(f"\nError del clasificador que siempre predice la clase mayoritaria = $errorMayoritaria%.4f")
println(f"Tasa de no clasificados en test = $tasaNoClasificados%.4f")
println(f"Error en test / tasa de no clasificados = ${errorTest / tasaNoClasificados}%.1f  (debe ser >= 10)")


// Paso 16: guardar el modelo final en el directorio ./Modelo
// overwrite: si el directorio ya existe (por ejemplo, al ejecutar el script dos veces) se sustituye
println("\nPaso 16: guardando el modelo final en ./Modelo")

DTtaxiModelFinal.write.overwrite().save("./Modelo")
println("Modelo guardado")