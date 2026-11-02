import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

// Pasos 1 a 7 (al unificar, se sustituye por el contenido de preparacionDatos.scala)
:load preparacionDatos.scala


// Paso 8: clase binaria
// Se agrupa la clase mayoritaria (short) frente al resto, como permite el enunciado:
//   0 = short (<= 10 min)   1 = más de 10 min (medium + long), clase positiva
println("\nPaso 8: paso a clase binaria (0 = short, 1 = más de 10 min)")

val trainBin = trainFinal.withColumn("label", when(col("label") === 0.0, 0.0).otherwise(1.0))
val testBin  = testFinal.withColumn("label", when(col("label") === 0.0, 0.0).otherwise(1.0))

// Se cachean porque se usan varias veces (el count los materializa)
trainBin.cache()
val nTrainBin = trainBin.count()   // esperado: 1155830
testBin.cache()
val nTestBin = testBin.count()     // esperado: 288869

println(s"\nClases en trainBin ($nTrainBin viajes):")
trainBin.groupBy("label").count().
  withColumn("percentage", round(col("count") / nTrainBin * 100, 2)).
  orderBy("label").show()

println(s"Clases en testBin ($nTestBin viajes):")
testBin.groupBy("label").count().
  withColumn("percentage", round(col("count") / nTestBin * 100, 2)).
  orderBy("label").show()


// Paso 9: trainBin se divide en trainSub (80 %, para entrenar los árboles)
// y validación (20 %, para estimar su error). El test no se usa en la exploración
println("\nPaso 9: división de trainBin en trainSub y validación (80/20)")

val Array(trainSub, validation) = trainBin.randomSplit(Array(0.8, 0.2), seed = 4676L)
trainSub.cache()
validation.cache()

val nTrainSub   = trainSub.count()     // esperado: 925752
val nValidation = validation.count()   // esperado: 230078
println(s"trainSub: $nTrainSub viajes | validación: $nValidation viajes")

println(s"\nClases en trainSub:")
trainSub.groupBy("label").count().
  withColumn("percentage", round(col("count") / nTrainSub * 100, 2)).
  orderBy("label").show()

println(s"Clases en validación:")
validation.groupBy("label").count().
  withColumn("percentage", round(col("count") / nValidation * 100, 2)).
  orderBy("label").show()


// Paso 10: se libera la memoria del preprocesado, porque los árboles profundos la necesitan
println("\nPaso 10: liberando memoria del preprocesado")

rawTaxiDF.unpersist()
taxiDF.unpersist()
classDF.unpersist()
train.unpersist()
test.unpersist()
trainClean.unpersist()
testClean.unpersist()


// Funciones para la exploración

// Todas las combinaciones (maxDepth, maxBins) de las dos listas
def mezclaParametros(depths: Seq[Int], bins: Seq[Int]): Seq[(Int, Int)] = {
  var combinaciones = Seq[(Int, Int)]()
  for (d <- depths) {
    for (b <- bins) {
      combinaciones = combinaciones :+ ((d, b))
    }
  }
  combinaciones
}

// Resultado de cada árbol
case class Resultado(maxDepth: Int, maxBins: Int, nodos: Int,
                     errorTrain: Double, errorVal: Double, segundos: Double)

// Error = 1 - tasa de acierto. La tasa de acierto la da el evaluador multiclase
// (sirve también con 2 clases); el binario solo calcula áreas bajo curvas
// No se usan pesos (weightCol) ni métricas ponderadas: las clases están casi equilibradas (55/45) y los dos errores cuestan lo mismo
val metrics = new MulticlassClassificationEvaluator()
metrics.setMetricName("accuracy")

// Entrena un árbol por combinación y calcula su error en train (solo para ver
// el sobreajuste) y en eval (el que se usa para elegir)
def explorar(combinaciones: Seq[(Int, Int)], train: DataFrame, eval: DataFrame): Seq[Resultado] = {
  var resultados = Seq[Resultado]()

  for ((maxDepth, maxBins) <- combinaciones) {
    // Solo se cambian maxDepth y maxBins; el resto, por defecto
    val DTtaxi = new DecisionTreeClassifier()
    DTtaxi.setMaxDepth(maxDepth)
    DTtaxi.setMaxBins(maxBins)

    val inicio = System.nanoTime()
    val DTtaxiModel = DTtaxi.fit(train)
    val segundos = (System.nanoTime() - inicio) / 1e9

    val nodos = DTtaxiModel.numNodes

    val predictionsAndLabelsDF_train = DTtaxiModel.transform(train).select("prediction", "label")
    val error_train = 1 - metrics.evaluate(predictionsAndLabelsDF_train)

    val predictionsAndLabelsDF_val = DTtaxiModel.transform(eval).select("prediction", "label")
    val error_val = 1 - metrics.evaluate(predictionsAndLabelsDF_val)

    println(f"Árbol maxDepth = $maxDepth%2d, maxBins = $maxBins%3d -> $nodos%6d nodos | " +
            f"error trainSub = $error_train%.4f | error validación = $error_val%.4f | $segundos%.1f s")

    resultados = resultados :+ Resultado(maxDepth, maxBins, nodos, error_train, error_val, segundos)
  }

  resultados
}

// Tabla con los resultados, del menor al mayor error en validación
def mostrarResumen(resultados: Seq[Resultado], titulo: String): Unit = {
  println(s"\n$titulo (de menor a mayor error en validación):")
  println("maxDepth  maxBins    nodos  err.trainSub  err.validación   tiempo")
  resultados.sortBy(_.errorVal).foreach { r =>
    println(f"${r.maxDepth}%8d  ${r.maxBins}%7d  ${r.nodos}%7d  ${r.errorTrain}%12.4f  ${r.errorVal}%14.4f  ${r.segundos}%6.1f s")
  }
}


// Paso 11: fase 1, búsqueda amplia (3 x 3 + 1 = 10 árboles)
//  - maxDepth 5 (por defecto), 10 y 15; maxBins 16, 32 (por defecto) y 64
//  - (20, 16) para comprobar el sobreajuste. Con profundidad 20 solo se usan 16 bins:
//    (20, 64) agotó la memoria y (20, 32) tardó más de 7 minutos
//  - maxBins mínimo válido: 3 (flag_idx tiene 3 valores por handleInvalid = keep)

// Activar o descativar dependiendos i qe quiere volver a comproabr de nuevo esta fase y sus parámetros
val ejecutarFase1 = true

val posibles_maxDepth = Seq(5, 10, 15)
val posibles_Bins     = Seq(16, 32, 64)
val parametros_fase1  = mezclaParametros(posibles_maxDepth, posibles_Bins) :+ ((20, 16))

println(s"\nPaso 11: fase 1 de la exploración (${parametros_fase1.size} árboles)")

var resultadosFase1 = Seq[Resultado]()
if (ejecutarFase1) {
  resultadosFase1 = explorar(parametros_fase1, trainSub, validation)
  mostrarResumen(resultadosFase1, "Resumen de la fase 1")
} else {
  println("Fase 1 omitida (ejecutarFase1 = false)")
}


// Paso 12: fase 2, búsqueda fina alrededor de la profundidad 15 (6 x 2 = 12 árboles)
//  - Profundidades que faltan entre 10 y 20 (la 15 ya está probada)
//  - Sin 64 bins: no mejoró en la fase 1 y es más lento
val profundidades_fase2 = Seq(12, 13, 14, 16, 17, 18)
val bins_fase2          = Seq(16, 32)
val parametros_fase2    = mezclaParametros(profundidades_fase2, bins_fase2)

println(s"\nPaso 12: fase 2 de la exploración (${parametros_fase2.size} árboles)")

val resultadosFase2 = explorar(parametros_fase2, trainSub, validation)
mostrarResumen(resultadosFase2, "Resumen de la fase 2")

if (ejecutarFase1) {
  mostrarResumen(resultadosFase1 ++ resultadosFase2, "Resumen de las dos fases")
}


// Paso 13: elección de los parámetros
//  - Margen de ruido al 95 %: sigma = sqrt(0,1513 * 0,8487 / 230 078) ~ 0,00075 -> +-0,0015
//  - Dentro del margen del mejor (error <= 0,1528) hay varios árboles empatados;
//    se elige el más sencillo: (13, 32), con 7 499 nodos y el menor sobreajuste
val mejorMaxDepth = 13
val mejorMaxBins  = 32

println(s"\nPaso 13: parámetros elegidos -> maxDepth = $mejorMaxDepth, maxBins = $mejorMaxBins")


// Paso 14: modelo final, entrenado con todo trainBin
println("\nPaso 14: modelo final (entrena con trainBin y se evalúa una sola vez con test)")

val DTtaxiFinal = new DecisionTreeClassifier()
DTtaxiFinal.setMaxDepth(mejorMaxDepth)
DTtaxiFinal.setMaxBins(mejorMaxBins)

val inicioFinal = System.nanoTime()
val DTtaxiModelFinal = DTtaxiFinal.fit(trainBin)
val segundosFinal = (System.nanoTime() - inicioFinal) / 1e9

val nodosFinal = DTtaxiModelFinal.numNodes
println(s"Clases del modelo: ${DTtaxiModelFinal.numClasses}")   // esperado: 2

val predictionsTrainFinal = DTtaxiModelFinal.transform(trainBin).select("prediction", "label")
val errorTrainFinal = 1 - metrics.evaluate(predictionsTrainFinal)

// Única vez que se usa el test. Se guarda probability para el AUC y la curva ROC
val predictionsTest = DTtaxiModelFinal.transform(testBin).
  select("prediction", "label", "probability").cache()
val errorTest = 1 - metrics.evaluate(predictionsTest)

println(f"Árbol final maxDepth = $mejorMaxDepth, maxBins = $mejorMaxBins -> $nodosFinal nodos | " +
        f"error trainBin = $errorTrainFinal%.4f | error test = $errorTest%.4f | $segundosFinal%.1f s")


// Paso 15: métricas sobre test
println("\nPaso 15: métricas del modelo final sobre test")

// ML: tasa de acierto, error y F1 ponderada
val evaluadorF1 = new MulticlassClassificationEvaluator()
evaluadorF1.setMetricName("f1")

println(f"Tasa de acierto: ${1 - errorTest}%.4f")
println(f"Tasa de error:   $errorTest%.4f")
println(f"F1 ponderada:    ${evaluadorF1.evaluate(predictionsTest)}%.4f")

// ML: áreas bajo las curvas ROC y PR. Se usa probability porque en un árbol
// rawPrediction son los recuentos de la hoja y daría un AUC incorrecto
val evaluadorROC = new BinaryClassificationEvaluator()
evaluadorROC.setMetricName("areaUnderROC")
evaluadorROC.setRawPredictionCol("probability")

val evaluadorPR = new BinaryClassificationEvaluator()
evaluadorPR.setMetricName("areaUnderPR")
evaluadorPR.setRawPredictionCol("probability")

println(f"Área bajo la curva ROC: ${evaluadorROC.evaluate(predictionsTest)}%.4f")
println(f"Área bajo la curva PR:  ${evaluadorPR.evaluate(predictionsTest)}%.4f")

// MLlib: matriz de confusión y métricas por clase con el umbral por defecto (0,5).
// La matriz sale con la clase 0 primero: | TN FP | y debajo | FN TP |
val predictionsAndLabelsRDD = predictionsTest.select("prediction", "label").rdd.
  map(r => (r.getDouble(0), r.getDouble(1)))
val metricsMLlib = new MulticlassMetrics(predictionsAndLabelsRDD)

println("\nMatriz de confusión (filas = clase real, columnas = clase predicha; 0 = short, 1 = más de 10 min):")
println(metricsMLlib.confusionMatrix)

println("\nMétricas por clase:")
println(f"short         (0): precisión = ${metricsMLlib.precision(0.0)}%.4f | recall = ${metricsMLlib.recall(0.0)}%.4f | F1 = ${metricsMLlib.fMeasure(0.0)}%.4f")
println(f"más de 10 min (1): precisión = ${metricsMLlib.precision(1.0)}%.4f | recall = ${metricsMLlib.recall(1.0)}%.4f | F1 = ${metricsMLlib.fMeasure(1.0)}%.4f")

println(f"\nPunto ROC con umbral 0,5: tp = ${metricsMLlib.truePositiveRate(1.0)}%.4f | fp = ${metricsMLlib.falsePositiveRate(1.0)}%.4f")

// MLlib: puntos de la curva ROC. El score de cada viaje es la probabilidad de la
// clase 1 de su hoja. Como el árbol tiene miles de hojas, hay miles de scores
// distintos; numBins = 20 los agrupa para obtener pocos puntos, suficientes para
// dibujar la curva y fáciles de copiar (con 0 saldría un punto por score distinto)
val scoresAndLabelsRDD = predictionsTest.select("probability", "label").rdd.
  map(r => (r.getAs[Vector](0)(1), r.getDouble(1)))
val metricsBinarias = new BinaryClassificationMetrics(scoresAndLabelsRDD, 20)

// Formato CSV (";" y coma decimal) para pegarlo en Excel u otra hoja de cálculo
println("\nPuntos de la curva ROC (copiar desde 'fp;tp' y pegar en Excel o otra herramienta apra visualizar):")
println("fp;tp")
metricsBinarias.roc().collect().foreach { case (fp, tp) =>
  println(f"$fp%.4f;$tp%.4f".replace('.', ','))
}

// Referencias para valorar el error:
//  - trivial: predice siempre la clase mayoritaria (1), así que falla todos los short
//  - no clasificados: unclassified y nTest vienen del preprocesado (pasos 1 a 7)
val nShortTest = predictionsTest.filter(col("label") === 0.0).count()
val errorMayoritaria = nShortTest.toDouble / nTestBin
val tasaNoClasificados = unclassified.toDouble / nTest

println(f"\nError del clasificador trivial (siempre 'más de 10 min'): $errorMayoritaria%.4f")
println(f"Tasa de no clasificados en test: $tasaNoClasificados%.4f")
println(f"Error en test / tasa de no clasificados: ${errorTest / tasaNoClasificados}%.1f (debe ser al menos 10)")


// Paso 16: guardar el modelo final (overwrite por si ./Modelo ya existe)
println("\nPaso 16: guardando el modelo final en ./Modelo")

DTtaxiModelFinal.write.overwrite().save("./Modelo")
println("Modelo guardado")