import org.apache.spark.sql.{Column, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StructType, StructField, StringType,
  DoubleType, LongType, IntegerType, TimestampType}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics


// Ruta completa al fichero (cambiar en base a donde se tenga el .csv)
var PATH = "./train.csv"

// Paso 1: cargar el dataset
println("Paso 1: cargando el dataset")

// Esquema del dataset: evita que Spark lea el fichero para inferir los tipos
val taxiSchema = StructType(Array(
  StructField("id", StringType, true),
  StructField("vendor_id", IntegerType, true),
  StructField("pickup_datetime", TimestampType, true),
  StructField("dropoff_datetime", TimestampType, true),
  StructField("passenger_count", IntegerType, true),
  StructField("pickup_longitude", DoubleType, true),
  StructField("pickup_latitude", DoubleType, true),
  StructField("dropoff_longitude", DoubleType, true),
  StructField("dropoff_latitude", DoubleType, true),
  StructField("store_and_fwd_flag", StringType, true),
  StructField("trip_duration", LongType, true)
))

// cache() guarda el DataFrame en memoria para no releer el CSV en cada count()/show()
val rawTaxiDF = spark.read.format("csv").
  option("header", "true").
  schema(taxiSchema).
  load(PATH).
  cache()

println("Esquema del dataset original:")
rawTaxiDF.printSchema()
println(s"Filas: ${rawTaxiDF.count()}  Columnas: ${rawTaxiDF.columns.length}")   //esperado: 1458644 y 11
println("Primeras 5 filas:")
rawTaxiDF.show(5, truncate = false)


// Paso 2: eliminar registros vacíos (algún nulo) y no válidos
println("\nPaso 2: eliminando registros vacíos y no válidos (nulos y duplicados)")

val taxiDF = rawTaxiDF.na.drop().dropDuplicates().cache()
println(s"Registros vacíos o no válidos eliminados: ${rawTaxiDF.count() - taxiDF.count()}")   // esperado: 0


// Paso 3: crear la clase (short / medium / long) a partir de trip_duration
// Cortes en segundos: 10 min = 600 s, 25 min = 1500 s (el 25 entra en medium)
println("\nPaso 3: creando la clase (short <= 10 min, medium <= 25 min, long > 25 min)")

// when evalúa en orden y se queda con la primera condición que se cumple (como if / else if / else)
val classDF = taxiDF.withColumn("trip_class",
  when(col("trip_duration") <= 600, "short").
    when(col("trip_duration") <= 1500, "medium").
    otherwise("long")
).cache()

val total = classDF.count()
println("Reparto de clases en el dataset completo:")

classDF.groupBy("trip_class").count().
  withColumn("percentage", round(col("count") / total * 100, 2)).
  show()


// Paso 4: partición train / test (80/20)
// Se hace antes de cualquier otra modificación; la semilla la hace reproducible
println("\nPaso 4: partición train / test (80/20)")
// se reparten las filas al azar; con la misma semilla siempre sale la misma partición
val Array(train, test) = classDF.randomSplit(Array(0.8, 0.2), seed = 42L)
train.cache()
test.cache()

val nTrain = train.count()
val nTest  = test.count()
println(s"Train: $nTrain  Test: $nTest  (total: $total)")

println("Reparto de clases en train (antes de limpiar):")
train.groupBy("trip_class").count().
  withColumn("percentage", round(col("count") / nTrain * 100, 2)).show()
println("Reparto de clases en test (antes de limpiar):")
test.groupBy("trip_class").count().
  withColumn("percentage", round(col("count") / nTest * 100, 2)).show()


// Paso 5: limpieza (criterios de la sección 9 de la exploración)
//
// Se aplica la misma función a train y a test. En test, las filas eliminadas
// son los no clasificados (NTAIE + NTOE).
//
// Inconsistentes (NTAIE):
//  - passenger_count fuera de [1, 6]: 0 y 7-9 no son valores válidos
//  - coordenadas de recogida o destino fuera del rectángulo amplio
//    lat [40.0, 41.5], lon [-74.8, -73.0]: imposibles para un taxi de NYC.
//    Es amplio a propósito para no eliminar viajes reales a Newark o
//    Long Island (901 de los 1.329 destinos fuera del rectángulo ajustado son long)
//  - distancia 0 km y duración > 5 min: contradice la variable más predictiva
//    (los 0 km de <= 5 min se mantienen, son coherentes con short)
//
// Outliers de duración (NTOE):
//  - trip_duration < 60 s: un viaje real de menos de 1 min es implausible
//  - trip_duration > 3 h: de los 2.112, 1.843 superan las 23 h (artefacto,
//    p. ej. taxímetro sin apagar); entre 2 y 3 h hay solo ~141 viajes
//
// Notas:
//  - distance_km supone que el destino se conoce al inicio del viaje

println("\nPaso 5: limpieza (misma función para train y test)")


// Distancia haversine en km entre dos puntos (lat/lon en grados)
def distanceKm(lat1: Column, lon1: Column, lat2: Column, lon2: Column): Column = {
  // mide lo separados que están los dos puntos sobre una esfera
  val a = pow(sin(radians(lat2 - lat1) / 2), 2) +
          cos(radians(lat1)) * cos(radians(lat2)) *
          pow(sin(radians(lon2 - lon1) / 2), 2)
  // pasa el valor a kilómetros
  lit(2 * 6371.0) * asin(sqrt(a))
}

// Criterios de limpieza, definidos una sola vez: los usan clean() y el desglose de NTAIE / NTOE
val dist = distanceKm(col("pickup_latitude"), col("pickup_longitude"),
                      col("dropoff_latitude"), col("dropoff_longitude"))

// Una fila es inconsistente (NTAIE) si cumple alguna de estas condiciones
val inconsistent = (
  !col("passenger_count").between(1, 6) ||
  !col("pickup_latitude").between(40.0, 41.5) ||
  !col("dropoff_latitude").between(40.0, 41.5) ||
  !col("pickup_longitude").between(-74.8, -73.0) ||
  !col("dropoff_longitude").between(-74.8, -73.0) ||
  (dist === 0.0 && col("trip_duration") > 300)
)

// Una fila es outlier de duración (NTOE) si dura menos de 60 s o más de 3 h (10800 s)
val outlier = !col("trip_duration").between(60, 10800)

// Se queda solo con las filas válidas (las que no son inconsistentes ni outliers)
def clean(df: DataFrame): DataFrame = {
  df.filter(!inconsistent && !outlier)
}

val trainClean = clean(train).cache()
val testClean  = clean(test).cache()

val nTrainClean = trainClean.count()
val nTestClean  = testClean.count()

val removedTrain = nTrain - nTrainClean
val unclassified = nTest - nTestClean   // NTAIE + NTOE

// Desglose de NTAIE y NTOE en test (el enunciado pide contar cada causa)
//  - NTAIE: filas eliminadas por inconsistentes (pasajeros fuera de 1-6, coordenadas
//    fuera del rectángulo amplio o 0 km con duración > 5 min)
//  - NTOE: filas eliminadas solo por duración (< 60 s o > 3 h)
//  - Si una fila cumple criterios de las dos causas, cuenta en NTAIE (así no se cuenta dos veces)
//  - NTAIE + NTOE = unclassified, y la tasa de no clasificados es unclassified / nTest
val ntaie = test.filter(inconsistent).count()
val ntoe  = test.filter(!inconsistent && outlier).count()   // si cumple las dos causas, ya está en NTAIE
println(s"NTAIE = $ntaie  NTOE = $ntoe  (suma = ${ntaie + ntoe})")
// Comprobación: los dos grupos tienen que sumar exactamente las filas eliminadas en test
if (ntaie + ntoe != unclassified)
  println(s"NTAIE + NTOE (${ntaie + ntoe}) no coincide con las filas eliminadas ($unclassified)")

//Fin desglose

println("Reparto de clases en train tras la limpieza:")
trainClean.groupBy("trip_class").count().
  withColumn("percentage", round(col("count") / nTrainClean * 100, 2)).show()
println("Reparto de clases en test tras la limpieza:")
testClean.groupBy("trip_class").count().
  withColumn("percentage", round(col("count") / nTestClean * 100, 2)).show()


// Paso 6: construir atributos y eliminar los que no se usan
//  - Se construyen: distance_km, pickup_hour, pickup_day
//  - Se eliminan: id (identificador), dropoff_datetime (fuga de información),
//    vendor_id (reparto de clases casi igual al global), trip_duration (solo
//    servía para crear la clase) y pickup_datetime (ya extraída hora y día)
//  - pickup_longitude/latitude y dropoff_* se mantienen como candidatas
//  - El mes se descarta: el dataset solo cubre 6 meses y no generaliza

println("\nPaso 6: construcción y selección de atributos")

// Añade los atributos nuevos: distancia, hora y día de la semana de la recogida.
// Va antes del drop porque necesita las coordenadas y pickup_datetime
def buildFeatures(df: DataFrame): DataFrame =
  df.withColumn("distance_km", distanceKm(col("pickup_latitude"), col("pickup_longitude"),
                                          col("dropoff_latitude"), col("dropoff_longitude"))).
    withColumn("pickup_hour", hour(col("pickup_datetime"))).
    withColumn("pickup_day", dayofweek(col("pickup_datetime")))

val unusedCols = Seq("id", "dropoff_datetime", "vendor_id", "trip_duration", "pickup_datetime")

// Construye los atributos y después quita las columnas que no se usan (el número de filas no cambia)
def selectFeatures(df: DataFrame): DataFrame =
  buildFeatures(df).drop(unusedCols: _*)

val trainSel = selectFeatures(trainClean)
val testSel  = selectFeatures(testClean)

println("Columnas eliminadas: " + unusedCols.mkString(", "))
println("Columnas que quedan: " + trainSel.columns.mkString(", "))
trainSel.printSchema()


// Paso 7: transformación (features y label)
// Los indexers son estimators: se ajustan (fit) SOLO con train y se aplican
// a train y a test. Ajustarlos con test sería fuga de información 

println("\nPaso 7: transformación de atributos (features y label)")

// StringIndexer: convierte el texto del flag (N / Y) en un número
val flagIndexer = new StringIndexer().
  setInputCol("store_and_fwd_flag").
  setOutputCol("flag_idx").
  setHandleInvalid("keep").
  fit(trainSel)

// Convierte la clase (short / medium / long) en la etiqueta numérica 0, 1, 2 (por frecuencia en train)
val labelIndexer = new StringIndexer().
  setInputCol("trip_class").
  setOutputCol("label").
  fit(trainSel)

// Todos los atributos salvo la etiqueta y el flag en texto (se usa su versión numérica)
val featureCols = trainSel.columns.diff(Array("trip_class", "store_and_fwd_flag")) :+ "flag_idx"
println("Atributos que entran en features: " + featureCols.mkString(", "))

// El assembler es un transformer: solo junta las columnas en un vector
val assembler = new VectorAssembler().
  setInputCols(featureCols).
  setOutputCol("features")

// Aplica las mismas transformaciones (ya ajustadas con train) a cualquier DataFrame
// y deja solo las dos columnas que necesita el clasificador: features y label
def transformPipeline(df: DataFrame): DataFrame = {
  val withFlag  = flagIndexer.transform(df)
  val withLabel = labelIndexer.transform(withFlag)
  assembler.transform(withLabel).select("features", "label")
}

val trainFinal = transformPipeline(trainSel)
val testFinal  = transformPipeline(testSel)

println("Primeras 5 filas de train transformado:")
trainFinal.show(5, truncate = false)
println(s"Train final: ${trainFinal.count()}  Test final: ${testFinal.count()}")
println("Correspondencia clase -> label:")
labelIndexer.labelsArray(0).zipWithIndex.foreach { case (c, i) => println(s"  $c -> $i") }
println("Correspondencia flag -> índice:")
flagIndexer.labelsArray(0).zipWithIndex.foreach { case (c, i) => println(s"  $c -> $i") }

// Opcional: guardar los conjuntos transformados (parquet, porque features es un vector)
//trainFinal.write.mode("overwrite").parquet("output/train")
//testFinal.write.mode("overwrite").parquet("output/test")


// Segunda etapa: entregable individual
println("\nSegunda etapa (entregable individual): selección de un árbol de decisión")

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

println("\nClases en trainSub:")
trainSub.groupBy("label").count().
  withColumn("percentage", round(col("count") / nTrainSub * 100, 2)).
  orderBy("label").show()

println("Clases en validación:")
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

// Poner a false solo para no repetir la fase 1 en pruebas (debe estar en true para ejecutar todo)
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
// distintos; numBins = 20 los agrupa para obtener menos puntos (salen unos 200,
// porque Spark agrupa cada partición por separado). Con 0 saldría un punto por score
val scoresAndLabelsRDD = predictionsTest.select("probability", "label").rdd.
  map(r => (r.getAs[Vector](0)(1), r.getDouble(1)))
val metricsBinarias = new BinaryClassificationMetrics(scoresAndLabelsRDD, 20)

// Formato CSV (";" y coma decimal) para pegarlo en Excel u otra hoja de cálculo
println("\nPuntos de la curva ROC (copiar desde 'fp;tp' y pegar en Excel o otra herramienta para visualizar):")
println("fp;tp")
metricsBinarias.roc().collect().foreach { case (fp, tp) =>
  println(f"$fp%.4f;$tp%.4f".replace('.', ','))
}
println(f"\nÁrea bajo la curva ROC con los puntos de MLlib (20 bins): ${metricsBinarias.areaUnderROC()}%.4f")

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