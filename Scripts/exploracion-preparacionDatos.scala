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

spark.sparkContext.setLogLevel("ERROR")   // oculta los avisos (WARN) de Spark; solo salen los errores

// Ubicación del fichero de datos original (cambiar en base a donde se tenga el .csv)
val PATH = "./"
val FILE = "train.csv"

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
  load(PATH + FILE).
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
// Criterio revisado tras la corrección del primer entregable ("varios criterios hard,
// 1 % es mucho"): solo se eliminan los registros imposibles. Lo raro pero posible se
// mantiene, porque son viajes reales y el modelo tiene que saber clasificarlos.
//
// Inconsistentes (NTAIE):
//  - coordenadas de recogida o destino fuera del rectángulo amplio
//    lat [40.0, 41.5], lon [-74.8, -73.0]: imposibles para un taxi de NYC
//    (49 viajes en el conjunto completo).
//    Es amplio a propósito para no eliminar viajes reales fuera de la ciudad
//    (901 de los 1.329 destinos fuera del rectángulo ajustado son long)
//  - Ya no se eliminan:
//    - passenger_count 0 o 7-9: pueden ser envíos de paquetes o vehículos grandes
//    - 0 km y duración > 5 min: pueden ser atascos o un destino mal registrado
//
// Outliers de duración (NTOE):
//  - trip_duration > 23 h: pico aislado en ~24 h (artefacto, p. ej. taxímetro sin apagar;
//    1.843 en el conjunto completo)
//  - Ya no se eliminan:
//    - trip_duration < 60 s: su clase (short) es coherente y no hay salto en la distribución
//    - trip_duration entre 3 y 23 h (269 en el conjunto completo): son long y es el
//      final normal de la distribución

println("\nPaso 5: limpieza (misma función para train y test)")


// Criterios de limpieza, definidos una sola vez: los usan clean() y el desglose de NTAIE / NTOE
// (ya no hace falta la distancia: el criterio de 0 km se ha quitado)

// Una fila es inconsistente (NTAIE) si alguna coordenada cae fuera del rectángulo amplio
val inconsistent = (
  !col("pickup_latitude").between(40.0, 41.5) ||
  !col("dropoff_latitude").between(40.0, 41.5) ||
  !col("pickup_longitude").between(-74.8, -73.0) ||
  !col("dropoff_longitude").between(-74.8, -73.0)
)

// Una fila es outlier de duración (NTOE) si dura más de 23 h (82800 s)
val outlier = col("trip_duration") > 82800

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
//  - NTAIE: filas eliminadas por inconsistentes (coordenadas fuera del rectángulo amplio)
//  - NTOE: filas eliminadas solo por duración (> 23 h)
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
//  - distance_km se calcula ahora con la distancia Manhattan en vez de la haversine:
//    se parece más al recorrido real por las calles (correlación con la duración
//    0,809 frente a 0,794). Se mantiene el nombre del atributo
//  - Se eliminan: id (identificador), dropoff_datetime (fuga de información),
//    vendor_id (reparto de clases casi igual al global), trip_duration (solo
//    servía para crear la clase), pickup_datetime (ya extraída hora y día) y
//    store_and_fwd_flag (describe cómo se transmitió el registro, no el viaje)
//  - pickup_longitude/latitude y dropoff_* se mantienen como candidatas
//  - El mes se descarta: el dataset solo cubre 6 meses y no generaliza
//  - distance_km supone que el destino se conoce al inicio del viaje

println("\nPaso 6: construcción y selección de atributos")

// Distancia Manhattan en km entre dos puntos (lat/lon en grados).
// Las calles de Manhattan forman una cuadrícula girada unos 29° respecto al norte:
// se proyecta el desplazamiento sobre las avenidas y sobre las calles y se suman los dos
val anguloCuadricula = math.toRadians(29)
val kmPorGradoLat = 110.574
val kmPorGradoLon = 111.320 * math.cos(math.toRadians(40.75))   // en la latitud de NYC

def manhattanKm(lat1: Column, lon1: Column, lat2: Column, lon2: Column): Column = {
  val dx = (lon2 - lon1) * kmPorGradoLon   // km hacia el este
  val dy = (lat2 - lat1) * kmPorGradoLat   // km hacia el norte
  val porAvenidas = dx * math.sin(anguloCuadricula) + dy * math.cos(anguloCuadricula)
  val porCalles   = dx * math.cos(anguloCuadricula) - dy * math.sin(anguloCuadricula)
  abs(porAvenidas) + abs(porCalles)
}

// Añade los atributos nuevos: distancia, hora y día de la semana de la recogida.
// Va antes del drop porque necesita las coordenadas y pickup_datetime
def buildFeatures(df: DataFrame): DataFrame =
  df.withColumn("distance_km", manhattanKm(col("pickup_latitude"), col("pickup_longitude"),
                                           col("dropoff_latitude"), col("dropoff_longitude"))).
    withColumn("pickup_hour", hour(col("pickup_datetime"))).
    withColumn("pickup_day", dayofweek(col("pickup_datetime")))

val unusedCols = Seq("id", "dropoff_datetime", "vendor_id", "trip_duration", "pickup_datetime",
                     "store_and_fwd_flag")

// Construye los atributos y después quita las columnas que no se usan (el número de filas no cambia)
def selectFeatures(df: DataFrame): DataFrame =
  buildFeatures(df).drop(unusedCols: _*)

val trainSel = selectFeatures(trainClean)
val testSel  = selectFeatures(testClean)

println("Columnas eliminadas: " + unusedCols.mkString(", "))
println("Columnas que quedan: " + trainSel.columns.mkString(", "))
trainSel.printSchema()


// Paso 7: transformación (features y label)
// El indexer es un estimator: se ajusta (fit) SOLO con train y se aplica
// a train y a test. Ajustarlo con test sería fuga de información
// (ya no hay indexer del flag: store_and_fwd_flag se elimina en el paso 6)

println("\nPaso 7: transformación de atributos (features y label)")

// Convierte la clase (short / medium / long) en la etiqueta numérica 0, 1, 2 (por frecuencia en train)
val labelIndexer = new StringIndexer().
  setInputCol("trip_class").
  setOutputCol("label").
  fit(trainSel)

// Todos los atributos salvo la etiqueta
val featureCols = trainSel.columns.diff(Array("trip_class"))
println("Atributos que entran en features: " + featureCols.mkString(", "))

// El assembler es un transformer: solo junta las columnas en un vector
val assembler = new VectorAssembler().
  setInputCols(featureCols).
  setOutputCol("features")

// Aplica las mismas transformaciones (ya ajustadas con train) a cualquier DataFrame
// y deja solo las dos columnas que necesita el clasificador: features y label
def transformPipeline(df: DataFrame): DataFrame = {
  val withLabel = labelIndexer.transform(df)
  assembler.transform(withLabel).select("features", "label")
}

val trainFinal = transformPipeline(trainSel)
val testFinal  = transformPipeline(testSel)

println("Primeras 5 filas de train transformado:")
trainFinal.show(5, truncate = false)
println(s"Train final: ${trainFinal.count()}  Test final: ${testFinal.count()}")
println("Correspondencia clase -> label:")
labelIndexer.labelsArray(0).zipWithIndex.foreach { case (c, i) => println(s"  $c -> $i") }

// Opcional: guardar los conjuntos transformados (parquet, porque features es un vector)
//trainFinal.write.mode("overwrite").parquet("output/train")
//testFinal.write.mode("overwrite").parquet("output/test")


// Liberar la caché de lo que ya no hace falta
// Los recuentos (nTrain, nTest, ntaie, ntoe...) ya están calculados y guardados en variables.
// Se mantienen trainClean y testClean, de los que salen trainFinal y testFinal: si se liberan
// antes de cachear trainFinal o testFinal, Spark tendría que rehacerlos desde el CSV.
// Cada script los libera después de cachear el conjunto que usa
rawTaxiDF.unpersist()
taxiDF.unpersist()
classDF.unpersist()
train.unpersist()
test.unpersist()