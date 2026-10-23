import org.apache.spark.sql.{Column, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StructType, StructField, StringType,
  DoubleType, LongType, IntegerType, TimestampType}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}

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


// Paso 2: eliminar registros vacíos (algún nulo) y no validos
println("\nPaso 2: eliminando registros vacíos y no válidos (nulos y duplicados)")

val taxiDF = rawTaxiDF.na.drop().dropDuplicates().cache()
println(s"Registros vacíos o no validos eliminados: ${rawTaxiDF.count() - taxiDF.count()}")   // esperado: 0


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