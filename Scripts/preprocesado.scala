import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}

println("\n=======================================================")
println("INICIANDO PREPROCESADO DEL DATASET NYC TAXI (SPARK-SHELL)")
println("=======================================================\n")

var PATH = "train.csv"

// --- PASO 1: LECTURA Y COMPRENSIÓN DE DATOS ---
println("1. Cargando y leyendo el dataset...")
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

val rawDF = spark.read.option("header", "true").schema(taxiSchema).csv(PATH)
println(s"-> Registros leídos inicialmente: ${rawDF.count()}")

// --- PASO 2: LIMPIEZA PREVIA (Valores nulos explícitos) ---
println("\n2. Eliminando valores nulos estructurales...")
val cleanInitialDF = rawDF.na.drop()

// --- PASO 3: INGENIERÍA DE LA VARIABLE OBJETIVO ---
println("\n3. Creando la variable clase objetivo ('trip_class') según las reglas de EDA...")
val dfWithClass = cleanInitialDF.
  withColumn("duration_min", col("trip_duration") / 60.0).
  withColumn("trip_class",
    when(col("duration_min") <= 10.0, "short").
    when(col("duration_min") <= 25.0, "medium").
    otherwise("long")
  )

// --- PASO 4: PARTICIÓN TRAIN/TEST ---
println("\n4. Realizando la partición Train/Test (80/20)...")
val Array(trainRaw, testRaw) = dfWithClass.randomSplit(Array(0.8, 0.2), seed = 42L)
trainRaw.cache()
testRaw.cache()

// --- PASO 5: LIMPIEZA PROFUNDA (Inconsistencias y Outliers basados en EDA) ---
println("\n5. Aplicando limpieza profunda de inconsistencias y valores atípicos...")

val haversineKm = udf((lat1: Double, lon1: Double, lat2: Double, lon2: Double) => {
  val R = 6371.0
  val dLat = math.toRadians(lat2 - lat1)
  val dLon = math.toRadians(lon2 - lon1)
  val a = math.sin(dLat / 2) * math.sin(dLat / 2) +
          math.cos(math.toRadians(lat1)) * math.cos(math.toRadians(lat2)) *
          math.sin(dLon / 2) * math.sin(dLon / 2)
  val c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
  R * c
})

def filterValidTrips(df: DataFrame): DataFrame = {
  val dfTemp = df.withColumn("temp_distance", haversineKm(col("pickup_latitude"), col("pickup_longitude"), col("dropoff_latitude"), col("dropoff_longitude")))
  
  dfTemp.filter(
    col("passenger_count").between(1, 6) &&
    col("trip_duration").between(60, 10800) &&
    col("pickup_latitude").between(40.477, 40.917) &&
    col("dropoff_latitude").between(40.477, 40.917) &&
    col("pickup_longitude").between(-74.256, -73.700) &&
    col("dropoff_longitude").between(-74.256, -73.700) &&
    !(col("temp_distance") === 0.0 && col("trip_duration") > 300)
  ).drop("temp_distance")
}

val trainClean = filterValidTrips(trainRaw)
val testClean = filterValidTrips(testRaw)

// --- PASO 6: EVALUACIÓN DE LIMPIEZA (Cálculo NTAIE + NTOE) ---
println("\n6. Evaluando métricas de limpieza en Test...")
val initialTestCount = testRaw.count()
val cleanTestCount = testClean.count()
val NTAIE_NTOE = initialTestCount - cleanTestCount

println("\n=== MÉTRICAS DE LIMPIEZA EN CONJUNTO DE PRUEBA ===")
println(s"Instancias iniciales en Test: $initialTestCount")
println(s"Instancias eliminadas (NTAIE + NTOE): $NTAIE_NTOE")
println(s"Tasa de no clasificados: ${NTAIE_NTOE.toDouble / initialTestCount.toDouble}\n")

// --- PASO 7: CREACIÓN DE NUEVOS ATRIBUTOS SINTÉTICOS ---
println("7. Creando atributos sintéticos (hora, mes, día, distancia)...")

def addFeatures(df: DataFrame): DataFrame = {
  df.withColumn("pickup_hour", hour(col("pickup_datetime"))).
    withColumn("pickup_dayofweek", dayofweek(col("pickup_datetime"))).
    withColumn("pickup_month", month(col("pickup_datetime"))).
    withColumn("distance_km", haversineKm(col("pickup_latitude"), col("pickup_longitude"), col("dropoff_latitude"), col("dropoff_longitude")))
}

val trainFeat = addFeatures(trainClean)
val testFeat = addFeatures(testClean)

// --- PASO 8: TRANSFORMACIÓN DE ETIQUETA (Estimators) ---
println("\n8. Entrenando StringIndexer en Train para la etiqueta...")
val labelIndexer = new StringIndexer().
  setInputCol("trip_class").
  setOutputCol("label").
  fit(trainFeat)

// --- PASO 9: SELECCIÓN FINAL DE ATRIBUTOS (Transformers) ---
println("\n9. Ensamblando VectorAssembler...")
val featureCols = Array(
  "vendor_id", "passenger_count", "pickup_hour", "pickup_dayofweek",
  "pickup_month", "distance_km",
  "pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"
)

val assembler = new VectorAssembler().
  setInputCols(featureCols).
  setOutputCol("features")

// --- PASO 10: APLICACIÓN DE TRANSFORMACIONES A TRAIN Y TEST ---
println("\n10. Aplicando transformaciones finales...")
val trainIndexed = labelIndexer.transform(trainFeat)
val testIndexed = labelIndexer.transform(testFeat)

val trainTransformed = assembler.transform(trainIndexed).select("features", "label")
val testTransformed = assembler.transform(testIndexed).select("features", "label")

println("\n=== PROCESAMIENTO COMPLETADO ===")
println("Muestra de datos de entrenamiento (trainTransformed):")
trainTransformed.show(5, truncate = false)

println("=======================================================\n")