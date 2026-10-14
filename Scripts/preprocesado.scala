// ============================================================================
// Script de Preprocesado, Limpieza y Transformacion en Spark ML (Scala)
// Asignatura: Tecnicas Escalables de Analisis de Datos (Clasificadores)
// ============================================================================

import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}

var PATH = "train.csv"

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

val rawDF = spark.read.
  option("header", "true").
  schema(taxiSchema).
  csv(PATH)

val cleanInitialDF = rawDF.na.drop()

val dfWithClass = cleanInitialDF.
  withColumn("duration_min", col("trip_duration") / 60.0).
  withColumn("trip_class",
    when(col("duration_min") <= 10.0, "short").
    when(col("duration_min") <= 25.0, "medium").
    otherwise("long")
  )

val Array(trainRaw, testRaw) = dfWithClass.randomSplit(Array(0.8, 0.2), seed = 42L)
trainRaw.cache()
testRaw.cache()

def filterValidTrips(df: DataFrame): DataFrame = {
  df.filter(
    col("passenger_count") > 0 &&
    col("trip_duration") >= 10 && col("trip_duration") <= 86400 &&
    col("pickup_latitude").between(40.4, 40.9) &&
    col("dropoff_latitude").between(40.4, 40.9) &&
    col("pickup_longitude").between(-74.3, -73.7) &&
    col("dropoff_longitude").between(-74.3, -73.7)
  )
}

val trainClean = filterValidTrips(trainRaw)
val testClean = filterValidTrips(testRaw)

val initialTestCount = testRaw.count()
val cleanTestCount = testClean.count()
val NTAIE_NTOE = initialTestCount - cleanTestCount

println("\n=== MÉTRICAS DE LIMPIEZA EN CONJUNTO DE PRUEBA ===")
println(s"Instancias iniciales en Test: $initialTestCount")
println(s"Instancias eliminadas (NTAIE + NTOE): $NTAIE_NTOE")
println(s"Tasa de no clasificados: ${NTAIE_NTOE.toDouble / initialTestCount.toDouble}\n")

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

def addFeatures(df: DataFrame): DataFrame = {
  df.withColumn("pickup_hour", hour(col("pickup_datetime"))).
    withColumn("pickup_dayofweek", dayofweek(col("pickup_datetime"))).
    withColumn("pickup_month", month(col("pickup_datetime"))).
    withColumn("distance_km", haversineKm(col("pickup_latitude"), col("pickup_longitude"), col("dropoff_latitude"), col("dropoff_longitude")))
}

val trainFeat = addFeatures(trainClean)
val testFeat = addFeatures(testClean)

val labelIndexer = new StringIndexer().
  setInputCol("trip_class").
  setOutputCol("label").
  fit(trainFeat)

val flagIndexer = new StringIndexer().
  setInputCol("store_and_fwd_flag").
  setOutputCol("flag_indexed").
  setHandleInvalid("keep").
  fit(trainFeat)

val featureCols = Array(
  "vendor_id", "passenger_count", "pickup_hour", "pickup_dayofweek",
  "pickup_month", "distance_km", "flag_indexed",
  "pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"
)

val assembler = new VectorAssembler().
  setInputCols(featureCols).
  setOutputCol("features")

val trainIndexed = flagIndexer.transform(labelIndexer.transform(trainFeat))
val testIndexed = flagIndexer.transform(labelIndexer.transform(testFeat))

val trainTransformed = assembler.transform(trainIndexed).select("features", "label")
val testTransformed = assembler.transform(testIndexed).select("features", "label")

println("=== PROCESAMIENTO COMPLETADO ===")
println("Muestra del conjunto de entrenamiento transformado (features, label):")
trainTransformed.show(5, truncate = false)