// Carga preparacionDatos.scala, que deja el conjunto de prueba limpio y transformado (testFinal),
// y define evaluar(), que calcula las métricas del enunciado, y errorEntrenamiento(), que
// calcula el error en entrenamiento para compararlo con el de prueba.
// No se lanza por separado: lo cargan evaluarArbolDecision.scala y evaluarRandomForest.scala

:load preparacionDatos.scala

import org.apache.spark.ml.functions.{vector_to_array, array_to_vector}
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, BinaryClassificationEvaluator}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.Transformer

testFinal.cache()
println(s"Viajes de prueba: ${testFinal.count()}")

// testFinal ya está en caché. trainFinal solo lo usa errorEntrenamiento, al final
trainClean.unpersist()
testClean.unpersist()

// Nombre de cada clase en el orden del StringIndexer (posición = etiqueta)
val clases = labelIndexer.labelsArray(0)

// Tasa de no clasificados en test (NTAIE + NTOE, calculados en preparacionDatos)
val tasaNoClasificados = (ntaie + ntoe).toDouble / nTest
println(f"Tasa de no clasificados: $tasaNoClasificados%1.5f  (NTAIE = $ntaie, NTOE = $ntoe)")

// Calcula y muestra las métricas del enunciado para clasificadores multiclase
def evaluar(prediccionesTest: DataFrame, modelo: String): Unit = {
    prediccionesTest.cache()
    val test_count = prediccionesTest.count()

    println(s"\nEvaluación del modelo ${modelo}, con ${test_count} viajes")

    // 1. Tasa de error, desviación estándar e intervalo de confianza (95%)
    val tasa_acierto = new MulticlassClassificationEvaluator().setMetricName("accuracy").evaluate(prediccionesTest)
    val error = 1 - tasa_acierto
    val sigma = math.sqrt((error * tasa_acierto) / test_count)
    val z_95 = 1.96

    println(f"\nTasa de error: $error%1.4f   desviación estándar: $sigma%1.5f")
    println(f"Intervalo de confianza al 95 %%: [${error - z_95 * sigma}%1.4f, ${error + z_95 * sigma}%1.4f]")


    // 2. Matriz de confusión
    // Convertimos las predicciones y etiquetas a un RDD de tuplas (Double, Double) para MLlib
    val prediccionesYEtiquetas = prediccionesTest.select("prediction", "label").rdd.map(row => (row.getDouble(0), row.getDouble(1)))

    val metricasMulticlase = new MulticlassMetrics(prediccionesYEtiquetas)
    println("\nMatriz de Confusión:")
    println("Filas: clase real, columnas: clase predicha, en orden " + clases.mkString(", "))
    println(metricasMulticlase.confusionMatrix)


    // 3.Tasa de ciertos positivos (TP), falsos positivos (FP) y F-measure (F1 Score) por clase
    println("\nMétricas por clase:")
    val etiquetas = metricasMulticlase.labels
    etiquetas.foreach { label =>
        println(s"Clase ${clases(label.toInt)}:")
        println(f"  - Tasa de ciertos positivos (TP): ${metricasMulticlase.truePositiveRate(label)}%.4f")
        println(f"  - Tasa de falsos positivos (FP): ${metricasMulticlase.falsePositiveRate(label)}%.4f")
        println(f"  - F-measure (F1 Score): ${metricasMulticlase.fMeasure(label)}%.4f")
    }


    // 4. Áreas ROC y PR de la clase minoritaria (long) frente al resto (ML)
    //    Puntuación = probabilidad de long que da el modelo
    val etiquetaLong = clases.indexOf("long")
    val pLong = vector_to_array(col("probability"))(etiquetaLong)

    // Se pasa a binario (long frente al resto): para cada viaje, la probabilidad de long
    // que da el modelo y si de verdad es long (1) o no (0)
    val binario = prediccionesTest.
        withColumn("probabilidadBinaria", array_to_vector(array(lit(1.0) - pLong, pLong))).
        withColumn("esLong", when(col("label") === etiquetaLong, 1.0).otherwise(0.0))

    val evaluadorBinario = new BinaryClassificationEvaluator().
        setRawPredictionCol("probabilidadBinaria").   // vector de 2: el evaluador coge el segundo (long)
        setLabelCol("esLong")

    val areaROC = evaluadorBinario.setMetricName("areaUnderROC").evaluate(binario)
    val areaPR  = evaluadorBinario.setMetricName("areaUnderPR").evaluate(binario)
    println(f"\nlong frente al resto: área ROC = $areaROC%1.4f   área PR = $areaPR%1.4f")

    // 5. El error debe ser al menos 10 veces la tasa de no clasificados
    if (tasaNoClasificados * 10 > error){
        println("\nAVISO: el error no llega a 10 veces la tasa de no clasificados")
    }else{
        println("\nEl error es al menos 10 veces la tasa de no clasificados")
    }

}

// Error del modelo sobre el propio entrenamiento (resustitución). No es una métrica de
// evaluación: sirve para compararla con el error en prueba y ver si el modelo sobreajusta
def errorEntrenamiento(modelo: Transformer): Double = {
  val acierto = new MulticlassClassificationEvaluator().
    setMetricName("accuracy").
    evaluate(modelo.transform(trainFinal))
  val error = 1 - acierto
  println(f"\nError en entrenamiento (resustitución): $error%1.4f")
  error
}