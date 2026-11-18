// Validación cruzada del random forest y guarda ./modeloRF
//
// Lanzar spark-shell con más memoria para el driver (en modo local hace también de executor):
//   spark-shell --driver-memory 8g
// Cada combinación entrena un bosque entero por pliegue, así que tarda bastante más que el DT.

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.ml.classification.{RandomForestClassifier, RandomForestClassificationModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.param.ParamMap

:load preparacionDatos.scala

// Carpeta donde se guarda el modelo final (en el enunciado se pide ./modeloXX)
val RUTA_MODELO = "./modeloRF"

// Se cachea porque la validación cruzada lo recorre muchas veces
trainFinal.cache()
println(s"Viajes de entrenamiento: ${trainFinal.count()}")

// Ya no hacen falta: trainFinal está en caché y el test no se usa al crear el modelo
trainClean.unpersist()
testClean.unpersist()

// Métrica de selección: tasa de acierto, la misma que en el DT para poder comparar
val metrics = new MulticlassClassificationEvaluator()
metrics.setMetricName("accuracy")

// Parámetros fijos (no entran en la rejilla)
val RFtaxi = new RandomForestClassifier().
  setMaxBins(32).                      // igual que en el DT: apenas influye en el error
  setImpurity("gini").                 // criterio de división por defecto
  setMinInstancesPerNode(50).          // en la fase 1, el mejor tamaño mínimo de hoja del DT;
                                       // además limita el tamaño de cada árbol. En la fase 2
                                       // entra en la rejilla y este valor se sustituye
  setFeatureSubsetStrategy("sqrt").    // modo para clasificación: sqrt(8) ≈ 3 atributos por nodo,
                                       // así los árboles salen distintos entre sí
  setSubsamplingRate(1.0).             // cada muestra bootstrap del tamaño del entrenamiento
  setBootstrap(true).                  // muestras con reemplazo (valor por defecto)
  setCacheNodeIds(true).               // acelera los árboles profundos; no cambia el resultado
  setSeed(2627L)                       // mismas muestras y atributos en cada ejecución

println(s"Fijos: maxBins = ${RFtaxi.getMaxBins}, impurity = ${RFtaxi.getImpurity}, " +
        s"featureSubsetStrategy = ${RFtaxi.getFeatureSubsetStrategy}, " +
        s"subsamplingRate = ${RFtaxi.getSubsamplingRate}")


// Lanza la validación cruzada con una rejilla y muestra el resultado de cada combinación
def validacionCruzada(rejilla: Array[ParamMap], titulo: String): CrossValidatorModel = {
  println(s"\n$titulo (${rejilla.length} combinaciones)")

  val cv = new CrossValidator().
    setEstimator(RFtaxi).
    setEvaluator(metrics).
    setEstimatorParamMaps(rejilla).
    setNumFolds(3).      // mismos pliegues que en el DT
    // 1 bosque a la vez: cada bosque ocupa mucha más memoria que un árbol, y Spark ya
    // entrena los árboles de un bosque usando los 4 núcleos
    setParallelism(1).
    setSeed(2627L)       // misma semilla que en el DT: mismos pliegues

  val inicio = System.nanoTime()
  val modelo = cv.fit(trainFinal)
  println(f"Tiempo de la validación cruzada: ${(System.nanoTime() - inicio) / 1e9}%.1f s")

  // Resultado de cada combinación (media de los 3 pliegues)
  println("Resultados:")
  modelo.getEstimatorParamMaps.zip(modelo.avgMetrics).foreach { case (pm, m) =>
    val config = pm.toSeq.map(pp => pp.param.name + " = " + pp.value).mkString(", ")
    println(f"  $config%-60s acierto = $m%1.4f  error = ${1 - m}%1.4f")
  }
  modelo
}


// Fase 1: búsqueda gruesa (minInstancesPerNode fijo en 50)
//  - numTrees: 20 y 50. Más árboles reduce la varianza, pero a partir de cierto número
//    apenas mejora y el tiempo crece en proporción
//  - maxDepth: 10, 15 y 20. En el DT el acierto subía hasta 20-25; en el RF el promedio
//    corrige el sobreajuste, pero los árboles muy profundos son mucho más lentos
//  - 2 x 3 = 6 combinaciones x 3 pliegues = 18 bosques

// Poner a false para saltarla (tarda unas 2 h) y usar los resultados ya obtenidos (en la memoria)
val ejecutarFase1 = true


// Mejor combinación de la fase 1 (se rellena en el if de abajo)
// Si se repite la fase 1, se sustituyen por los que salgan
var mejorArbolesFase1 = 0
var mejorProfundidadFase1 = 0
var mejorMinInstanciasFase1 = 0
var mejorAciertoFase1 = 0.0

// Mejor combinación de la fase 1: se calcula si se ejecuta la fase, y si no,
// se toman los valores de la ejecución (con esta misma semilla) ya hecha (resultados en la memoria)
if (ejecutarFase1) {
  val rejillaGruesa = new ParamGridBuilder().
    addGrid(RFtaxi.numTrees, Array(20, 50)).
    addGrid(RFtaxi.maxDepth, Array(10, 15, 20)).
    build()
  val cvGruesa = validacionCruzada(rejillaGruesa, "Fase 1: búsqueda gruesa")

  val mejorFase1 = cvGruesa.bestModel.asInstanceOf[RandomForestClassificationModel]
  mejorArbolesFase1 = mejorFase1.getNumTrees
  mejorProfundidadFase1 = mejorFase1.getMaxDepth
  mejorMinInstanciasFase1 = mejorFase1.getMinInstancesPerNode
  mejorAciertoFase1 = cvGruesa.avgMetrics.max
}else{
  mejorArbolesFase1 = 50
  mejorProfundidadFase1 = 20
  mejorMinInstanciasFase1 = 50
  mejorAciertoFase1 = 0.8085
}

println(f"\nMejor combinación de la fase 1: numTrees = $mejorArbolesFase1, " +
        f"maxDepth = $mejorProfundidadFase1, minInstancesPerNode = $mejorMinInstanciasFase1  " +
        f"(acierto = $mejorAciertoFase1%1.4f)")
println("La fase 2 busca alrededor de esa combinación")


// Fase 2: búsqueda fina a partir de lo visto en la fase 1
//  - numTrees fijo en 20: en la fase 1, pasar de 20 a 50 solo mejora 0,0009 (en el límite del
//    margen de ruido) y tarda más del doble
//  - maxDepth: el acierto seguía subiendo en 20, el borde de la rejilla, así que se prueba
//    más profundidad (30 es el máximo de Spark)
//  - minInstancesPerNode: en la fase 1 se fijó en 50 (el mejor del DT) para reducir la
//    rejilla. Como el acierto sigue subiendo al dar más capacidad a los árboles, y el
//    promedio del RF corrige el sobreajuste, se prueban también hojas más pequeñas (10)
//  - 1 x 3 x 2 = 6 combinaciones x 3 pliegues = 18 bosques
val rejillaFina = new ParamGridBuilder().
  addGrid(RFtaxi.numTrees, Array(20)).
  addGrid(RFtaxi.maxDepth, Array(20, 25, 30)).
  addGrid(RFtaxi.minInstancesPerNode, Array(10, 50)).
  build()
val cvModel = validacionCruzada(rejillaFina, "Fase 2: búsqueda fina")


// Modelo final: el mejor de la fase 2, ya reentrenado con todo el entrenamiento
val bestModel = cvModel.bestModel.asInstanceOf[RandomForestClassificationModel]
val mejorAcierto = cvModel.avgMetrics.max

println(s"\nMejor configuración: numTrees = ${bestModel.getNumTrees}, " +
        s"maxDepth = ${bestModel.getMaxDepth}, minInstancesPerNode = ${bestModel.getMinInstancesPerNode}")
println(f"Tasa de acierto media: $mejorAcierto%1.4f  (error = ${1 - mejorAcierto}%1.4f)")
println(s"Nodos totales del bosque: ${bestModel.totalNumNodes}")

// Comparación con la fase 1
// (numTrees no se compara: en la fase 2 se fija en 20 a propósito)
if (bestModel.getMaxDepth == mejorProfundidadFase1 && bestModel.getMinInstancesPerNode == mejorMinInstanciasFase1) {
  println("\nLa fase 2 confirma la profundidad y el tamaño de hoja de la fase 1")
} else {
  println(s"\nLa fase 2 cambia la mejor combinación: " +
          s"fase 1 (maxDepth = $mejorProfundidadFase1, minInstancesPerNode = $mejorMinInstanciasFase1) -> " +
          s"fase 2 (maxDepth = ${bestModel.getMaxDepth}, minInstancesPerNode = ${bestModel.getMinInstancesPerNode})")
}
println(f"Acierto: fase 1 = $mejorAciertoFase1%1.4f, fase 2 = $mejorAcierto%1.4f " +
        f"(diferencia = ${mejorAcierto - mejorAciertoFase1}%+1.4f)")

// Importancia de cada atributo en el bosque (cuánto reduce el Gini en total)
println("\nImportancia de los atributos:")
featureCols.zip(bestModel.featureImportances.toArray).sortBy(-_._2).foreach { case (a, imp) =>
  println(f"  $a%-20s $imp%1.4f")
}

// Comentar estas líneas para no sobrescribir el modelo ya guardado en ./modeloRF
bestModel.write.overwrite().save(RUTA_MODELO)
println(s"Modelo guardado en ${RUTA_MODELO}")
