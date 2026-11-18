// Validación cruzada del árbol de decisión y guarda ./modeloDT
//
// Lanzar spark-shell con más memoria para el driver (en modo local hace también de executor):
//   spark-shell --driver-memory 8g
// La máquina tiene 15 GiB (11 GiB libres); por defecto Spark solo usa 1 GiB y los árboles
// profundos pueden agotarla. 8 GiB deja margen para el sistema operativo.

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.ml.classification.{DecisionTreeClassifier, DecisionTreeClassificationModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.param.ParamMap

:load preparacionDatos.scala

// Carpeta donde se guarda el modelo final (en el enunciado se pide ./modeloXX)
val RUTA_MODELO = "./modeloDT"

// Se cachea porque la validación cruzada lo recorre muchas veces
trainFinal.cache()
println(s"Viajes de entrenamiento: ${trainFinal.count()}")

// Ya no hacen falta: trainFinal está en caché y el test no se usa al crear el modelo
trainClean.unpersist()
testClean.unpersist()

// Métrica de selección: tasa de acierto (= 1 - error, la métrica principal del enunciado)
val metrics = new MulticlassClassificationEvaluator()
metrics.setMetricName("accuracy")

// Parámetros fijos (no entran en la rejilla): se escriben aunque sean los valores por defecto
val DTtaxi = new DecisionTreeClassifier().
  setMaxBins(32).        // el entregable individual mostró que apenas influye en el error
  setImpurity("gini").   // criterio de división por defecto
  setMinInfoGain(0.0)    // sin ganancia mínima: el tamaño de las hojas lo controla minInstancesPerNode

println(s"maxBins = ${DTtaxi.getMaxBins}, impurity = ${DTtaxi.getImpurity}, minInfoGain = ${DTtaxi.getMinInfoGain}")


// Lanza la validación cruzada con una rejilla y muestra el resultado de cada combinación
def validacionCruzada(rejilla: Array[ParamMap], titulo: String): CrossValidatorModel = {
  println(s"\n$titulo (${rejilla.length} combinaciones)")

  val cv = new CrossValidator().
    setEstimator(DTtaxi).
    setEvaluator(metrics).
    setEstimatorParamMaps(rejilla).
    // 3 pliegues: cada uno valida con ~388 000 viajes, suficiente para una estimación estable.
    // Con 5 o 10 la estimación apenas mejoraría y el tiempo se multiplicaría
    setNumFolds(3).
    // 2 árboles a la vez: con 8 GiB de driver funciona y tarda unas 3 veces menos que de uno
    // en uno. Más de 2 apenas acelera (la máquina tiene 4 núcleos, que se reparten entre los
    // modelos) y aumenta el riesgo de quedarse sin memoria con los árboles profundos
    setParallelism(2).
    setSeed(2627L)       // mismos pliegues en cada ejecución (resultados reproducibles)

  val inicio = System.nanoTime()
  val modelo = cv.fit(trainFinal)
  println(f"Tiempo de la validación cruzada: ${(System.nanoTime() - inicio) / 1e9}%.1f s")

  // Resultado de cada combinación (media de los 3 pliegues)
  println("Resultados:")
  modelo.getEstimatorParamMaps.zip(modelo.avgMetrics).foreach { case (pm, m) =>
    val config = pm.toSeq.map(pp => pp.param.name + " = " + pp.value).mkString(", ")
    println(f"  $config%-45s acierto = $m%1.4f  error = ${1 - m}%1.4f")
  }
  modelo
}


// Fase 1: búsqueda gruesa
//  - maxDepth de 5 (por defecto) a 25, en pasos de 5
//  - minInstancesPerNode: 1 (por defecto, sin restricción), 50, 100 y 1000, para ver
//    desde árboles sin límite (sobreajuste) hasta hojas muy grandes (subajuste)

// Poner a false para saltarla (tarda alrededor de 1 h) y usar los resultados ya obtenidos (en la memoria)
val ejecutarFase1 = true

// Mejor combinación de la fase 1 (se rellena en el if de abajo)
var mejorProfundidadFase1 = 0
var mejorMinInstanciasFase1 = 0
var mejorAciertoFase1 = 0.0

// Mejor combinación de la fase 1: se calcula si se ejecuta la fase, y si no,
// se toman los valores de la ejecución (con esta misma semilla) ya hecha (resultados en la memoria)
if (ejecutarFase1) {
  val rejillaGruesa = new ParamGridBuilder().
    addGrid(DTtaxi.maxDepth, Array(5, 10, 15, 20, 25)).
    addGrid(DTtaxi.minInstancesPerNode, Array(1, 50, 100, 1000)).
    build()
  val cvGruesa = validacionCruzada(rejillaGruesa, "Fase 1: búsqueda gruesa")

  val mejorFase1 = cvGruesa.bestModel.asInstanceOf[DecisionTreeClassificationModel]
  mejorProfundidadFase1 = mejorFase1.getMaxDepth
  mejorMinInstanciasFase1 = mejorFase1.getMinInstancesPerNode
  mejorAciertoFase1 = cvGruesa.avgMetrics.max
}else{
  mejorProfundidadFase1 = 25
  mejorMinInstanciasFase1 = 50
  mejorAciertoFase1 = 0.7993
}

println(f"\nMejor combinación de la fase 1: maxDepth = $mejorProfundidadFase1, " +
        f"minInstancesPerNode = $mejorMinInstanciasFase1  (acierto = $mejorAciertoFase1%1.4f)")
println("La fase 2 busca alrededor de esa combinación")


// Fase 2: búsqueda fina alrededor de lo mejor de la fase 1
//  - La fase 1 da el mejor resultado con profundidad 20-25 y un mínimo de 50-100 viajes
//    por hoja; con mínimo 1 los árboles profundos sobreajustan y con 1000 pierden detalle
//  - maxDepth: valores alrededor de 20-25
//  - minInstancesPerNode: valores entre 25 y 150, alrededor de 50-100
val rejillaFina = new ParamGridBuilder().
  addGrid(DTtaxi.maxDepth, Array(18, 21, 23, 25)).
  addGrid(DTtaxi.minInstancesPerNode, Array(25, 50, 75, 150)).
  build()
val cvModel = validacionCruzada(rejillaFina, "Fase 2: búsqueda fina")


// Modelo final: el mejor de la fase 2, ya reentrenado con todo el entrenamiento
val bestModel = cvModel.bestModel.asInstanceOf[DecisionTreeClassificationModel]
val mejorAcierto = cvModel.avgMetrics.max

println(s"\nMejor configuración: maxDepth = ${bestModel.getMaxDepth}, " +
        s"minInstancesPerNode = ${bestModel.getMinInstancesPerNode}")
println(f"Tasa de acierto media: $mejorAcierto%1.4f  (error = ${1 - mejorAcierto}%1.4f)")
println(s"Profundidad real: ${bestModel.depth}  Nodos: ${bestModel.numNodes}  maxBins: ${bestModel.getMaxBins}")


// Comparación con la fase 1
if (bestModel.getMaxDepth == mejorProfundidadFase1 && bestModel.getMinInstancesPerNode == mejorMinInstanciasFase1){
  println("\nLa fase 2 confirma la mejor combinación de la fase 1")
}else{
  println(s"\nLa fase 2 cambia la mejor combinación: " +
          s"fase 1 (maxDepth = $mejorProfundidadFase1, minInstancesPerNode = $mejorMinInstanciasFase1) -> " +
          s"fase 2 (maxDepth = ${bestModel.getMaxDepth}, minInstancesPerNode = ${bestModel.getMinInstancesPerNode})")
}
println(f"Acierto: fase 1 = $mejorAciertoFase1%1.4f, fase 2 = $mejorAcierto%1.4f " +
        f"(diferencia = ${mejorAcierto - mejorAciertoFase1}%+1.4f)")

// Importancia de cada atributo en el árbol (cuánto reduce el Gini en total)
println("\nImportancia de los atributos:")
featureCols.zip(bestModel.featureImportances.toArray).sortBy(-_._2).foreach { case (a, imp) =>
  println(f"  $a%-20s $imp%1.4f")
}

// Comentar estas líneas para no sobrescribir el modelo ya guardado en ./modeloDT
bestModel.write.overwrite().save(RUTA_MODELO)
println(s"Modelo guardado en ${RUTA_MODELO}")