package functionalModel

import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD

/**
 * Created by philippy on 2015/11/23.
 */
object sparkMlp {
  def main(args:Array[String]): Unit = {
    val sc = new SparkContext("local", "NNtrain")
    val sqlContext= new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._
    val data = MLUtils.loadLibSVMFile(sc, "/home/philippy/Programs/spark-1.5.1/data/mllib/sample_multiclass_classification_data.txt").toDF()
    // Split the data into train and test
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
    val train = splits(0)
    val test = splits(1)
    // specify layers for the neural network:
    // input layer of size 4 (features), two intermediate of size 5 and 4 and output of size 3 (classes)
    val layers = Array[Int](4, 5, 4, 3)
    // create the trainer and set its parameters
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100)
    // train the model
    val model = trainer.fit(train)
    // compute precision on the test set
    val result = model.transform(test)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("precision")
    println("Precision:" + evaluator.evaluate(predictionAndLabels))
  }
}
