package functionalModel

import org.apache.spark.SparkContext
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD

/**
 * Created by philippy on 2015/11/23.
 */
object brainwave {
  def main(args:Array[String]): Unit = {
    val r = scala.util.Random
    val sc = new SparkContext("local", "NN");
    var datastr = sc.textFile("/home/philippy/Desktop/data.txt");
    var labelstr = sc.textFile("/home/philippy/Desktop/label.txt");
    var data: RDD[Array[Double]] = datastr.map { line =>
      line.split("\t").map(_.toDouble)
    }
    var label: RDD[Double] = labelstr.map { line =>
      line.toDouble * 2 - 3
    }
    var dataset = data.zip(label);
    var nn = new mlnn(128, Array(300), 1, 1);

    for (i <- 1 to 10) {
      val splits = dataset.randomSplit(Array(0.8, 0.2), seed = System.currentTimeMillis);

      val trainingSet: Array[(Array[Double], Double)] = splits(0).collect()
      val testSet: Array[(Array[Double], Double)] = splits(1).collect()

      println("trainingset length:" + trainingSet.length)
      println("testset length:" + testSet.length)
      println("feature length:" + trainingSet(0)._1.length)
      println("label:" + trainingSet(0)._2)

      for (i <- 0 to trainingSet.length - 1) {
        nn.train(trainingSet(i)._1, Array(trainingSet(i)._2))
      }

      var right = 0;
      for (i <- 0 to testSet.length - 1) {
        var r: Double = nn.predict(testSet(i)._1)(0)
        var rev: Double = 0;
        if (r < 0.5)
          rev = -1.0
        else
          rev = 1.0
        if (rev == testSet(i)._2)
          right = right + 1
      }
      println(right.toDouble / testSet.length.toDouble)
    }
  }
}
