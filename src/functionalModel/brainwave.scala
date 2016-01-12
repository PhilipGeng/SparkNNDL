package functionalModel

import org.apache.spark.SparkContext

/**
 * Created by philippy on 2015/12/21.
 */
object brainwave {
  def main(args:Array[String]): Unit = {
    val sc = new SparkContext("local", "NNtrain")
    var mlp = new multilayerPerceptron(128, Array(300), 2, 0.9, 0.2)
    val data = sc.textFile("/home/philippy/Desktop/data.txt").collect().map { line =>
      line.split("\t").map(_.toDouble);
    }
    val label = sc.textFile("/home/philippy/Desktop/label.txt").collect().map { line =>
      line.toInt - 1
    }
    val dataset = sc.parallelize(label.zip(data))
    val split = dataset.randomSplit(Array(0.8, 0.2))
    val training = split(0).collect()
    val testing = split(1).collect()
    println("training:"+training.length)
    println("test:"+testing.length)
    for (i <- 1 to 500){
      println(i)
      training.foreach { item =>
        var l = Array(0d,0d)
        l(item._1.toInt) = 1d
        mlp.train(item._2, l)
      }
    }
    var cnt = 0;
    var correct = 0;
    testing.foreach{item =>
      var res = mlp.predict(item._2)
      if(res(item._1)==res.max)
        correct = correct+1
      cnt=cnt+1
    }
    println("cnt:"+cnt);
    println("correct:"+correct)
  }
}
