package functionalModel

import org.apache.spark.SparkContext

/**
 * Created by philippy on 2015/12/14.
 */
object sampledata {
  def main(args:Array[String]): Unit = {
    val train = "data/train"
    val test = "data/test"
    val sc = new SparkContext("local", "NNtrain")
    var mlp = new multilayerPerceptron(4, Array(5, 4), 3,0.9,0)
    val trstr = sc.textFile(train).collect();
    val teststr = sc.textFile(test).collect();
    for(i<-0 to 5){
      trstr.foreach{line=>
        val parts = line.split(" ");
        var label = Array(0d,0d,0d)
        label(parts(0).toInt) = 1d
        var feature = Array(parts(1).split(":")(1).toDouble,parts(2).split(":")(1).toDouble,parts(3).split(":")(1).toDouble,parts(4).split(":")(1).toDouble)
        mlp.train(feature,label)
      }
    }

    var r = 0
    var t = 0
    teststr.foreach{line=>
      val parts = line.split(" ");
      var label = parts(0).toInt
      var feature = Array(parts(1).split(":")(1).toDouble,parts(2).split(":")(1).toDouble,parts(3).split(":")(1).toDouble,parts(4).split(":")(1).toDouble)
      var res = mlp.predict(feature)
      if(res(label)==res.max){
        r=r+1
        print(label+" ")
        res.foreach(print)
        println()
      }
      else{
        print("false "+label+" ")
        res.foreach(print)
        println()
      }
      t=t+1
    }
    println("r: = "+r+" t= "+t+" ratio: "+r.toDouble/t.toDouble)
  }
}
