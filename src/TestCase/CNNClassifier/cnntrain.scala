package TestCase.CNNClassifier

import breeze.linalg.{DenseMatrix => DM, DenseVector => DV}

/**
 * Created by philippy on 2016/2/12.
 */
object cnntrain {
  def main(args:Array[String]): Unit = {val trainfile:Array[String] = Array("data/mnist_train1","data/mnist_train2","data/mnist_train3","data/mnist_train4","data/mnist_train5","data/mnist_train6")
  /*  val sc = new SparkContext("local", "NNtrain")
    var net:CNN = new CNN
    val vert = DM.fill(2,28){0d}
    val horz = DM.fill(32,2){0d}
    for(k<-0 to 5){
      for(i<-0 to 5) {
        println("training sample "+i+"*"+k)
        var trstr = sc.textFile(trainfile(i)).map{line=>
          val parts = line.split(",")
        }

        trstr.foreach { line =>
          val parts = line.split(",")
          val label: Int = parts(0).toInt
          val data: Array[Double] = parts.drop(1).map(_.toDouble / 255)
          var labarr: Array[Double] = Array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
          labarr(label) = 1
          val input = DM.horzcat(horz,DM.vertcat(vert,new DM(28,28,data),vert),horz)
          val target = new DV(labarr)
          net.train(input, target);
        }
      }
    }
    val fos = new FileOutputStream("CNNRDD.out")
    val oos = new ObjectOutputStream(fos)
    oos.writeObject(net)
    println("your neural network is successfully saved")
    oos.close()*/
  }
}
