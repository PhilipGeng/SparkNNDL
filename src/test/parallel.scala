package test

/**
 * Created by philippy on 2016/2/14.
 */

import java.io.{ObjectOutputStream, FileOutputStream}

import CNN.CNN
import breeze.linalg.{DenseVector=>DV, DenseMatrix=>DM}
import org.apache.spark._
import org.apache.spark.rdd.RDD

object parallel {
  def main(args:Array[String]): Unit = {
    val sc = new SparkContext("local","SparkCNN")
    val trainfile:Array[String] = Array("data/mnist_train1","data/mnist_train2","data/mnist_train3","data/mnist_train4","data/mnist_train5","data/mnist_train6")
    var net:CNN = new CNN
    val vert = DM.fill(2,28){0d}
    val horz = DM.fill(32,2){0d}
    var cnt = 0
    for(k<-0 to 2){
      for(i<-0 to 5) {
        println("training sample "+i+"*"+k)
        val trstr:RDD[String] = sc.textFile(trainfile(i))
        val input_tuple:RDD[(DM[Double],DV[Double])] = trstr.map { line =>
          val parts = line.split(",")
          val label: Int = parts(0).toInt
          val data: Array[Double] = parts.drop(1).map(_.toDouble / 255)
          var labarr: Array[Double] = Array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
          labarr(label) = 1
          val input = DM.horzcat(horz,DM.vertcat(vert,new DM(28,28,data),vert),horz)
          val target = new DV(labarr)
          (input, target)
        }
      }
    }
    val fos = new FileOutputStream("CNNMnist3ite.out")
    val oos = new ObjectOutputStream(fos)
    oos.writeObject(net)
    println("your neural network is successfully saved")
    oos.close()
    sc.stop()

  }

}
