package test

import java.io.{FileInputStream, ObjectInputStream}

import CNN.CNN
import breeze.linalg.{DenseMatrix => DM, DenseVector => DV}
import org.apache.spark.SparkContext

/**
 * Created by philippy on 2016/2/12.
 */
object cnnclassify {
  def main(args:Array[String]): Unit ={

    val fis = new FileInputStream("CNNMnist3ite.out")
    val ois = new ObjectInputStream(fis)

    val net:CNN = ois.readObject.asInstanceOf[CNN]

    val sc = new SparkContext("local", "NNtest")
    val testfile = "data/mnist_test.csv"
    var cnt = 0
    var correct = 0
    val vert = DM.fill(2,28){0d}
    val horz = DM.fill(32,2){0d}

    var teststr = sc.textFile(testfile).collect()
    teststr.foreach { line =>
      val parts = line.split(",")
      val label: Int = parts(0).toInt
      val data: Array[Double] = parts.drop(1).map(_.toDouble / 255)
      val input = DM.horzcat(horz,DM.vertcat(vert,new DM(28,28,data),vert),horz)
      val res = net.classify(input).toArray
      var resstr = "wrong"
      if(res.max == res(label)){
        resstr = "correct"
        correct+=1
      }
      cnt=cnt+1
      println("testing sample "+cnt+"  "+resstr)
    }
    println("correct rate: "+correct.toDouble/cnt.toDouble)
  }

}
