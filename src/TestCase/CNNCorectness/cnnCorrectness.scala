package TestCase.CNNCorectness

import java.io._

import CNNNet.CNN
import breeze.linalg.{DenseMatrix => DM, DenseVector => DV}
import org.apache.spark._
import org.apache.spark.rdd.RDD

/**
 * Created by philippy on 2016/1/19.
 */

object cnnCorrectness {
  def main(args:Array[String]): Unit ={
    val sc = new SparkContext("local","SparkCNN")
    var cnt = 0
    val inputrdd:RDD[Array[DM[Double]]] = sc.parallelize(Array(Array(DM.fill(32,32){1d}),Array(DM.fill(32,32){1d})))
    val targetrdd:RDD[DV[Double]] = sc.parallelize(Array(new DV(Array(1d,0d,0d,0d,0d,0d,0d,0d,0d,0d)),new DV(Array(1d,0d,0d,0d,0d,0d,0d,0d,0d,0d))))
    val input:Array[DM[Double]] = Array(DM.fill(32,32){1d})
    val target:DV[Double] = new DV(Array(1d,0d,0d,0d,0d,0d,0d,0d,0d,0d))

    val net:CNN = new CNN
    val t1 = System.nanoTime()

   for(i<-0 to 150){
       println("cnt:"+cnt)
       cnt = cnt+1
       val e = net.train(input,target)
     }
    net.setEta(0.25)
    for(i<-0 to 100){
      println("cnt:"+cnt)
      cnt = cnt+1
      net.train(inputrdd,targetrdd)
    }
    val t2 = System.nanoTime()
    val exec_time = t2-t1

    val file = "error.txt"
    val writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file)))
    val errarr = net.getErr()
    for (x <- errarr) {
      writer.write(x + ",")  // however you want to format it
    }
    writer.write("\n"+exec_time+",")
    writer.close()
  }

}
