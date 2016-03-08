package TestCase.CNNClassifier

import java.io.{FileOutputStream, ObjectOutputStream}

import CNNNet.CNN
import breeze.linalg.{DenseMatrix => DM, DenseVector => DV}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by philippy on 2016/2/12.
 */
object cnnBatchTrain {
  var net:CNN = new CNN
  var fos: FileOutputStream = _
  var oos: ObjectOutputStream = _
  def main(args:Array[String]): Unit = {
    var trainfile:Array[String] = Array("mnist_train1","mnist_train2","mnist_train3","mnist_train4","mnist_train5","mnist_train6")
    val dir:String = "hdfs:///mnist/"
    trainfile = trainfile.map(x=>dir+x)
    val conf = new SparkConf().setAppName("CNN batch train")
    val sc = new SparkContext(conf)
    //cluster training
    val cluster_ite: Int = 5
    val cluster_eta: Array[Double] = Array(0.1,0.07,0.05,0.05,0.05)
    val cluster_updateWhenWrong: Array[Boolean] = Array(false,false,false,true,true)
    require(cluster_ite == cluster_eta.length)
    require(cluster_ite == cluster_updateWhenWrong.length)

    val start = System.nanoTime()
    fos = new FileOutputStream("CNNtrainstart.out")
    oos = new ObjectOutputStream(fos)
    oos.writeObject(net)

    //net.setNumPartition(9)

    //cluster
    for(k<-0 to cluster_ite-1){
      net.setEta(cluster_eta(k))
      net.setUpdateWhenWrong(cluster_updateWhenWrong(k))
      for(i<- 0 to trainfile.length-1){ // 10000 samples
        println("cluster training: ite "+k+" sample "+i+"0000")
        val trstr: RDD[(DM[Double],DV[Double])] = sc.textFile(trainfile(i)).map{rddline=>
          val arr: Array[String] = rddline.split(",")
          val label: Int = arr(0).toInt
          val data = arr.drop(1).map(_.toDouble/255)
          val labarr: Array[Double] = Array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
          labarr(label) = 1
          val input:DM[Double] = padDM(new DM(28,28,data),2)
          val target:DV[Double] = new DV(labarr)
          (input,target)
        }
        val in:RDD[DM[Double]] = trstr.map(_._1)
        val t:RDD[DV[Double]] = trstr.map(_._2)
        net.train(in,t)
      }
      fos = new FileOutputStream("CNNclusterIte"+k+"t"+(System.nanoTime()-start).toString+".out")
      oos = new ObjectOutputStream(fos)
      oos.writeObject(net)
      println("your neural network is successfully saved cluster ite"+k+"! time:"+(System.nanoTime()-start).toString+" --finished--")
    }
    oos.close()
  }
  def padDM(mat:DM[Double], padding: Int):DM[Double] = {
    val vert = DM.fill(padding,mat.cols){0d}
    val horz = DM.fill(mat.rows+2*padding,padding){0d}
    DM.horzcat(horz,DM.vertcat(vert,mat,vert),horz)
  }
}
