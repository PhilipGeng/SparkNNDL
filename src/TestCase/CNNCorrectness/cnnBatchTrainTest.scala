package TestCase.CNNCorrectness

import java.io.{FileOutputStream, ObjectOutputStream}

import CNNNet.CNN
import breeze.linalg.{DenseMatrix => DM, DenseVector => DV}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD

/**
 * Created by philippy on 2016/2/12.
 */
object cnnBatchTrainTest {
  var net:CNN = new CNN
  def main(args:Array[String]): Unit = {
    var trainfile:Array[String] = Array("mnist_train1","mnist_train2","mnist_train3","mnist_train4","mnist_train5","mnist_train6")
   // val dir:String = "hdfs:///mnist/"
    val dir:String = "/home/philippy/IdeaProjects/NeuralNetwork/data/"
    trainfile = trainfile.map(x=>dir+x)
    val conf = new SparkConf().setAppName("CNN batch train").setMaster("local")
    val sc = new SparkContext(conf)
    //local training
    val local_ite: Int = 3
    val local_eta: Array[Double] = Array(1,1,1)
    //cluster training
    val cluster_ite: Int = 5
    val cluster_eta: Array[Double] = Array(0.5,0.5,0.5,0.5,0.5)

    val start = System.nanoTime()
    //var fos = new FileOutputStream("CNNtrain.out")
    //var oos = new ObjectOutputStream(fos)

    net.setUpdateWhenWrong(true)
    net.setNumPartition(9)
    //cluster
    for(k<-0 to 0){
    //for(k<-0 to cluster_ite-1){
      net.setEta(cluster_eta(k))
      for(i<- 0 to 1){
        println("cluster training: ite "+k+" sample "+i+"0000")
        val trstr: RDD[(DM[Double],DV[Double])] = sc.parallelize(sc.textFile(trainfile(i)).take(15)).map{rddline=>
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
      //fos = new FileOutputStream("CNNclusterIte"+k+"t"+(System.nanoTime()-start).toString+".out")
      //oos = new ObjectOutputStream(fos)
      //oos.writeObject(net)
      println("your neural network is successfully saved cluster ite"+k+"! time:"+(System.nanoTime()-start).toString+" --finished--")
    }
    //oos.close()
  }
  def padDM(mat:DM[Double], padding: Int):DM[Double] = {
    val vert = DM.fill(padding,mat.cols){0d}
    val horz = DM.fill(mat.rows+2*padding,padding){0d}
    DM.horzcat(horz,DM.vertcat(vert,mat,vert),horz)
  }
}
