package TestCase.CNNClassifier

import java.io.{FileOutputStream, ObjectOutputStream}

import CNNNet.CNN
import breeze.linalg.{DenseMatrix => DM, DenseVector => DV}
import org.apache.spark.SparkContext

/**
 * Created by philippy on 2016/2/12.
 */
object cnnLocalTrain {
  var net:CNN = new CNN
  var fos:FileOutputStream = _
  var oos:ObjectOutputStream = _

  def main(args:Array[String]): Unit = {
    var trainfile:Array[String] = Array("mnist_train1","mnist_train2","mnist_train3","mnist_train4","mnist_train5","mnist_train6")
    val dir:String = "/home/philippy/IdeaProjects/NeuralNetwork/data/"
    trainfile = trainfile.map(x=>dir+x)

    val sc = new SparkContext("local", "cnnLocalTrain")
    //local training
    val local_ite: Int = 5
    val local_eta: Array[Double] = Array(1,1,1,1,1)
    require(local_ite==local_eta.length)
    val start = System.nanoTime()
    var cnt = 0

/*  // load model from file
    val modelPath = "model/CNNlocal6.out"
    val fis1 = new FileInputStream(modelPath)
    val ois1 = new ObjectInputStream(fis1)

    val net:CNN = ois1.readObject.asInstanceOf[CNN]
*/
    //local training
    fos = new FileOutputStream("CNNtrainstart.out")
    oos = new ObjectOutputStream(fos)
    oos.writeObject(net)
    for(k<-0 to local_ite-1){ // 1 iterations = 60000 sample
      net.setEta(local_eta(k))
      for(i<- 0 to trainfile.length-1){ // 10000 samples
        val trstr: Array[String] = sc.textFile(trainfile(i)).collect()
        trstr.foreach{line=>
          val parts = line.split(",")
          val label: Int = parts(0).toInt
          val data: Array[Double] = parts.drop(1).map(_.toDouble/255)
          val labarr: Array[Double] = Array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
          labarr(label) = 1
          val input:DM[Double] = padDM(new DM(28,28,data),2)
          val target:DV[Double] = new DV(labarr)
          cnt = cnt+1
          net.train(input,target)
          println("sample:"+cnt+"err:"+net.localerr)
        }
      }
      fos = new FileOutputStream("CNNlocal"+k+".out")
      oos = new ObjectOutputStream(fos)
      oos.writeObject(net)
      println("your neural network is successfully saved local ite"+k+"! time:"+(System.nanoTime()-start).toString+" --finished--")
    }
    oos.close()
  }
  def padDM(mat:DM[Double], padding: Int):DM[Double] = {
    val vert = DM.fill(padding,mat.cols){0d}
    val horz = DM.fill(mat.rows+2*padding,padding){0d}
    DM.horzcat(horz,DM.vertcat(vert,mat,vert),horz)
  }
}
