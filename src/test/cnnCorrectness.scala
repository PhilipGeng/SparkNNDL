package test

import java.io.{FileOutputStream, ObjectOutputStream}

import CNN.CNN
import breeze.linalg.{DenseMatrix => DM, DenseVector => DV}

/**
 * Created by philippy on 2016/1/19.
 */
object cnnCorrectness {
  def main(args:Array[String]): Unit ={

    val net:CNN = new CNN
    var cnt = 0
    val input:Array[DM[Double]] = Array(DM.fill(32,32){1})
    val target:DV[Double] = new DV(Array(1d,0d,0d,0d,0d,0d,0d,0d,0d,0d))

    for(i<-0 to 10000){
      println("cnt:"+cnt)
      cnt = cnt+1
      net.train(input,target)
    }

    val fos = new FileOutputStream("CNNcorrectness.out")
    val oos = new ObjectOutputStream(fos)
    oos.writeObject(net)
    println("your neural network is successfully saved")
    oos.close()
  }
}
