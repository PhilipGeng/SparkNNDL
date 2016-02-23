package CNNNet

import CNNLayer.{CL, FL, OL, SL}
import breeze.linalg.{DenseMatrix => DM, DenseVector => DV, sum}
import org.apache.spark.rdd.RDD

/**
 * Created by philippy on 2016/1/8.
 */
@SerialVersionUID(-1000L)
class CNN extends Serializable{
  val c1_fm_in_map = Array(Array(0),Array(0),Array(0),Array(0),Array(0),Array(0))
  val c3_fm_in_map = Array(Array(0,1,2),Array(1,2,3),Array(2,3,4),Array(3,4,5),Array(4,5,0),Array(5,0,1),Array(0,1,2,3),Array(1,2,3,4),Array(2,3,4,5),Array(3,4,5,0),Array(4,5,0,1),Array(5,0,1,2),Array(0,1,3,4),Array(1,2,4,5),Array(0,2,3,5),Array(0,1,2,3,4,5))
  val c5_fm_in_map = Array.fill[Array[Int]](120){(0 to 15).toArray}
  var c1:CL = new CL(6,5)
  c1.set_fm_input_map(c1_fm_in_map)
  var s2:SL = new SL(6,2)
  var c3:CL = new CL(16,5)
  c3.set_fm_input_map(c3_fm_in_map)
  var s4:SL = new SL(16,2)
  var c5:CL = new CL(120,5)
  c5.set_fm_input_map(c5_fm_in_map)
  var f6:FL = new FL(120,84)
  var o7:OL = new OL(84,10)
  var network = Array(c1,s2,c3,s4,c5,f6,o7)
  var errarr: Array[Double] = Array()
  /**
   * single node mode
   * @param input image
   * @param target label
   */
  def train(input:RDD[Array[DM[Double]]], target:RDD[DV[Double]]): Unit ={
    val fpres:RDD[DV[Double]] = classify(input)
    val a:Array[DV[Double]] = fpres.collect()
    val b:Array[DV[Double]] = target.collect()
    val err:DV[Double] = a(0)-b(0)
    val Error:Double = sum(err:*err)

    errarr = errarr++Array(Error)

    o7.calErr(target)
    f6.calErr(o7)
    c5.calErr(f6)
    s4.calErr(c5)
    c3.calErr(s4)
    s2.calErr(c3)
    c1.calErr(s2)

    c1.adjWeight()
    s2.adjWeight()
    c3.adjWeight()
    s4.adjWeight()
    c5.adjWeight()
    f6.adjWeight()
    o7.adjWeight()

  }

  def train(input:Array[DM[Double]], target:DV[Double]): Double ={
    val fpres:DV[Double] = classify(input)
    val err:DV[Double] = fpres-target
    val Error:Double = sum(err:*err)

    println(Error)

    o7.calErrLocal(target)
    f6.calErrLocal(o7)
    c5.calErrLocal(f6)
    s4.calErrLocal(c5)
    c3.calErrLocal(s4)
    s2.calErrLocal(c3)
    c1.calErrLocal(s2)

    c1.adjWeightLocal()
    s2.adjWeightLocal()
    c3.adjWeightLocal()
    s4.adjWeightLocal()
    c5.adjWeightLocal()
    f6.adjWeightLocal()
    o7.adjWeightLocal()
    Error
  }

  def classify(input:RDD[Array[DM[Double]]]):RDD[DV[Double]] = {
    val c1res:RDD[Array[DM[Double]]] = c1.forward(input)
    val s2res:RDD[Array[DM[Double]]] = s2.forward(c1res)
    val c3res:RDD[Array[DM[Double]]] = c3.forward(s2res)
    val s4res:RDD[Array[DM[Double]]] = s4.forward(c3res)
    val c5res:RDD[Array[DM[Double]]] = c5.forward(s4res)
    val f6res:RDD[Array[DM[Double]]] = f6.forward(c5res)
    val o7res:RDD[Array[DM[Double]]] = o7.forward(f6res)
    o7res.map(x=>o7.flattenOutput(x))
  }

  def classify(input:Array[DM[Double]]):DV[Double] = {
    val c1res:Array[DM[Double]] = c1.forwardLocal(input)
    val s2res:Array[DM[Double]] = s2.forwardLocal(c1res)
    val c3res:Array[DM[Double]] = c3.forwardLocal(s2res)
    val s4res:Array[DM[Double]] = s4.forwardLocal(c3res)
    val c5res:Array[DM[Double]] = c5.forwardLocal(s4res)
    val f6res:Array[DM[Double]] = f6.forwardLocal(c5res)
    val o7res:Array[DM[Double]] = o7.forwardLocal(f6res)
    c1.flattenOutput(o7res)
  }

  def setEta(eta:Double): Unit ={
    c1.seteta(eta)
    s2.seteta(eta)
    c3.seteta(eta)
    s4.seteta(eta)
    c5.seteta(eta)
    f6.seteta(eta)
    o7.seteta(eta)
  }

  def getErr(): Array[Double] = errarr
}
