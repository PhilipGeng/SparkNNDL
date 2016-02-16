package CNN

import breeze.linalg.{DenseVector => DV, DenseMatrix => DM, sum}
import org.apache.spark.rdd.RDD

/**
 * Created by philippy on 2016/1/8.
 */
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

  /**
   * single node mode
   * @param input image
   * @param target label
   */
  def train(input:DM[Double],target:DV[Double]): Unit = train(Array(input),target)
  def train(input:Array[DM[Double]],target:DV[Double]): Unit ={
    val o7res = classify(input)
    val Err = sum(o7res-target)
    println(Err)

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

  def classify(input:DM[Double]):DV[Double]=classify(Array(input))
  def classify(input:Array[DM[Double]]): DV[Double] ={
    val c1res:Array[DM[Double]] = c1.forward(input)
    val s2res:Array[DM[Double]] = s2.forward(c1res)
    val c3res:Array[DM[Double]] = c3.forward(s2res)
    val s4res:Array[DM[Double]] = s4.forward(c3res)
    val c5res:Array[DM[Double]] = c5.forward(s4res)
    val f6res:Array[DM[Double]] = f6.forward(c5res)
    val o7res:Array[DM[Double]] = o7.forward(f6res)
    c1.flattenOutput(o7res)
  }

  /**
   * parallel training
   * @param input RDD[image,label]
   */
  def parallelTrain(input:RDD[(DM[Double],DV[Double])]): Unit ={

  }
  def classify(input:RDD[DM[Double]]): DV[Double] ={
    val c1res:Array[DM[Double]] = c1.forward(input)
    val s2res:Array[DM[Double]] = s2.forward(c1res)
    val c3res:Array[DM[Double]] = c3.forward(s2res)
    val s4res:Array[DM[Double]] = s4.forward(c3res)
    val c5res:Array[DM[Double]] = c5.forward(s4res)
    val f6res:Array[DM[Double]] = f6.forward(c5res)
    val o7res:Array[DM[Double]] = o7.forward(f6res)
    c1.flattenOutput(o7res)
  }

}
