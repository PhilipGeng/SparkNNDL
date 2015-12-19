package functionalModel

import org.apache.spark.SparkContext
import breeze.linalg.{*, DenseMatrix => DM, DenseVector => DV}

/**
 * Created by philippy on 2015/12/15.
 */
object correctness {
  def main(args:Array[String]): Unit ={
    val sc = new SparkContext("local", "NNtrain")
    var mlp = new multilayerPerceptron(3, Array(2),1,0.9,0d)
    mlp.weight(0) = new DM(4,3)
    mlp.weight(1) = new DM(3,2)
    mlp.weight(0)(0,0) = 0
    mlp.weight(0)(0,1) = -0.4
    mlp.weight(0)(0,2) = 0.2
    mlp.weight(0)(1,0) = 0
    mlp.weight(0)(1,1) = 0.2
    mlp.weight(0)(1,2) = -0.3
    mlp.weight(0)(2,0) = 0
    mlp.weight(0)(2,1) = 0.4
    mlp.weight(0)(2,2) = 0.1
    mlp.weight(0)(3,0) = 0
    mlp.weight(0)(3,1) = -0.5
    mlp.weight(0)(3,2) = 0.2
    mlp.weight(1)(0,0) = 0
    mlp.weight(1)(0,1) = 0.1
    mlp.weight(1)(1,0) = 0
    mlp.weight(1)(1,1) = -0.3
    mlp.weight(1)(2,0) = 0
    mlp.weight(1)(2,1) = -0.2
    mlp.layers(0)(0)=0
    mlp.layers(1)(0)=0
    mlp.layers(2)(0)=0

    mlp.train(Array(1,0,1),Array(1))
    println(mlp.weight(0)(0,0))
    println(mlp.weight(0)(0,1))
    println(mlp.weight(0)(0,2))
    println(mlp.weight(0)(1,0))
    println(mlp.weight(0)(1,1))
    println(mlp.weight(0)(1,2))
    println(mlp.weight(0)(2,0))
    println(mlp.weight(0)(2,1))
    println(mlp.weight(0)(2,2))
    println(mlp.weight(0)(3,0))
    println(mlp.weight(0)(3,1))
    println(mlp.weight(0)(3,2))
  }
}
