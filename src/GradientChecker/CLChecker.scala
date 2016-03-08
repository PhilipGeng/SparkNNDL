package GradientChecker

import CNNLayer.CL

/**
 * Created by philippy on 2016/3/5.
 */
object CLChecker {
  def main(args:Array[String]): Unit ={
    val c1_fm_in_map = Array(Array(0),Array(0),Array(0),Array(0),Array(0),Array(0))
    var c1:CL = new CL(6,5)
    c1.set_fm_input_map(c1_fm_in_map)

  }
}
