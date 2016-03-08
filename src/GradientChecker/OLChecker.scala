package GradientChecker

import CNNLayer._
import breeze.linalg.{DenseMatrix=>DM,DenseVector=>DV}
/**
  * Created by philippy on 2016/3/5.
  */
object OLChecker {
   def main(args:Array[String]): Unit ={
     /*val layer = new OL(2,4)
     var w = DM.tabulate(layer.weight.rows,layer.weight.cols){case(i,j)=> layer.weight(i,j)}
     var target = new DV(Array(0.1d,0.2d,0.3d,0.4d))
     val in = new DV(Array(0.1d,0.2d))
     val ep = 0.005
     var output = layer.flattenOutput(layer.forwardLocal(layer.formatOutput(in)))
     layer.calErrLocal(target)
     //val v1:DM[Double] = layer.caladj()
     val v2:DM[Double] = DM.tabulate(v1.rows,v1.cols){case(i,j)=>
       layer.weight(i,j) = layer.weight(i,j)+ep
       var out1 = layer.flattenOutput(layer.forwardLocal(layer.formatOutput(in)))
       layer.weight(i,j) = layer.weight(i,j)-(ep*2)
       var out2 = layer.flattenOutput(layer.forwardLocal(layer.formatOutput(in)))
       layer.weight = w
       (layer.softmaxLoss(out1,target)-layer.softmaxLoss(out2,target))/(2*ep)
     }
     println(v1)
     println(v2)*/
   }

 }
