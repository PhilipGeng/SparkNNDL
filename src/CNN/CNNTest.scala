package CNN

/**
 * Created by philippy on 2016/1/8.
 */
import breeze.linalg.{DenseVector => DV, DenseMatrix => DM, sum}
object CNNTest {
  def main(args:Array[String]): Unit ={/*
    val id = 5
    val cd = 3
    val rd = id-cd+1
    val image = Array(1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1).map(_.toDouble)
    val conv = Array(1,0,1,0,1,0,1,0,1).map(_.toDouble)
    val rdata = Array(0,0,0,0,0,0,0,0,0,0).map(_.toDouble)
    val im = new DM(id,id,image)
    val cm = new DM(cd,cd,conv)
    var rm = new DM(rd,rd,rdata)
    for(i<-0 to id-cd){
      for(j<-0 to id-cd){
        val res = im(i to i+cd-1,j to j+cd-1):*cm
        rm(i,j) = sum(res)
      }
    }
    println(rm)
    var net = new CL(1,3)
    net.conv = Array(cm)
    println(net.forward(im)(0))
*/
    val image = Array(1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1).map(_.toDouble)
    val im = new DM(4,4,image)
    var net = new SL(2)
    println(im)
    println(net.forward(Array(im))(0))
  }
}
