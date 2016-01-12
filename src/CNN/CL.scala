package CNN

/**
 * Created by philippy on 2016/1/8.
 */
import breeze.linalg.{DenseVector=>DV,DenseMatrix=>DM,sum}
class CL(val numfm:Int,val dim_conv:Int) extends Serializable {
  var conv:Array[DM[Double]] = Array.fill(numfm){DM.rand(dim_conv,dim_conv):*=2d:-=1d}
  var bias:Array[Double] = Array.fill(numfm){scala.util.Random.nextDouble()*2d-1d}
  def forward(input:DM[Double]):Array[DM[Double]]={
    val rd = input.cols-dim_conv+1
    var rm:Array[DM[Double]] = Array.fill(numfm){DM.zeros(rd,rd)}
    if(input.cols!=input.rows){
      print("error input")
      return rm
    }
    for(m<-0 to numfm-1){
      for(i<- 0 to rd-1){
        for(j<- 0 to rd-1){
          rm(m)(i,j) = sum(input(i to i+dim_conv-1,j to j+dim_conv-1):*conv(m))
        }
      }
    }
    return rm
  }
}
