package CNN

/**
 * Created by philippy on 2016/1/8.
 */
import breeze.linalg.{DenseVector=>DV, DenseMatrix=>DM, sum}
class SL (val dim_neightbor:Int){
  def forward(input:Array[DM[Double]]): Array[DM[Double]] ={
    val rm_dim:Int = input(0).cols/2
    var rm:Array[DM[Double]] = Array.fill(input.length){DM.zeros(rm_dim,rm_dim)}
    for(k<-0 to input.length-1){
      val im = input(k)
      for(i<- 0 to rm_dim-1){
        for(j<- 0 to rm_dim-1){
          rm(k)(i,j) = im(i*2,j*2)+im(i*2+1,j*2)+im(i*2,j*2+1)+im(i*2+1,j*2+1)
        }
      }
    }
    return rm
  }
}
