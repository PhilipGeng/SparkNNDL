package CNN

/**
 * @author Philip GENG
 *         gengxu.heartfire@gmail.com
 * This is the sub-sampling layer in CNN
 */
import breeze.linalg.{DenseVector=>DV, DenseMatrix=>DM, sum}
import breeze.numerics.sigmoid

/**
 *
 * @param numfm number of featuremaps
 * @param dim_neighbor width/height of the subsampling block
 * @param eta learning rate
 */
class SL (val numfm: Int, val dim_neighbor:Int, val eta:Double=0.5) extends layer{
  var weight:Array[Double] = Array.fill(numfm){scala.util.Random.nextDouble()*2-1d}
  var bias:Array[Double] = Array.fill(numfm){scala.util.Random.nextDouble()*2-1d}
  var output:Array[DM[Double]]=_
  var delta:Array[DM[Double]]=_
  var input:Array[DM[Double]]=_
  var weightAdj:List[Array[Double]] = _
  var biasAdj:List[Array[Double]] = _

  /**
   * forward
   * @param input_arr input image
   * @return output featuremap
   */
  override def forward(input_arr:Array[DM[Double]]): Array[DM[Double]] ={
    input = input_arr
    val rm_dim:Int = input(0).cols/dim_neighbor
    val rm:Array[DM[Double]] = Array.fill(input.length){DM.zeros(rm_dim,rm_dim)}
    for(k<-input.indices){
      val im = input(k)
      for(i<- 0 to rm_dim-1){
        for(j<- 0 to rm_dim-1){
          //squeeze a dim_neighbor*dim_neighbor subsampling block into one number by getting the mean
          val summation:Double = sum(im(i*dim_neighbor to (i+1)*dim_neighbor-1,j*dim_neighbor to (j+1)*dim_neighbor-1))
          //get output by activation
          rm(k)(i,j) = sigmoid(summation*weight(k)+bias(k))
        }
      }
    }
    output = rm
    output
  }

  /**
   * calculate Err when it's before CL
   * @param nextLayer
   */
  override def calErr(nextLayer:CL): Unit ={
    delta = calErrBeforeCl(nextLayer)
  }

  def calErrBeforeCl(nextLayer:CL): Array[DM[Double]] ={
    val deltashape = output.map(x=>x:*0d)
    delta = nextLayer.calErrForSl(deltashape)
    for(i<-output.indices)
      delta(i) = output(i):*(output(i):-1d):*(-1d):*delta(i)
    delta
  }

  /**
   * calculate Err for CL
   * @param CLDelta CL Delta shape
   * @return
   */
  def calErrForCl(CLDelta:Array[DM[Double]]): Array[DM[Double]] ={
    for(j<-CLDelta.indices){
      for(m<-0 to delta(j).cols-1){
        for(n<-0 to delta(j).rows-1){
          //subset range assignment: all delta in the block, when passed to previous CL, should be the same
          for(a<- 0 to dim_neighbor-1){
            for(b<- 0 to dim_neighbor-1){
              CLDelta(j)(dim_neighbor*m+a,dim_neighbor*n+b) = delta(j)(m,n)
            }
          }
        }
      }
      CLDelta(j) = CLDelta(j):*weight(j)
    }
    CLDelta
  }

  /**
   * adjust weight
   * single node mode
   */
  override def adjWeight(): Unit ={
    val adjw:Array[Double] = weight.map(x=>0d)
    val adjb:Array[Double] = bias.map(x=>0d)
    for(i<-delta.indices){
      var adjw_part = 0d
      var adjb_part = 0d
      for(m<-0 to delta(i).cols-1){
        for(n<-0 to delta(i).rows-1){
          val sumin = sum(input(i)(dim_neighbor*m to dim_neighbor*(m+1)-1,dim_neighbor*n to dim_neighbor*(n+1)-1))
          adjw_part = adjw_part + sumin*delta(i)(m,n)
          adjb_part = adjb_part + delta(i)(m,n)
        }
      }
      adjw(i) = adjw_part*eta
      adjb(i) = adjb_part*eta
      weight(i) = weight(i)+adjw_part*eta
      bias(i) = bias(i)+adjb_part*eta
    }
  }

  /**
   * calculate weight adjustments and queue up in the list
   * cluster mode, for batch update
   */
  override def calWeightAdj(): Unit ={
    val adjw:Array[Double] = weight.map(x=>0d)
    val adjb:Array[Double] = bias.map(x=>0d)
    for(i<-delta.indices){
      var adjw_part = 0d
      var adjb_part = 0d
      for(m<-0 to delta(i).cols-1){
        for(n<-0 to delta(i).rows-1){
          val sumin = sum(input(i)(dim_neighbor*m to dim_neighbor*(m+1)-1,dim_neighbor*n to dim_neighbor*(n+1)-1))
          adjw_part = adjw_part + sumin*delta(i)(m,n)
          adjb_part = adjb_part + delta(i)(m,n)
        }
      }
      adjw(i) = adjw_part*eta
      adjb(i) = adjb_part*eta
    }
    weightAdj = weightAdj ++ List(adjw)
    biasAdj = biasAdj ++ List(adjb)
  }

  /**
   * batch update weight
   * cluster node mode
   */
  override def collectWeightAdj(): Unit ={
    val adjw: Array[Double] = weightAdj.reduce{(a,b)=>
      val sumadjw = a
      for(i<-a.indices)
        sumadjw(i) = a(i)+b(i)
      sumadjw
    }
    val adjb: Array[Double] = biasAdj.reduce{(a,b)=>
      val sumadjb = a
      for(i<-a.indices)
        sumadjb(i) = a(i)+b(i)
      sumadjb
    }
    for(i<-weight.indices){
      weight(i) = weight(i)+adjw(i)
      bias(i) = bias(i)+adjb(i)
    }
  }
}
