package CNN

/**
 * @author Philip GENG
 *         gengxu.heartfire@gmail.com
 * This convolutional feature maps in CNN
 */
import breeze.linalg.{DenseVector=>DV,DenseMatrix=>DM,sum}
import breeze.numerics.{sigmoid,tanh}

/**
 * constructor:
 * @param inputmap index of input image
 * @param dim_conv height/width of convolutional musk/perception field
 * @param eta learning rate
 */
class convFeatureMap(val inputmap:Array[Int],val dim_conv:Int, val eta:Double = 0.5d) extends Serializable {
  //conv: weights of convolutional musks according to different input maps
  var conv:Array[DM[Double]] = Array.fill(inputmap.length){DM.rand(dim_conv,dim_conv):*=2d:-=1d}
  var bias:Double = scala.util.Random.nextDouble()*2d-1d
  var output:DM[Double] =_
  var delta:DM[Double] = _
  var input:Array[DM[Double]] = _
  //convAdj/biasAdj: queue of weight/bias adjustment
  var convAdj:List[Array[DM[Double]]] = _
  var biasAdj:List[Double] = _

  /**
   * forward function of each featuremap
   * @param input_arr input image
   * @return output featuremap
   */
  def forward(input_arr:Array[DM[Double]]): DM[Double] ={
    // get coresponding input images
    val in:Array[DM[Double]] = inputmap.map(input_arr(_))
    input = in
    // initialize output
    val outdim = in(0).rows-dim_conv+1
    output = DM.zeros(outdim,outdim)
    //construct output of this feature map
    for(i<- 0 to outdim-1){
      for(j<- 0 to outdim-1){
        for(m<- in.indices){
          output(i,j) = output(i,j)+sum(in(m)(i to i+dim_conv-1, j to j+dim_conv-1):*conv(m))
        }
        output(i,j) = sigmoid(output(i,j)+bias)
      }
    }
    output
  }

  /**
   * assign delta/Err
   * @param indelta
   */
  def setDelta(indelta:DM[Double]): Unit ={
    delta = indelta
  }

  /**
   * calculate Err for SL
   * @return
   */
  def calErrForSl(): Array[DM[Double]] ={
    //initialize shape of Err
    val dim_Err:Int = dim_conv+output.cols-1
    val Err:Array[DM[Double]] = Array.fill(inputmap.length){DM.fill(dim_Err,dim_Err){0}}
    for(i<- inputmap.indices)//foreach input image
      for(m<- 0 to output.rows-1)
        for(n<- 0 to output.cols-1){
          // subset range assignment: Err(i)(m to m+dim_conv-1,n to n+dim_conv-1) = conv(i):*delta(m,n)
          val submat = conv(i):*delta(m,n)
          for(a<- m to m+dim_conv-1){
            for(b<- n to n+dim_conv-1){
              Err(i)(a,b) = submat(a-m,b-n)
            }
          }
        }
    Err
  }

  /**
   * adjust weight
   * single node mode
   */
  def adjustWeight(): Unit ={
    val adjw: Array[DM[Double]] = conv.map(c=>DM.fill(c.rows,c.cols){0d})
    val adjb: Double = sum(delta)
    for(i<- 0 to delta.cols-1){
      for(j<- 0 to delta.rows-1){
        for(m<- conv.indices){
          adjw(m) = adjw(m)+input(m)(i to i+dim_conv-1, j to j+dim_conv-1):*delta(i,j)
        }
      }
    }
    bias = bias+adjb
    for(i<-conv.indices)
      conv(i) = conv(i)+adjw(i)
  }

  /**
   * calculate and queue up weight adjustment
   * cluster mode, batch update
   */
  def calWeightAdj(): Unit ={
    val adjw: Array[DM[Double]] = conv.map(c=>DM.fill(c.rows,c.cols){0d})
    val adjb: Double = sum(delta)
    for(i<- 0 to delta.cols-1){
      for(j<- 0 to delta.rows-1){
        for(m<- conv.indices){
          adjw(m) = adjw(m)+input(m)(i to i+dim_conv-1, j to j+dim_conv-1):*delta(i,j)
        }
      }
    }
    convAdj = convAdj++List(adjw)
    biasAdj = biasAdj++List(adjb)
  }

  /**
   * collect weight adjustments and clear list
   * cluster mode, batch update
   */
  def collectWeightAdj(): Unit ={
    val adjw: Array[DM[Double]] = convAdj.reduce{(a,b) =>
      val sum: Array[DM[Double]] = a
      for (i <- a.indices) {
        sum(i) = a(i) + b(i)
      }
      sum
    }
    val adjb: Double = biasAdj.sum

    for(i<-conv.indices)
      conv(i) = conv(i)+adjw(i)
    bias = bias+adjb

    convAdj = List()
    biasAdj = List()
  }
}
