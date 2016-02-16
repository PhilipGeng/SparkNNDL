package CNN

/**
 * @author Philip GENG
 *         gengxu.heartfire@gmail.com
 * This fully connected layer in CNN (multi-layer perceptron)
 */
import breeze.linalg.{DenseVector=>DV,DenseMatrix=>DM,sum}
import breeze.numerics.sigmoid

/**
 * constructor
 * @param num_in number of incoming features
 * @param num_out number of output vector
 * @param eta learning rate, optional
 */
class FL (val num_in:Int, val num_out:Int, val eta:Double=0.5) extends layer{
  var weight:DM[Double] = DM.fill[Double](num_in, num_out){scala.util.Random.nextDouble()*2-1d}
  var bias: DV[Double] = DV.fill[Double](num_out){scala.util.Random.nextDouble()*2-1d}
  var delta: DV[Double] = DV.fill[Double](num_out){0d}
  var output: DV[Double] = DV.fill[Double](num_out){0d}
  var input: DV[Double] = DV.fill[Double](num_in){0d}
  //convAdj/biasAdj: queue of weight/bias adjustment
  var weightAdj: List[DM[Double]] = _
  var biasAdj: List[DV[Double]] = _

  /**
   * override forward function
   * @param in_vec input vector(learned features) in matrix format (to be transformed)
   * @return output
   */
  override def forward(in_vec:Array[DM[Double]]): Array[DM[Double]] ={
    //transform Array[DM[Double]] => DV[Double]
    input = flattenOutput(in_vec)
    //dot product to get output
    output = (input.t * weight).t + bias
    //activate using sigmoid function
    output = output.map(x=>sigmoid(x))
    //format output DV[Double] => Array[DM[Double]]
    formatOutput(output)
  }

  override def forward(in_vec:Array[DM[Double]]): Array[DM[Double]] ={
    //transform Array[DM[Double]] => DV[Double]
    input = flattenOutput(in_vec)
    //dot product to get output
    output = (input.t * weight).t + bias
    //activate using sigmoid function
    output = output.map(x=>sigmoid(x))
    //format output DV[Double] => Array[DM[Double]]
    formatOutput(output)
  }

  /**
   * calculate Err for this FL if it's before a FL
   * @param nextDelta
   * @param nextWeight
   */
  def calErrBeforeFl(nextDelta:DV[Double], nextWeight:DM[Double]): Unit ={
    val sum:DV[Double]= nextWeight*nextDelta
    val o:DV[Double] = output
    delta = o :* (o:-1d) :* sum :* -1d
  }

  /**
   * task distributor
   * @param nextLayer
   */
  override def calErr(nextLayer:FL): Unit = calErrBeforeFl(nextLayer.delta,nextLayer.weight)
  override def calErr(nextLayer:OL): Unit = calErrBeforeFl(nextLayer.delta,nextLayer.weight)

  /**
   * adjust weight
   * single node mode
   */
  override def adjWeight(): Unit ={
    val adj:DM[Double] = input*delta.t:*eta
    weight = weight:+adj
    val adjb:DV[Double] = delta:*eta
    bias = bias+adjb
  }

  /**
   * calculate weight adjustment and queue up
   * for cluster mode batch update
   */
  override def calWeightAdj(): Unit ={
    val adjw:DM[Double] = input*delta.t:*eta
    val adjb:DV[Double] = delta:*eta
    weightAdj = weightAdj ++ List(adjw)
    biasAdj = biasAdj ++ List(adjb)
  }

  /**
   * batch update and clear list
   */
  override def collectWeightAdj(): Unit ={
    val adjw = weightAdj.reduce(_:+_)
    val adjb = biasAdj.reduce(_+_)
    weight = weight:+adjw
    bias = bias+adjb
  }
}
