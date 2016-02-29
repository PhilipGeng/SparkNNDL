package CNNLayer

/**
 * @author Philip GENG
 *         gengxu.heartfire@gmail.com
 * This fully connected layer in CNN (multi-layer perceptron)
 */
import breeze.linalg.{DenseVector=>DV,DenseMatrix=>DM,sum}
import breeze.numerics.sigmoid
import org.apache.spark.rdd.RDD

/**
 * constructor
 * @param num_in number of incoming features
 * @param num_out number of output vector
 * @param eta learning rate, optional
 */
class FL (val num_in:Int, val num_out:Int, var eta:Double=0.5) extends layer{
  var weight:DM[Double] = DM.fill[Double](num_in, num_out){scala.util.Random.nextDouble()*2-1d}
  var bias: DV[Double] = DV.fill[Double](num_out){scala.util.Random.nextDouble()*2-1d}
  var delta: RDD[DV[Double]] = _
  var output: RDD[DV[Double]] = _
  var input: RDD[DV[Double]] = _
  var inputLocal: DV[Double] = _
  var outputLocal: DV[Double] = _
  var deltaLocal: DV[Double] = _

  def seteta(e:Double): Unit ={
    this.eta = e
  }
  /**
   * override forward function
   * @param in_vec input vector(learned features) in matrix format (to be transformed)
   * @return output
   */
  override def forward(in_vec:RDD[Array[DM[Double]]]): RDD[Array[DM[Double]]] ={
    //transform Array[DM[Double]] => DV[Double]
    input = in_vec.map(flattenOutput(_)).cache()
    //dot product to get output
    output = input.map { vec=> sigmoid((vec.t * weight).t + bias)}
    //format output DV[Double] => Array[DM[Double]]
    output.cache()
    output.map(formatOutput(_))
  }

  override def forwardLocal(in_vec:Array[DM[Double]]): Array[DM[Double]] = {
      //transform Array[DM[Double]] => DV[Double]
      inputLocal = flattenOutput(in_vec)
      //dot product to get output
      outputLocal = sigmoid((inputLocal.t * weight).t + bias)
      //format output DV[Double] => Array[DM[Double]]
      formatOutput(outputLocal)
  }
  /**
   * calculate Err for this FL if it's before a FL
   * @param nextDelta
   * @param nextWeight
   */
  def calErrBeforeFl(nextDelta:RDD[DV[Double]], nextWeight:DM[Double]): Unit ={
    delta = nextDelta.zip(output).map{vec=>
      val deltaVec:DV[Double] = vec._1
      val outputVec:DV[Double] = vec._2
      val sum:DV[Double] = nextWeight*deltaVec
      outputVec:*(outputVec:-1d):*sum:*(-1d)
    }.cache()
  }
  def calErrBeforeFlLocal(nextDelta:DV[Double], nextWeight:DM[Double]): Unit ={
    val sum:DV[Double]= nextWeight*nextDelta
    val o:DV[Double] = outputLocal
    deltaLocal = o :* (o:-1d) :* sum :* -1d
  }

  /**
   * task distributor
   * @param nextLayer
   */
  override def calErr(nextLayer:FL): Unit = calErrBeforeFl(nextLayer.delta,nextLayer.weight)
  override def calErr(nextLayer:OL): Unit = calErrBeforeFl(nextLayer.delta,nextLayer.weight)
  override def calErrLocal(nextLayer:FL): Unit = calErrBeforeFlLocal(nextLayer.deltaLocal,nextLayer.weight)
  override def calErrLocal(nextLayer:OL): Unit = calErrBeforeFlLocal(nextLayer.deltaLocal,nextLayer.weight)

  /**
   * adjust weight
   * single node mode
   */
  override def adjWeight(): Unit ={
    val adj:RDD[(DM[Double],DV[Double])] = delta.zip(input).map{vec=>
      val d:DV[Double] = vec._1
      val i:DV[Double] = vec._2
      val adjw:DM[Double] = i*d.t:*eta
      val adjb:DV[Double] = d:*eta
      (adjw,adjb)
    }
    val redadj:(DM[Double],DV[Double]) = adj.reduce{(i,j)=>(i._1:+j._1,i._2+j._2)}

    weight = weight:+redadj._1
    bias = bias:+redadj._2
  }
  override def adjWeightLocal(): Unit ={
    val adj:DM[Double] = inputLocal*deltaLocal.t:*eta
    weight = weight:+adj
    val adjb:DV[Double] = deltaLocal:*eta
    bias = bias+adjb
  }
  override def clearCache(): Unit ={
    input.unpersist()
    output.unpersist()
    delta.unpersist()
  }
  override def filterInput(inputFilter:RDD[Int]): Unit = {
    //only keep wrong results
    input = input.zip(inputFilter).filter(_._2 == 0).map(_._1)
    input.coalesce(numPartition,true)  }
}
