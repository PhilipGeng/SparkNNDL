package CNN

/**
 * @author Philip GENG
 *         gengxu.heartfire@gmail.com
 * This convolutional layer in CNN
 */
import breeze.linalg.{DenseVector=>DV,DenseMatrix=>DM,sum}
import breeze.numerics.{sigmoid,tanh}
import org.apache.spark.rdd.RDD

/**
 * constructor:
 * @param numfm number of output feature maps
 * @param dim_conv width/height of perception field/convolutional musk
 */
class CL(val numfm:Int, val dim_conv:Int) extends layer {
  //fm_input_map: by default, input image and feature maps are one-to-one aligned and mapped
  var fm_input_map: Array[Array[Int]] = (0 to numfm-1).map(x=>Array(x)).toArray
  var featuremaps: Array[convFeatureMap] = _
  var delta: Array[DM[Double]] = _
  var output: Array[DM[Double]]= _

  /**
   * set map relationship between input images/maps with output feature maps
   * @param map input image/map of the ith output feature map
   */
  def set_fm_input_map(map:Array[Array[Int]]): Unit ={
    fm_input_map = map
    featuremaps = fm_input_map.map(fim=>new convFeatureMap(fim, dim_conv))
  }

  /**
   *
   * @param input input Matrix of image or maps
   *              will be distributed to different featuremaps observing rules defined in fm_input_map
   * @return output feature maps in Array of matrix
   */
  override def forward(input: Array[DM[Double]]): Array[DM[Double]] = {
    output = featuremaps.map(fm=>fm.forward(input))
    output
  }
 /* override def forward(input: RDD[Array[DM[Double]]]): RDD[Array[DM[Double]]] = {
    output = input.map(each=>featuremaps.map(fm=>fm.forward(each)))
    output
  }*/

  /**
   * calculate Err of this CL when it's before a FL
   * @param nextLayer
   * @return Err of this CL
   */
  def calErrBeforeFl(nextLayer:FL): Array[DM[Double]] ={
    //get data from next layer
    val nextDelta: DV[Double] = nextLayer.delta
    val nextWeight: DM[Double] = nextLayer.weight
    //back prop delta by weight matrix to this CL
    val sum:DV[Double]= nextWeight*nextDelta
    //flatten output Array[DM[Double]] to DV[Double]
    val o:DV[Double] = flattenOutput(output)
    //calculate delta of this CL based on output value
    val deltaVector:DV[Double] = o :* (o:-1d) :* sum :* -1d
    //distribute delta to featuremaps
    delta = deltaVector.map(d=>DM.fill(1,1){d}).toArray
    for(i<- delta.indices)
      featuremaps(i).setDelta(delta(i))
    delta
  }
  /**
   * calculate Err of this CL when it's before a SL
   * @param nextLayer
   * @return Err of this CL
   */
  def calErrBeforeSl(nextLayer:SL): Array[DM[Double]] ={
    //get shape of delta matrix
    val deltashape: Array[DM[Double]] = output.map(x=>x:*0d)
    //ask next SL to calculate delta for this CL
    delta = nextLayer.calErrForCl(deltashape)
    //distribute delta to featuremaps
    for(i<-delta.indices){
      delta(i) = output(i):*(output(i):-1d):*(-1d):*delta(i)
      featuremaps(i).setDelta(delta(i))
    }
    delta
  }
  /**
   * calculate Err for SL(when the SL is before this CL)
   * @param SLDelta shape of delta of the SL
   * @return delta of the SL
   */
  def calErrForSl(SLDelta:Array[DM[Double]]): Array[DM[Double]] ={
    for(i<- featuremaps.indices){
      //ask each featuremap to calculate Err
      val Err:Array[DM[Double]] = featuremaps(i).calErrForSl()
      for(j<- fm_input_map(i).indices){
        //integrate Err according to input images map of each featuremaps
        val index = fm_input_map(i)(j)
        SLDelta(index) = SLDelta(index):+ Err(j)
      }
    }
    SLDelta
  }
  /**
   * distribute task according to type of layers
   * @param nextLayer
   */
  override def calErr(nextLayer:FL): Unit =calErrBeforeFl(nextLayer)
  override def calErr(nextLayer:SL): Unit =calErrBeforeSl(nextLayer)

  /**
   * adjust weight
   */
  override def adjWeight(): Unit ={
    //ask each feature map to adjust weight, according to delta been assigned
    featuremaps.foreach(fm=> fm.adjustWeight())
  }
  /**
   * calculate and queue up weight adjustment
   */
  override def calWeightAdj(): Unit ={
    //ask each feature map to calculate weight adjustment, according to delta been assigned
    featuremaps.foreach(fm=>fm.calWeightAdj())
  }
  /**
   * process weight adjustment list
   */
  override def collectWeightAdj(): Unit ={
    //ask each feature map to adjust weight, according to delta been assigned, and empty adjustment list
    featuremaps.foreach(fm=>fm.collectWeightAdj())
  }

}
