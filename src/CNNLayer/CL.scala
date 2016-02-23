package CNNLayer

/**
 * @author Philip GENG
 *         gengxu.heartfire@gmail.com
 * This convolutional layer in CNN
 */
import breeze.linalg.{DenseVector=>DV,DenseMatrix=>DM,sum}
import breeze.numerics.{sigmoid}
import org.apache.spark.rdd.RDD

/**
 * constructor:
 * @param numfm number of output feature maps
 * @param dim_conv width/height of perception field/convolutional musk
 * @param eta learning rate, optional
 */
class CL(val numfm:Int, val dim_conv:Int, var eta:Double=0.5) extends layer {
  //fm_input_map: by default, input image and feature maps are one-to-one aligned and mapped
  var fm_input_map: Array[Array[Int]] = (0 to numfm-1).map(x=>Array(x)).toArray
  //outputIndex: a invert index for the position of each input map
  var outputIndex: Array[Array[(Int,Int)]] = calcInvIndex(fm_input_map)
  //convolve kernel, learnable weights, init:(-1,1)
  var kernel: Array[Array[DM[Double]]] = fm_input_map.map(arr=>arr.map(fm=>DM.rand[Double](dim_conv,dim_conv):*=2d:-=1d))
  //bias, learnable, init:(-1,1)
  var bias: Array[Double] = (DV.rand(numfm):*=2d:-=1d).toArray
  //[cluster]rdd
  var input: RDD[Array[DM[Double]]] = _
  var delta: RDD[Array[DM[Double]]] = _
  var output: RDD[Array[DM[Double]]]= _
  //[local]
  var inputLocal: Array[DM[Double]] = _
  var deltaLocal: Array[DM[Double]] = _
  var outputLocal: Array[DM[Double]]= _
  /**
   * set map relationship between input maps with output feature maps
   * @param map input image/map of the ith output feature map
   */
  def seteta(e:Double): Unit ={
    this.eta = e
  }

  def set_fm_input_map(map:Array[Array[Int]]): Unit ={
    fm_input_map = map
    outputIndex = calcInvIndex(map)
    kernel = fm_input_map.map(arr=>arr.map(fm=>DM.rand[Double](dim_conv,dim_conv):*=2d:-=1d))
  }

  /**
   * calculate position of each input map
   * @param input_map fm_input_map
   * @return each input[(outputfm, index in kernel)]
   */
  def calcInvIndex(input_map:Array[Array[Int]]): Array[Array[(Int,Int)]] ={
    val numOfInputMap:Int = input_map.flatMap(x=>x).max+1
    val ii:Array[Array[(Int,Int)]] = Array.fill(numOfInputMap){Array()}
    for(i<-input_map.indices)
      for(j<-input_map(i).indices)
        ii(input_map(i)(j)) = ii(input_map(i)(j))++Array((i,j))
    ii
  }

  /**
   *  [cluster] batch processing in rdd
   * @param input input Matrix of image or maps
   *              will be distributed to different featuremaps following rules defined in fm_input_map
   * @return output feature maps in Array of matrix
   */

  override def forward(input: RDD[Array[DM[Double]]]): RDD[Array[DM[Double]]] = {
    this.input = input
    output = input.map{rddarr=> //each rdd
      kernel.zip(bias).zip(fm_input_map).map{fm=> //each fm
        val k: Array[DM[Double]] = fm._1._1 //kernel
        val b: Double = fm._1._2 //bias
        val i: Array[Int] = fm._2 //input map
        k.zip(i).map{ker=> //each convolution
          val convKernel:DM[Double] = ker._1 //kernel
          val index:Int = ker._2 //index of inputmap
          convolveDM(rddarr(index),convKernel) //get map and convolve
        }.reduce((x,y)=>x+y).map(x=>sigmoid(x+b)) //convoltuion end, sum up with bias and sigmoid
      }//fm end
    }//rdd end
    output
  }
  /**
   * [local]
   * @param input input Matrix of image or maps
   *              will be distributed to different featuremaps observing rules defined in fm_input_map
   * @return output feature maps in Array of matrix
   */
  override def forwardLocal(input: Array[DM[Double]]): Array[DM[Double]] = {
    this.inputLocal = input
    outputLocal = kernel.zip(bias).zip(fm_input_map).map{fm=> //each fm
      val k: Array[DM[Double]] = fm._1._1 //kernel
      val b: Double = fm._1._2 //bias
      val i: Array[Int] = fm._2 //input map
      k.zip(i).map{ker=> //each convoltuion
        val convKernel: DM[Double] = ker._1 //kernel
        val index: Int = ker._2 //index of inputmap
        convolveDM(inputLocal(index),convKernel) //get inputmap and convolve
      }.reduce((x,y)=>x+y).map(x=>sigmoid(x+b)) //end convolution, sum up with bias and sigmoid
    }//end fm
    outputLocal
  }

  /**
   * [cluster] batch processing in rdd
   * calculate Err of this CL when it's before a FL
   * @param nextLayer
   * @return Err of this CL
   */
  def calErrBeforeFl(nextLayer:FL): RDD[Array[DM[Double]]] ={
    //get data from next layer
    val nextDelta: RDD[DV[Double]] = nextLayer.delta
    val nextWeight: DM[Double] = nextLayer.weight
    //back prop delta by weight matrix to this CL
    val sum:RDD[DV[Double]]= nextDelta.map{vec=>nextWeight*vec}
    //flatten output Array[DM[Double]] to DV[Double]
    val o:RDD[DV[Double]] = output.map(flattenOutput(_))
    //calculate delta of this CL based on output value
    delta = o.zip(sum).map{elem=> // each rdd
      val e_o:DV[Double] = elem._1
      val e_sum:DV[Double] = elem._2
      formatOutput(e_o :* (e_o:-1d) :* e_sum :* -1d)
    }//end rdd
    delta
  }

  /**
   * [local]
   * calculate Err of this CL when it's before a FL
   * @param nextLayer
   * @return Err of this CL
   */
  def calErrBeforeFlLocal(nextLayer:FL): Array[DM[Double]] ={
    //get data from next layer
    val nextDelta: DV[Double] = nextLayer.deltaLocal
    val nextWeight: DM[Double] = nextLayer.weight
    //back prop delta by weight matrix to this CL
    val sum:DV[Double]= nextWeight*nextDelta
    //flatten output Array[DM[Double]] to DV[Double]
    val o:DV[Double] = flattenOutput(outputLocal)
    //calculate delta of this CL based on output value
    val deltaVector:DV[Double] = o :* (o:-1d) :* sum :* -1d
    deltaLocal = formatOutput(deltaVector)
    deltaLocal
  }
  /**
   * [cluster]
   * calculate Err of this CL when it's before a SL
   * @param nextLayer
   * @return Err of this CL
   */
  def calErrBeforeSl(nextLayer:SL): RDD[Array[DM[Double]]] ={
    //ask next SL to calculate delta for this CL
    val propErr:RDD[Array[DM[Double]]] = nextLayer.calErrForCl()
    //distribute delta to featuremaps
    delta = propErr.zip(output).map{arr=>
      arr._1.zip(arr._2).map{item=>
        val d:DM[Double] = item._1
        val o:DM[Double] = item._2
        o:*(o:-1d):*(-1d):*d
      }
    }
    delta
  }

  /**
   * [local]
   * calculate Err of this CL when it's before a SL
   * @param nextLayer
   * @return Err of this CL
   */
  def calErrBeforeSlLocal(nextLayer:SL): Array[DM[Double]] ={
    //ask next SL to calculate delta for this CL
    val SLDelta: Array[DM[Double]] = nextLayer.calErrForClLocal()
    deltaLocal = SLDelta.zip(outputLocal).map{each=>
      val d = each._1 //delta
      val o = each._2 //output
      o:*(o:-1d):*(-1d):*d
    }
    deltaLocal
  }
  /**
   * [cluster]
   * calculate Err for SL(when the SL is before this CL)
   * @return delta of the SL
   */
  def calErrForSl(): RDD[Array[DM[Double]]] ={
    val SLErr:RDD[Array[DM[Double]]] = delta.map{rddarr=>
      outputIndex.map{sm=>
        sm.map { cm =>
          val d = rddarr(cm._1)
          val SLDim = dim_conv+d.rows-1
          val k = rot180(kernel(cm._1)(cm._2))
          val expand_d:DM[Double] = padDM(d,dim_conv-1)
          convolveDM(expand_d,k)
        }.reduce((a,b)=>a+b)
      }
    }
    SLErr
  }

  /**
   * [local]
   * calculate Err for SL(when the SL is before this CL)
   * @return delta of the SL
   */
  def calErrForSlLocal(): Array[DM[Double]] ={
    val SLErr:Array[DM[Double]] = outputIndex.map{sm=> //each SL map
      sm.map { cm => //corresponding CL map
        val d = deltaLocal(cm._1) //get CL map delta
        val SLDim = dim_conv+d.rows-1 //calculate SLmap dimension
        val k = rot180(kernel(cm._1)(cm._2)) //rot 180 of conv kernel
        val expand_d:DM[Double] = padDM(d,dim_conv-1)
        convolveDM(expand_d,k)
      }.reduce((a,b)=>a+b)
    }
    SLErr
  }

  /**
   * distribute task according to type of layers
   * @param nextLayer
   */
  override def calErr(nextLayer:FL): Unit =calErrBeforeFl(nextLayer)
  override def calErr(nextLayer:SL): Unit =calErrBeforeSl(nextLayer)
  override def calErrLocal(nextLayer:FL): Unit =calErrBeforeFlLocal(nextLayer)
  override def calErrLocal(nextLayer:SL): Unit =calErrBeforeSlLocal(nextLayer)

  /**
   * [cluster]
   * adjust weight
   */
  override def adjWeight(): Unit ={
    val adj:Array[(Array[DM[Double]],Double)] =delta.zip(input).map{rddarr=> //each iteration
      val err:Array[DM[Double]] = rddarr._1 //delta
      val input:Array[DM[Double]] = rddarr._2 //input
      err.zip(fm_input_map.map(x=>x.map(input(_)))).map{d=> //each delta or each kernel
        val delta:DM[Double] = d._1 //delta
        val input:Array[DM[Double]] = d._2 //input
        val adjb:Double = sum(delta) //bias
        val adjw:Array[DM[Double]] = input.map{in=>convolveDM(in,delta)} //adj Kernel
        (adjw,adjb)
      }
    }.reduce{case(a,b)=>
      a.zip(b).map{pair=>(
          pair._1._1.zip(pair._2._1).map(mat=>mat._1+mat._2), // sum adjw matrix
          pair._1._2+pair._2._2 //sum bias double
        )
      }
    }
    for(i<-adj.indices){
      val adjw:Array[DM[Double]] = adj(i)._1
      val adjb:Double = adj(i)._2
      bias(i) = bias(i)+eta*adjb
      for(j<-adjw.indices)
        kernel(i)(j) = kernel(i)(j) + adjw(j):*eta
    }
  }
  /**
   * [local]
   * adjust weight
   */
  override def adjWeightLocal(): Unit ={
    val adj:Array[(Array[DM[Double]],Double)] = deltaLocal.zip(fm_input_map.map(x=>x.map(inputLocal(_)))).map{d=>
      val delta:DM[Double] = d._1
      val input:Array[DM[Double]] = d._2
      val adjb:Double = sum(delta)
      val adjw:Array[DM[Double]] = input.map{in=>convolveDM(in,delta)}
      (adjw,adjb)
    }

    for(i<-adj.indices){
      val adjw:Array[DM[Double]] = adj(i)._1
      val adjb:Double = adj(i)._2
      bias(i) = bias(i)+eta*adjb
      for(j<-adjw.indices)
        kernel(i)(j) = kernel(i)(j) + adjw(j):*eta
    }
  }
}
