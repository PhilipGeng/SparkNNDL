package CNNLayer

/**
 * @author Philip GENG
 *         gengxu.heartfire@gmail.com
 * This is the sub-sampling layer in CNN
 */
import breeze.linalg.{DenseVector=>DV, DenseMatrix=>DM, sum}
import breeze.numerics.sigmoid
import org.apache.spark.rdd.RDD

/**
 *
 * @param numfm number of featuremaps
 * @param dim_neighbor width/height of the subsampling block
 * @param eta learning rate
 */
class SL (val numfm: Int, val dim_neighbor:Int, var eta:Double=0.5) extends layer{
  //weight and bias
  var weight:Array[Double] = Array.fill(numfm){scala.util.Random.nextDouble()*2-1d}
  var bias:Array[Double] = Array.fill(numfm){scala.util.Random.nextDouble()*2-1d}
  //[cluster]rdd
  var output:RDD[Array[DM[Double]]]=_
  var delta:RDD[Array[DM[Double]]]=_
  var input:RDD[Array[DM[Double]]]=_
  //[local]
  var outputLocal:Array[DM[Double]]=_
  var deltaLocal:Array[DM[Double]]=_
  var inputLocal:Array[DM[Double]]=_

  def seteta(e:Double): Unit ={
    this.eta = e
  }
  /**
   * forward
   * @param input_arr input image
   * @return output featuremap
   */
  override def forward(input_arr:RDD[Array[DM[Double]]]): RDD[Array[DM[Double]]] ={
    input = input_arr //save input
    input.cache()
    output = input.map{arr=> //rdd, each iteration
      val rm_dim:Int = arr(0).cols/dim_neighbor //output matrix dimension
      arr.zip(weight.zip(bias)).map{mat=> //(input:DM,(weight:Double,bias:Double))
        val dm:DM[Double] = mat._1 //matrix
        val w:Double = mat._2._1 //weight
        val b:Double = mat._2._2 //bias
        DM.tabulate(rm_dim,rm_dim){case (i,j)=> //indices in output matrix
          //mean subsampling
          val mean:Double = sum(dm(i*dim_neighbor to (i+1)*dim_neighbor-1,j*dim_neighbor to (j+1)*dim_neighbor-1))/(dim_neighbor*dim_neighbor)
          sigmoid(mean*w+b)
        }
      }
    }.cache()
    output
  }

  override def forwardLocal(input_arr:Array[DM[Double]]): Array[DM[Double]] ={
    inputLocal = input_arr
    val rm_dim:Int = inputLocal(0).cols/dim_neighbor
    val rm:Array[DM[Double]] = Array.fill(inputLocal.length){DM.zeros(rm_dim,rm_dim)}
    for(k<-inputLocal.indices){
      val im = inputLocal(k)
      for(i<- 0 to rm_dim-1){
        for(j<- 0 to rm_dim-1){
          //squeeze a dim_neighbor*dim_neighbor subsampling block into one number by getting the mean
          val summation:Double = sum(im(i*dim_neighbor to (i+1)*dim_neighbor-1,j*dim_neighbor to (j+1)*dim_neighbor-1))
          rm(k)(i,j) = sigmoid(summation*weight(k)+bias(k))
        }
      }
    }
    outputLocal = rm
    outputLocal
  }
  /**
   * calculate Err when it's before CL
   * @param nextLayer
   */
  override def calErr(nextLayer:CL): Unit ={
    delta = calErrBeforeCl(nextLayer)
    delta.cache()
  }
  override def calErrLocal(nextLayer:CL): Unit ={
    deltaLocal = calErrBeforeClLocal(nextLayer)
  }

  def calErrBeforeCl(nextLayer:CL): RDD[Array[DM[Double]]] ={
    val propErr:RDD[Array[DM[Double]]] = nextLayer.calErrForSl() //receive proped err
    delta = propErr.zip(output).map{arr=> //each arr
      arr._1.zip(arr._2).map{Mat=> //zipped matrix
        val d:DM[Double] = Mat._1 //delta
        val o:DM[Double] = Mat._2 //output
        o:*(o:-1d):*(-1d):*d //calculate as matrix
      }
    }
    delta
  }
  def calErrBeforeClLocal(nextLayer:CL): Array[DM[Double]] ={
    val SLDelta = nextLayer.calErrForSlLocal()
    deltaLocal = outputLocal.zip(SLDelta).map{x=>
      val o = x._1
      val d = x._2
      o:*(o:-1d):*(-1d):*d
    }
    deltaLocal
  }

  /**
   * calculate Err for CL
   * @return
   */
  def calErrForCl(): RDD[Array[DM[Double]]] ={
    val CLDelta: RDD[Array[DM[Double]]] = delta.map{arr=> //each arr
      arr.zip(weight).map{Mat=> //each mat zip with weight
        val d:DM[Double] = Mat._1 //delta mat
        val w:Double = Mat._2 //weight
        expandDM(d,dim_neighbor):*w
      }
    }
    CLDelta
  }

  def calErrForClLocal(): Array[DM[Double]] ={
    val CLDelta: Array[DM[Double]] = deltaLocal.zip(weight).map{Mat=>
      val d:DM[Double] = Mat._1 //delta mat
      val w:Double = Mat._2 //weight
      expandDM(d,dim_neighbor):*w
    }
    CLDelta
  }



  /**
   * calculate weight adjustments and queue up in the list
   * cluster mode, for batch update
   */
  override def adjWeight(): Unit ={
    val adj:RDD[Array[(Double, Double)]] = delta.zip(input) map { arr => // each arr
      arr._1.zip(arr._2).map { mat=> //each delta mat zip with input mat
        val d:DM[Double] = mat._1 //delta
        val i:DM[Double] = mat._2 //input
        val summat:DM[Double] = DM.tabulate(d.rows,d.cols){case(m,n)=>sum(i(dim_neighbor*m to dim_neighbor*(m+1)-1,dim_neighbor*n to dim_neighbor*(n+1)-1))/(dim_neighbor*dim_neighbor)}
        val adjw:Double = sum(summat:*d) //weight adjustment
        val adjb:Double = sum(d) //bias adjustment
        (adjw,adjb)
      }
    }
    val redadj:Array[(Double,Double)] = adj.reduce{(a,b)=>
      a.zip(b).map{each=>
        (each._1._1+each._2._1,each._1._2+each._2._2) //reduce(sum up) adjustments
      }
    }
    weight = weight.zip(redadj).map{arr=>arr._1+arr._2._1*eta}
    bias = bias.zip(redadj).map{arr=>arr._1+arr._2._2*eta}
  }

  override def adjWeightLocal(): Unit ={
    val adjw:Array[Double] = weight.map(x=>0d)
    val adjb:Array[Double] = bias.map(x=>0d)
    for(i<-deltaLocal.indices){
      var adjw_part = 0d
      var adjb_part = 0d
      for(m<-0 to deltaLocal(i).cols-1){
        for(n<-0 to deltaLocal(i).rows-1){
          val sumin = sum(inputLocal(i)(dim_neighbor*m to dim_neighbor*(m+1)-1,dim_neighbor*n to dim_neighbor*(n+1)-1))
          adjw_part = adjw_part + sumin*deltaLocal(i)(m,n)
          adjb_part = adjb_part + deltaLocal(i)(m,n)
        }
      }
      adjw(i) = adjw_part*eta
      adjb(i) = adjb_part*eta
      weight(i) = weight(i)+adjw_part*eta
      bias(i) = bias(i)+adjb_part*eta
    }
  }
  override def clearCache(): Unit ={
    input.unpersist()
    output.unpersist()
    delta.unpersist()
  }
  override def filterInput(inputFilter:RDD[Int]): Unit = {
    //only keep wrong results
    input = input.zip(inputFilter).filter(_._2 == 0).map(_._1)
    input.coalesce(numPartition,true)
  }
}
