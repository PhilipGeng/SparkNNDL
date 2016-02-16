package CNN

/**
 * @author Philip GENG
 *         gengxu.heartfire@gmail.com
 * This is the generic layer model contains functions in common
 */
import breeze.linalg.{DenseMatrix=>DM,DenseVector=>DV}
import org.apache.spark.rdd.RDD

class layer extends Serializable {
  def forward(input_arr:Array[DM[Double]]):Array[DM[Double]] = {Array(DM.fill(1,1){0})}
  def forward(input_arr:RDD[Array[DM[Double]]]):RDD[Array[DM[Double]]] = {Array(DM.fill(1,1){0})}
  def calErr(nextLayer:SL): Unit = {}
  def calErr(nextLayer:CL): Unit = {}
  def calErr(nextLayer:FL): Unit = {}
  def calErr(nextLayer:OL): Unit = {}
  def calErr(target: DV[Double]): Unit = {}
  def adjWeight(): Unit = {}
  def calWeightAdj(): Unit ={}
  def collectWeightAdj(): Unit ={}
  def flattenOutput(output: Array[DM[Double]]): DV[Double] ={
    val flatOutput: DV[Double] = new DV(output.map(x=>x(0,0)))
    flatOutput
  }
  def formatOutput(output: DV[Double]): Array[DM[Double]] ={
    val format:Array[DM[Double]] = output.toArray.map(x=>DM.fill(1,1){x})
    format
  }
}
