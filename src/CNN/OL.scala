package CNN

/**
 * @author Philip GENG
 *         gengxu.heartfire@gmail.com
 * This is fully connected layer as the output layer in CNN (multi-layer perceptron)
 */
import breeze.linalg.{DenseVector=>DV,DenseMatrix=>DM}
import breeze.numerics.sigmoid

/**
 * constructor
 * @param num_in number of incoming features
 * @param num_out number of output vector
 * @param eta learning rate, optional
 */
class OL (override val num_in:Int, override val num_out:Int, override val eta:Double=0.5) extends FL(num_in, num_out, eta){
  /**
   * override calculation of Delta, cuz it doesn't have next layer
   * @param target desired output
   */
  override def calErr(target: DV[Double]): Unit ={
    delta = output :* (output :- 1d) :* (output :- target)
  }
}
