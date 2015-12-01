package functionalModel

import java.util
import java.util.Random

import org.apache.spark.util.random
import spire.std.{double, int}

/**
 * Created by philippy on 2015/11/23.
 */
object integerTest {
  def main(args:Array[String]): Unit = {
    val rd = scala.util.Random;
    //var nn: neuralNetwork = new neuralNetwork(32,15, 4);
    var nn: mlnn = new mlnn(32,Array(2),4)
    var list = new Array[(Array[Double], Array[Double])](1000);


    for (i <- 0 to list.length - 1) {
      var number = rd.nextInt(Math.pow(2, 30).toInt);
      var sign = rd.nextInt(2) * 2 - 1;
      var odd = number % 2;
      number = number * sign;
      nn.train(toBinary(number), toLabel(sign, odd))
    }


    var right = 0;
    for (i <- 0 to 200) {
      var nm = rd.nextInt(Math.pow(2, 31).toInt) - Math.pow(2, 30).toInt
      var n: Array[Double] = toBinary(nm)
      var r = nn.predict(n)
      var str1 = "negative";
      var str2 = "even";
      if (r(0) > r(1)) str1 = "positive";
      if (r(2) > r(3)) str2 = "odd";
      if(nm*(r(0)-r(1))>0){
        if(r(2)>r(3)){
          if(Math.abs(nm)%2==1)right+=1
          else println("err1: "+nm+str1+str2)
        }
        else{
          if(Math.abs(nm)%2==0)right+=1
          else println("err2: "+nm+str1+str2)
        }
      }
      else{
        println("err3: "+nm+str1+str2)
      }
    }
    println(right)
  }


    def toBinary(num: Int): Array[Double] = {
      var value = num
      var index = 31;
      var binary: Array[Double] = Array.fill(32){0};
      while (value != 0) {
        binary(index) = (value & 1).toDouble;
        index = index - 1;
        value >>>= 1;
      }
      binary
    }

    def toLabel(sign: Int, odd: Int): Array[Double] = {
      var l = Array.fill(4){0d}
      if (sign > 0) {
        l(0) = 1d
      }
      else {
        l(1) = 1d
      }
      if (odd == 1) {
        l(2) = 1d
      }
      else {
        l(3) = 1d
      }
      l
    }
}
