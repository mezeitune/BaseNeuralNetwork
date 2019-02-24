package nnw

/** Created by Matias Zeitune feb 2019 **/
package object nnw {

  type TrainingSet = List[(Input, Result)]
  type Input = List[Double]
  type Result = List[Double]



}

object nnwMath {
  /**
    * Gradient of the sigmoid (used for calculate gradient)
    */
  def sigmoidGradient(x: Double): Double =  sigmoid(x)*(1-sigmoid(x))

  def sigmoid(x:Double): Double = 1 / (1 + Math.exp(-x))
}
