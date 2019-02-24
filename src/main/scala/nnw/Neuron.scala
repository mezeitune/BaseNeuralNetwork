package nnw

/**
  * Define a Neuron of a Neural network
  */
case class Neuron(theta: List[Double]) {

  /**
    * Alternative constructor for deactivated neuron
    */
  def this(numInputs: Int) =
    this(List.fill(numInputs)(0.0))

  /**
    * Theta must be of the same length as the input.
    * X * theta => Double
    * Image 1
    */
  def apply(input:List[Double]) : Double = {
    val zipped = input zip (theta) //zip values with thetas to value according the importance of each one
    val multiplied = zipped map {case (x,y) => x*y}
    multiplied.sum //sum all the values received from the back neurons
  }

  /**
    * Helper for cost function with regularization
    * Image 2
    */
  def thetasSumSquare: Double = theta.foldLeft(0.0)((a,b) => a + b*b)

}
