import nnw.Neuron
import nnw.nnw.TrainingSet

import scala.util.Random

/** Created by Matias Zeitune feb 2019 **/


object Trainer {
  def apply() = new Trainer()
}

/**
  * In charge of train the Neural Network selecting initial values for theta
  */
case class Trainer(maxIter: Int, lambda: Double, step: Double) {

  def this() = {
    this(400, 1.0, 0.15) //Default values
  }

  /**
    * Calculate the cost of the current neural network for a training set given with the expected results
    * (As we are on a classifying problem we expect a vector of (1,0,0,0...) for detect a 0
    * (0,1,0,0...) for detect a 1 etc
    *
    */
  def costFunction(network: NeuralNetwork, trainingSet: TrainingSet, lambda: Double): Double = {
    val costPerExample = trainingSet map {
      case (input, expected) => {
        val calculated = network.apply(input)
        val calcAndExp = calculated zip expected
        val diff = calcAndExp map {
          case (calc, exp) => Math.abs(-exp * Math.log(calc) - (1 - exp) * Math.log(1 - calc))
        }
        diff.sum
      }
    }
    val noReg = costPerExample.sum / trainingSet.length

    val hiddenReg = network.hiddenLayer.foldLeft(0.0)((a, b) => a + b.thetasSumSquare)
    val outputReg = network.outputLayer.foldLeft(0.0)((a, b) => a + b.thetasSumSquare)

    noReg + lambda / (2 * trainingSet.length) * (hiddenReg + outputReg)
  }

  /**
    * From Stanford Coursera Machine Learning Course. Recommendation
    */
  def epsilonInitForHidden(network: NeuralNetwork): Double =
    Math.sqrt(6) / Math.sqrt(network.hiddenLayer.length + network.inputLayer)

  /**
    * From Stanford Coursera Machine Learning Course. Recommendation
    */
  def epsilonInitForOutput(network: NeuralNetwork): Double =
    Math.sqrt(6) / Math.sqrt(network.hiddenLayer.length + network.outputLayer.length)

  /**
    * Generate a random theta for a neuron in the Hidden layer
    */
  def initThetaForHidden(network: NeuralNetwork): List[Double] =
    List.fill(network.inputLayer + 1)(-epsilonInitForHidden(network) + (2 * epsilonInitForHidden(network)) * Random.nextDouble())

  /**
    * Generate a random theta for a neuron in the Output layer
    */
  def initThetaForOutput(network: NeuralNetwork): List[Double] =
    List.fill(network.hiddenLayer.length + 1)(-epsilonInitForOutput(network) + (2 * epsilonInitForOutput(network)) * Random.nextDouble())

  /**
    * Return the trained neural network
    */
  def train(network: NeuralNetwork, trainingSet: TrainingSet): NeuralNetwork = {
    //Start with random thetas
    val hiddenLayer = (1 to network.hiddenLayer.length) map { x => Neuron(initThetaForHidden(network)) } toList
    val outputLayer = (1 to network.outputLayer.length) map { x => Neuron(initThetaForOutput(network)) } toList

    val initialNetwork = NeuralNetwork(network.inputLayer, hiddenLayer, outputLayer)

    customFmin(initialNetwork, trainingSet, maxIter)

  }



    //todo: modificar esto para checkear si usar el gradiente o no , esto esta medio a piacere del chabon

  def customFmin(network: NeuralNetwork, training: TrainingSet, iteration: Int): NeuralNetwork = {

    def newNetworkFromGradient(stepUsed: Double): NeuralNetwork = {

      val currCost = costFunction(network, training, lambda)
      println("Current cost: " + currCost)

      val grad = gradient(network, training)

      //Modify theta
      val newNetwork = network.updateThetasWithGradient(grad, stepUsed)
      val newCost = costFunction(newNetwork, training, lambda)

      if (newCost > currCost) newNetworkFromGradient(stepUsed / 2) //step too big
      else {
        println("New cost: " + newCost)
        newNetwork
      }

    }

    println("Iter:" + iteration)
    if (iteration == 0) network
    else customFmin(newNetworkFromGradient(step), training, iteration - 1)

  }

  //todo: agregar toda la parte de backprop
  def gradient(network: NeuralNetwork, set: TrainingSet) = ???


}

