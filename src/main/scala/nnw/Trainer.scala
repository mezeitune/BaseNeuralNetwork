import nnw.{Neuron, nnwMath}
import nnw.nnw.{Input, Result, TrainingSet}

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


  def customFmin(network: NeuralNetwork, training: TrainingSet, iteration: Int): NeuralNetwork = {

    def newNetworkFromGradient(stepUsed: Double): NeuralNetwork = {

      //val currCost = costFunction(network, training, lambda)
      //println("Current cost: " + currCost)

      val grad = gradient(network, training)

      //Modify theta
      val newNetwork = network.updateThetasWithGradient(grad, stepUsed)
      //val newCost = costFunction(newNetwork, training, lambda)

      //if (newCost > currCost) newNetworkFromGradient(stepUsed / 2) //step too big
      //else {
        //println("New cost: " + newCost)
        newNetwork
      //}

    }

    println("Iter:" + iteration)
    if (iteration == 0) network
    else customFmin(newNetworkFromGradient(step), training, iteration - 1)

  }

  def outputDeltas(squareError: List[Double],
                   network: NeuralNetwork,
                   trainingExample: Input): List[Double] = {

    network.outputLayer.zip(squareError).flatMap{
      case(neuron,error) => {
        neuron.theta.map{_ =>
          error*
            nnwMath.sigmoidGradient(neuron.apply(trainingExample))*
            nnwMath.sigmoid(neuron.apply(trainingExample))
        }
      }
    }
  }

  def hiddenDeltas(squareError: List[Double],
                   network: NeuralNetwork,
                   trainingExample: Input): List[Double] = {
    val totalErrorSum = squareError.sum

    val aux2 = network.hiddenLayer.map{
      neuron => {
        totalErrorSum*
          nnwMath.sigmoidGradient(neuron.apply(trainingExample))
      }
    }
    val hiddenThetas = network.hiddenLayer.zipWithIndex.map{ case (x,neuronNumber) =>(x.theta,neuronNumber) }

    val hiddenThetasFlatten = for(h <- hiddenThetas) yield {
      val thetas = h._1
      val allThetas = thetas.tail.zipWithIndex.map{case (theta,index) => (theta,(index+1,h._2))}
      List(((1.0),(0,h._2))):::allThetas
    }

    hiddenThetasFlatten.flatten.map{
      case (x,(thetaNumber,neuronNumber)) => {
        val inputNumber = thetaNumber match {
          case 0 => 0
          case _ => (thetaNumber - 1)
        }
        aux2(neuronNumber)*trainingExample(inputNumber)
      }
    }
  }


  def calculateSquareError(network: NeuralNetwork, trainingExample: Input, expectedResult: Result): List[Double] = {
    val resul = network.apply(trainingExample)
    val zippedResult = resul zip expectedResult
    zippedResult map { case (res,exp) => nnwMath.totalErrorDerivate(exp,res)}
  }


  def gradient(network: NeuralNetwork, set: TrainingSet): (List[Double],List[Double]) = {


    val intermediateGradients = set.map{
      case(trainingExample, expectedResult) => {
        val squareError = calculateSquareError(network,trainingExample,expectedResult)
        val outputD = outputDeltas(squareError,network,trainingExample)
        val hiddenD = hiddenDeltas(squareError,network,trainingExample)

        (hiddenD,outputD)
      }
    }

    val incrementsHidden = intermediateGradients.map{ x => x._1}
    val incrementsOutput = intermediateGradients.map{ x => x._2}

    val finalHidden = incrementsHidden.transpose.map( x => x.sum / set.length)
    val finalOutput = incrementsOutput.transpose.map( x => x.sum / set.length)

    (finalHidden,finalOutput)

  }


}

