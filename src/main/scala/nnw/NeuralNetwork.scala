import nnw.{Neuron, nnwMath}
import nnw.nnw.{Input, Result}

/**
  * For us a Neural network is made by two layers. The hidden layer, and the output layer.
  * Depending of your needs you could consider add more layers (or remove them)
  *
  * Each of the neurons of the hidden layer receive all the inputs and generate a response
  * Each of the neurons of the output layer receive all the responses of the hidden layer and generate a response
  */
case class NeuralNetwork(inputLayer: Int, //Number of inputs of the neural Network (also called input neurons /input units)
                          hiddenLayer: List[Neuron],
                          outputLayer: List[Neuron]) {

  /**
    * Alternative constructor for deactivateds neuron layers
    * Used to Initialize real NeuralNetwork in a Trainer
    */
  def this(inputLayer: Int, hiddenLayerCount: Int, outputLayerCount: Int) =
    this(inputLayer,
      List.fill(hiddenLayerCount)(new Neuron(inputLayer+1)), //fill a plus for bias to the theta neuron
      List.fill(outputLayerCount)(new Neuron(hiddenLayerCount+1))) //fill a plus for bias to the theta neuron

  require(inputLayer > 0)
  require(hiddenLayer.foldLeft(true)((actual,neuron)=> actual && neuron.theta.length == inputLayer + 1))
  require(outputLayer.foldLeft(true)((actual,neuron)=> actual && neuron.theta.length == hiddenLayer.length + 1))


  //todo: it is very coupled to how many layers it has. make it more generic
  /**
    * a3 = Sigmoid(z3)
    */
  def apply(input: Input): Result = {
    sigmoidList(resultsOutputLayer(input))
  }

  /**
    * z2
    * Image 3
    */
  def resultsHiddenLayer(input:Input):Result = {
    val inputWithBias = 1.0 :: input
    hiddenLayer map { x => x.apply(inputWithBias) }
  }

  /**
    * a2 = Sigmoid(z2)
    * Image 4
    */
  def resultsHiddenLayerSigmoided(input:Input): Result = sigmoidList(resultsHiddenLayer(input))

  /**
    * z3
    */
  def resultsOutputLayer(input:Input): Result = {
    val resultHiddenWithBias = 1.0 :: resultsHiddenLayerSigmoided(input)
    outputLayer map { x => x.apply(resultHiddenWithBias)}
  }

  def sigmoidList(input: List[Double]): List[Double] = input map { x => nnwMath.sigmoid(x)}

  /**
    * Retrieve the list of thetas from hiddenLayer & outputLayer (theta1; theta2)
    */
  def thetas: List[Double] = {
    val hidTheta = hiddenLayer flatMap {x => x.theta}
    val outputTheta = outputLayer flatMap {x => x.theta}
    hidTheta:::outputTheta
  }


  def updateThetasWithGradient(grad: (List[Double],List[Double]), alpha: Double): NeuralNetwork = {

    val inputSize = inputLayer + 1
    val hiddenLayerSize = hiddenLayer.length + 1

    val hiddenThetas = hiddenLayer.flatMap(x=>x.theta)
    val outputThetas = outputLayer.flatMap(x=>x.theta)

    val newHiddenThetas = (hiddenThetas zip grad._1).map{case(x,y) => x-alpha*y}
    val newOutputThetas = (outputThetas zip grad._2).map{ case(x,y) => x-alpha*y}

    val newHiddenLayer = newHiddenThetas.grouped(inputSize).map{x=>Neuron(x)}.toList
    val newOutputLayer = newOutputThetas.grouped(hiddenLayerSize).map{x=>Neuron(x)}.toList

    val das = NeuralNetwork(inputLayer,newHiddenLayer,newOutputLayer)
    das
  }
}