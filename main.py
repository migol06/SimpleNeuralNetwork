import numpy as np

class NeuralNetwork():
  
  def __init__(self):
    np.random.seed(1)

    self.synaptic_weights = 2 * np.random.random((10,1)) - 1

  def sigmoid(self, x):
    return 1/(1 + np.exp(-x))

  def sigDerivative(self, x):
    return x * (1-x)

  def getInput(self, outlook = "sunny", temparature = "hot", humidity = "high", wind = "weak"):

    training_inputs = []

    outlook = outlook.lower()
    temparature = temparature.lower()
    humidity = humidity.lower()
    wind = wind.lower()

    if outlook == "sunny":
      training_inputs.append(1)
    else:
      training_inputs.append(0)
    
    if outlook == "overcast":
      training_inputs.append(1)
    else:
      training_inputs.append(0)
    
    if outlook == "rain":
      training_inputs.append(1)
    else:
      training_inputs.append(0)


    if temparature == "hot":
      training_inputs.append(1)
    else:
      training_inputs.append(0)

    if temparature == "mild":
      training_inputs.append(1)
    else:
      training_inputs.append(0)

    if temparature == "cold":
      training_inputs.append(1)
    else:
      training_inputs.append(0)
    

    if humidity == "high":
      training_inputs.append(1)
    else:
      training_inputs.append(0)
    
    if humidity == "normal":
      training_inputs.append(1)
    else:
      training_inputs.append(0)

    if wind == "strong":
      training_inputs.append(1)
    else:
      training_inputs.append(0)
    
    if wind == "weak":
      training_inputs.append(1)
    else:
      training_inputs.append(0)

    return training_inputs

  def getTrainingOutput(self, play ="yes"):
    play = play.lower()

    if play == "yes":
      return 1
    else:
      return 0
  
  def think(self, inputs):
    inputs = inputs.astype(float)
    output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
    return output

  def train(self, training_inputs, training_outputs, training_iterations):
      for iteration in range(training_iterations):
        output = self.think(training_inputs)

        error = training_outputs - output
        
        adjustments = np.dot(training_inputs.T, error * self.sigDerivative(output))

        self.synaptic_weights += adjustments


if __name__ == "__main__":

  neuralNetwork = NeuralNetwork()

  print('Random starting weights: ')
  print(neuralNetwork.synaptic_weights)


  training_inputs = np.array(
    [
      #1
      neuralNetwork.getInput(
        "sunny", "hot","high", "weak"
      ),
      #2
      neuralNetwork.getInput(
        "sunny", "hot","high", "strong"
      ),
      #3
      neuralNetwork.getInput(
        "overcast", "hot","high", "weak"
      ),
      #4
      neuralNetwork.getInput(
        "rain", "mild","high", "weak"
      ),
      #5
      neuralNetwork.getInput(
        "rain", "cold","normal", "weak"
      ),
      #6
      neuralNetwork.getInput(
        "rain", "cold","normal", "strong"
      ),
      #7
      neuralNetwork.getInput(
        "overcast", "cold","normal", "strong"
      ),
      #8
      neuralNetwork.getInput(
        "sunny", "mild","high", "weak"
      ),
      #9
      neuralNetwork.getInput(
        "sunny", "cold","normal", "weak"
      ),
      #10
      neuralNetwork.getInput(
        "rain", "mild","normal", "weak"
      ),
      #11
      neuralNetwork.getInput(
        "sunny", "mild","normal", "strong"
      ),
      #12
      neuralNetwork.getInput(
        "overcast", "mild","high", "strong"
      ),
      #13
      neuralNetwork.getInput(
        "overcast", "hot","normal", "weak"
      ),
      #14
      neuralNetwork.getInput(
        "rain", "mild","high", "strong"
      ),
    ]
  )

  training_outputs = np.array(
    [
      [
        #1
        neuralNetwork.getTrainingOutput("no"),
        #2
        neuralNetwork.getTrainingOutput("no"),
        #3
        neuralNetwork.getTrainingOutput("yes"),
        #4
        neuralNetwork.getTrainingOutput("yes"),
        #5
        neuralNetwork.getTrainingOutput("yes"),
        #6
        neuralNetwork.getTrainingOutput("no"),
        #7
        neuralNetwork.getTrainingOutput("yes"),
        #8
        neuralNetwork.getTrainingOutput("no"),
        #9
        neuralNetwork.getTrainingOutput("yes"),
        #10
        neuralNetwork.getTrainingOutput("yes"),
        #11
        neuralNetwork.getTrainingOutput("yes"),
        #12
        neuralNetwork.getTrainingOutput("yes"),
        #13
        neuralNetwork.getTrainingOutput("yes"),
        #14
        neuralNetwork.getTrainingOutput("no"),
      ]
    ]
  ).T

  neuralNetwork.train(training_inputs, training_outputs, 100000)

  print('Weights after training')
  print(neuralNetwork.synaptic_weights)

  outlook = input("Input Outlook[Sunny, Overcast,Rain]: ")
  temperature = input("Input Temperatutre[Hot, Cold, Mild]: ")
  humidity = input("Input Humidity[High,Normal]: ")
  wind = input("Input Wind[Weak,Strong]: ")

  print('Input data = ', outlook, temperature, humidity, wind)

  inputs = np.array(neuralNetwork.getInput(
    outlook,temperature,humidity,wind
  ))

  output = neuralNetwork.think(inputs)

  if output > 0.8:
    print("Play?: Yes")
  else:
    print("Play?: No")

  


   



  