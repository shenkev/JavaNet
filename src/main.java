import java.util.Random;

import NeuralNet.NeuralNet;
import NeuralNet.Costs.LogisticLoss;
import NeuralNet.Costs.Loss;
import NeuralNet.NonLinFuncs.NonLinFunction;
import NeuralNet.NonLinFuncs.Sigmoid;
import NeuralNet.Optimizers.GradientDescent;
import NeuralNet.Optimizers.Optimizer;

/**
 * 
 */

/**
 * @author kevin
 *
 */
public class main {

	/**
	 * @param args
	 */
	public static void main(String[] args) {

		// initialize
		
		int noFeatures = 2;
		int batchSize = 1;
		int noLayers = 2;
		int[] layerDims = { 4, 1 };
		double trainRate = 0.2;
		double momentum = 0.0;
		NonLinFunction nonLinFunction = new Sigmoid();
		Loss lossFunc = new LogisticLoss();
		Optimizer optimizer = new GradientDescent();
		
		
		NeuralNet nn = new NeuralNet(noFeatures, batchSize, noLayers, layerDims,
				trainRate, momentum, nonLinFunction, lossFunc, optimizer);

		// training
		int iter = 1000;
		int printPer = 10;
		
		for ( int i = 0; i < iter; i++ ) {
			// double loss = nn.runOnePass(X, y);
			
			if ( i % printPer == 0 ) {
				System.out.printf("Loss for iteration %i is %d. \n", i, loss);
			}
		}
		
	}
	


}

