import java.util.Random;

import org.la4j.Matrix;
import org.la4j.Vector;
import org.la4j.matrix.DenseMatrix;

import NeuralNet.NeuralNet;
import NeuralNet.Costs.LogisticLoss;
import NeuralNet.Costs.Loss;
import NeuralNet.Costs.SquareLoss;
import NeuralNet.NonLinFuncs.NonLinFunction;
import NeuralNet.NonLinFuncs.Sigmoid;
import NeuralNet.NonLinFuncs.Tanh;
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
		
		// get data
		double[][] Xarr = new double[][]{
		  { 0, 0 },
		  { 1, 0 },
		  { 0, 1 },
		  { 1, 1 }
		};	
		
		double[] yarr = new double[] {
			0,
			1,
			1,
			0
		};
		
		Matrix X = DenseMatrix.from2DArray(Xarr);
		Matrix y = Vector.fromArray(yarr).toColumnMatrix();
		
		// initialize
		
		int noFeatures = 2;
		int batchSize = 4;
		int noLayers = 2;
		int[] layerDims = new int[] { 4, 1 };
		double trainRate = 0.2;
		double momentum = 0.0;
		NonLinFunction nonLinFunction = new Sigmoid();
		Loss lossFunc = new SquareLoss();
		Optimizer optimizer = new GradientDescent();
		int randSeed = 500;
		
		
		NeuralNet nn = new NeuralNet(noFeatures, batchSize, noLayers, layerDims,
				trainRate, momentum, nonLinFunction, lossFunc, optimizer, randSeed);

		// training
		int iter = 1000;
		int printPer = 10;
		
		for ( int i = 0; i < iter; i++ ) {
			 double loss = nn.runOnePass(X, y);
			
			if ( i % printPer == 0 ) {
				System.out.println( "Loss for iteration " + i + " is: " + loss );
			}
		}
		
		System.out.println(nn.predict(X));
		
	}
	


}

