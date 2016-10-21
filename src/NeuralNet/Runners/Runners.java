package NeuralNet.Runners;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import org.la4j.Matrix;
import org.la4j.Vector;
import org.la4j.matrix.DenseMatrix;

import NeuralNet.NeuralNet;
import NeuralNet.Costs.Loss;
import NeuralNet.Costs.SquareLoss;
import NeuralNet.NonLinFuncs.BipolarSigmoid;
import NeuralNet.NonLinFuncs.NonLinFunction;
import NeuralNet.NonLinFuncs.ReLu;
import NeuralNet.NonLinFuncs.Sigmoid;
import NeuralNet.NonLinFuncs.Tanh;
import NeuralNet.Optimizers.GradientDescent;
import NeuralNet.Optimizers.Optimizer;

public class Runners {
	
/*
 * 
 */	
	public static void NonStochastic(double[][] Xarr, double[] yarr) {
		
		Matrix X = DenseMatrix.from2DArray(Xarr);
		Matrix y = Vector.fromArray(yarr).toColumnMatrix();
	
		// Setting batch size
		int numberOfDataPoints = 4;
		int batchSize = numberOfDataPoints;
		
		// NN hyperparams
		int noFeatures = 2;
		int[] layerDims = new int[] { 4, 1 };
		int noLayers = layerDims.length;
		double trainRate = 0.2;
		double momentum = 0.9;
		NonLinFunction nonLinFunction = new ReLu();
		NonLinFunction outputFunction = new Tanh();
		Loss lossFunc = new SquareLoss();
		Optimizer optimizer = new GradientDescent(trainRate, momentum, noLayers, layerDims, noFeatures);
		int randSeed = 800;		
		NeuralNet nn = new NeuralNet(noFeatures, batchSize, noLayers, layerDims, 
				nonLinFunction, outputFunction, lossFunc, optimizer, randSeed);

		// training params
		int iter = 40;
		int printPer = 1;
		int convergedIteration = 0;
		boolean converged = false;
		
		// default use all data
		Matrix X_batch = X;
		Matrix y_batch = y;
		double[] losses = new double[iter];
		
		for ( int i = 0; i < iter; i++ ) {
			
			 double loss = nn.runOnePass(X_batch, y_batch);
			 losses[i] = loss;
			
			 if ( loss < 0.05 && converged == false ) {
				 converged = true;
				 convergedIteration = i;
			 }
			 
			 if ( i % printPer == 0 ) {
				 System.out.println( "Loss for iteration " + i + " is: " + loss );
			 }
		}
		
		try {
			write("data.txt", losses);
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		System.out.println(nn.predict(X));
		System.out.println("Converged at iteration: " + convergedIteration);	
	}
	
	
/*
 * 
 */
	public static void BatchRun(double[][] Xarr, double[] yarr) {
		Matrix X = DenseMatrix.from2DArray(Xarr);
		Matrix y = Vector.fromArray(yarr).toColumnMatrix();
		
		// Running options
//		boolean useBatching = false;
//		int desiredBatchSize = 1;
		
		// Setting batch size
		int numberOfDataPoints = 4;
		int batchSize = numberOfDataPoints;
//		if (useBatching) {
//			batchSize = desiredBatchSize;
//		}
		
		// NN hyperparams
		int noFeatures = 2;
		int[] layerDims = new int[] { 4, 1 };
		int noLayers = layerDims.length;
		double trainRate = 0.2;
		double momentum = 0.9;
		NonLinFunction nonLinFunction = new Sigmoid();
		NonLinFunction outputFunction = new Tanh();
		Loss lossFunc = new SquareLoss();
		Optimizer optimizer = new GradientDescent(trainRate, momentum, noLayers, layerDims, noFeatures);
		int randSeed = 800;
		
		NeuralNet nn = new NeuralNet(noFeatures, batchSize, noLayers, layerDims, 
				nonLinFunction, outputFunction, lossFunc, optimizer, randSeed);

		// training params
		int iter = 1000;
		int printPer = 1;
		int convergedIteration = 0;
		boolean converged = false;
		
		// default use all data
		Matrix X_batch = X;
		Matrix y_batch = y;
//		if (useBatching) {
//			// randomly pick training batch
//			int[] batchRows = generateBatch(batchSize, X.rows());
//			X_batch = X.select(batchRows, generateZeroToNm1Array(X.columns()));
//			y_batch = y.select(batchRows, generateZeroToNm1Array(y.columns()));
//		}
		
		for ( int i = 0; i < iter; i++ ) {
			
			 double loss = nn.runOnePass(X_batch, y_batch);
			
			 if ( loss < 0.05 && converged == false ) {
				 converged = true;
				 convergedIteration = i;
			 }
			 
			 if ( i % printPer == 0 ) {
				 System.out.println( "Loss for iteration " + i + " is: " + loss );
			 }
		}
		
//		for (int i = 0; i < 4; i++) {
//			System.out.println(nn.predict(X.getRow(i).toRowMatrix()));
//		}
//		nn.setBatchSize(numberOfDataPoints);
		System.out.println(nn.predict(X));
		System.out.println("Converged at iteration: " + convergedIteration);
		
	}
	
//	public static int[] generateBatch(int size, int upper) {
//		
//		if (size > upper) {
//			throw new IllegalArgumentException("You shouldn't ask for more values than can provide");
//		}
//		
//		Set<Integer> set = new HashSet<Integer>();
//		Random rand = new Random();
//		while (set.size() < size) {
//			set.add(rand.nextInt(upper));
//		}
//		
//		Integer[] arr = new Integer[size];
//		arr = set.toArray(arr);
//		int[] arr2 = new int[size];
//		for (int i = 0; i < arr.length; i++) {
//			arr2[i] = arr[i].intValue();
//		}
//		
//		return arr2;
//	}
//	
//	public static int[] generateZeroToNm1Array(int n) {
//		
//		int[] arr = new int[n];
//		
//		for (int i = 0; i < n; i++) {
//			arr[i] = i;
//		}
//		return arr;
//	}
//}
	public static void write (String filename, double[] x) throws IOException {
		  BufferedWriter outputWriter = null;
		  outputWriter = new BufferedWriter(new FileWriter(filename));
		  for (int i = 0; i < x.length; i++) {
		    outputWriter.write(Double.toString(x[i]));
		    outputWriter.newLine();
		  }
		  outputWriter.flush();  
		  outputWriter.close();  
	}
}
