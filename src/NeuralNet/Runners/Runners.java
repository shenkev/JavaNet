package NeuralNet.Runners;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import org.la4j.Matrix;
import org.la4j.Vector;
import org.la4j.matrix.DenseMatrix;

import IO.WeightIO;
import NeuralNet.NeuralNet;
import NeuralNet.Costs.Loss;
import NeuralNet.Costs.SquareLoss;
import NeuralNet.LossAndDerivs.XSquared;
import NeuralNet.NonLinFuncs.BipolarSigmoid;
import NeuralNet.NonLinFuncs.NonLinFunction;
import NeuralNet.NonLinFuncs.ReLu;
import NeuralNet.NonLinFuncs.Sigmoid;
import NeuralNet.NonLinFuncs.Tanh;
import NeuralNet.NonLinFuncs.doNothing;
import NeuralNet.Optimizers.GradientDescent;
import NeuralNet.Optimizers.Optimizer;

public class Runners {
	static String folderPath = "./results/NNWeights/";
/*
 * 
 */	
	public static void NonStochastic(double[][] Xarr, double[][] yarr) {
		
		Matrix X = DenseMatrix.from2DArray(Xarr);
		Matrix y = DenseMatrix.from2DArray(yarr);
	
		// Setting batch size
		int numberOfDataPoints = Xarr.length;
		int batchSize = numberOfDataPoints;
		
		// NN hyperparams
		int noFeatures = Xarr[0].length;
		int[] layerDims = new int[] { 4, yarr[0].length };
		int noLayers = layerDims.length;
		double trainRate = 0.2;
		double momentum = 0.9;
		NonLinFunction nonLinFunction = new BipolarSigmoid(10);
		NonLinFunction outputFunction = new BipolarSigmoid(10);
		Loss lossFunc = new SquareLoss();
		Optimizer optimizer = new GradientDescent(trainRate, momentum, noLayers, layerDims, noFeatures, 0);
		int randSeed = 800;		

		NeuralNet nn = new NeuralNet(noFeatures, batchSize, noLayers, layerDims, 
				nonLinFunction, outputFunction, lossFunc, optimizer, randSeed);
		nn.setDropout(1.0);
		// training params
		int iter = 300;
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
		public static void Stochastic(double[][] Xarr, double[] yarr) {
		
			// Setting batch size
			int batchSize = 1;
			
			// NN hyperparams
			int noFeatures = 2;
			int[] layerDims = new int[] { 4, 1 };
			int noLayers = layerDims.length;
			double trainRate = 0.2;
			double momentum = 0.9;
			NonLinFunction nonLinFunction = new ReLu();
			NonLinFunction outputFunction = new BipolarSigmoid();
			Loss lossFunc = new SquareLoss();
			Optimizer optimizer = new GradientDescent(trainRate, momentum, noLayers, layerDims, noFeatures, 0);
			int randSeed = 800;	
			
			NeuralNet nn = new NeuralNet(noFeatures, batchSize, noLayers, layerDims, 
					nonLinFunction, outputFunction, lossFunc, optimizer, randSeed);
			nn.setDropout(1.0);
			// training params
			int iter = 1000;
			int printPer = 1;
			int convergedIteration = 0;
			boolean converged = false;
			Random rand = new Random();

			
			// default use all data
			Matrix X = DenseMatrix.from2DArray(Xarr);
			Matrix y = Vector.fromArray(yarr).toColumnMatrix();
			
			double[] losses = new double[iter];
			
			for ( int i = 0; i < iter; i++ ) {
				
				 int index = rand.nextInt(X.rows());
				 Matrix X_batch = DenseMatrix.from1DArray(1, X.columns(), Xarr[index]);
				 Matrix y_batch = Vector.fromArray(new double[]{yarr[index]}).toColumnMatrix();
				
				 double loss = nn.runOnePass(X_batch, y_batch);
				 losses[i] = loss;
				 
				 if ( i % printPer == 0 ) {
					 double totalLoss = stochasticLoss(nn, X, y, Xarr, yarr).loss;
					 if ( totalLoss < 0.05 && converged == false ) {
						 converged = true;
						 convergedIteration = i;
					 }
					 System.out.println( "Loss for iteration " + i + " is: " + totalLoss );
				 }
			}
			
			try {
				write("data.txt", losses);
			} catch (IOException e) {
				e.printStackTrace();
			}
			
			System.out.println(Arrays.toString(stochasticLoss(nn, X, y, Xarr, yarr).predictions));;
			System.out.println("Converged at iteration: " + convergedIteration);	
		}
		
	/**
	 * I feel dirty doing this but yolo	
	 * @return
	 */
	private static lossPredictionPair stochasticLoss(NeuralNet nn, Matrix X,
			Matrix Y, double[][] Xarr, double[] yarr) {

		double[] predictions = new double[X.rows()];
		for (int i=0; i<X.rows(); i++) {
			Matrix X_singlet = DenseMatrix.from1DArray(1, X.columns(), Xarr[i]);
			// only supports 1D right now
			predictions[i] = nn.predict(X_singlet).getRow(0).get(0);
		}
		
		double accumLoss = 0.0;
		for (int i=0; i<X.rows(); i++) {
			accumLoss = accumLoss + (predictions[i] - yarr[i])*(predictions[i] - yarr[i]);
		}
		
		return new lossPredictionPair(0.5*accumLoss, predictions);	
	}
	
	private static class lossPredictionPair{
		
		// only supports 1D predictions
		double loss;
		double[] predictions;
		
		lossPredictionPair(double loss, double[] predictions) {
			this.loss = loss;
			this.predictions = predictions;
		}
	}
/*
 * 
 */
	public static void BatchRun(double[][] Xarr, double[] yarr) {
		// Setting batch size
		int batchSize = 1;
		
		// NN hyperparams
		int noFeatures = 2;
		int[] layerDims = new int[] { 4, 1 };
		int noLayers = layerDims.length;
		double trainRate = 0.2;
		double momentum = 0.9;
		NonLinFunction nonLinFunction = new ReLu();
		NonLinFunction outputFunction = new BipolarSigmoid();
		Loss lossFunc = new SquareLoss();
		Optimizer optimizer = new GradientDescent(trainRate, momentum, noLayers, layerDims, noFeatures, 0);
		int randSeed = 800;	
		
		NeuralNet nn = new NeuralNet(noFeatures, batchSize, noLayers, layerDims, 
				nonLinFunction, outputFunction, lossFunc, optimizer, randSeed);
		nn.setDropout(1.0);
		// training params
		int iter = 1000;
		int printPer = 1;
		int convergedIteration = 0;
		boolean converged = false;
		Random rand = new Random();

		
		// default use all data
		Matrix X = DenseMatrix.from2DArray(Xarr);
		Matrix y = Vector.fromArray(yarr).toColumnMatrix();
		
		double[] losses = new double[iter];
		
		for ( int i = 0; i < iter; i++ ) {
			
			 int index = rand.nextInt(X.rows());
			 Matrix X_batch = DenseMatrix.from1DArray(1, X.columns(), Xarr[index]);
			 Matrix y_batch = Vector.fromArray(new double[]{yarr[index]}).toColumnMatrix();
			
			 double loss = nn.runOnePass(X_batch, y_batch);
			 losses[i] = loss;
			 
			 if ( i % printPer == 0 ) {
				 double totalLoss = stochasticLoss(nn, X, y, Xarr, yarr).loss;
				 if ( totalLoss < 0.05 && converged == false ) {
					 converged = true;
					 convergedIteration = i;
				 }
				 System.out.println( "Loss for iteration " + i + " is: " + totalLoss );
			 }
		}
		
	   // save weights
	   WeightIO.save(folderPath, "w/", "weight", nn.getW());
	   
	   // save biases
	   Vector[] bb = nn.getb();
	   Matrix[] mb = new Matrix[bb.length];
	   for(int i=0; i < mb.length; i++) {
	   	mb[i] = bb[i].toRowMatrix();
	   }
	   WeightIO.save(folderPath, "b/", "bias", mb);
		
		// Setting batch size
		int batchSizeTest = 4;
		
		NeuralNet nnTest = new NeuralNet(noFeatures, batchSizeTest, noLayers, layerDims, 
				nonLinFunction, outputFunction, lossFunc, optimizer, randSeed);
		
		try {
			// load weights
			nnTest.setW(WeightIO.load(folderPath, "w/", new int[][]{
					{2, 4},
					{4, 1}
			}));
			
			// load biases
			Matrix[] lmb;
			lmb = WeightIO.load(folderPath, "b/", new int[][]{
				{1, 4},
				{1, 1}
			});
			Vector[] lbb = new Vector[lmb.length];
			for(int i=0; i < lbb.length; i++) {
				lbb[i] = lmb[i].toRowVector();
			}
			nnTest.setb(lbb);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		System.out.println(nnTest.predict(X));
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
