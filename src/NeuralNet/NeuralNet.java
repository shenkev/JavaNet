package NeuralNet;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

import org.la4j.Matrix;
import org.la4j.Vector;
import org.la4j.matrix.DenseMatrix;
import org.la4j.matrix.functor.MatrixFunction;

import NeuralNet.Costs.Loss;
import NeuralNet.NonLinFuncs.NonLinFunction;
import NeuralNet.Optimizers.Optimizer;
import NeuralNet.Optimizers.Optimizer.OptimizationResult;

public class NeuralNet implements NeuralNetInterface {

	// Neural net parameters
	private int noFeatures;
	private int batchSize;
	private int noLayers;	// greater than or equal to 1
	private int[] layerDimensions;		// last layer is number of outputs and should be number of class

	// Computation parameters
	private Matrix[] Hhat;
	private Matrix[] H;		// includes the last layer
	private Matrix del;
	private Matrix[] W;
	private Vector[] b;
	private Matrix[] dW;
	private Vector[] db;
	private Matrix[] HdropoutMask;
	private double loss = 0;
	private Vector ones;
	private Random rand;
	
	// Functions
	private MatrixFunction noneLinearity;
	private MatrixFunction noneLinearityDerivative;
	private MatrixFunction outputFunction;
	private MatrixFunction outputFunctionDerivative;
	private Loss lossFunction;
	
	// Optimization
	private Optimizer optimizer;
	
	// Hyper-parameters
	private double dropout = 1.0;	// expressed as a decimal, probability neuron is present
	
	public NeuralNet(int noFeatures, int batchSize, int noLayers, int[] layerDims,
			NonLinFunction noLinFunc, NonLinFunction outputFunc, Loss lossObj, Optimizer optimizer, int randSeed) {
				
		this.noFeatures = noFeatures;
		this.batchSize = batchSize;
		this.noLayers = noLayers;
		this.layerDimensions = layerDims;
		this.noneLinearity = noLinFunc.fnc;
		this.noneLinearityDerivative = noLinFunc.derivative;
		this.outputFunction = outputFunc.fnc;
		this.outputFunctionDerivative = outputFunc.derivative;
		this.lossFunction = lossObj;
		this.optimizer = optimizer;
		this.ones = Vector.zero(this.batchSize).add(1.0);
		
		this.rand = new Random(randSeed);

		// Initialize weight matrix
		W = new Matrix[this.noLayers];
		W[0] = DenseMatrix.random(noFeatures, layerDimensions[0], rand);
		for (int i = 1; i < W.length; i++) {
			W[i] = DenseMatrix.random(layerDimensions[i-1], layerDimensions[i], rand);
		}
		
		b = new Vector[this.noLayers];
		for (int i = 0; i < b.length; i++) {
			b[i] = Vector.zero(layerDimensions[i]);
		}
		
		// Initialize weight gradients
		dW = new Matrix[this.noLayers];
		dW[0] = DenseMatrix.zero(this.noFeatures, layerDimensions[0]);
		for (int i = 1; i < dW.length; i++) {
			dW[i] = DenseMatrix.zero(layerDimensions[i-1], layerDimensions[i]);
		}
		
		db = new Vector[this.noLayers];
		for (int i = 0; i < db.length; i++) {
			db[i] = Vector.zero(layerDimensions[i]);
		}
		
		// Initialize states
		H = new Matrix[this.noLayers + 1];
		Hhat = new Matrix[this.noLayers];
		HdropoutMask = new Matrix[this.noLayers - 1]; // don't drop first layer
	}

	// ========================== Propagation Functions ========================== //
	
	@Override
	public Matrix forwardProp(Matrix batchData) {
		
		// iterate through each layer
		for (int k = 0; k < noLayers-1; k++) {
			
			// Do dropout
			if (dropout < 0.99 && k!=0) { // we actually intend to do dropout
				
				// Need to mask H[k]
				HdropoutMask[k-1] = createMask(
						H[k].rows(), H[k].columns(), dropout).divide(dropout);
				H[k] = H[k].hadamardProduct(HdropoutMask[k-1]);
				
			}

			Hhat[k] = H[k].multiply(W[k]).add(ones.outerProduct(b[k]));
			H[k+1] = Hhat[k].transform(noneLinearity);
		}
		
		// treat output with a different nonlinear function
		if (dropout < 0.99) { // we actually intend to do dropout
			HdropoutMask[noLayers-2] = createMask(
					H[noLayers-1].rows(), H[noLayers-1].columns(), dropout).divide(dropout);
			H[noLayers-1] = H[noLayers-1].hadamardProduct(HdropoutMask[noLayers-2]);
		}
		Hhat[noLayers-1] = H[noLayers-1].multiply(W[noLayers-1]).add(ones.outerProduct(b[noLayers-1]));
		H[noLayers] = Hhat[noLayers-1].transform(outputFunction);
		
		return H[noLayers];
	}

	@Override
	public void backwardProp(Matrix y) {
		
		// prepare last layer
		del = lossFunction.computeGradient(H[noLayers], y)
							.hadamardProduct(Hhat[noLayers - 1].transform(outputFunctionDerivative));
		
		// deal with bias
		db[noLayers - 1] = ones.multiply(del);
		dW[noLayers - 1] = H[noLayers-1].transpose().multiply(del);
		// iterate backwards through layers
		for (int k = noLayers-2; k >= 0; k--) {
			
			del = del.multiply(W[k+1].transpose())
					.hadamardProduct(Hhat[k].transform(noneLinearityDerivative));
			// dropout
			if (dropout < 0.99) {
				del = del.hadamardProduct(HdropoutMask[k]);
			}
			// deal with bias
			db[k] = ones.multiply(del);
			dW[k] = H[k].transpose().multiply(del);

		}
		
	}
	
	@Override
	public double runOnePass(Matrix X, Matrix y) {
		
		if ( X.rows() != batchSize ) {
			throw new IllegalArgumentException("Batch size is not as promised.");
		}
		
		if ( y.columns() != layerDimensions[noLayers - 1] ) {
			throw new IllegalArgumentException("Number of outputs doesn't match number of truth classes.");
		}
				
		H[0] = X;
		forwardProp(X);
		loss = lossFunction.computeLoss(H[noLayers], y);
		backwardProp(y);
		OptimizationResult result = optimizer.optimize(W, b, dW, db);
		
		if (result == null) {
			System.out.println("Error: called invalid optimization function.");
		}
		
		W = result.newW;
		b = result.newb;
		
//		debugPrint();
		return loss;
	}
	
	@Override
	public Matrix predict(Matrix Xhat) {
		
		H[0] = Xhat;
		forwardProp(Xhat);
		return H[noLayers];
	}
	
	// ========================== Helper Functions ========================== //
	
	public void setDropout(double threshold) {
		if (threshold > 0.0 && threshold <= 1.0) {
			this.dropout = threshold;
		} else {
			throw new IllegalArgumentException("Dropout needs to be between 0 and 1");
		}
	}
	
	public Matrix createMask(int rows, int cols, double probOfOne) {
		
		double[] maskArr = new double[rows*cols];
		for (int i=0; i<rows*cols; i++) {
			if (this.rand.nextDouble() < probOfOne) {
				maskArr[i] = 1;
			} else {
				maskArr[i] = 0;
			}
		}
		
		return Matrix.from1DArray(rows, cols, maskArr);
	}
	
	// Need to update this when we're predicting which is annoying
	public void setBatchSize(int s) {
		this.batchSize = s;
		this.ones = Vector.zero(this.batchSize).add(1.0);
	}
}
