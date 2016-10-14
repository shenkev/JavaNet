package NeuralNet;

import java.util.Random;

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
	private Matrix[] del;
	private Matrix[] W;
	private Vector[] b;
	private Matrix[] dW;
	private Vector[] db;
	private double loss = 0;
	
	// Functions
	private MatrixFunction noneLinearity;
	private MatrixFunction noneLinearityDerivative;
	private Loss lossFunction;
	
	// Optimization
	private double trainRate;
	private double momentum;
	private Optimizer optimizer;
	
	public NeuralNet(int noFeatures, int batchSize, int noLayers,
			int[] layerDims, double trainRate, double momentum, 
			NonLinFunction noLinFunc, Loss lossObj, Optimizer optimizer) {
		
		this.noFeatures = noFeatures;
		this.batchSize = batchSize;
		this.noLayers = noLayers;
		this.layerDimensions = layerDims;
		this.noneLinearity = noLinFunc.fnc;
		this.noneLinearityDerivative = noLinFunc.derivative;
		this.lossFunction = lossObj;
		this.trainRate = trainRate;
		this.momentum = momentum;
		this.optimizer = optimizer;
		
		Random rand = new Random(800);

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
		H = new Matrix[this.noLayers];
		for (int i = 0; i < H.length; i++) {
			H[i] = DenseMatrix.zero(batchSize, layerDimensions[i]);
		}
		
		Hhat = new Matrix[this.noLayers];
		for (int i = 0; i < Hhat.length; i++) {
			Hhat[i] = DenseMatrix.zero(batchSize, layerDimensions[i]);
		}
		
		del = new Matrix[this.noLayers];
		for (int i = 0; i < del.length; i++) {
			del[i] = DenseMatrix.zero(batchSize, layerDimensions[i]);
		}
		
	}

	@Override
	public void forwardProp(Matrix batchData) {
		
		// prepare first layer
		Hhat[0] = batchData.multiply(W[0]);
		// add bias
		for (int i = 0; i < batchSize; i++) {
			Hhat[0].setRow(i, Hhat[0].getRow(i).add(b[0]));
		}
		H[0] = Hhat[0].transform(noneLinearity);
		
		// iterate through each layer
		for (int k = 1; k < noLayers; k++) {
			Hhat[k] = H[k-1].multiply(W[k]);
			// add bias
			for (int i = 0; i < batchSize; i++) {
				Hhat[k].setRow(i, Hhat[k].getRow(i).add(b[k]));
			}
			H[k] = Hhat[k].transform(noneLinearity);
		}
		
	}

	@Override
	public void backwardProp(Matrix y) {
		
		// prepare last layer
		del[noLayers-1] = lossFunction.computeGradient(H[noLayers - 1], y)
							.hadamardProduct(Hhat[noLayers - 1].transform(noneLinearityDerivative));
						
		// iterate backwards through layers
		for (int k = noLayers-2; k >= 0; k--) {
			del[k] = del[k+1].multiply(W[k+1].transpose())
					.hadamardProduct(Hhat[k].transform(noneLinearityDerivative));
		}
		
	}
	
	@Override
	public double runOnePass(Matrix X, Matrix y) {
		
		if ( y.columns() != H[noLayers - 1].columns() ) {
			throw new IllegalArgumentException("Number of outputs doesn't match number of truth classes.");
		}
		
		forwardProp(X);
		loss = lossFunction.computeLoss(H[noLayers - 1], y);
		backwardProp(y);
		computeGradients(X);
		OptimizationResult result = optimizer.optimize(W, b, dW, db, trainRate, momentum);
		if (result == null) {
			System.out.println("Error: called invalid optimization function.");
		}
		W = result.newW;
		b = result.newb;
		
		return loss;
	}

	@Override
	public void computeGradients(Matrix batchData) {
		
		// gradient for first layer
		dW[0] = batchData.transpose().multiply(del[0]);
		// bias for first layer = sum up rows of del
		db[0] = del[0].getRow(0);
		for (int i = 1; i < batchSize; i++) {
			db[0] = db[0].add(del[0].getRow(i));
		}
		
		for (int k = 0; k < noLayers; k++) {
			dW[k] = H[k-1].transpose().multiply(del[k]);
			// bias = sum up rows of del
			db[k] = del[k].getRow(0);
			for (int i = 1; i < batchSize; i++) {
				db[k] = db[k].add(del[k].getRow(i));
			}
		}
	}
	
	@Override
	public Matrix predict(Matrix Xhat) {
		
		forwardProp(Xhat);
		return H[noLayers - 1];
	}
	
}
