package NeuralNet.Optimizers;

import org.la4j.Matrix;
import org.la4j.Vector;
import org.la4j.matrix.DenseMatrix;

public class GradientDescent extends Optimizer{

	private double momentum;
	private Matrix[] previous_dW;
	private Vector[] previous_db;
	private double lambda = 0;
	
	public GradientDescent(double learnRate, double momentum,
			int noLayers, int[] layerDimensions, int noFeatures, double lambda) {
		
		this.learnRate = learnRate;
		this.momentum = momentum;
		this.lambda = lambda;
		
		// Initialize dW and db previous
		this.previous_dW = new Matrix[noLayers];
		this.previous_dW[0] = DenseMatrix.zero(noFeatures, layerDimensions[0]);
		for (int i = 1; i < this.previous_dW.length; i++) {
			this.previous_dW[i] = DenseMatrix.zero(layerDimensions[i-1], layerDimensions[i]);
		}
		
		this.previous_db = new Vector[noLayers];
		for (int i = 0; i < this.previous_db.length; i++) {
			this.previous_db[i] = Vector.zero(layerDimensions[i]);
		}
		
	}
	
	@Override
	public OptimizationResult optimize(Matrix[] W, Vector[] b, Matrix[] dW, Vector[] db) {
		
		for (int i = 0; i < W.length; i++) {
			
			
			// Calculate the change
			previous_dW[i] = dW[i].multiply(this.learnRate)
								.add(previous_dW[i].multiply(this.momentum));
			previous_db[i] = db[i].multiply(this.learnRate)
								.add(previous_db[i].multiply(this.momentum));
			
			// Update weights
			W[i] = W[i].multiply(1-this.learnRate*this.lambda);
			W[i] = W[i].subtract(previous_dW[i]);
			b[i] = b[i].subtract(previous_db[i]);
			
		}
		
		return new OptimizationResult(W, b);
	}

}
