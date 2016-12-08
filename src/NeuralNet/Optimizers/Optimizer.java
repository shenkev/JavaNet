package NeuralNet.Optimizers;

import org.la4j.Matrix;
import org.la4j.Vector;

public abstract class Optimizer {
	protected double learnRate;

	public OptimizationResult optimize(Matrix[] W, Vector[] b, Matrix[] dW, Vector[] db) {
		throw new IllegalArgumentException(); // Do not call this abstract method
	};
	
	public class OptimizationResult {
		public Matrix[] newW;
		public Vector[] newb;
		
		public OptimizationResult(Matrix[] inputW, Vector[] inputb) {
			this.newW = inputW;
			this.newb = inputb;
		}
	}
	
	public void setLearnRate(double rate) {
		this.learnRate = rate;
	}
	
	public double getLearnRate() {
		return this.learnRate;
	}
}
