package NeuralNet.Optimizers;

import org.la4j.Matrix;
import org.la4j.Vector;

public abstract class Optimizer {

	public OptimizationResult optimize(Matrix[] W, Vector[] b, Matrix[] dW, Vector[] db, double rate, double momentum) {
		return null; // Do not call this abstract method
	};
	
	public class OptimizationResult {
		public Matrix[] newW;
		public Vector[] newb;
		
		public OptimizationResult(Matrix[] inputW, Vector[] inputb) {
			this.newW = inputW;
			this.newb = inputb;
		}
	}
}
