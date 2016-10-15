package NeuralNet.Optimizers;

import org.la4j.Matrix;
import org.la4j.Vector;

public class GradientDescent extends Optimizer{

	@Override
	public OptimizationResult optimize(Matrix[] W, Vector[] b, Matrix[] dW, Vector[] db, double rate, double momentum) {
		
		for (int i = 0; i < W.length; i++) {
			
			W[i] = W[i].subtract(dW[i].multiply(rate));
			b[i] = b[i].subtract(db[i].multiply(rate));
			
		}
		
		return new OptimizationResult(W, b);
	}
	


}
