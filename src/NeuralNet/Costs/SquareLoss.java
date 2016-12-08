package NeuralNet.Costs;

import org.la4j.Matrix;
import org.la4j.Vector;

import NeuralNet.LossAndDerivs.XSquared;

// works with any dimension of predictions
public class SquareLoss extends Loss {

	@Override
	public double computeLoss(Matrix prediction, Matrix truth) {
		
		Matrix result = truth.subtract(prediction);
		result.update(new XSquared());
		return 0.5*result.sum();
	}
	
	@Override
	public Matrix computeGradient(Matrix prediction, Matrix truth) {
		
		return prediction.subtract(truth);
	}
	
}
