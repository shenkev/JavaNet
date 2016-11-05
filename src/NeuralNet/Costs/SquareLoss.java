package NeuralNet.Costs;

import org.la4j.Matrix;
import org.la4j.Vector;

import NeuralNet.LossAndDerivs.XSquared;

// works with any range of predictions
public class SquareLoss extends Loss {

	@Override
	public double computeLoss(Matrix prediction, Matrix truth) {
		
//		if ( prediction.columns() != 1 || truth.columns() != 1) {
//			throw new IllegalArgumentException("Squared function only takes 1D vectors");
//		}
		
//		Vector result = truth.getColumn(0).subtract(prediction.getColumn(0));
		Matrix result = truth.subtract(prediction);
		result.update(new XSquared());
		return 0.5*result.sum();
	}
	
	@Override
	public Matrix computeGradient(Matrix prediction, Matrix truth) {
		
//		if ( prediction.columns() != 1 || truth.columns() != 1) {
//			throw new IllegalArgumentException("Squared function only takes 1D vectors");
//		}
		
//		Vector result = prediction.getColumn(0).subtract(truth.getColumn(0));
//		return result.toColumnMatrix();
		return prediction.subtract(truth);
	}
	
}
