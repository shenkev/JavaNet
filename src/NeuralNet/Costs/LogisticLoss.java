package NeuralNet.Costs;

import org.la4j.Matrix;
import org.la4j.Vector;

import NeuralNet.LossAndDerivs.LogisticDerivative;
import NeuralNet.LossAndDerivs.LogisticFunction;

public class LogisticLoss extends Loss {

	public LogisticLoss() {
		this.loss = new LogisticFunction();
		this.derivative = new LogisticDerivative();
	}
	
	@Override
	public double computeLoss(Matrix prediction, Matrix truth) {
		
		if ( prediction.columns() != 1 || truth.columns() != 1) {
			throw new IllegalArgumentException("Logistic function only takes 1D vectors");
		}
		
		Vector result = truth.getColumn(0).hadamardProduct(prediction.getColumn(0)).transform(new LogisticFunction());
		return result.sum();
	}
	
	@Override
	public Matrix computeGradient(Matrix prediction, Matrix truth) {
		
		if ( prediction.columns() != 1 || truth.columns() != 1) {
			throw new IllegalArgumentException("Logistic function only takes 1D vectors");
		}
		
		Vector result = truth.getColumn(0).hadamardProduct(prediction.getColumn(0)).transform(new LogisticDerivative());
		result = result.hadamardProduct(truth.getColumn(0)); // extra times y from derivative
		return result.toColumnMatrix();
	}
}
