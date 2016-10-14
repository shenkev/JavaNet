package NeuralNet.LossAndDerivs;

import org.la4j.vector.functor.VectorFunction;

public class LogisticDerivative implements VectorFunction {

	@Override
	public double evaluate(int i, double value) {
		
		return -1.0/(1.0 + Math.pow( Math.E, value));
	}





}
