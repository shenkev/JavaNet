package NeuralNet.LossAndDerivs;

import org.la4j.vector.functor.VectorFunction;

public class LogisticFunction implements VectorFunction {

	@Override
	public double evaluate(int i, double value) {

		return Math.log1p(Math.pow( Math.E, -value));
	}


}
