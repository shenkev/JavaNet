package NeuralNet.LossAndDerivs;

import org.la4j.vector.functor.VectorFunction;

public class XSquared implements VectorFunction {

	@Override
	public double evaluate(int i, double value) {

		return value*value;
	}

}