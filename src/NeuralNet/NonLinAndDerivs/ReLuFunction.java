package NeuralNet.NonLinAndDerivs;

import org.la4j.matrix.functor.MatrixFunction;

public class ReLuFunction implements MatrixFunction {

	@Override
	public double evaluate(int i, int j, double value) {
		
		if (value > 0.0) {
			return value;
		} else {
			return 0.0;
		}

	}

}
