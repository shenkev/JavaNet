package NeuralNet.NonLinAndDerivs;

import org.la4j.matrix.functor.MatrixFunction;

import NeuralNet.NonLinFuncs.BipolarSigmoid;

public class BipolarSigmoidDerivative implements MatrixFunction {

	@Override
	public double evaluate(int i, int j, double value) {
		
		double bsig = BipolarSigmoid.evaluateBipolarSigmoid(value);
		return 0.5*(1-bsig*bsig);
	}

}
