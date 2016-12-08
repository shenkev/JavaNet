package NeuralNet.NonLinAndDerivs;

import org.la4j.matrix.functor.MatrixFunction;

import NeuralNet.NonLinFuncs.BipolarSigmoid;

public class BipolarSigmoidFunction implements MatrixFunction {

	double multiple = 1;

	public BipolarSigmoidFunction(double multiple) {
		this.multiple = multiple;
	}
	
	@Override
	public double evaluate(int i, int j, double value) {
		return BipolarSigmoid.evaluateBipolarSigmoid(value);
	}

}
