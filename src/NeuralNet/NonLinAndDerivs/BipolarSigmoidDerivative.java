package NeuralNet.NonLinAndDerivs;

import org.la4j.matrix.functor.MatrixFunction;

import NeuralNet.NonLinFuncs.BipolarSigmoid;

public class BipolarSigmoidDerivative implements MatrixFunction {

	double multiple = 1;
	
	public BipolarSigmoidDerivative(double multiple) {
		this.multiple = multiple;
	}
	
	@Override
	public double evaluate(int i, int j, double value) {
		
		double bsig = BipolarSigmoid.evaluateBipolarSigmoid(value);
		return 0.5*(1-bsig*bsig);
	}

}
