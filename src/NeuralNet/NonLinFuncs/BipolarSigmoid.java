package NeuralNet.NonLinFuncs;

import NeuralNet.NonLinAndDerivs.BipolarSigmoidDerivative;
import NeuralNet.NonLinAndDerivs.BipolarSigmoidFunction;

// Outputs from -1 to 1
public class BipolarSigmoid extends NonLinFunction{

	double multiple = 1;
	
	public BipolarSigmoid() {
		this.fnc = new BipolarSigmoidFunction(1);
		this.derivative = new BipolarSigmoidDerivative(1);
	}
	
	public BipolarSigmoid(double multiple) {
		this.multiple = multiple;
		this.fnc = new BipolarSigmoidFunction(multiple);
		this.derivative = new BipolarSigmoidDerivative(multiple);
	}
	
	public static double evaluateBipolarSigmoid(double z) {
		return ( -1 + 2/( 1 + Math.pow( Math.E,(-1*z) ) ) );
	}
}
