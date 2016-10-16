package NeuralNet.NonLinFuncs;

import NeuralNet.NonLinAndDerivs.BipolarSigmoidDerivative;
import NeuralNet.NonLinAndDerivs.BipolarSigmoidFunction;

// Outputs from -1 to 1
public class BipolarSigmoid extends NonLinFunction{

	public BipolarSigmoid() {
		this.fnc = new BipolarSigmoidFunction();
		this.derivative = new BipolarSigmoidDerivative();
	}
	
	public static double evaluateBipolarSigmoid(double z) {
		return ( -1 + 2/( 1 + Math.pow( Math.E,(-1*z) ) ) );
	}
}
