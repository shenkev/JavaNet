package NeuralNet.NonLinFuncs;

import NeuralNet.NonLinAndDerivs.SigmoidDerivative;
import NeuralNet.NonLinAndDerivs.SigmoidFunction;

// Outputs from 0 to 1
public class Sigmoid extends NonLinFunction{

	public Sigmoid() {
		this.fnc = new SigmoidFunction();
		this.derivative = new SigmoidDerivative();
	}
	
	public static double evaluateSigmoid(double z) {
		return ( 1/( 1 + Math.pow( Math.E,(-1*z) ) ) );
	}
}
