package NeuralNet.NonLinFuncs;

import NeuralNet.NonLinAndDerivs.ReLuDerivative;
import NeuralNet.NonLinAndDerivs.ReLuFunction;
import NeuralNet.NonLinAndDerivs.leaveUnchanged;

// This outputs a value >= 0
public class ReLu extends NonLinFunction{

	public ReLu() {
		this.fnc = new ReLuFunction();
		this.derivative = new ReLuDerivative();
	}
}