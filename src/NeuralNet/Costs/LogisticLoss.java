package NeuralNet.Costs;

import NeuralNet.LossAndDerivs.LogisticDerivative;
import NeuralNet.LossAndDerivs.LogisticFunction;

public class LogisticLoss extends Loss {

	public LogisticLoss() {
		this.loss = new LogisticFunction();
		this.derivative = new LogisticDerivative();
	}
}
