import java.awt.Color;
import java.util.Random;

import org.la4j.Matrix;
import org.la4j.matrix.DenseMatrix;

import NeuralNet.NeuralNet;
import NeuralNet.Costs.Loss;
import NeuralNet.Costs.SquareLoss;
import NeuralNet.NonLinFuncs.BipolarSigmoid;
import NeuralNet.NonLinFuncs.NonLinFunction;
import NeuralNet.NonLinFuncs.ReLu;
import NeuralNet.Optimizers.GradientDescent;
import NeuralNet.Optimizers.Optimizer;
import robocode.AdvancedRobot;
import robocode.BulletHitEvent;
import robocode.DeathEvent;
import robocode.HitWallEvent;
import robocode.ScannedRobotEvent;
import robocode.WinEvent;

public class NNRLRobot extends AdvancedRobot {
	// ========================== RL Hyper Parameters ============================ //
   final double gamma = 0.975;
   final double epsilon = 0.1;	// should start at 1 and decrement to 0.1
   
   final double rewardOnWin = 10.0;
   
	// ========================== State Representation ========================== //
   static int iter = 0;
   Random rand = new Random();
    
   static final int numberOfTotalStateParams = 13;
   static final int numberOfActions = 10;
   static final int numberOfFeatures = numberOfTotalStateParams;
   
   // indexes of features
   static final int ownEnergy = 0;
   static final int ownVelocity = 1;
   static final int ownX = 2;
   static final int ownY = 3;
   static final int ownHeadingCos = 4;
   static final int ownHeadingSin = 5;
   static final int enemyEnergy = 6;
   static final int enemyBearingCos = 7;
   static final int enemyBearingSin = 8;
   static final int enemyDistance= 9;
   static final int enemyVelocity = 10;
   static final int enemyHeadingCos = 11;
   static final int enemyHeadingSin = 12;
   
   // constants for trying to normalize input
   static final double energyScale = 100.0;
   static final double velocityScale = 8.0;
   double fieldWidth;
   double fieldHeight;
   
   // State-Actions
   Matrix currentState;
   int currentAction;
   Matrix previousState;
   
   double reward;
   
	// ========================== Neural Network Setup ========================== //
   static final int batchSize = 1;
   static final int[] layerDims = new int[] { 4, numberOfActions };
   static final int noLayers = layerDims.length;
   static double trainRate = 0.2;
   static double momentum = 0.9;
   static final NonLinFunction nonLinFunction = new ReLu();
   static final NonLinFunction outputFunction = new BipolarSigmoid();
   static final Loss lossFunc = new SquareLoss();	// need to support multi dimensions
   static final Optimizer optimizer = new GradientDescent(
		   trainRate, momentum, noLayers, layerDims, numberOfFeatures);
   static final int randSeed = 800;	
   
   static final NeuralNet nn = new NeuralNet(numberOfFeatures, batchSize, noLayers,
		   layerDims, nonLinFunction, outputFunction, lossFunc, optimizer, randSeed);
   
	// ========================== Start of Robot Code ========================== //
   public void run() {
	    fieldWidth = getBattleFieldWidth();
	    fieldHeight = getBattleFieldHeight();
        
		setAdjustRadarForRobotTurn(true);//keep the radar still while we turn
		setBodyColor(Color.red);
		setGunColor(Color.white);
		setRadarColor(Color.white);
		setScanColor(Color.white);
		setBulletColor(Color.red);
		setAdjustGunForRobotTurn(true); // Keep the gun still when we turn
		turnRadarRightRadians(Double.POSITIVE_INFINITY);//keep turning radar right
    	while (true) {

        }
   }
 
   // ============================ Q-Learning ============================= //
   public void onScannedRobot(ScannedRobotEvent e) {
       // choose current action from previous state using epsilon policy
	   Matrix QforPrevStateActionsMatrix = nn.forwardProp(previousState);
	   currentAction = epsilonGreedyPolicyChooseAction(
			   QforPrevStateActionsMatrix.getRow(0).toDenseVector().toArray());
	   
	   // always track enemy with gun
	   double absBearing=e.getBearingRadians()+getHeadingRadians();//enemies absolute bearing
	   double latVel=e.getVelocity() * Math.sin(e.getHeadingRadians() -absBearing);//enemies later velocity
	   setTurnRadarLeftRadians(getRadarTurnRemainingRadians());//lock on the radar
	   double gunTurnAmt = robocode.util.Utils.normalRelativeAngle(absBearing- getGunHeadingRadians()+latVel/22);//amount to turn our gun, lead just a little bit
	   setTurnGunRightRadians(gunTurnAmt); //turn our gun
	   
	   // take current action
	   takeAction(currentAction);
	   
	   // observe current state
	   double[] currentStateArr = returnState(getEnergy(), getVelocity(),
			   getX(), getY(),
			   getHeadingRadians(), e.getEnergy(), e.getBearingRadians(),
			   e.getDistance(), e.getVelocity(), e.getHeadingRadians());
	   currentState = DenseMatrix.from1DArray(
			   1, numberOfTotalStateParams, currentStateArr);
	   
	   reward = 0;
	   
	   // find maxQ for s'
	   Matrix QforCurrStateActionsMatrix = nn.forwardProp(currentState);
	   double maxQ = bestQ(
			   QforCurrStateActionsMatrix.getRow(0).toDenseVector().toArray());
	   double target = reward + gamma*maxQ;
	   
	   // QforPrevStateActionsMatrix is now the "truth" or target
	   QforPrevStateActionsMatrix.set(0, currentAction, target);
	   
	   // train Q(previousState, currentAction) via backpropagation
	   double loss = nn.runOnePass(previousState, QforPrevStateActionsMatrix);
	   
	   // update previous state to current state
	   previousState = currentState;

   }

   public void onWin(WinEvent event) {

	   // update previous action-state with reward only
	   Matrix QforPrevStateActionsMatrix = nn.forwardProp(previousState);
	   QforPrevStateActionsMatrix.set(0, currentAction, rewardOnWin);
	   double loss = nn.runOnePass(previousState, QforPrevStateActionsMatrix);
	   
	   // save
   }
   
   public void onDeath(DeathEvent event) {
	  
	   // update previous action-state with reward only
	   Matrix QforPrevStateActionsMatrix = nn.forwardProp(previousState);
	   QforPrevStateActionsMatrix.set(0, currentAction, (-1.0*rewardOnWin));
	   double loss = nn.runOnePass(previousState, QforPrevStateActionsMatrix);
	   
	   // save
   }
   
   public void onBulletHit(BulletHitEvent event) {
	   
   }
   
   public void onHitWall(HitWallEvent event) {
	   double hitWallPenalty = 0.5;
	   
	   // update previous action-state with reward only
	   Matrix QforPrevStateActionsMatrix = nn.forwardProp(previousState);
	   QforPrevStateActionsMatrix.set(0, currentAction, (-1.0*hitWallPenalty));
	   double loss = nn.runOnePass(previousState, QforPrevStateActionsMatrix);

   }
   
   private double bestQ(double[] QforActions) {
	   
	   double bestQ = QforActions[0];
	   for (int i=1; i<numberOfActions; i++) {
		   if (QforActions[i] > bestQ) {
			   bestQ = QforActions[i];
		   }
	   }
	   
	   return bestQ;
   }
   
   public int epsilonGreedyPolicyChooseAction(double[] QforActions) {
	   
	   if (rand.nextDouble() < epsilon) {
		   
		   return rand.nextInt(numberOfActions);
	   } else {
		  
		   int bestAction = 0;
		   double bestQ = QforActions[0];
		   
		   for (int i=1; i<numberOfActions; i++) {
			   if (QforActions[i] > bestQ) {
				   bestAction = i;
				   bestQ = QforActions[i];
			   }
		   }
		   
		   return bestAction;
	   }
   }   
   
   // =================== Helpers for Converting to State ====================== //
   public double[] returnState(double ownEnergy, double ownVelocity, double ownX,
		   double ownY, double ownHeading, double enemyEnergy,
		   double enemyBearing, double enemyDistance,
		   double enemyVelocity, double enemyHeading) {
	   
	   return new double[] {
			   
			   ownEnergy/energyScale, ownVelocity/velocityScale,
			   (ownX - (fieldWidth/2.0))/fieldWidth, 
			   (ownY - (fieldHeight/2.0))/fieldHeight,
			   Math.cos(ownHeading), Math.sin(ownHeading),
			   enemyEnergy/energyScale,
			   Math.cos(enemyBearing), Math.sin(enemyBearing),
			   enemyDistance/fieldWidth, enemyVelocity/velocityScale,
			   Math.cos(enemyHeading), Math.sin(enemyHeading)
			   
	   };
   }
   
   public void takeAction(int action) {
	   
	   double moveStep = 50.0;
	   
	   
	   switch(action) {
	   case 0:
		   setAhead(moveStep);
		   break;
		   
	   case 1:
		   setTurnRight(45.0);
		   setAhead(moveStep);
		   break;
		   
	   case 2:
		   setTurnRight(90.0);
		   setAhead(moveStep);
		   break;
		   
	   case 3:
		   setTurnRight(135.0);
		   setAhead(moveStep);
		   break;
		   
	   case 4:
		   setTurnRight(180.0);
		   setAhead(moveStep);
		   break;
		   
	   case 5:
		   setTurnRight(-135.0);
		   setAhead(moveStep);
		   break;
		   
	   case 6:
		   setTurnRight(-90.0);
		   setAhead(moveStep);
		   break;
		   
	   case 7:
		   setTurnRight(-45.0);
		   setAhead(moveStep);
		   break;
		   
	   case 8:
		   setFire(1);
		   break;
		  
	   case 9:
		   setFire(3);
		   break;
		   
	   }
   }
 
}