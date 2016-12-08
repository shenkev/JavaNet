import java.awt.Color;
import java.io.IOException;
import java.util.Random;

import org.la4j.Matrix;
import org.la4j.Vector;
import org.la4j.matrix.DenseMatrix;

import IO.LUTIO;
import IO.ResultsIO;
import IO.WeightIO;
import NeuralNet.NeuralNet;
import NeuralNet.Costs.Loss;
import NeuralNet.Costs.SquareLoss;
import NeuralNet.NonLinFuncs.BipolarSigmoid;
import NeuralNet.NonLinFuncs.NonLinFunction;
import NeuralNet.NonLinFuncs.ReLu;
import NeuralNet.NonLinFuncs.Tanh;
import NeuralNet.Optimizers.GradientDescent;
import NeuralNet.Optimizers.Optimizer;
import robocode.AdvancedRobot;
import robocode.BulletHitEvent;
import robocode.DeathEvent;
import robocode.HitWallEvent;
import robocode.ScannedRobotEvent;
import robocode.WinEvent;

public class NNRLRobot extends AdvancedRobot {
	
	// ============================ Local Storage ============================= //
   static int iter = 0;
   static int previousWins = 0;
   static int previousLosses = 0;
   static final int saveWeightsEvery = 1000;
   static final int saveResultsEvery = 100;
   static final int printLossEvery = 1000;
   static double[] errs = new double[10000];
   static int lossIter = 0;
   static int printIter = 0;
   static String folderPath = "./../workspace/Robocode/results/NNWeights/";
   static String saveWName = "weight";
   static String savebName = "bias";
   static String resultFileName = "results.txt";
   static String lossFileName = "loss.txt";
   static String loadWfolder = "w/";
   static String loadbfolder = "b/";
	
	// ========================== RL Hyper Parameters ============================ //
   final double gamma = 0.975;
   final double epsilon = 0.1;	// should start at 1 and decrement to 0.1
   
   final double rewardOnWin = 1.0;
   
	// ========================== State Representation ========================== //
   Random rand = new Random();
    
   static final int numberOfTotalStateParams = 13;
   static final int numberOfActions = 6;
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
   static final double energyScale = 26.0;
   static final double energyMean = 40.0;
   static final double velocityScale = 1.4;
   static final double velocityMean = 2.5;
   static final double xScale = 240.0;
   static final double yScale = 180.0;
   static final double distScale = 210.0;
   static final double angleScale = 1.4;
   
   double fieldWidth;
   double fieldHeight;
   
   // State-Actions
   Matrix currentState;
   int currentAction;
   // Initialize state
   static final double[] prevStateArr = new double[]{
		   100/energyScale, 0,
		   0, 0,
		   1, 0,
		   100/energyScale,
		   1, 0,
		   300/800, 0,
		   1, 0
   };
   Matrix previousState = DenseMatrix.from1DArray(
		   1, numberOfTotalStateParams, prevStateArr);;
   
   double reward;
   
	// ========================== Neural Network Setup ========================== //
   static final int batchSize = 1;
   static final int[] layerDims = new int[] { 60, 20, numberOfActions };
   static final int[][] dims = new int[][]{ // Make sure these match
		{ numberOfFeatures, 60 },
		{ 60, 20 },
		{ 20, numberOfActions }
   };
   static final int[][] dimsb = new int[][]{ // Also make sure these match
		{ 1, 60 },
		{ 1, 20 },
		{ 1, numberOfActions }
	};
   static final int noLayers = layerDims.length;
   static double trainRate = 0.0001;
   static double momentum = 0.9;
   static double lambda = 0.1;
   static final NonLinFunction nonLinFunction = new Tanh();
   static final NonLinFunction outputFunction = new Tanh();
   static final Loss lossFunc = new SquareLoss();	// need to support multi dimensions
   static final Optimizer optimizer = new GradientDescent(
		   trainRate, momentum, noLayers, layerDims, numberOfFeatures, lambda);
   static final int randSeed = 800;	
   
   static final NeuralNet nn = new NeuralNet(numberOfFeatures, batchSize, noLayers,
		   layerDims, nonLinFunction, outputFunction, lossFunc, optimizer, randSeed);

   // ========================== Start of Robot Code ========================== //
   public void run() {
   	// run at beginning of EACH GAME
	   	if (iter == 0) {
	   		try {
	   			// load weights
				nn.setW(WeightIO.load(folderPath, loadWfolder, dims));
				
				// load biases
				Matrix[] mb = WeightIO.load(folderPath, loadbfolder, dimsb);
				Vector[] bb = new Vector[mb.length];
				for(int i=0; i < bb.length; i++) {
					bb[i] = mb[i].toRowVector();
				}
				nn.setb(bb);
				
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
	   	}
	   
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
	   
	   if (lossIter % printLossEvery == 0) {
		   errs[printIter] = loss;
		   printIter++;
		   lossIter = 0;
	   }
	   lossIter++;
	   
	   // update previous state to current state
	   previousState = currentState;

   }

   public void onWin(WinEvent event) {

	   // update previous action-state with reward only
	   Matrix QforPrevStateActionsMatrix = nn.forwardProp(previousState);
	   QforPrevStateActionsMatrix.set(0, currentAction, rewardOnWin);
	   double loss = nn.runOnePass(previousState, QforPrevStateActionsMatrix);

	   if (lossIter % printLossEvery == 0) {
		   errs[printIter] = loss;
		   printIter++;
		   lossIter = 0;
	   }
	   lossIter++;
	   // save
	   iter++;
	   previousWins++;
	   // save results
	   if (iter % saveResultsEvery == 0) {
		   ResultsIO.writeWins(resultFileName, previousWins);
		   previousWins = 0;
		   previousLosses = 0;
		   ResultsIO.save(lossFileName, errs);
	   }
	   // save Weights
	   if (iter % saveWeightsEvery == 0) {
		   // save weights
		   WeightIO.save(folderPath, loadWfolder, saveWName, nn.getW());
		   
		   // save biases
		   Vector[] bb = nn.getb();
		   Matrix[] mb = new Matrix[bb.length];
		   for(int i=0; i < mb.length; i++) {
		   	mb[i] = bb[i].toRowMatrix();
		   }
		   WeightIO.save(folderPath, loadbfolder, savebName, mb);
	   }
   }
   
   public void onDeath(DeathEvent event) {
	  
	   // update previous action-state with reward only
	   Matrix QforPrevStateActionsMatrix = nn.forwardProp(previousState);
	   QforPrevStateActionsMatrix.set(0, currentAction, (-1.0*rewardOnWin));
	   double loss = nn.runOnePass(previousState, QforPrevStateActionsMatrix);

	   if (lossIter % printLossEvery == 0) {
		   errs[printIter] = loss;
		   printIter++;
		   lossIter = 0;
	   }
	   lossIter++;
	   // save
	   iter++;
	   previousLosses++;
	   // save results
	   if (iter % saveResultsEvery == 0) {
		   ResultsIO.writeWins(resultFileName, previousWins);
		   previousWins = 0;
		   previousLosses = 0;
		   ResultsIO.save(lossFileName, errs);
	   }
	   // save Weights
	   if (iter % saveWeightsEvery == 0) {
		   // save weights
		   WeightIO.save(folderPath, loadWfolder, saveWName, nn.getW());
		   
		   // save biases
		   Vector[] bb = nn.getb();
		   Matrix[] mb = new Matrix[bb.length];
		   for(int i=0; i < mb.length; i++) {
		   	mb[i] = bb[i].toRowMatrix();
		   }
		   WeightIO.save(folderPath, loadbfolder, savebName, mb);
	   }
   }
   
   public void onBulletHit(BulletHitEvent event) {
	   
   }
   
   public void onHitWall(HitWallEvent event) {
	   double hitWallPenalty = 0.1;
	   
	   // update previous action-state with reward only
	   Matrix QforPrevStateActionsMatrix = nn.forwardProp(previousState);
	   QforPrevStateActionsMatrix.set(0, currentAction, (-1.0*hitWallPenalty));
	   double loss = nn.runOnePass(previousState, QforPrevStateActionsMatrix);

	   if (lossIter % printLossEvery == 0) {
		   errs[printIter] = loss;
		   printIter++;
		   lossIter = 0;
	   }
	   lossIter++;
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
			   
			   (ownEnergy-energyMean)/energyScale,
			   (ownVelocity-velocityMean)/velocityScale,
			   (ownX - (800.0/2.0))/xScale, 
			   (ownY - (600.0/2.0))/yScale,
			   angleScale*Math.cos(ownHeading), angleScale*Math.sin(ownHeading),
			   (enemyEnergy-energyMean)/energyScale,
			   angleScale*Math.cos(enemyBearing), angleScale*Math.sin(enemyBearing),
			   (enemyDistance-200.0)/distScale,
			   (enemyVelocity-velocityMean)/velocityScale,
			   angleScale*Math.cos(enemyHeading), angleScale*Math.sin(enemyHeading)
			   
	   };
   }
   
   public void takeAction(int action) {
	   
	   double moveStep = 50.0;
	   
	   
	   switch(action) {
	   case 0:
		   setAhead(moveStep);
		   break;
		   
	   case 1:
		   setBack(moveStep);
		   break;
		   
	   case 2:
		   setTurnLeft(90.0);
		   setAhead(moveStep);
		   break;
		   
	   case 3:
		   setTurnRight(90.0);
		   setAhead(moveStep);
		   break;
		   
	   case 4:
		   setFire(1);
		   break;
		   
	   case 5:
		   setFire(2);
		   break;
		   
	   }
   }
 
}