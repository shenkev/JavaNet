import org.deeplearning4j.arbiter.DL4JConfiguration;
import org.deeplearning4j.arbiter.MultiLayerSpace;
import org.deeplearning4j.arbiter.data.DataSetIteratorProvider;
import org.deeplearning4j.arbiter.layers.DenseLayerSpace;
import org.deeplearning4j.arbiter.layers.OutputLayerSpace;
import org.deeplearning4j.arbiter.optimize.api.CandidateGenerator;
import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultReference;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultSaver;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxCandidatesCondition;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxTimeCondition;
import org.deeplearning4j.arbiter.optimize.api.termination.TerminationCondition;
import org.deeplearning4j.arbiter.optimize.candidategenerator.RandomSearchGenerator;
import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration;
import org.deeplearning4j.arbiter.optimize.parameter.continuous.ContinuousParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.integer.IntegerParameterSpace;
import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;
import org.deeplearning4j.arbiter.optimize.runner.LocalOptimizationRunner;
import org.deeplearning4j.arbiter.optimize.ui.ArbiterUIServer;
import org.deeplearning4j.arbiter.optimize.ui.listener.UIOptimizationRunnerStatusListener;
import org.deeplearning4j.arbiter.saver.local.multilayer.LocalMultiLayerNetworkSaver;
import org.deeplearning4j.arbiter.scoring.RegressionValue;
import org.deeplearning4j.arbiter.scoring.multilayer.TestSetAccuracyScoreFunction;
import org.deeplearning4j.arbiter.scoring.multilayer.TestSetRegressionScoreFunction;
import org.deeplearning4j.arbiter.task.MultiLayerNetworkTaskCreator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;

public class RandomGridSearch {

	public static void main(String[] args) {

		// Network Parameters
		int rngSeed = 123; // random number seed for reproducibility
		final Random rng = new Random(rngSeed);
		OptimizationAlgorithm algo = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
		int  iterations = 1; //Number of iterations per minibatch
		String hiddenAct = "tanh";
		String outAct = "identity";
		Updater updater = Updater.NESTEROVS;
		
		// Learning Parameters
		int numEpochs = 300;
		int batchSize = 18;
		
		// Dimensions
		int features = 13;
		int lay1 = 60;
		int lay2 = 20;
		int lay3 = 80;
		int outs = 6;
		
        //First: Set up the hyperparameter configuration space. This is like a MultiLayerConfiguration, but can have either
        // fixed values or values to optimize, for each hyperparameter

        ParameterSpace<Double> learningRateHyperparam = new ContinuousParameterSpace(0.001, 0.01);  //Values will be generated uniformly at random between 0.0001 and 0.1 (inclusive)
        ParameterSpace<Double> dropoutHyperparam = new ContinuousParameterSpace(0.5, 0.9);
        ParameterSpace<Double> L2RegularizationHyperparam = new ContinuousParameterSpace(0.00005, 0.002);
        ParameterSpace<Double> momentumHyperparam = new ContinuousParameterSpace(0.85, 0.93);
        ParameterSpace<Integer> layerSizeHyperparam1 = new IntegerParameterSpace(10,50);            //Integer values will be generated uniformly at random between 16 and 256 (inclusive)
        ParameterSpace<Integer> layerSizeHyperparam2 = new IntegerParameterSpace(10,50);            //Integer values will be generated uniformly at random between 16 and 256 (inclusive)
//        ParameterSpace<Integer> layerSizeHyperparam3 = new IntegerParameterSpace(10,50);            //Integer values will be generated uniformly at random between 16 and 256 (inclusive)
        
        MultiLayerSpace hyperparameterSpace = new MultiLayerSpace.Builder()
            //These next few options: fixed values for all models
        	.seed(rngSeed)
            .optimizationAlgo(algo)
            .iterations(iterations)
            .weightInit(WeightInit.XAVIER_UNIFORM)
        	.updater(updater)
            .regularization(true).l2(L2RegularizationHyperparam)
            //Learning rate: this is something we want to test different values for
            .learningRate(learningRateHyperparam)
            .addLayer( new DenseLayerSpace.Builder()
                    //Fixed values for this layer:
                    .activation(hiddenAct)
            		.adamMeanDecay(0.9)
            		.adamVarDecay(0.99)
            		.momentum(momentumHyperparam)
                    .nIn(features)  //Fixed input: 13
                    .nOut(lay1)
                    .build())
            .addLayer( new DenseLayerSpace.Builder()
                    //Fixed values for this layer:
                    .activation(hiddenAct)
            		.adamMeanDecay(0.9)
            		.adamVarDecay(0.99)
            		.momentum(momentumHyperparam)
                    .dropOut(dropoutHyperparam)
                    .nIn(lay1)  //Fixed input: 13
                    .nOut(lay2)
                    .build())
//            .addLayer( new DenseLayerSpace.Builder()
//                    //Fixed values for this layer:
//                    .activation(hiddenAct)
//            		.adamMeanDecay(0.9)
//            		.adamVarDecay(0.99)
//                    .dropOut(dropoutHyperparam)
//                    .nIn(layerSizeHyperparam2)  //Fixed input: 13
//                    .nOut(layerSizeHyperparam3)
//                    .build())
            .addLayer( new OutputLayerSpace.Builder()
                .activation(outAct)
                .lossFunction(LossFunctions.LossFunction.L2)
        		.adamMeanDecay(0.9)
        		.adamVarDecay(0.99)
        		.momentum(momentumHyperparam)
                .nIn(lay2)
                .nOut(outs)
                .build())
            .pretrain(false).backprop(true)
            .build();


        //Now: We need to define a few configuration options
        // (a) How are we going to generate candidates? (random search or grid search)
        CandidateGenerator<DL4JConfiguration> candidateGenerator = new RandomSearchGenerator<>(hyperparameterSpace);    //Alternatively: new GridSearchCandidateGenerator<>(hyperparameterSpace, 5, GridSearchCandidateGenerator.Mode.RandomOrder);

        // (b) How are going to provide data? For now, we'll use a simple built-in data provider for DataSetIterators
		Object[] dat = offlineTraining.loadOfflineDat();
		double[][] Xarr = (double[][])dat[0];
		double[][] yarr = (double[][])dat[1];
		
		INDArray X = Nd4j.create(Xarr);
		INDArray y = Nd4j.create(yarr);
		
		final DataSet allData = new DataSet(X,y);
		final List<DataSet> list = allData.asList();
		Collections.shuffle(list, rng);
		final List<DataSet> test = new ArrayList<DataSet>(list.subList(0, list.size()/8));
		final List<DataSet> train = new ArrayList<DataSet>(list.subList(list.size()/8, list.size()));
        
        DataSetIterator datTrain = new MultipleEpochsIterator(numEpochs, new ListDataSetIterator(train, batchSize));
        DataSetIterator datTest = new ListDataSetIterator(test, batchSize);
        DataProvider<DataSetIterator> dataProvider = new DataSetIteratorProvider(datTrain, datTest);

        // (c) How we are going to save the models that are generated and tested?
        //     In this example, let's save them to disk the working directory
        //     This will result in examples being saved to arbiterExample/0/, arbiterExample/1/, arbiterExample/2/, ...
        String baseSaveDirectory = "./2Layer_nesterov_hyperParamSearch/";
        File f = new File(baseSaveDirectory);
        if(f.exists()) f.delete();
        f.mkdir();
        ResultSaver<DL4JConfiguration,MultiLayerNetwork,Object> modelSaver = new LocalMultiLayerNetworkSaver<>(baseSaveDirectory);

//        // (d) What are we actually trying to optimize?
//        //     In this example, let's use classification accuracy on the test set
        ScoreFunction<MultiLayerNetwork,DataSetIterator> scoreFunction = new TestSetRegressionScoreFunction(RegressionValue.MSE);
//
//        // (e) When should we stop searching? Specify this with termination conditions
//        //     For this example, we are stopping the search at 15 minutes or 20 candidates - whichever comes first
        TerminationCondition[] terminationConditions = {new MaxTimeCondition(600, TimeUnit.MINUTES), new MaxCandidatesCondition(2000)};



//        //Given these configuration options, let's put them all together:
        OptimizationConfiguration<DL4JConfiguration, MultiLayerNetwork, DataSetIterator, Object> configuration
            = new OptimizationConfiguration.Builder<DL4JConfiguration, MultiLayerNetwork, DataSetIterator, Object>()
                .candidateGenerator(candidateGenerator)
                .dataProvider(dataProvider)
                .modelSaver(modelSaver)
                .scoreFunction(scoreFunction)
                .terminationConditions(terminationConditions)
                .build();

//        //And set up execution locally on this machine:
        IOptimizationRunner<DL4JConfiguration,MultiLayerNetwork,Object> runner
            = new LocalOptimizationRunner<>(configuration, new MultiLayerNetworkTaskCreator<>());


//        //Start the UI
        ArbiterUIServer server = ArbiterUIServer.getInstance();
        runner.addListeners(new UIOptimizationRunnerStatusListener(server));


//        //Start the hyperparameter optimization
        runner.execute();


//        //Print out some basic stats regarding the optimization procedure
        StringBuilder sb = new StringBuilder();
        sb.append("Best score: ").append(runner.bestScore()).append("\n")
            .append("Index of model with best score: ").append(runner.bestScoreCandidateIndex()).append("\n")
            .append("Number of configurations evaluated: ").append(runner.numCandidatesCompleted()).append("\n");
        System.out.println(sb.toString());


//        //Get all results, and print out details of the best result:
        int indexOfBestResult = runner.bestScoreCandidateIndex();
        List<ResultReference<DL4JConfiguration,MultiLayerNetwork,Object>> allResults = runner.getResults();

        OptimizationResult<DL4JConfiguration, MultiLayerNetwork, Object> bestResult;
		try {
			bestResult = allResults.get(indexOfBestResult).getResult();
			
	        MultiLayerNetwork bestModel = bestResult.getResult();

	        System.out.println("\n\nConfiguration of best model:\n");
	        System.out.println(bestModel.getLayerWiseConfigurations().toJson());
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

//        //Note: UI server will shut down once execution is complete, as JVM will exit
//        //So do a Thread.sleep(1 minute) to keep JVM alive, so that network configurations can be viewed
        try {
			Thread.sleep(60000);
	        System.exit(0);
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

}
