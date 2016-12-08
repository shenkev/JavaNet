

import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class offlineTrain {

	public static MultiLayerNetwork main(String[] args) {

		//Initialize the user interface backend
	    UIServer uiServer = UIServer.getInstance();

	    //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
	    StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later
	    
	    //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
	    uiServer.attach(statsStorage);
		
		Object[] dat = offlineTraining.loadOfflineDat();
		double[][] Xarr = (double[][])dat[0];
		double[][] yarr = (double[][])dat[1];
		
		INDArray X = Nd4j.create(Xarr);
		INDArray y = Nd4j.create(yarr);
				
		System.out.println(y);
		
		// Network Parameters
		int rngSeed = 123; // random number seed for reproducibility
		final Random rng = new Random(rngSeed);
		OptimizationAlgorithm algo = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
		int  iterations = 1; //Number of iterations per minibatch
		String hiddenAct = "tanh";
		String outAct = "identity";
		Updater updater = Updater.ADAM;
		
		// Learning Parameters
		double rate = 0.03;
		double regularize = 0.002;
		double dropOut = 0.7;
		int numEpochs = 200;
		int batchSize = 18;
		int printEvery = Xarr.length/batchSize;
		
		// Dimensions
		int features = 13;
		int lay1 = 60;
		int lay2 = 20;
//		int lay3 = 80;
		int outs = 6;
		
		final DataSet allData = new DataSet(X,y);
		final List<DataSet> list = allData.asList();
		Collections.shuffle(list, rng);
		DataSetIterator iterator = new ListDataSetIterator(list, batchSize);
		
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngSeed)
                .optimizationAlgo(algo)
                .iterations(iterations)
                .activation(hiddenAct)
                .weightInit(WeightInit.XAVIER_UNIFORM)
                .updater(updater)
                .regularization(true).l2(regularize).dropOut(dropOut)
                .list()
                .layer(0, new DenseLayer.Builder()
                		.learningRate(rate)
                		.adamMeanDecay(0.9)
                		.adamVarDecay(0.99)
                        .nIn(features)
                        .nOut(lay1)
                        .build())
                .layer(1, new DenseLayer.Builder()
                		.learningRate(rate)
                		.adamMeanDecay(0.9)
                		.adamVarDecay(0.99)
                        .nIn(lay1)
                        .nOut(lay2)
                        .build())
//                .layer(2, new DenseLayer.Builder()
//                		.learningRate(rate)
//                		.adamMeanDecay(0.9)
//                		.adamVarDecay(0.99)
//                        .nIn(lay2)
//                        .nOut(lay3)
//                        .build())
                .layer(2, new OutputLayer.Builder()
                        .activation(outAct)
                        .lossFunction(LossFunctions.LossFunction.L2)
                        .learningRate(rate)
                		.adamMeanDecay(0.9)
                		.adamVarDecay(0.99)
                        .nIn(lay2)
                        .nOut(outs)
                        .build())
                .pretrain(true).backprop(true)
                .build();
		

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        
        //Load the model
        try {
			model = ModelSerializer.restoreMultiLayerNetwork("./results/offline/MyMultiLayerNetwork.zip");
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
        model.init();
        model.setListeners(new ScoreIterationListener(printEvery));
        
	    //Then add the StatsListener to collect this information from the network, as it trains
	    model.setListeners(new StatsListener(statsStorage));
        
        System.out.println("Training model");
        for( int i=0; i< numEpochs; i++ ){
        	iterator.reset();
            model.fit(iterator);
        }
        
        System.out.println(model.output(X, false));
                
        //Save the model
        File locationToSave = new File("./results/offline/MyMultiLayerNetwork.zip");      //Where to save the network. Note: the file is in .zip format - can be opened externally
        boolean saveUpdater = true;                                     //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
        try {
			ModelSerializer.writeModel(model, locationToSave, saveUpdater);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        
        return model;
        
	}
	
}
