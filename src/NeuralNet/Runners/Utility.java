package NeuralNet.Runners;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;

public class Utility {

	/**
	 * Saves the weights to file. Format of the output is as follows:
	 * 		numInputs
	 * 		numHidden
	 * 		weight input 0 to hidden 0
	 * 		weight input 0 to hidden 1
	 * 		.....
	 * 		weight input j to hidden i
	 * 		weight hidden 0 to output
	 * 		weight hidden i to output
	 * @param argFile A "File" handle to where the weights are to be written
	 */
//	public void save ( File argFile ) {
//		PrintStream saveFile = null;
//		
//		try {
//			saveFile = new PrintStream( new RobocodeFileOutputStream( argFile ));
//		}
//		catch (IOException e) {
//			System.out.println( "*** Could not create output stream for NN save file.");
//		}
//		
//		saveFile.println( numInputs );
//		saveFile.println( numHidden );
//		
//		// First save the weights from the input to hidden neurons (one line per weight)
//		for ( int i=0; i<numHidden; i++) {
//			for ( int j=0; j<numInputs; j++) {
//				saveFile.println( weightInputToHidden [i][j] );
//			}
//			saveFile.println( weightInputToHidden [i][numInputs] ); // Save bias weight for this hidden neuron too
//		}
//		// Now save the weights from the hidden to the output neuron
//		for (int i=0; i<numHidden; i++) {
//			saveFile.println( weightHiddenToOutput [i] );
//		}	
//		saveFile.println( weightHiddenToOutput [numHidden] ); // Save bias weight for output neuron too.
//		saveFile.close();
//	}
//	
//	/**
//	 * Loads the weights from file. Format of the file is expected to follow
//	 * that specified in the "save" method specified elsewhere in this class.
//	 * @param argFileName the name of the file where the weights are to be found
//	 */
//	public void load ( String argFileName ) throws IOException {
//
//		FileInputStream inputFile = new FileInputStream( argFileName );
//		BufferedReader inputReader = new BufferedReader(new InputStreamReader( inputFile ));		
//				
//		// Check that NN defined for file matches that created
//		int numInputInFile = Integer.valueOf( inputReader.readLine() );
//		int numHiddenInFile = Integer.valueOf( inputReader.readLine() );
//		// System.out.println("--- File: #inputs=" + numInputInFile + ", #hidden=" + numHiddenInFile);
//		// System.out.println("--- NNet: #inputs=" + numInputs + ", #hidden=" + numHidden);
//		
//		if ( numInputInFile != numInputs ) {
//			System.out.println ( "*** Number of inputs in file is " + numInputInFile + " Expected " + numInputs );
//			throw new IOException();
//		}
//		if ( numHiddenInFile != numHidden ) {
//			System.out.println ( "*** Number of hidden in file is " + numHiddenInFile + " Expected " + numHidden );
//			throw new IOException();
//		}
//		if ( (numInputInFile != numInputs) || (numHiddenInFile != numHidden) ) return;
//		
//		// First load the weights from the input to hidden neurons (one line per weight)
//		for ( int i=0; i<numHidden; i++) {
//			for ( int j=0; j<numInputs; j++) {
//				weightInputToHidden [i][j] = Double.valueOf( inputReader.readLine() );
//			}
//			weightInputToHidden [i][numInputs] = Double.valueOf( inputReader.readLine() ); // Load bias weight for this hidden neuron too
//		}
//		// Now load the weights from the hidden to the output neuron
//		for (int i=0; i<numHidden; i++) {
//			weightHiddenToOutput [i] = Double.valueOf( inputReader.readLine() );
//		}	
//		weightHiddenToOutput [numHidden] = Double.valueOf( inputReader.readLine() ); // Load bias weight for output neuron too.
//	}
	
}
