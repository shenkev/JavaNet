package IO;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.text.DecimalFormat;
import java.util.Arrays;

import org.la4j.Matrix;
import org.la4j.matrix.DenseMatrix;

public class WeightIO {

	public static void save(String folderPath, String subfolder,
			String fileName, Matrix[] weights) {

		   String rootFileName = folderPath + subfolder + fileName;
		   DecimalFormat numberFormat = new DecimalFormat("#.0000");
		   PrintStream w = null;
		   
		   for(int i=0; i < weights.length; i++) {
			   
			   double[][] arri = weights[i].toDenseMatrix().toArray();
			   int rows = arri.length;
			   int cols = arri[0].length;
			   
			   String fullPath = rootFileName + Integer.toString(i+1) + ".txt";
			   
				try {
					w = new PrintStream(new FileOutputStream(fullPath));
					
					for(int j=0; j < rows ; j++) {
						
						for(int k=0; k < cols; k++) {
							
							w.println(numberFormat.format(arri[j][k]));	
						}
						
					}
				} catch (IOException e) {
					e.printStackTrace();
				} finally {
					w.flush();
					w.close();
				}
		   }
		   
	}

	public static Matrix[] load(String folderPath, String subfolder,
			int[][] dims) throws IOException {
		
			String rootFolder = folderPath + subfolder;
			File folder = new File(rootFolder);
			String[] listOfFiles = folder.list();
			if(listOfFiles == null || listOfFiles.length == 0) {
				throw new IOException("Couldn't load weights, proceeding with fresh weights");
			}
			Arrays.sort(listOfFiles);
	
			Matrix[] W = new Matrix[dims.length];
			
			String line = null;
			
			for (int i=0; i < listOfFiles.length; i++) {
				
				double[][] m = new double[dims[i][0]][dims[i][1]];

				BufferedReader reader = new BufferedReader(
						new FileReader(rootFolder + listOfFiles[i]));
				
				try {
			        
			        for(int j=0; j < dims[i][0]; j++) {
			        	
			        	for(int k=0; k < dims[i][1]; k++) {
				        	line = reader.readLine();
				        	m[j][k] = Double.parseDouble(line);
			        	}
			        }
				} catch (IOException e) {
					e.printStackTrace();
				} finally {
					reader.close();
				}
				
				W[i] = DenseMatrix.from2DArray(m);
			}	
			
			return W;
	}
	
}
