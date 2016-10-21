import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.la4j.Matrix;
import org.la4j.Vector;
import org.la4j.matrix.DenseMatrix;

public class libraryTest {

	public static void main(String[] args) {

//		double[] b2 = new double[] {
//			7, 8, 9
//		};
//		Vector v2 = Vector.fromArray(b2);
//		Vector v3 = Vector.zero(3).add(1.0);
//		System.out.println(v3);
//		System.out.println(v3.multiply(v3.outerProduct(v2)));
		
	double[][] a1 = new double[][]{
		  { 1, 2, 3 },
		  { 4, 5, 6 },
		  { 7, 8, 9 }
		};
	double[][] a2 = new double[][]{
		  { 2, 2, 2 },
		  { 2, 2, 2 },
		  { 2, 2, 2 }
		};	
	Matrix m1 = DenseMatrix.from2DArray(a1);
	Matrix m2 = DenseMatrix.from2DArray(a2);
	
	
	int[] arr = {0, 1, 2};
	
	System.out.println(m1.select(generateBatch(2, 3), arr));
	System.out.println(m1);

	System.out.println(Arrays.toString(generateZeroToNm1Array(5)));


//
//	Vector v1 = Vector.random(3, new Random(500));
	
	
//	System.out.println(m1.multiply(m2));
//	System.out.println(m1.hadamardProduct(m2));
	
//	System.out.println(m1);
//	System.out.println(DenseMatrix.random(3, 3, new Random()));
//	System.out.println(v1);
//	v1 = v1.add(v1);
//	System.out.println(v1);
//	for (int i = 0; i < 3; i++) {
//		m1.setRow(i, m1.getRow(i).add(v1));
//	}
//	System.out.println(m1);
//	System.out.println(m1.multiply(m2));
//	System.out.println(m1.multiply(m2).add(m2));
//	System.out.println(m1.transform(new xsquared()));

		
	}
	
	public static int[] generateBatch(int size, int upper) {
		
		if (size > upper - 1) {
			throw new IllegalArgumentException("You shouldn't ask for more values than can provide");
		}
		
		Set<Integer> set = new HashSet<Integer>();
		Random rand = new Random();
		while (set.size() < size) {
			set.add(rand.nextInt(upper));
		}
		
		Integer[] arr = new Integer[size];
		arr = set.toArray(arr);
		int[] arr2 = new int[size];
		for (int i = 0; i < arr.length; i++) {
			arr2[i] = arr[i].intValue();
		}
		
		System.out.println(Arrays.toString(arr2));
		return arr2;
	}
	
	public static int[] generateZeroToNm1Array(int n) {
		
		int[] arr = new int[n];
		
		for (int i = 0; i < n; i++) {
			arr[i] = i;
		}
		return arr;
	}
}
