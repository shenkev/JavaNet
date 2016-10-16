import org.la4j.Vector;

public class libraryTest {

	public static void main(String[] args) {

		double[] b2 = new double[] {
			7, 8, 9
		};
		Vector v2 = Vector.fromArray(b2);
		Vector v3 = Vector.zero(3).add(1.0);
//		System.out.println(v3);
		System.out.println(v3.multiply(v3.outerProduct(v2)));
		
		//		double[][] a1 = new double[][]{
//		  { 1, 2, 3 },
//		  { 4, 5, 6 },
//		  { 7, 8, 9 }
//		};
//	double[][] a2 = new double[][]{
//		  { 2, 2, 2 },
//		  { 2, 2, 2 },
//		  { 2, 2, 2 }
//		};	
//	Matrix m1 = DenseMatrix.from2DArray(a1);
//	Matrix m2 = DenseMatrix.from2DArray(a2);
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

}
