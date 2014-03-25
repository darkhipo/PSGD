package sgd;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import Jama.Matrix;

public class UtilLib
{
    static int START_OFFSET     = 0;
    
    public static double colMean(Matrix m)
    {
        double mean = 0;
        for (int i=0; i<m.getRowDimension(); i++)
        {
            mean += m.get(i,0);
        }
        mean /= (double) m.getRowDimension();
        return mean;
    }
    public static double colMag(Matrix m)
    {
        double mag = 0;
        for (int i=0; i<m.getRowDimension(); i++)
        {
            mag += Math.pow(m.get(i,0),2);
        }
        mag = Math.sqrt(mag);
        return mag;
    }
    public static Matrix colAbs(Matrix m)
    {
        Matrix ret = m.copy();
        for (int i=0; i<m.getRowDimension(); i++)
        {
            ret.set(i,0,m.get(i,0));
        }
        return ret;
    }
    public static double colAbsDiff(Matrix m1, Matrix m2 )
    {
        double ad = 0;
        for (int i=0; i<m1.getRowDimension(); i++)
        {
            ad += Math.abs( m2.get(i,0) - m1.get(i,0) );
        }
        return ad;
    }
    public static double colMax(Matrix m)
    {
        double max = -Double.MAX_VALUE;
        for (int i=0; i<m.getRowDimension(); i++)
        {
            max = Math.max(m.get(i,0),max);
        }
        //System.out.println(1/max);
        return max;
    }
    public static double colMin(Matrix m)
    {
        double min = Double.MAX_VALUE;
        for (int i=0; i<m.getRowDimension(); i++)
        {
            min = Math.min(m.get(i,0),min);
        }
        //System.out.println(1/max);
        return min;
    }
    public static Matrix colEucNorm(Matrix m)
    {
        //double acc = 0;
        //for (int i=0; i<m.getRowDimension(); i++)
        //{
        //    acc += Math.pow(m.get(i,0),2);
        //}
        //acc = Math.sqrt(acc);
        //System.out.println(acc + " " + m.normF());
        //System.out.println(1/max);
        //return m.times( 1.0 / acc );
        //System.out.println(m.normF());
        return m.times( 1.0 / m.normF() );
    }
    public static Matrix getX(Matrix dat)
    {
        return dat.getMatrix(START_OFFSET,dat.getRowDimension()-1,START_OFFSET,dat.getColumnDimension()-2);
    }
    public static Matrix getY(Matrix dat)
    {
        return dat.getMatrix(START_OFFSET,dat.getRowDimension()-1,dat.getColumnDimension()-1,dat.getColumnDimension()-1);
    }
    public static Matrix matMergeXY(Matrix x, Matrix y)
    {
        Matrix ret = new Matrix (x.getRowDimension(),x.getColumnDimension()+1);
        ret.setMatrix(START_OFFSET, x.getRowDimension()-1, START_OFFSET, x.getColumnDimension()-1, x);
        ret.setMatrix(START_OFFSET, x.getRowDimension()-1, x.getColumnDimension(), x.getColumnDimension(), y);
        return ret;
    }
    public static Map<String, Matrix> splitXYs(Map<String, Matrix> map)
    {
        map.put("train_X" , sgd.UtilLib.getX(map.get("train_dat")));
        map.put("train_Y" , sgd.UtilLib.getY(map.get("train_dat" )));
        map.put("test_X"  , sgd.UtilLib.getX(map.get("test_dat" )));
        map.put("test_Y"  , sgd.UtilLib.getY(map.get("test_dat" )));
        return map;
    }
    public static Matrix matMergeX(Matrix m1, Matrix m2)
    {
        Matrix rna_lump = new Matrix(m1.getRowDimension()+m2.getRowDimension(),m1.getColumnDimension(),0);
        rna_lump.setMatrix(0, m2.getRowDimension()-1, 0, m2.getColumnDimension()-1, m2);
        rna_lump.setMatrix(m2.getRowDimension(), rna_lump.getRowDimension()-1, 0, m1.getColumnDimension()-1, m1);
        return rna_lump;
    }
    public static double errorRate( SGD sgd, Matrix test_Y)
    {
        Matrix Y_hat = sgd.predict();
        Matrix err   = Y_hat.minus(test_Y);
        
        double absErr = 0;
        for (int i=0; i<err.getRowDimension(); i++ )
        {
            absErr += Math.abs(err.get(i, 0));
        }
        absErr /= err.getRowDimension();
        return absErr;
    }
    public static Matrix randomRowShuffle(Matrix dat)
    {
        return new Matrix(randomRowShuffle(dat.getArray()));
    }
    public static double [] [] randomRowShuffle(double [][] dat )
    {
        ArrayList <Integer> a = new ArrayList<Integer>();
        for (int i=0; i<dat.length; i++)
        {
            a.add(i);
        }
        Collections.shuffle(a);
        double [][] dat_2 = new double [dat.length] [dat[0].length];
        for(int i = 0; i< dat.length;i++)
        {
            dat_2[i] = dat[a.get(i)];
        }
        return dat_2;
    }   
    public static Map<String,Matrix> trainTestSplit(Matrix dat, double split)
    {
        HashMap<String,Matrix> args = new HashMap<String, Matrix> ();
        if ( split == 0 || split == 1 )
        {
            args.put("train_dat",dat);
            args.put("test_dat" ,dat);
            return args;
        }
        int trainTop                = (int) Math.floor( split*dat.getRowDimension() );
        int testBottom              = trainTop+1;
        
        args.put("train_dat",dat.getMatrix(START_OFFSET,trainTop,START_OFFSET,dat.getColumnDimension()-1));
        args.put("test_dat" ,dat.getMatrix(testBottom,dat.getRowDimension()-1,START_OFFSET,dat.getColumnDimension()-1));
              
        return args;
    }
    public static Matrix standardize(Matrix m)
    {
        Matrix r = new Matrix(m.getRowDimension(),m.getColumnDimension(),0);
        for (int i=0; i<m.getColumnDimension(); i++)
        {
            Matrix col  = m.getMatrix(0,m.getRowDimension()-1,i,i);
            Matrix ones = (new Matrix(m.getRowDimension(),1,1));
            Matrix mean = ones.times(colMean(col));
            Matrix temp = col.minus(mean);
            temp        = temp.arrayTimes(temp);
            Double std  = Math.sqrt(colMean(temp));
            Double cm   = colMax(col);
            Double ra   = colMax(col) - colMin(col);
            Double coef = 1.0/ra;//1.0/(colMax(col) - colMin(col));
            coef        = Double.isInfinite(coef) ? 1.0 : coef;
            col         = col.minus(mean); //Mean Normalization.
            col         = col.times(coef);
            r.setMatrix(0,m.getRowDimension()-1,i,i,col);
        }
        return r;
    }    
    public static void writeMat(Matrix mat, String fname)
    {
        try
        {
            ObjectOutputStream writer = new ObjectOutputStream (new FileOutputStream(fname));
            writer.writeObject(mat);
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
    }
    public static Matrix readMat(String fname)
    {
        Matrix ret = null;
        try
        {
            ObjectInputStream reader = new ObjectInputStream (new FileInputStream(fname));
            ret = (Matrix) reader.readObject();
        }
        catch ( ClassNotFoundException e)
        {
            e.printStackTrace();
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
        return ret;
    }
    public static Matrix parseLibSvmFast(String fname, int rowC, int colC)
    {
        Matrix m = new Matrix(rowC, colC+1, 0.0);
        try
        {
            BufferedReader br                    = new BufferedReader(new InputStreamReader(new FileInputStream(fname)));
            ArrayList<ArrayList<Double>> mat     = new ArrayList<ArrayList<Double>>(rowC);
            ArrayList<Double>            classes = new ArrayList<Double>(rowC);
            ArrayList<String []>         blob    = new ArrayList<String []>();
            
            //Read into mem.
            String strLine;
            while( ( strLine = br.readLine() ) != null )
            {
                String [] tokens = strLine.split("\\s+");
                Integer d_class  = Double.valueOf(tokens[0].replaceFirst("\\+", "")).intValue();
                d_class          = (d_class > 0) ? 1 : 0; 
                tokens           = Arrays.copyOfRange(tokens, 1, tokens.length);
                classes.add(d_class.doubleValue());
                blob.add(tokens);
            }
            for (int r=0; r<blob.size(); r++)
            {
                String [] tokens = blob.get(r);
                for ( String tok : tokens )
                {
                    String [] tuple    = tok.split(":");
                    Integer   colIndex = Integer.valueOf(tuple[0]);
                    Double    value    = Double.valueOf (tuple[1]);
                    m.set(r, colIndex-1, value);
                }
                m.set( r, colC, classes.get(r) );
                //System.out.printf(r+"~Read %.8f\n", ((double)1+r++)/rowC);
            }
        }
        catch(Exception e)
        {
            e.printStackTrace();
            System.out.println(e);
        }
        return m;
    }
    public static Matrix parseLibSvm(String fname, int rowC, int colC)
    {
        try
        {
            BufferedReader br                    = new BufferedReader(new InputStreamReader(new FileInputStream(fname)));
            ArrayList<ArrayList<Double>> mat     = new ArrayList<ArrayList<Double>>(rowC);
            ArrayList<Double>            classes = new ArrayList<Double>(rowC);
            int cols                             = colC;
            String [] tokens;
            String strLine;
            
            while( ( strLine = br.readLine() ) != null )
            {
                tokens           = strLine.split("\\s+");
                Integer d_class  = Double.valueOf(tokens[0].replaceFirst("\\+", "")).intValue();
                d_class          = (d_class > 0) ? 1 : 0; 
                tokens           = Arrays.copyOfRange(tokens, 1, tokens.length);
                classes.add(d_class.doubleValue());
                
                ArrayList<Double> row = new ArrayList<Double>(Collections.nCopies(rowC, 0.0));
                for ( String tok : tokens )
                {
                    String [] tuple    = tok.split(":");
                    Integer   colIndex = Integer.valueOf(tuple[0]);
                    Double    value    = Double.valueOf (tuple[1]);
                    
                    if ( row.size() < colIndex )
                    {
                        for (int i=row.size(); i<colIndex-1; i++)
                        {
                            row.add(i, 0.0);
                        }
                    }
                    cols = Math.max(cols, colIndex);
                    row.add(colIndex-1, value); 
                }
                //System.out.printf("~Read %.2f\n", (z++)/rowC);
                mat.add(row);
            }
            for (int i=0; i<mat.size(); i++)
            {
                if ( mat.get(i).size() < cols )
                {
                    for (int j=mat.get(i).size()-1; j<cols-1; j++)
                    {
                        mat.get(i).add(j,0.0);
                    }
                }
            }           
            Matrix m = new Matrix( mat.size(), mat.get(0).size()+1 );
            for (int i = 0; i<mat.size(); i++)
            {
                Double [] d    = mat.get(i).toArray(new Double [1]);
                double [][] d2 = new double [1][d.length+1];
                
                for (int j = 0; j<d.length; j++)
                {
                    d2[0][j] = d[j];
                }
                d2[0][d.length] = classes.get(i);
                Matrix row      = new Matrix(d2);
                m.setMatrix(i,i,0,mat.get(0).size(),row);
            }
            return m;
        }
        catch(Exception e)
        {
            e.printStackTrace();
            System.out.println(e);
        }
        return null;
    }
}