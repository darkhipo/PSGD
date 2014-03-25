package sgd;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import javax.rmi.CORBA.Util;

import Jama.*;
import sgd.UtilLib;

public class SGD_Test 
{
	static int START_OFFSET     = 0;
	static double [][] iris2    = new double[][] {{5.2,4.1,1.5,0.1,1}, {6.7,3,5.2,2.3,0}, {6.5,3.2,5.1,2,0}, {4.7,3.2,1.3,0.2,1}, {4.4,3.2,1.3,0.2,1}, {5.1,3.5,1.4,0.2,1}, {4.8,3.4,1.9,0.2,1}, {4.6,3.1,1.5,0.2,1}, {5.7,3.8,1.7,0.3,1}, {5.4,3.9,1.7,0.4,1}, {5,3.2,1.2,0.2,1}, {6.9,3.1,5.4,2.1,0}, {7.2,3.2,6,1.8,0}, {7.3,2.9,6.3,1.8,0}, {5.1,3.8,1.5,0.3,1}, {4.9,3.1,1.5,0.1,1}, {7.7,2.6,6.9,2.3,0}, {6.2,3.4,5.4,2.3,0}, {4.9,3.1,1.5,0.2,1}, {7.2,3.6,6.1,2.5,0}, {5.4,3.4,1.7,0.2,1}, {4.9,2.5,4.5,1.7,0}, {5.9,3,5.1,1.8,0}, {5.8,2.8,5.1,2.4,0}, {7.7,2.8,6.7,2,0}, {4.9,3.6,1.4,0.1,1}, {6.1,3,4.9,1.8,0}, {7.4,2.8,6.1,1.9,0}, {6.8,3.2,5.9,2.3,0}, {6.3,2.8,5.1,1.5,0}, {5.1,3.8,1.9,0.4,1}, {7.9,3.8,6.4,2,0}, {4.8,3.1,1.6,0.2,1}, {6.3,3.3,6,2.5,0}, {6.5,3,5.2,2,0}, {5.1,3.4,1.5,0.2,1}, {6.4,3.1,5.5,1.8,0}, {6.4,3.2,5.3,2.3,0}, {5.5,3.5,1.3,0.2,1}, {5.6,2.8,4.9,2,0}, {6.4,2.8,5.6,2.1,0}, {5.1,3.7,1.5,0.4,1}, {5,3,1.6,0.2,1}, {5.2,3.5,1.5,0.2,1}, {5.3,3.7,1.5,0.2,1}, {4.8,3,1.4,0.3,1}, {6.3,2.5,5,1.9,0}, {6.3,2.7,4.9,1.8,0}, {5.4,3.4,1.5,0.4,1}, {5.4,3.9,1.3,0.4,1}, {5.7,4.4,1.5,0.4,1}, {6.5,3,5.5,1.8,0}, {6.7,3.3,5.7,2.1,0}, {4.4,3,1.3,0.2,1}, {4.8,3.4,1.6,0.2,1}, {5,3.6,1.4,0.2,1}, {5,3.4,1.6,0.4,1}, {6.3,2.9,5.6,1.8,0}, {5.1,3.8,1.6,0.2,1}, {6.4,2.7,5.3,1.9,0}, {7.2,3,5.8,1.6,0}, {6,3,4.8,1.8,0}, {6.2,2.8,4.8,1.8,0}, {6.7,3.3,5.7,2.5,0}, {6.4,2.8,5.6,2.2,0}, {4.7,3.2,1.6,0.2,1}, {6.9,3.2,5.7,2.3,0}, {4.3,3,1.1,0.1,1}, {5.2,3.4,1.4,0.2,1}, {5.4,3.7,1.5,0.2,1}, {7.1,3,5.9,2.1,0}, {7.6,3,6.6,2.1,0}, {5.7,2.5,5,2,0}, {4.9,3,1.4,0.2,1}, {6.5,3,5.8,2.2,0}, {5.8,2.7,5.1,1.9,0}, {4.4,2.9,1.4,0.2,1}, {6.3,3.4,5.6,2.4,0}, {5,3.5,1.6,0.6,1}, {5.8,4,1.2,0.2,1}, {5.1,3.3,1.7,0.5,1}, {5,3.4,1.5,0.2,1}, {5.8,2.7,5.1,1.9,0}, {4.6,3.4,1.4,0.3,1}, {6.1,2.6,5.6,1.4,0}, {5.1,3.5,1.4,0.3,1}, {4.6,3.6,1,0.2,1}, {6,2.2,5,1.5,0}, {6.9,3.1,5.1,2.3,0}, {6.8,3,5.5,2.1,0}, {6.7,3.1,5.6,2.4,0}, {4.5,2.3,1.3,0.3,1}, {4.6,3.2,1.4,0.2,1}, {5,3.5,1.3,0.3,1}, {4.8,3,1.4,0.1,1}, {7.7,3.8,6.7,2.2,0}, {7.7,3,6.1,2.3,0}, {5,3.3,1.4,0.2,1}, {5.5,4.2,1.4,0.2,1}, {6.7,2.5,5.8,1.8,0}};  			
	static double trainingSplit = 0.40;
	static boolean doRand       = true;
	
	static ArrayList<Double> w1 = new ArrayList<Double> ();
	static ArrayList<Double> w2 = new ArrayList<Double> ();
	static ArrayList<Double> w3 = new ArrayList<Double> ();
    
	public static double sgd_0_1 ( Map<String, Matrix> args )
	{
	    boolean includeOffsetTerm = true;
		SGD sgd        = new SGD(0);
		Matrix train_X = args.get("train_X"); 
		Matrix train_Y = args.get("train_Y");
		Matrix test_X  = args.get("test_X");
		Matrix test_Y  = args.get("test_Y");
		sgd.setIterationCount(1);
		
		sgd.setData(train_X, includeOffsetTerm, true);
		sgd.setDataClass(train_Y, true);
		
		sgd.setData(test_X, includeOffsetTerm, false);
		sgd.setDataClass(test_Y, false);
		
		sgd.learn();
		//sgd.batchLearn();
		
		double absErr = UtilLib.errorRate(sgd, test_Y);
		System.out.println("ABS ERR: " + absErr*100 + " %");
		return absErr;
	}
	


	public static void sgd_0_1_threads ( double [][] d, double split, int threads, int iters )
	{
		sgd_0_1_threads ( sgd.UtilLib.trainTestSplit(new Matrix(d), split), threads, iters );
	}
	
	public static double sgd_0_1_threads ( Map<String,Matrix> map, int threads, int iters )
	{    
		int threadCount     = threads;
		int iterations      = iters;
		boolean mixPerIter  = false;
		boolean mixPerPiece = false;
		
		Matrix train_X  = map.get("train_X");
	    Matrix test_X   = map.get("test_X");
		Matrix train_Y  = map.get("train_Y");
		Matrix test_Y   = map.get("test_Y");
		
		 //The +1 accounts for the intercept weight.
		Matrix iterWeights = new Matrix(1,train_X.getColumnDimension()+1,0).transpose();
		for ( int j = 0; j<iterations; j++ )
		{
		    if (mixPerIter)
		    {
                Matrix temp    =  sgd.UtilLib.matMergeXY(train_X, train_Y);
                train_X        =  sgd.UtilLib.getX(temp);
                train_Y        =  sgd.UtilLib.getY(temp);
		    }
		    SGD[] sgds     = new SGD[threadCount];
			//System.out.println("AAA: "+j);
			//mergeWeights.print(4, 4);
			int lastEnd = 0;
			for (int i = 0; i<threadCount; i++)
			{
				sgds[i] = new SGD (i);  
				sgds[i].setIterationCount(1);
				
				int newEnd = (int) ( lastEnd + Math.ceil(train_X.getRowDimension()/threadCount) );
				
				Matrix train_X_piece = train_X.getMatrix(lastEnd, Math.min(newEnd,train_X.getRowDimension()-1), 0, train_X.getColumnDimension()-1);
				Matrix train_Y_piece = train_Y.getMatrix(lastEnd, Math.min(newEnd,train_Y.getRowDimension()-1), 0, train_Y.getColumnDimension()-1);
				
				if(mixPerPiece)
				{
    				Matrix temp2  = sgd.UtilLib.matMergeXY(train_X_piece, train_Y_piece);
    				temp2         = sgd.UtilLib.randomRowShuffle(temp2);
    				train_X_piece = sgd.UtilLib.getX(temp2);
    				train_Y_piece = sgd.UtilLib.getY(temp2);
				}
				
				sgds[i].setData(train_X_piece, true, true);
				sgds[i].setDataClass(train_Y_piece, true);
				sgds[i].setWeights(iterWeights);
				
				lastEnd = newEnd + 1;
				sgds[i].start();
			}
			//The +1 accounts for the intercept weight.
			Matrix mergeWeights = new Matrix(1,train_X.getColumnDimension()+1,0).transpose();
			for (int i = 0; i<threadCount; i++)
			{
				try 
				{
					sgds[i].join();
					mergeWeights = mergeWeights.plus( (sgds[i].getW()) );
				} 
				catch (InterruptedException e) 
				{
					e.printStackTrace();
				};
			}
			mergeWeights = mergeWeights.times( 1.0 / ((double)threadCount) );
			iterWeights  = mergeWeights;
		}
	    //	mergeWeights = sgd.UtilLib.colEucNorm(mergeWeights);
		
		SGD merged   = new SGD();
		merged.setWeights(iterWeights);
		merged.setData(test_X, true, false);
		merged.setDataClass(test_Y, false);
		
		w_new = merged.getW().copy();
		//merged.W.print(4, 4);
		//System.out.println("ABS ERR: " + absErr*100 + " %");
		double absErr = sgd.UtilLib.errorRate(merged, test_Y);
		return absErr*100 ;
	}
	public static void irisSuite()
	{
		//double trainingSplit    = 1.2/2.0;
		int threads             = 1;
		int threadIter          = 1;
		long startTime			= 0;
		long endTime			= 0;
		double exeTime			= 0;
		ArrayList<Double> error = new ArrayList<Double> ();
		ArrayList<Double> times = new ArrayList<Double> ();
		
		Matrix lump = new Matrix((iris2));
		startTime = System.nanoTime();
		
		if (doRand)
		{
		    lump        = sgd.UtilLib.randomRowShuffle(lump);
		}
		Map<String, Matrix> map = sgd.UtilLib.trainTestSplit(lump,trainingSplit);
        sgd.UtilLib.splitXYs(map);
        map.put("train_X", sgd.UtilLib.standardize(map.get("train_X")));
        map.put("test_X" , sgd.UtilLib.standardize(map.get("test_X" )));
        sgd_0_1(map);
                
		endTime = System.nanoTime();
		exeTime = (double)(endTime-startTime) * Math.pow(10,-9);
		times.add(exeTime);
		startTime = System.nanoTime();
		sgd_0_1_threads(map, threads, threadIter);
		endTime = System.nanoTime();
		exeTime = (double)(endTime-startTime) * Math.pow(10,-9);
		times.add(exeTime);
	}
    public static void ausSuite()
    {
        //double trainingSplit    = 0.5;///2.0;
        
        Matrix lump = sgd.UtilLib.parseLibSvm("./tests/australian/australian_scale"  ,1,1);
        if (doRand)
        {
            lump        = sgd.UtilLib.randomRowShuffle(lump);
        }
        Map<String, Matrix> map = sgd.UtilLib.trainTestSplit(lump,trainingSplit);
        sgd.UtilLib.splitXYs(map);
        map.put("train_X", sgd.UtilLib.standardize(map.get("train_X")));
        map.put("test_X" , sgd.UtilLib.standardize(map.get("test_X" )));
        sgd_0_1(map);
    }
    public static void bcanSuite()
    {
        //double trainingSplit    = 0.5;///2.0;
        Matrix lump = sgd.UtilLib.parseLibSvmFast("./tests/breast-cancer/breast-cancer_scale_",683,10);
        if (doRand)
        {
            lump        = sgd.UtilLib.randomRowShuffle(lump);
        }
        Map<String, Matrix> map = sgd.UtilLib.trainTestSplit(lump,trainingSplit);
        sgd.UtilLib.splitXYs(map);
        map.put("train_X", sgd.UtilLib.standardize(map.get("train_X")));
        map.put("test_X" , sgd.UtilLib.standardize(map.get("test_X" )));
        sgd_0_1(map);
    }
    public static void ccanSuite()
    {
        //double trainingSplit    = 0.30;///2.0;
        
        Matrix lump = sgd.UtilLib.parseLibSvm("./tests/colon-cancer/colon-cancer",1,1);
        if (doRand)
        {
            lump        = sgd.UtilLib.randomRowShuffle(lump);
        }
        Map<String, Matrix> map = sgd.UtilLib.trainTestSplit(lump,trainingSplit);
        sgd.UtilLib.splitXYs(map);
        map.put("train_X", sgd.UtilLib.standardize(map.get("train_X")));
        map.put("test_X" , sgd.UtilLib.standardize(map.get("test_X" )));
        sgd_0_1(map);
    }
    public static void diabSuite()
    {
        //double trainingSplit    = 0.50;///2.0;
        
        Matrix lump = sgd.UtilLib.parseLibSvmFast("./tests/diabetes/diabetes_scale",768,8);
        if (doRand)
        {
            lump        = sgd.UtilLib.randomRowShuffle(lump);
        }
        Map<String, Matrix> map = sgd.UtilLib.trainTestSplit(lump,trainingSplit);
        sgd.UtilLib.splitXYs(map);
        map.put("train_X", sgd.UtilLib.standardize(map.get("train_X")));
        map.put("test_X" , sgd.UtilLib.standardize(map.get("test_X" )));
        sgd_0_1(map);
    }
    public static void dukeSuite()
    {
        //double trainingSplit    = 0.30;///2.0;
        
        Matrix lump = sgd.UtilLib.parseLibSvmFast("./tests/duke/duke",44,7129);
        if (doRand)
        {
            lump        = sgd.UtilLib.randomRowShuffle(lump);
        }
        Map<String, Matrix> map = sgd.UtilLib.trainTestSplit(lump,trainingSplit);
        sgd.UtilLib.splitXYs(map);
        map.put("train_X", sgd.UtilLib.standardize(map.get("train_X")));
        map.put("test_X" , sgd.UtilLib.standardize(map.get("test_X" )));
        sgd_0_1(map);
    }
	public static void a1aSuite()
	{
		//double trainingSplit    = 0.5;///2.0;
		int threads             = 1;
		int threadIter          = 1;
		long startTime			= 0;
		long endTime			= 0;
		double exeTime			= 0;
		ArrayList<Double> error = new ArrayList<Double> ();
		ArrayList<Double> times = new ArrayList<Double> ();
		
		Matrix a1a_train = sgd.UtilLib.parseLibSvm("./tests/a1a/a1a"  ,1,1);
		Matrix a1a_test  = sgd.UtilLib.parseLibSvm("./tests/a1a/a1a.t",1,1);
		Matrix lump  = sgd.UtilLib.matMergeX(a1a_test, a1a_train);

        if (doRand)
        {
            lump        = sgd.UtilLib.randomRowShuffle(lump);
        }
		Map<String, Matrix> map = sgd.UtilLib.trainTestSplit(lump,trainingSplit);
		sgd.UtilLib.splitXYs(map);
        map.put("train_X", sgd.UtilLib.standardize(map.get("train_X")));
        map.put("test_X" , sgd.UtilLib.standardize(map.get("test_X" )));
        
        sgd_0_1(map);
		//for ( threads = 1; threads<9; threads++)
		/*
		{
			for ( threadIter = 1; threadIter<=4; threadIter++)
			{
				System.out.print(".");
				startTime = System.nanoTime();
				error.add( sgd_0_1_threads(lump_rand, trainingSplit, threads, threadIter) );
				endTime = System.nanoTime();
				exeTime = (double)(endTime-startTime) * Math.pow(10,-9);
				times.add(exeTime);
			}
			System.out.println(threads + " threads. ");
			System.out.println(error);
			System.out.println(times);
			error.clear();
		}*/
	}
	
	public static void a2aSuite()
    {
        //double trainingSplit    = 0.5;///2.0;
        int threads             = 1;
        int threadIter          = 1;
        long startTime          = 0;
        long endTime            = 0;
        double exeTime          = 0;
        ArrayList<Double> error = new ArrayList<Double> ();
        ArrayList<Double> times = new ArrayList<Double> ();
        
        Matrix a2a_train = sgd.UtilLib.parseLibSvm("./tests/a2a/a2a"  ,1,1);
        Matrix a2a_test  = sgd.UtilLib.parseLibSvm("./tests/a2a/a2a.t",1,1);
        Matrix lump      = sgd.UtilLib.matMergeX(a2a_test, a2a_train);

        if (doRand)
        {
            lump        = sgd.UtilLib.randomRowShuffle(lump);
        }
        Map<String, Matrix> map = sgd.UtilLib.trainTestSplit(lump,trainingSplit);
        sgd.UtilLib.splitXYs(map);
        map.put("train_X", sgd.UtilLib.standardize(map.get("train_X")));
        map.put("test_X" , sgd.UtilLib.standardize(map.get("test_X" )));
        sgd_0_1(map);
        //for ( threads = 1; threads<9; threads++)
        /*
        {
            for ( threadIter = 1; threadIter<=4; threadIter++)
            {
                System.out.print(".");
                startTime = System.nanoTime();
                error.add( sgd_0_1_threads(lump_rand, trainingSplit, threads, threadIter) );
                endTime = System.nanoTime();
                exeTime = (double)(endTime-startTime) * Math.pow(10,-9);
                times.add(exeTime);
            }
            System.out.println(threads + " threads. ");
            System.out.println(error);
            System.out.println(times);
            error.clear();
        }*/
    }
	
	public static void a3aSuite()
    {
        //double trainingSplit    = 0.5;///2.0;
        int threads             = 1;
        int threadIter          = 1;
        long startTime          = 0;
        long endTime            = 0;
        double exeTime          = 0;
        ArrayList<Double> error = new ArrayList<Double> ();
        ArrayList<Double> times = new ArrayList<Double> ();
        
        Matrix train = sgd.UtilLib.parseLibSvm("./tests/a3a/a3a"  ,1,1);
        Matrix test  = sgd.UtilLib.parseLibSvm("./tests/a3a/a3a.t",1,1);
        Matrix lump  = sgd.UtilLib.matMergeX(test, train);

        if (doRand)
        {
            lump        = sgd.UtilLib.randomRowShuffle(lump);
        }
        Map<String, Matrix> map = sgd.UtilLib.trainTestSplit(lump,trainingSplit);
        sgd.UtilLib.splitXYs(map);
        map.put("train_X", sgd.UtilLib.standardize(map.get("train_X")));
        map.put("test_X" , sgd.UtilLib.standardize(map.get("test_X" )));
        sgd_0_1(map);
        //for ( threads = 1; threads<9; threads++)
        /*
        {
            for ( threadIter = 1; threadIter<=4; threadIter++)
            {
                System.out.print(".");
                startTime = System.nanoTime();
                error.add( sgd_0_1_threads(lump_rand, trainingSplit, threads, threadIter) );
                endTime = System.nanoTime();
                exeTime = (double)(endTime-startTime) * Math.pow(10,-9);
                times.add(exeTime);
            }
            System.out.println(threads + " threads. ");
            System.out.println(error);
            System.out.println(times);
            error.clear();
        }*/
    }
	public static void a4aSuite()
    {
        //double trainingSplit    = 0.5;///2.0;
        int threads             = 1;
        int threadIter          = 1;
        long startTime          = 0;
        long endTime            = 0;
        double exeTime          = 0;
        ArrayList<Double> error = new ArrayList<Double> ();
        ArrayList<Double> times = new ArrayList<Double> ();
        
        Matrix train = sgd.UtilLib.parseLibSvm("./tests/a4a/a4a"  ,1,1);
        Matrix test  = sgd.UtilLib.parseLibSvm("./tests/a4a/a4a.t",1,1);
        Matrix lump = sgd.UtilLib.matMergeX(test, train);

        if (doRand)
        {
            lump        = sgd.UtilLib.randomRowShuffle(lump);
        }
        Map<String, Matrix> map = sgd.UtilLib.trainTestSplit(lump,trainingSplit);
        sgd.UtilLib.splitXYs(map);
        map.put("train_X", sgd.UtilLib.standardize(map.get("train_X")));
        map.put("test_X" , sgd.UtilLib.standardize(map.get("test_X" )));
        sgd_0_1(map);
        //for ( threads = 1; threads<9; threads++)
        /*
        {
            for ( threadIter = 1; threadIter<=4; threadIter++)
            {
                System.out.print(".");
                startTime = System.nanoTime();
                error.add( sgd_0_1_threads(lump_rand, trainingSplit, threads, threadIter) );
                endTime = System.nanoTime();
                exeTime = (double)(endTime-startTime) * Math.pow(10,-9);
                times.add(exeTime);
            }
            System.out.println(threads + " threads. ");
            System.out.println(error);
            System.out.println(times);
            error.clear();
        }*/
    }
	static int run = 0;
	static Matrix w_old;
	static Matrix w_new;
	public static void codRnaSuite()
    {
        int threadMax            = -1;
        int threadIterMax        = -1;
        int threads              = 1;
        int threadIter           = 1;
        long startTime           = 0;
        long endTime             = 0;
        double exeTimeInSec      = 0;
        //double trainingSplit     = 1.2/2.0;
        
        ArrayList<Double>  error       = new ArrayList<Double>  ();
        ArrayList<Double>  times       = new ArrayList<Double>  ();
        ArrayList<Double>  convergence = new ArrayList<Double>  ();
        
        Matrix rna_r    = sgd.UtilLib.parseLibSvmFast("./tests/cod-rna/cod-rna.r",157413,8);
        Matrix rna_te   = sgd.UtilLib.parseLibSvmFast("./tests/cod-rna/cod-rna.t",271617,8);
        Matrix rna_tr   = sgd.UtilLib.parseLibSvmFast("./tests/cod-rna/cod-rna"  ,59535 ,8);
        
        Matrix lump = sgd.UtilLib.matMergeX(rna_tr, rna_te);
        lump        = sgd.UtilLib.matMergeX(lump, rna_r);
        if (doRand)
        {
            lump        = sgd.UtilLib.randomRowShuffle(lump);
        }
        
        //Map<String,Matrix> map = new HashMap<String,Matrix>();
        Map<String, Matrix> map = sgd.UtilLib.trainTestSplit(lump,trainingSplit);
        sgd.UtilLib.splitXYs(map);
        map.put("train_X", sgd.UtilLib.standardize(map.get("train_X")));
        map.put("test_X" , sgd.UtilLib.standardize(map.get("test_X" )));
        
        //map.put("train_X", sgd.UtilLib.standardize(sgd.UtilLib.readMat("./rna_train_X")));
        //map.put("test_X" , sgd.UtilLib.standardize(sgd.UtilLib.readMat("./rna_test_X")));
        //map.put("train_Y", sgd.UtilLib.readMat("./rna_train_Y"));
        //map.put("test_Y" , sgd.UtilLib.readMat("./rna_test_Y"));
        sgd_0_1(map);
         
        for ( threads = 1; threads<=threadMax; threads++ )
        {
            run++;
            for ( threadIter = 1; threadIter<=threadIterMax; threadIter++ )
            {
                System.out.print(" " + threadIter);
                startTime = System.nanoTime();
                    double err = sgd_0_1_threads(map, threads, threadIter);
                endTime = System.nanoTime();
                exeTimeInSec = (double)(endTime-startTime)*Math.pow(10,-9);
                times.add(exeTimeInSec);
                convergence.add(SGD.closeness(w_old, w_new));
                w_old = w_new.copy();
                //w_old.print(2, 2);
                error.add(err);
            }
            System.out.println();
            System.out.println(threads + " : threads. ");
            System.out.println(error);
            System.out.println(convergence);
            System.out.println(times);
            error.clear();
        }
    }
	
	/*
	 * The rows of the matrix are data points. 
	 * All but the last column are features.
	 * The last column of the matrix is a 0-1 class label.  
	 */
	public static void main(String[] args) 
	{
		/*
		 *Time of parallel sgd vs single thread sgd.
		 *Convergent rate.
		 *Test log likelihood for the prediction performance .
		 *Basically, we need to report classification accuracy(roc, auc or log likelihood ), 
		 *convergence rate, 
		 *speed up (running time by sequential/ by parallel)
		*/
	    
	    irisSuite();
	    bcanSuite();
	    ausSuite();
	    dukeSuite();	    
	    ccanSuite();
        diabSuite();
	    a1aSuite();
	    a2aSuite();
	    a3aSuite();
	    a4aSuite();
	    codRnaSuite();
	}
}
