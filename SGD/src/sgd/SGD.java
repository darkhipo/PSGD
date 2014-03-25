package sgd;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.Random;

import Jama.*;
import sgd.UtilLib;

/*
 * Use: 
 * 
 * 		sgd.setData(train_X, true, true);
 *		sgd.setDataClass(train_Y, true);
 *	    sgd.setData(test_X, true, false);
 *		sgd.setDataClass(test_Y, false);
 *		sgd.learn();
 *		Matrix Y_hat = sgd.predict();
 * 
 */
public class SGD extends Thread
{
	static final int START_OFFSET= 0;
	static final int AVG_OVER    = 5000;
	volatile Matrix X;           //Train Data
	volatile Matrix Y;           //Train Data Class
	volatile Matrix X_t;         //Test Data
	volatile Matrix Y_t;         //Test Data Class
	volatile Matrix W; 	         //Assigned weights. Synonym: theta
	volatile double lr           = 0.200;//0.16000;//0.240;//0.16;//0.0001;//0.20000;//1;//0.8;//0.005;   //Learning rate.
	volatile double regulParam   = 0.005;//0.00008;//0.028;//0.5 
	volatile int iterationCount  = 1;
	volatile int threadId        = 0;
	volatile boolean learnMode   = true;
	
	volatile boolean normalize1  = false;
	volatile boolean normalize2  = true;
	volatile boolean adaptLR     = true;
	volatile boolean randBoot    = true;
    volatile boolean randPerIter = false;
	
	//0: +(regul*lr)*W
	//1: +(regul*lr)*Math.abs(W)
	//2: +(regul*lr)*W^2
    //3: +0.5 \lambda * w' w + log loss function
	//default: no regularization.
	volatile int regularizationOption = 1;
	
	volatile ArrayList<Double> logLike;
	volatile ArrayList<Double> closeNess;
	volatile ArrayList<Double> convergence;
	
	SGD ( )
	{
		logLike = new ArrayList<Double>();
	}
	
	SGD( int id )
	{ 
		this.threadId    = id;
		this.logLike     = new ArrayList<Double>();
		this.closeNess   = new ArrayList<Double>();
	    this.convergence = new ArrayList<Double>();
	}
	
	public double lossFunction(Matrix prediction, Matrix truth)
	{
	    Matrix err = truth.minus(prediction);
	    return err.transpose().times(err).norm1();
	}
	public double logLoss(Matrix prediction, Matrix truth)
    {
        return Math.log(lossFunction(prediction, truth));
    }
	public Matrix getY_t() 
	{
		return Y_t;
	}
	
	public Matrix getW() 
	{
		return W;
	}
	
	public ArrayList<Double> getll()
	{
		return this.logLike;
	}
	
	public ArrayList<Double> getCloseness() 
	{
	    return closeNess;
	}

	public boolean setLearnMode(boolean mode)
	{
		return this.learnMode = mode;
	}
	
	public int setIterationCount(int i)
	{
		return this.iterationCount = i;
	}
	
	public Matrix setWeights(double [][] w)
	{
		return this.W = new Matrix (w);
	}
	
	 public Matrix setWeights(Matrix w)
	{
		return this.W = (w);
	}
	
	public Matrix setDataClass(double [][] y, boolean isTrain)
	{
		Matrix mat = new Matrix (y);
		if (isTrain) 
		{ 
			this.Y = mat;
		}
		else
		{
			this.Y_t = new Matrix(mat.getRowDimension(),mat.getColumnDimension(),0);
		}
		return mat;
	}
	
	public Matrix setDataClass(Matrix y, boolean isTrain)
	{
		Matrix mat = y;
		if (isTrain) 
		{ 
			this.Y = mat;
		}
		else
		{
			this.Y_t = new Matrix(mat.getRowDimension(),mat.getColumnDimension(),0);
		}
		return mat;
	}
	
	public int setData(double [][] x, boolean includeOffset, boolean isTrain)
	{
		Matrix matToSet = isTrain ? this.X : this.X_t;
		matToSet        = new Matrix (x);
		
		int featureCount= matToSet.getColumnDimension();
		int datumCount  = matToSet.getRowDimension();
		
		if(includeOffset)
		{
			Matrix X2    = new Matrix(datumCount, featureCount+1, 1.0);
			X2.setMatrix(START_OFFSET, datumCount-1, 1, featureCount, matToSet);
			featureCount++;
			matToSet = X2;
		}
		if (isTrain)
		{
			this.X = matToSet;
		}
		else
		{
			this.X_t = matToSet;
		}
		return featureCount;
	}
	
	public int setData(Matrix x, boolean includeOffset, boolean isTrain)
	{
		Matrix matToSet = isTrain ? this.X : this.X_t;
		matToSet = (x);

		int featureCount = matToSet.getColumnDimension();
		int datumCount   = matToSet.getRowDimension();
		
		if(includeOffset)
		{
			Matrix X2    = new Matrix(datumCount, featureCount+1, 1.0);
			X2.setMatrix(START_OFFSET, datumCount-1, 1, featureCount, matToSet);
			featureCount++;
			matToSet = X2;
		}
		if (isTrain)
		{
			this.X = matToSet;
		}
		else
		{
			this.X_t = matToSet;
		}
		return featureCount;
	}
	
	public void run()
	{
		if( this.learnMode )
		{
			learn();
		}
		else
		{
			predict();
		}
	}
	
	public double sigmoid(double z)
	{
	    //return 1.0/(1.0 + Math.pow(Math.E, -z));
		return (z > 0) ? (1.0/(1.0 + Math.pow(Math.E, -z))) : 1.0-(1.0/(1.0 + Math.pow(Math.E, z)));
	}
	
	public double logLikelihood(double y_i, double y_i_hat)
	{
	    //System.out.println( Math.log(y_i_hat) + " " + Math.log(1.0-y_i_hat) );
	    return y_i > 0 ? Math.log(y_i_hat) : Math.log(1.0-y_i_hat);
	    //return Math.max( Math.log(y_i_hat), Math.log(1.0-y_i_hat) );
	}
	
	public static  Double closeness(Matrix w_old, Matrix w_new)
    {
	    if(w_old == null)
	    {
	        w_old = new Matrix(w_new.getRowDimension(),w_new.getColumnDimension(),0);
	    }
	    
	    //return sgd.UtilLib.colAbsDiff(w_old, w_new);
	    Matrix temp  = w_new.minus(w_old);
	    return Math.sqrt( ( temp.transpose().times(temp) ).get(0, 0) );
    }
    
	public  Double updateLr(double currLr, double k)
	{
	    //System.out.println((X.getRowDimension()-k) / X.getRowDimension());

	    //return ( currLr / Math.pow(k, 2) );
	    //return ( currLr / k ) ;
	    //return ( currLr / Math.sqrt(k) );
	    if(adaptLR)
	    {
	        return currLr - currLr*( 1.0 / X.getRowDimension() );
	    }
	    else
	    {
	        return ( currLr ) ;
	    }
	        //return 1.0;
	    //return 0.0 ;
	    //
	    //return currLr*( (double)( X.getRowDimension()+1-k) / (double) (X.getRowDimension()+1) );
	}
	public Matrix batchLearn()
	{
	    int batchSize = this.X.getRowDimension();
	    if( this.W == null)
        {
            this.W = new Matrix(this.X.getColumnDimension(),1,0);
            Random rand = new Random();
            if (randBoot)
            {
                for(int i=0; i<this.W.getColumnDimension(); i++)
                {
                    this.W.set(i,START_OFFSET,rand.nextDouble());
                }
                this.W = sgd.UtilLib.colEucNorm(this.W);
            }
        }
	    iterationCount = 50;
	    for (int j=0; j<this.iterationCount; j++)
	    {
    	    Matrix update = new Matrix(this.W.getRowDimension(),this.W.getColumnDimension(),0);
    	    for (int i=0; i< batchSize; i++)
    	    {
    	        Matrix datum         = this.X.getMatrix(i,i,START_OFFSET,X.getColumnDimension()-1);
                double prediction    = datum.times(this.W).get(START_OFFSET,START_OFFSET);
                prediction           = this.sigmoid(prediction);
                double errorResidual = prediction-this.Y.get(i,START_OFFSET);
                double coef          = errorResidual; 
                Matrix gradient      = datum.times(coef).transpose();
                update               = update.plus(gradient);
    	    }
    	    update = update.times( this.lr / ( (double)batchSize ) );
    	    this.W = this.W.minus(update);
	    }
    	return this.W;
	}
	public  Matrix learn()
	{
		//If weights were not set, initialize them to 0s.
		if( this.W == null)
		{
			this.W = new Matrix(this.X.getColumnDimension(),1,0);
			Random rand = new Random();
			if (randBoot)
			{
    			for(int i=0; i<this.W.getColumnDimension(); i++)
    			{
    			    this.W.set(i,START_OFFSET,rand.nextDouble());
    			}
    			this.W = sgd.UtilLib.colEucNorm(this.W);
			}
		}
		for (int j=0; j<this.iterationCount; j++)
		{
		    if (randPerIter)
		    {
    		    Matrix temp = sgd.UtilLib.matMergeXY(this.X, this.Y);
    		    temp        = sgd.UtilLib.randomRowShuffle(temp);
    		    this.X      = sgd.UtilLib.getX(temp);
    		    this.Y      = sgd.UtilLib.getY(temp);
		    }
			for (int i=0; i<X.getRowDimension(); i++)
			{
			    double convAcc       = 0.0;
			    double llAcc         = 0.0;
	            double clAcc         = 0.0;
			    Matrix w_old         = (Matrix) this.W.clone();
			    
				Matrix datum         = this.X.getMatrix(i,i,START_OFFSET,X.getColumnDimension()-1);
				double prediction    = datum.times(this.W).get(START_OFFSET,START_OFFSET);
				prediction           = this.sigmoid(prediction);
				double errorResidual = prediction-this.Y.get(i,START_OFFSET);
                Double logLossTerm   = Math.log(Math.pow(errorResidual,2));
                logLossTerm          = ( logLossTerm.isInfinite() || logLossTerm.isNaN() ) ?  (0) : (logLossTerm) ;
				double coef          = errorResidual; 
				Matrix gradient      = datum.times(coef).transpose(); //Neg
				Matrix step          = gradient.times(this.lr);
				Matrix regul         = new Matrix(this.W.getRowDimension(),this.W.getColumnDimension(),0);
				
				if(this.regularizationOption == 0) //By weight regularization.
				{
				    regul = this.W.times(this.lr*this.regulParam);
				}
				else if(this.regularizationOption == 1) //Lasso regularization.
				{
				    regul = sgd.UtilLib.colAbs(this.W.times(this.lr*this.regulParam));
				}
				else if (this.regularizationOption == 2)//Squared regularization
				{
				    regul = (this.W.arrayTimes(this.W)).times(this.lr*this.regulParam);
				}
				else if (this.regularizationOption == 3)
				{
				    regul = new Matrix(this.W.getRowDimension(),this.W.getColumnDimension(),this.W.transpose().times(this.W).times(0.5*this.lr*this.regulParam).get(0,0) + logLossTerm.doubleValue()); 
				}
				//W.print(2,2);
                this.lr              = updateLr(this.lr,i);
				//Update the weights.
				this.W               = this.W.minus(step);  //Update the weights.
				this.W               = this.W.plus(regul);  //Apply regularization.
		        if (normalize1)
		        {
		            this.W = sgd.UtilLib.colEucNorm(this.W);
		        }
				//Collect statistics.
				convAcc += 0.5*Math.pow((prediction-this.Y.get(i,START_OFFSET)),2);
				llAcc   += logLikelihood(this.Y.get(i,START_OFFSET),prediction);
				clAcc   += closeness(w_old, this.W);
				if (i%AVG_OVER == 0)
				{
				    convAcc /= (1.0*AVG_OVER);
				    llAcc   /= (1.0*AVG_OVER);
				    clAcc   /= (1.0*AVG_OVER);
				    convergence.add(convAcc);
				    closeNess.add(clAcc);
				    logLike.add(llAcc);
				    //this.closeNess.add( closeness(w_old, this.W) );
	                //this.logLike.add  ( logLikelihood(this.Y.get(i,START_OFFSET),prediction) );
				    convAcc = 0.0;
				    llAcc   = 0.0;
				    clAcc   = 0.0;
				}
				//this.W.transpose().print(4, 4);
				//System.out.println(closeness(w_old, this.W));
			}
		}
		//writeLog();
		//System.out.println("Done " +this.lr+ " "+ threadId);
		if (normalize2)
		{
		    this.W = sgd.UtilLib.colEucNorm(this.W);
		}
        //System.out.println("C="+convergence);
        //System.out.println("CL="+closeNess);
        //System.out.println("LL="+logLike);
		return this.W;
	}
	
	public Matrix predict()
	{
		for ( int i=0; i<this.X_t.getRowDimension(); i++ )
		{
			Matrix datum      = X_t.getMatrix(i,i,START_OFFSET,this.X_t.getColumnDimension()-1);
			double prediction = datum.times(this.W).get(0,0);
			prediction        = this.sigmoid(prediction);
			this.Y_t.set(i, START_OFFSET, prediction > 0.5 ? 1 : 0);
		}
		return this.Y_t;
	}
	
	public void writeLog()
	{
	    try
	    {
	        String timeStamp = new SimpleDateFormat("yyyy:MM:dd:HH:mm:ss").format(Calendar.getInstance().getTime());
	        File file = new File("./logs/"+timeStamp+".m");
	        FileWriter writer = new FileWriter(file);
	        PrintWriter printWriter = new PrintWriter(writer);
	        printWriter.print("LL=[");
	        for (Double s : logLike)
	        {
	            printWriter.print(s + " ");
	        }
            printWriter.println("];");
            printWriter.print("CL=[");
            for (Double s : closeNess)
            {
                printWriter.print(s + " ");
            }
            printWriter.println("];");
	        
	        printWriter.close();
	    }
	    catch (IOException e)
	    {
	        e.printStackTrace();
	    }
	}
}
