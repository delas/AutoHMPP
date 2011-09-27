package it.processmining.autohmpp.thread;

import it.processmining.autohmpp.AutoHMPP;
import it.processmining.autohmpp.utils.Utils;
import it.processmining.hmpp.models.HMPPHeuristicsNet;
import it.processmining.hmpp.models.HMPPParameters;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import org.processmining.framework.log.LogReader;
import org.processmining.mining.geneticmining.fitness.duplicates.DTContinuousSemanticsFitness;

/**
 * This is the class with the thread used to search for the best parameters
 * configuration.
 * 
 * @author Andrea Burattin
 * @version 0.1
 */
public class ParameterSearchThread extends Thread {
	
	private AutoHMPP algorithm = null;
	private HMPPParameters parameters = null;
	private LogReader log = null;
	private int greatestNetworkSize;
	private double finalSolutionCost = Double.MAX_VALUE;
	private ArrayList<Double[]> discretizedParameters;
	private int steps = 0;
	
	/**
	 * Default thread serach constructor
	 * 
	 * @param name the name of this thread
	 * @param algorithm the Heuristics Miner++ algorithm instance
	 */
	public ParameterSearchThread(String name, AutoHMPP algorithm) {
		super(name);
		this.algorithm = algorithm;
		this.log = algorithm.getLogReader();
		this.greatestNetworkSize = algorithm.getGreatestNetworkSize();
		this.parameters = new HMPPParameters();
	}
	
	
	@Override
	public void run() {

		Random randomGenerator = new Random();
		ArrayList<Double> costs = new ArrayList<Double>();
		discretizedParameters = algorithm.getDiscretizedParameters();
		/*
		 * ArrayList with all the discretized values for each parameters. Each
		 * element is an array of Double containing the possible values for the
		 * parameter. This is the map from the index to the parameter
		 * discratization:
		 *  0 - dependency threshold
		 *  1 - positive observations
		 *  2 - relative to best
		 *  3 - and threshold
		 *  4 - length one loop
		 *  5 - length two loop
		 *  6 - long distance dep
		 */
		/* dependency thresholds */
		int index_dt = randomGenerator.nextInt(discretizedParameters.get(0).length);
		/* positive observations */
		int index_po = randomGenerator.nextInt(discretizedParameters.get(1).length);
		/* relative to best */
		int index_rb = randomGenerator.nextInt(discretizedParameters.get(2).length);
		/* AND threshold */
		int index_and = randomGenerator.nextInt(discretizedParameters.get(3).length);
		/* length one/two loops */
		int index_l1l = randomGenerator.nextInt(discretizedParameters.get(4).length);
		int index_l2l = randomGenerator.nextInt(discretizedParameters.get(5).length);
		/* long distance dep */
		int index_ldd = randomGenerator.nextInt(discretizedParameters.get(6).length);
		
		/*
		 * We are moving in a 7 variables space, example:
		 *
		 *         A
		 *   .     | 
		 *    `.   |
		 *      `. |
		 *  -------+-------- B
		 *         |`.
		 *         |  `.
		 *         |    ` C
		 * 
		 * As example assume:
		 *   A as the dependency threshold
		 *   B as the positive observations
		 *   C as the relative to best
		 *  
		 */

		/*               .----------- first cluster -----------.              */
		int indexes[] = {index_dt, index_po, index_rb, index_and,
		/*      .------ second cluster -------.                               */
				index_l1l, index_l2l, index_ldd};
		
		int jump = 1;
		int plateauStepLeft = algorithm.getMaxPlateauStep();
		
		double currentCost;
		double newCost;
		
		int stepsFirstCluster = 1;
		/* =============== CHECKING PARAMETERS IN FIRST CLUSTER ================
		 * 
		 * In this first block we are searching for the best parameters
		 * configuration looking only at the related parameters:
		 *  - dependency threshold
		 *  - positive observation threshold
		 *  - relative to best
		 *  - and threshold
		 */
		do {
			if (validateArrayIndexes(indexes, discretizedParameters)) {
				currentCost = getMinedNetworkCost(indexes, discretizedParameters, false, true, false);
			} else {
				currentCost = Double.MAX_VALUE;
			}
			newCost = currentCost;
			int newDirection = -1;
			dbg("current position cost = " + currentCost);
			
			/* Variations in the indexes array:
			 *
			 *                  ,-- dependency threshold
			 *                  |  ,-- positive observations
			 *                  |  |  ,-- relative to best
			 *                  |  |  |  ,-- and threshold
			 *                  |  |  |  |  ,-- l1l
			 *                  |  |  |  |  |  ,-- l2l
			 *                  |  |  |  |  |  |  ,-- long distance dep
			 *                  v  v  v  v  v  v  v
			 *         indexes: 0  1  2  3  4  5  6                           */
			int variations[] = {0, 0, 0, 0, 0, 0, 0};
			
			/* dependency threshold */
			for (variations[0] = -jump; variations[0] <= jump; variations[0]++) {
				/* positive observations */
				for (variations[1] = -jump; variations[1] <= jump; variations[1]++) {
					/* relative to best */
					for (variations[2] = -jump; variations[2] <= jump; variations[2]++) {
						/* and threshold */
						for (variations[3] = -jump; variations[3] <= jump; variations[3]++) {
							
							/* id of the possible new direction */
							int direction =	
								(variations[0] + jump + 1) * 1 + 
								(variations[1] + jump + 1) * 10 + 
								(variations[2] + jump + 1) * 100 +
								(variations[3] + jump + 1) * 1000;
							/* this index, to check we are not going back 
							 * into the coming direction... */
							int fromDirection = 0;
							for (int i = 0; i < 4; i++) {
								int multiplier = (i > 0)? 10*i : 1;
								if (variations[i] == 1) {
									fromDirection += 1 * multiplier;
								}
								if (variations[i] == -1) {
									fromDirection += 3 * multiplier;
								}
							}
							
							if (direction != fromDirection) {
								double possibleCost = Double.MAX_VALUE;
								String log = "";
								if (validateArrayIndexes(indexes, variations, discretizedParameters)) {
									possibleCost = getMinedNetworkCost(indexes, variations, discretizedParameters, false, true, false);
									log = "direction = " + direction + " ; cost = " + possibleCost;
								}
								if (possibleCost <= newCost) {
									log += " -- best";
									newCost = possibleCost;
									newDirection = direction;
								}
								if (!log.equals("")) {
									dbg(log);
								}
							}
						}
					}
				}
			}
			dbg("new hip cost = " + newCost);
			dbg("new direction = " + newDirection);
			
			if (newDirection > 0) {
				int newPossibleDirection;
				/* there is a best place to move! */
				for (variations[0] = -jump; variations[0] <= jump; variations[0]++) {
					for (variations[1] = -jump; variations[1] <= jump; variations[1]++) {
						for (variations[2] = -jump; variations[2] <= jump; variations[2]++) {
							for (variations[3] = -jump; variations[3] <= jump; variations[3]++) {

								newPossibleDirection = 
									(variations[0] + jump + 1) * 1 + 
									(variations[1] + jump + 1) * 10 + 
									(variations[2] + jump + 1) * 100 +
									(variations[3] + jump + 1) * 1000;

								if (newPossibleDirection == newDirection) {
									for(int i = 0; i < 4; i++) {
										indexes[i] += variations[i];
									}
								}

							}
						}
					}
				}
				if (currentCost == newCost) {
					plateauStepLeft--;
					dbg("still in plateau [" + plateauStepLeft + "]");
				} else {
					plateauStepLeft = algorithm.getMaxPlateauStep();
					stepsFirstCluster++;
					dbg("this is new hipothesis");
				}
			}
			costs.add(newCost);
			
		} while (newCost <= currentCost && plateauStepLeft > 0);
		
		double bestCostBeforeLoop = currentCost;
		HMPPParameters parametersBeforeLoop = new HMPPParameters();
		parametersBeforeLoop.setDependencyThreshold(discretizedParameters.get(0)[indexes[0]]);
		parametersBeforeLoop.setPositiveObservationsThreshold(discretizedParameters.get(1)[indexes[1]].intValue());
		parametersBeforeLoop.setRelativeToBestThreshold(discretizedParameters.get(2)[indexes[2]]);
		parametersBeforeLoop.setAndThreshold(discretizedParameters.get(3)[indexes[3]]);
		parametersBeforeLoop.setL1lThreshold(0.0);
		parametersBeforeLoop.setL2lThreshold(0.0);
		parametersBeforeLoop.setLDThreshold(0.0);
		parametersBeforeLoop.useAllConnectedHeuristics = true;
		parametersBeforeLoop.useLongDistanceDependency = false;
		
		dbg("\n\n=========== CHECKING PARAMETERS IN SECOND CLUSTER ===========\n");
		int stepsSecondCluster = 1;
		/* =============== CHECKING PARAMETERS IN SECOND CLUSTER ===============
		 * 
		 * Now we have a potentially sub-optimal solution, without considering
		 * the others parameters that are:
		 *  - length 1 loop
		 *  - length 2 loop
		 *  - long distance dependency
		 * We now try to extend our solution including these ones and, if we
		 * obtain a better hypothesis, than we can continue.
		 */
		plateauStepLeft = algorithm.getMaxPlateauStep();
		
		do {
			if (validateArrayIndexes(indexes, discretizedParameters)) {
				currentCost = getMinedNetworkCost(indexes, discretizedParameters, true, true, true);
			} else {
				currentCost = Double.MAX_VALUE;
			}
			newCost = currentCost;
			int newDirection = -1;
			dbg("current position cost = " + currentCost);
			
			/* Variations in the indexes array:
			 *
			 *                  ,-- dependency threshold
			 *                  |  ,-- positive observations
			 *                  |  |  ,-- relative to best
			 *                  |  |  |  ,-- and threshold
			 *                  |  |  |  |  ,-- l1l
			 *                  |  |  |  |  |  ,-- l2l
			 *                  |  |  |  |  |  |  ,-- long distance dep
			 *                  v  v  v  v  v  v  v
			 *         indexes: 0  1  2  3  4  5  6                           */
			int variations[] = {0, 0, 0, 0, 0, 0, 0};
			
			/* length 1 loop */
			for (variations[4] = -jump; variations[4] <= jump; variations[4]++) {
				/* length 2 loop */
				for (variations[5] = -jump; variations[5] <= jump; variations[5]++) {
					/* long distance dependency */
					for (variations[6] = -jump; variations[6] <= jump; variations[6]++) {
						
						/* id of the possible new direction */
						int direction =	
							(variations[4] + jump + 1) * 1 + 
							(variations[5] + jump + 1) * 10 + 
							(variations[6] + jump + 1) * 100;
						/* this index, to check we are not going back 
						 * into the coming direction... */
						int fromDirection = 0;
						for (int i = 4; i < 7; i++) {
							int multiplier = (i > 0)? 10*i : 1;
							if (variations[i] == 1) {
								fromDirection += 1 * multiplier;
							}
							if (variations[i] == -1) {
								fromDirection += 3 * multiplier;
							}
						}
						
						if (direction != fromDirection) {
							double possibleCost = Double.MAX_VALUE;
							String log = "";
							if (validateArrayIndexes(indexes, variations, discretizedParameters)) {
								possibleCost = getMinedNetworkCost(indexes, variations, discretizedParameters, true, true, true);
								log = "direction = " + direction + " ; cost = " + possibleCost;
							}
							if (possibleCost <= newCost) {
								log += " -- best";
								newCost = possibleCost;
								newDirection = direction;
							}
							if (!log.equals("")) {
								dbg(log);
							}
						}
					}
				}
			}
			dbg("new hip cost = " + newCost);
			dbg("new direction = " + newDirection);
			
			if (newDirection > 0) {
				int newPossibleDirection;
				/* there is a best place to move! */
				for (variations[4] = -jump; variations[4] <= jump; variations[4]++) {
					for (variations[5] = -jump; variations[5] <= jump; variations[5]++) {
						for (variations[6] = -jump; variations[6] <= jump; variations[6]++) {
							
							newPossibleDirection = 
								(variations[4] + jump + 1) * 1 + 
								(variations[5] + jump + 1) * 10 + 
								(variations[6] + jump + 1) * 100;

							if (newPossibleDirection == newDirection) {
								for(int i = 4; i < 7; i++) {
									indexes[i] += variations[i];
								}
							}
							
						}
					}
				}
				if (currentCost == newCost) {
					plateauStepLeft--;
					dbg("still in plateau [" + plateauStepLeft + "]");
				} else {
					plateauStepLeft = algorithm.getMaxPlateauStep();
					stepsSecondCluster++;
					dbg("not in plateau");
				}
			}
			costs.add(newCost);
			
		} while (newCost <= currentCost && plateauStepLeft > 0);
		
		double bestCostAfterLoop = currentCost;
		HMPPParameters parametersAfterLoop = new HMPPParameters();
		parametersAfterLoop.setDependencyThreshold(discretizedParameters.get(0)[indexes[0]]);
		parametersAfterLoop.setPositiveObservationsThreshold(discretizedParameters.get(1)[indexes[1]].intValue());
		parametersAfterLoop.setRelativeToBestThreshold(discretizedParameters.get(2)[indexes[2]]);
		parametersAfterLoop.setAndThreshold(discretizedParameters.get(3)[indexes[3]]);
		parametersAfterLoop.setL1lThreshold(discretizedParameters.get(4)[indexes[4]]);
		parametersAfterLoop.setL2lThreshold(discretizedParameters.get(5)[indexes[5]]);
		parametersAfterLoop.setLDThreshold(discretizedParameters.get(6)[indexes[6]]);
		parametersAfterLoop.useAllConnectedHeuristics = true;
		parametersAfterLoop.useLongDistanceDependency = true;
		
		
		/*
		 * ============================ CONCLUSIONS ============================
		 */
		/* is better the network with or without loops and long distance? */
		if (bestCostAfterLoop < bestCostBeforeLoop) {
			/* network with loop */
			parameters = parametersAfterLoop;
			finalSolutionCost = bestCostAfterLoop;
			steps = stepsFirstCluster + stepsSecondCluster;
			dbg("best network is with loops and long distance");
		} else {
			/* network without loop */
			parameters = parametersBeforeLoop;
			finalSolutionCost = bestCostBeforeLoop;
			steps = stepsFirstCluster;
			dbg("best network without loops");
		}
		
		HMPPHeuristicsNet n = algorithm.makeHeuristicsRelations(log, parameters);
		dbg("tested miner size: " + Utils.calculateNetworkSize(n) + " ; hash: " + n.hashCode());

		dbg("");
		algorithm.addParametersCosts(getName(), costs);
	}
	
	
	/**
	 * This method checks if the new indexes summed with the relative variations
	 * are correct indexes for the correspondent items. In other words, it must
	 * happen, for each elements i, that:
	 * 
	 *   (indexes[i] + variations[i] &gt;= 0 && 
	 *      indexes[i] + variations[i] &lt; items.get(i).length)
	 * 
	 * @param indexes the original indexes
	 * @param variations the variations to each indexes (summed to the index)
	 * @param items the array elements
	 * @return true if, for all the items, the indexes are correct, false
	 * otherwise
	 */
	private boolean validateArrayIndexes(int indexes[], int variations[], ArrayList<Double[]> items) {
		boolean toReturn = true;
		int max = indexes.length;
		if (max < variations.length) {
			max = variations.length;
		}
		for (int i = 0; i < max; i++) {
			toReturn = toReturn && 
				(indexes[i] + variations[i] >= 0 && 
						indexes[i] + variations[i] < items.get(i).length);
		}
		return toReturn;
	}
	
	
	/**
	 * This method checks if the indexes are correct indexes for the 
	 * correspondent items. In other words, it must happen, for each elements i, 
	 * that:
	 * 
	 *   (indexes[i] &gt;= 0 && indexes[i] &lt; items.get(i).length)
	 * 
	 * @param indexes the original indexes
	 * @param items the array elements
	 * @return true if, for all the items, the indexes are correct, false
	 * otherwise
	 */
	private boolean validateArrayIndexes(int indexes[], ArrayList<Double[]> items) {
		int variations[] = new int[indexes.length];
		Arrays.fill(variations, 0);
		return validateArrayIndexes(indexes, variations, items);
	}
	
	
	/**
	 * This method mines and returns the cost of the hypothesis built with the
	 * given parameters.
	 * 
	 * The returned value is build considering the MDL approach and is
	 * calculated as follows:
	 * 
	 *    L(h) + L(D|h)
	 * 
	 * where L(h) is the "size" of the network h and L(D|h) is the fitness of
	 * the not processed observations.
	 * 
	 * @param p the parameters object instance
	 * @return the hypothesis cost
	 */
	private Double getMinedNetworkCost(HMPPParameters p) {
		dbg("getMinedNetworkCost {");
		Double[] data = {0., 0.};
		Double networkHypCost = 0.;
		
		HMPPHeuristicsNet result = algorithm.makeHeuristicsRelations(log, p);
		data[0] = new Double(Utils.calculateNetworkSize(result));
		
		HMPPHeuristicsNet[] population = new HMPPHeuristicsNet[1];
		population[0] = result;
		
		DTContinuousSemanticsFitness fitnessContinuousSemantics = new DTContinuousSemanticsFitness(log);
		fitnessContinuousSemantics.calculate(population);
		Double fitness = population[0].getFitness();
		
		data[1] = fitness;
		networkHypCost = (data[0] / greatestNetworkSize) + (1 - data[1]);
		
		dbg("} getMinedNetworkCost");
		return networkHypCost;
	}
	
	
	/**
	 * Shortcut for the same method, with the parameters as array
	 * 
	 * @param indexes indexes to use for the call
	 * @param variations values to be summed to each index before the call
	 * @param discretizedParameters parameter values
	 * @param useLongDistanceDependency
	 * @param useAllConnectedHeuristics
	 * @param useLoops
	 * @return the hypothesis cost 
	 */
	private Double getMinedNetworkCost(
			int[] indexes, 
			int[] variations, 
			ArrayList<Double[]> discretizedParameters, 
			boolean useLongDistanceDependency,
			boolean useAllConnectedHeuristics,
			boolean useLoops) {
		
		/*
		 *							| discr param	| indexes	| variations
		 *--------------------------+---------------+-----------+-----------
		 * dependency thresholds	| 0				| 0			| 0
		 * positive observations	| 1				| 1			| 1
		 * relative to best			| 2				| 2			| 2
		 * AND threshold			| 3				| 3			| 3
		 * length 1 loops			| 4				| 4			| 4
		 * length 2 loops			| 5				| 5			| 5
		 * long distance dep		| 6				| 6			| 6
		 */
		
		Double l1loopThreshold = (useLoops)? discretizedParameters.get(4)[indexes[4] + variations[4]] : 0.0;
		Double l2loopThreshold = (useLoops)? discretizedParameters.get(5)[indexes[5] + variations[5]] : 0.0;
		Double ldThreshold = (useLongDistanceDependency)? discretizedParameters.get(6)[indexes[6] + variations[6]] : 0.0;
		
		HMPPParameters p = getParameters();
		p.setDependencyThreshold(discretizedParameters.get(0)[indexes[0] + variations[0]]);
		p.setPositiveObservationsThreshold(discretizedParameters.get(1)[indexes[1] + variations[1]].intValue());
		p.setRelativeToBestThreshold(discretizedParameters.get(2)[indexes[2] + variations[2]]);
		p.setAndThreshold(discretizedParameters.get(3)[indexes[3] + variations[3]]);
		p.setL1lThreshold(l1loopThreshold);
		p.setL2lThreshold(l2loopThreshold);
		p.setLDThreshold(ldThreshold);
		p.setUseLongDistanceDependency(useLongDistanceDependency);
		p.setUseAllConnectedHeuristics(useAllConnectedHeuristics);
		
		return getMinedNetworkCost(p);
	}
	
	
	/**
	 * Shortcut for the same method, with the parameters as array
	 * 
	 * @param indexes indexes to use for the call
	 * @param discretizedParameters parameter values
	 * @param useLongDistanceDependency
	 * @param useAllConnectedHeuristics
	 * @param useLoops
	 * @return the hypothesis cost 
	 */
	private Double getMinedNetworkCost(
			int[] indexes,  
			ArrayList<Double[]> discretizedParameters, 
			boolean useLongDistanceDependency,
			boolean useAllConnectedHeuristics,
			boolean useLoops) {
		
		int variations[] = new int[indexes.length];
		Arrays.fill(variations, 0);
		return getMinedNetworkCost(indexes, variations, discretizedParameters, 
				useLongDistanceDependency, useAllConnectedHeuristics, useLoops);
	}
	
	
	/**
	 * This method to get the set of found parameters
	 * 
	 * @return get the set of found parameters 
	 */
	public HMPPParameters getParameters() {
		return parameters;
	}
	
	
	/**
	 * This method to get the cost of the final solution
	 * 
	 * @return the final solution cost
	 */
	public Double getFinalCost() {
		return finalSolutionCost;
	}
	
	
	/**
	 * This method returns the number of step of the procedure
	 * 
	 * @return the number of step
	 */
	public int getExecutionStep() {
		return steps;
	}
	
	
	/**
	 * This method prints can be used to print some debug information.
	 * 
	 * @param msg the message text
	 */
	private void dbg(String msg) {
		algorithm.dbg("Thread " + getName() + " -- " + msg);
	}
}
