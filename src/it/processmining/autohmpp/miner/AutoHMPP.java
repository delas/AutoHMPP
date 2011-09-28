package it.processmining.autohmpp.miner;

import it.processmining.autohmpp.miner.notifier.Notifier;
import it.processmining.autohmpp.utils.Utils;
import it.processmining.hmpp.HMPP;
import it.processmining.hmpp.HMPPResult;
import it.processmining.hmpp.models.HMPPHeuristicsNet;
import it.processmining.hmpp.models.HMPPParameters;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;

import javax.swing.JPanel;

import org.processmining.framework.log.AuditTrailEntry;
import org.processmining.framework.log.AuditTrailEntryList;
import org.processmining.framework.log.LogEvent;
import org.processmining.framework.log.LogEvents;
import org.processmining.framework.log.LogReader;
import org.processmining.framework.log.LogSummary;
import org.processmining.framework.log.ProcessInstance;
import org.processmining.framework.models.heuristics.HNSet;
import org.processmining.framework.models.heuristics.HNSubSet;
import org.processmining.framework.util.PluginDocumentationLoader;
import org.processmining.mining.MiningResult;

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleFactory3D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.DoubleMatrix3D;


/**
 * This is the main class for the HeuristicsMiner++ Algorithm
 * 
 * @author Andrea Burattin
 * @version 0.4
 */
public class AutoHMPP extends HMPP {
	
	/* The plugin name... */
	private final String PLUGIN_NAME = "HeuristicsMiner++ (with MDL)";
	private HMPPParameters parameters;

	/** Maximum number of step while in a plateau */
	private int maxPlateauSteps = 1;
	private int numberOfSearchThread = 5;


	/* The events log */
	private LogEvents events;
	private LogReader log;
	/* An array list with all the observed events (just one entry for each
	 * event, without considering the cardinality and the event type) */
	private ArrayList<String> transitions;
	/* The number of atomic events (for caching purpose) */
	private int transitionsSize;
	private int eventsSize;

	
	/* The greatest network size */
	private int greatestNetworkSize = -1;
	/* The maximum number of step in the search thread */
	private int maxSearchSteps = 0;
	
	
	/* ===================== DATA FROM HEURISTICS MINER ===================== */
	/* Support matrices for the start and finish event detection */
	private DoubleMatrix1D startCount;
	private DoubleMatrix1D endCount;
	/* Matrix with the direct dependency measures */
//	private DoubleMatrix2D dependencyMeasures;	
	private DoubleMatrix2D longRangeSuccessionCount;
//	private DoubleMatrix2D causalSuccession;
	/* Information about the longrange dependecy relation */
	private DoubleMatrix2D longRangeDependencyMeasures;
	private DoubleMatrix2D dependencyMeasuresAccepted;
	/* Counts the total wrong dependency observations in the log */
	private DoubleMatrix2D noiseCounters;
	
	private DoubleMatrix1D L1LdependencyMeasuresAll;
	private boolean[] L1Lrelation;
	private DoubleMatrix2D L2LdependencyMeasuresAll;
	private int[] L2Lrelation;
	private DoubleMatrix2D ABdependencyMeasuresAll;
	private boolean[] alwaysVisited;
	
	private DoubleMatrix3D allAndMeasures;
	private DoubleMatrix2D andInMeasuresAll;
	private DoubleMatrix2D andOutMeasuresAll;
	
	private DoubleMatrix2D directSuccessionCount;
	private DoubleMatrix2D succession2Count;
	private DoubleMatrix2D parallelCount;
	
	private DoubleMatrix1D totalActivityCounter;
	private DoubleMatrix1D totalActivityTime;
	private DoubleMatrix2D totalOverlappingTime;
	
	/* list of parameters cost */
	private HashMap<String, ArrayList<Double>> parametersCosts;
	private ArrayList<Double[]> discretizedParameters = null;
	
//	double[] bestInputMeasure;
//	double[] bestOutputMeasure;
//	int[] bestInputEvent;
//	int[] bestOutputEvent;
	
	private boolean basicRelationsMade = false;
	
	/* debug variables */
	private final boolean DEBUG = false;
	private final boolean DEBUG_START_END = false;
	private int CALLS_DEEP = -1;
	
	private Notifier notifier = null;
	
	
	/**
	 * Default plugin constructor
	 */
	public AutoHMPP(Notifier notifier) {
		dbgStart();
		parameters = new HMPPParameters();
		this.notifier = notifier;
		dbgEnd();
	}
	
	
	@Override
	public String getName() {
		return PLUGIN_NAME;
	}

	
	@Override
	public JPanel getOptionsPanel(LogSummary summary) {
		return null;
	}
	
	/* not required anymore */
	/**
	 * WARNING: this method required modifications in the MiningSettings.java
	 *          file. Added lines 92, 93 and 94 
	 * 
	 * @param log
	 * @return actually there is no option to configure, return null
	 */
	public JPanel getOptionsPanel(LogReader log) {
		return null;
	}
	
	
	@Override
	public String getHtmlDescription() {
		return PluginDocumentationLoader.load(this);
	}
	
	
	@Override
	public MiningResult mine(LogReader log) {
		dbgStart();
		
		/* ===================== SUPPORT DATA POPULATION ==================== */
		// TODO: sistemare sta cosa
//		int[] exc = {};
//		this.log = log.clone(exc);
		this.log = log;
		this.discretizedParameters = null;
		dataInitialization(log);
		makeBasicRelations(log, 0.8);
		
		/* ================= LEARNING BEST PARAMETERS' VALUE ================ */
		calculateGreatestNetworkSize(log);
		
		/* build and construct each thread */
		ParameterSearchThread[] t = new ParameterSearchThread[numberOfSearchThread];
		for(int i = 0; i < numberOfSearchThread; i++) {
			t[i] = new ParameterSearchThread(new Integer(i).toString(), this);
			t[i].start();
		}
		/* wait for each thread to finish */
		try {
			for(int i = 0; i < numberOfSearchThread; i++) {
				t[i].join();
			}
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		/* ok, we got a winner! find it out... :) */
		Double bestCost = Double.MAX_VALUE;
		for(int i = 0; i < numberOfSearchThread; i++) {
			System.out.print("Thread " + i + " -- cost: " + t[i].getFinalCost());
			if (t[i].getFinalCost() < bestCost) {
				bestCost = t[i].getFinalCost();
				parameters = t[i].getParameters();
				maxSearchSteps = t[i].getExecutionStep();
				System.out.print(" -- best");
			}
			System.out.print("\n");
		}
		System.out.println("using this: " + bestCost);
		System.out.println("finished!");
		
		dbg("Mining with these parameters");
		dbg("============================");
		dbg("  dependency threshold = " + parameters.getDependencyThreshold());
		dbg(" positive observations = " + parameters.getPositiveObservationsThreshold());
		dbg("      relative to best = " + parameters.getRelativeToBestThreshold());
		dbg("         and threshold = " + parameters.getAndThreshold());
		dbg("         l1l threshold = " + parameters.getL1lThreshold());
		dbg("         l2l threshold = " + parameters.getL2lThreshold());
		dbg("          ld threshold = " + parameters.getLDThreshold());
		dbg(" use all act connected = " + parameters.useAllConnectedHeuristics);
		dbg(" use long distance dep = " + parameters.useLongDistanceDependency);
		
		/* =========================== DATA OUTPUT ========================== */
		MiningResult res = new HMPPResult(this, log, false, transitions);
		
		dbgEnd();
		return res;
	}
	
	
	/**
	 * This method to get the current activity time vector
	 * 
	 * @return the current activity time vector
	 */
	protected DoubleMatrix1D getActivityTime() {
		return totalActivityTime;
	}

	
	/**
	 * This method to get the current activity counter vector
	 * 
	 * @return the current activity counter vector
	 */
	protected DoubleMatrix1D getActivityCounter() {
		return totalActivityCounter;
	}

	
	/**
	 * This method to get the current overlapping time matrix
	 * 
	 * @return the current overlapping time matrix
	 */
	protected DoubleMatrix2D getOverlappingTime() {
		return totalOverlappingTime;
	}

	
	/**
	 * This method to get the current parallel count matrix
	 * 
	 * @return the current parallel count matrix
	 */
	protected DoubleMatrix2D getParallelCount() {
		return parallelCount;
	}

	
	/**
	 * This method to get the current algorithm parameters object
	 * 
	 * @return the current algorithm parameter object
	 */
	public HMPPParameters getParameters() {
		return parameters;
	}


	/**
	 * This method to set the current algorithm parameters object
	 * 
	 * @param param the new parameters
	 */
	public void setParameters(HMPPParameters param) {
		this.parameters = param;
	}
	
	
	/**
	 * This method builds the main object instances
	 * 
	 * @param log the log to analyse
	 */
	private void dataInitialization(LogReader log) {
		dbgStart();
		
		/* ====================== DATA INITIALIZATION ======================= */
		/* Build the single events array */
//		eventsFiltered = new LogEvents();
		transitions = new ArrayList<String>(Arrays.asList(log.getLogSummary().getModelElements()));
		transitionsSize = transitions.size();
		events = log.getLogSummary().getLogEvents();
		eventsSize = events.size();
		
		startCount = DoubleFactory1D.dense.make(eventsSize, 0.0);
		endCount = DoubleFactory1D.dense.make(eventsSize, 0.0);
		
		longRangeSuccessionCount = DoubleFactory2D.dense.make(eventsSize, eventsSize, 0);
		longRangeDependencyMeasures = DoubleFactory2D.dense.make(eventsSize, eventsSize, 0);
//		causalSuccession = DoubleFactory2D.dense.make(logAtomicEventsSize, logAtomicEventsSize, 0);
		longRangeSuccessionCount = DoubleFactory2D.dense.make(eventsSize, eventsSize, 0);
		dependencyMeasuresAccepted = DoubleFactory2D.sparse.make(eventsSize, eventsSize, 0.0);
		noiseCounters = DoubleFactory2D.sparse.make(events.size(), events.size(), 0);
		
		L1LdependencyMeasuresAll = DoubleFactory1D.sparse.make(eventsSize, 0);
		L2LdependencyMeasuresAll = DoubleFactory2D.sparse.make(eventsSize, eventsSize, 0);
		ABdependencyMeasuresAll = DoubleFactory2D.sparse.make(eventsSize, eventsSize, 0);
		
		allAndMeasures = DoubleFactory3D.sparse.make(eventsSize, eventsSize, eventsSize, 0);
		andInMeasuresAll = DoubleFactory2D.sparse.make(eventsSize, eventsSize, 0);
		andOutMeasuresAll = DoubleFactory2D.sparse.make(eventsSize, eventsSize, 0);
		
		directSuccessionCount = DoubleFactory2D.dense.make(eventsSize, eventsSize, 0);
		succession2Count = DoubleFactory2D.dense.make(eventsSize, eventsSize, 0);
		/* This matrix considers just the parallel relations between activity,
		 * not between events (so between A and B instead of A-start, B-start,
		 * A-finish, B-finish) */
		parallelCount = DoubleFactory2D.dense.make(transitionsSize, transitionsSize, 0);
		
		totalActivityCounter = DoubleFactory1D.dense.make(transitionsSize, 0);
		totalActivityTime = DoubleFactory1D.dense.make(transitionsSize, 0);
		totalOverlappingTime = DoubleFactory2D.dense.make(transitionsSize, transitionsSize, 0);
		
		/* loop relations matrix */
		L1Lrelation = new boolean[eventsSize];
		L2Lrelation = new int[eventsSize];
		for (int i = 0; i < eventsSize; i++) {
			L1Lrelation[i] = false;
			L2Lrelation[i] = -10;
		}
		
		dbgEnd();
	}
	
	
	/**
	 * This method builds all the basic relations, populating the long range
	 * succession count and invoking the calculateEventFrequencies for each
	 * process instance.
	 * 
	 * @param log
	 * @param causalityFall
	 */
	public void makeBasicRelations(LogReader log, double causalityFall) {
		dbgStart();
		
		/* Iterate through all log events */
		@SuppressWarnings("unchecked")
		Iterator<ProcessInstance> it = log.instanceIterator();
		while (it.hasNext()) {
			/* Extract the current process and its activity list */
			ProcessInstance pi = it.next();
			AuditTrailEntryList atel = pi.getAuditTrailEntryList();
			
			/* Update the successors and parallels matrices */
			calculateEventsFrequencies(atel);
			
			int i = 0;
			boolean terminate = false;
			while (!terminate) {
				@SuppressWarnings("unchecked")
				Iterator<AuditTrailEntry> it2 = atel.iterator();
				/* Skip the first i entries of the trace */
				for (int j = 0; j < i; j++) {
					it2.next();
				}
				/* The starting element */
				AuditTrailEntry begin = it2.next();
				LogEvent beginEvent = new LogEvent(begin.getElement(), begin.getType());
				/* Find the correct row of the matices */
				int row = events.indexOf(beginEvent);
				
				int distance = 0;
				boolean foundSelf = false;
				HNSubSet done = new HNSubSet();
				terminate = (!it2.hasNext());
				while (it2.hasNext() && (!foundSelf)) {
					/* The ending element */
					AuditTrailEntry end = it2.next();
					LogEvent endEvent = new LogEvent(end.getElement(), end.getType());
					/* Find the correct column of the matrices */
					int column = events.indexOf(endEvent);
					/* Is it the same? */
					foundSelf = (row == column);
					distance++;
					
					if (done.contains(column)) {
						continue;
					}
					done.add(column);
					
					/* Update long range matrix */
					longRangeSuccessionCount.set(row, column, longRangeSuccessionCount.get(row, column) + 1);

					/* Update causal matrix */
//					causalSuccession.set(row, column, causalSuccession.get(row, column) + Math.pow(causalityFall, distance - 1));
				}
				i++;
			}
		}

		// calculate longRangeDependencyMeasures
//		for (int i = 0; i < longRangeDependencyMeasures.rows(); i++) {
//			for (int j = 0; j < longRangeDependencyMeasures.columns(); j++) {
//				if (events.getEvent(i).getOccurrenceCount() == 0) {
//					continue;
//				}
//				longRangeDependencyMeasures.set(i, j, calculateLongDistanceDependencyMeasure(i, j));
//			}
//		}
		
		dbgEnd();
	}
	
	
	/**
	 * This method returns a map to track the number of connection for each
	 * possible positive observations threshold
	 * 
	 * @return a hash map where the key is the possible threshold and the value
	 * is the number of connection for that threshold (counting also all the 
	 * lower values)
	 */
	public HashMap<Double, Integer> getDiscretizedPositiveObsThresholds() {
		dbgStart();
		
		HashMap<Double, Integer> toret = new HashMap<Double, Integer>();
		Double key;
		Integer val;
		/* extracts the exact count for each threshold */
		for (int i = 0; i < directSuccessionCount.columns(); i++) {
			for (int j = 0; j < directSuccessionCount.rows(); j++) {
				key = directSuccessionCount.get(i, j);
				if (key > 0) {
					val = toret.get(key);
					if (val == null)
						val = new Integer(0);
					toret.put(key, val + 1);
				}
			}
		}
		
		/* sums all bigger thresholds value */
		Object[] keys = toret.keySet().toArray();
		Arrays.sort(keys);
		Integer curTot = 0;
		for (int i = keys.length - 1; i >= 0; i--) {
			curTot += toret.get(keys[i]);
			toret.put((Double) keys[i], curTot);
		}
		
		dbgEnd();
		return toret;
	}
	
	
	/**
	 * This method returns a map to track the number of connection for each
	 * possible dependency threshold 
	 * 
	 * @return a hash map where the key is the possible threshold and the value
	 * is the number of connection for that threshold (counting also all the 
	 * lower values)
	 */
	public HashMap<Double, Integer> getDiscretizedDependencyThresholds() {
		dbgStart();
		
		HashMap<Double, Integer> toret = new HashMap<Double, Integer>();
		Double key;
		Integer val;
		/* extracts the exact count for each threshold */
		for (int i = 0; i < directSuccessionCount.columns(); i++) {
			for (int j = 0; j < directSuccessionCount.rows(); j++) {
				boolean sameEvent = events.get(i).getModelElementName().equals(events.get(j).getModelElementName());
				key =  calculateDependencyMeasure(i, j);
				if (key > 0 && !sameEvent) {
					val = toret.get(key);
					if (val == null)
						val = new Integer(0);
					toret.put(key, val + 1);
				}
			}
		}
		
		/* sums all bigger thresholds value */
		Object[] keys = toret.keySet().toArray();
		Arrays.sort(keys);
		Integer curTot = 0;
		for (int i = keys.length - 1; i >= 0; i--) {
			curTot += toret.get(keys[i]);
			toret.put((Double) keys[i], curTot);
		}
		
		dbgEnd();
		return toret;
	}
	
	
	/**
	 * This method returns an array to track the possible values for the
	 * relative to best parameter
	 * 
	 * @return a set with a discretization of all possible relative to best
	 * values (for which there are changes in the output)
	 */
	public Double[] getDiscretizedRelativeToBests() {
		dbgStart();
		
		HashSet<Double> temp = new HashSet<Double>();
		
		double[] bestInputMeasure = new double[eventsSize];
		double[] bestOutputMeasure = new double[eventsSize];
		int[] bestInputEvent = new int[eventsSize];
		int[] bestOutputEvent = new int[eventsSize];
		calculateBestRelations(bestInputMeasure, bestOutputMeasure, bestInputEvent, bestOutputEvent);
		double measure;
		for (int i = 0; i < eventsSize; i++) {
			for (int j = 0; j < eventsSize; j++) {
				measure = calculateDependencyMeasure(i, j);
				temp.add(bestOutputMeasure[i] - measure);
			}
		}
		
		Double[] a = (Double[]) temp.toArray(new Double[temp.size()]);
		Arrays.sort(a);
		
		dbgEnd();
		return a;
	}
	
	
	/**
	 * This method returns an array to track the possible values for the loops
	 * of length one
	 * 
	 * @return a set with a discretization of all possible length one loops
	 * values (for which there are changes in the output)
	 */
	public Double[] getDiscretizedLength1Loop() {
		dbgStart();
		
		HashSet<Double> temp = new HashSet<Double>();
//		calculateBestRelations();
		double measure;
		for (int i = 0; i < eventsSize; i++) {
			measure = calculateL1LDependencyMeasure(i);
			if (measure <= 1.0) {
				temp.add(measure);
			}
		}
		
		Double[] a = (Double[]) temp.toArray(new Double[temp.size()]);
		Arrays.sort(a);
		
		dbgEnd();
		return a;
	}
	
	
	/**
	 * This method returns an array to track the possible values for the loops
	 * of length two
	 * 
	 * @return a set with a discretization of all possible length two loops
	 * values (for which there are changes in the output)
	 */
	public Double[] getDiscretizedLength2Loop() {
		dbgStart();
		
		HashSet<Double> temp = new HashSet<Double>();
//		calculateBestRelations();
		double measure;
		for (int i = 0; i < eventsSize; i++) {
			for (int j = 0; j < eventsSize; j++) {
				measure = calculateL2LDependencyMeasure(i, j);
				if (measure <= 1.0) {
					temp.add(measure);
				}
			}
		}
		
		Double[] a = (Double[]) temp.toArray(new Double[temp.size()]);
		Arrays.sort(a);
		
		dbgEnd();
		return a;
	}
	
	
	/**
	 * This method returns an array to track the possible values for the long
	 * distance parameter
	 * 
	 * @return a set with a discretization of all possible long distance
	 * thresholds (for which there are changes in the output)
	 */
	public Double[] getDiscretizedLongDistanceThreshold() {
		dbgStart();
		
		HashSet<Double> temp = new HashSet<Double>();
		
		double measure;
		for (int i = 0; i < longRangeDependencyMeasures.rows(); i++) {
			for (int j = 0; j < longRangeDependencyMeasures.columns(); j++) {
				if (events.getEvent(i).getOccurrenceCount() == 0) {
					continue;
				}
				measure = calculateLongDistanceDependencyMeasure(i, j);
				if (measure <= 1.0) {
					temp.add(measure);
					longRangeDependencyMeasures.set(i, j, measure);
				}
			}
		}
		
		Double[] a = (Double[]) temp.toArray(new Double[temp.size()]);
		Arrays.sort(a);
		
		dbgEnd();
		return a;
	}
	
	
	/**
	 * This method returns an array to track the possible values for the AND
	 * threshold parameter
	 * 
	 * @return a set with a discretization of all possible AND thresholds (for
	 * which there are changes in the output)
	 */
	public Double[] getDiscretizedANDThreshold() {
		dbgStart();
		
		HashSet<Double> temp = new HashSet<Double>();
		
		double measure;
		for(int i = 0; i < eventsSize; i++) {
			for(int j = 0; j < eventsSize; j++) {
				for(int k = 0; k < eventsSize; k++) {
					if (i!=j && i!=k && j!=k)
					{
						measure = andInMeasureF(i, j, k);
						if (measure <= 1.0) {
							temp.add(measure);
							allAndMeasures.set(i, j, k, measure);
						}
					}
				}
			}
		}
		
		Double[] a = (Double[]) temp.toArray(new Double[temp.size()]);
		Arrays.sort(a);
		
		dbgEnd();
		return a;
	}
	
	
	/**
	 * This method returns an ArrayList with all the discretized values for each
	 * parameter. Each element is an array of Double containing the
	 * possible values for the parameter. This is the map from the
	 * index to the parameter discratization:
	 * <ul>
	 * 	<li>0. dependency threshold</li>
	 *  <li>1. positive observations</li>
	 *  <li>2. relative to best</li>
	 *  <li>3. and threshold</li>
	 *  <li>4. length one loop</li>
	 *  <li>5. length two loop</li>
	 *  <li>6. long distance dep</li>
	 * </ul>
	 * 
	 * @return the discretized parameters
	 */
	public synchronized ArrayList<Double[]> getDiscretizedParameters() {
		dbgStart();
		
		notifier.stepStarts("NULL", Notifier.STEPS.PARAMETER_DISCRETIZATION);
		
		if (discretizedParameters == null) {
			discretizedParameters = new ArrayList<Double[]>(7);
			/* dependency thresholds */
			discretizedParameters.add(0, getDiscretizedDependencyThresholds().keySet().toArray(new Double[0]));
			/* positive observations */
			discretizedParameters.add(1, getDiscretizedPositiveObsThresholds().keySet().toArray(new Double[0]));
			/* relative to best */
			discretizedParameters.add(2, getDiscretizedRelativeToBests());
			/* AND threshold */
			discretizedParameters.add(3, getDiscretizedANDThreshold());
			/* length one/two loops */
			discretizedParameters.add(4, getDiscretizedLength1Loop());
			discretizedParameters.add(5, getDiscretizedLength2Loop());
			/* long distance dep */
			discretizedParameters.add(6, getDiscretizedLongDistanceThreshold());
		}
		
		notifier.stepEnds("NULL", Notifier.STEPS.PARAMETER_DISCRETIZATION);
		
		dbgEnd();
		return discretizedParameters;
	}
	
	
	private void calculateBestRelations(double[] bestInputMeasure, double[] bestOutputMeasure, int[] bestInputEvent, int[] bestOutputEvent) {
		dbgStart();
		
//		bestInputMeasure = new double[eventsSize];
//		bestOutputMeasure = new double[eventsSize];
//		bestInputEvent = new int[eventsSize];
//		bestOutputEvent = new int[eventsSize];
		double measure;
		
		for (int i = 0; i < eventsSize; i++) {
			bestInputMeasure[i] = -10.0;
			bestOutputMeasure[i] = -10.0;
			bestInputEvent[i] = -1;
			bestOutputEvent[i] = -1;
		}
		/* Search the beste ones */
		for (int i = 0; i < eventsSize; i++) {
			for (int j = 0; j < eventsSize; j++) {
				if (i != j) {
					measure = calculateDependencyMeasure(i, j);
//					measure = dependencyMeasuresAccepted.get(i, j);
//					dependencyMeasuresAccepted.set(i, j, measure);
					ABdependencyMeasuresAll.set(i, j, measure);

					if (measure > bestOutputMeasure[i]) {
						bestOutputMeasure[i] = measure;
						bestOutputEvent[i] = j;
					}
					if (measure > bestInputMeasure[j]) {
						bestInputMeasure[j] = measure;
						bestInputEvent[j] = i;
					}
				}
			}
		}
		
		dbgEnd();
	}
	
	
	/**
	 * This method extracts information on the parameter instance, calculating
	 * the direct successions matrix and the parallel events matrix
	 * 
	 * @param atel the process instance's activities
	 */
	@SuppressWarnings("unchecked")
	private void calculateEventsFrequencies(AuditTrailEntryList atel) {
//		dbgStart();

		/* All the activities finished just before the current one */
		HashMap<String, Integer> finishedActivities = new HashMap<String, Integer>();
		/* All the activities finished from ever */
		HashMap<String, Integer> finishedActivitiesEver = new HashMap<String, Integer>();
		/* All the activities started but not yet finished */
		HashMap<String, Long[]> startedNotFinishedActivities = new HashMap<String, Long[]>();
		/* This object contains all the events finished next to last */
		HashMap<String, Integer> nextToLastFinishedActivities = null;
		
		/* Starting and ending elements for this process instance */
		int startElement = -1;
		int endElement = -1;
		
		/* We have to iterate throughout the process instance */
		Iterator<AuditTrailEntry> i = atel.iterator();
		/* We need to remember if the last activity was a finish so if we have
		 * no other direct successors */
		boolean previousEventWasComplete = false;

		while (i.hasNext()) {	
			AuditTrailEntry ate = i.next();
			LogEvent le = new LogEvent(ate.getElement(), ate.getType());
			String leName = ate.getName();
			String leType = ate.getType();
			
			int indexOfAct = events.indexOf(le);
			int indexOfTransition = transitions.indexOf(leName);
			
			if (leType.equals("start")) {
				
				/* If required, update the starting activity
				 */
				if (startElement == -1) {
					startElement = indexOfAct;
				}
				
				/* This is the start of a new activity, all the activities
				 * started but not finished are overlapped with this one and all
				 * the activities already finished are before this one.
				 */
				/* Set up the activity direct successors */
				for (String act : finishedActivities.keySet()) {
					int indexOfCurrAct = events.indexOf(new LogEvent(act, "complete"));
					double old = directSuccessionCount.get(indexOfCurrAct, indexOfAct);
					directSuccessionCount.set(indexOfCurrAct, indexOfAct, old + 1);
//					System.out.println("   Added "+ act +" => "+ leName);
				}
				
				/* Set up the activity successors */
				// TODO: here i have to correctly populate the succession2Count
				//       array, in order to mine correctly the length two loop
//				if (nextToLastFinishedActivities != null) {
//					for (String act : nextToLastFinishedActivities.keySet()) {
//						int indexOfCurrAct = events.indexOf(new LogEvent(act, "complete"));
//						double old = succession2Count.get(indexOfCurrAct, indexOfAct);
//						succession2Count.set(transitions.indexOf(act), indexOfAct, old + 1);
//						System.out.println("   Added "+ act +" ===> "+ leName);
//					}
//				}
				
				/* Overlapped activities */
				for (String act : startedNotFinishedActivities.keySet()) {
					double old = parallelCount.get(transitions.indexOf(act), indexOfTransition);
					parallelCount.set(transitions.indexOf(act), indexOfTransition, old + 1);
					parallelCount.set(indexOfTransition, transitions.indexOf(act), old + 1);
//					System.out.println("   Added "+ act +" || "+ leName);
				}
				
				/* Started not finished increment */
				Long[] val_started_not_finished = {1L, ate.getTimestamp().getTime()};
				if (startedNotFinishedActivities.containsKey(leName)) {
					val_started_not_finished[0] += startedNotFinishedActivities.get(leName)[0];
				}
				startedNotFinishedActivities.put(leName, val_started_not_finished);
				
				previousEventWasComplete = false;
				
			} else if (leType.equals("complete")) {
				
				/* Update the current end activity  */
				endElement = indexOfAct;
				
				/* Update the activity counter and the total activity time */
				double oldOccur = totalActivityCounter.get(indexOfTransition);
				totalActivityCounter.set(indexOfTransition, oldOccur+1);
				oldOccur = totalActivityTime.get(indexOfTransition);

				/* We have to clean this because we want to keep only the DIRECT
				 * successors of the activity, just if there are no other
				 * acrivity ended before */
				if (!previousEventWasComplete) {
					nextToLastFinishedActivities = (HashMap<String, Integer>)finishedActivities.clone();
					finishedActivities.clear();
				}
				
				/* This is the finish of an activity, I have just to terminate
				 * the start.  
				 */
				/* Eventual started but not finished removal */
				if (startedNotFinishedActivities.containsKey(leName))
				{
					Long[] val_started_not_finished = startedNotFinishedActivities.get(leName);
					/* Update the total activity time */
					double time = totalActivityTime.get(transitions.indexOf(leName));
					time += ((ate.getTimestamp().getTime() - val_started_not_finished[1]) / 1000);
					totalActivityTime.set(transitions.indexOf(leName), time);

					/* Update the started not finished map */
					long val = val_started_not_finished[0];
					if (val == 1) {
						startedNotFinishedActivities.remove(leName);
					} else {
						val_started_not_finished[0] = val - 1L;
						startedNotFinishedActivities.put(leName, val_started_not_finished);
					}
					
					/* Update the total overlapping time */
					for (String act : startedNotFinishedActivities.keySet()) {
						/* Update the overlapping time only for the activities
						 * different from the current one */
						if (!act.equals(leName)) {
							int indexOfCurrAct = transitions.indexOf(act);
							time = ate.getTimestamp().getTime() - startedNotFinishedActivities.get(act)[1];
							time /= 1000;
							time += totalOverlappingTime.get(indexOfTransition, indexOfCurrAct);
							totalOverlappingTime.set(indexOfTransition, indexOfCurrAct, time);
							totalOverlappingTime.set(indexOfCurrAct, indexOfTransition, time);
						}
					}
				}
				
				/* Finished activities increment */
				int val_finished = 1;
				int val_finished_ever = 1;
				if (finishedActivities.containsKey(leName)) {
					val_finished += finishedActivities.get(leName);
				}
				finishedActivities.put(leName, val_finished);
				
				if (finishedActivitiesEver.containsKey(leName)) {
					val_finished_ever += finishedActivitiesEver.get(leName);
				}
				finishedActivitiesEver.put(leName, val_finished_ever);
				
				previousEventWasComplete = true;
				
			}
		}
		/* Update the start / finish process counter */
		if (startElement >= 0) {
			startCount.set(startElement, startCount.get(startElement) + 1);
		}
		if (endElement >= 0) {
			endCount.set(endElement, endCount.get(endElement) + 1);
		}
		
//		dbgEnd();
	}


	/**
	 * This method uses the support data to build the heuristics relations.
	 * These are the main steps of this procedure:
	 *   - Best start and end activities calculation
	 *   - Build dependency measures
	 *   - Given the InputSets and OutputSets build OR-subsets
	 *   - Build the HeuristicsNetwork as output
	 * 
	 * @param log the current log
	 * @param parameters the parameter configuration
	 * @return the heuristics net from the log
	 */
	public HMPPHeuristicsNet makeHeuristicsRelations(LogReader log, HMPPParameters parameters) {
		dbgStart();
		
		/* Step 0 =========================================================== */
		dbg("Step 0");
		/* Data initialization */
		dependencyMeasuresAccepted = DoubleFactory2D.sparse.make(eventsSize, eventsSize, 0.0);
		
		/* The net we are going to build... */
//		DependencyHeuristicsNet result = new DependencyHeuristicsNet(eventsFiltered,
//				dependencyMeasuresAccepted, directSuccessionCount);
		HMPPHeuristicsNet result = new HMPPHeuristicsNet(events, dependencyMeasuresAccepted, directSuccessionCount);
//		HeuristicsNet result = new DependencyHeuristicsNet()

//		L1Lrelation = new boolean[eventsSize];
//		L2Lrelation = new int[eventsSize];
		
		HNSubSet[] inputSet = new HNSubSet[eventsSize];
		HNSubSet[] outputSet = new HNSubSet[eventsSize];
		
		for (int i = 0; i < eventsSize; i++) {
			inputSet[i] = new HNSubSet();
			outputSet[i] = new HNSubSet();
//			L1Lrelation[i] = false;
//			L2Lrelation[i] = -10;
		}
		
		/* Step 1 =========================================================== */
		dbg("Step 1");
		/* Best start and end activities calculation */
		int bestStart = 0;
		int bestEnd = 0;
		for (int i = 0; i < eventsSize; i++) {
			if (startCount.get(i) > startCount.get(bestStart)) {
				bestStart = i;
			}
			if (endCount.get(i) > endCount.get(bestEnd)) {
				bestEnd = i;
			}
		}
		/* Set the start task */
		HNSubSet startTask = new HNSubSet();
		startTask.add(bestStart);
		result.setStartTasks(startTask);
		/* Set the end task */
		HNSubSet endTask = new HNSubSet();
		endTask.add(bestEnd);
		result.setEndTasks(endTask);
		/* Update noiseCounters */
		noiseCounters.set(bestStart, 0, log.getLogSummary().getNumberOfProcessInstances() - startCount.get(bestStart));
		noiseCounters.set(0, bestEnd, log.getLogSummary().getNumberOfProcessInstances() - endCount.get(bestEnd));
		
		/* Step 2 =========================================================== */
		dbg("Step 2");
		/* Build dependency measures */
		double measure = 0.0;
		
		/* Step 2.1 - L1L loops ............................................. */
		for (int i = 0; i < eventsSize; i++) {
			measure = calculateL1LDependencyMeasure(i);
			L1LdependencyMeasuresAll.set(i, measure);
			if (measure >= parameters.getL1lThreshold() &&
					directSuccessionCount.get(i, i) >= parameters.getPositiveObservationsThreshold()) {
				dependencyMeasuresAccepted.set(i, i, measure);
				L1Lrelation[i] = true;
				inputSet[i].add(i);
				outputSet[i].add(i);
			}
		}
		
		/* Step 2.2 - L2L loops ............................................. */
		for (int i = 0; i < eventsSize; i++) {
			for (int j = 0; j < eventsSize; j++) {
				measure = calculateL2LDependencyMeasure(i, j);
				L2LdependencyMeasuresAll.set(i, j, measure);
				L2LdependencyMeasuresAll.set(j, i, measure);
				
				if ((i != j) && (measure >= parameters.getL2lThreshold()) && 
						((succession2Count.get(i, j) + succession2Count.get(j,i)) >= parameters.getPositiveObservationsThreshold())) {
					dependencyMeasuresAccepted.set(i, j, measure);
					dependencyMeasuresAccepted.set(j, i, measure);
					L2Lrelation[i] = j;
					L2Lrelation[j] = i;
					inputSet[i].add(j);
					outputSet[j].add(i);
					inputSet[j].add(i);
					outputSet[i].add(j);
				}
			}
		}
		
		/* Step 2.3 - Normal dependecy measure .............................. */
		/* Independent of any threshold search the best input and output
		 * connection */
//		int[] bestInputEvent = new int[eventsSize];
//		int[] bestOutputEvent = new int[eventsSize];
//		for (int i = 0; i < eventsSize; i++) {
//			bestInputMeasure[i] = -10.0;
//			bestOutputMeasure[i] = -10.0;
//			bestInputEvent[i] = -1;
//			bestOutputEvent[i] = -1;
//		}
//		/* Search the beste ones */
//		for (int i = 0; i < eventsSize; i++) {
//			for (int j = 0; j < eventsSize; j++) {
//				if (i != j) {
//					measure = calculateDependencyMeasure(i, j);
////					measure = dependencyMeasuresAccepted.get(i, j);
////					dependencyMeasuresAccepted.set(i, j, measure);
//					ABdependencyMeasuresAll.set(i, j, measure);
//
//					if (measure > bestOutputMeasure[i]) {
//						bestOutputMeasure[i] = measure;
//						bestOutputEvent[i] = j;
//					}
//					if (measure > bestInputMeasure[j]) {
//						bestInputMeasure[j] = measure;
//						bestInputEvent[j] = i;
//					}
//				}
//			}
//		}
		double[] bestInputMeasure = new double[eventsSize];
		double[] bestOutputMeasure = new double[eventsSize];
		int[] bestInputEvent = new int[eventsSize];
		int[] bestOutputEvent = new int[eventsSize];
		calculateBestRelations(bestInputMeasure, bestOutputMeasure, bestInputEvent, bestOutputEvent);
		
		/* Extra check for best compared with L2L-loops */
		for (int i = 0; i < eventsSize; i++) {
			if ((i!=bestStart) && (i!=bestEnd)) {
				for (int j = 0; j < eventsSize; j++) {
					measure = calculateL2LDependencyMeasure(i, j);
					
					if (measure > bestInputMeasure[i]) {
						dependencyMeasuresAccepted.set(i, j, measure);
						dependencyMeasuresAccepted.set(j, i, measure);
						L2Lrelation[i] = j;
						L2Lrelation[j] = i;
						inputSet[i].add(j);
						outputSet[j].add(i);
						inputSet[j].add(i);
						outputSet[i].add(j);
					}
				}
			}
		}
		/* Update the dependencyMeasuresAccepted matrix, the inputSet, outputSet
		 * arrays and the noiseCounters matrix */
		if (parameters.useAllConnectedHeuristics) {
			for (int i = 0; i < eventsSize; i++) {
				/* consider each case */
				int j = L2Lrelation[i];
				if (i != bestStart) {
					if ((j > -1) && (bestInputMeasure[j] > bestInputMeasure[i])) {
						/* i is in a L2L relation with j but j has a stronger
						 * input connection do nothing */
					} else {
						dependencyMeasuresAccepted.set(bestInputEvent[i], i, bestInputMeasure[i]);
						inputSet[i].add(bestInputEvent[i]);
						outputSet[bestInputEvent[i]].add(i);
						noiseCounters.set(bestInputEvent[i], i, directSuccessionCount.get(i, bestInputEvent[i]));
					}
				}
				if (i != bestEnd) {
					if ((j > -1) && (bestOutputMeasure[j] > bestOutputMeasure[i])) {
						/* i is in a L2L relation with j but j has a stronger
						 * input connection do nothing */
					} else {
						dependencyMeasuresAccepted.set(i, bestOutputEvent[i], bestOutputMeasure[i]);
						inputSet[bestOutputEvent[i]].add(i);
						outputSet[i].add(bestOutputEvent[i]);
						noiseCounters.set(i, bestOutputEvent[i], directSuccessionCount.get(bestOutputEvent[i], i));
					}
				}
			}
		} else {
			/* Connect all starts with the relative finish */
			for (int i = 0; i < eventsSize; i++) {
				for (int j = 0; j < eventsSize; j++) {
					LogEvent leI = events.get(i);
					LogEvent leJ = events.get(j);
					boolean sameEvent = leI.getModelElementName().equals(leJ.getModelElementName());
					boolean isIStart = leI.getEventType().equals("start");
					boolean isIFinish = leI.getEventType().equals("complete");
					boolean isJStart = leJ.getEventType().equals("start");
					boolean isJFinish = leJ.getEventType().equals("complete");
					if (sameEvent /*&& isIStart && isJFinish*/) {
						if (isIStart && isJFinish) {
							outputSet[i].add(j);
							inputSet[j].add(i);
						} else if (isJStart && isIFinish) {
							outputSet[j].add(i);
							inputSet[i].add(j);
						}
					}
				}
			}
		}
		/* Search for other connections that fulfill all the thresholds */
		for (int i = 0; i < eventsSize; i++) {
			for (int j = 0; j < eventsSize; j++) {
				if (dependencyMeasuresAccepted.get(i, j) <= 0.0001) {
					measure = calculateDependencyMeasure(i, j);
					if (((bestOutputMeasure[i] - measure) <= parameters.getRelativeToBestThreshold()) &&
							(directSuccessionCount.get(i, j) >= parameters.getPositiveObservationsThreshold()) &&
							(measure >= parameters.getDependencyThreshold())) {
						dependencyMeasuresAccepted.set(i, j, measure);
						inputSet[j].add(i);
						outputSet[i].add(j);
						noiseCounters.set(i, j, directSuccessionCount.get(j, i));
					}
				}
			}
		}
		
		/* Step 3 =========================================================== */
		dbg("Step 3");
		/* Given the InputSets and OutputSets build OR-subsets */
		double score;
		alwaysVisited = new boolean[eventsSize];
		for (int i = 0; i < eventsSize; i++) {
			result.setInputSet(i, buildOrInputSets(i, inputSet[i]));
			result.setOutputSet(i, buildOrOutputSets(i, outputSet[i]));
		}
//		System.out.println(andInMeasuresAll);
//		System.out.println(andOutMeasuresAll);
		/* Update the HeuristicsNet with non binairy dependecy relations */
		/* Search for always visited activities */
		if (parameters.useLongDistanceDependency) {
			alwaysVisited[bestStart] = false;
			for (int i = 1; i < eventsSize; i++) {
				BitSet h = new BitSet();
				if (escapeToEndPossibleF(bestStart, i, h, result)) {
					alwaysVisited[i] = false;
				} else {
					alwaysVisited[i] = true;
				}
			}
//		/* Why close the if and than re-open it? :-/ */
//		}
//		if (USE_LONG_DISTANCE_CONNECTIONS) {
//		if (parameters.useLongDistanceDependency) {
			for (int i = (eventsSize - 1); i >= 0; i--) {
				for (int j = (eventsSize - 1); j >= 0; j--) {
					if ((i == j) || (alwaysVisited[j] && (j != bestEnd))) {
						continue;
					}
					score = calculateLongDistanceDependencyMeasure(i, j);
					if (score > parameters.getLDThreshold()) {
						BitSet h = new BitSet();
						if (escapeToEndPossibleF(i, j, h, result)) {
							// HNlongRangeFollowingChance.set(i, j, hnc);
							dependencyMeasuresAccepted.set(i, j, score);

							// update heuristicsNet
							HNSubSet helpSubSet = new HNSubSet();
							HNSet helpSet = new HNSet();

							helpSubSet.add(j);
							helpSet = result.getOutputSet(i);
							helpSet.add(helpSubSet);
							result.setOutputSet(i, helpSet);

							helpSubSet = new HNSubSet();
							helpSet = new HNSet();

							helpSubSet.add(i);
							helpSet = result.getInputSet(j);
							helpSet.add(helpSubSet);
							result.setInputSet(j, helpSet);
						}
					}
				}
			}
		}
		/*int numberOfConnections = 0;
		for (int i = 0; i < dependencyMeasuresAccepted.rows(); i++) {
			for (int j = 0; j < dependencyMeasuresAccepted.columns(); j++) {
				if (dependencyMeasuresAccepted.get(i, j) > 0.01) {
					numberOfConnections = numberOfConnections + 1;
				}
			}
		}
		result.setConnections(numberOfConnections);*/
		int noiseTotal = 0;
		for (int i = 0; i < noiseCounters.rows(); i++) {
			for (int j = 0; j < noiseCounters.columns(); j++) {
				noiseTotal = noiseTotal + (int) noiseCounters.get(i, j);
			}
		}

		/* Step 4 =========================================================== */
		dbg("Step 4");
		/* Building the output */
//		HMPPHeuristicsNet[] population = new HMPPHeuristicsNet[1];
//		population[0] = result;
		
//		System.out.println("Input-output set, before disconnection:");
//		for (int i = 0; i < eventsSize; i++) {
//			System.out.println(events.get(i) +"  in = "+ result.getInputSet(i));
//			System.out.println(events.get(i) +" out = "+ result.getOutputSet(i));
//			System.out.println();
//		}
		
//		DTContinuousSemanticsFitness fitness1 = new DTContinuousSemanticsFitness(log);
//		fitness1.calculate(population);
//		System.out.println("Continuous semantics fitness = " + population[0].getFitness());
//		
//		DTImprovedContinuousSemanticsFitness fitness2 = new DTImprovedContinuousSemanticsFitness(log);
//		fitness2.calculate(population);
//		System.out.println("Improved Continuous semantics fitness = " + population[0].getFitness());
//		System.out.println("=============================================================");
//		DTImprovedContinuousSemanticsFitness fitness2 = new DTImprovedContinuousSemanticsFitness(log);
//		fitness2.calculate(population);
		
//		population[0].disconnectUnusedElements();
		
//		for (int i = 0; i < dependencyMeasuresAccepted.rows(); i++) {
//			for (int j = 0; j < dependencyMeasuresAccepted.columns(); j++) {
//				System.out.print(dependencyMeasuresAccepted.get(i, j) + " ");
//			}
//		}
//		System.out.println("");
		
		dbgEnd();
		return result;
	}
	

	/**
	 * This method calculates the long distance dependency measure between two
	 * activities
	 *  
	 * @param i the first activity index
	 * @param j the second activity index
	 * @return the dependency measure
	 */
	private double calculateLongDistanceDependencyMeasure(int i, int j) {
		return ((double) longRangeSuccessionCount.get(i, j) / (events.getEvent(i).getOccurrenceCount() + parameters.getDependencyDivisor())) -
				(5.0 * (Math.abs(events.getEvent(i).getOccurrenceCount() - events.getEvent(j).getOccurrenceCount())) / events.getEvent(i).getOccurrenceCount());

	}
	
	
	/**
	 * This method calculates the length one loop between two activities
	 *  
	 * @param i the activity index
	 * @return the dependency measure
	 */
	private double calculateL1LDependencyMeasure(int i) {
		return ((double) directSuccessionCount.get(i, i)) /
				(directSuccessionCount.get(i, i) + parameters.getDependencyDivisor());
	}
	
	

	/**
	 * This method calculates the length two loop distance dependency measure
	 * between two activities
	 *  
	 * @param i the first activity index
	 * @param j the second activity index
	 * @return the dependency measure
	 */
	private double calculateL2LDependencyMeasure(int i, int j) {
		/* Problem if, for instance, we have a A -> A loop in parallel with B
		 * the |A > B > A|-value can be high without a L2L-loop
		 */
		if ((L1Lrelation[i] && succession2Count.get(i, j) >= parameters.getPositiveObservationsThreshold()) ||
			(L1Lrelation[j] && succession2Count.get(j, i) >= parameters.getPositiveObservationsThreshold())) {
			return 0.0;
		} else {
//			LogEvent leI = events.get(i);
//			LogEvent leJ = events.get(j);
//			int transitionIndexI = transitions.indexOf(leI.getModelElementName());
//			int transitionIndexJ = transitions.indexOf(leJ.getModelElementName());
			return ((double) succession2Count.get(i, j) + succession2Count.get(j, i)) /
					(succession2Count.get(i, j) + 
					 succession2Count.get(j, i) + 
					 /*(parallelCount.get(transitionIndexI, transitionIndexJ) * parameters.getIntervalsOverlapMultiplier()) +*/ 
					 parameters.getDependencyDivisor());
		}
	}
	

	/**
	 * This method calculates the dependency measure between two activities
	 *  
	 * @param i the first activity index
	 * @param j the second activity index
	 * @return the dependency measure
	 */
	private double calculateDependencyMeasure(int i, int j) {
		LogEvent leI = events.get(i);
		LogEvent leJ = events.get(j);
		boolean sameEvent = leI.getModelElementName().equals(leJ.getModelElementName());
		boolean isIStart = leI.getEventType().equals("start");
		boolean isIFinish = leI.getEventType().equals("complete");
		boolean isJStart = leJ.getEventType().equals("start");
		boolean isJFinish = leJ.getEventType().equals("complete");
		if (sameEvent && isIStart && isJFinish) {
			return 1.0;
		} else if ((!sameEvent) && isIFinish && isJStart) {
			int transitionIndexI = transitions.indexOf(leI.getModelElementName());
			int transitionIndexJ = transitions.indexOf(leJ.getModelElementName());
			
			double calc;
			/* TODO Check the use of direct succession or simply succession */
			calc = (directSuccessionCount.get(i, j) - directSuccessionCount.get(j, i)) / 
				   (directSuccessionCount.get(i, j) + 
					directSuccessionCount.get(j, i) + 
					(parallelCount.get(transitionIndexI, transitionIndexJ) * parameters.getIntervalsOverlapMultiplier()) + 
					parameters.getDependencyDivisor());
			return calc;
		} else {
			return 0.0;
		}
	}

	
	/**
	 * This method builds the or input set for the event
	 * 
	 * @param ownerE the current event index
	 * @param inputSet the input events set
	 * @return the corrent input set
	 */
	private HNSet buildOrInputSets(int ownerE, HNSubSet inputSet) {
		HNSet h = new HNSet();
		int currentE;
		
		// using the welcome method,
		// distribute elements of TreeSet inputSet over the elements of HashSet h
		boolean minimalOneOrWelcome;
		//setE = null;
		//Iterator hI = h.iterator();
		HNSubSet helpTreeSet;
		for (int isetE = 0; isetE < inputSet.size(); isetE++) {
			currentE = inputSet.get(isetE);
			minimalOneOrWelcome = false;
			for (int ihI = 0; ihI < h.size(); ihI++) {
				helpTreeSet = h.get(ihI);
				if (xorInWelcome(ownerE, currentE, helpTreeSet)) {
					minimalOneOrWelcome = true;
					helpTreeSet.add(currentE);
				}
			}
			if (!minimalOneOrWelcome) {
				helpTreeSet = new HNSubSet();
				helpTreeSet.add(currentE);
				h.add(helpTreeSet);
			}
		}

		// look to the (A v B) & (B v C) example with B A C in the inputSet;
		// result is [AB] [C]
		// repeat to get [AB] [BC]

		for (int isetE = 0; isetE < inputSet.size(); isetE++) {
			currentE = inputSet.get(isetE);
			for (int ihI = 0; ihI < h.size(); ihI++) {
				helpTreeSet = h.get(ihI);
				if (xorInWelcome(ownerE, currentE, helpTreeSet)) {
					helpTreeSet.add(currentE);
				}
			}
		}
		return h;
	}


	/**
	 * This method builds the or output set for the event
	 * 
	 * @param ownerE the current event index
	 * @param outputSEt the output events set
	 * @return the corrent output set
	 */
	private HNSet buildOrOutputSets(int ownerE, HNSubSet outputSet) {
		HNSet h = new HNSet();
		int currentE;

		// using the welcome method,
		// distribute elements of TreeSet inputSet over the elements of HashSet h
		boolean minimalOneOrWelcome;
		//setE = null;
		HNSubSet helpTreeSet;
		for (int isetE = 0; isetE < outputSet.size(); isetE++) {
			currentE = outputSet.get(isetE);
			minimalOneOrWelcome = false;
			for (int ihI = 0; ihI < h.size(); ihI++) {
				helpTreeSet = h.get(ihI);
				if (xorOutWelcome(ownerE, currentE, helpTreeSet)) {
					minimalOneOrWelcome = true;
					helpTreeSet.add(currentE);
				}
			}
			if (!minimalOneOrWelcome) {
				helpTreeSet = new HNSubSet();
				helpTreeSet.add(currentE);
				h.add(helpTreeSet);
			}
		}

		// look to the (A v B) & (B v C) example with B A C in the inputSet;
		// result is [AB] [C]
		// repeat to get [AB] [BC]
		for (int isetE = 0; isetE < outputSet.size(); isetE++) {
			currentE = outputSet.get(isetE);
			for (int ihI = 0; ihI < h.size(); ihI++) {
				helpTreeSet = h.get(ihI);
				if (xorOutWelcome(ownerE, currentE, helpTreeSet)) {
					helpTreeSet.add(currentE);
				}
			}
		}

		return h;
	}


	/**
	 * This method determines if two elements are in a XOR split
	 * 
	 * @param ownerE first element
	 * @param newE second element
	 * @param h the elements subset
	 * @return true if the elements are in a XOR splir
	 */
	private boolean xorInWelcome(int ownerE, int newE, HNSubSet h) {
		boolean welcome = true;
		int oldE;
		double andValue;

		for (int ihI = 0; ihI < h.size(); ihI++) {
			oldE = h.get(ihI);
			andValue = andInMeasureF(ownerE, oldE, newE);
			if (newE != oldE) {
				andInMeasuresAll.set(newE, oldE, andValue);
			}
			if (andValue > parameters.getAndThreshold()) {
				welcome = false;
			}
		}
		return welcome;
	}

	
	/**
	 * This method determines if two elements are in a XOR join
	 * 
	 * @param ownerE first element
	 * @param newE second element
	 * @param h the elements subset
	 * @return true if the elements are in a XOR splir
	 */
	private boolean xorOutWelcome(int ownerE, int newE, HNSubSet h) {
		boolean welcome = true;
		int oldE;
		double andValue;

		for (int ihI = 0; ihI < h.size(); ihI++) {
			oldE = h.get(ihI);
			andValue = andOutMeasureF(ownerE, oldE, newE);
			if (newE != oldE) {
				andOutMeasuresAll.set(newE, oldE, andValue);
			}
			if (andValue > parameters.getAndThreshold()) {
				welcome = false;
			}
		}
		return welcome;
	}
	
	
	/**
	 * This method determines if two elements are in a AND split
	 * 
	 */
	private double andInMeasureF(int ownerE, int oldE, int newE) {
		double toret = 0.0;
		if (ownerE == newE) {
			toret = 0.;
		/* TODO: verify if it's correct to not consider this case */
//		} else if ((directSuccessionCount.get(oldE, newE) < parameters.getPositiveObservationsThreshold()) ||
//				(directSuccessionCount.get(newE, oldE) < parameters.getPositiveObservationsThreshold())) {
//			toret = 0.;
		} else {
			int pcIndexNewE = transitions.indexOf(events.get(newE).getModelElementName());
			int pcIndexOldE = transitions.indexOf(events.get(oldE).getModelElementName());
			toret = ((double) directSuccessionCount.get(oldE, newE) + 
					         directSuccessionCount.get(newE, oldE) + 
					         (parallelCount.get(pcIndexNewE, pcIndexOldE) * parameters.getIntervalsOverlapMultiplier())) /
					// relevantInObservations;
					(directSuccessionCount.get(newE, ownerE) + 
					 directSuccessionCount.get(oldE, ownerE) + 1);
		}
		return toret;
	}

	
	/**
	 * This method determines if two elements are in a AND join
	 * 
	 */
	private double andOutMeasureF(int ownerE, int oldE, int newE) {
		double toret = 0.0;
		if (ownerE == newE) {
			toret = 0.;
		/* TODO: verify if it's correct to not consider this case */
//		} else if ((directSuccessionCount.get(oldE, newE) < parameters.getPositiveObservationsThreshold()) ||
//				(directSuccessionCount.get(newE, oldE) < parameters.getPositiveObservationsThreshold())) {
//			toret = 0.;
		} else {
			int pcIndexNewE = transitions.indexOf(events.get(newE).getModelElementName());
			int pcIndexOldE = transitions.indexOf(events.get(oldE).getModelElementName());
			toret = ((double) directSuccessionCount.get(oldE, newE) + 
					         directSuccessionCount.get(newE, oldE) + 
					         (parallelCount.get(pcIndexNewE, pcIndexOldE) * parameters.getIntervalsOverlapMultiplier())) /
					// relevantOutObservations;
					(directSuccessionCount.get(ownerE, newE) + 
					 directSuccessionCount.get(ownerE, oldE) + 1);
		}
		return toret;
	}
	
	
	/**
	 * 
	 * @param x
	 * @param y
	 * @param alreadyVisit
	 * @param result
	 * @return
	 */
	private boolean escapeToEndPossibleF(int x, int y, BitSet alreadyVisit,
			HMPPHeuristicsNet result) {
		HNSet outputSetX, outputSetY = new HNSet();
		//double max, min, minh;
		boolean escapeToEndPossible;
		int minNum;

		//          [A B]
		// X        [C]     ---> Y
		//          [D B F]

		// build subset h = [A B C D E F] of all elements of outputSetX
		// search for minNum of elements of min subset with X=B as element: [A B] , minNum = 2

		outputSetX = result.getOutputSet(x);
		outputSetY = result.getOutputSet(y);

		HNSubSet h = new HNSubSet();
		minNum = 1000;
		for (int i = 0; i < outputSetX.size(); i++) {
			HNSubSet outputSubSetX = new HNSubSet();
			outputSubSetX = outputSetX.get(i);
			if ((outputSubSetX.contains(y)) && (outputSubSetX.size() < minNum)) {
				minNum = outputSubSetX.size();
			}
			for (int j = 0; j < outputSubSetX.size(); j++) {
				h.add(outputSubSetX.get(j));
			}
		}

		if (alreadyVisit.get(x)) {
			return false;
		} else if (x == y) {
			return false;
		} else if (outputSetY.size() < 0) {
			// y is an eEe element
			return false;
		} else if (h.size() == 0) {
			// x is an eEe element
			return true;
		} else if (h.contains(y) && (minNum == 1)) {
			// x is unique connected with y
			return false;
		} else if (events.get(x).getEventType().equals(events.get(y).getEventType())) {
			// even here the we are in the same event!
			return false;
		} else {
			// iteration over OR-subsets in outputSetX
			for (int i = 0; i < outputSetX.size(); i++) {
				HNSubSet outputSubSetX = new HNSubSet();
				outputSubSetX = outputSetX.get(i);
				escapeToEndPossible = false;
				for (int j = 0; j < outputSubSetX.size(); j++) {
					int element = outputSubSetX.get(j);
					BitSet hulpAV = (BitSet) alreadyVisit.clone();
					hulpAV.set(x);
					if (escapeToEndPossibleF(element, y, hulpAV, result)) {
						escapeToEndPossible = true;
					}

				}
				if (!escapeToEndPossible) {
					return false;
				}
			}
			return true;
		}
	}
	
	
	/**
	 * Method to know if it has already build the basic relations
	 * 
	 * @return 
	 */
	public boolean getBasicRelationsMade() {
		return basicRelationsMade;
	}
	
	
	/**
	 * Method to set if it has already build the basic relations
	 * 
	 * @param val the new value
	 */
	public void setBasicRelationsMade(boolean val) {
		basicRelationsMade = val;
	}
	
	
//	/**
//	 * This method mines and returns the cost of the hypothesis built with the
//	 * given parameters.
//	 * 
//	 * The returned value is build considering the MDL approach and is
//	 * calculated as follows:
//	 * 
//	 *    L(h) + L(D|h)
//	 * 
//	 * where L(h) is the "size" of the network h and L(D|h) is the fitness of
//	 * the not processed observations.  
//	 * 
//	 * @param dependencyThreshold
//	 * @param positiveObservationsThreshold
//	 * @param relativeToBestThreshold
//	 * @param andThreshold
//	 * @param L1LThreshold length one loops threshold
//	 * @param L2LThreshold length two loops threshold
//	 * @param LDThreshold long distance threshold
//	 * @param useLongDistanceDependency
//	 * @param useAllConnectedHeuristics
//	 * @return the hypothesis cost
//	 */
//	private Double getHFromParam(
//			Double dependencyThreshold, 
//			Double positiveObservationsThreshold, 
//			Double relativeToBestThreshold,
//			Double andThreshold,
//			Double L1LThreshold,
//			Double L2LThreshold,
//			Double LDThreshold,
//			boolean useLongDistanceDependency,
//			boolean useAllConnectedHeuristics) {
//		
//		Double[] data = {0., 0.};
//		Double networkHypCost = 0.;
//		
//		HMPPParameters p = getParameters();
//		p.setDependencyThreshold(dependencyThreshold);
//		p.setPositiveObservationsThreshold(positiveObservationsThreshold.intValue());
//		p.setRelativeToBestThreshold(relativeToBestThreshold);
//		p.setAndThreshold(andThreshold);
//		p.setL1lThreshold(L1LThreshold);
//		p.setL2lThreshold(L2LThreshold);
//		p.setLDThreshold(LDThreshold);
//		p.setUseLongDistanceDependency(useLongDistanceDependency);
//		p.setUseAllConnectedHeuristics(useAllConnectedHeuristics);
//		
//		HMPPHeuristicsNet result = makeHeuristicsRelations(log, p);
//		data[0] = new Double(result.getNetworkSize());
//		
//		HMPPHeuristicsNet[] population = new HMPPHeuristicsNet[1];
//		population[0] = result;
//		
//		DTContinuousSemanticsFitness fitnessContinuousSemantics = new DTContinuousSemanticsFitness(log);
//		fitnessContinuousSemantics.calculate(population);
//		Double fitness = population[0].getFitness();
//		
////		DTImprovedContinuousSemanticsFitness fitnessImprovedContinuousSemantics = new DTImprovedContinuousSemanticsFitness(log);
////		fitnessImprovedContinuousSemantics.calculate(population);
////		Double fitness = population[0].getFitness();
//		
//		data[1] = fitness;
//		
//		networkHypCost = (data[0] / greatestNetworkSize) + (1 - data[1]);
////		dbg("      biggest: " + greatestNetworkSize);
////		dbg("         this: " + result.hashCode() +" -- "+ data[0]);
////		dbg("      L(h) = " + (data[0] / greatestNetworkSize) + "  ;  L(D|h) = " + (1 - data[1]));
////		dbg("===================");
////		dbg("	dep thr: " + dependencyThreshold); // dependencyThreshold
////		dbg("	pos obs: " + positiveObservationsThreshold); // positiveObservationsThreshold
////		dbg("	rel bst: " + relativeToBestThreshold); // relativeToBestThreshold
////		dbg("	and thr: " + andThreshold); // andThreshold
//		dbg("cost: " + networkHypCost + " -- size: " + data[0] + " -- hash: "+ result.hashCode());
//		
//		return networkHypCost;
//	}
//	
//	
//	/**
//	 * Shortcut for the same method, with the parameters as array
//	 * 
//	 * @param indexes indexes to use for the call
//	 * @param variations values to be summed to each index before the call
//	 * @param discretizedParameters parameter values
//	 * @param useLongDistanceDependency
//	 * @param useAllConnectedHeuristics
//	 * @param useLoops
//	 * @return the hypothesis cost 
//	 */
//	public Double getHFromParam(
//			int[] indexes, 
//			int[] variations, 
//			ArrayList<Double[]> discretizedParameters, 
//			boolean useLongDistanceDependency,
//			boolean useAllConnectedHeuristics,
//			boolean useLoops) {
//		
//		/*
//		 *							| discr param	| indexes	| variations
//		 *--------------------------+---------------+-----------+-----------
//		 * dependency thresholds	| 0				| 0			| 0
//		 * positive observations	| 1				| 1			| 1
//		 * relative to best			| 2				| 2			| 2
//		 * AND threshold			| 3				| 3			| 3
//		 * length 1 loops			| 4				| 4			| 4
//		 * length 2 loops			| 5				| 5			| 5
//		 * long distance dep		| 6				| 6			| 6
//		 */
//		
//		Double l1loopThreshold = (useLoops)? discretizedParameters.get(4)[indexes[4] + variations[4]] : 0.0;
//		Double l2loopThreshold = (useLoops)? discretizedParameters.get(5)[indexes[5] + variations[5]] : 0.0;
//		Double ldThreshold = (useLongDistanceDependency)? discretizedParameters.get(6)[indexes[6] + variations[6]] : 0.0;
//		
//		return getHFromParam(
//				discretizedParameters.get(0)[indexes[0] + variations[0]], // dependencyThreshold
//				discretizedParameters.get(1)[indexes[1] + variations[1]], // positiveObservationsThreshold
//				discretizedParameters.get(2)[indexes[2] + variations[2]], // relativeToBestThreshold
//				discretizedParameters.get(3)[indexes[3] + variations[3]], // andThreshold
//				l1loopThreshold, // L1LThreshold length one loops threshold
//				l2loopThreshold, // L2LThreshold length two loops threshold
//				ldThreshold, // LDThreshold long distance threshold
//				useLongDistanceDependency, // useLongDistanceDependency
//				useAllConnectedHeuristics // useAllConnectedHeuristics
//				);
//	}
//	
//	
//	/**
//	 * Shortcut for the same method, with the parameters as array
//	 * 
//	 * @param indexes indexes to use for the call
//	 * @param discretizedParameters parameter values
//	 * @param useLongDistanceDependency
//	 * @param useAllConnectedHeuristics
//	 * @param useLoops
//	 * @return the hypothesis cost 
//	 */
//	public Double getHFromParam(
//			int[] indexes,  
//			ArrayList<Double[]> discretizedParameters, 
//			boolean useLongDistanceDependency,
//			boolean useAllConnectedHeuristics,
//			boolean useLoops) {
//		
//		int variations[] = new int[indexes.length];
//		Arrays.fill(variations, 0);
//		return getHFromParam(indexes, variations, discretizedParameters, 
//				useLongDistanceDependency, useAllConnectedHeuristics, useLoops);
//	}
	
	
	/**
	 * This method mines the network with the parameters setted in order to get
	 * the biggest network (whose value will be used to normalize the value
	 * between 0 and 1).
	 * 
	 * @param log
	 */
	private void calculateGreatestNetworkSize(LogReader log) {
		dbgStart();
		HMPPParameters p = new HMPPParameters();
		p.setDependencyThreshold(0.);
		p.setPositiveObservationsThreshold(1);
		p.setRelativeToBestThreshold(1.);
		p.setAndThreshold(0.);
		p.setL1lThreshold(0.);
		p.setL2lThreshold(0.);
		p.setLDThreshold(0.);
		p.setUseAllConnectedHeuristics(true);
		p.setUseLongDistanceDependency(true);
		
		HMPPHeuristicsNet greatestResult = makeHeuristicsRelations(log, p);
		greatestNetworkSize = Utils.calculateNetworkSize(greatestResult);
		
//		dbg(new Integer(log.getInstances().size()).toString());
//		dbg(p.toString());
		dbgEnd();
	}
	
	
	/**
	 * This method prints can be used to print some debug information.
	 * 
	 * @param msg the message text
	 */
	public void dbg(String msg) {
		if (DEBUG) {
			Throwable t = new Throwable();
			StackTraceElement[] stackTraceElements = t.getStackTrace();
			StackTraceElement ste = stackTraceElements[1];
			
			for (int i = 0; i <= CALLS_DEEP; i++) {
				System.out.print("    ");
			}
			
			java.util.Calendar calendar = java.util.Calendar.getInstance();
			java.util.Date now = calendar.getTime();
			java.sql.Timestamp currentTimestamp = new java.sql.Timestamp(now.getTime());
			
			String ste_getClassName = ste.getClassName();
			ste_getClassName = ste_getClassName.substring(ste_getClassName.lastIndexOf(".") + 1);
			System.out.print("[DBG] " + currentTimestamp.toString() + " " + ste_getClassName + "::" + ste.getMethodName());
			if (!msg.equals("")) {
				System.out.print(" -- " + msg);
			}
			System.out.println("");
			System.out.flush();
		}
	}
	
	
	/**
	 * This method helps keep track of the start of one method (must be inserted
	 * as first call into a method).
	 */
	private void dbgStart() {
		if (DEBUG_START_END) {
			Throwable t = new Throwable();
			StackTraceElement[] stackTraceElements = t.getStackTrace();
			StackTraceElement ste = stackTraceElements[1];
			
			for (int i = 0; i <= CALLS_DEEP; i++) {
				System.out.print("    ");
			}
			
			java.util.Calendar calendar = java.util.Calendar.getInstance();
			java.util.Date now = calendar.getTime();
			java.sql.Timestamp currentTimestamp = new java.sql.Timestamp(now.getTime());
			
			String ste_getClassName = ste.getClassName();
			ste_getClassName = ste_getClassName.substring(ste_getClassName.lastIndexOf(".") + 1);
			System.out.println("[DBG:{] " + currentTimestamp.toString() + " " + ste_getClassName + "::" + ste.getMethodName());
			System.out.flush();
			CALLS_DEEP++;
		}
	}
	
	
	/**
	 * This method helps keep track of the end of one method (must be inserted
	 * as last call into a method, just before the possible return statement).
	 */
	private void dbgEnd() {
		if (DEBUG_START_END) {
			Throwable t = new Throwable();
			StackTraceElement[] stackTraceElements = t.getStackTrace();
			StackTraceElement ste = stackTraceElements[1];
			
			for (int i = 0; i < CALLS_DEEP; i++) {
				System.out.print("    ");
			}
			
			java.util.Calendar calendar = java.util.Calendar.getInstance();
			java.util.Date now = calendar.getTime();
			java.sql.Timestamp currentTimestamp = new java.sql.Timestamp(now.getTime());
			
			String ste_getClassName = ste.getClassName();
			ste_getClassName = ste_getClassName.substring(ste_getClassName.lastIndexOf(".") + 1);
			System.out.println("[DBG:}] " + currentTimestamp.toString() + " " + ste_getClassName + "::" + ste.getMethodName());
			if (CALLS_DEEP == 0) {
				System.out.println("");
			}
			System.out.flush();
			CALLS_DEEP--;
		}
	}


	/**
	 * @param key
	 * @param parametersCosts
	 */
	public void addParametersCosts(String key, ArrayList<Double> parametersCosts) {
		if (this.parametersCosts == null) {
			this.parametersCosts = new HashMap<String, ArrayList<Double>>();
		}
		this.parametersCosts.put(key, parametersCosts);
	}
	
	
	/**
	 * @return
	 */
	public HashMap<String, ArrayList<Double>> getParametersCosts() {
		return parametersCosts;
	}


	/**
	 * @param key
	 * @return
	 */
	public ArrayList<Double> getParametersCosts(String key) {
		return parametersCosts.get(key);
	}
	
	
	public LogReader getLogReader() {
		return log;
	}
	
	
	public int getGreatestNetworkSize() {
		return greatestNetworkSize;
	}
	
	
	public HMPPParameters getLastUsedParameters() {
		return parameters;
	}


	public void setMaxPlateauStep(int max) {
		maxPlateauSteps = max;
	}


	public int getMaxPlateauStep() {
		return maxPlateauSteps;
	}

	
	public int getNumberOfSearchThread() {
		return numberOfSearchThread;
	}


	public void setNumberOfSearchThread(int numberOfSearchThread) {
		this.numberOfSearchThread = numberOfSearchThread;
	}
	
	public int getMaxExecutionSteps() {
		return maxSearchSteps;
	}
	
	public Notifier getNotifier() {
		return notifier;
	}
}

