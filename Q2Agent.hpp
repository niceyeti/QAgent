#ifndef Q2_AGENT
#define Q2_AGENT

#include "../ANN/MultilayerNetwork.hpp"
#include "Q2Util.hpp"
#include "World.hpp"
#include "unistd.h" //these linux headers could all be eliminated. If the short-term stochastic training code works out, the code should be cleaned to remove fork()/execve() patterns
#include "sys/types.h"
#include "sys/wait.h"

//smaller values (3-5) have worked the best; larger values produce oscillations. The effect of a smaller number of memorized locations
//is to let only the most recent locations push the agent, hence pushing the opposite in a more orthogonal direct wrt the visited region's radius
#define NUM_MEMORIZED_LOCATIONS 10

class kvector{
	public:
		kvector()=delete;
		kvector(const vector<double>& state, double reward, char label);
		vector<double> xs; //x values (state vector) that occurred when this reward was received
		double r;		//some external reward
		char alpha;		//the label for this external reward event
};

class World;

//for minibatch implementations
class Experience{
	public:
		Experience();
		Action PerformedAction;
		vector<double> BatchedState;
		double QTarget;
		double QEstimate;
};

/*
A highly similar implementation to QAgent, but this agent uses only a single network to represent
all possible actions. This is just an experiment to see how the agent performs. The QAgent showed
great promise, converging to local qvalues for the last action performed; however, the other actions
were not convergent, so the policy of choosing the best next action was a choice among one really good
q-value estimate for the last action, and then 3 other noisier estimates.

Theoretically combining all the actions into a single network would solve this, since the looped-update
will update all action values, not just one. The architecture is to add two inputs to the network
for x and y velocity, and to have a single linear output. _recentLocationsThe "action" values are given solely by inputing
different values for the x and y velocity inputs, which in this case are just 1 or 0 (exclusively).

*/
//not much rhyme or reason as to how the class decomposition was done.
class Q2Agent{
	private:
		int _repetitionCounter;
		int _episodeCount;
		int _epochCount;
		vector<double> _currentActionValues; 
		double _epochReward;
		//the reward function coefficients		
		double _coefGoalCos, _coefVisitedCos, _coefCollisionProx;
		int _totalEpisodes;
		double _lastEpochCollisionRate;
		int _lastEpochActionCount;
		double _lastEpochReward;
		int _t;
		double _eta;
		double _gamma;
		double _totalExternalReward;
		vector<kvector> _kVectors;
		string _historyFilePath;
		fstream _outputFile;
		fstream _kVectorFile;
		fstream _rewardParamsFile;
		fstream _prototypeFile; //for storing state vectors reresenting terminal states (goal-reached, agent crashed, etc)
		vector<vector<vector<double> > > _stateHistory; //likely only two states for now: t and t+1
		//storage for the recent locations; note this gives agent access to global knowledge, but the assumption is that it is limited to its previous, relative n locations.
		//the agent thus has access to exponentially decreasing approximations of its recent locations, using an EMA of previous k locations
		vector<pair<double, double> > _recentLocations;
		int _locationRingIndex;
		//These are how the agent approximates its recent, approximate location; it remains to be seen which is better, something centroidish, or exponentially weighted, etc.
		pair<double, double> _locationEma;
		pair<double, double> _locationAvg;
		//in this implementation, the agent has only a single q network
		MultilayerNetwork _qNet;
		MultilayerNetwork _rewardApproximator;
		double _experiencedQValue;
		//short-term memory for the agent
		vector<Experience> _batch;
		int _batchIndex;

		map<char,Neuron> _alphaNeurons;
		void _takeAction(Action nextAction);
		double _normalizedCosSim(double x1, double y1, double x2, double y2);
		double _cosSim(double x1, double y1, double x2, double y2);
		void _copyVec(vector<double>& v1, vector<double>& v2);
		double _dist(double x1, double y1, double x2, double y2);
		double _getStrikeLikelihood(const vector<Missile>& missiles);
		double _getMissileLikelihood(const vector<Missile>& missiles);
		//double _nearestObstacleDist(const World* world);
		double _nearestObjectOnHeading(double headingX, double headingY, double posX, double posY, const World* world, const vector<Missile>& missiles);
		double _getCurrentRewardValue_Logistic(const World* world, const vector<Missile>& missiles);
		double _getCurrentRewardValue_Manual1(const World* world, const vector<Missile>& missiles);
		double _getCurrentRewardValue_Manual2(const World* world, const vector<Missile>& missiles);
		double _getCurrentRewardValue_Terminal(const World* world, const vector<Missile>& missiles);
		void _updateCurrentActionStates(const World* world, const vector<Missile>& missiles);
		double _updateExternalReward(const World* world, const vector<Missile>& missiles);
		void _updateLocationMemory();
		void _estimateSubsequentState(double xHeading, double yHeading, double destX, double destY, const World* world, const vector<Missile>& missiles, vector<double>& subsequentState);
		
		//experimental logging
		void _flushRewardVectors(bool clearVecs);
		void _storeRewardParams(const vector<double>& state, double totalReward);
		void _storeTerminalState(const vector<double>& state, double terminalValue);
		void _storeLabeledVector(const vector<double>& state, double terminalValue, fstream& outputFile);
		void _recordExample(const vector<double>& state, double qTarget, double qEstimate, Action action);

		Action _getStochasticOptimalAction();
		Action _searchForOptimalAction(const World* world, const vector<Missile>& missiles, const int searchHorizon);
		Action _selectionMethod1();
		Action _selectionMethod2();
		double _absDiff(double d1, double d2);
		void _normalizeStateVector(const World* world, vector<double>& state);
		void _zeroMeanStateVector(const World* world, vector<double>& state);
		void _normalizeVector(vector<double>& vec);
		void _takeContinuousAction();
		void _deriveActionStates(const World* world, const vector<Missile>& missiles, vector<vector<double>>& actionStates);
		const vector<double>& _getPreviousState(Action action);
		const vector<double>& _getCurrentState(Action action);
		bool _isWallCollision(const World* world);
		void _tokenize(const string &s, char delim, vector<string> &tokens);
	public:
		Q2Agent()=delete;
		Q2Agent(int initX, int initY);
		~Q2Agent();
		void PrintState();
		Agent agent;
		//some performance parameters
		double EpochGoalCount;
		double EpochActionCount;
		double EpochCollisionCount;
		double GoalResetThreshold;
		void StoreTerminalState(double terminalValue);
		void StoreRewardParams(const vector<double>& rewardParams, double reward);
		void ResetEpoch();
		void PrintCurrentStateEstimates();
		const char* GetActionStr(int i);
		void LoopedUpdate(const World* world, const vector<Missile>& missiles);
		void AverageUpdate(const World* world, const vector<Missile>& missiles);
		void AdvantageUpdate(const World* world, const vector<Missile>& missiles);
		//void DiscriminativeUpdate(const World* world, const vector<Missile>& missiles);
		//void MinibatchUpdate(const World* world, const vector<Missile>& missiles);
		//void OfflineUpdate(const World* world, const vector<Missile>& missiles);
		//void EpochalUpdate(const World* world, const vector<Missile>& missiles);
		void DirectApproximationUpdate(const World* world, const vector<Missile>& missiles);
		void LogisticRewardApproximationUpdate(const World* world, const vector<Missile>& missiles);
		void ClassicalUpdate(const World* world, const vector<Missile>& missiles);
		void SarsaUpdate(const World* world, const vector<Missile>& missiles);
		void Update(const World* world, const vector<Missile>& missiles);
		Action CurrentAction;
};

#endif


