#ifndef Q2_AGENT
#define Q2_AGENT

#include "../ANN/MultilayerNetwork.hpp"
#include "Q2Util.hpp"
#include "World.hpp"
#include "unistd.h" //these linux headers could all be eliminated. If the short-term stochastic training code works out, the code should be cleaned to remove fork()/execve() patterns
#include "sys/types.h"
#include "sys/wait.h"


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
great promise, convering to local qvalues for the last action performed; however, the other actions
were not convergent, so the policy of choosing the best next action was a choice among one really good
q-value estimate for the last action, and then 3 other noisier estimates.

Theoretically combining all the actions into a single network would solve this, since the looped-update
will update all action values, not just one. The architecture is to add two inputs to the network
for x and y velocity, and to have a single linear output. The "action" values are given solely by inputing
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
		int _totalEpisodes;
		double _lastEpochCollisionRate;
		int _lastEpochActionCount;
		double _lastEpochReward;
		int _t;
		double _eta;
		double _gamma;
		string _historyFilePath;
		fstream _outputFile;
		vector<vector<vector<double> > > _stateHistory; //likely only two states for now: t and t+1
		//in this implementation, the agent has only a single q network		
		MultilayerNetwork _qNet;
		double _experiencedQValue;
		//short-term memory for the agent
		vector<Experience> _batch;
		int _batchIndex;

		void _takeAction(Action nextAction);
		double _normalizedCosSim(double x1, double y1, double x2, double y2);
		double _cosSim(double x1, double y1, double x2, double y2);
		void _copyVec(vector<double>& v1, vector<double>& v2);
		double _dist(double x1, double y1, double x2, double y2);
		double _getStrikeLikelihood(const vector<Missile>& missiles);
		double _getMissileLikelihood(const vector<Missile>& missiles);
		double _nearestObstacleDist(const World* world);
		double _nearestObjectOnHeading(double headingX, double headingY, const World* world, const vector<Missile>& missiles);
		double _getCurrentRewardValue(const World* world, const vector<Missile>& missiles);
		void _updateCurrentState(const World* world, const vector<Missile>& missiles);

		void _recordExample(const vector<double>& state, double qTarget, double qEstimate, Action action);

		Action _getStochasticOptimalAction();
		Action _selectionMethod1();
		Action _selectionMethod2();
		double _absDiff(double d1, double d2);
		void _normalizeStateVector(const World* world, vector<double>& state);
		void _zeroMeanStateVector(const World* world, vector<double>& state);
		void _normalizeVector(vector<double>& vec);
		void _takeContinuousAction();
		void _deriveCurrentState(const World* world, const vector<Missile>& missiles);
		const vector<double>& _getPreviousState(Action action);
		const vector<double>& _getCurrentState(Action action);
		bool _isWallCollision(const World* world);
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
		void ResetEpoch();
		void PrintCurrentStateEstimates();
		const char* GetActionStr(int i);
		void LoopedUpdate(const World* world, const vector<Missile>& missiles);	
		//void DiscriminativeUpdate(const World* world, const vector<Missile>& missiles);
		//void MinibatchUpdate(const World* world, const vector<Missile>& missiles);
		//void OfflineUpdate(const World* world, const vector<Missile>& missiles);
		//void EpochalUpdate(const World* world, const vector<Missile>& missiles);
		void Update(const World* world, const vector<Missile>& missiles);
		Action CurrentAction;
};

#endif


