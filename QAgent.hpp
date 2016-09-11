#ifndef Q_AGENT
#define Q_AGENT

#include "../ANN/MultilayerNetwork.hpp"
#include "QUtil.hpp"
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
An implementation of approximate q-learning for discrete state/action spaces,
in this case just a grid-world. This differs from tabular q-learning in that
a neural network is used to estimate Q-values instead of a huge table. This uses 
a discrete state/action representation as a personal tutorial, and extension to
continuous state spaces should be trivial once the discrete-world version works.

Neural nets for learning q-values are entirely defined by the user; some have 
a separate neural net for each action, and all generally have different architectures.
In other words, I have no idea which is best, this is just an attempted approximate
q-learning implementation for a very simple environment, as a self tutorial.

The architecture and update method is directly from Haykin, Neural Networks: A Comprehensive...
*/
//not much rhyme or reason as to how the class decomposition was done.
class QAgent{
	private:
		int _repetitionCounter;
		int _episodeCount;
		int _epochCount;
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
		vector<MultilayerNetwork> _qNets;
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
		void _deriveCurrentState(const World* world, const vector<Missile>& missiles, vector<double>& state);
		const vector<double>& _getPreviousState(Action action);
		const vector<double>& _getCurrentState(Action action);
		bool _isWallCollision(const World* world);
	public:
		QAgent()=delete;
		QAgent(int initX, int initY);
		~QAgent();
		void PrintState();
		void Train();
		Agent agent;
		//some performance parameters
		double EpochGoalCount;
		double EpochActionCount;
		double EpochCollisionCount;
		double GoalResetThreshold;
		void ResetEpoch();
		const char* GetActionStr(int i);
		void LoopedUpdate(const World* world, const vector<Missile>& missiles);
		void DiscriminativeUpdate(const World* world, const vector<Missile>& missiles);
		void MinibatchUpdate(const World* world, const vector<Missile>& missiles);
		void OfflineUpdate(const World* world, const vector<Missile>& missiles);
		void EpochalUpdate(const World* world, const vector<Missile>& missiles);
		void Update(const World* world, const vector<Missile>& missiles);
		//Action LastAction;
		Action CurrentAction;
};

#endif


