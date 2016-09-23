#ifndef WORLD_HPP
#define WORLD_HPP

#include "Q2Util.hpp"
#include "Q2Agent.hpp"
class Q2Agent;

//#include "QUtil.hpp"
//#include "QAgent.hpp"
//class QAgent;

class World{
	private:
		int _maxX;
		int _maxY;
		char* _worldString; // the 'view'
		vector<vector<WorldCell> > _world; //the map
		void _setMaxGoalDist();
		void _resetTraversalFlags();
		void _setRandomGoalLocation();
		void _setRandomAgentStartLocation(Agent& agent);
		void _restartAgent(Q2Agent* qagent, double terminalValue);
	public:
		World();
		World(int x, int y);
		~World();
		int GOAL_X; //goal location
		int GOAL_Y;

		//these return the max VALID x and y coordinates, not vector sizes
		int MaxX() const;
		int MaxY() const;
		//This may not belong as a world property, but makes it easily accessible
		double MaxDistanceToGoal;
		void ResizeWorld(int x, int y);
		//void PrintState(vector<Missile>& missiles, DiscreteQ2Agent& qagent);
		void InitializeWorld(vector<Missile>& missiles, Agent& agent);
		void InitRandomMissiles(vector<Missile>& missiles, int numMissiles);
		void Update(vector<Missile>& missiles, Q2Agent* agent, double timeStep);
		bool IsValidPosition(int x, int y) const;
		void Draw(vector<Missile>& missiles, Agent& agent);
		WorldCell& GetCell(int x, int y);
		const WorldCell& GetCell(int x, int y) const;
};

#endif

