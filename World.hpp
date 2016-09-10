#ifndef WORLD_HPP
#define WORLD_HPP

#include "QUtil.hpp"
#include "QAgent.hpp"

class QAgent;

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
		void _restartAgent(QAgent& qagent);
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
		//void PrintState(vector<Missile>& missiles, DiscreteQAgent& qagent);
		void InitializeWorld(vector<Missile>& missiles, Agent& agent);
		void InitRandomMissiles(vector<Missile>& missiles, int numMissiles);
		void UpdateWorld(vector<Missile>& missiles, QAgent& agent, double timeStep);
		bool IsValidPosition(int x, int y) const;
		void DrawWorld(vector<Missile>& missiles, Agent& agent);
		WorldCell& GetCell(int x, int y);
		const WorldCell& GetCell(int x, int y) const;
};

#endif

