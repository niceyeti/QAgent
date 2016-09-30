#include "Q2Agent.hpp"


int main(int argc, char** argv, char** env)
{
	int t = 0;
	string dummy;
	World world(120, 35);
	Q2Agent q2Agent(5,5);
	vector<Missile> missiles;

	srand(time(NULL));
	usleep(rand() % 50000);
	world.InitializeWorld(missiles, q2Agent.agent, 30);

	while(true){
		//tell the agent the current world state, letting it take some action
		//qAgent.OfflineUpdate(&world,missiles);
		//qAgent.MinibatchUpdate(&world,missiles);
		//qAgent.DiscriminativeUpdate(&world,missiles);
		//q2Agent.LoopedUpdate(&world,missiles);
		//qAgent.EpochalUpdate(&world,missiles);

		//Looping the Update() function: this heuristic works well in practice. Why?
		//Also that it is sort of invalid, since Update() assumes CurrentAction was the previously asserted action, which is no longer true after the first call.
		/*
		for(int i = 0; i < 1; i++){
			q2Agent.Update(&world,missiles);
		}
		*/
		
		q2Agent.Update(&world,missiles);
		//q2Agent.LoopedUpdate(&world,missiles);
		//q2Agent.ClassicalUpdate(&world,missiles);

		//Update and draw the world; this is just updating a model and displaying a view
		world.Update(missiles, &q2Agent, 1.0);
		//only draw and delay once we want to see the agent's behavior
		if(t > 100000){
			world.Draw(missiles, q2Agent.agent);
			q2Agent.PrintState();
			usleep(5000);
		}
		t++;
	}

	return 0;
}






