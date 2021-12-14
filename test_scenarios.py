## Serving as our testing script for testing scenarios 
# Note: Change the parameter env_name to the scenario you want to make. They can be found under PATH gfootball/scenarios/
# Created: Nov 30, 2021

import gfootball.env as env
from absl import flags 

#run to score on a close (open) goal, single person 
env1 = env.create_environment(env_name='test_scenario_1v1',render=True,representation='simple115') 
#train on simple115: able to render. pixels constraint was cap 

done = False

for i in range(500):
    env1.reset()

    done = False
    while not done:
        action = env1.action_space.sample() #sampling that is done for the model testing 
        observation,reward,done,info = env1.step(action) #done outside of this file 
        

    #for each action, tell us what the action was, observation, reward, and info 
        print(action)
        print(observation)
        print(reward)
        print("DONE_VAL: ", done)
        print(info)
