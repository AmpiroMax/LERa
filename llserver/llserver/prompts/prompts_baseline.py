# look_prompt = """
# You are a helpful embodied assistant. You are helping with resolving errors in robot execution.
# During execution of last action you have encountered a problem. Robot failed executing last action.
# You are given a goal and a current plan. 
# Look at the scene and describe what you see. Provide a detailed description of the scene. 
# Describe objects and their positions. Describe robot arm and its position. Describe gripper, its position and its contents.
# HINT: robot may accidently drop objects during gripper movement.

# Successfully executed actions:
# [success_actions]

# Current plan:
# [current_plan]

# Observation:
# <image>

# Current goal:
# [goal]
# """

# ------------------------------------------------------------------------------------------------
# Pybullet prompts
# ------------------------------------------------------------------------------------------------

replan_prompt = """
You are a helpful embodied assistant. You are helping with resolving errors in robot execution.
During execution of last action you have encountered a problem. Robot failed executing last action.
You are given a goal and a current plan and error resolving ideas. 
Predict new plan to achieve the goal.
HINT: robot may accidently drop objects during gripper movement.

Action description:
locate('object') - function that give you position of the object in the scene. Allows to move robot arm to the object in order to pick it up or place_on_top_of another object on it.
pick('object') - function that pick up an object. Can be used only if object location is known. Thus you must use locate('object') before pick('object').
place_on_top_of('object') - function that put something from gripper on top of the object, which name is provided to function. Example: place_on_top_of('blue block') - put anything from gripper on top of the blue block.
done() - function that indicate that the task is completed.
Each of locate, pick and place_on_top_of actions takes one argument - object name. You can not use any other arguments or multiple objects.

Strong advice: 
- always use locate('object') before pick('object')
- always use locate('object') before place_on_top_of('object')
- pick('object') can be used only if gripper is empty and object is located
- place_on_top_of('object') can be used only if gripper is NOT empty and object is located

Answer format:
You must answer using actions from avaliable_actions. Pay attention on action description.
Do not use any other actions and formatting. Your output must be a list of actions.
Do not coment on your answer. Do not explain it. Return only new plan as a list of actions.
Separate actions by comma. Do not use any other formatting.

Answer examples:
["locate('cyan block')", "pick('cyan block')", "locate('blue block')", "place_on_top_of('blue block')", "done()"]
["locate('cyan block')", "pick('cyan block')", "locate('red bowl')", "place_on_top_of('cyan block')", "done()"]
["locate('blue block')", "place_on_top_of('blue block')", "done()"]

Successfully executed actions:
[success_actions]

Current plan:
[current_plan]

Avaliable actions:
[available_actions]

Current goal:
[goal]
"""

# ------------------------------------------------------------------------------------------------
# Alfred prompts
# ------------------------------------------------------------------------------------------------


replan_prompt_alfred_2 = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. You are able to take one of this actions at a time [move, slice, open, close, put, pick_up, turn_on, turn_off]. Imagine, you are a robot operating in a house. I can ask you to do various tasks and you're gonna tell me the sequence of actions you would do to accomplish your task.
### Human: What is correct plan?
Current goal: 
pick up the alarm clock and turn on the lamp
Current plan: 
turn_on("desk lamp")
Description: 
1. The first step of the current plan has failed because the lamp is already turned on.
2. The goal is achived, no need to do anything.
3. The updated plan would be: done().
4. This updated plan will still accomplish the goal of looking at the clock under the lamp but without trying to turn the lamp on again, which is already turned on.
Available actions: 
[move, slice, open, close, put, pick_up, turn_on, turn_off, done()]
### Assistant: 1. done().
### Human: What is correct plan?
Current goal: 
carry an alarm clock while turning on a lamp
Current plan: 
turn_on("desk lamp")
Description: 
1. The first step of the current plan has failed because the lamp is already turned on.
2. The goal is achived, no need to do anything.
3. The updated plan would be: done().
4. This updated plan will still accomplish the goal of looking at the clock under the lamp but without trying to turn the lamp on again, which is already turned on.
Available actions: 
[move, slice, open, close, put, pick_up, turn_on, turn_off, done()]
### Assistant: 1. done().
### Human: What is correct plan?
Current goal: 
place a microwaved apple slice inside of the garbage can
Current plan: 
open("microwave"), put("microwave"), close("microwave"), turn_on("microwave"), turn_off("microwave"), open("microwave"), pick_up("apple sliced"), close("microwave"), move("garbage can"), put("garbage can")
Description: 
1. The first step of the current plan has failed because the microwave oven is already open.
2. The next step would be to put apple slice in the microwave and close the microwave door, since the robot cannot open it any further.
3. The updated plan would be: put("microwave"), close("microwave"), turn_on("microwave"), turn_off("microwave"), open("microwave"), pick_up("apple sliced"), close("microwave"), move("garbage can"), put("garbage can")
4. This updated plan will still accomplish the goal of placing a microwaved apple slice inside of the garbage can, but without trying to open the microwave door again, which is already open.
Available actions: 
[move, slice, open, close, put, pick_up, turn_on, turn_off, done()]
### Assistant: 1. put("microwave"), 2. close("microwave"), 3. turn_on("microwave"), 4. turn_off("microwave"), 5. open("microwave"), 6. pick_up("apple sliced"), 7. close("microwave"), 8. move("garbage can"), 9. put("garbage can"), 10. done().
### Human: What is correct plan?
Current goal: 
place a heated cup on a counter
Current plan: 
open("microwave"), put("microwave"), close("microwave"), turn_on("microwave"), turn_off("microwave"), open("microwave"), pick_up("cup"), close("microwave"), move("counter"), put("counter")
Description: 
1. The first step of the current plan has failed because the microwave oven is already open.
2. The next step would be to put cup in the microwave and close the microwave door, since the robot cannot open it any further.
3. The updated plan would be: put("microwave"), close("microwave"), turn_on("microwave"), turn_off("microwave"), open("microwave"), pick_up("cup"), close("microwave"), move("counter"), put("counter")
4. This updated plan will still accomplish the goal of placing a heated cup on a counter, but without trying to open the microwave door again, which is already open.
Available actions: 
[move, slice, open, close, put, pick_up, turn_on, turn_off, done()]
### Assistant: 1. put("microwave"), 2. close("microwave"), 3. turn_on("microwave"), 4. turn_off("microwave"), 5. open("microwave"), 6. pick_up("cup"), 7. close("microwave"), 8. move("counter"), 9. put("counter"), 10. done().
### Human: What is correct plan?
Current goal: 
place a chilled bowl in a cabinet
Current plan: 
open("fridge"), put("fridge"), close("fridge"), open("fridge"), pick_up("bowl"), close("fridge"), move("cabinet"), put("cabinet")
Description: 
1. The first step of the current plan has failed because the fridge is already open.
2. The next step would be to put bowl in the fridge and close the fridge door, since the robot cannot open it any further.
3. The updated plan would be: put("fridge"), close("fridge"), open("fridge"), pick_up("bowl"), close("fridge"), move("cabinet"), put("cabinet")
4. This updated plan will still accomplish the goal of placing a chilled bowl in a cabinet, but without trying to open the fridge door again, which is already open.
Available actions: 
[move, slice, open, close, put, pick_up, turn_on, turn_off, done()]
### Assistant: 1. put("fridge"), 2. close("fridge"), 3. open("fridge"), 4. pick_up("bowl"), 5. close("fridge"), 6. move("cabinet"), 7. put("cabinet"), 8. done(). 
### Human: What is correct plan?
Current goal: 
put a chilled tomato in the microwave
Current plan: 
open("fridge"), put("fridge"), close("fridge"), open("fridge"), pick_up("bowl"), close("fridge"), move("cabinet"), put("cabinet")
Description: 
1. The first step of the current plan has failed because the fridge is already open.
2. The next step would be to put tomato in the fridge and close the fridge door, since the robot cannot open it any further.
3. The updated plan would be: put("fridge"), close("fridge"), open("fridge"), pick_up("tomato"), close("fridge"), move("microwave"), open("microwave"), put("microwave"), close("microwave")
4. This updated plan will still accomplish the goal of placing a chilled bowl in a cabinet, but without trying to open the fridge door again, which is already open.
Available actions: 
[move, slice, open, close, put, pick_up, turn_on, turn_off, done()]
### Assistant: 1. put("fridge"), 2. close("fridge"), 3. open("fridge"), 4. pick_up("tomato"), 5. close("fridge"), 6. move("microwave"), 7. open("microwave"), 8. put("microwave"), 9. close("microwave"), 10. done().
### Human: What is correct plan?
Current goal: 
[goal]
Current plan: 
[current_plan]
Available actions: 
[move, slice, open, close, put, pick_up, turn_on, turn_off, done()]
Action description:
You are able to take one of this actions at a time [move, slice, open, close, put, pick_up, turn_on, turn_off]
move('object') - move towards position of the object
slice('object') - slice the object. May be used only with objects that can be sliced. (e.g. tomato, apple, etc.)
open('object') - open the object. May be used only with objects that can be opened. (e.g. fridge, drawer, microwave, etc.)
close('object') - close the object. May be used only with objects that can be closed. (e.g. fridge, drawer, microwave, etc.)
put('object') - put everything you are holding on top of the object
pick_up('object') - pick up the object
turn_on('object') - turn on the object. May be used only with objects that can be turned on. (e.g. tv, lamp, microwave, etc.)
turn_off('object') - turn off the object. May be used only with objects that can be turned off. (e.g. tv, lamp, microwave, etc.)
### Assistant:
"""


# ------------------------------------------------------------------------------------------------
# All prompts
# ------------------------------------------------------------------------------------------------

PROMPTS = {
    "replan": replan_prompt,
    "alfred_2": {
        "replan": replan_prompt_alfred_2
    }
}