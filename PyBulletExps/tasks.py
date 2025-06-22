task1 = {
    "name": "pick_n_place",
    "base_random_seed": 100,
    "goal": "put cyan block in red bowl",
    "obj_list": ['cyan block', 'gray block', 'pink block', 'brown block', 'red bowl'],
    "plan": """locate('cyan block')
pick('cyan block')
locate('red bowl')
place('red bowl')
done()""",
    "validation_rule": [('cyan block', 'red bowl')]
}

task2 = {
    "name": "pick_n_place_same_color",
    "base_random_seed": 200,
    "goal": "put blocks in bowls with same color",
    "obj_list": ['blue block', 'red block', 'blue bowl', 'red bowl'],
    "plan": """locate('blue block')
pick('blue block')
locate('blue bowl')
place('blue bowl')
locate('red block')
pick('red block')
locate('red bowl')
place('red bowl')
done()""",
    "validation_rule": [
        ('blue block', 'blue bowl'),
        ('red block', 'red bowl')
    ]
}

task3 = {
    "name": "stack_blocks_easy",
    "base_random_seed": 300,
    "goal": "put cyan block in red bowl and then put brown block on cyan block",
    "obj_list": ['cyan block', 'gray block', 'pink block', 'brown block', 'red bowl'],
    "plan": """locate('cyan block')
pick('cyan block')
locate('red bowl')
place('red bowl')
locate('brown block')
pick('brown block')
locate('cyan block')
place('cyan block')
done()""",
    "validation_rule": [('cyan block', 'red bowl'), ('brown block', 'cyan block')]
}

task4 = {
    "name": "stack_blocks_hard",
    "base_random_seed": 400,
    "goal": "stack all blocks on each other in the red bowl",
    "obj_list": ['blue block', 'red block', 'green block', 'red bowl'],
    "plan": """locate('blue block')
pick('blue block')
locate('red bowl')
place('red bowl')
locate('red block')
pick('red block')
locate('blue block')
place('blue block')
locate('green block')
pick('green block')
locate('red block')
place('red block')
done()""",
    "validation_rule": [
        ('blue block', 'red bowl'),
        ('red block', 'red bowl'),
        ('green block', 'red bowl')
    ]
}


task5 = {
    "name": "rotating_blocks",
    "base_random_seed": 500,
    "goal": "place blocks in bowls according to clockwise color rotation of blocks colors",
    "obj_list": ['blue block', 'red block', 'green block', 'blue bowl', 'red bowl', 'green bowl'],
    "plan": """locate('blue block')
pick('blue block')
locate('red bowl')
place('red bowl')
locate('red block')
pick('red block')
locate('green bowl')
place('green bowl')
locate('green block')
pick('green block')
locate('blue bowl')
place('blue bowl')
done()""",
    "validation_rule": [
        ('blue block', 'red bowl'),
        ('red block', 'green bowl'),
        ('green block', 'blue bowl')
    ]
}

task6 = {
    "name": "rotating_blocks_hard",
    "base_random_seed": 600,
    "goal": "place blocks in bowls according to two clockwise color rotations of blocks colors",
    "obj_list": ['blue block', 'red block', 'green block', 'blue bowl', 'red bowl', 'green bowl'],
    "plan": """locate('blue block')
pick('blue block')
locate('green bowl')
place('green bowl')
locate('red block')
pick('red block')
locate('blue bowl')
place('blue bowl')
locate('green block')
pick('green block')
locate('red bowl')
place('red bowl')
done()""",
    "validation_rule": [
        ('blue block', 'green bowl'),
        ('red block', 'blue bowl'),
        ('green block', 'red bowl')
    ]
}

task7 = {
    "name": "two_towers",
    "base_random_seed": 700,
    "goal": "build two towers with two blocks each: blue on red and yellow on green",
    "obj_list": ['blue block', 'red block', 'yellow block', 'green block'],
    "plan": """locate('blue block')
pick('blue block')
locate('red block')
place('red block')
locate('yellow block')
pick('yellow block')
locate('green block')
place('green block')
done()""",
    "validation_rule": [
        ('blue block', 'red block'),
        ('yellow block', 'green block')
    ]
}

task8 = {
    "name": "two_tower_hard",
    "base_random_seed": 800,
    "goal": "build two towers in bowls: blue on red, yellow on green. Tower should be located in bowl with the same color as the block on top of the tower",
    "obj_list": ['blue block', 'red block', 'yellow block', 'green block', 'blue bowl', 'yellow bowl'],
    "plan": """locate('red block')
pick('red block')
locate('blue bowl')
place('blue bowl')
locate('blue block')
pick('blue block')
locate('red block')
place('red block')
locate('green block')
pick('green block')
locate('yellow bowl')
place('yellow bowl')
locate('yellow block')
pick('yellow block')
locate('green block')
place('green block')
done()""",
    "validation_rule": [
        ('blue block', 'red block'),
        ('red block', 'blue bowl'),
        ('yellow block', 'green block'),
        ('green block', 'yellow bowl')
    ]
}

task9 = {
    "name": "pyramid_of_opposites",
    "base_random_seed": 900,
    "goal": "build a pyramid of blocks where top is contrasting to bottom block. You have blue, yellow, red blocks. Start with blue one.",
    "obj_list": ['blue block', 'yellow block', 'red block'],
    "plan": """locate('yellow block')
pick('yellow block')
locate('blue block')
place('blue block')
locate('red block')
pick('red block')
locate('yellow block')
place('yellow block')
done()""",
    "validation_rule": [
        ('red block', 'yellow block'),
        ('yellow block', 'blue block')
    ]
}

task10 = {
    "name": "pyramid_of_opposites_non_matching_bowl",
    "base_random_seed": 1000,
    "goal": "build a pyramid of blocks where top is contrasting to bottom block. You have blue, yellow, red blocks. Start with blue one. Build in a bowl that does not match any block color.",
    "obj_list": ['blue block', 'yellow block', 'red block', 'blue bowl', 'green bowl'],
    "plan": """locate('blue block')
pick('blue block')
locate('green bowl')
place('green bowl')
locate('yellow block')
pick('yellow block')
locate('blue block')
place('blue block')
locate('red block')
pick('red block')
locate('yellow block')
place('yellow block')
done()""",
    "validation_rule": [
        ('red block', 'yellow block'),
        ('yellow block', 'blue block'),
        ('blue block', 'green bowl')
    ]
}