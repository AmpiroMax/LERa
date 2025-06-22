from typing import Tuple


class Subtask:
    """Represents a subtask.
     
    A subtask has 1 of 8 actions (PickupObject, PutObject, OpenObject, 
    CloseObject, ToggleObjectOn, ToggleObjectOff, GotoLocation), an object 
    that the action is being performed on, and optionally a receptacle 
    containing the object. For GotoLocation (navigational) action any object 
    (regular or receptacle) is considered as regular object.

    Attributes
    ----------
    action : str
        An action to execute in the AI2THOR simulator.
    obj : str
        The object on which the action is performed.
    recept : str, optional
        A receptacle containing the object.
    """

    def __init__(self, subtask_tuple: Tuple):
        self.obj = None
        self.recept = None
        self.action = None
        # If receptacle is present in subtask
        # (e.g. (Apple, Fridge, PickupObject))
        if len(subtask_tuple) == 3:
            self.obj, self.recept, self.action = subtask_tuple
        # Else, if there is no receptacle (e.g. (Apple, None, PickupObject))
        elif len(subtask_tuple) == 2:
            self.obj, self.action = subtask_tuple
        # Otherwise, corrupted subgoal
        else:
            # TODO: this needs to be properly handled
            print(f'Corrupted subtask: {subtask_tuple}')
            self.obj, self.action = ('Apple', 'PickupObject')

    def __str__(self):
        return f'({self.action}, {self.obj}, {self.recept})'
    
    def __eq__(self, other):
        if (
            hasattr(other, 'obj') and hasattr(other, 'recept')
            and hasattr(other, 'action') and self.obj == other.obj
            and self.recept == self.recept and self.action == other.action
        ):
            return True
        else:
            return False
