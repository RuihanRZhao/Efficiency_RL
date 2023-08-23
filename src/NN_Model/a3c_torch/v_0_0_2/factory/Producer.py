from . import Fac_Value as FV
from .Material import Material


class Producer(object):
    def __init__(self, name, origin, low_cost, max_times):
        self.product = name
        self.origin = origin
        self.lowest_maintain_cost = low_cost  # not in use
        self.max_action_in_1_step = max_times

    def __repr__(self):
        return ("\n(object: Producer, name: {}, \n       origin: {}, \norigin_volume: {}, lowest_maintain_cost: {}, "
                "max_action_in_1_step: {})").format(self.name, self.origin,
                                                    self.lowest_maintain_cost, self.max_action_in_1_step)

    def produce(self, times, mat_list: list[Material]):
        reward = 0
        # if too much times / origin not enough, start again
        if times < 0:
            reward -= FV.Cost_Do_Nothing * 2
        for i in self.origin:
            for m in mat_list:
                if m.name == i[0] and m.storage < i[1] * int(times):
                    reward -= FV.Cost_Do_Nothing * 1
                    # raise ValueError("Material not enough.")
        if self.max_action_in_1_step < int(times):
            # raise ValueError("Product time Overflow.")
            reward -= FV.Cost_Do_Nothing * 1
        if reward < 0:
            return reward, 0
        else:
            reward += FV.Cost_Do_Nothing * 3
        # produce
        for m in mat_list:
            for i in self.origin:
                if m.name == i[0]:
                    m.Update_Material_Stock(-i[1] * int(times))
            if m.name == self.product:
                m.Update_Material_Stock(int(times))

        return reward, 0
