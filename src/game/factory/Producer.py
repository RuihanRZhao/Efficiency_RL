import DB
from Material import Material
class Producer(object):
    def __init__(self, name, origin, low_cost, max_times):
        self.name =name
        self.origin = origin
        self.lowest_maintain_cost = low_cost
        self.max_action_in_1_step = max_times

    def __repr__(self):
        return ("\n(object: Producer, name: {}, \n       origin: {}, \norigin_volume: {}, lowest_maintain_cost: {}, "
                "max_action_in_1_step: {})").format(self.name, self.origin,
                                                    self.lowest_maintain_cost, self.max_action_in_1_step)

    def produce(self, times, mat_list: list[Material]):
        reward = 0

        # if too much times / origin not enough, start again
        for i in self.origin:
            for m in mat_list:
                if m.name == i[0] and m.storage < i[1]:
                    self.produce(int(input(
                        "Origin {} not enough: {} needed per unit\ntry again: \nInput: {}".format(i[0], i[1],times))))
        if self.max_action_in_1_step < times:
            self.produce(int(input(
                "Produce {} times of {} more than available: {} units\ntry again: \nInput: {}".format(times, self.name,
                                                                                           self.max_action_in_1_step,times))))

        # produce
        for i in self.origin:
            for m in mat_list:
                if m.name == i[0]:
                    m.Update_Material_Stock(-i[1])
                if m.name == self.name:
                    m.Update_Material_Stock(times)

        return reward

