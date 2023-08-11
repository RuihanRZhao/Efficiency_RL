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
        for i in self.origin:
            for m in mat_list:
                if m.name == i[0] and m.storage < i[1]:
                    raise ValueError("Material not enough.")
        if self.max_action_in_1_step < times:
            raise ValueError("Product time Overflow.")

        # produce
        for i in self.origin:
            for m in mat_list:
                if m.name == i[0]:
                    m.Update_Material_Stock(-i[1])
                if m.name == self.product:
                    m.Update_Material_Stock(times)

        return reward