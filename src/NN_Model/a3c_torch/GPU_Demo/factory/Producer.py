class Productor(object):
    def __init__(self, origin, origin_volume, product, product_volume, low_cost, max_times):
        self.origin = origin
        self.origin_volume = origin_volume
        self.product = product
        self.product_volume = product_volume
        self.lowest_maintain_cost = low_cost
        self.max_action_in_1_step = max_times

    def produce(self, times):

