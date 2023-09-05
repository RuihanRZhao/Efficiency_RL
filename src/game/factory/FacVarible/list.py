class FacList:
    def __init__(self, _type: type = None):
        self.type = _type
        self.check_type(items)
        self.list : list[_type] = []


    def append(self, item):

    def check_type(self, _input: list | object):
        if type(_input) == list:
            for i in _input:
                if not type(i) == self.type: raise TypeError(
                    f"TypeError: Varible {i} is a {type(i)}, \n"
                    f"The list requires {self.type}"
                )
        else:
            if not type(_input) == self.type: raise TypeError(
                f"TypeError: Varible \n"
                f"\t{_input}\n "
                f"is a {type(_input)}, \n"
                f"The list requires {self.type}"
            )
