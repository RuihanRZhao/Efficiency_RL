
def write(target, type, content):
    f = open(target, type)
    f.write(content)
    f.close()