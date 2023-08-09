import DB
import random
price_info = []

obj = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
o_p = [ 10,  12,  24,   5,  62,  30,  13, 170]
for d in range(1,366):
    for i in range(8):
        price_info.append([obj[i], d, round(o_p[i] * (0.5 + random.random()), 2)])



DB.Write_File("data/material_price.csv","csv",["name","day","price"],price_info)

