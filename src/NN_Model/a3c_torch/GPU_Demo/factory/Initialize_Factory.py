drop schema efficiency_rl_ori;

create database Efficiency_RL_Ori;

use Efficiency_RL_Ori;
create table material(
    name varchar(256),
    storage int,
    Max_Store int,
    Max_Extra_Store int
) SELECT * FROM efficiency_rl.material;

create table material_price(
                         name varchar(256),
                         day int,
                         price float
) SELECT * FROM efficiency_rl.material_price;

create table producer(

                               Produce varchar(256),
                               Origin varchar(256),
                               Origin_Volume float
) SELECT * FROM efficiency_rl.producer;

