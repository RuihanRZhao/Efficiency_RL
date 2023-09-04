# Efficiency_RL

## Documentation
Click [here](https://ruihanrzhao.github.io/Efficiency_RL/index.html) to the documentation.

## Requirement 需求

| name     | version | require |
|----------|---------|---------|
| python   | 3.10    | Must    |
| anaconda | 23.3.1  | Must    |
| cuda     |         | Flex    |

## Install Instructions 安装指南
### anaconda
##### Install the GUI version (best for new to Python)<br>安装图形操作界面 (新手友好)
官网下载: https://www.anaconda.com/download/ <br>
Official Website download: https://www.anaconda.com/download/

##### Install by terminal<br>利用终端安装
_[中文指南](https://zhuanlan.zhihu.com/p/397096022)<br>
[English Instruction](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)_

### Python
Official Website 官网: https://www.python.org/

### Cuda
To be complete

## Environment Config 环境配置 
To be complete

## Release 发行
Please check link below
请查看下方的链接<br>
[Release](https://github.com/RuihanRZhao/Efficiency_RL/tree/master/release)

## mySQL Server Structure 服务器结构
```mysql
use Factory;

create table Factory.Material(
    un_id           varchar(255)    primary key     unique,
    name            varchar(255),
    inventory       int,
    inventory_cap   int,
    cache           int,
    cache_cap       int
);

create table Factory.Price(
    un_id           varchar(255),
    date            datetime,
    price           float
);

create table Factory.Producer(
    un_id               varchar(255),
    Material_id         varchar(255),
    Material_amount     int,
    daily_low_cost      float,
    daily_produce_cap   int
);
```
