
'''
多臂老虎机
由k个老虎机，要判断我拉动哪个的效益最好。
我要执行n次，每轮我都计算一个分数，来表示悔恨值（我把每轮最好的结果当作我的目标值，当前实际结果与目标值的差值就是悔恨值）

过程（伪代码）：
for i in N:
    score_i = score_i-1+(A_i)/N_ai
'''
import numpy as np;
import matplotlib.pyplot as plt
# 老虎机
class BernoulliBandit():
    # k指的是拉杆个数
    def __init__(self,k):
        # 初始化k个获奖概率不同的老虎机
        self.bb_machine = np.random.uniform(size = k)
        # 记录概率最高的老虎机(记录索引值)
        self.best_machine = np.argmax(self.bb_machine)
        # 记录概率值
        self.best_machine_score = self.bb_machine[self.best_machine]
        # 记录k值
        self.k = k
    def get_score(self,i):
        if np.random.rand()<self.bb_machine[i]:
            return 1
        else:
            return 0
# 老虎机测试
# np.random.seed(1)
# k = 10
# bb = BernoulliBandit(k)
# print("老虎机获胜概率最大的机器是%d"%bb.best_mechine)
# print("概率是%.4f"%bb.best_machine_score)

# 解决方案
class Solver():
    def __init__(self,bb):
        # 记录老虎机
        self.bb = bb
        # 记录悔过值
        self.regret_val  = 0

        # 记录行为
        self.action = []
        # 记录每步的悔恨
        self.regrets = []
        # 每根拉杆的次数
        self.count = np.zeros(bb.k)

    def cal_regret(self,k):
        self.regret_val = self.bb.bb_machine[self.bb.best_machine] - self.bb.bb_machine[k]
        self.regrets.append(self.regret_val)

        

    def run_one_step(self):
        raise NotImplementedError

    def run(self,total_step):
        for i in range(total_step):
            k = self.run_one_step()
            self.cal_regret(k)
            self.action.append(k)
            self.count[k]+=1
            
            
'''
方法一：ε-贪心
使用ε作为一个概率，当大于概率值时，拉动估值最大的拉杆（利用过程），当小于概率值时，随机拉动杠杆（探索过程）
'''
class EGreedy(Solver):
    def __init__(self,bb,e = 0.01, init_pro = 1.0):
        super(EGreedy,self).__init__(bb)
        # 记录老虎机
        self.bb = bb
        # 记录拉杆权重
        self.evaluation = np.array([init_pro]*self.bb.k)
        # 记录e值
        self.e = e
    def run_one_step(self):
        # 判断概率
        if np.random.rand()>self.e:
            k = np.argmax(self.evaluation)
        else:
            k = np.random.randint(0,self.bb.k)
        # 更新权重值
        r = self.bb.get_score(k)
        self.evaluation[k] += 1./(self.count[k]+1)* (r-self.evaluation[k])
            
        return k
# 绘图函数
def plot_results(solvers, solver_names):
    """生成累积懊悔随时间变化的图像。输入solvers是一个列表,列表中的每个元素是一种特定的策略。
    而solver_names也是一个列表,存储每个策略的名称"""
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bb.k)
    plt.legend()
    plt.show()
# test
np.random.seed(1)
# 初始化老虎机
k = 10
bb = BernoulliBandit(k)
e = 0.01
egreedy = EGreedy(bb,e)
egreedy.run(5000)
print('epsilon-贪婪算法的累积懊悔为：', egreedy.regrets)
plot_results([egreedy], ["EGreedy"])



