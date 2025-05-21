
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
        self.regret_val += self.bb.bb_machine[self.bb.best_machine] - self.bb.bb_machine[k]
        self.regrets.append(self.regret_val)

        

    def run_one_step(self):
        raise NotImplementedError

    def run(self,total_step):
        for i in range(total_step):
            k = self.run_one_step()
            self.count[k]+=1
            self.cal_regret(k)
            self.action.append(k)
            
            
            
'''
方法一：ε-贪心
使用ε作为一个概率，当大于概率值时，拉动估值最大的拉杆（利用过程），当小于概率值时，随机拉动杠杆（探索过程）
'''
class EGreedy(Solver):
    def __init__(self,bb,e = 0.01, init_pro = 1.0):
        super(EGreedy,self).__init__(bb)

        # 记录拉杆权重
        self.evaluation = np.array([init_pro]*self.bb.k)
        # 记录e值
        self.e = e
    def run_one_step(self):
        # 判断概率
        if np.random.random()>self.e:
            k = np.argmax(self.evaluation)
        else:
            k = np.random.randint(0,self.bb.k)
        # 更新权重值
        r = self.bb.get_score(k)
        self.evaluation[k] += 1./(self.count[k]+1)* (r-self.evaluation[k])
            
        return k
    
# 方法二：随时间衰减的e贪心算法
class EGreedy_time(Solver):
    def __init__(self,bb,e = 0.01, init_pro = 1.0):
        super(EGreedy_time,self).__init__(bb)
        self.total_counts = 0

        # 记录拉杆权重
        self.evaluation = np.array([init_pro]*self.bb.k)
        # # 记录e值
        # self.e = e
    def run_one_step(self):
        self.total_counts+=1
        # 判断概率
        if np.random.random()<(1/self.total_counts):
            k = np.random.randint(0,self.bb.k)
        else:
            k = np.argmax(self.evaluation)
        # 更新权重值
        r = self.bb.get_score(k)
        self.evaluation[k] += 1./(self.count[k]+1)* (r-self.evaluation[k])
            
        return k
    
'''
上置信界算法：A拉杆拉过很多次，B拉杆只拉过一次，该算法认为被拉过次数越少，这个拉杆越值得被探索，因此也引入了霍夫丁不等式中的不确定值的变量，随着次数增加，这个值降低
霍夫丁不等式：有多个随机变量，每个随机变量的期望=经验期望（总值除以次数）＋不确定值
'''

class UCB(Solver):
    def __init__(self, bb, coef, init_prob = 1.0):
        super(UCB, self).__init__(bb)
        # 总次数
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bb.k)
        self.coef = coef
    def run_one_step(self):
        self.total_count+=1
        ucb = self.estimates + self.coef * np.sqrt(np.log(self.total_count)/(2*(self.count+1)))
        k = np.argmax(ucb)
        r = self.bb.get_score(k)
        self.estimates[k] += 1. / (self.count[k]+1) * (r-self.estimates[k])
        return k


'''
汤普森采样：假定每根拉杆有一个概率，然后每次迭代根据实际拉杆的结果情况调整概率偏移，使其逐渐倾向于正确的概率分布，比如左边正确的多，右边少，那么就会倾向于左边

'''

class TS(Solver):
    def __init__(self, bb):
        super(TS,self).__init__(bb)
        self.a = np.ones(self.bb.k)
        self.b = np.ones(self.bb.k)
    def run_one_step(self):
        samples = np.random.beta(self.a, self.b)
        k = np.argmax(samples)
        r = self.bb.get_score(k)

        self.a[k]+=r
        self.b[k]+=(1-r)
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

def test(name):
    if name == "egreedy":
        # test
        np.random.seed(1)
        # 初始化老虎机
        k = 20
        bb = BernoulliBandit(k)
        e = 0.2
        egreedy = EGreedy(bb,e)
        egreedy.run(500)
        print('epsilon-贪婪算法的累积懊悔为：', egreedy.regret_val)
        plot_results([egreedy], ["EGreedy"])
    elif name == "egreedy-time":
        # test
        np.random.seed(1)
        # 初始化老虎机
        k = 20
        bb = BernoulliBandit(k)
        e = 0.2
        egreedy = EGreedy_time(bb,e)
        egreedy.run(500)
        print('衰减epsilon-贪婪算法的累积懊悔为：', egreedy.regret_val)
        plot_results([egreedy], ["EGreedy_time"])
    elif name == "UCB":
        np.random.seed(1)
        # 初始化老虎机
        k = 20
        bb = BernoulliBandit(k)
        # 初始化参数
        coef = 1
        ucb_test = UCB(bb, coef)
        ucb_test.run(500)
        print('上置信界算法的累积懊悔为：', ucb_test.regret_val)
        plot_results([ucb_test], ["UCB"])
    elif name=="ts":
        np.random.seed(1)
        # 初始化老虎机
        k = 20
        bb = BernoulliBandit(k)
        # 初始化参数
        coef = 1
        ts_test = TS(bb)
        ts_test.run(500)
        print('汤普森的累积懊悔为：', ts_test.regret_val)
        plot_results([ts_test], ["TS"])
    else:
        print("没有这个方法")

test("ts")
