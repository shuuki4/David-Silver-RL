# David Silver - Reinforcement Learning

## Lecture 1 : Introduction to Reinforcement Learning

#### Rewards
- $$$R_t$$$ 로 표기 : agent가 time t에 얼마나 잘 하고 있는지?
- agent의 목표는 $$$\sum_t R_t$$$를 maximize 하는 것
- ** Reward Hypothesis ** : Goal = Maximisation of expected cumulative reward

#### General System
- Agent는 time t에 observation $$$O_t$$$를 보고, reward $$$R_t$$$를 받는다. 이 정보를 통해 Action $$$A_t$$$를 행한다.
- History $$$H_t$$$ : sequence of observations, actions, rewards
	 - = every observable variables till time t
- State $$$S_t$$$ : 다음에 무엇이 일어날지 예상하기 위해 필요한 정보
	- any function of the history, $$$ f(H_t) $$$
	- 즉, 단순하게 last observation만 보는 것도 일종의 'state' 라고 할 수 있다
	- environment state $$$S_t^e$$$ : time t에서 environment의 representation
		- usually not visible to agent. visible하다고 하더라도 필요하지 않은 정보가 많을 수도 있음
	- agent state $$$ S_t^a$$$ : agent의 internal representation / any information used by RL algorithm

###### Information State (Markov State)
- all useful information from the history
- state $$$S_t$$$가 Markov라는 것은 $$$P[S_{t+1}|S_t] = P[S_{t+1}|S_1, S_2, ..., S_t]$$$ 라는 것
	- 현재 State $$$S_t$$$만 알고 있으면, 현재까지 쌓아온 History는 필요 없다는 것
	- environment state $$$S_t^e$$$ 는 정의에 의해 markov임. History H도 정의에 의해 markov임.

#### Environments
- Fully Observable Environments
	- agent가 environment state를 모두 관찰할 수 있음 : agent state = environment state = information state
	- 이걸 Markov Decision Process (MDP) 라고 함.
- Partially Observable Environments
	- agent가 environment를 indirect하게 보고, 일부 정보만 가지고 있을 경우
	- 실제로 대부분의 경우는 partially observable일 것
	- agent state != environment state이다 :'Partially observable Markov decision process (POMDP)'
		- agent가 자기의 state를 실제로 만들어야함.
		- Ex : Complete History, 'Beliefs' of environment state, RNN...

#### RL Agents
- RL agent는 다음과 같은 component중 한개 이상을 가지고 있어야함
	1. 'policy' : 어떻게 action을 결정하는가?
	2. 'value function' : state나 action이 얼마나 좋은지 어떻게 평가하는가?
	3. 'model' : agent가 environment를 어떻게 표현하는가?

###### Policy
- Map from state to action
	- deterministic : $$$ a = \pi(s) $$$
	- probabilistic : $$$ \pi(a|s) = P[A=a|S=s] $$$

###### Value function
- Prediction of future reward
- ex ) expected future reward의 식
	$$$ v_{\pi}(S) = E_{\pi}[R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + ... | S_t = s] $$$

###### Model
- environment가 다음 step에 무엇을 할지 예측함
- transitions : 다음 state를 예측한다. (helicopter의 경우, 동역학적인 면에서 나의 현재 상황을 예측)

###### Agents
- Value based (policy는 implicit이고, value function만 가지고 판단)
- Policy based (value function이 없고, policy만 가지고 함)
- Actor Critic (both policy & value function)
- Model Free (No Model : do not try to understand environment)
- Model Based (Use Model)

#### Atari Example

##### Reinforcement Learning View
- 실제로 emulator가 어떻게 돌아가는지는 모르지만, Observation (화면) 과 Reward (점수)를 가지고 나의 action을 결정함

##### Planning View
- Game의 Rule이 알려져있는 경우, 이 경우에는 simulator를 이용하여 tree search 등을 이용하여 action을 plan할 수 있음

#### Exploration & Exploitation
- Reinforcement Learning은 기본적으로 Trial & Error
- Agent는 good policy를 찾아야함과 동시에, 여러가지 action을 해보면서 경험도 해야함 : exploration과 exploitation의 균형을 잘 맞춰야 함

## Chapter 2 : Markov Decision Process

### MDP
- Environment for reinforcement learning
- environment가 'fully observable' 할때만을 고려한다
- Optimal control은 continuous MDP로 변환 가능하고, Partially observable problem도 MDP로 conversion이 가능하다

#### Markov Processes
- State Transition Matrix : $$$ P_{ss'} = P[S_{t+1}=s' | S_t = s] $$$ 이고, P = [P_{ij}]. row stochastic matrix인 matrix로 나타낼 수 있고, 이걸 state transition matrix라고 함.
- ** Markov Process ** (Markov Chain) : sequence of random states with the markov property
	- denote as (S, P) : S = set of states, P = state transition probability matrix

#### Markov Reward Processes
- tuple (S, P, R, $$$\gamma$$$)
	- S : set of states, P : state transition matrix
	- R : Reward Function, $$$R_s = E[R_{t+1}|S_t=s]$$$ : 내가 현재 state에 있을 때, 다음에 받을 reward의 expectation
	- $$$\gamma$$$ : discount factor
- return $$$G_t$$$ : total *discounted* reward from timestep t (of specific sample)
	- $$$G_t = R_{t+1} + \gamma R_{t+2} + .. = \sum_{k=0} \gamma^k R_{t+k+1} $$$
	- notation이 time이 하나씩 밀려있는데, 어떤 state를 '벗어날때' reward가 주어진다고 생각하면 이해됨. 물론 time을 하나씩 땡겨서 생각해도 결론적으로는 무방하다
- Discount Factor가 있는 이유 :
	1. mathematically convenient (발산의 위험을 없애므로)
		- infinite reward 등을 없앰
	2. uncertainty를 반영
- ** Value Function ** v(s) : long-term value of state s
	- MRP에서 value function은 reward의 expectation in specific state
	- $$$ E[G_t | S_t = s] $$$

##### Bellman Equation
$$ v(s) = E[G_t | S_t = s]$$
$$ = E[R_{t+1}+\gamma R_{t+2}+.... | S_t = s] $$
$$ = E[R_{t+1} + \gamma G_{t+1} | S_t = s ]$$
$$ = E[R_{t+1} + \gamma v(S_{t+1})] $$
where $$$ \gamma v(S_{t+1}) $$$ = discounted value of successor state
$$ = R_s + \gamma \sum_{s'} P_{ss'} v(s')$$
- Bellman Equiation을 행렬로 표현할 수도 있음. v와 R를 각각 value function / reward function의 column vector라고 하면,
$$ v = R + \gamma Pv $$
As a result,
$$ v = (I-\gamma P)^{-1}R $$
- 따라서 O(n^3) (n= number of states)에 value vector를 구할 수 있고, 이렇게 구하지 않더라도 large problem에 대해서는 iterative method를 이용할 수도 있음
	- DP, Monte-Carlo evaluation, TD learning...

#### Markov Decision Process
- tuple (S, A, P, R, $$$\gamma$$$)
	- MRP에 A (finite set of actions) 가 추가된 형태
	- 이제 P를 $$$P_{ss'}^a = P[S_{t+1}=s' | S_t = s, A_t = a]$$$로 생각하고, reward function R도 $$$R_s^a = E[R_{t+1} | S_t=s , A_t=a] $$$ 라고 생각한다
- Policy : distribution of actions, given states
	- $$$ \pi(a|s) = P[A_t=a | S_t = s] $$$
	- MDP에서는 policy가 history가 아니라 현재 state에만 관련이 있다 (markov)
		- state 자체가 reward를 fully describe하기 때문에 reward 등은 신경쓰지 않아도 됨
	- MDP와 policy가 주어졌을 때, 이를 이용해서 고른 state sequence는 Markov process이다. 또한, state와 reward의 sequence $$$ S_1, R_2, S_2, ...$$$를 markov reward process라고 한다.
- Value Functions
	- state-value function $$$v_{\pi}(s)$$$ : policy $$$\pi$$$를 따를 때, s에 있을 때 expected return
		- $$$v_{\pi}(s) = E_{\pi}[G_t|S_t=s] $$$
	- action-value function $$$ q_{\pi}(s,a) $$$ : state s에서 action a를 하고, policy $$$\pi$$$를 따를 때의 expected return
		- $$$q_{\pi}(s,a) = E_{\pi}[G_t|S_t=s, A_t=a] $$$

##### Bellman Expectation Equation
- 위의 Bellman Equation과 유사한 방법으로 state-value, action-value를 각각 decompose 할 수 있다
$$ v_{\pi}(s) = E_{\pi}[R_{t+1} + \gamma v_\pi(S_{t+1})|S_t=s] $$
$$ q_{\pi}(s,a) = E_{\pi}[R_{t+1}+\gamma q_\pi (S_{t+1}, A_{t+1})|S_t = s] $$
- value function의 경우 다음과 같이 생각해볼 수 있다
	- 현재 state에서 policy의 distribution을 가지고 action을 취했을 때, 각각의 action을 취했을 때의 reward를 q를 이용하여 나타낼 수 있다.
- action function의 경우 다음과 같이 생각할 수 있다
	- 현재 state s에서 action a를 취했을 때, 그 이후의 reward를 새로운 state의 value function으로 생각할 수 있다.
- 이 두가지 관계를 서로의 식에 넣어주는 식으로 v, q 각각을 recursive하게 만들어줄 수 있다.
$$ v_\pi = R^\pi + \gamma P^\pi v_\pi, so$$
$$ v_\pi = (I-\gamma P^\pi)^{-1} R^\pi$$
이와 같은 식으로 풀어줄 수 있음.
	- 이때 $$$P^\pi_{ss'} = \sum_{a \in A} \pi(a|s)P^a_{ss'} $$$ (원래 state transition function에 policy의 분포를 고려해서 만든 이동 분포)
	- 또 $$$ R^\pi_s = \sum_{a \in A} \pi(a|s)R_s^a $$$ (policy의 이동 분포를 고려한 현 state의 reward expectation) 

##### Optimal Value Function
- optimal state-value function $$$v_*(s)$$$ : 현재 state s에서 value function의 값을 maximize하는 policy $$$ {max}_\pi \: v_\pi(s) $$$
- optimal action-value function $$$ q_*(s, a) $$$ : $$$ {max}_\pi \: q_\pi (s,a) $$$
	 - 이 qstar를 알면 결국 MDP를 푼것. 어떤 모든 상황에서 action을 할 지 결정이 되므로.

##### Optimal Policy
- policy에 대한 partial ordering 정의 : 어떤 policy가 다른 policy에 비해, 모든 state에서 value function의 값이 같거나 클 경우 order가 같거나 크다고 정의한다.
$$ \pi \ge \pi' \quad if \quad v_\pi(s) \ge v_{\pi'}(s) \; for \; \forall s $$
- 이렇게 정의할 경우 항상 optimal policy가 하나는 존재. 이러한 policy는 항상 optimal state-value, action-value function을 만든다.
- optimal policy를 만드는 방법 :
	- $$$a = argmax \:q_*(s,a) $$$ 이면 policy를 1을 주고, 아닌 action에 대해서는 0을 주면 optimal policy가 만들어진다.

##### Bellman Optimality Equation for $$$v_*$$$ : What we usually call *'Bellman Equation'*
- 결국 v와 q는 $$$ v_*(s) = {max}_a \: q_*(s,a) $$$ 라는 식으로 연결되어 있음. state의 value를 maximize하는 방법은 그 state에서 택할 수 있는 action 중 가장 좋은 action을 취하는 것.
- q의 경우 원래 state에서 받는 reward에, action을 취함으로서 갈 수 있는 optimal value의 expectation으로 나타낼 수 있다. $$$ q_*(s,a) = \: R_S^a + \gamma \sum_{s'\in S} P_{ss'}^a v_*(s') $$$
- 두 식을 합침으로서 v와 q 각각에 대해 recursive한 식을 만들 수 있다. 그러나 max가 들어가기 때문에 non-linear이고, closed-form solution이 없음.
	- 다양한 iterative solution이 존재 : Value iteration, Policy iteration, Q-learning, Sarsa...

#### Extension to MDPs
- infinite / continuous MDPs
- partially observable MDPs
- undiscounted, average reward MDPs

## Chapter 3 : Planning by Dynamic Programming
- Dynamic Programming이라고 말하고 있기는 하지만 내 입장에서는 그냥 matrix(vector) 이용해서 iteratively converge하는 algorithm이라고 생각하는게 편할거 같음 (실제로 그렇고)

### Policy Evaluation
- Given MDP와 given policy $$$\pi$$$에 대해 Bellman Expectation을 계산해서 value function $$$v_\pi$$$를 계산하는 것
- $$$ v^{k+1} = R^\pi + \gamma P^\pi v^{k} $$$ 를 반복
	- 모든 state에 대해 한 step에 다 신경써줘서 이를 'synchronous'라고 부름
- grid world 예시를 보면, uniform random policy로 value function을 구하고, 이 value function 값으로 greedy하게 policy를 새로 만든다고 생각하면 (주변 중 argmax v(s)로 간다고 생각하면) optimal policy가 나옴
	 - any value function can be used to compute better value function
	 - policy iteration의 근간이 되는 아이디어

### Policy Iteration
- given policy $$$\pi$$$에 대해, policy evaluation을 이용해서 evaluate를 한 뒤 그 value function값을 이용한 greedy policy를 새로운 policy로 차용한다
	- 여기서의 policy는 deterministic한 policy (모든 MDP는 deterministic한 optimal policy를 가지므로 deterministic한 policy만 봐도 충분)
	- EM Algorithm과 비슷하게 느껴질 수 있지만, 완전한 EM Algorithm은 아님. 그러나 어느 정도의 연관관계는 있음.
- 항상 $$$\pi$$$가 $$$\pi_*$$$로 수렴함이 알려져있음
	- V도 V*로 가고, policy도 best policy로 가면서 점점 수렴
- Formal한 Policy Improvement 방법
	1. deterministic policy $$$\pi$$$가 있다
	2. new policy $$$ \pi'(s) = {argmax}_{a \in A} q_\pi(s,a) $$$
- Policy Iteration의 올바른 수렴성 확인
	1. policy iteration의 한 step은 모든 state의 value를 improve 함
		- $$$ q_\pi(s, \pi'(s)) = {max}_{a \in A} q_\pi(s,a) \ge q_\pi (s, \pi(s)) = v_\pi(s) $$$
	2. 1에 의해 한 step에서 value function도 반드시 증가함
		- $$$ v_\pi(s) \le q_\pi(s, \pi'(s)) = E_{\pi'}[R_{t+1}+\gamma v_\pi(S_{t+1})|S_t = s]$$$
		$$$ \le  E_{\pi'}[R_{t+1}+\gamma q_\pi(S_{t+1}, \pi'(S_{t+1}))|S_t = s] \quad via \: 1$$$
        $$$ \le  E_{\pi'}[R_{t+1}+\gamma R_{t+2} + \gamma^2 q_\pi(S_{t+2}, \pi'(S_{t+2}))|S_t = s] ...$$$
        $$$ \le  E_{\pi'}[R_{t+1}+\gamma R_{t+2} +\gamma^2 R_{t+3} + ...|S_t = s] = v_{\pi'}(s)$$$
    3. Improve가 멈추는 (수렴하는) 상황일 경우, 모든 state에 대해 max action이 골라진 상태이다. 즉 현재 policy는 bellman optimality equation을 만족하고, 결국 수렴을 통해 찾은 $$$\pi$$$는 optimal policy이다.

#### Modified Policy Iteration
- Stop early (value function 변화량이 일정 threshold 이하이면)
- 정해진 step 수만 진행 (ex : 3 step..)
- k = 1만 보고 stop하는 건? -> value iteration과 equivalent한 방법임

### Value Iteration
- optimal policy를 'optimal first action / successor state로부터의 optimal policy' 로 breakdown 한다
- ** Principle of Optimality ** : policy $$$\pi(a|s)$$$ is optimal iff s에서 갈 수 있는 모든 state s'에 대해 $$$v_\pi(s') = v_*(s') $$$
- 이런 관점에서 생각할 경우 $$$ v_*(s) <- {max}_a R_s^a + \gamma \sum P^a_{ss'} v_*(s') $$$ 로 쓸 수 있음. (Bellman Expectation Equation) 이걸 iterative하게 반복함으로서 value iteration을 행한다.
- Formal Algorithm
	- synchronous하게, 위의 식을 이용하여 $$$v^k(s)$$$를 통해 $$$v^{k+1}(s)$$$를 구하는 것을 반복한다.
		- 행렬로 나타내면, $$$v_{k+1} = {max}_{a \in A} R^a + \gamma P^a v_k $$$ where R^a, P^a is reward vector / transition matrix about particular action a
	- 반드시 optimal value $$$v_*$$$로 수렴한다
	- policy iteration과 다른 점은, value만 구하기 때문에 explicit한 policy가 나오는 것은 아님. 또한, 수렴 중간에 나오는 intermediate value function이 any policy에도 대응되지 않을 수 있음.
	- policy iteration에서 value만 계속 업데이트 하고, 마지막에 k=1로 policy를 바꾸는 것과 equivalent함.

### Synchronous DP Algorithms : sum
- state-value function에 대해 DP를 하면 O(mn^2) per iteration이지만 (m=number of actions, n=number of states), action-value function에 대해 DP를 하면 O(m^2 n^2) per iteration이다
	- state-value에 대해 DP를 하는 이유

### Extension : Asynchronous DP Algorithms
- 반드시 한 iteration에 모든 state를 update하는 것이 아니라, 일부 state에 대해서만 update를 하는 것.
	- reduce computation
- 모든 state가 전부 선택된다는 보장만 있으면, 역시 $$$v_*$$$로 수렴함

#### In-place value iteration
- 원래 value iteration은 old와 new vector를 따로 고려했지만, in-place value iteration에서는 vector를 하나만 두고 그 vector 안에서 update 하는 것을 반복한다
- 이렇게 방법을 사용할 경우 행렬로 업데이트 하면 안되고, for문을 이용해서 state별로 하나하나 update해야 할 것

#### Prioritised Sweeping
- 어떤 state부터 update 해야할 지, 어떤 순서로 state update를 해야할지에 대한 접근
- 'Bellman Error' (한번 v(s)를 update 했을 때 값이 변하는 정도의 절대값) 가 큰 state부터 접근한다
- 한 state를 update했을 경우, 그 state와 연관된 다른 state들의 Bellman Error를 update 해준다. Predecessor state에 대한 정보를 알고, Priority Queue를 이용하면 이러한 방식을 구현할 수 있다.

#### Real-time Dynamic Programming
- 실제로 agent가 방문하는 state에 대해서 update

#### Sample Backups
- Full-width backups : DP 방식의 접근은 모든 successor state, action, MDP transition, reward function에 대한 정보를 알고 있어야 한다. 문제가 커지면 제대로 해결하지 못할 수 있음.
- Further lecture에서는 실제로 경험해본 결과를 토대로 'sampling' 하면서 update 하는 방법에 대해 말할 것
	- full model에 대해 알 필요가 없으므로, real-world problem solve에 더 적합함 : lead to 'Model-free Prediction'
	- curse of dimensionality도 줄일 수 있음

### Contraction Mapping
- value iteration, policy evaluation, policy iteration 등이 optimal solution에 converge하는지, solution이 unique한지, 얼마나 빨리 converge하는지 등을 'contraction mapping theorem'을 이용하여 알 수 있다.
- TODO : slide 보고 update

## Lecture 4 : Model-Free Prediction
- MDP가 주어지지 않은 상황에서 MDP의 value function을 estimate 하는 방법
	- 이 lecture에서는 policy evaluation에 focus
	- 이 lecture에서 배운 core idea를 토대로 최적화 과정에 대해 배울 것

### Monte-Carlo Learning
- 'episode' 관점으로 경험을 해나가면서 학습을 한다.
- 목표 : policy $$$\pi$$$를 따를 때, value function $$$v_\pi$$$를 예측한다
- 이 때, reward function으로는 discounted score의 sum이 아니라 'empirical mean' 을 사용 (경험을 통해 얻은 점수의 평균)

##### First-Visit MC Policy Evaluation
- 한 episode에서 state s가 처음 방문되었을 경우에만
	- counter N(s) <- N(s) + 1를 늘림
	- total return S(s) <- S(s) + $$$G_t$$$ ($$$G_t$$$ : return, =  $$$ R_{t+1} + \gamma R_{t+2} + .... + \gamma ^{T-1} R_T $$$)
	- value function은 mean으로 생각. V(s) = S(s)/N(s). 큰 수의 법칙에 의해 방문 수가 늘어나면 실제 값으로 수렴함.

##### Every-Visit MC Policy Evaluation
- first-visit과 똑같은 방법으로 하되, 처음 방문할 경우가 아니라 state s를 방문한 모든 경우에 대해 갱신을 해준다

#### Incremental MC updates
- mean $$$\mu_k = \mu_{k-1} + \frac{1}{k} (x_k - \mu_{k-1}) $$$ 로 생각 가능
- sum을 생각할 필요 없이, incremental하게 MC를 update하는 것도 가능하다.
	- 한 episode가 끝나면 그 episode에서 방문한 state들에 대해 다음과 같은 update를 함
		- $$$ N(S_t) = N(S_t) + 1 $$$
		- $$$ V(S_t) = V(S_t) + \frac{1}{N(S_t)}(G_t - V(S_t)) $$$
- 문제가 non-stationary 할 경우 (시간에 따라 바뀌는 것이 있을 경우) 완전한 mean을 계산하는 대신 old episode를 까먹는 것이 좋을 수도 있다. 이럴 때는 1/N(St)을 이용하여 완전한 평균을 계산하는 대신 지수평균 적으로 계산함.
	- $$$ V(S_t) = V(S_t) + \alpha (G_t - V(S_t)) $$$

### TD-Learning (Temporal Difference)
- TD Learning에서는 episode가 끝날 필요가 없음 : bootstrapping
	- 다음에 일어날 행동을 예측하는 방식
- TD(0)
	- return을 'estimate' 한다 : $$$R_{t+1} + \gamma V(S_{t+1}) $$$
	- Monte-Carlo의 real return 대신 estimated return을 사용.
		- $$$ V(S_t) = V(S_t) + \alpha(R_{t+1} + \gamma V(S_{t+1}) - V(S_t))$$$
		- 이 때 $$$R_{t+1} + \gamma V(S_{t+1})$$$를 TD-target, $$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t) $$$를 TD error라고 함
- Monte-carlo와는 다르게, 어떤 일련의 과정이 끝나기 전에 즉각즉각 value function의 update를 할 수 있음. 또한 episode가 완전히 끝날 필요도 없음.
- 예를 들어 하나의 episode의 좋은과정과 나쁜 과정이 섞여있었고 결과적으로 나쁜 결과가 나왔을 경우, MC를 사용하면 모든 action들이 negative feedback을 받지만 TD를 사용할 경우 각각의 action이 다른 feedback을 받을 수 있음.

##### Bias / Variance
- true return이나 true TD target과는 달리, TD target $$$ R_{t+1} + \gamma V(S_{t+1}) $$$는 *biased estimate* 임. 내가 임의로 생각하는 V가 어느쪽으로 초기화되있을지도 모르고, 임의의 가정으로 생각하기 때문.
- 반면 TD target은 return보다 *lower variance* 임. return은 여러 개의 action / transition / reward에 의존하는 반면, TD target은 하나의 결과에만 의존하기 때문.
- Monte Carlo : High variance, zero bias, good convergence
- TD : Low variance, some bias, usually efficient than MC, but more sensitive to initial value
	- TD(0)은 $$$v_\pi(s)$$$로 수렴함. 그러나 function approximation을 하면 그렇지 않은 specific case들도 있음.

### MC and TD
- Monte Carlo는 minimum mean-squared error를 만드는 쪽으로 수렴 (all episode, all timestep에 대한 square error sum)
- 반면 TD(0)은 maximum likelihood Markov model로 수렴함. Data를 보고, data에 가장 fit되는 MDP의 solution으로 수렴함. (state 하나만 보고 다음 action을 예측하기 때문일 것)
- 따라서 TD는 Markov property를 사용한다. (모든 궤적을 보지 않고 전 state만 보기 때문) 즉, markov property가 성립하는 환경에서는 TD가 더 efficient하다. 반면, MC는 markov property를 사용하지 않지만 그렇기에 non-markov environment라면 MC가 efficient하다.

#### United View of RL
- Boostrapping과 sampling의 관점에서 다음과 같이 바라볼 수 있음.
	- boostrapping + sampling = TD learning
	- boostrapping + not sampling = DP
	- not boostrapping + sampling = Monte Carlo
	- not boostrapping + not sampling = Exhaustive Search

### TD($$$\lambda$$$)
- TD가 1 step만 보는게 아니라, n-step을 본 뒤 그 결과들로 현재 state의 value function을 update한다고 해보자.
- n-step reward를 $$$G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + .. + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n}) $$$ 이라고 하자. 그러면 n-step TD는 $$$V(S_t) = V(S_t) + \alpha (G_t^{(n)}-V(S_t)) $$$ 이다.
- 이 때 n을 하나만 정해서 해당 n-step reward만 보는 것이 아니라, 여러 n-step reward를 보고 평균을 내자는 것이 TD(lambda)의 목표.
- TD($$$\lambda$$$)
	- $$$G_t^\lambda  = (1-\lambda) \sum_{n=1}^{\inf} \lambda^{n-1} G_t^{(n)}$$$
	- $$$G_t^\lambda$$$는 n-step reward들의 geometric average. weight들을 다 합치면 1이 됨.
	- update 식은 $$$ V(S_t) = V(S_t) + \alpha (G_t^\lambda - V(S_t)) $$$
	- 이 때 geometric weight를 사용하는 이유는, geometric weight를 사용하면 memoryless하게 계산할 수 있기 때문.

#### Backward-view TD($$$\lambda$$$)
- Eligibility Trace : state의 frequency와 recency를 모두 고려한 값
	- $$$E_0(s) = 0, \quad E_t(s) = \gamma \lambda E_{t-1}(s) + 1(S_t = s) $$$
- V(s)를 업데이트 할 때, TD-error와 eligibility trace값을 둘다 이용하여 업데이트한다
	- $$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t) $$$
	- $$$ V(s) = V(s) + \alpha \delta_t E_t(s) $$$
- final reward를 이용해서 error를 구하고, 그 error가 backpropagate 되는 느낌?
- Theorem : episodic & offline case 에서, TD-lambda의 forward-view와 backward view는 equivalent하다.
	- online update의 경우 forward와 backward가 다르지만, eligibility trace 식을 조금 바꿈으로서 equivalence를 만들 수 있다 (Sutton and von Seijen, ICML 2014)
	- forward 방식을 생각할 경우 직접 n-step reward들을 계산해야하고 모든 계산이 끝날때 까지 기다려야함. 반면 backward 방식으로 생각하면 단순히 step마다 trace update를 해가면서 쉽게 값을 갱신할 수 있다. 즉 backward 방식으로 편하게 구현해도 forward와 동일한 효과를 볼 수 있으므로 이득
- lambda가 0일때는 TD(0)과 같고, lambda가 1일때는 monte-carlo와 같다. (효과만 같은 것이지, 실제로는 MC에 비해 benefit이 있음) 즉, TD-lambda는 TD와 MC의 benefit을 둘 다 누리기 위한 방법이다.

### Equivalence of Forward & Backward
- Mathematical Proof
- TODO : 슬라이드 보고 update

## Lecture 5 : Model-Free Control
- How to optimise value function of unknown MDP?

### On-policy & Off-policy
- On-policy : policy $$$\pi$$$에 대한 정보를, $$$\pi$$$를 따라서 경험해보면서 얻는 방법
- Off-policy : 다른 behavior를 보면서 policy $$$\pi$$$에 대해 학습하는 방법

### Generalized Policy Iteration
- Policy evaluation과 improvement를 반복하면서 좋은 해로의 수렴을 기대할 수 있을것
- policy evaluation을 MC로 하고, improvement를 greedy policy improvement를 통해 한다고 생각해보자.
	- 그러나 state-value function을 사용할 경우 greedy policy improvement를 할 때 MDP에 대한 정보가 있어야함 (P 및 이동가능한 상태..)
	- 반면 action-value function을 사용하면 greedy policy improvement 할 수 있음 (단순히 $$$\pi'(s) = {argmax}_a Q(s,a) $$$)
- 이제 Q를 MC를 통해서 알고, greedy policy improvement를 한다고 생각해보자. 그러나 이 방법도 문제점이 있는 것이, MC에서 샘플링을 하면서 모든 (s,a) pair를 볼 수 있는 것이 아니기 때문에 보지 못한 pair에 대한 평가는 하지 못한다는 단점이 있다.

#### $$$\epsilon$$$-Greedy exploration
- simple idea : epsilon의 확률로는 random하게 action을 고르고, 1-epsilon의 확률로는 greedy action을 고른다.
- Theorem : epsilon-greedy를 통해 고른 policy는 **항상 improvement가 있다**
	$$ q_\pi(s, \pi'(s)) = \sum_a \pi'(a|s)q_\pi(s,a)$$
    $$ = \frac{\epsilon}{m} \sum_a q_\pi(s,a) + (1-\epsilon) {max}_a q_\pi(s,a)$$
    $$ \ge \frac{\epsilon}{m} \sum_a q_\pi(s,a) + (1-\epsilon) \sum_a \frac{\pi(a|s)-\epsilon/m}{1-\epsilon} q_\pi(s,a) $$
    <center> (by definition of $$$ \pi(a|s) $$$ in epsilon-greedy) </center>
    $$ = \sum_a \pi(a|s)q_\pi(s,a) = v_\pi(s)$$

### Monte-Carlo control
- Lecture 3에서 보았듯이, Policy iteration을 할 때, 꼭 policy evaluation을 끝까지 해서 완전한 답을 알 필요는 없다
- 따라서, 한 episode를 진행할 때마다 evaluation (MC policy evaluation),  improvement (epsilon-greedy)를 반복한다.

#### GLIE (Greedy in the Limit with Infinite Exploration)
- GLIE는 다음과 같은 상황을 말한다
	- episode를 무한정 늘리면 모든 state, action을 무한정 방문한다
	- episode를 무한정 늘리면 policy가 greedy policy로 converge 한다
- ex) $$$\epsilon$$$-greedy를 사용할 때, $$$\epsilon_k = \frac{1}{k} $$$로 잡으면 GLIE 성질을 만족함
- GLIE Monte-Carlo Control
	- kth episode를 $$$\pi$$$를 이용해 sample
	- episode가 끝나면 state, action pair의 value function을 update
		- $$$ N(S_t, A_t) = N(S_t, A_t) + 1 $$$
		- $$$ Q(S_t, A_t) = Q(S_t, A_t) + \frac{1}{N(S_t, A_t)} (G_t - Q(S_t, A_t)) $$$
	- 그 후 policy를 갱신함
		- $$$ \epsilon = 1/k $$$
		- $$$ \pi = \epsilon-greedy(Q) $$$
	- 이 알고리즘은 optimal action-value function $$$q_*(s,a)$$$로 수렴한다.

### TD using : SARSA
- 처음 state S와 action A를 통해, reward R를 받고 새로운 state S'로 간다. 그 뒤, S'에서 할 수 있는 새로운 action A'에 대해 다음과 같이 Q를 update 한다.
	$$$ Q(S,A) = Q(S,A) + \alpha (R + \gamma Q(S', A') - Q(S,A)) $$$
- MC와는 달리 매 timestep마다 policy의 업데이트가 이루어짐.
- On-policy Control with Sarsa의 Psuedocode
	- Q를 랜덤하게 초기화한다. 단, Q(terminal state, :)는 0으로 초기화 한다.
	- Repeat :
		1. S를 초기화 (맨 처음 S)
		2. Q에 대해 epsilon-greedy등을 해서 policy $$$\pi$$$를 만든다. 이 policy로 취할 action A를 결정한다.
		3. repeat for each step in episode :
			 (1) A를 행하고, R과 S'를 observe함
             (2) 현재 Q를 통해 새로운 policy를 만들고, S'에서 이 policy를 통해 A'를 만듬
             (3) $$$ Q(S,A) = Q(S,A) + \alpha (R + \gamma Q(S', A') - Q(S,A)) $$$
             (4) S를 S'로, A를 A'로 이동
        until terminate
- 실제로 optimal로 converge 하기 위해서는 다음과 같은 조건들이 필요하다
	- GLIE여야 함
	- $$$ \sum_t a_t = \inf, \; \sum_t a_t^2 < \inf $$$
	- 물론 실제 구현할 때는 이 조건들 만족 안해도 웬만하면 돌아감

#### n-step sarsa (sarsa($$$\lambda$$$))
- TD-lambda와 마찬가지로, Q_t를 n-step return으로 생각하는 것
- Forward View로 생각하는 것은 TD-lambda와 같음. geometric weight.
- backward view도 TD-lambda와 유사하다. eligibility를 통해 쉽게 업데이트 하면서 forward view와 같은 효과를 볼 수 있다. 반면, eligibility trace를 state, action pair에 대해 하나씩 놓고 update한다.
	- $$$ E_0(s, a) = 0 $$$
	- $$$ E_t(s, a) = \gamma \lambda E_{t-1}(s,a) + 1(S_t = s, A_t = a) $$$
- Q는 다음과 같이 update 한다. 이 때, every state s, every action a (그 episode에서 방문한 s, a pair만 해도 상관없다) 에 대해 update 한다.
	 - $$$ \delta_t = R_{t+1} +\gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) $$$
	 - $$$ Q(s,a) = Q(s,a) + \alpha \delta_t E_t(s,a) $$$

### Off-policy Learning
- off-policy learning을 사용하면 human experience 등을 참고할 수 있고, old policy에서의 경험을 재활용할 수 있음

#### Importance Sampling
- $$$ E_{X \sim P}[f(X)] = E_{X \sim Q}[\frac{P(X)}{Q(X)}f(X)] $$$
	- 다른 distribution에서 나온 function을 보정하는 방법
- Importance Sampling for Off-policy monte carlo
	- $$$\mu$$$에서 나온 result를 통해 $$$\pi$$$를 evaluate한다
	- 그러나 실제로는 episode의 모든 step에 의존하기 떄문에, 너무 high variance이기 때문에 useless.
- Importance Sampling for Off-Policy TD
	- importance sampling을 이용해서 TD target을 sample한다
	- 지금까지의 경험을 토대로 만든 policy $$$\mu$$$를 가지고, 내가 evaluate 하려는 policy $$$\pi$$$가 얼마나 비슷한 방향으로 가는지를 보정해주는 느낌
	- $$$ V(S_t) = V(S_t) + \alpha ( {\frac{\pi(A_t|S_t)} {\mu(A_t|S_t)}} (R_{t+1}+\gamma V(S_{t+1}))-V(S_t)) $$$

### Q-Learning
- Next Action은 $$$\mu$$$를 이용해서 고르되, alternative successor action A'는 현재 policy $$$\pi$$$를 이용해서 고른다.
	- sarsa update를 다음과 같이 바꾼다 : $$$ Q(S_t, A_t) = Q(S_t, A_t) + \alpha (R_{t+1} + \gamma Q(S_{t+1}, A') - Q(S_t, A_t)) $$$
- behavior과 target policy가 모두 improve 할 수 있다고 해보자.
	- Target policy가 아예 greedy일 경우를 상정.
	- 이 경우 $$$ R_{t+1} + \gamma Q(S_{t+1}, A') $$$를 $$$ R_{t+1} + {max}_{a'} \gamma Q(S_{t+1}, a') $$$ 로 바꿀 수가 있다.

#### Q-learning control algorithm (SARSAMAX)
- Sarsa처럼 S, A에서 reward R를 받고 다음 state S'로 이동한 상황이다. 다음 action은 A'라고 denote하자.
- $$$ Q(S,A) <- Q(S,A) + \alpha (R + \gamma {max}_{a'} Q(S', a') - Q(S,A) ) $$$ 로 업데이트
	- 일단 한 step을 간 후, 그 곳에서 갈 수 있는 가장 좋은 곳을 택한다고 생각하고 update 하는 느낌.
	- 이 식을 따르면 optimal $$$q_*$$$로 수렴한다는 것이 알려져 있음.

##### pseudocode
- initialize Q : randomly, but Q(terminal, :) = 0
- repeat for each episode :
    1. set S to first state
    2. repeat for each step (until terminate) :
        - choose A by policy from Q (epsilon-greedy)
        - take action A, observe R & S'
        - $$$ Q(S,A) <- Q(S,A) + \alpha (R + \gamma {max}_{a'} Q(S', a') - Q(S,A) ) $$$
        - S to S'

### Relationship between DP and TD
|                                    | DP                | TD          |
|:----------------------------------:|-------------------|-------------|
| Bellman Expectation Equation for v | policy evaluation | TD learning |
| Bellman Expectation Equation for q | policy iteration  | SARSA       |
| Bellman Optimality Equation for q* | value iteration   | Q-learning  |

## Lecture 6 : Value Function Approximation
- 지금까지 배운 기법은 모든 state와 action에 대해 value를 저장할 수 있다는 생각 하에 진행해왔음 (Q-learning의 table 등). 그러나 state의 수가 매우 크거나, continuous한 경우 이러한 접근법은 사용 불가능하다.
	- value function v와 q를 approximate할 수 있는 함수를 만들어서 사용한다. 이때, 함수에서 사용하는 weight는 MC나 TD learning을 이용하여 갱신한다.
	- (이 수업에서) function approximator로 주로 사용하는 것은 linear combination이나 neural network (which is differentiable)
	- plus, supervised learning과는 다르게 non-stationary하고, non-iid한 (sequence들이 계속 들어오므로) 데이터에 대한 훈련방법이 필요함

### Incremental Methods
- J(w)를 mean-squared error로 잡고 stochastic gradient descent
	- $$$ \Delta_w = \alpha (v_\pi(S) - \hat{v}(S, w))\nabla_w \hat{v}(S, w) $$$
- state를 feature vector로 나타낸다고 생각하자.
- linear combination으로 나타냈을 때, $$$ \Delta_w $$$를 간단히 계산할 수 있음
	- 위의 식에서 Graident 부분을 단순히 $$$x(S)$$$ (feature vector)로 바꾸면 됨
	- 전 lecture까지 생각했던 table lookup은 linear combination의 special case라고 생각할 수 있음 (feature vector를 state의 one-hot vector라고 생각하면)

#### Incremental Prediction Algorithms
- gradient descent를 하기 위해서는 real value function의 값이 필요한데, 우리는 이 값을 알지 못한다.
- 따라서 target $$$v_\pi(s)$$$ 를 다른 target으로 바꾼다
	- MC의 경우 return $$$G_t$$$
	- TD(0)의 경우 TD target $$$R_{t+1} + \gamma \hat{v}(S_{t+1}, w) $$$
	- TD($$$\lambda$$$)의 경우 $$$G_t^\lambda$$$
- MC의 경우, 경험을 통해 얻은 state-return 쌍들을 supervised learning의 training set처럼 사용할 수 있다
- TD의 경우, TD-target은 biased-sample이긴 하지만 여전히 state-TD target 쌍들을 training set처럼 사용할 수 있다. 이렇게 해도 여전히 optimal에 가깝게 수렴함.
- TD-lambda의 경우, forward-view의 경우의 식은 명백하지만 backward는 자명하지 않음
	- function approx.를 사용할 경우 eligibility trace $$$E_t = \gamma \lambda E_{t-1} + x(S_t) $$$로 놓고 $$$ \Delta_w = \alpha \delta_t E_t $$$로 놓는다.

#### Control with Approximation
- Evaluation은 approximation으로 하고, improvement는 epsilon-greedy로
- control 하기 위해서는 action-value function q가 필요함. q도 v와 마찬가지의 방법으로 SGD를 할 수 있다.
- q의 경우,
	 - MC는 target $$$G_t$$$로 바꿈
	 - TD(0)은 TD target $$$R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) $$$로 바꿈
	 - TD-lambda의 backward의 경우, eligibility trace $$$E_t = \gamma \lambda E_{t-1} + \nabla_w \hat{q} (S_t, A_t, w) $$$로 놓음
- lambda : 1일 경우 (MC에 가까울 경우) 보통 결과가 좋지 않음. 적당한 lambda를 찾는 것이 좋다.

#### Baird's Counterexample & Algorithm Convergence
- simple MDP이지만, TD를 사용했을때 diverge하는 반례
- Evaluation Algorithm의 convergence
    - On-policy일 경우
        - linear combination이면 MC, TD0, TD-lam 다 converge 하지만 non-linear function approximation을 사용할 경우 TD-0과 TD-lambda는 diverge할 수 있다.
    - Off-policy일 경우
        - MC는 모든 경우에 대해 converge하지만, TD-0과 TD-lambda는 linear combination을 쓰더라도 diverge할 수 있다.
    - ** Gradient TD** 라고 해서, 어느 경우에도 bellman error의 true gradient를 따르는 TD learning이 있음
- Control의 경우,
	- non-linear이면 어느 경우에도 수렴의 보장이 없음
	- linear의 경우, MC control과 SARSA는 near optimal로 수렴하지만, optimal 근처에서 맴돔. Q-learning은 수렴의 보장성 없음
	- **Gradient Q-learning**이라고 해서, linear combination의 경우 반드시 수렴한다고 보장할 수 있는 알고리즘이 있다

### Batch RL
- 여러 (state, value) pair를 가지고 sum of square error를 loss로 해서 gradient descent를 한다
- Experience Replay를 이용한 SGD
	- experience D를 쌓아놓고, 다음을 반복
		 - D에서 (sample, value)쌍을 하나 sample한다
		 - 그 이후 이 쌍을 이용하여 SGD update
    - 이를 반복하면 least square를 만족하는 w로 수렴함

#### DQN's Algorithm
- epsilon-greedy policy를 이용하여 action $$$a_t$$$를 정한다
- $$$(s_t, a_t, r_{t+1}, s_{t+1}) $$$을 experience D에 저장한다
- D에서 mini-batch를 sample한다
- old weight를 이용해 q-learning target을 계산하고, SSE를 minimize하도록 SGD 한다
	- loss : $$$E[(r + \gamma \;{max}_{a'} Q(s', a'; w^{old})-Q(s, a; w))^2] $$$
	- SGD의 variant를 이용
- DQN이 stabilize 될 수 있었던 이유는 다음 두가지임
	1. experience replay : highly correlated 된 연속된 경험들을 보는 대신, 순서를 randomize하여 안정화
	2. fixed Q-targets : Network의 weight을 원래 방식처럼 update하는 것이 아니라, 원래 weight을 'freeze' 시키고 새로운 weight을 학습하는 방식을 사용했다. 이렇게 fixed된 오래된 weight를 사용함으로서 stabilize가 됨 (TD learning의 단점이 계속 bootstrap하면서 biased 된 결과를 얻을 수 있다는 것이므로?)
		 - old weight은 몇 step의 update 후 다시 바꾸는 형식으로
- 이 때 neural network는 last 4 frame을 stack 한 단순 CNN.

#### Linear Least Squares Prediction
- weight w를 이용하는 Least Square LS(w)에 대해, 우리의 minimum에서의 delta w의 expected update를 0임
- 이 식을 이용하면 그 시점에서의 w를 정확하게 알 수 있음 (normal equation처럼 생각해서)
	- $$$ w = (\sum_t x(s_t)x(s_t)^T)^{-1} \sum_t x(s_t)v_t^\pi $$$
- Least Square MC / TD / TD-lambda 다 동일한 방식을 이용해서 direct solution을 얻을 수 있음
	- incremental algorithm보다 better convergence property를 가짐

##### Least Squares Policy Iteration
- 위 LS algorithm들을 이용한 control 알고리즘
- policy evaluation은 LS Q-learning, improvement는 epsilon-greedy 이용