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
- state $$$S_t$$$가 Markov라는 것은 P[S_{t+1}|S_t] = P[S_{t+1}|S_1, S_2, ..., S_t] 라는 것
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

###### Planning View
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