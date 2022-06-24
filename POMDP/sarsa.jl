# Packages
using POMDPs, POMDPModelTools, QuickPOMDPs
# Solver
using DiscreteValueIteration, TabularTDLearning
# Policy
using POMDPPolicies
# Standard Library
using Random


struct State
    x::Int
end

@enum Action LEFT RIGHT

null = State(-1)
S = [[State(x) for x = 1:7]..., null]

A = [LEFT, RIGHT]

const MOVEMENTS = Dict(
    LEFT => State(-1),
    RIGHT => State(1)
)

Base.:+(s1::State, s2::State) = State(s1.x + s2.x)

function T(s::State, a::Action)
    if R(s) != 0
        return Deterministic(null)
    end
    
    len_a = length(A)
    next_states = Vector{State}(undef, len_a + 1)
    probabilities = zeros(len_a + 1)

    for (index, a_prime) in enumerate(A)
        prob = (a_prime == a) ? 0.8 : 0.2
        dest = s + MOVEMENTS[a_prime]
        next_states[index + 1] = dest

        if 1 <= dest.x <= 7
            probabilities[index + 1] += prob
        end
    end

    next_states[1] = s
    probabilities[1] = 1 - sum(probabilities)

    return SparseCat(next_states, probabilities)
end
    
function R(s, a = missing)
    if s == State(1)
        return -1
    elseif s == State(7)
        return 1
    end
    return 0
end
    
gamma = 0.95

termination(s::State) = s == null
abstract type GridCity <: MDP{State, Action} end

mdp = QuickMDP(GridCity,
    states = S,
    actions = A,
    transition = T,
    reward = R,
    discount = gamma,
    isterminal = termination
)

solver = ValueIterationSolver(max_iterations = 30)
policy = solve(solver, mdp)

# SARSA
s_mdp = QuickMDP(GridCity,
    states = S,
    actions = A,
    transition = T,
    reward = R,
    discount = gamma,
    initialstate = S,
    isterminal = termination
)

s_alpha = 0.9
s_n_episodes = 10

s_solver = SARSASolver(
    n_episodes = s_n_episodes,
    learning_rate = s_alpha,
    exploration_policy = EpsGreedyPolicy(s_mdp, 0.5),
    verbose = false
)

s_policy = solve(s_solver, s_mdp)

# Q-Learning
q_mdp = QuickMDP(GridCity,
    states = S,
    actions = A,
    transition = T,
    reward = R,
    discount = gamma,
    initialstate = S,
    isterminal = termination
)

q_alpha = 0.9
q_n_episodes = 15

q_solver = QLearningSolver(
    n_episodes = q_n_episodes,
    learning_rate = q_alpha,
    exploration_policy = EpsGreedyPolicy(q_mdp, 0.5),
    verbose = false
)

q_policy = solve(q_solver, q_mdp)