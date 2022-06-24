# Packages
using POMDPs, POMDPModelTools, QuickPOMDPs
# Solver
using DiscreteValueIteration

# Define State
struct State
    x::Int
    y::Int
end

@enum Action UP DOWN LEFT RIGHT
Action

null = State(-1, -1)
S = [
    [State(x, y) for x = 1:4, y = 1:3]..., null
]

# Define Action
#@enum Action UP DOWN LEFT RIGHT
A = [UP, DOWN, LEFT, RIGHT]

const MOVEMENTS = Dict(
    UP => State(0, 1),
    DOWN => State(0, -1),
    LEFT => State(-1, 0),
    RIGHT => State(1, 0)
)

Base.:+(s1::State, s2::State) = State(
    s1.x + s2.x, s1.y + s2.y
)

# transition function
function T(s::State, a::Action)
    if R(s) != 0
        return Deterministic(null)
    end

    len_a = length(A)
    next_states = Vector{State}(undef, len_a + 1)
    probabilities = zeros(len_a + 1)
    
    for (index, a_prime) in enumerate(A)
        prob =(a_prime == a) ? 0.7 : 0.1
        dest = s + MOVEMENTS[a_prime]
        next_states[index + 1] = dest

        if dest.x == 2 && dest.y == 2
            probabilities[index + 1] = 0
        elseif 1 <= dest.x <= 4 && 1 <= dest.y <= 3
            probabilities[index + 1] += prob
        end
    end
    next_states[1] = s
    probabilities[1] = 1 - sum(probabilities)
    return SparseCat(next_states, probabilities)
end

# Reward function
function R(s, a = missing)
    if s == State(4, 2)
        return -100
    elseif s == State(4, 3)
        return 10
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
value_view = [S policy.util]