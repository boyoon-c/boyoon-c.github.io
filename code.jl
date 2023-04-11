# how to assign values of different types to variables 
x = 1
안녕하세요 = 2

y = "안녕하세요"
x 
y
a = [1;2;3;4;5]
b = [1 2;3 4;5 6;7 8; 9 10]
c=[6,7,8,9,10]
γ, δ, σ, ϵ, ε, ξ = 1, 2, 3, 4, 5, 6
γ
δ
σ
ε

# we can use Unicode math symbols by backslash Latex symbol followed by tab. For example to type alpha symbol, you coud simply do \alpha-tab. 
α = 3
β = 4


# how to express in mathematics 
x+1
α + β

# function 
function gFunction(x,y) 
    x + y 
end 

gFunction(1,2)

fFunction(x,y) = x-y
fFunction(5,2)

fib(n::Integer) = n ≤ 2 ? one(n) : fib(n-1) + fib(n-2)

fib(1)
fib(2)
fib(3)
fib(4)
fib(5)


hcat(a,b)
vcat(a,c)

# map 
x -> x+ 3
function (x) 
    x+3
end 

map(1:10) do x 
    2x 
end 
map(x->2x, 1:10)

map(1:10, 11:20) do x, y
    x + y
end

map((x,y)->x+y, 1:10, 11:20)

# plots 
using Plots 

x = range(0, 10, length=100)
y1 = sin.(x)
y2 = cos.(x)
y3=range(0, 10, length=100)
plot(x, [y1 y2], title="Trigonometric functions", label=["sin(x)" "cos(x)"])
plot(x,y3, title="Linear Plot", label="y3")

plot(x, x->x^2, label="y")
plot(x, sin.(x), title="Sin Graph", label="y")

# for loop
for i in 1:10 
    println(i)
end


using Gadfly, Statistics
plot(x=1:100, y=simulateAR1(0.8, 1., 0., 0, 100), Geom.line())
Gadfly.push_theme(Theme(background_color=colorant"white"))
plot(x->x^2, 0, 100)


struct Foo 
    bar
    baz 
end 

foo = Foo("happy","bear")
foo = Foo("bear","happy?")
foo.bar
foo.baz

# include("testModule.jl")
using .myModule
import .myModule
myModule.myfunction()
myModule.mysecretfunction()

using Optim, Gadfly 
# univariate 
f1(x) = (x-1)*(x-2)*(x-3)*(x-4)
f1(4)
Gadfly.push_theme(Theme(background_color=colorant"white"))
p=Gadfly.plot(x->f1(x), -5.0, 5.0, Coord.cartesian(xmin=-10, ymin=-5, xmax=10, ymax=5))
img = SVG("./quickstart/content/posts/img/iris_plot.svg", 6inch, 4inch)
draw(img, p)
optimize(f1, -10.0, 10.0)

f2(x) =-1*( 1-x[1]^2-x[2]^2+2*x[1]+4*x[2])
f3(x,y)= 1-x^2-y^2+2*x+4*y
using Plots
x = range(0, 10, length=100)
y= range(0,10,length=100)
z=[f3(i,j) for i in x, j in y]
Plots.plot(x, y, z, linetype=:surface)


x0=[0.0, 0.0]
res=optimize(f2, x0)
minimum(res)
Optim.minimizer(res)

"""
    Comprehension
"""

x = LinRange(1,10,10)
x = rand(8)
# moving average
[ x[i-1]+x[i] for i=2:length(x)-1 ]
[ i + j + k for i=1:3, j=1:3, k=1:3]

map(x->x+2, 1:10)
[ x + 2 for x=1:10]

map((x, y)->x-2*y+2, 1:10, 11:20)
map((x,y)->x-y + 2, 1:10, -10:-1)
print(-10:-1)

## punctuation 
T1=[1;2;3]
T2=[3;4;5]
T1 <: T2
Int8 <: Float64
Float64 <: Int8 
d=Dict(:a=>1, :b=>2)
d[:a]

a=[1,2]
b=[1 2 ; 3 4]
broadcast(x->x+2, b)
broadcast(+, a, b)
map(x->x+2, b)


"""
Growth model example
"""

"""
a ={consume, store}
s = stock 
action a c = s-a 
storage limited by upper bound M 
u(c) = c^α (reward function)
output drawon from discrete uniform distribution 
s' = a + U ( state update)
S = {0, ..., M + B }
A(s) = {0, ..., min(s, M)}
"""


using BenchmarkTools, Plots, QuantEcon, Parameters

SimpleOG=@with_kw (B = 10, M=5, α=0.5, β=0.9)

function transition_matrices(g)
    (B, M, α, β) = g
    u(c) = c^α
    n = B + M + 1
    m = M + 1

    R = zeros(n, m)
    Q = zeros(n, m, n)

    for a in 0:M
        Q[:, a + 1, (a:(a + B)) .+ 1] .= 1 / (B + 1)
        for s in 0:(B + M)
            R[s + 1, a + 1] = (a≤s ? u(s - a) : -Inf)
        end
    end

    return (Q = Q, R = R)
end

g = SimpleOG();
Q, R = transition_matrices(g);
Q

function verbose_matrices(g)
    (;B, M, α, β) = g
    u(c) = c^α

    #Matrix dimensions. The +1 is due to the 0 state.
    n = B + M + 1
    m = M + 1

    R = fill(-Inf, n, m) #Start assuming nothing is feasible
    Q = zeros(n,m,n) #Assume 0 by default

    #Create the R matrix
    #Note: indexing into matrix complicated since Julia starts indexing at 1 instead of 0
    #but the state s and choice a can be 0
    for a in 0:M
         for s in 0:(B + M)
            if a <= s #i.e. if feasible
                R[s + 1, a + 1] = u(s - a)
            end
        end
    end

    #Create the Q multi-array
    for s in 0:(B+M) #For each state
        for a in 0:M #For each action
            for sp in 0:(B+M) #For each state next period
                if( sp >= a && sp <= a + B) # The support of all realizations
                    Q[s + 1, a + 1, sp + 1] = 1 / (B + 1) # Same prob of all
                end
            end
            @assert sum(Q[s + 1, a + 1, :]) ≈ 1 #Optional check that matrix is stochastic
         end
    end
    return (Q = Q, R = R)
end
