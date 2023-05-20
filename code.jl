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
using Pkg
Pkg.rm("Plots")
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


using BenchmarkTools, QuantEcon, Parameters

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

using DataFrames
df = DataFrame(
    A=1:2:1000, 
    B= repeat(1:10, inner=50), 
    C= 1:500)

first(df, 10)
last(df, 10)

df[[1,3,5], :]
df[:, [:A, :C]]
select(df, :A)
df[:, [:A]]
df[:, :A]
df[:, :x1].=0
df[:, :x2].=1
df
df[:, r"x"]
df[:, Not(r"x")]
df[:, Not(:A)][1:10, :]
df[(df.A .> 100), :]

# subsetting functions 
subset(df, :A=>a-> a .<10)

# rename 
df1=select(df,  :A => :a, :B=>:b )
first(df1, 2)

select(df1, :a, :b, [:a, :b]=>ByRow((x1, x2)-> x1/x2) => :z  )

select(df, [:A, :B].=>[:a, :b])

vec=[1,2,3,4,5]

vec .^3 |> sum
[vec[i]^3 for i in 1:5] |>sum
vec |> x->x.^3|>x->sum(x)


using BenchmarkTools, LinearAlgebra, Plots, QuantEcon, Statistics
using SparseArrays
using Parameters

SimpleOG = @with_kw (B = 10, M = 5, α = 0.5, β = 0.9)
g = SimpleOG()
Q, R = transition_matrices(g);
ddp=DiscreteDP(R, Q, g.β)
results=solve(ddp, PFI)
results.v
results.num_iter
results.sigma .- 1
fieldnames(typeof(results))

using Gadfly
plot(x=1:16, y=results.v, Geom.line(), 
    Geom.point(), 
    Guide.ylabel("value"), 
    Guide.xlabel("state"))


using SymPy
a,b,c,x = symbols("a, b, c, x")
x, y = symbols("x, y")
p1=a*x^2 + b*x + c
p2 = x + 0.5*y + 1 
typeof(p1)
p1.coeff(1)
p2.coeff(y)
p2.coeff(1)
p = [2*x+ 3*y; 1*x+5*y+6]
θ=zeros(2, length(p))
M=length(p)
for i in 1:M
    θ[i, 1]=p[i].coeff(x)
    θ[i, 2]=p[i].coeff(y) 
end 
θ*[x,  y]
p

p[1].coeff(x)
c₁
c₁= [p[i].coeff(x) for i in 1:2]
c₂=[p[i].coeff(y) for i in 1:2]
c=[c₁ c₂]
c*[1 , 2]
θ=[1, -4]
c*θ


for i in 1:2:10
    println(i)
end

for i ∈ 1:2:10
    println(i)
end 
for i = 1:2:10
    println(i)
end 
for i = eachindex(1:2:10)
    println(i)
end 


N=4
T=2 
P=2

using SymPy
a11, a12, a13, a14 = 1,2,3,4
a21, a22, a23, a24 = 5,6,7,8
b11, b12, b13, b14 = -1, -2, -3, -4
b21, b22, b23, b24 = -5, -6, -7, -8

θ, ρ = symbols("θ, ρ")
A = [a11*θ + b11*ρ a12*θ + b12*ρ a13*θ + b13*ρ a14*θ + b14*ρ; 
a21*θ + b21*ρ a22*θ + b22*ρ a23*θ + b23*ρ a24*θ + b24*ρ]
Ψ=Array{Any}(undef, T, P, N) 

# Ψ contains above written metrices for each n 
for n in 1:N 
    Ψ[:, :, n]=
    hcat(
        [A[t, n].coeff(θ) for t in 1:T],
        [A[t, n].coeff(ρ) for t in 1:T]
    )
end
Ψ
β=0.95
βm= [β^t for t in 1:T]
1/N * reduce(+, Ψ[:,:, n] for n in 1:N )
W = transpose(βm) .* 1/N * reduce(+, Ψ[:,:, n] for n in 1:N )


x=2
x=2;


countries = ("Korea", "Japan", "China")
for (index, country) in enumerate(countries) 
    println("Country $(country) in index $(index)")
end
collect(enumerate(countries))[1][1]

function dynamic(x_in, slope, intercept)
    x_out=slope*x_in + intercept
    return x_out 
end 



p = (A=2, s=0.3, α=0.3, δ=0.4, xmin=0, xmax=4)

g(k; p) = p.A * p.s * k^p.α + (1-p.δ)*k

x = zeros(6)
x[1]=2.0
for t in 2:100
    x[t]=g(x[t-1];p)
end 


plot(1:10, [1,2,3,4,5,6,7,8,9,10])
plot!(1:10, 21:30)


function ts_plot(f, xmin, xmax, x0; ts_length=6)
    x = zeros(ts_length)
    x[1] = x0
    for t in 1:(ts_length-1)
        x[t+1] = f(x[t])
    end
    plot(1:ts_length, x, ylim=(xmin, xmax), linecolor=:blue, lw=2, alpha=0.7)
    scatter!(x, mc=:blue, alpha=0.7, legend=false)
end

using Plots 
k0=0.25
ts_plot(k -> g(k; p), p.xmin, p.xmax, k0)


a=0.9
b=0.1
c=0.5
mu=-3.0
v=0.6

using StatsPlots 
using Distributions

sim_length=10
x_grid=range(-5, 7, length=120)

plt=plot() 
for t in 1:sim_length 
    mu = a * mu + b
    v = a^2 * v + c^2
    dist = Normal(mu, sqrt(v))
    plot!(plt, x_grid, pdf.(dist,x_grid), 
    label="\\psi_{$t}",
    linealpha=0.7)
end 
plt

function plot_density_seq(mu_0=-3.0, v_0=0.6; sim_length=60)
    mu = mu_0
    v = v_0
    plt = plot()
    for t in 1:sim_length
        mu = a * mu + b
        v = a^2 * v + c^2
        dist = Normal(mu, sqrt(v))
        plot!(plt, x_grid, pdf.(dist, x_grid), label=nothing, linealpha=0.5)
    end
    return plt
end
plot_density_seq()
plot_density_seq(3.0)


d = Categorical([0.5, 0.3, 0.2])
@show rand(d,5)
@show pdf(d,1)

@assert size([0.5, 0.3, 0.2])[1] == size([0.5, 0.3, 0.2])[2]
function mc_sample_path(P; init=1, sample_size=1000)
    @assert size(P)[1] == size(P)[2] # transition matrix should be a square matrix 
    
    # N be the number of rows of transition matrix (or the number of initial states)
    N = size(P)[1]
    # dists be the state transition probabiliteis for each initial state; for example dists[1] will be state-transition probabilities of state 1 transitioning to state 1 , 2, and 3 respectively
    dists = [Categorical(P[i, :]) for i in 1:N]

    X = fill(0, sample_size)
    X[1] = init 

    for t in 2:sample_size 
        dist=dists[X[t-1]]
        X[t]=rand(dist)
    end 
    return X 
end 

P=[0.4 0.6; 0.2 0.8]
X = mc_sample_path(P, sample_size = 100_000)
μ1 = count(X.==1)/length(X)

mc=MarkovChain(P, ["unemployed", "employed"])


function mccallbellmanmap(v,  w,p,c,β)

    v_reject = c + β * dot(p,v) 
    v_accept = w/(1-β)
    

    v_out = max.(v_reject,v_accept)

    S = length(w)
    for s in 1:S
        v_out[s] = max(v_reject,w[s]/(1-β))
    end
    
    return v_out
end

A = [1 2; 3 4]
argmax(A)
B=[-1 2; 10 -10]
argmax(B)
argmin(B)

argmax(x->x^2-3*x,-10:10 )
argmin(x->x^2-3*x,-10:10 )
plot(x->x^2-3*x, -10:10)

A = rand(5, 2,3)
# column-wise reshape
reshape(A, (2,3,5))
B = rand(1,4)
reshape(B, (2,2))
using LinearAlgebra
Bt=transpose(B)
reshape(Bt, (2,2))

C = [1,2,3,4,5,6]
C′= reshape(C, (2,3))

permutedims(C′, (2, 1))
transpose(C′)

permutedims(reshape(C, (3,2)), (2,1))

Threads.nthreads()
Threads.threadid()

a = zeros(10)
Threads.@threads for i =1:10 
    a[i]=Threads.threadid()
end

eng2kor=Dict()
eng2kor["one"]="일"
eng2kor["two"]="이"
eng2kor
eng2kor=Dict(
    "one"=>"일",
    "two"=>"이",
    "three"=>"삼",
    "four"=>"사"    
)
eng2kor
values(eng2kor)
keys(eng2kor)

function histogram(s)
    d = Dict()
    for c in s
        if c ∉ keys(d)
            d[c] = 1
        else
            d[c] += 1
        end
    end
    d
end

histogram("장보윤장보윤")

array=[1,2,3,4,5]
push!(array,6)
pop!(array)
array

pop!(eng2kor, "one")
push!(eng2kor, "four"=>"😃")
push!(eng2kor, "five"=>"다섯")
push!(eng2kor, "one"=>"🍔")
eng2kor["four"]
eng2kor

a=[1,2,3,4,5]
push!(a, 6,7)
pop!(a)

fun(a,x)=a.-x 
fun.([1,2], [3,4])
fun.([1,2], Ref([3,4]))
fun.(Ref([1,2]), [3,4])
@. fun([1,2], [3,4])

using Distributions
n, a, b= 50, 200,100
q_default = BetaBinomial(n, a, b).pdf() 