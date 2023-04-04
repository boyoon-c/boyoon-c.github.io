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