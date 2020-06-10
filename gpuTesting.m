gpuDevice;

a = linspace(1,10,10);
b = linspace(11,20,10);

aGpu = gpuArray(a);
bGpu = gpuArray(b);

%c = arrayfun(@(x)arrayTest(x),a)  %1./(1-arrayfun(@(x)exp(-x),a))
%cGpu = arrayfun(@(x)arrayTest(x),aGpu) %1./(1-arrayfun(@(x)exp(-x),aGpu))

x = linspace(1,10,10)
x = gpuArray(x);

option = 2

if(option == 1)
        fun = @(x)deepFunction1(x);
    end    
    if(option == 2)
        fun = @(x)deepFunction2(x);
    end

y = surfaceFun1(x,fun)


function arrayTest = arrayTest(x)
arrayTest = 1/(1+arrayfun(@(x)exp(-x),x));
end

function surfaceFun1 = surfaceFun1(x,fun)
    surfaceFun1 = fun(x);
end

function deepFunction1 = deepFunction1(x)
    disp('Deep function 1')
    deepFunction1 = x;
end

function deepFunction2 = deepFunction2(x)
    disp('Deep function 2')
    deepFunction2 = x.^2;
end