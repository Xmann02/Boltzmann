gpuDevice

a = linspace(1,10,10)
b = linspace(11,20,10)

aGpu = gpuArray(a)
bGpu = gpuArray(b)

c = arrayfun(@(x)arrayTest(x),a)  %1./(1-arrayfun(@(x)exp(-x),a))
cGpu = arrayfun(@(x)arrayTest(x),aGpu) %1./(1-arrayfun(@(x)exp(-x),aGpu))


function arrayTest = arrayTest(x)
arrayTest = 1/(1+arrayfun(@(x)exp(-x),x));
end