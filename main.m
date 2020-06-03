%Programm by Xmann02 and DontStealMyAccount
%

%% Global parameters

sizeH = 2000; %size of the h-vector
alpha = 0.001 %learning rate
n = 10000 % number of images in training routine
k = 50 %number of iterations of gibbs sample


%% Load data set
[images, labels] = mnist_parse('train-images.idx3-ubyte','train-labels.idx1-ubyte');


%% Image preparation

%Simple cutoff
%images = (images > 120);

%ooooorrrrr
%image in grayscale
images = double(images)/255;


%imshow(images(:,:,50)) %test to see effect on image


%% initialization
%[a,b,W] = ini(images,sizeH);


%% Working/Testing area


% Temporary 
%A = images(:,:,1);
%v = vectorizeImage(A,28,28);

%Training Loop
for j=1:n
j    
%alpha = 1/(100+j/10);
%use this to train network for specific number, needs to be removed later
%if labels(j)==4
A = images(:,:,randi([1,50000],1,1));
[a,b,W] = trainingStep(A,a,b,W,alpha,k);  
%end    
  


end
%pVH = prod(pHj)
%pHV = prod(pVj)



%Plotting
figure(1)
for i=1:16
subplot(4,8,2*i-1)
A = images(:,:,randi([50000,60000],1,1));
v = vectorizeImage(A,28,28);
imshow(A)
subplot(4,8,2*i)
%v = rand(28*28,1);
%sample an image to generate new sample
[pVj,pHj] = GibbsSample(v,a,b,W,50);
%v = pVj > rand(28*28,1);
A = reconstructImage(pVj,28,28);
imshow(A)
end    



%energy(v,pHj,a,b,W)






%% Functions 


%% Initialization values 
function [a,b,W] = ini(images,sizeH)
disp("initalizing Parameters ...");
%Get sizes of dataset
imageSize = size(images(:,:,1));
imageNumber = length(images(1,1,:));

%Random initialization of W matrix
%Values from Mehta p 98 on the bottom
W = normrnd(0,2/sqrt(sizeH+imageSize(1)*imageSize(2)),[sizeH,imageSize(1)*imageSize(2)]);

%Random intialization for input bias vector
%Values from Mehta p 98 on the bottom
initalizationInputBiasSigma = 1/mean(mean(mean(images)));
a = normrnd(0,initalizationInputBiasSigma,[imageSize(1)*imageSize(2),1]);

%Zero initialization for hidden layer bias vector
b = zeros(sizeH,1);

a = gpuArray(a);
b = gpuArray(b);
W = gpuArray(W);

disp("Done")
end

%% Do training step on sample image

function [a,b,W] = trainingStep(image,a,b,W,alpha,k)

vData = gpuArray(vectorizeImage(image,28,28));
pHjData = gpuArray(1./(1+arrayfun(@(x)exp(-x), b + W*vData)));
%pHj = pHjData;
%hData = gpuArray(rand(size(pHj))<pHj);

    
[pVj,pHj] = GibbsSample(vData,a,b,W,k);    



%Update Network parameters
a = a + alpha * (vData-pVj);
b = b + alpha * (pHjData-pHj);
W = W + alpha * (kron(pHjData,vData')-kron(pHj,pVj'));
end




%% Calculate a gibbs sample
%input vector from data, biases for visible/hidden layer, interaction
%Matrix, num iterations
%Returns propability vectors for v and h
function [pVj,pHj] = GibbsSample(vData,a,b,W,k) 
vModel = gpuArray(vData);
for i=1:k
pHj = gpuArray(1./(1+arrayfun(@(x)exp(-x), b + W*vModel)));
hModel = gpuArray(rand(size(pHj))<pHj);

pVj = gpuArray(1./(1+arrayfun(@(x)exp(-x), a + W'*hModel)));
vModel = gpuArray(rand(size(pVj))<pVj);
end    

end

%% Function to calculate sigmoid function
% Not used, because it is incompatible with gpu computing
function sigmoidCustom = sigmoidCustom(x)
sigmoidCustom = 1./(1+arrayfun(@(x)exp(-x),x));
end

%% Calculate Energy from vector
function energy = energy(x,h,a,b,W) %input Vector, hidden layer Vector, input bias, hidden layer Bias, exchange Matrix W
energy = -h'*W*x-a'*x-b'*h;
end


%% Reshape image to vector
function vectorX = vectorizeImage(image,sizeX,sizeY)
vectorX = reshape(image,[sizeX*sizeY,1]);
end

%% Reconstruct image from a vector
function image = reconstructImage(vector,sizeX,sizeY)
image = reshape(vector,[sizeX,sizeY]);
end



