%Programm by Xmann02 and DontStealMyAccount
%

%% Global parameters

sizeH = 2000; %size of the h-vector
alpha = 0.1 %learning rate
n = 1000 % number of images in training routine
k = 20 %number of iterations of gibbs sample


%% Load data set
[images, labels] = mnist_parse('train-images.idx3-ubyte','train-labels.idx1-ubyte');


%% Image preparation

%Simple cutoff
images = (images > 120);

%ooooorrrrr
%image in grayscale
%images = double(images)/255;


%imshow(images(:,:,50)) %test to see effect on image


%% initialization
[a,b,W] = ini(images,sizeH);

%% Working/Testing area


% Temporary 
%A = images(:,:,1);
%v = vectorizeImage(A,28,28);

for j=1:n
j    

%use this to train network for specific number, needs to be removed later
if labels(j)==8
A = images(:,:,j);
[a,b,W] = trainingStep(A,a,b,W,alpha,k);  
end    
  


end
%pVH = prod(pHj)
%pHV = prod(pVj)


% Show images (original vs reconstructed)
figure(1)
A = images(:,:,1);
v = vectorizeImage(A,28,28);
imshow(A)
figure(2)
%v = rand(28*28,1);
%sample an image to generate new sample
[pVj,pHj] = GibbsSample(v,a,b,W,500);
v = pVj > rand(28*28,1);
A = reconstructImage(v,28,28);
imshow(A)



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
disp("Done")
end

%% Do training step on sample image

function [a,b,W] = trainingStep(image,a,b,W,alpha,k)

vData = vectorizeImage(image,28,28);
vModel = vData;
pHjData = arrayfun(@(x)sigmoidCustom(x), b + W*vData);
pHj = pHjData;
hData = rand(size(pHj))<pHj;
hModel = hData;
    
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
vModel = vData;
for i=1:k
%Random initalization for hidden layer vector h
pHj = arrayfun(@(x)sigmoidCustom(x), b + W*vModel);
hModel = rand(size(pHj))<pHj;

pVj = arrayfun(@(x)sigmoidCustom(x), a + W'*hModel);
vModel = rand(size(pVj))<pVj;
end    

end

function sigmoidCustom = sigmoidCustom(x)
sigmoidCustom = 1/(1+arrayfun(@(x)exp(-x),x));
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



