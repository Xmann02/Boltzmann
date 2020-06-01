%Programm by Xmann02 and DontStealMyAccount
%

%% Global parameters

sizeH = 500; %size of the h-vector


%% Load data set
[images, labels] = mnist_parse('train-images.idx3-ubyte','train-labels.idx1-ubyte');


%% Convert to binary

%Simple cutoff
images = (images > 120);
%imshow(images(:,:,2)) %test to see effect on image


%% Initialization values 

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



%% Working/Testing area


% Temporary 
A = images(:,:,1);
v = vectorizeImage(A,28,28);
k = 1;
alpha = 0.01; %Propably needs to be dynamic
n = 10;
% Training loop for single image -> move to function, make it work with
% different images for input
for j=1:n

    vData = vectorizeImage(A,28,28);
    vModel = vData;
    pHj = arrayfun(@(x)sigmoidCustom(x), b + W*vData);
    hData = rand(size(pHj))<pHj;
    hModel = hData;
    
    %Gibbs sampling  -> move to function
for i = 1:k 
%Random initalization for hidden layer vector h
pHj = arrayfun(@(x)sigmoidCustom(x), b + W*vModel);
hModel = rand(size(pHj))<pHj;

pVj = arrayfun(@(x)sigmoidCustom(x), a + W'*hModel);
vModel = rand(size(pVj))<pVj;
end

%Update Network parameters
a = a + alpha * (vData-vModel);
b = b + alpha * (hData-hModel);
W = W + alpha * (kron(hData,vData')-kron(hModel,vModel'));


end
pVH = prod(pHj)
pHV = prod(pVj)


% Show images (original vs reconstructed)
figure(1)
C = reconstructImage(vData,28,28);
imshow(C)
figure(2)
C = reconstructImage(vModel,28,28);
imshow(C)


%energy(v,pHj,a,b,W)






%% Functions 

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



