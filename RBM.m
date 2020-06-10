%Programm by Xmann02 and DontStealMyAccount
%

% Global parameters

prop.sizeH = 2000; %size of the h-vector
prop.startLearningRate = 0.01; %learning rate
prop.endLearningRage = 0.001; %
prop.numTrainingImages = 1000; % number of images in training routine
prop.numGibbsIterations = 50; %number of iterations of gibbs sample
prop.gibbsSampleType = 'Propability'; % options 'Reconstruct', 'Propability'
prop.gibbsSampleInput = 'Sample'; %options: 'Sample', 'Random' NOT YET IMPLEMENTED
prop.regularizer = 'L2'; %reguarlizer options: 'None', 'L1', L2' 
prop.regularizerLambda = 0.0001; %regularizer Lambda 
prop.imageType = 'Grayscale'; %options: 'Grayscale', 'BW' 
prop.imageSamples = 'All'; %options: 'All' or any single digit 
prop.trainingPercentage = 0.8; %percentage of data used for training

% Options for video Generation
vidOpt.numGibbsIterations = 500;


funs = iniFunctions(prop);

% Image preparation

[images,labels] = prepareTrainingData(prop);

% initialization
%load or initialize values, note loading will overwrite set Global
%parameters
[a,b,W] = ini(images,prop.sizeH);
%[a,b,W,prop] = loadNetwork(3);



%Working area
%Train network or create video of training
%[a,b,W] = trainNetwork(images,a,b,W,prop,funs);

[a,b,W] = TrainingVideo(images,a,b,W,prop,funs);

%Create video of gibbs sample
GibbsSampleVideo(vectorizeImage(images(:,:,5),28,28),a,b,W,vidOpt);


% Save Network
%saveNetwork(a,b,W,prop);


% Show Samples
plotSamples(images,a,b,W,prop,funs);




%% Functions 

%% Functions for video making

function [a,b,W] = TrainingVideo(images,a,b,W,prop,funs)
v = VideoWriter('TrainingVideo.avi');
open(v);
%Training Loop
for j=1:prop.numTrainingImages
j    
image = images(:,:,randi([1,round(prop.trainingPercentage * length(images))],1,1));
vData = gpuArray(vectorizeImage(image,28,28));
pHjData = gpuArray(1./(1+arrayfun(@(x)exp(-x), b + W*vData)));
%pHj = pHjData;
%hData = gpuArray(rand(size(pHj))<pHj);
[pVj,pHj] = funs.gibbsSample(vData,a,b,W,prop);    
%Update Network parameters
[a,b,W] = funs.updateNetwork(a,b,W,vData,pVj,pHjData,pHj,prop);
prop.learningRate = prop.startLearningRate*(1-j/prop.numTrainingImages) + prop.endLearningRage*j/prop.numTrainingImages; 

subplot(1,2,1)
imshow(image);
subplot(1,2,2)
imshow(reconstructImage(pVj,28,28));
frame = getframe(gcf);
writeVideo(v,frame);

end

close(v);

end

function GibbsSampleVideo(vData,a,b,W,prop)
v = VideoWriter('GibbsSample.avi');
open(v);
vModel = gpuArray(vData);
for i=1:prop.numGibbsIterations
pHj = gpuArray(1./(1+arrayfun(@(x)exp(-x), b + W*vModel)));
hModel = gpuArray(rand(size(pHj))<pHj);

pVj = gpuArray(1./(1+arrayfun(@(x)exp(-x), a + W'*hModel)));
vModel = gpuArray(rand(size(pVj))<pVj);

subplot(1,3,1);
imshow(reconstructImage(vData,28,28));
subplot(1,3,2);
imshow(reconstructImage(pVj,28,28));
subplot(1,3,3);
imshow(reconstructImage(vModel,28,28));

frame = getframe(gcf);
writeVideo(v,frame);

end    

close(v);

end

%% initialize functions
function funs = iniFunctions(prop)
    
    if(strcmp(prop.gibbsSampleType,'Reconstruct'))
        funs.gibbsSample = @(vData,a,b,W,prop)GibbsSampleReconstruct(vData,a,b,W,prop);
    end
    if(strcmp(prop.gibbsSampleType,'Propability'))
        funs.gibbsSample = @(vData,a,b,W,prop)GibbsSamplePropability(vData,a,b,W,prop);
    end
    
    if(strcmp(prop.regularizer,'None'))
        funs.updateNetwork = @(a,b,W,vData,pVj,pHjData,pHj,prop)updateNoRegularizer(a,b,W,vData,pVj,pHjData,pHj,prop);
    end    
    if(strcmp(prop.regularizer,'L1'))
        funs.updateNetwork = @(a,b,W,vData,pVj,pHjData,pHj,prop)updateL1(a,b,W,vData,pVj,pHjData,pHj,prop);
    end 
    if(strcmp(prop.regularizer,'L2'))
        funs.updateNetwork = @(a,b,W,vData,pVj,pHjData,pHj,prop)updateL2(a,b,W,vData,pVj,pHjData,pHj,prop);
    end 
    

end

%% Plot 
function plotSamples(images,a,b,W,prop,funs)
figure(1)
for i=1:16
subplot(4,8,2*i-1)
A = images(:,:,randi([round(prop.trainingPercentage * length(images)),round(length(images))],1,1));
v = vectorizeImage(A,28,28);
imshow(A)
subplot(4,8,2*i)
%v = rand(28*28,1);
%sample an image to generate new sample
[pVj,pHj] = funs.gibbsSample(v,a,b,W,prop);
if strcmp(prop.imageType,'BW')
    v = pVj > rand(28*28,1);
    A = reconstructImage(v,28,28);
end    
if strcmp(prop.imageType,'Grayscale')
    A = reconstructImage(pVj,28,28);
end    
imshow(A)
end    
end

%% Training loop

function [a,b,W] = trainNetwork(images,a,b,W,prop,funs)

%Training Loop
for j=1:prop.numTrainingImages
j    

A = images(:,:,randi([1,round(prop.trainingPercentage * length(images))],1,1));
[a,b,W] = trainingStep(A,a,b,W,prop,funs);  
prop.learningRate = prop.startLearningRate*(1-j/prop.numTrainingImages) + prop.endLearningRage*j/prop.numTrainingImages;
%end    

end
end

%% image preparation
function [images,labels] = prepareTrainingData(prop)

[images, labels] = mnist_parse('train-images.idx3-ubyte','train-labels.idx1-ubyte');

if strcmp(prop.imageType,'BW')
    %Simple cutoff
    images = (images > 120);
else if strcmp(prop.imageType,'Grayscale')
    %image in grayscale
    images = double(images)/255;
    else
        disp('Mistake in prop.imageType');
    end
end        

if ~strcmp(prop.imageSamples,'All')
    mask = (labels == str2num(prop.imageSamples));
    images = images(:,:,mask);
end    

end

%% saving and loading trained networks
function saveNetwork(a,b,W,prop)
disp('saving network')
for i=1:10
    if ~exist(strcat("RBMs/", int2str(i)), 'dir')
        mkdir(strcat("RBMs/", int2str(i)))
        break;
    end
end    

save(strcat(strcat("RBMs/",int2str(i)),"/a.mat"),"a")
save(strcat(strcat("RBMs/",int2str(i)),"/b.mat"),"b")
save(strcat(strcat("RBMs/",int2str(i)),"/W.mat"),"W")
save(strcat(strcat("RBMs/",int2str(i)),"/prop.mat"),"prop")

end

function [a,b,W,prop] = loadNetwork(i)
disp('loading network')
a = matfile(strcat(strcat("RBMs/",int2str(i)),"/a.mat"));
b = matfile(strcat(strcat("RBMs/",int2str(i)),"/b.mat"));
W = matfile(strcat(strcat("RBMs/",int2str(i)),"/W.mat"));
prop = matfile(strcat(strcat("RBMs/",int2str(i)),"/prop.mat"));

a = a.a;
b = b.b;
W = W.W;
prop = prop.prop

end

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

function [a,b,W] = trainingStep(image,a,b,W,prop,funs)

vData = gpuArray(vectorizeImage(image,28,28));
pHjData = gpuArray(1./(1+arrayfun(@(x)exp(-x), b + W*vData)));
%pHj = pHjData;
%hData = gpuArray(rand(size(pHj))<pHj);

    
[pVj,pHj] = funs.gibbsSample(vData,a,b,W,prop);    

%Update Network parameters
[a,b,W] = funs.updateNetwork(a,b,W,vData,pVj,pHjData,pHj,prop);

end

function [a,b,W] = updateNoRegularizer(a,b,W,vData,pVj,pHjData,pHj,prop)
a = a + prop.learningRate * (vData-pVj);
b = b + prop.learningRate * (pHjData-pHj);
W = W + prop.learningRate * (kron(pHjData,vData')-kron(pHj,pVj'));
end

function [a,b,W] = updateL1(a,b,W,vData,pVj,pHjData,pHj,prop)
a = a + prop.learningRate * (vData-pVj);
b = b + prop.learningRate * (pHjData-pHj);
W = W + prop.learningRate * (kron(pHjData,vData')-kron(pHj,pVj')) - prop.regularizerLambda * sign(W) ;
end

function [a,b,W] = updateL2(a,b,W,vData,pVj,pHjData,pHj,prop)
a = a + prop.learningRate * (vData-pVj);
b = b + prop.learningRate * (pHjData-pHj);
W = W + prop.learningRate * (kron(pHjData,vData')-kron(pHj,pVj')) - 2 * prop.regularizerLambda * W ;
end

%% Calculate a gibbs sample
%input vector from data, biases for visible/hidden layer, interaction
%Matrix, num iterations
%Returns propability vectors for v and h
function [pVj,pHj] = GibbsSampleReconstruct(vData,a,b,W,prop) 
vModel = gpuArray(vData);
for i=1:prop.numGibbsIterations
pHj = gpuArray(1./(1+arrayfun(@(x)exp(-x), b + W*vModel)));
hModel = gpuArray(rand(size(pHj))<pHj);

pVj = gpuArray(1./(1+arrayfun(@(x)exp(-x), a + W'*hModel)));
vModel = gpuArray(rand(size(pVj))<pVj);
end    

end

function [pVj,pHj] = GibbsSamplePropability(vData,a,b,W,prop) 
vModel = gpuArray(vData);
pHj = gpuArray(1./(1+arrayfun(@(x)exp(-x), b + W*vModel)));
pVj = gpuArray(1./(1+arrayfun(@(x)exp(-x), a + W'*pHj)));

for i=1:prop.numGibbsIterations
pHj = gpuArray(1./(1+arrayfun(@(x)exp(-x), b + W*pVj)));
pVj = gpuArray(1./(1+arrayfun(@(x)exp(-x), a + W'*pHj)));
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



