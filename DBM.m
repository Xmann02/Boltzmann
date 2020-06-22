%Programm by Xmann02 and DontStealMyAccount
%

% Global parameters
prop.netDepth = 3;
prop.size = [400,200,100]; %size of the h-vector
prop.learningRate = 0.01
prop.startLearningRate = 0.001; %learning rate
prop.endLearningRate = 0.0001; %
prop.numTrainingImages = 1000; % number of images in training routine
prop.gibbsSampleInputNoise = 0.0; % Overlay starting sample for gibbs sample with noise options: [0:1]
prop.numGibbsIterations = 10; %number of iterations of gibbs sample
prop.gibbsSampleType = 'Propability'; % options 'Reconstruct', 'Propability'
prop.regularizer = 'None'; %reguarlizer options: 'None', 'L1', L2' 
prop.regularizerLambda = 0.0001; %regularizer Lambda 
prop.imageType = 'Grayscale'; %options: 'Grayscale', 'BW' 
prop.imageSamples = '8'; %options: 'All' or any single digit 
prop.trainingPercentage = 0.8; %percentage of data used for training

% Options for video Generation
vidOpt = prop;
vidOpt.numGibbsIterations = 500;
vidOpt.gibbsSampleInputNoise = 0.0;


funs = iniFunctions(prop);

% Image preparation

[images,labels] = prepareTrainingData(prop);

% initialization
%load or initialize values, note loading will overwrite set Global
%parameters

%net = ini(images,prop);
%[net,prop] = loadNetwork(2); 



%Working area
%Train network or create video of training
net = preTrainNetwork(images,net,prop,funs);
net = deepTraining(images,net,prop,funs);


net = deepTrainingVideo(images,net,prop,funs); 
%GibbsSampleVideo(vectorizeImage(images(:,:,1),28,28),net,vidOpt);
%Create video of gibbs sample




% Save Network
%saveNetwork(net,prop);

%Plot layers of network and biases
plotNetwork(net); 

% Show Samples
plotSamples(images,net,prop,funs); 




%% Functions 

function net = deepTraining(images,net,prop,funs)

%Training Loop
for j=1:prop.numTrainingImages
j    

A = images(:,:,randi([1,round(prop.trainingPercentage * length(images))],1,1));
vData = gpuArray(vectorizeImage(A,28,28));
net = deepTrainingStep(vData,net,prop,funs);  
prop.learningRate = prop.startLearningRate*exp(log(prop.startLearningRate)*(1-j)/prop.numTrainingImages + log(prop.endLearningRate)*j/prop.numTrainingImages);
%end    

end

end

function net = deepTrainingStep(vData,net,prop,funs)

data.layer(1).prop = gpuArray(vData);
for i=2:(prop.netDepth+1)
    data.layer(i).prop = gpuArray(1./(1+arrayfun(@(x)exp(-x), net.layer(i-1).b + net.layer(i-1).W*data.layer(i-1).prop)));
end


%pHjData = gpuArray(1./(1+arrayfun(@(x)exp(-x), b + W*vData)));

%pHj = pHjData;
%hData = gpuArray(rand(size(pHj))<pHj);

gibbsStart = vData * (1-prop.gibbsSampleInputNoise) + prop.gibbsSampleInputNoise * rand(size(vData));    
model = deepGibbsSample(gibbsStart,net,prop);    

%Update Network parameters
net = deepUpdateNetwork(net,data,model,prop);
end


function model = deepGibbsSample(gibbsStart,net,prop)

model.layer(1).prop = gpuArray(gibbsStart);
    for i=1:prop.numGibbsIterations
    
        for j=2:(prop.netDepth+1)
        model.layer(j).prop = gpuArray(1./(1+arrayfun(@(x)exp(-x), net.layer(j-1).b + net.layer(j-1).W*model.layer(j-1).prop)));
        end

        for j=(prop.netDepth):-1:1
        model.layer(j).prop = gpuArray(1./(1+arrayfun(@(x)exp(-x), net.layer(j).a + net.layer(j).W'*model.layer(j+1).prop)));
        end
    end
    
    

end

function net = deepUpdateNetwork(net,data,model,prop)

for i=1:prop.netDepth
    net.layer(i).a = net.layer(i).a + prop.learningRate*(data.layer(i).prop-model.layer(i).prop); 
    net.layer(i).b = net.layer(i).b + prop.learningRate*(data.layer(i+1).prop-model.layer(i+1).prop);
    net.layer(i).W = net.layer(i).W + prop.learningRate*(kron(data.layer(i+1).prop,data.layer(i).prop')-kron(model.layer(i+1).prop,model.layer(i).prop'));
end 
end



%% Functions for seeing what W,a,b do
function plotNetwork(net)
a = net.layer(1).a;
W = net.layer(1).W;
figure(8);
subplot(4,8,1);
A = reconstructImage((a-min(a))/max(abs(a-min(a))),28,28);
imshow(A);
for i=2:32
subplot(4,8,i);
w = reconstructImage((W(i,:)-min(W(:,:))/max(abs(W(:,:)-min(W(:,:))))),28,28);
imshow(w);
end


end


%% Functions for video making


function net = deepTrainingVideo(images,net,prop,funs)
v = VideoWriter('TrainingVideo.avi');
open(v);
%Training Loop
for j=1:prop.numTrainingImages
j    
image = images(:,:,randi([1,round(prop.trainingPercentage * length(images))],1,1));
vData = gpuArray(vectorizeImage(image,28,28));
data.layer(1).prop = gpuArray(vData);


for i=2:(prop.netDepth+1)
    data.layer(i).prop = gpuArray(1./(1+arrayfun(@(x)exp(-x), net.layer(i-1).b + net.layer(i-1).W*data.layer(i-1).prop)));
end


gibbsStart = vData * (1-prop.gibbsSampleInputNoise) + prop.gibbsSampleInputNoise * rand(size(vData));
%pHj = pHjData;
%hData = gpuArray(rand(size(pHj))<pHj);
model = deepGibbsSample(gibbsStart,net,prop);       
%Update Network parameters
net = deepUpdateNetwork(net,data,model,prop);
prop.learningRate = prop.startLearningRate*exp(log(prop.startLearningRate)*(1-j)/prop.numTrainingImages + log(prop.endLearningRate)*j/prop.numTrainingImages); 


subplot(1,3,1)
imshow(image);
subplot(1,3,2)
imshow(reconstructImage(gibbsStart,28,28));
subplot(1,3,3)
imshow(reconstructImage(model.layer(1).prop,28,28));
frame = getframe(gcf);
writeVideo(v,frame);

end

close(v);

end



function [a,b,W] = TrainingVideo(images,a,b,W,prop,funs)
v = VideoWriter('TrainingVideo.avi');
open(v);
%Training Loop
for j=1:prop.numTrainingImages
j    
image = images(:,:,randi([1,round(prop.trainingPercentage * length(images))],1,1));
vData = gpuArray(vectorizeImage(image,28,28));
pHjData = gpuArray(1./(1+arrayfun(@(x)exp(-x), b + W*vData)));
gibbsStart = vData * (1-prop.gibbsSampleInputNoise) + prop.gibbsSampleInputNoise * rand(size(vData));
%pHj = pHjData;
%hData = gpuArray(rand(size(pHj))<pHj);
[pVj,pHj] = funs.gibbsSample(gibbsStart,a,b,W,prop);    
%Update Network parameters
[a,b,W] = funs.updateNetwork(a,b,W,vData,pVj,pHjData,pHj,prop);
prop.learningRate = prop.startLearningRate*exp(log(prop.startLearningRate)*(1-j)/prop.numTrainingImages + log(prop.endLearningRate)*j/prop.numTrainingImages); 

subplot(1,3,1)
imshow(image);
subplot(1,3,2)
imshow(reconstructImage(gibbsStart,28,28));
subplot(1,3,3)
imshow(reconstructImage(pVj,28,28));
frame = getframe(gcf);
writeVideo(v,frame);

end

close(v);

end

function GibbsSampleVideo(vData,net,prop)
v = VideoWriter('GibbsSample.avi');
open(v);
gibbsStart = vData * (1-prop.gibbsSampleInputNoise) + prop.gibbsSampleInputNoise * rand(size(vData));
model.layer(1).prop = gpuArray(gibbsStart);
for i=1:prop.numGibbsIterations
   
     for j=2:(prop.netDepth+1)
     model.layer(j).prop = (1./(1+arrayfun(@(x)exp(-x), net.layer(j-1).b + net.layer(j-1).W*model.layer(j-1).prop)));
     end
 
     for j=(prop.netDepth):-1:1
     model.layer(j).prop = (1./(1+arrayfun(@(x)exp(-x), net.layer(j).a + net.layer(j).W'*model.layer(j+1).prop)));   
     end
     
vModel = gpuArray(rand(size(model.layer(1).prop))<model.layer(1).prop);    
    

subplot(1,4,1);
imshow(reconstructImage(vData,28,28));
subplot(1,4,2);
imshow(reconstructImage(gibbsStart,28,28));
subplot(1,4,3);
imshow(reconstructImage(model.layer(1).prop,28,28));
subplot(1,4,4);
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
function plotSamples(images,net,prop,funs)
figure(1)
for i=1:16
subplot(4,8,2*i-1)
A = images(:,:,randi([round(prop.trainingPercentage * length(images)),round(length(images))],1,1));
A = A * (1-prop.gibbsSampleInputNoise) + prop.gibbsSampleInputNoise * rand(size(A));
v = vectorizeImage(A,28,28);
imshow(A)
subplot(4,8,2*i)
%v = rand(28*28,1);
%sample an image to generate new sample
model = deepGibbsSample(v,net,prop);
if strcmp(prop.imageType,'BW')
    v = model.layer(1).prop > rand(28*28,1);
    A = reconstructImage(v,28,28);
end    
if strcmp(prop.imageType,'Grayscale')
    A = reconstructImage(model.layer(1).prop,28,28);
end    
imshow(A)
end    
end

%% Training loop

function net = preTrainNetwork(images,net,prop,funs)

%Training Loop
for i=1:prop.netDepth
    a = net.layer(i).a;
    b = net.layer(i).b;
    W = net.layer(i).W;
    
    
    
    for j=1:prop.numTrainingImages
    j    
    A = images(:,:,randi([1,round(prop.trainingPercentage * length(images))],1,1));
    vData = gpuArray(vectorizeImage(A,28,28));
    
    for k=1:(i-1)
        
        vData = net.layer(k).b + net.layer(k).W*vData;
    end
    [a,b,W] = trainingStep(vData,a,b,W,prop,funs);  
    prop.learningRate = prop.startLearningRate*exp(log(prop.startLearningRate)*(1-j)/prop.numTrainingImages + log(prop.endLearningRate)*j/prop.numTrainingImages);
    end    

    net.layer(i).a = a; 
    net.layer(i).b = b;
    net.layer(i).W = W;
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
function saveNetwork(net,prop)
disp('saving network')
for i=1:10
    if ~exist(strcat("DBMs/", int2str(i)), 'dir')
        mkdir(strcat("DBMs/", int2str(i)))
        break;
    end
end    

save(strcat(strcat("DBMs/",int2str(i)),"/net.mat"),"net")
save(strcat(strcat("DBMs/",int2str(i)),"/prop.mat"),"prop")

end

function [net,prop] = loadNetwork(i)
disp('loading network')
net = matfile(strcat(strcat("DBMs/",int2str(i)),"/net.mat"));
prop = matfile(strcat(strcat("DBMs/",int2str(i)),"/prop.mat"));

net = net.net;
prop = prop.prop;

end



%% Initialization values 
function net = ini(images,prop)
disp("initalizing Parameters ...");
%Get sizes of dataset
imageSize = size(images(:,:,1));
imageNumber = length(images(1,1,:));



%Random initialization of W matrix
%Values from Mehta p 98 on the bottom
net.layer(1).W = normrnd(0,2/sqrt(prop.size(1)+imageSize(1)*imageSize(2)),[prop.size(1),imageSize(1)*imageSize(2)]);
net.layer(1).W = gpuArray(net.layer(1).W);
for i=2:prop.netDepth
    net.layer(i).W = normrnd(0,2/sqrt(prop.size(i-1)+prop.size(i)),[prop.size(i),prop.size(i-1)]);
    net.layer(i).W = gpuArray(net.layer(i).W);
end

%Random intialization for input bias vector
%Values from Mehta p 98 on the bottom
initalizationInputBiasSigma = 1/mean(mean(mean(images)));
net.layer(1).a = normrnd(0,initalizationInputBiasSigma,[imageSize(1)*imageSize(2),1]);
net.layer(1).a = gpuArray(net.layer(1).a);

%Zero initialization for hidden layer bias vector

for i=1:prop.netDepth
    net.layer(i).b = zeros(prop.size(i),1);
    net.layer(i).b = gpuArray(net.layer(i).b);
end  

for i=2:prop.netDepth
    net.layer(i).a = zeros(prop.size(i-1),1);
    net.layer(i).a = gpuArray(net.layer(i).a);
end    



disp("Done")
end

%% Do training step on sample image

function [a,b,W] = trainingStep(vData,a,b,W,prop,funs)

pHjData = gpuArray(1./(1+arrayfun(@(x)exp(-x), b + W*vData)));
%pHj = pHjData;
%hData = gpuArray(rand(size(pHj))<pHj);

gibbsStart = vData * (1-prop.gibbsSampleInputNoise) + prop.gibbsSampleInputNoise * rand(size(vData));    
[pVj,pHj] = funs.gibbsSample(gibbsStart,a,b,W,prop);    

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



