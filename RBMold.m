%Programm by Xmann02 and DontStealMyAccount
%

function RBMold(gpuID)

gpuID=gpuID+1; % Important! Transfers Grid engine counting 0,1,2,... to Matlab counting 1,2,3,....
info.gpu=gpuDevice(gpuID); % Chooses the GPU 
info.gpuCount=gpuDeviceCount; % Number of GPUs in the system, just for info
gpurng('shuffle'); % random number generator initialised randomly. Important if one uses a random number generator 
maxNumCompThreads(1); % Restrict number of CPU cores to 1


% Global parameters

prop.sizeH = 200; %size of the h-vector

prop.learningRates = [0.01,0.01]; %learning rate
prop.numTrainingImages = 1000; % number of images in training routine
prop.gibbsSampleInputNoise = 0.0; % Overlay starting sample for gibbs sample with noise options: [0:1]
prop.numGibbsIterations = 100; %number of iterations of gibbs sample
prop.gibbsSampleCD = 'CD'; %options 'CD', 'CDP'
prop.regularizer = 'None'; %reguarlizer options: 'None', 'L1', L2' 
prop.regularizerLambdas = [0.0,0.0]; %regularizer Lambda 
prop.imageType = 'BW'; %options: 'Grayscale', 'BW' 
prop.imageSamples = 'All'; %options: 'All' or any single digit 
prop.numEpochs = 1;

% Options for video Generation
vidOpt.numGibbsIterations = 500;
vidOpt.gibbsSampleInputNoise = 0.0;


funs = iniFunctions(prop);

% Image preparation

[imagesTraining,~] = prepareTrainingData(prop,'training');
[imagesTesting,~] = prepareTrainingData(prop,'testing');

% initialization
%load or initialize values, note loading will overwrite set Global
%parameters
net = ini(imagesTraining,prop.sizeH);
%[net,prop] = loadNetwork(10);



%Working area
%Train network or create video of training
%net = trainNetwork(imagesTraining,net,prop,funs);

net = TrainingVideo(imagesTraining,net,prop,funs);

%Create video of gibbs sample
%GibbsSampleVideo(vectorizeImage(images(:,:,5),28,28),net,vidOpt);


% Save Network
%saveNetwork(net,prop);

%Plot layers of network and biases
%plotNetwork(net);

% Show Samples
plotSamples(imagesTesting,net,prop,funs);

end

%% Functions 

%% Functions for seeing what W,a,b do
function plotNetwork(net)
figure(8);
subplot(4,8,1);
A = reconstructImage((net.a-min(net.a))/max(abs(net.a-min(net.a))),28,28);
imshow(A);
subplot(4,8,2);
B = reconstructImage((net.W'*net.b-min(net.W'*net.b))/max(net.W'*net.b-min(net.W'*net.b)),28,28);
imshow(B);
for i=3:32
subplot(4,8,i);
w = reconstructImage((net.W(i,:)-min(net.W(:,:))/max(abs(net.W(:,:)-min(net.W(:,:))))),28,28);
imshow(w);
end


end


%% Functions for video making

function net = TrainingVideo(images,net,prop,funs)
v = VideoWriter('TrainingVideo.avi');
open(v);
net.gibbsSample = gpuArray(vectorizeImage(images(:,:,1),28,28));
%Training Loop
for i = 1:prop.numEpochs
    prop.learningRate = prop.learningRates(i);
    prop.regularizerLambda = prop.regularizerLambdas(i);
        for j=1:prop.numTrainingImages
        j    
        image = images(:,:,j);
        vData = gpuArray(vectorizeImage(image,28,28));
        pHjData =  net.b + net.W*vData;
        gibbsStart = gibbsSampleStart(vData,net,prop);

        [net,model] = funs.gibbsSample(gibbsStart,net,prop);    

        net = funs.updateNetwork(net,vData,model.v,pHjData,model.h,prop);

        subplot(1,3,1)
        imshow(image);
        subplot(1,3,2)
        imshow(reconstructImage(gibbsStart,28,28));
        subplot(1,3,3)
        imshow(reconstructImage(model.v,28,28));
        frame = getframe(gcf);
        writeVideo(v,frame);

        end
    end
close(v);

end

function GibbsSampleVideo(vData,net,prop)
v = VideoWriter('GibbsSample.avi');
open(v);
gibbsStart = vData * (1-prop.gibbsSampleInputNoise) + prop.gibbsSampleInputNoise * rand(size(vData));
vModel = gpuArray(gibbsStart);
for i=1:prop.numGibbsIterations
pHj = gpuArray(1./(1+arrayfun(@(x)exp(-x), net.b + net.W*vModel)));
hModel = gpuArray(120/255<pHj);

pVj = gpuArray(1./(1+arrayfun(@(x)exp(-x), net.a + net.W'*hModel)));
vModel = gpuArray(120/255<pVj);

subplot(1,4,1);
imshow(reconstructImage(vData,28,28));
subplot(1,4,2);
imshow(reconstructImage(gibbsStart,28,28));
subplot(1,4,3);
imshow(reconstructImage(pVj,28,28));
subplot(1,4,4);
imshow(reconstructImage(vModel,28,28));

frame = getframe(gcf);
writeVideo(v,frame);

end    

close(v);

end

%% initialize functions
function funs = iniFunctions(prop)
    
    if(strcmp(prop.imageType,'BW'))
        funs.gibbsSample = @(vData,net,prop)GibbsSampleBW(vData,net,prop);
    end
    if(strcmp(prop.imageType,'Grayscale'))
        funs.gibbsSample = @(vData,net,prop)GibbsSampleGrayscale(vData,net,prop);
    end
    
    if(strcmp(prop.regularizer,'None'))
        funs.updateNetwork = @(net,vData,pVj,pHjData,pHj,prop)updateNoRegularizer(net,vData,pVj,pHjData,pHj,prop);
    end    
    if(strcmp(prop.regularizer,'L1'))
        funs.updateNetwork = @(net,vData,pVj,pHjData,pHj,prop)updateL1(net,vData,pVj,pHjData,pHj,prop);
    end 
    if(strcmp(prop.regularizer,'L2'))
        funs.updateNetwork = @(net,vData,pVj,pHjData,pHj,prop)updateL2(net,vData,pVj,pHjData,pHj,prop);
    end 
    

end

%% Plot 
function plotSamples(images,net,prop,funs)
figure(1)
for i=1:16
subplot(4,8,2*i-1)
A = images(:,:,i);
A = A * (1-prop.gibbsSampleInputNoise) + prop.gibbsSampleInputNoise * rand(size(A));
v = vectorizeImage(A,28,28);
imshow(A)
subplot(4,8,2*i)
%v = rand(28*28,1);
%sample an image to generate new sample
[net,pVj,pHj] = funs.gibbsSample(v,net,prop);
if strcmp(prop.imageType,'BW')
    v = pVj > 120/255;
    A = reconstructImage(v,28,28);
end    
if strcmp(prop.imageType,'Grayscale')
    A = reconstructImage(pVj,28,28);
end    
imshow(A)
end    
end

%% Training loop

function net = trainNetwork(images,net,prop,funs)
for i=1:prop.numEpochs
%Training Loop
prop.learningRate = prop.learningRates(i);
prop.regularizerLambda = prop.regularizerLambdas(i);
for j=1:prop.numTrainingImages
%j    

if(mod(j,10)==0)
    j
end

A = images(:,:,j);
net = trainingStep(A,net,prop,funs);  
%prop.learningRate = prop.startLearningRate*exp(log(prop.startLearningRate)*(1-j)/prop.numTrainingImages + log(prop.endLearningRate)*j/prop.numTrainingImages);
%end    

end
end
end

%% Image preparation

function [images,labels] = prepareTrainingData(prop,usage)

if(strcmp(usage,'training'))
    [images, labels] = mnist_parse('train-images.idx3-ubyte','train-labels.idx1-ubyte');
end

if(strcmp(usage,'testing'))
    [images, labels] = mnist_parse('t10k-images.idx3-ubyte','t10k-labels.idx1-ubyte');
end

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

if ~strcmp(prop.imageSamples,'All') % If needed, use only single digit
    mask = (labels == str2num(prop.imageSamples));
    images = images(:,:,mask);
end    

if(strcmp(usage,'training')) % Choose a random permutation of data
    p = randperm(length(images),prop.numTrainingImages);
    images = images(:,:,p(:));
end
images = gpuArray(images);
end

%% saving and loading trained networks
function saveNetwork(net,prop)
disp('saving network')
for i=1:10
    if ~exist(strcat("RBMs/", int2str(i)), 'dir')
        mkdir(strcat("RBMs/", int2str(i)))
        break;
    end
end    

save(strcat(strcat("RBMs/",int2str(i)),"/net.mat"),"net")
save(strcat(strcat("RBMs/",int2str(i)),"/prop.mat"),"prop")

end

function [net,prop] = loadNetwork(i)
disp('loading network')

net = matfile(strcat(strcat("RBMs/",int2str(i)),"/net.mat"));
prop = matfile(strcat(strcat("RBMs/",int2str(i)),"/prop.mat"));


net = net.net;
prop = prop.prop

end

%% Initialization values 
function net = ini(images,sizeH)
disp("initalizing Parameters ...");
%Get sizes of dataset
imageSize = size(images(:,:,1));
imageNumber = length(images(1,1,:));

%Random initialization of W matrix
%Values from Mehta p 98 on the bottom
net.W = normrnd(0,2/sqrt(sizeH+imageSize(1)*imageSize(2)),[sizeH,imageSize(1)*imageSize(2)]);

%Random intialization for input bias vector
%Values from Mehta p 98 on the bottom
initalizationInputBiasSigma = 1/mean(mean(mean(images)));
net.a = normrnd(0,initalizationInputBiasSigma,[imageSize(1)*imageSize(2),1]);

%Zero initialization for hidden layer bias vector
net.b = zeros(sizeH,1);

net.a = gpuArray(net.a);
net.b = gpuArray(net.b);
net.W = gpuArray(net.W);

disp("Done")
end

%% Do training step on sample image

function net = trainingStep(image,net,prop,funs)
net.gibbsSample = gpuArray(vectorizeImage(image,28,28));
vData = gpuArray(vectorizeImage(image,28,28));
pHjData = net.b + net.W*vData;
%pHj = pHjData;
%hData = gpuArray(rand(size(pHj))<pHj);
gibbsStart = gibbsSampleStart(vData,net,prop);
    
[net,model] = funs.gibbsSample(gibbsStart,net,prop);    
%Update Network parameters
net = funs.updateNetwork(net,vData,model.v,pHjData,model.h,prop);

end

function net = updateNoRegularizer(net,vData,pVj,pHjData,pHj,prop)
net.a = net.a + prop.learningRate * (vData-pVj);
net.b = net.b + prop.learningRate * (pHjData-pHj);
net.W = net.W + prop.learningRate * (kron(pHjData,vData')-kron(pHj,pVj'));
end

function net = updateL1(net,vData,pVj,pHjData,pHj,prop)
net.a = net.a + prop.learningRate * (vData-pVj);
net.b = net.b + prop.learningRate * (pHjData-pHj);
net.W = net.W + prop.learningRate * (kron(pHjData,vData')-kron(pHj,pVj')) - prop.regularizerLambda * sign(net.W) ;
end

function net = updateL2(net,vData,pVj,pHjData,pHj,prop)
net.a = net.a + prop.learningRate * (vData-pVj);
net.b = net.b + prop.learningRate * (pHjData-pHj);
net.W = net.W + prop.learningRate * (kron(pHjData,vData')-kron(pHj,pVj')) - 2 * prop.regularizerLambda * net.W ;
end

%% Calculate a gibbs sample
%input vector from data, biases for visible/hidden layer, interaction
%Matrix, num iterations
%Returns propability vectors for v and h
function [net,model] = GibbsSampleBW(vData,net,prop) 
vModel = gpuArray(vData);
for i=1:prop.numGibbsIterations
model.h = gpuArray(1./(1+arrayfun(@(x)exp(-x), net.b + net.W*vModel)));
hModel = gpuArray(rand(size(model.h))<model.h);

model.v = gpuArray(1./(1+arrayfun(@(x)exp(-x), net.a + net.W'*hModel)));
vModel = gpuArray(rand(size(model.v))<model.v);
end    
net.gibbsSample = model.v > rand(size(model.v)); 
model.v = model.v > rand(size(model.v));
model.h = model.h > rand(size(model.h));
end

function [net,model] = GibbsSampleGrayscale(vData,net,prop) 
model.v = gpuArray(vData);

for i=1:prop.numGibbsIterations
model.h = gpuArray(1./(1+arrayfun(@(x)exp(-x), net.b + net.W*model.v)));
model.v = gpuArray(1./(1+arrayfun(@(x)exp(-x), net.a + net.W'*model.h)));
end    
net.gibbsSample = model.v;
end

function gibbsStart = gibbsSampleStart(image,net,prop)
    if(strcmp(prop.gibbsSampleCD,'CD'))
    gibbsStart = image * (1-prop.gibbsSampleInputNoise) + prop.gibbsSampleInputNoise * rand(size(image));   
    end
    if(strcmp(prop.gibbsSampleCD,'CDP'))
    gibbsStart = net.gibbsSample * (1-prop.gibbsSampleInputNoise) + prop.gibbsSampleInputNoise * rand(size(image));   
    end
    if(strcmp(prop.imageType,'BW'))
    gibbsStart = gibbsStart > rand(size(gibbsStart));
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



