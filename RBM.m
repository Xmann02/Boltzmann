%Main program

function RBM(gpuID,prop) %Input gpu and options to run RBM

gpuID=gpuID+1; % Important! Transfers Grid engine counting 0,1,2,... to Matlab counting 1,2,3,....
info.gpu=gpuDevice(gpuID); % Chooses the GPU 
info.gpuCount=gpuDeviceCount; % Number of GPUs in the system, just for info
gpurng('shuffle'); % random number generator initialised randomly. Important if one uses a random number generator 
maxNumCompThreads(1); % Restrict number of CPU cores to 1


%Prepare images (Labels not needed if image samples set to 'All'

[trainingImages,trainingLabels] = prepareTrainingData(prop,'training');
[testingImages,testingLabels] = prepareTrainingData(prop,'testing');


%Choose one option
net = ini(trainingImages,prop);     %Randomly initialize RBM
%[net,prop] = loadNetwork(9);       %Load previously trained RBM, you must
%remember number of loaded Network, your set properties will be overwritten
%by this option!

%Always activate
funs = iniFunctions(prop);      %Initializes needed functions


%Choose one option
net = trainNetwork(trainingImages,net,prop,funs);      %Normal Training process
%net = trainNetworkVideo(trainingImages,net,prop,funs);  %Create training video (Warning: Not on scc, takes longer)


%GibbsSampleVideo(vectorizeImage(trainingImages(:,:,1),28,28),net,prop);
%%Create Sample video

saveNetwork(net,prop);  %Save network, else will be lost


%Change options for new sample creation

prop.gibbsSampleInputNoise = 0.0; %No noise on input samples
plotNetwork(net);   %Plot layers of weight matrix

%Set CD-n sampling and 100 steps
prop.gibbsSampleCD = 'CD';  
prop.numGibbsIterations = 100;

%Plot new samples
plotSamples(testingImages,net,prop,funs);

end


%% Make a video

%Video of gibbs sampling process
function GibbsSampleVideo(vData,net,prop)
%Video setup
v = VideoWriter('GibbsSample.avi');
open(v);
model.v = gpuArray(vData);
model.v = gibbsSampleStart(model.v,net,prop);
gibbsStart = model.v;
for i=1:prop.numGibbsIterations
pHj = gpuArray(1./(1+arrayfun(@(x)exp(-x), net.b + net.W*model.v)));    %Hidden layer probability calculation
model.h = gpuArray(pHj>rand(size(pHj)));        %Hidden layer activation
pVj = gpuArray(1./(1+arrayfun(@(x)exp(-x), net.a + net.W'*model.h)));   %Visible layer probability
model.v = gpuArray(pVj>rand(size(pVj)));    %Visible layer activation
    
net.gibbsSample = model.v; 

subplot(1,4,1);
imshow(reconstructImage(vData,28,28));  %Originial data sample
subplot(1,4,2);
imshow(reconstructImage(gibbsStart,28,28)); %Gibbs start 
subplot(1,4,3);
imshow(reconstructImage(pVj,28,28)); %Visible layer activation probability
subplot(1,4,4);
imshow(reconstructImage(model.v,28,28));    %Visible layer activation

frame = getframe(gcf);
writeVideo(v,frame);
end
close(v);
end    


% Video of training process
function net = trainNetworkVideo(images,net,prop,funs)
%Setup
VW = VideoWriter('TrainingSamples.avi');
VN = VideoWriter('TrainingNetworks.avi');
open(VW);
open(VN);
    for i = 1:prop.numEpochs
        %Setup parameters and images for learning (one epoch)
        i
        prop.learningRate = prop.learningRates(i)
        prop.regularizerLambda = prop.regularizerLambdas(i)
        p = randperm(length(images),prop.numTrainingImages);
        images = images(:,:,p);
        %Training Loop
        net.gibbsSample = images(1); % Set default gibbsSample, for case CDP is chosen
        for j=1:prop.numTrainingImages   

            if(mod(j,2000)==0)
            j
            end

    
            v = vectorizeImage(images(:,:,j),28,28);    %Gibbs sample setup
            gibbsStart = gibbsSampleStart(v,net,prop);
            dOut =  rand(1,prop.sizeH,1) > prop.dropoutPropability; %Prepare dropout
    
            [net,model] = funs.gibbsSample(gibbsStart,net,prop,dOut);    %Calc gibbs sample
            data = dataSample(v,net);
            model.dOut = dOut;

            %Update Network parameters
            net = funs.updateNetwork(net,data,model,prop);  
            
            if(mod(j,2000)==0)  %Frequency of sample images
                figure(1)
                subplot(1,3,1)
                imshow(images(:,:,j));  %Data sample
                subplot(1,3,2)
                imshow(reconstructImage(gibbsStart,28,28)); %Start of gibbs sample (Depends on PCD-n and noise)
                subplot(1,3,3)
                imshow(reconstructImage(model.v,28,28));    %Gibbs sample
                frame = getframe(gcf);
                writeVideo(VW,frame);
            end
            
            if(mod(j,2000)==0)  %Frequency of weight matrix images
                figure(8);
                for k=1:32
                subplot(4,8,k);
                w = reconstructImage((net.W(k,:)-min(net.W(:,:))/max(abs(net.W(:,:)-min(net.W(:,:))))),28,28);  
                imshow(w);  %Show matrix layers
                end
                frame = getframe(gcf);
                writeVideo(VN,frame);
            end
            
        end
        
    end
close(VW);
close(VN);
end



%% Generate some Samples

function plotSamples(images,net,prop,funs)

dOut = ones(size(net.b));
for i=1:16

figure(1)

subplot(4,4,i)

A = vectorizeImage(images(:,:,sampleChoise(i)),28,28);  %Choose image from est set
A = gibbsSampleStart(A,net,prop);       %Start gibbs sample on image

imshow(reconstructImage(A,28,28));  
[net,A] = funs.gibbsSample(A,net,prop,dOut');   %Technically net update not needed (formality issue with gibbs sample function)
A = reconstructImage(A.v,28,28);        %Reshape vector to image
figure(2)
subplot(4,4,i)  
imshow(A)       %Show image


end
end

%Handpicked selection of testing set for sample creation
function i = sampleChoise(j)
    switch j
        case 1
            i=100;
        case 2
            i=2;
        case 3
            i=3;
        case 4
            i=4;
        case 5
            i=33;
        case 6
            i=7;
        case 7
            i=8; 
        case 8
            i=12;
        case 9
            i=6858;
        case 10
            i=16; 
        case 11
            i=17;
        case 12
            i=20;
        case 13
            i=22;
        case 14
            i=26;
        case 15
            i=48;
        case 16
            i=4790;     
    end  
end


%% Functions for seeing what W does
function plotNetwork(net)
figure(8);
for i=1:16  %First 16 rows
subplot(4,4,i);
w = reconstructImage((net.W(i,:)-min(net.W(:,:))/max(abs(net.W(:,:)-min(net.W(:,:))))),28,28);  %Reshape row of W-matrix into image
imshow(w);
end


end


%% Training loop
function net = trainNetwork(images,net,prop,funs)
    for i = 1:prop.numEpochs
        i %Output training epoch
        prop.learningRate = prop.learningRates(i);  %Set epoch learning rate
        prop.regularizerLambda = prop.regularizerLambdas(i);    %Set epoch regularizer lambda
        p = randperm(length(images),prop.numTrainingImages);    %Random training set permutation
        images = images(:,:,p);                                 
        net = trainingEpoch(images,net,prop,funs);          %Do training epoch
	   saveNetwork(net,prop);               %Saves every epoch
    end

end


function net = trainingEpoch(images,net,prop,funs)
    %Training Loop
    net.gibbsSample = images(1); % Set default gibbsSample, for case PCD-n is chosen (else error thrown)
    for j=1:prop.numTrainingImages   

        if(mod(j,1000)==0)
        j   %Output for actual gibbs step
        end

    
        net = trainingStep(images(:,:,j),net,prop,funs);  %Do one training step
      

    end
end


function net = trainingStep(image,net,prop,funs)
v = vectorizeImage(image,28,28);
gibbsStart = gibbsSampleStart(v,net,prop);  %Set gibbs sample start

dOut =  rand(1,prop.sizeH,1) > prop.dropoutPropability; %Prepare dropout
    
[net,model] = funs.gibbsSample(gibbsStart,net,prop,dOut);   %Run gibbs sample
data = dataSample(v,net);   
model.dOut = dOut;

%Update Network parameters
net = funs.updateNetwork(net,data,model,prop);

end



%% initialize functions
function funs = iniFunctions(prop)
    
    %Set Gibbs sample function (BW or experimental Grayscale calculation)
    if(strcmp(prop.imageType,'BW'))
        funs.gibbsSample = @(vData,net,prop,dOut)GibbsSampleBW(vData,net,prop,dOut);
    end
    if(strcmp(prop.imageType,'Grayscale'))
        funs.gibbsSample = @(vData,net,prop,dOut)GibbsSampleGrayscale(vData,net,prop,dOut);
    end
    
    %Set Regularizer function
    if(strcmp(prop.regularizer,'None'))
        funs.updateNetwork = @(net,data,model,prop)updateNoRegularizer(net,data,model,prop);
    end    
    if(strcmp(prop.regularizer,'L1'))
        funs.updateNetwork = @(net,data,model,prop)updateL1(net,data,model,prop);
    end 
    if(strcmp(prop.regularizer,'L2'))
        funs.updateNetwork = @(net,data,model,prop)updateL2(net,data,model,prop);
    end 
    

end

%% Calculate data part for cd
function dataI = dataSample(vData,net)
dataI.v = vData;
dataI.h = gpuArray(1./(1+arrayfun(@(x)exp(-x),net.b + net.W*vData)));
end




%% Calculate a gibbs sample
%input vector from data, biases for visible/hidden layer, interaction
%Matrix, num iterations
%Returns propability vectors for v and h
function [net,model] = GibbsSampleBW(vData,net,prop,dOut) 
model.v = gpuArray(vData);

%Gibbs iterations
for i=1:prop.numGibbsIterations
pHj = gpuArray(1./(1+arrayfun(@(x)exp(-x), net.b + net.W*model.v))); %Hidden layer activation chance
model.h = gpuArray(pHj>rand(size(pHj)));    %Hidden layer activation
model.h = model.h.*dOut';                   %Dropout calculation
pVj = gpuArray(1./(1+arrayfun(@(x)exp(-x), net.a + net.W'*model.h)));   %Visible layer activation chance
model.v = gpuArray(pVj>rand(size(pVj)));     %Visible layer activation
end    
net.gibbsSample = model.v; %Set output
end

%Outdated experimental option
function [net,model] = GibbsSampleGrayscale(vData,net,prop,dOut) 
model.v = gpuArray(vData);

for i=1:prop.numGibbsIterations
model.h = gpuArray(1./(1+arrayfun(@(x)exp(-x), net.b + net.W*model.v)));
model.h = model.h.*dOut';
model.v = gpuArray(1./(1+arrayfun(@(x)exp(-x), net.a + net.W'*model.h)));
end    
net.gibbsSample = model.v;
end


%Start a gibbs sample (depends on CD-n or PCD-n and input noise)
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



%% Update Network parameters
%options for no regularizer, L1 and L2 
function net = updateNoRegularizer(net,data,model,prop)
net.a = net.a + prop.learningRate * (data.v-model.v);
net.b = net.b + prop.learningRate * (data.h-model.h).*model.dOut';
net.W = net.W + prop.learningRate * (kron(data.h.*model.dOut',data.v')-kron(model.h.*model.dOut',model.v'));
end

function net = updateL1(net,data,model,prop)
net.a = net.a + prop.learningRate * (data.v-model.v);
net.b = net.b + prop.learningRate * (data.h-model.h).*model.dOut';
net.W = net.W + prop.learningRate * (kron(data.h.*model.dOut',data.v')-kron(model.h.*model.dOut',model.v')) -  prop.regularizerLambda * kron(model.dOut',ones(size(data.v'))) .* sign(net.W) ;
end

function net = updateL2(net,data,model,prop)
net.a = net.a + prop.learningRate * (data.v-model.v);
net.b = net.b + prop.learningRate * (data.h-model.h).*model.dOut';
net.W = net.W + prop.learningRate * (kron(data.h.*model.dOut',data.v')-kron(model.h.*model.dOut',model.v')) - 2 * prop.regularizerLambda * kron(model.dOut',ones(size(data.v'))) .*  net.W ;
end


%% Initialization values 
function net = ini(images,prop)
disp("initalizing Parameters ...");
%Get sizes of dataset
imageSize = size(images(:,:,1));
imageNumber = length(images(1,1,:));

%Random initialization of W matrix
%Values from Mehta p 98 on the bottom
net.W = normrnd(0,2/sqrt(prop.sizeH+imageSize(1)*imageSize(2)),[prop.sizeH,imageSize(1)*imageSize(2)]);

%Random intialization for input bias vector
%Values from Mehta p 98 on the bottom
initalizationInputBiasSigma = 1/mean(mean(mean(images)));
net.a = normrnd(0,initalizationInputBiasSigma,[imageSize(1)*imageSize(2),1]);

%Zero initialization for hidden layer bias vector
net.b = zeros(prop.sizeH,1);

%Make into gpuArray
net.a = gpuArray(net.a);
net.b = gpuArray(net.b);
net.W = gpuArray(net.W);

disp("Done")
end





%% Image preparation

function [images,labels] = prepareTrainingData(prop,usage)

if(strcmp(usage,'training'))
    [images, labels] = mnist_parse('train-images.idx3-ubyte','train-labels.idx1-ubyte'); %mnist_parse loads images and labels (Not our own)
end

if(strcmp(usage,'testing'))
    [images, labels] = mnist_parse('t10k-images.idx3-ubyte','t10k-labels.idx1-ubyte');
end

if strcmp(prop.imageType,'BW')
    %Simple cutoff for Black/White images
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

%Load net and properties
net = matfile(strcat(strcat("RBMs/",int2str(i)),"/net.mat"));
prop = matfile(strcat(strcat("RBMs/",int2str(i)),"/prop.mat"));
net = net.net;
prop = prop.prop

%Some info about loaded net
disp(strcat("Learning Rate ",num2str(prop.learningRates(1))))
disp(strcat("Lambda/Learning Rate ",num2str(prop.regularizerLambdas(1)/prop.learningRates(1))))
end








%% Reshape image to vector
function vectorX = vectorizeImage(image,sizeX,sizeY)
vectorX = reshape(image,[sizeX*sizeY,1]);
end

%% Reconstruct image from a vector
function image = reconstructImage(vector,sizeX,sizeY)
image = reshape(vector,[sizeX,sizeY]);
end