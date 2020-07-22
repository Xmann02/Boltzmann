

function RBM(gpuID,prop)

gpuID=gpuID+1; % Important! Transfers Grid engine counting 0,1,2,... to Matlab counting 1,2,3,....
info.gpu=gpuDevice(gpuID); % Chooses the GPU 
info.gpuCount=gpuDeviceCount; % Number of GPUs in the system, just for info
gpurng('shuffle'); % random number generator initialised randomly. Important if one uses a random number generator 
maxNumCompThreads(1); % Restrict number of CPU cores to 1



[trainingImages,trainingLabels] = prepareTrainingData(prop,'training');
[testingImages,testingLabels] = prepareTrainingData(prop,'testing');



%net = ini(trainingImages,prop);
[net,prop] = loadNetwork(10);

funs = iniFunctions(prop);
%GibbsSampleVideo(vectorizeImage(trainingImages(:,:,1),28,28),net,prop);
%net = trainNetwork(trainingImages,net,prop,funs);
%net = trainNetworkVideo(trainingImages,net,prop,funs);

%saveNetwork(net,prop);


plotNetwork(net);
prop.gibbsSampleCD = 'CD';
prop.numGibbsIterations = 2;
plotSamples(testingImages,net,prop,funs);

end


%% Do a video


function GibbsSampleVideo(vData,net,prop)
v = VideoWriter('GibbsSample.avi');
open(v);
model.v = gpuArray(vData);
model.v = gibbsSampleStart(model.v,net,prop);
gibbsStart = model.v;
for i=1:prop.numGibbsIterations
pHj = gpuArray(1./(1+arrayfun(@(x)exp(-x), net.b + net.W*model.v)));
model.h = gpuArray(pHj>rand(size(pHj)));
pVj = gpuArray(1./(1+arrayfun(@(x)exp(-x), net.a + net.W'*model.h)));
model.v = gpuArray(pVj>rand(size(pVj)));
    
net.gibbsSample = model.v; 

subplot(1,4,1);
imshow(reconstructImage(vData,28,28));
subplot(1,4,2);
imshow(reconstructImage(gibbsStart,28,28));
subplot(1,4,3);
imshow(reconstructImage(pVj,28,28));
subplot(1,4,4);
imshow(reconstructImage(model.v,28,28));

frame = getframe(gcf);
writeVideo(v,frame);
end
close(v);
end    



function net = trainNetworkVideo(images,net,prop,funs)
VW = VideoWriter('TrainingSamples.avi');
VN = VideoWriter('TrainingNetworks.avi');
open(VW);
open(VN);
    for i = 1:prop.numEpochs
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

    
            v = vectorizeImage(images(:,:,j),28,28);
            gibbsStart = gibbsSampleStart(v,net,prop);

            dOut =  rand(1,prop.sizeH,1) > prop.dropoutPropability;
            %size(dOut);
    
            [net,model] = funs.gibbsSample(gibbsStart,net,prop,dOut);    
            data = dataSample(v,net);
            model.dOut = dOut;

            %Update Network parameters
            net = funs.updateNetwork(net,data,model,prop);  
            
            if(mod(j,2000)==0)
            figure(1)
            subplot(1,3,1)
            imshow(images(:,:,j));
            subplot(1,3,2)
            imshow(reconstructImage(gibbsStart,28,28));
            subplot(1,3,3)
            imshow(reconstructImage(model.v,28,28));
            frame = getframe(gcf);
            writeVideo(VW,frame);
            end
            
            if(mod(j,100)==0)
            figure(8);
            for k=1:32
            subplot(4,8,k);
            w = reconstructImage((net.W(k,:)-min(net.W(:,:))/max(abs(net.W(:,:)-min(net.W(:,:))))),28,28);
            imshow(w);
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

A = vectorizeImage(images(:,:,sampleChoise(i)),28,28);
A = gibbsSampleStart(A,net,prop);
imshow(reconstructImage(A,28,28));
[net,A] = funs.gibbsSample(A,net,prop,dOut');    
A = reconstructImage(A.v,28,28);
figure(2)
subplot(4,4,i)
imshow(A)


end
end

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
       % case 5
       %     i=36;
        case 5
            i=33;
        case 6
            i=7;
        case 7
            i=8; 
        %case 9
        %    i=3836;
        %case 10
        %    i=10;
        %case 11
        %    i=11;
        case 8
            i=12;
        %case 13
        %    i=7828;
        %case 14
        %    i=45;
        case 9
            i=6858;
        case 10
            i=16; 
        case 11
            i=17;
        %case 18
        %    i=18;
        %case 19
        %    i=19;
        case 12
            i=20;
        %case 21
        %    i=37;
        case 13
            i=22;
        %case 23
        %    i=23;
        %case 24
        %    i=24; 
        %case 25
        %    i=25;
        case 14
            i=26;
        case 15
            i=48;
        %case 28
        %    i=42;
        %case 29
        %    i=39;
        %case 30
        %    i=30;
        case 16
            i=4790;
        %case 32
        %    i=46;      
    end  
end


%% Functions for seeing what W does
function plotNetwork(net)
figure(8);
for i=1:32
subplot(4,8,i);
w = reconstructImage((net.W(i,:)-min(net.W(:,:))/max(abs(net.W(:,:)-min(net.W(:,:))))),28,28);
imshow(w);
end


end


%% Training loop
function net = trainNetwork(images,net,prop,funs)
    for i = 1:prop.numEpochs
        i
        prop.learningRate = prop.learningRates(i);
        prop.regularizerLambda = prop.regularizerLambdas(i);
        p = randperm(length(images),prop.numTrainingImages);
        images = images(:,:,p);
        net = trainingEpoch(images,net,prop,funs);
	   saveNetwork(net,prop);
    end

end


function net = trainingEpoch(images,net,prop,funs)
    %Training Loop
    net.gibbsSample = images(1); % Set default gibbsSample, for case CDP is chosen
    for j=1:prop.numTrainingImages   

        if(mod(j,1000)==0)
        j
        end

    
        net = trainingStep(images(:,:,j),net,prop,funs);  
      

    end
end


function net = trainingStep(image,net,prop,funs)
v = vectorizeImage(image,28,28);
gibbsStart = gibbsSampleStart(v,net,prop);

dOut =  rand(1,prop.sizeH,1) > prop.dropoutPropability;
    
[net,model] = funs.gibbsSample(gibbsStart,net,prop,dOut);    
data = dataSample(v,net);
model.dOut = dOut;

%Update Network parameters
net = funs.updateNetwork(net,data,model,prop);

end



%% initialize functions
function funs = iniFunctions(prop)
    
    if(strcmp(prop.imageType,'BW'))
        funs.gibbsSample = @(vData,net,prop,dOut)GibbsSampleBW(vData,net,prop,dOut);
    end
    if(strcmp(prop.imageType,'Grayscale'))
        funs.gibbsSample = @(vData,net,prop,dOut)GibbsSampleGrayscale(vData,net,prop,dOut);
    end
    
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

for i=1:prop.numGibbsIterations
pHj = gpuArray(1./(1+arrayfun(@(x)exp(-x), net.b + net.W*model.v)));
model.h = gpuArray(pHj>120/255);
model.h = model.h.*dOut';
pVj = gpuArray(1./(1+arrayfun(@(x)exp(-x), net.a + net.W'*model.h)));
model.v = gpuArray(pVj>120/255);
end    
net.gibbsSample = model.v; 
end

function [net,model] = GibbsSampleGrayscale(vData,net,prop,dOut) 
model.v = gpuArray(vData);

for i=1:prop.numGibbsIterations
model.h = gpuArray(1./(1+arrayfun(@(x)exp(-x), net.b + net.W*model.v)));
model.h = model.h.*dOut';
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



%% Update Network parameters
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

net.a = gpuArray(net.a);
net.b = gpuArray(net.b);
net.W = gpuArray(net.W);

disp("Done")
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