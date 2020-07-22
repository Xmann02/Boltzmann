
function staticRBMinput(gpuID)

%Options for RBM


%Optins for Network
prop.sizeH = 32; %size of the h-vector

%Options for Training
prop.numEpochs = 1;
prop.learningRates = [0.0005];
prop.numTrainingImages = 60000; % number of images in training epoch optimally at 60000 (all Images)
prop.gibbsSampleCD = 'CDP'; %options 'CD', 'CDP'
prop.numGibbsIterations = 1; %number of iterations of gibbs sample
prop.regularizer = 'None'; %reguarlizer options: 'None', 'L1', L2' 
prop.regularizerLambdas = prop.learningRates*0.0001; %regularizer Lambda 
prop.dropoutPropability = 0.5;

prop.gibbsSampleInputNoise = 0.0; % Overlay starting sample for gibbs sample with noise options: [0:1]

%Options for Training Data
prop.imageType = 'BW'; %options: 'Grayscale', 'BW' 
prop.imageSamples = 'All'; %options: 'All' or any single digit 


RBM(gpuID,prop)

end