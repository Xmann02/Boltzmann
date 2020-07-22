
function randomRBMinput(gpuID)

%Options for RBM
 

% %Optins for Network
% prop.sizeH = 200; %size of the h-vector
% 
% %Options for Training
% prop.numEpochs = 1;
% prop.learningRates = [0.05];
% prop.numTrainingImages = 60000; % number of images in training epoch optimally at 60000 (all Images)
% prop.gibbsSampleCD = 'CDP'; %options 'CD', 'CDP'
% prop.numGibbsIterations = 1; %number of iterations of gibbs sample
% prop.regularizer = 'L2'; %reguarlizer options: 'None', 'L1', L2'
% prop.regularizerLambdas = prop.learningRates*0.05; %regularizer Lambda
% prop.dropoutPropability = 0.5;
% 
% prop.gibbsSampleInputNoise = 0.0; % Overlay starting sample for gibbs sample with noise options: [0:1]


for num=1:5
    
    k=10000;
    while(k>700||k<5)
        k = round(10^normrnd(2,1)); 
    end
    
    prop.sizeH = k; %size of the h-vector
    prop.numEpochs = randi([1 10],1,1);
    
    
    
    k=10000;
    while(k>1||k<1e-3)
        k = 10^normrnd(-1.5,1); 
    end
    a = 100;
    while(a>1||a<1e-2)
        a = 10^normrnd(-1,1);
    end
    b = 0;
    while(round(b)<1)
        b = rand()*prop.numEpochs;
    end
    
    
    prop.learningRates = k*ones(prop.numEpochs,1);
    for c=round(b):prop.numEpochs
        prop.learningRates(c) = k*a;
    end 
    
    prop.numTrainingImages = 60000;
    
    d = randi([1 2]);
    if(d==1)
        prop.gibbsSampleCD = 'CD';
    end
    if(d==2)
        prop.gibbsSampleCD = 'CDP';
    end
    
    
    f=10000;
    while(f>500||f<1)
        f = round(10^normrnd(1,1)); 
    end
    
    prop.numGibbsIterations = f;
    
    
    g = randi([1 3]);
    if(g==1)
        prop.regularizer = 'None';
    end
    if(g==2)
        prop.regularizer = 'L1';
    end
    if(g==3)
        prop.regularizer = 'L2';
    end
    
    h=1;
    while(h>0.01||h<0)
        h = 10^normrnd(-2.3,0.8); 
    end
    
    prop.regularizerLambdas = prop.learningRates*h;
    
    j= 100;
    while(j>0.5||j<0)
        j = normrnd(0.5,0.5);
    end
    
    prop.dropoutPropability = j; 
    
    m = 100;
    while(m>1.0||m<0)
        m = normrnd(0.5,0.5);
    end
    
    prop.gibbsSampleInputNoise = 0.0;
    
    %Options for Training Data
    prop.imageType = 'BW'; %options: 'Grayscale', 'BW'
    prop.imageSamples = 'All'; %options: 'All' or any single digit
    
    prop
    disp(strcat('Inital Learning Rate ',num2str(k)));
    disp(strcat('Learning Rate Reduction ',num2str(a)));
    disp(strcat('Lambda/Learning Rate ',num2str(h)));
    RBM(gpuID,prop);
    
end



end