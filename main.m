
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



%% Working area

%Random initalization for hidden layer vector h
h = rand(sizeH,1);

A = images(:,:,2);
B = vectorizeImage(A,28,28);



energy(B,h,a,b,W)

%C = reconstructImage(B,28,28);

%figure(1)
%imshow(A)
%figure(2)
%imshow(C)



%for i=1:length(images(1,1,:))




%% Functions 

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



